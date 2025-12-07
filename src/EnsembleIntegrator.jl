module EnsembleIntegrator

using LinearAlgebra
using Random
using CUDA
using Flux
using ProgressMeter

export evolve_sde, evolve_sde_snapshots, SnapshotIntegrator, build_snapshot_integrator

# --- Diagnostic message flags (print only once per session) ---
const MULTI_GPU_MESSAGE_PRINTED = Ref(false)
const SINGLE_GPU_MESSAGE_PRINTED = Ref(false)
const CPU_MESSAGE_PRINTED = Ref(false)

# --- 1. Memory Cache for Zero-Allocation Steps ---

# Per-wrapper CPU integration state
struct CPUIntegratorState{TX, TP, TS, TM, TC}
    x::TX          # Working state on CPU
    Phi::TP        # Drift matrix on CPU
    Sigma::TS      # Diffusion factor (UpperTriangular) on CPU
    model::TM      # Frozen model used for integration
    cache::TC      # Reusable IntegratorCache
end

const CPU_STATE_CACHE = IdDict{Any, CPUIntegratorState}()

# Per-wrapper single-GPU integration state
struct GPUIntegratorState{TX, TP, TS, TM, TC}
    x::TX          # Working state on GPU
    Phi::TP        # Drift matrix on GPU
    Sigma::TS      # Diffusion factor (UpperTriangular) on GPU
    model::TM      # Frozen model on GPU
    cache::TC      # Reusable IntegratorCache
end

const GPU_STATE_CACHE = IdDict{Any, GPUIntegratorState}()

struct IntegratorCache{M1, M2}
    drift_out::M1    # Stores output of s(x)
    drift_term::M2   # Stores Phi * s(x)
    noise::M2        # Stores Gaussian noise
    diff_term::M2    # Stores Sigma * noise
end

# Progress helpers (throttled updates to avoid CPU/GPU sync overhead)
const MAX_PROGRESS_UPDATES = 200
const PROGRESS_MIN_INTERVAL = 0.5

function init_progress(total_steps::Int;
                       progress::Union{Bool, ProgressMeter.Progress, Nothing},
                       desc::AbstractString)
    if total_steps <= 0 || progress === false || progress === nothing
        return nothing, 0, false
    end

    stride = max(1, fld(total_steps, MAX_PROGRESS_UPDATES))

    if progress isa ProgressMeter.Progress
        return progress, stride, false
    end

    p = Progress(total_steps; desc = desc, dt = PROGRESS_MIN_INTERVAL)
    return p, stride, true
end

@inline function tick_progress!(p, step::Int, stride::Int, total_steps::Int)
    if p !== nothing && (step % stride == 0 || step == total_steps)
        ProgressMeter.update!(p, step)
    end
end

@inline function finish_progress!(p, owns::Bool)
    if owns && p !== nothing
        ProgressMeter.finish!(p)
    end
end

function setup_cache(x::AbstractMatrix, s_model)
    # Run dummy pass to get output size/type
    dummy_out = s_model(x)

    drift_out  = similar(dummy_out)
    drift_term = similar(x)
    noise      = similar(x)
    diff_term  = similar(x)

    return IntegratorCache(drift_out, drift_term, noise, diff_term)
end

# --- 2. The Core Stepper (Euler-Maruyama) ---
"""
    em_step!(x, cache, s_model, Phi, Sigma, dt, sq_2dt)
In-place update: x += dt*(Phi*s(x)) + sqrt(2dt)*(Sigma*xi)
"""
function em_step!(x::AbstractMatrix, cache::IntegratorCache, s_model, Phi, Sigma, dt, sq_2dt)
    # 1. Evaluate Score s(x)
    # Flux allocates the output; we copy it to cache to keep the rest in-place.
    copyto!(cache.drift_out, s_model(x))

    # 2. Drift Term: Phi * s(x)
    mul!(cache.drift_term, Phi, cache.drift_out)

    # 3. Diffusion Term: Sigma * Noise
    randn!(cache.noise)
    mul!(cache.diff_term, Sigma, cache.noise)

    # 4. Update
    # Precision Fix: Ensure dt/sq_2dt match the array type (usually Float32)
    T = eltype(x)
    dt_T = T(dt)
    sq_2dt_T = T(sq_2dt)
    
    @. x += (dt_T * cache.drift_term) + (sq_2dt_T * cache.diff_term)
    return nothing
end

# --- 3. Single Device Integration Loop ---
function integrate_on_device!(x,
                              s_model,
                              Phi,
                              Sigma,
                              dt,
                              T,
                              cache::IntegratorCache;
                              progress::Union{Bool, ProgressMeter.Progress, Nothing} = false,
                              progress_desc::Union{Nothing, AbstractString} = nothing)
    sq_2dt = sqrt(2 * dt)
    n_steps = floor(Int, T / dt)

    desc = something(progress_desc, "Ensemble integration")
    p, stride, owns = init_progress(n_steps; progress = progress, desc = desc)

    for step in 1:n_steps
        em_step!(x, cache, s_model, Phi, Sigma, dt, sq_2dt)
        tick_progress!(p, step, stride, n_steps)
    end

    finish_progress!(p, owns)
    return x
end

function integrate_on_device!(x,
                              s_model,
                              Phi,
                              Sigma,
                              dt,
                              T;
                              progress::Union{Bool, ProgressMeter.Progress, Nothing} = false,
                              progress_desc::Union{Nothing, AbstractString} = nothing)
    cache = setup_cache(x, s_model)
    return integrate_on_device!(x, s_model, Phi, Sigma, dt, T, cache;
                                progress = progress, progress_desc = progress_desc)
end

# --- 4. Multi-GPU Orchestration (kept for compatibility, but
#       the main entry point below always uses the single-GPU path
#       for better performance and simpler caching). ---
"""
    solve_multi_gpu(x_cpu, model_cpu, Phi, Sigma, dt, T)
Splits the batch across ALL available GPUs, runs integration, and gathers results.
"""
function solve_multi_gpu(x_cpu, model_cpu, Phi, Sigma, dt, T)
    # NOTE: This path is no longer used by `evolve_sde`; the optimized
    # single-GPU dispatcher below gives better end-to-end performance.
    devices = collect(CUDA.devices())
    n_dev = length(devices)
    N = size(x_cpu, 2)

    # Calculate chunks
    chunk_size = div(N, n_dev)
    residuals = N % n_dev

    results = Vector{Any}(undef, n_dev)

    # Helper function to process a chunk on a specific GPU
    function process_chunk(gpu_idx::Int)
        dev = devices[gpu_idx]
        CUDA.device!(dev) # Switch CUDA context to this GPU

        # Determine batch slice for this GPU
        start_idx = (gpu_idx-1)*chunk_size + 1 + min(gpu_idx-1, residuals)
        end_idx = start_idx + chunk_size - 1 + (gpu_idx <= residuals ? 1 : 0)

        if start_idx <= end_idx
            # 1. Move Data to GPU
            x_gpu = CUDA.cu(x_cpu[:, start_idx:end_idx])
            Phi_gpu = CUDA.cu(Phi)
            
            # Optimization: Ensure Sigma is UpperTriangular on GPU for cublasTrmm
            # We strip the wrapper to move data, then re-wrap on device
            Sigma_gpu = UpperTriangular(CUDA.cu(Matrix(Sigma)))

            # 2. Move Model to GPU and set TEST MODE
            # Deepcopy is vital because Flux models are mutable
            model_gpu = deepcopy(model_cpu) |> Flux.gpu
            Flux.testmode!(model_gpu) # <--- CRITICAL FIX

            # 3. Integrate
            integrate_on_device!(x_gpu, model_gpu, Phi_gpu, Sigma_gpu, dt, T)

            # 4. Bring back to CPU
            return Array(x_gpu)
        else
            return similar(x_cpu, size(x_cpu, 1), 0)
        end
    end

    # Process chunks in parallel using tasks
    tasks = [Threads.@spawn process_chunk(i) for i in 1:n_dev]

    for (i, task) in enumerate(tasks)
        results[i] = fetch(task)
    end

    return cat(results..., dims=2)
end

# --- 5. Main Dispatcher ---
"""
    evolve_sde(s_model, x0, Phi, Sigma, dt, T; device="cpu", progress=false, progress_desc=nothing)

Main entry point replacing FastSDE.evolve. Set `progress=true` (or pass an
existing `ProgressMeter.Progress`) to display a throttled progress bar during
integration on either CPU or GPU.
"""
function evolve_sde(s_model,
                    x0::AbstractMatrix,
                    Phi,
                    Sigma,
                    dt,
                    T;
                    device = "cpu",
                    progress::Union{Bool, ProgressMeter.Progress, Nothing} = false,
                    progress_desc::Union{Nothing, AbstractString} = nothing)
    desc_gpu = progress_desc === nothing ? "Ensemble integration (GPU)" : progress_desc
    desc_cpu = progress_desc === nothing ? "Ensemble integration (CPU)" : progress_desc

    if device == "gpu"
        if CUDA.functional()
            # Optimized single-GPU path (used even when multiple GPUs are present).
            if !SINGLE_GPU_MESSAGE_PRINTED[]
                println("EnsembleIntegrator: Running on Single GPU.")
                SINGLE_GPU_MESSAGE_PRINTED[] = true
            end

            # Reuse cached GPU state per score wrapper to avoid repeated
            # allocations and model deep-copies across calls.
            state = get!(GPU_STATE_CACHE, s_model) do
                x_gpu = CUDA.cu(x0)
                Phi_gpu = CUDA.cu(Phi)
                Sigma_gpu = Sigma isa UpperTriangular ? UpperTriangular(CUDA.cu(Matrix(Sigma))) :
                             Sigma isa LowerTriangular ? LowerTriangular(CUDA.cu(Matrix(Sigma))) :
                             CUDA.cu(Matrix(Sigma))

                model_gpu = Flux.gpu(s_model)
                Flux.testmode!(model_gpu)

                cache = setup_cache(x_gpu, model_gpu)
                GPUIntegratorState(x_gpu, Phi_gpu, Sigma_gpu, model_gpu, cache)
            end

            # Refresh working state from current x0 values but reuse buffers/model.
            CUDA.@sync copyto!(state.x, x0)
            integrate_on_device!(state.x, state.model, state.Phi, state.Sigma, dt, T, state.cache;
                                 progress = progress, progress_desc = desc_gpu)
            return Array(state.x)
        else
            @warn "GPU requested but CUDA not functional. Falling back to CPU."
        end
    end

    # CPU Path
    if !CPU_MESSAGE_PRINTED[]
        println("EnsembleIntegrator: Running on CPU.")
        CPU_MESSAGE_PRINTED[] = true
    end

    # Reuse cached CPU state per score wrapper to avoid repeated allocations
    # and model deep-copies across calls.
    state = get!(CPU_STATE_CACHE, s_model) do
        x_curr = Array(x0)
        Phi_cpu = Array(Phi)
        Sigma_cpu = Sigma isa Union{UpperTriangular, LowerTriangular} ? Sigma : Array(Sigma)

        model_cpu = Flux.cpu(s_model)
        # Work on a dedicated copy to decouple from any training model
        model_cpu = deepcopy(model_cpu)
        Flux.testmode!(model_cpu)

        cache = setup_cache(x_curr, model_cpu)
        CPUIntegratorState(x_curr, Phi_cpu, Sigma_cpu, model_cpu, cache)
    end

    # Refresh working state from current x0 values but reuse buffers/model.
    copyto!(state.x, x0)
    integrate_on_device!(state.x, state.model, state.Phi, state.Sigma, dt, T, state.cache;
                         progress = progress, progress_desc = desc_cpu)
    return Array(state.x)
end

"""
    evolve_sde_snapshots(s_model, x0, Phi, Sigma;
                         dt, n_steps, burn_in=0, resolution=1,
                         device="cpu", boundary=nothing,
                         progress=false, progress_desc=nothing)

Integrate the SDE

    dx = Φ * s(x) dt + √2 Σ dW

for an ensemble of initial conditions `x0` and a score model `s_model`, returning
snapshot states at regular intervals. The state `x0` is a matrix of size
`(dim, n_ensembles)`. The result is an array of size `(dim, n_snapshots, n_ensembles)`,
where `n_snapshots` is determined by `n_steps`, `burn_in`, and `resolution`.

If `boundary` is provided as a tuple `(min, max)`, then on each Euler–Maruyama
step a simple reflecting condition is enforced: whenever a state component
leaves the interval, the corresponding ensemble member is reset to its initial
condition `x0`. The total number of such resets is reported via an
informational log message. This behaviour is supported on both the CPU and GPU
paths.

This function reuses the optimized device-specific caches and model copies
from [`evolve_sde`] for efficiency and scalability. Set `progress=true` (or
provide an existing `ProgressMeter.Progress`) to display a throttled progress
bar during integration on CPU or GPU.
"""
function evolve_sde_snapshots(s_model,
                              x0::AbstractMatrix,
                              Phi,
                              Sigma;
                              dt::Real,
                              n_steps::Integer,
                              burn_in::Integer = 0,
                              resolution::Integer = 1,
                              device::AbstractString = "cpu",
                              boundary = nothing,
                              progress::Union{Bool, ProgressMeter.Progress, Nothing} = false,
                              progress_desc::Union{Nothing, AbstractString} = nothing)
    steps_per_snapshot = max(Int(resolution), 1)
    total_steps = max(Int(n_steps), 1)

    total_snapshots = fld(total_steps, steps_per_snapshot)
    total_snapshots > 0 ||
        error("resolution too large or n_steps too small to produce any snapshots")

    burn_in_steps = max(Int(burn_in), 0)
    burn_in_snapshots = fld(burn_in_steps, steps_per_snapshot)
    keep_snapshots = total_snapshots - burn_in_snapshots
    keep_snapshots > 0 ||
        error("Burn-in removes all samples; increase n_steps or reduce burn_in/resolution.")

    dim, n_ens = size(x0)
    traj = Array{eltype(x0)}(undef, dim, keep_snapshots, n_ens)

    # Precompute constants for the Euler–Maruyama stepper
    sq_2dt = sqrt(2 * dt)

    # Track how many ensemble members are reset to their initial conditions
    reset_counter = Ref{Int}(0)
    desc_gpu = progress_desc === nothing ? "Snapshot integration (GPU)" : progress_desc
    desc_cpu = progress_desc === nothing ? "Snapshot integration (CPU)" : progress_desc

    # GPU path (single-device, cached state) mirrors `evolve_sde`
    if device == "gpu"
        if CUDA.functional()
            if !SINGLE_GPU_MESSAGE_PRINTED[]
                println("EnsembleIntegrator: Running on Single GPU.")
                SINGLE_GPU_MESSAGE_PRINTED[] = true
            end

            # Reuse cached GPU state per score wrapper
            state = get!(GPU_STATE_CACHE, s_model) do
                x_gpu = CUDA.cu(x0)
                Phi_gpu = CUDA.cu(Phi)
                Sigma_gpu = Sigma isa UpperTriangular ? UpperTriangular(CUDA.cu(Matrix(Sigma))) :
                             Sigma isa LowerTriangular ? LowerTriangular(CUDA.cu(Matrix(Sigma))) :
                             CUDA.cu(Matrix(Sigma))

                model_gpu = Flux.gpu(s_model)
                Flux.testmode!(model_gpu)

                cache = setup_cache(x_gpu, model_gpu)
                GPUIntegratorState(x_gpu, Phi_gpu, Sigma_gpu, model_gpu, cache)
            end

            CUDA.@sync copyto!(state.x, x0)

            # Preserve a copy of the initial conditions on the GPU for
            # boundary enforcement. This is kept local to the call to
            # avoid assumptions about reuse of x0 across runs.
            x0_gpu = copy(state.x)

            snapshot_idx = 0
            p, stride, owns = init_progress(total_steps; progress = progress, desc = desc_gpu)
            for step in 1:total_steps
                em_step!(state.x, state.cache, state.model, state.Phi, state.Sigma, dt, sq_2dt)
                if boundary !== nothing
                    enforce_boundary!(state.x, x0_gpu, boundary, reset_counter)
                end
                if step > burn_in_steps && step % steps_per_snapshot == 0
                    snapshot_idx += 1
                    @inbounds traj[:, snapshot_idx, :] .= Array(state.x)
                end
                tick_progress!(p, step, stride, total_steps)
            end
            finish_progress!(p, owns)

            if boundary !== nothing
                @info "Boundary resets during SDE integration" resets=reset_counter[]
            end

            return traj
        else
            @warn "GPU requested but CUDA not functional. Falling back to CPU."
        end
    end

    # CPU path with cached state per score wrapper (matches `evolve_sde`)
    if !CPU_MESSAGE_PRINTED[]
        println("EnsembleIntegrator: Running on CPU.")
        CPU_MESSAGE_PRINTED[] = true
    end

    state = get!(CPU_STATE_CACHE, s_model) do
        x_curr = Array(x0)
        Phi_cpu = Array(Phi)
        Sigma_cpu = Sigma isa Union{UpperTriangular, LowerTriangular} ? Sigma : Array(Sigma)

        model_cpu = Flux.cpu(s_model)
        model_cpu = deepcopy(model_cpu)
        Flux.testmode!(model_cpu)

        cache = setup_cache(x_curr, model_cpu)
        CPUIntegratorState(x_curr, Phi_cpu, Sigma_cpu, model_cpu, cache)
    end

    copyto!(state.x, x0)

    snapshot_idx = 0
    p, stride, owns = init_progress(total_steps; progress = progress, desc = desc_cpu)
    for step in 1:total_steps
        em_step!(state.x, state.cache, state.model, state.Phi, state.Sigma, dt, sq_2dt)
        if boundary !== nothing
            enforce_boundary!(state.x, x0, boundary, reset_counter)
        end
        if step > burn_in_steps && step % steps_per_snapshot == 0
            snapshot_idx += 1
            @inbounds traj[:, snapshot_idx, :] .= state.x
        end
        tick_progress!(p, step, stride, total_steps)
    end
    finish_progress!(p, owns)

    if boundary !== nothing
        @info "Boundary resets during SDE integration" resets=reset_counter[]
    end

    return traj
end

"""
    enforce_boundary!(x, x0, boundary, reset_counter)

Helper that enforces a simple boundary condition on the ensemble
state matrix `x` of size `(dim, n_ens)`. For each ensemble member (column),
if any component leaves the interval specified by `boundary = (min, max)`,
the entire column is reset to the corresponding initial condition stored in
`x0`. The total number of resets is accumulated in `reset_counter[]`.
"""
function enforce_boundary!(x::AbstractMatrix{<:Real},
                           x0::AbstractMatrix{<:Real},
                           boundary::Tuple{<:Real,<:Real},
                           reset_counter::Base.RefValue{Int})
    lo, hi = boundary
    dim, n_ens = size(x)

    @inbounds for j in 1:n_ens
        reset_column = false
        for i in 1:dim
            v = x[i, j]
            if v < lo || v > hi
                reset_column = true
                break
            end
        end
        if reset_column
            @inbounds @simd for i in 1:dim
                x[i, j] = x0[i, j]
            end
            reset_counter[] += 1
        end
    end
    return nothing
end

function enforce_boundary!(x::CUDA.CuArray{T,2},
                           x0::CUDA.CuArray{T,2},
                           boundary::Tuple{<:Real,<:Real},
                           reset_counter::Base.RefValue{Int}) where {T<:Real}
    lo, hi = boundary
    lo_T = T(lo)
    hi_T = T(hi)

    # Compute, on the GPU, which columns have at least one out-of-bounds
    # entry, then bring that small mask back to the CPU and reset full
    # columns via GPU broadcasts.
    viol = (x .< lo_T) .| (x .> hi_T)
    col_violation = vec(reduce(|, viol; dims=1))
    col_violation_host = Array(col_violation)

    local_resets = 0
    @inbounds for (j, do_reset) in pairs(col_violation_host)
        if do_reset
            @views x[:, j] .= x0[:, j]
            local_resets += 1
        end
    end

    reset_counter[] += local_resets
    return nothing
end

"""
    SnapshotIntegrator(s_model, device="cpu")

Callable object that wraps a score model `s_model` together with a device
selection string (`\"cpu\"` or `\"gpu\"`) for repeated snapshot-based SDE
integration.

Typically constructed via [`build_snapshot_integrator`] rather than directly.
"""
struct SnapshotIntegrator{M}
    s_model::M
    device::String
end

"""
    build_snapshot_integrator(s_model; device="cpu")

Construct a cached snapshot integrator for the score model `s_model`.

The returned object is callable with the signature

```julia
integrator(x0, Phi, Sigma;
           dt, n_steps;
           burn_in=0,
           resolution=1,
           boundary=nothing,
           progress=false,
           progress_desc=nothing)
```

If `boundary` is supplied as `(min, max)`, it is forwarded to
[`evolve_sde_snapshots`] to enable optional boundary enforcement on the CPU
and GPU paths (see that docstring for details). Progress settings are forwarded to
enable optional progress bars during integration.

and internally dispatches to [`evolve_sde_snapshots`], reusing the optimized
per-model CPU/GPU caches for efficient repeated Langevin integrations.
"""
function build_snapshot_integrator(s_model; device::AbstractString = "cpu")
    return SnapshotIntegrator(s_model, String(device))
end

function (integ::SnapshotIntegrator)(x0::AbstractMatrix,
                                     Phi,
                                     Sigma;
                                     dt::Real,
                                     n_steps::Integer,
                                     burn_in::Integer = 0,
                                     resolution::Integer = 1,
                                     boundary = nothing,
                                     progress::Union{Bool, ProgressMeter.Progress, Nothing} = false,
                                     progress_desc::Union{Nothing, AbstractString} = nothing)
    return evolve_sde_snapshots(integ.s_model,
                                x0,
                                Phi,
                                Sigma;
                                dt = dt,
                                n_steps = n_steps,
                                burn_in = burn_in,
                                resolution = resolution,
                                device = integ.device,
                                boundary = boundary,
                                progress = progress,
                                progress_desc = progress_desc)
end

end # module
