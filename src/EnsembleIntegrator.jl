module EnsembleIntegrator

using LinearAlgebra
using Random
using CUDA
using Flux

export evolve_sde

# --- 1. Memory Cache for Zero-Allocation Steps ---
struct IntegratorCache{M1, M2}
    drift_out::M1    # Stores output of s(x)
    drift_term::M2   # Stores Phi * s(x)
    noise::M2        # Stores Gaussian noise
    diff_term::M2    # Stores Sigma * noise
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
function integrate_on_device!(x, s_model, Phi, Sigma, dt, T)
    cache = setup_cache(x, s_model)
    sq_2dt = sqrt(2 * dt)
    n_steps = floor(Int, T / dt)

    for _ in 1:n_steps
        em_step!(x, cache, s_model, Phi, Sigma, dt, sq_2dt)
    end
    return x
end

# --- 4. Multi-GPU Orchestration ---
"""
    solve_multi_gpu(x_cpu, model_cpu, Phi, Sigma, dt, T)
Splits the batch across ALL available GPUs, runs integration, and gathers results.
"""
function solve_multi_gpu(x_cpu, model_cpu, Phi, Sigma, dt, T)
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
    evolve_sde(s_model, x0, Phi, Sigma, dt, T; device="cpu")

Main entry point replacing FastSDE.evolve.
"""
function evolve_sde(s_model, x0::AbstractMatrix, Phi, Sigma, dt, T; device="cpu")
    if device == "gpu"
        if CUDA.functional()
            if length(CUDA.devices()) > 1
                println("EnsembleIntegrator: Distributing across $(length(CUDA.devices())) GPUs.")
                return solve_multi_gpu(Array(x0), Flux.cpu(s_model), Phi, Sigma, dt, T)
            else
                # Single GPU path
                println("EnsembleIntegrator: Running on Single GPU.")
                x_gpu = CUDA.cu(x0)
                Phi_gpu = CUDA.cu(Phi)
                
                # Optimization: UpperTriangular wrapper on GPU
                Sigma_gpu = UpperTriangular(CUDA.cu(Matrix(Sigma)))
                
                model_gpu = Flux.gpu(s_model)
                Flux.testmode!(model_gpu) # <--- CRITICAL FIX
                
                integrate_on_device!(x_gpu, model_gpu, Phi_gpu, Sigma_gpu, dt, T)
                return Array(x_gpu)
            end
        else
            @warn "GPU requested but CUDA not functional. Falling back to CPU."
        end
    end

    # CPU Path
    println("EnsembleIntegrator: Running on CPU.")
    
    x_curr = Array(x0)
    Phi_cpu = Array(Phi)
    # Ensure Sigma is explicitly UpperTriangular for CPU speed optimization
    Sigma_cpu = isa(Sigma, UpperTriangular) ? Sigma : UpperTriangular(Array(Sigma))
    
    # Ensure we work on a copy to avoid side effects (e.g. testmode! on training model)
    model_cpu = deepcopy(Flux.cpu(s_model))
    Flux.testmode!(model_cpu)

    integrate_on_device!(x_curr, model_cpu, Phi_cpu, Sigma_cpu, dt, T)
    return x_curr
end

end # module