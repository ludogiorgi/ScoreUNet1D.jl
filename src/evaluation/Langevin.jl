using LinearAlgebra
using Random
using Statistics
using StatsBase
using KernelDensity
using Base.Threads
import CUDA
using Dates

Base.@kwdef mutable struct LangevinConfig
    dt::Float64 = 1e-2              # integrator step
    sample_dt::Float64 = 1e-1       # physical time between stored snapshots
    nsteps::Int = 50_000
    resolution::Int = 10
    n_ensembles::Int = 64
    burn_in::Int = 5_000
    nbins::Int = 128
    sigma::Float32 = 0.05f0
    seed::Int = 21
    mode::Union{Symbol,Int} = :all
    boundary::Union{Nothing,Tuple{Float64,Float64}} = (-10.0, 10.0)
end

struct LangevinResult
    trajectory::Array{Float32,3}
    bin_centers::Vector{Float64}
    simulated_pdf::Vector{Float64}
    observed_pdf::Vector{Float64}
    kl_divergence::Float64
    simulated_acf::Union{Nothing,Vector{Float64}}
    observed_acf::Union{Nothing,Vector{Float64}}
    simulated_time::Union{Nothing,Vector{Float64}}
    observed_time::Union{Nothing,Vector{Float64}}
    decorrelation_time::Union{Nothing,Float64}
end

Base.@kwdef mutable struct CorrelationConfig
    dt::Float64 = 1.0
    max_lag::Int = 512
    stride::Int = 1
    threshold::Float64 = exp(-1)
    multiple::Float64 = 5.0
end

struct CorrelationInfo
    observed::Vector{Float64}
    time::Vector{Float64}
    decorrelation_time::Float64
    plot_lag::Int
    dt::Float64
end

struct DriftWorkspace
    tensor::Array{Float32,3}
    flat::Vector{Float32}
    score::Vector{Float64}
end

function DriftWorkspace(L::Int, C::Int, dim::Int)
    tensor = Array{Float32}(undef, L, C, 1)
    flat = reshape(tensor, :)
    score = zeros(Float64, dim)
    return DriftWorkspace(tensor, flat, score)
end

function create_langevin_drift(model, sigma::Real, L::Int, C::Int, dim::Int;
                               device::ExecutionDevice=CPUDevice(),
                               device_index::Int=1)
    workspaces = [DriftWorkspace(L, C, dim) for _ in 1:nthreads()]
    function drift_single!(du, u, _, _)
        ws = workspaces[threadid()]
        @inbounds @simd for i in 1:dim
            ws.flat[i] = Float32(u[i])
        end
        score_input = device isa GPUDevice ? move_array(ws.tensor, device; device_index=device_index) : ws.tensor
        scores = score_from_model(model, score_input, sigma)
        score_vec = Array(reshape(scores, dim))
        @inbounds @simd for i in 1:dim
            ws.score[i] = Float64(score_vec[i])
        end
        copyto!(du, ws.score)
        return nothing
    end
    drift_single!(du, u, t) = drift_single!(du, u, nothing, t)
    return drift_single!
end

"""
    create_langevin_drift_cpu(model, sigma, L, C, dim, workspaces)

Creates a CPU-optimized drift function for Langevin dynamics.
This version eliminates all GPU transfers and uses thread-local workspaces
for maximum performance.
"""
function create_langevin_drift_cpu(model, sigma::Real, L::Int, C::Int, dim::Int,
                                   workspaces::Vector{DriftWorkspace})
    sigma_f32 = Float32(sigma)
    sigma_tensor = fill(sigma_f32, 1, 1, 1)

    function drift_cpu!(du, u, _, _)
        tid = threadid()
        ws = workspaces[tid]

        # Copy current state into workspace (in-place)
        @inbounds @simd for i in 1:dim
            ws.flat[i] = Float32(u[i])
        end

        # Score evaluation on CPU
        # Model call is thread-safe in Flux.jl
        scores = score_from_model(model, ws.tensor, sigma_f32)

        # Copy scores to output (in-place)
        @inbounds @simd for i in 1:dim
            du[i] = Float64(scores[i])
        end

        return nothing
    end

    # FastSDE compatibility wrapper
    drift_cpu!(du, u, t) = drift_cpu!(du, u, nothing, t)
    return drift_cpu!
end

struct ScoreWrapper{M}
    model::M
    sigma::Float32
    L::Int
    C::Int
    dim::Int
end

Flux.@functor ScoreWrapper (model,)

function (sw::ScoreWrapper)(x::AbstractMatrix)
    B = size(x, 2)
    # Convert to Float32 to match model precision
    x_f32 = Float32.(x)
    x_reshaped = reshape(x_f32, sw.L, sw.C, B)
    scores = score_from_model(sw.model, x_reshaped, sw.sigma)
    # Convert back to match input type for consistency
    return eltype(x).(reshape(scores, sw.dim, B))
end

"""
    run_langevin(model, dataset, cfg)

Integrates the Langevin dynamics `dx = Phi*s(x)dt + sqrt(2)*Sigma*dW` using EnsembleIntegrator
with Phi=Sigma=Identity, which reduces to the standard form `dx = s(x)dt + sqrt(2)dW`.
Uses `cfg.resolution` and `cfg.burn_in` to control how often snapshots are stored and
how many are discarded as burn-in before estimating the steady-state PDF.
"""
function run_langevin(model, dataset::NormalizedDataset, cfg::LangevinConfig,
                      corr_info::Union{Nothing,CorrelationInfo}=nothing;
                      device::ExecutionDevice=CPUDevice())
    # Get dataset dimensions
    L = sample_length(dataset)
    C = num_channels(dataset)
    dim = L * C
    n_ens = max(cfg.n_ensembles, 1)

    # Use identity matrices for Phi and Sigma as specified in AGENTS.md
    Phi = Matrix{Float64}(I, dim, dim)
    Sigma = Matrix{Float64}(I, dim, dim)

    # Derive integration schedule from config
    dt = cfg.dt
    total_steps = max(cfg.nsteps, 1)
    steps_per_snapshot = max(cfg.resolution, 1)
    burn_in_steps = max(cfg.burn_in, 0)

    total_snapshots = fld(total_steps, steps_per_snapshot)
    total_snapshots > 0 || error("resolution too large or nsteps too small to produce any Langevin snapshots")

    burn_in_snapshots = fld(burn_in_steps, steps_per_snapshot)
    keep_snapshots = max(total_snapshots - burn_in_snapshots, 0)
    keep_snapshots > 0 || error("Burn-in removes all Langevin samples. Increase nsteps or reduce burn_in/resolution.")

    # Prefer cfg.sample_dt if it matches dt * resolution, otherwise fall back to that product.
    segment_time = dt * steps_per_snapshot
    if cfg.sample_dt > 0
        # Only warn if there's a significant mismatch; keep dt/resolution authoritative
        rel_err = abs(cfg.sample_dt - segment_time) / max(segment_time, eps(Float64))
        rel_err > 1e-6 && @warn "LangevinConfig.sample_dt is inconsistent with dt * resolution; using dt * resolution" dt cfg_sample_dt=cfg.sample_dt resolution=steps_per_snapshot
    end

    total_time = segment_time * total_snapshots

    # Determine device string for evolve_sde
    device_str = is_gpu(device) ? "gpu" : "cpu"

    # Prepare score model wrapper
    # Ensure model is on CPU initially for the wrapper, EnsembleIntegrator handles moving to GPU
    cpu_model = Flux.cpu(model)
    wrapper = ScoreWrapper(cpu_model, cfg.sigma, L, C, dim)

    # Seed RNG for reproducible ensemble initialization
    Random.seed!(cfg.seed)

    # Sample initial conditions
    # We need a (dim, n_ens) matrix
    x0 = Matrix{Float64}(undef, dim, n_ens)
    for i in 1:n_ens
        idx = rand(1:length(dataset))
        # dataset.data is (L, C, N)
        x0[:, i] = reshape(dataset.data[:, :, idx], dim)
    end

    @info "Running $(n_ens) ensembles using EnsembleIntegrator" device=device_str dt=dt nsteps=total_steps resolution=steps_per_snapshot burn_in=burn_in_steps snapshots=keep_snapshots total_time=total_time

    # Integrate using evolve_sde in segments, storing snapshots every `steps_per_snapshot`
    traj = Array{Float32}(undef, dim, keep_snapshots, n_ens)
    x_curr = x0
    snapshot_idx = 0
    for step_block in 1:total_snapshots
        # Advance by one snapshot interval
        x_curr = evolve_sde(wrapper, x_curr, Phi, Sigma, dt, segment_time; device=device_str)
        if step_block > burn_in_snapshots
            snapshot_idx += 1
            traj[:, snapshot_idx, :] .= Float32.(x_curr)
        end
    end

    flattened = reshape(traj, dim, :)
    observed = reshape(dataset.data, dim, :)

    sim_modes = select_modes(flattened, L, C, cfg.mode)
    obs_modes = select_modes(observed, L, C, cfg.mode)

    obs_min = Float64(minimum(observed))
    obs_max = Float64(maximum(observed))
    if obs_min == obs_max
        δ = max(abs(obs_min), 1.0) * 1e-3
        obs_min -= δ
        obs_max += δ
    elseif obs_min > obs_max
        obs_min, obs_max = obs_max, obs_min
    end

    centers, sim_pdf, obs_pdf, sim_raw, obs_raw = compare_pdfs(sim_modes, obs_modes;
                                                               nbins=cfg.nbins,
                                                               bounds=(obs_min, obs_max))
    kl = relative_entropy(obs_raw, sim_raw)
    log_kl_history(kl, cfg)

    # Autocorrelation analysis (simplified - would need full trajectory)
    sim_acf = nothing
    obs_acf = nothing
    sim_time = nothing
    obs_time = nothing
    decor_time = nothing

    return LangevinResult(traj, centers, sim_pdf, obs_pdf, kl,
                          sim_acf, obs_acf, sim_time, obs_time, decor_time)
end



function log_kl_history(kl::Real, cfg::LangevinConfig)
    runs_dir = normpath(joinpath(@__DIR__, "..", "..", "runs"))
    mkpath(runs_dir)
    history_path = joinpath(runs_dir, "kl_history.csv")
    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    open(history_path, "a") do io
        println(io, "$timestamp,$(Float64(kl)),$(cfg.n_ensembles),$(cfg.nsteps),$(cfg.resolution)")
    end
end

function finite_minimum(values)
    min_val = Inf
    found = false
    @inbounds for v in values
        if isfinite(v)
            found = true
            if v < min_val
                min_val = v
            end
        end
    end
    return found ? min_val : NaN
end

function finite_maximum(values)
    max_val = -Inf
    found = false
    @inbounds for v in values
        if isfinite(v)
            found = true
            if v > max_val
                max_val = v
            end
        end
    end
    return found ? max_val : NaN
end

function select_modes(values::AbstractMatrix, L::Int, C::Int, mode::Symbol)
    mode === :all && return values
    throw(ArgumentError("Unsupported mode selector $mode"))
end

function select_modes(values::AbstractMatrix, L::Int, C::Int, mode::Integer)
    1 <= mode <= C || throw(ArgumentError("Mode index must be between 1 and $C"))
    start = (mode - 1) * L + 1
    stop = start + L - 1
    return values[start:stop, :]
end

"""
    compare_pdfs(sim_values, obs_values; nbins=128)

Computes averaged PDFs (over modes) for simulated and observed data.
"""
function flatten_samples(values::AbstractMatrix)
    return Float64.(vec(values))
end

function compare_pdfs(sim_values::AbstractMatrix, obs_values::AbstractMatrix;
                      nbins::Int=128, bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    sim_flat = flatten_samples(sim_values)
    obs_flat = flatten_samples(obs_values)
    support = bounds
    if support === nothing
        mins = filter(!isnan, (minimum(sim_flat), minimum(obs_flat)))
        maxs = filter(!isnan, (maximum(sim_flat), maximum(obs_flat)))
        support = isempty(mins) || isempty(maxs) ? (-1.0, 1.0) :
            (Float64(minimum(mins)), Float64(maximum(maxs)))
    else
        support = bounds
    end
    grid = range(support[1], support[2]; length=nbins)
    sim_kde = kde(sim_flat, grid)
    obs_kde = kde(obs_flat, grid)
    centers = sim_kde.x
    sim_pdf = sim_kde.density
    obs_pdf = obs_kde.density
    dx = length(centers) > 1 ? (centers[2] - centers[1]) : 1.0
    sim_mass = sim_pdf .* dx
    obs_mass = obs_pdf .* dx
    sim_mass ./= sum(sim_mass)
    obs_mass ./= sum(obs_mass)
    return centers, sim_pdf, obs_pdf, sim_mass, obs_mass
end

function pair_samples(tensor::Array{Float32,3}, j::Int)
    L, C, B = size(tensor)
    L > j || error("Lag j=$j exceeds spatial length L=$L")
    total = (L - j) * C * B
    xs = Vector{Float64}(undef, total)
    ys = Vector{Float64}(undef, total)
    idx = 1
    @inbounds for b in 1:B, c in 1:C, i in 1:(L - j)
        xs[idx] = Float64(tensor[i, c, b])
        ys[idx] = Float64(tensor[i + j, c, b])
        idx += 1
    end
    return xs, ys
end

function kde_heatmap(tensor::Array{Float32,3}, j::Int,
                     bounds::Tuple{Float64,Float64}, npoints::Int)
    xs, ys = pair_samples(tensor, j)
    kd = kde(xs, ys;
             xmin=bounds[1], xmax=bounds[2],
             ymin=bounds[1], ymax=bounds[2],
             npoints=(npoints, npoints))
    return kd.x, kd.y, kd.density
end

"""
    relative_entropy(p, q)

Computes KL(p || q) with numerical stabilization.
"""
function relative_entropy(p::AbstractVector, q::AbstractVector)
    length(p) == length(q) || throw(ArgumentError("PDFs must have the same length"))
    eps_val = 1e-12
    acc = 0.0
    @inbounds for i in eachindex(p)
        pi = max(p[i], eps_val)
        qi = max(q[i], eps_val)
        acc += pi * log(pi / qi)
    end
    return acc
end

function compute_stein_matrix(model,
                              dataset::NormalizedDataset,
                              sigma::Real;
                              batch_size::Int=256,
                              device::ExecutionDevice=CPUDevice(),
                              sample_stride::Int=1)
    L = sample_length(dataset)
    C = num_channels(dataset)
    dim = L * C
    idxs = collect(1:sample_stride:length(dataset))
    isempty(idxs) && error("No samples available for Stein estimation")
    batches = collect(Iterators.partition(idxs, batch_size))
    total = 0

    if device isa GPUDevice && gpu_count(device) > 1
        ndev = min(gpu_count(device), length(batches))
        partials = [zeros(Float64, dim, dim) for _ in 1:ndev]
        counts = zeros(Int, ndev)
        Threads.@threads for worker in 1:ndev
            gpu_idx = worker
            local_model = move_model(model, device; device_index=gpu_idx)
            local_V = partials[gpu_idx]
            local_count = 0
            for batch_id in worker:ndev:length(batches)
                batch_idxs = batches[batch_id]
                batch_cpu = Array(dataset.data[:, :, batch_idxs])
                dev_batch = move_array(batch_cpu, device; device_index=gpu_idx)
                
                # Add noise to sample from p_sigma
                noise = similar(dev_batch)
                CUDA.randn!(noise)
                noisy_batch = dev_batch .+ sigma .* noise
                
                scores = score_from_model(local_model, noisy_batch, sigma)
                score_mat = Array(reshape(scores, dim, size(batch_cpu, 3)))
                x_mat = Array(reshape(noisy_batch, dim, size(batch_cpu, 3)))
                local_V .+= score_mat * transpose(x_mat)
                local_count += length(batch_idxs)
            end
            partials[gpu_idx] = local_V
            counts[gpu_idx] = local_count
        end
        total = sum(counts)
        total > 0 || error("No samples processed for Stein estimation")
        return reduce(+, partials) ./ total
    elseif device isa GPUDevice
        model_on_device = move_model(model, device)
        stein_acc = zeros(Float64, dim, dim)
        for batch_idxs in batches
            batch_cpu = Array(dataset.data[:, :, batch_idxs])
            dev_batch = move_array(batch_cpu, device)
            
            # Add noise to sample from p_sigma
            noise = similar(dev_batch)
            CUDA.randn!(noise)
            noisy_batch = dev_batch .+ sigma .* noise
            
            scores = score_from_model(model_on_device, noisy_batch, sigma)
            score_mat = Array(reshape(scores, dim, size(batch_cpu, 3)))
            x_mat = Array(reshape(noisy_batch, dim, size(batch_cpu, 3)))
            stein_acc .+= score_mat * transpose(x_mat)
            total += length(batch_idxs)
        end
        total > 0 || error("No samples processed for Stein estimation")
        return stein_acc ./ total
    else
        model_on_device = model
        nworkers = nthreads()
        partials = [zeros(Float64, dim, dim) for _ in 1:nworkers]
        counts = zeros(Int, nworkers)
        Threads.@threads for batch_id in 1:length(batches)
            tid = threadid()
            batch_idxs = batches[batch_id]
            batch_cpu = Array(dataset.data[:, :, batch_idxs])
            
            # Add noise to sample from p_sigma
            noise = randn(eltype(batch_cpu), size(batch_cpu))
            noisy_batch = batch_cpu .+ sigma .* noise
            
            scores = score_from_model(model_on_device, noisy_batch, sigma)
            score_mat = Float64.(reshape(scores, dim, size(batch_cpu, 3)))
            x_mat = Float64.(reshape(noisy_batch, dim, size(batch_cpu, 3)))
            partials[tid] .+= score_mat * transpose(x_mat)
            counts[tid] += length(batch_idxs)
        end
        total = sum(counts)
        total > 0 || error("No samples processed for Stein estimation")
        return reduce(+, partials) ./ total
    end
end

function average_mode_acf(values::AbstractMatrix{<:Real}, max_lag::Int)
    D, T = size(values)
    max_lag = min(max_lag, T - 1)
    max_lag >= 0 || error("Insufficient samples to compute autocorrelation")
    acf = zeros(Float64, max_lag + 1)
    for d in 1:D
        series = values[d, :]
        μ = mean(series)
        centered = series .- μ
        variance = sum(abs2, centered) / max(T, 1)
        variance <= eps(Float64) && continue
        for lag in 0:max_lag
            total = T - lag
            total <= 0 && break
            acf[lag + 1] += dot(view(centered, 1:total),
                                view(centered, 1 + lag:lag + total)) / (total * variance)
        end
    end
    acf ./= D
    return acf
end

function compute_correlation_info(dataset::NormalizedDataset,
                                  cfg::CorrelationConfig)
    L = sample_length(dataset)
    C = num_channels(dataset)
    stride = max(cfg.stride, 1)
    full_matrix = reshape(dataset.data, L * C, :)
    subsampled = full_matrix[:, 1:stride:end]
    samples = size(subsampled, 2)
    samples > 1 || error("Not enough samples to compute autocorrelation")
    max_lag = min(cfg.max_lag, samples - 1)
    acf_full = average_mode_acf(subsampled, max_lag)
    idx = findfirst(i -> acf_full[i] <= cfg.threshold, 1:length(acf_full))
    decor_idx = idx === nothing ? length(acf_full) : idx
    decor_time = (decor_idx - 1) * cfg.dt
    target_time = min(cfg.multiple * decor_time, max_lag * cfg.dt)
    plot_lag = clamp(ceil(Int, target_time / cfg.dt), 1, max_lag)
    time_axis = (0:plot_lag) .* cfg.dt
    observed = acf_full[1:plot_lag + 1]
    return CorrelationInfo(observed, time_axis, decor_time, plot_lag, cfg.dt)
end
