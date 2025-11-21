using FastSDE
using LinearAlgebra
using Random
using StatsBase
using KernelDensity
using Base.Threads
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
    phi::Union{Nothing,Matrix{Float64}} = nothing
    diffusion::Union{Nothing,Matrix{Float64}} = nothing
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

struct DriftWorkspace
    tensor::Array{Float32,3}
    flat::Vector{Float32}
    score::Vector{Float64}
    drift::Union{Nothing,Vector{Float64}}
end

function DriftWorkspace(L::Int, C::Int, dim::Int, need_drift::Bool)
    tensor = Array{Float32}(undef, L, C, 1)
    flat = reshape(tensor, :)
    score = zeros(Float64, dim)
    drift = need_drift ? zeros(Float64, dim) : nothing
    return DriftWorkspace(tensor, flat, score, drift)
end

function create_langevin_drift(model, sigma::Real, phi::Union{Nothing,Matrix{Float64}},
                               L::Int, C::Int, dim::Int)
    workspaces = [DriftWorkspace(L, C, dim, phi !== nothing) for _ in 1:nthreads()]
    function drift_single!(du, u, _, _)
        ws = workspaces[threadid()]
        @inbounds @simd for i in 1:dim
            ws.flat[i] = Float32(u[i])
        end
        scores = score_from_model(model, ws.tensor, sigma)
        score_vec = reshape(scores, dim)
        @inbounds @simd for i in 1:dim
            ws.score[i] = Float64(score_vec[i])
        end
        if phi === nothing
            copyto!(du, ws.score)
        else
            tmp = ws.drift::Vector{Float64}
            mul!(tmp, phi, ws.score)
            copyto!(du, tmp)
        end
        return nothing
    end
    drift_single!(du, u, t) = drift_single!(du, u, nothing, t)
    return drift_single!
end

function make_diffusion(sigma_matrix::Union{Nothing,Matrix{Float64}})
    if sigma_matrix === nothing
        return sqrt(2.0)
    else
        return sqrt(2.0) .* sigma_matrix
    end
end

"""
    run_langevin(model, dataset, cfg)

Integrates the Langevin dynamics `dx = s(x)dt + sqrt(2)dW` using FastSDE and
compares the resulting steady-state PDF with the observed data distribution.
"""
function run_langevin(model, dataset::NormalizedDataset, cfg::LangevinConfig,
                      corr_info::Union{Nothing,CorrelationInfo}=nothing)
    L = sample_length(dataset)
    C = num_channels(dataset)
    dim = L * C
    n_ens = max(cfg.n_ensembles, 1)
    phi = cfg.phi
    sigma_matrix = cfg.diffusion
    diffusion_arg = make_diffusion(sigma_matrix)
    dt = Float64(cfg.dt)

    drift_single! = create_langevin_drift(model, cfg.sigma, phi, L, C, dim)

    trajectories = Vector{Matrix{Float64}}(undef, n_ens)
    Threads.@threads for ens in 1:n_ens
        tid = threadid()
        local_rng = MersenneTwister(cfg.seed + 10_000 * tid + ens)
        idx = rand(local_rng, 1:length(dataset))
        sample = Array(dataset.data[:, :, idx:idx])
        u0 = Float64.(reshape(sample, dim))
        seed = rand(local_rng, 1:typemax(Int))
        traj = FastSDE.evolve(u0, dt, cfg.nsteps, drift_single!, diffusion_arg;
                              resolution=cfg.resolution,
                              seed=seed,
                              timestepper=:euler,
                              boundary=cfg.boundary,
                              flatten=true,
                              manage_blas_threads=true)
        trajectories[ens] = Array(traj)
    end

    snapshots_per_traj = div(cfg.nsteps, cfg.resolution) + 1
    drop_cols = min(snapshots_per_traj, fld(cfg.burn_in, cfg.resolution) + 1)
    keep_cols = snapshots_per_traj - drop_cols
    keep_cols > 0 || error("Burn-in removes all Langevin samples. Increase nsteps or reduce burn_in.")
    traj = Array{Float32}(undef, dim, keep_cols, n_ens)
    for (ens, mat) in enumerate(trajectories)
        start_idx = drop_cols + 1
        traj[:, :, ens] .= Float32.(mat[:, start_idx:end])
    end
    post_burn = traj
    flattened = reshape(post_burn, dim, :)
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

    sim_acf = nothing
    obs_acf = nothing
    sim_time = nothing
    obs_time = nothing
    decor_time = nothing
    if corr_info !== nothing
        obs_steps = min(corr_info.plot_lag, length(corr_info.observed) - 1)
        obs_steps = max(obs_steps, 0)
        obs_acf = corr_info.observed[1:obs_steps + 1]
        obs_time = corr_info.time[1:obs_steps + 1]
        target_time = obs_time[end]
        sim_dt = cfg.dt * cfg.resolution
        max_sim_steps = max(size(flattened, 2) - 1, 0)
        sim_steps = clamp(floor(Int, target_time / sim_dt), 0, max_sim_steps)
        sim_series = view(flattened, :, 1:sim_steps + 1)
        sim_acf = average_mode_acf(sim_series, sim_steps)
        sim_time = (0:sim_steps) .* sim_dt
        decor_time = corr_info.decorrelation_time
    end

    return LangevinResult(Float32.(post_burn), centers, sim_pdf, obs_pdf, kl,
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
