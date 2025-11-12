using FastSDE
using LinearAlgebra
using Random
using StatsBase
using KernelDensity

Base.@kwdef mutable struct LangevinConfig
    dt::Float64 = 1e-2
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
    acf_time::Union{Nothing,Vector{Float64}}
    decorrelation_time::Union{Nothing,Float64}
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
    rng = MersenneTwister(cfg.seed)
    phi = cfg.phi
    sigma_matrix = cfg.diffusion
    tensor_buf = Array{Float32}(undef, L, C, 1)
    tensor_flat = reshape(tensor_buf, :)
    score64 = zeros(Float64, dim)
    drift_out = zeros(Float64, dim)

    function compute_drift!(du, u)
        @inbounds for i in 1:dim
            tensor_flat[i] = Float32(u[i])
        end
        scores = score_from_model(model, tensor_buf, cfg.sigma)
        score_vec32 = reshape(scores, dim)
        @inbounds for i in 1:dim
            score64[i] = Float64(score_vec32[i])
        end
        if phi === nothing
            @inbounds @simd for i in 1:dim
                du[i] = score64[i]
            end
        else
            mul!(drift_out, phi, score64)
            @inbounds @simd for i in 1:dim
                du[i] = drift_out[i]
            end
        end
        return nothing
    end
    drift_single!(du, u, p, t) = compute_drift!(du, u)
    drift_single!(du, u, t) = compute_drift!(du, u)

    diffusion_arg = sigma_matrix === nothing ? sqrt(2.0) :
        (sqrt(2.0) .* sigma_matrix)

    function integrate_once(seed::Int, u0::Vector{Float64})
        FastSDE.evolve(u0, cfg.dt, cfg.nsteps, drift_single!, diffusion_arg;
                       resolution=cfg.resolution,
                       seed=seed,
                       timestepper=:euler,
                       boundary=cfg.boundary)
    end

    initial_idx = rand(rng, 1:length(dataset))
    init_sample = Array(dataset.data[:, :, initial_idx:initial_idx])
    u0 = Float64.(vec(init_sample))
    traj_sample = integrate_once(rand(rng, 1:typemax(Int)), u0)
    saves = size(traj_sample, 2)
    traj = Array{Float32}(undef, dim, saves, n_ens)
    traj[:, :, 1] .= Float32.(traj_sample)

    for ens in 2:n_ens
        idx = rand(rng, 1:length(dataset))
        sample = Array(dataset.data[:, :, idx:idx])
        u0 = Float64.(vec(sample))
        seed = rand(rng, 1:typemax(Int))
        traj_single = integrate_once(seed, u0)
        traj[:, :, ens] .= Float32.(traj_single)
    end

    burn_saved = min(size(traj, 2) - 1, fld(cfg.burn_in, cfg.resolution) + 1)
    post_burn = traj[:, burn_saved+1:end, :]
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

    sim_acf = nothing
    obs_acf = nothing
    acf_time = nothing
    decor_time = nothing
    if corr_info !== nothing
        sim_matrix = reshape(Float32.(flattened), dim, :)
        available_lag = min(corr_info.plot_lag, size(sim_matrix, 2) - 1)
        if available_lag >= 0
            sim_acf = average_mode_acf(sim_matrix, available_lag)
            obs_acf = corr_info.observed[1:available_lag + 1]
            acf_time = corr_info.time[1:available_lag + 1]
        end
        decor_time = corr_info.decorrelation_time
    end

    return LangevinResult(Float32.(post_burn), centers, sim_pdf, obs_pdf, kl,
                          sim_acf, obs_acf, acf_time, decor_time)
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
