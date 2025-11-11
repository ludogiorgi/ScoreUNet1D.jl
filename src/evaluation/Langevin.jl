using FastSDE
using Random
using StatsBase

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
end

struct LangevinResult
    trajectory::Array{Float32,3}
    bin_centers::Vector{Float64}
    simulated_pdf::Vector{Float64}
    observed_pdf::Vector{Float64}
    kl_divergence::Float64
end

"""
    run_langevin(model, dataset, cfg)

Integrates the Langevin dynamics `dx = s(x)dt + sqrt(2)dW` using FastSDE and
compares the resulting steady-state PDF with the observed data distribution.
"""
function run_langevin(model, dataset::NormalizedDataset, cfg::LangevinConfig)
    L = sample_length(dataset)
    C = num_channels(dataset)
    dim = L * C
    n_ens = max(cfg.n_ensembles, 1)
    rng = MersenneTwister(cfg.seed)
    init_idx = rand(rng, 1:length(dataset))
    init_sample = Array(dataset.data[:, :, init_idx:init_idx])
    u0 = vec(copy(init_sample))
    u0 = Float32.(u0)

    drift! = function (DU, U, p, t)
        state = reshape(U, L, C, n_ens)
        scores = score_from_model(model, state, cfg.sigma)
        du_view = reshape(DU, L, C, n_ens)
        copyto!(du_view, scores)
        return nothing
    end

    traj = FastSDE.evolve_ens(u0, cfg.dt, cfg.nsteps, drift!, sqrt(2f0);
                              n_ens=n_ens,
                              resolution=cfg.resolution,
                              seed=cfg.seed,
                              timestepper=:euler,
                              batched_drift=true)

    burn_saved = min(size(traj, 2) - 1, fld(cfg.burn_in, cfg.resolution) + 1)
    post_burn = traj[:, burn_saved+1:end, :]
    flattened = reshape(post_burn, dim, :)
    observed = reshape(dataset.data, dim, :)

    sim_modes = select_modes(flattened, L, C, cfg.mode)
    obs_modes = select_modes(observed, L, C, cfg.mode)
    centers, sim_pdf, obs_pdf = compare_pdfs(sim_modes, obs_modes; nbins=cfg.nbins)
    kl = relative_entropy(obs_pdf, sim_pdf)

    return LangevinResult(Float32.(post_burn), centers, sim_pdf, obs_pdf, kl)
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
function compare_pdfs(sim_values::AbstractMatrix, obs_values::AbstractMatrix;
                      nbins::Int=128)
    mins = [val for val in (finite_minimum(sim_values), finite_minimum(obs_values)) if !isnan(val)]
    maxs = [val for val in (finite_maximum(sim_values), finite_maximum(obs_values)) if !isnan(val)]
    if isempty(mins) || isempty(maxs)
        range = (-1.0, 1.0)
    else
        range = (Float64(minimum(mins)), Float64(maximum(maxs)))
    end

    centers, sim_pdf = averaged_histogram(sim_values, nbins, range)
    _, obs_pdf = averaged_histogram(obs_values, nbins, range)
    return centers, sim_pdf, obs_pdf
end

function averaged_histogram(values::AbstractMatrix, nbins::Int, range::Tuple{Float64,Float64})
    nmodes = size(values, 1)
    accum = zeros(Float64, nbins)
    centers = nothing
    for i in 1:nmodes
        centers, pdf = histogram_pdf(view(values, i, :), nbins, range)
        accum .+= pdf
    end
    accum ./= nmodes
    return centers, accum
end

function histogram_pdf(data_view, nbins::Int, bounds::Tuple{Float64,Float64})
    lo, hi = bounds
    if !isfinite(lo) || !isfinite(hi)
        lo, hi = -1.0, 1.0
    elseif lo == hi
        δ = max(abs(lo), 1.0) * 1e-3 + eps(Float64)
        lo -= δ
        hi += δ
    elseif lo > hi
        lo, hi = hi, lo
    end
    edges = collect(range(lo, hi; length=nbins + 1))
    finite_values = filter(isfinite, vec(data_view))
    if isempty(finite_values)
        centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
        return centers, zeros(Float64, nbins)
    end
    hist = fit(Histogram, finite_values, edges; closed=:left)
    weights = copy(hist.weights)
    total = sum(weights)
    pdf = total == 0 ? fill(0.0, nbins) : vec(weights) ./ total
    edges = hist.edges[1]
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return centers, pdf
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
