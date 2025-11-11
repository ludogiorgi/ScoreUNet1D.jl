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
    rng = MersenneTwister(cfg.seed)
    init_idx = rand(rng, 1:length(dataset))
    init_sample = Array(dataset.data[:, :, init_idx:init_idx])
    u0 = Float64.(vec(init_sample))

    tensor_buf = Array{Float32}(undef, L, C, 1)
    flat_tensor = reshape(tensor_buf, :)
    score_flat = similar(flat_tensor)

    drift! = function (du, u, t)
        @inbounds @simd for i in eachindex(u)
            flat_tensor[i] = Float32(u[i])
        end
        score = score_from_model(model, tensor_buf, cfg.sigma)
        score_flat .= vec(score)
        @inbounds @simd for i in eachindex(du)
            du[i] = Float64(score_flat[i])
        end
    end

    traj = FastSDE.evolve_ens(u0, cfg.dt, cfg.nsteps, drift!, sqrt(2.0);
                              n_ens=cfg.n_ensembles,
                              resolution=cfg.resolution,
                              seed=cfg.seed,
                              timestepper=:euler)

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
    min_val = min(minimum(sim_values), minimum(obs_values))
    max_val = max(maximum(sim_values), maximum(obs_values))
    range = (Float64(min_val), Float64(max_val))

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
    edges = collect(range(bounds[1], bounds[2]; length=nbins + 1))
    hist = fit(Histogram, vec(data_view), edges; closed=:left)
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
