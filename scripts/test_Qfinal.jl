#########################
# Packages & imports
#########################

# nohup julia --project=. scripts/test_Q.jl > test_Q.log 2>&1 &

using LinearAlgebra
using Statistics
using CairoMakie

using MarkovChainHammer
using MarkovChainHammer.TransitionMatrix: generator

using ScoreUNet1D
using ScoreUNet1D: NormalizedDataset, apply_stats,
    ScoreWrapper, build_snapshot_integrator, plot_langevin_vs_observed

using StateSpacePartitions
using StateSpacePartitions.Trees
using Plots

using BSON

#########################
# Data loading
#########################


"""
    load_low_dataset(; resolution=1)

Load the low-resolution KS dataset and optionally subsample in time.
"""
function load_low_dataset(; resolution::Int = 1)
    base = load_hdf5_dataset(LOW_DATA_PATH;
                             dataset_key = DATASET_KEY,
                             samples_orientation = :columns)

    data = base.data
    if resolution > 1
        @info "Subsampling low-resolution dataset" stride = resolution original_size = size(data, 3)
        data = data[:, :, 1:resolution:end]
        @info "After subsampling" new_size = size(data, 3)
    end

    return NormalizedDataset(data, base.stats)
end

"""
    load_datasets(; resolution_low=1)

Load both low- and high-resolution datasets.
The high-resolution data are normalized with the statistics of the low dataset.
"""
function load_datasets(; resolution_low::Int = 1)
    low = load_low_dataset(; resolution = resolution_low)

    hr_raw = load_hdf5_dataset(HR_DATA_PATH;
                               dataset_key = DATASET_KEY,
                               samples_orientation = :columns,
                               normalize = false)
    hr_norm = apply_stats(hr_raw.data, low.stats)
    hr_dataset = NormalizedDataset(hr_norm, low.stats)
    return low, hr_dataset
end

"""
    extract_trajectory(dataset)

Return the 2D trajectory matrix `X` (d × T) and 3D field `u` (d × 1 × T).
"""
function extract_trajectory(dataset::NormalizedDataset)
    u = dataset.data           # (nx, 1, nt)
    nx, _, nt = size(u)
    X = reshape(u, nx, nt)
    return X, u
end

#########################
# Observables for KS
#########################

# Compute the three observables used in Fig. 11:
# 1) u(0, t)
# 2) ∫ u(x,t)² dx   (energy, up to a constant factor)
# 3) Re[û(k_four)]  ("four oscillations" mode – choose an index)
function ks_observables(u::Array{<:Real,3}; mode_index::Int = 4)
    nx, _, nt = size(u)

    obs_u0      = zeros(Float64, nt)
    obs_energy  = zeros(Float64, nt)
    obs_fourier = zeros(Float64, nt)

    # Single spatial Fourier mode via precomputed weights
    k = clamp(mode_index, 1, nx)
    ω = -2π * (k - 1) / nx
    weights = Vector{ComplexF64}(undef, nx)
    @inbounds for j in 1:nx
        weights[j] = cis(ω * (j - 1))
    end

    Base.Threads.@threads for t in 1:nt
        ut = view(u, :, 1, t)

        obs_u0[t] = ut[1]

        s_energy = 0.0
        s_fourier = 0.0 + 0.0im
        @inbounds for j in 1:nx
            val = ut[j]
            s_energy += val * val
            s_fourier += val * weights[j]
        end

        obs_energy[t] = s_energy / nx
        obs_fourier[t] = real(s_fourier)
    end

    return Dict(
        :u0      => obs_u0,
        :energy  => obs_energy,
        :fourier => obs_fourier,
    )
end

#########################
# Empirical ACF from time series
#########################

"""
    empirical_acf(x, max_lag; normalize=false)

Compute the empirical autocovariance (or normalized ACF) of a scalar time series.
"""
function empirical_acf(x::AbstractVector, max_lag::Int; normalize::Bool = false)
    x̄ = mean(x)
    xc = x .- x̄
    var0 = dot(xc, xc) / max(length(xc) - 1, 1)

    acf = zeros(Float64, max_lag + 1)
    for lag in 0:max_lag
        n = length(xc) - lag
        if n <= 1
            acf[lag + 1] = NaN
            continue
        end
        acf[lag + 1] = dot(@view(xc[1:n]), @view(xc[1+lag:lag+n])) / max(n - 1, 1)
    end
    return normalize && var0 > 0 ? acf ./ var0 : acf
end

#########################
# Stationary distribution and cluster averages
#########################

function stationary_from_labels(labels::AbstractVector{<:Integer}, n_states::Int)
    counts = zeros(Float64, n_states)
    for s in labels
        counts[s] += 1
    end
    π = counts ./ sum(counts)
    return π
end

function cluster_means(g_time::AbstractVector,
                       labels::AbstractVector{<:Integer},
                       n_states::Int)
    @assert length(g_time) == length(labels)
    sums   = zeros(Float64, n_states)
    counts = zeros(Int, n_states)

    @inbounds for (g, s) in zip(g_time, labels)
        sums[s]   += g
        counts[s] += 1
    end

    @inbounds for i in 1:n_states
        if counts[i] > 0
            sums[i] /= counts[i]
        else
            sums[i] = 0.0
        end
    end
    return sums
end

#########################
# Generator-based ACF (full Q)
#########################

struct GeneratorEigenCache{T}
    V::Matrix{T}
    Vinv::Matrix{T}
    λ::Vector{T}
end

function GeneratorEigenCache(Q::AbstractMatrix)
    eig = eigen(Matrix(Q))
    V = eig.vectors
    Vinv = inv(V)
    λ = eig.values
    return GeneratorEigenCache(V, Vinv, λ)
end

function generator_acf(cache::GeneratorEigenCache,
                       π::AbstractVector,
                       g_vals::AbstractVector;
                       dt::Real = 1.0,
                       max_lag::Int = 200,
                       normalize::Bool = false)

    π = π ./ sum(π)
    M = Diagonal(π)

    μ = dot(g_vals, π)
    g̃ = g_vals .- μ
    v0 = M * g̃

    V    = cache.V
    Vinv = cache.Vinv
    λ    = cache.λ

    v0̂ = Vinv * v0
    acf = zeros(ComplexF64, max_lag + 1)
    for (k, lag) in enumerate(0:max_lag)
        t = lag * dt
        ŵ = exp.(λ .* t) .* v0̂
        w  = V * ŵ
        acf[k] = dot(g̃, w)
    end

    acf_real = real.(acf)
    return normalize ? acf_real ./ acf_real[1] : acf_real
end

function generator_acf(Q::AbstractMatrix,
                       π::AbstractVector,
                       g_vals::AbstractVector;
                       dt::Real = 1.0,
                       max_lag::Int = 200,
                       normalize::Bool = false)
    n = size(Q, 1)
    if n <= 1000
        cache = GeneratorEigenCache(Q)
        return generator_acf(cache, π, g_vals; dt = dt, max_lag = max_lag, normalize = normalize)
    else
        return generator_acf_direct(Q, π, g_vals; dt = dt, max_lag = max_lag, normalize = normalize)
    end
end

"""
    generator_acf_direct(Q, π, g_vals; dt, max_lag, expQtdt=nothing, normalize=false)

Direct ACF computation for large Q using exp(Qᵀ dt) and repeated matrix-vector products.
"""
function generator_acf_direct(Q::AbstractMatrix,
                              π::AbstractVector,
                              g_vals::AbstractVector;
                              dt::Real = 1.0,
                              max_lag::Int = 200,
                              expQtdt::Union{Nothing, AbstractMatrix} = nothing,
                              normalize::Bool = false)
    n = size(Q, 1)
    π_norm = π ./ sum(π)
    M = Diagonal(π_norm)

    μ = dot(g_vals, π_norm)
    g_centered = g_vals .- μ
    v0 = M * g_centered

    acf = zeros(Float64, max_lag + 1)
    acf[1] = dot(g_centered, v0)

    if expQtdt === nothing
        Qt = Matrix(transpose(Q))
        @info "Computing exp(Q'*dt) for ACF propagation" n_states = n
        expQtdt = exp(Qt .* dt)
    end

    w = Vector{Float64}(v0)
    for lag in 1:max_lag
        w = expQtdt * w
        acf[lag + 1] = real(dot(g_centered, w))
    end

    return normalize ? acf ./ acf[1] : acf
end

#########################
# Clustering: StateSpacePartitions
#########################

"""
    cluster_trajectory(X; minimum_probability)

Cluster the trajectory `X` (d × T) into Markov states using an unstructured
tree partition with probability-based stopping.
"""
function cluster_trajectory(X::AbstractMatrix;
                            minimum_probability::Real)
    @info "Clustering trajectory" minimum_probability = minimum_probability

    tree_method = Tree(false, minimum_probability)
    partition = StateSpacePartition(X; method = tree_method)

    labels = partition.partitions
    actual_n_states = maximum(labels)

    @info "Partitioning complete" actual_n_states = actual_n_states

    d = size(X, 1)
    centers = zeros(Float64, d, actual_n_states)
    counts = zeros(Int, actual_n_states)

    for (i, s) in enumerate(labels)
        centers[:, s] .+= X[:, i]
        counts[s] += 1
    end

    for j in 1:actual_n_states
        if counts[j] > 0
            centers[:, j] ./= counts[j]
        end
    end

    return labels, centers, actual_n_states
end

#########################
# Φ–Σ construction from Q (phi_sigma.txt)
#########################

function load_score_model()
    contents = BSON.load(MODEL_PATH)
    model = contents[:model]
    sigma =
        haskey(contents, :trainer_cfg) ? contents[:trainer_cfg].sigma :
        0.05
    trainer_cfg = ScoreUNet1D.ScoreTrainerConfig(; sigma = sigma)
    return model, trainer_cfg
end

"""
    construct_phi_sigma(Q, centers, π;
                        regularization=1e-6, drift_scale=1.0)

Construct drift matrix Φ and diffusion factor Σ from generator Q using the
relations derived in `phi_sigma.txt`, combined with Stein's identity.
In the notation of the text, this corresponds to using V ≈ -I so that
Φ ≈ -M, where

    M_{i,j} = ∑_{n,m} x_j^n π_n x_i^m Q_{mn}.
"""
function construct_phi_sigma(Q::AbstractMatrix,
                             centers::AbstractMatrix,
                             π::AbstractVector;
                             regularization::Real = 1e-6,
                             drift_scale::Real = 1.0)
    D, n_states = size(centers)
    @assert size(Q) == (n_states, n_states)
    @assert length(π) == n_states

    π_norm = π ./ sum(π)

    # M_{i,j} = Σ_{n,m} x_j^n π_n x_i^m Q_{mn}
    # With centers[:, n] = x^n this can be written as
    #   M = centers * Q * Diagonal(π) * centers'
    # since
    #   (centers * Q * Diagonal(π) * centers')_{i,j}
    #     = Σ_{m,n} x_i^m Q_{m n} π_n x_j^n
    # which matches the definition in phi_sigma.txt.
    M = centers * Q * Diagonal(π_norm) * centers'
    # By Stein's identity, V ≈ -I, so Φ ≈ -M.
    Φ = - M

    Φ_S = 0.5 * (Φ + Φ')
    Φ_A = 0.5 * (Φ - Φ')

    @info "Φ decomposition" norm_symmetric = norm(Φ_S) norm_antisymmetric = norm(Φ_A)

    # Diffusion factor from the symmetric part: Φ_S ≈ Σ Σᵀ.
    # Apply a simple Tikhonov-type regularization on the diagonal to
    # ensure positive definiteness, then take the upper Cholesky factor
    # to match the usage of `UpperTriangular(Σ)` in the integrator.
    Φ_S_reg = Φ_S #+ regularization * I
    chol_U = cholesky(Symmetric(Φ_S_reg), check = true).U
    Σ = Matrix(chol_U)

    return Φ, Σ
end

#########################
# Configuration constants
#########################
PROJECT_ROOT  = dirname(@__DIR__)
LOW_DATA_PATH = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
HR_DATA_PATH  = joinpath(PROJECT_ROOT, "data", "new_ks_hr.hdf5")
MODEL_PATH = joinpath(PROJECT_ROOT, "scripts", "model.bson")
DATASET_KEY   = "timeseries"

# Physical sampling intervals
DT_LOW = 1  # time step for new_ks.hdf5
DT_HR  = 0.1    # time step for new_ks_hr.hdf5

Q_MIN_PROB      = 1e-4
Q_MAX_LAG       = 500
Q_MODE_INDEX    = 4

LANGEVIN_TOTAL_STEPS = 5_000
LANGEVIN_BURN_IN     = 0
LANGEVIN_DT          = 0.01
LANGEVIN_RESOLUTION  = 100
LANGEVIN_ENSEMBLES   = 1
LANGEVIN_MAX_LAG     = 500
LANGEVIN_REGULARIZATION = 1e-3
LANGEVIN_DRIFT_SCALE    = 1e-2
USE_GPU = false

#########################
# ACF from data vs full Q (low dataset)
#########################

# Load datasets
@info "Loading datasets (low and high resolution)..."
low_dataset, hr_dataset = load_datasets()
X_hr, u_hr = extract_trajectory(hr_dataset)
D_hr, T_hr = size(X_hr)
@info "High-resolution dataset loaded" D_hr = D_hr T_hr = T_hr

# Observables and empirical ACFs on low dataset
@info "Computing KS observables on high-resolution dataset..."
obs_dict_hr  = ks_observables(u_hr; mode_index = Q_MODE_INDEX)
obs_names = [:u0, :energy, :fourier]

acf_data_hr = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    acf_data_hr[obs] = empirical_acf(obs_dict_hr[obs], Q_MAX_LAG; normalize = true)
end

# Cluster low-resolution trajectory and build Q
@info "Clustering low-resolution trajectory and building generator Q..."
labels, centers, n_states = cluster_trajectory(X_hr; minimum_probability = Q_MIN_PROB)
mc = reshape(labels, 1, :)
Q = generator(mc; dt = DT_HR)
pi_vec = stationary_from_labels(labels, n_states)
@info "Generator Q constructed" n_states = n_states
Plots.plot([Q[i,i] for i in 1:n_states], title = "Generator Q diagonal", xlabel = "State", ylabel = "Q[ii]")

## Generator-based ACFs from full Q
@info "Computing generator-based ACFs (full Q)..."
acf_Q = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    g_t    = obs_dict_hr[obs]
    g_vals = cluster_means(g_t, labels, n_states)
    acf_Q[obs] = generator_acf(Q, pi_vec, g_vals; dt = DT_HR, max_lag = Q_MAX_LAG, normalize = true)
end

# Plot ACF comparison (data vs full Q)
@info "Plotting ACF comparison (data vs full Q)..."
τs_low = collect(0:Q_MAX_LAG) .* DT_HR

obs_labels = Dict(
    :u0      => "Autocorrelation of u(0)",
    :energy  => "Autocorrelation of (1/L)∫u(x)² dx",
    :fourier => "Autocorrelation of Re û(4k)"
)

fig_acf_Q = Figure(size = (1200, 900))
for (row, obs) in enumerate(obs_names)
    ax = Axis(fig_acf_Q[row, 1],
              xlabel = "Time",
              ylabel = get(obs_labels, obs, string(obs)))

    lines!(ax, τs_low, acf_data_hr[obs];
           label = "Data", color = :black, linewidth = 3.0)
    lines!(ax, τs_low, acf_Q[obs];
           label = "Generator Q", color = :red, linewidth = 2.5)
    axislegend(ax; position = :rt)
end

acf_q_path = joinpath(@__DIR__, "ks_acf_Q_vs_data_fullQ.png")
save(acf_q_path, fig_acf_Q)
@info "ACF figure saved" path = acf_q_path
display(fig_acf_Q)

#########################
# Φ, Σ from Q and Langevin integration
#########################

## Load score model
@info "Loading score model..."
model, trainer_cfg = load_score_model()

# Construct Φ and Σ from Q
@info "Constructing Φ and Σ from Q..."
Φ, Σ = construct_phi_sigma(Q, centers, pi_vec;
                           regularization = 0.0,
                           drift_scale    = 0.0)
@info "Φ and Σ constructed" size_Φ = size(Φ) size_Σ = size(Σ)

eigvals_sym = eigvals(Symmetric(Φ + Φ'))

Plots.plot(eigvals_sym)

##
min_eig_sym = minimum(real.(eigvals_sym))
@info "Eigenvalues of Φ + Φ'" min_eig = min_eig_sym
display(Plots.heatmap(Φ, title = "Drift matrix Φ", colorbar_title = "Value"))

## Prepare Langevin integration on high-resolution dataset
@info "Preparing Langevin integration on high-resolution dataset..."
L_hr = size(hr_dataset.data, 1)
C_hr = size(hr_dataset.data, 2)
D_hr = L_hr * C_hr

@assert D_hr == size(Φ, 1) "Dimension mismatch between Φ and data"

n_ens = max(LANGEVIN_ENSEMBLES, 1)
x0 = Matrix{Float32}(undef, D_hr, n_ens)
for i in 1:n_ens
    idx = rand(1:size(hr_dataset.data, 3))
    x0[:, i] = reshape(hr_dataset.data[:, :, idx], D_hr)
end

device_str = USE_GPU ? "gpu" : "cpu"

score_wrapper = ScoreWrapper(model, Float32(trainer_cfg.sigma), L_hr, C_hr, D_hr)
integrator = build_snapshot_integrator(score_wrapper; device = device_str)

# Optional boundary: clamp to the range observed in the high-resolution dataset.
data_min_hr = Float32(minimum(hr_dataset.data))
data_max_hr = Float32(maximum(hr_dataset.data))
boundary = (data_min_hr, data_max_hr)

@info "Integrating Langevin SDE with Φ, Σ" dt = LANGEVIN_DT effective_dt = LANGEVIN_DT * LANGEVIN_RESOLUTION steps = LANGEVIN_TOTAL_STEPS burn_in = LANGEVIN_BURN_IN ensembles = n_ens device = device_str

Φ_test = Φ * 1
Σ_test = Σ * √1

# Φ = 0.2 * Matrix(I, 32, 32)
# Σ = √0.2 * Matrix(I, 32, 32)


@time traj_state = integrator(x0, Φ_test, Σ_test;
                              dt        = 0.01,
                              n_steps   = 50_000,
                              burn_in   = 0,
                              resolution = 10,
                              boundary   = boundary,
                              progress   = true,
                              progress_desc = "Langevin ensemble integration") 
@info "Raw Langevin state trajectory" size = size(traj_state)
flattened = reshape(traj_state, D_hr, :)
langevin_traj = reshape(flattened, L_hr, C_hr, :)
@info "Reshaped Langevin trajectory" size = size(langevin_traj)

#########################
# Performance comparison with high-resolution data
#########################
##
@info "Generating comparison plots (Langevin vs high-resolution data)..."
fig_langevin = plot_langevin_vs_observed(langevin_traj, hr_dataset; max_lag = 500)

langevin_fig_path = joinpath(@__DIR__, "langevin_vs_hr_data_comparison.png")
save(langevin_fig_path, fig_langevin)
@info "Langevin vs high-resolution data figure saved" path = langevin_fig_path
display(fig_langevin)

@info "Computing KS observables for Langevin trajectory..."
obs_dict_langevin = ks_observables(langevin_traj; mode_index = Q_MODE_INDEX)

acf_langevin = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    acf_langevin[obs] = empirical_acf(obs_dict_langevin[obs], Q_MAX_LAG; normalize = true)
end

@info "Plotting ACF comparison (data vs Langevin)..."
# Use the same physical time axis as in fig_acf_Q (dt = DT_HR),
# so this figure is a direct Langevin-vs-data analogue of that one.
τs_langevin = collect(0:Q_MAX_LAG) .* DT_HR

fig_acf_langevin = Figure(size = (1200, 900))
for (row, obs) in enumerate(obs_names)
    ax = Axis(fig_acf_langevin[row, 1],
              xlabel = "Time",
              ylabel = get(obs_labels, obs, string(obs)))

    lines!(ax, τs_langevin, acf_data_hr[obs];
           label = "Data", color = :black, linewidth = 3.0)
    lines!(ax, τs_langevin, acf_langevin[obs];
           label = "Langevin", color = :blue, linewidth = 2.5)
    axislegend(ax; position = :rt)
end

acf_langevin_path = joinpath(@__DIR__, "langevin_vs_data_acf.png")
save(acf_langevin_path, fig_acf_langevin)
@info "Langevin ACF figure saved" path = acf_langevin_path
display(fig_acf_langevin)
