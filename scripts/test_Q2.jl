#########################
# Φ and ACF comparisons for new_ks datasets
#########################

using LinearAlgebra
using MarkovChainHammer
using MarkovChainHammer.TransitionMatrix: generator
using ScoreUNet1D
using ScoreUNet1D: NormalizedDataset, apply_stats, load_hdf5_dataset,
    ScoreWrapper, build_snapshot_integrator
using StateSpacePartitions
using StateSpacePartitions.Trees
using Plots
using BSON

#########################
# Paths and constants
#########################
const PROJECT_ROOT = dirname(@__DIR__)
const LOW_DATA_PATH = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
const HR_DATA_PATH  = joinpath(PROJECT_ROOT, "data", "new_ks_hr.hdf5")
const MODEL_PATH    = joinpath(PROJECT_ROOT, "scripts", "model.bson")
const DATASET_KEY   = "timeseries"

const DT_LOW = 1.0      # new_ks.hdf5
const DT_HR  = 0.1      # new_ks_hr.hdf5

const Q_MIN_PROB           = 1e-4
const PHI_REGULARIZATION   = 3.5e-2
const LANGEVIN_DRIFT_SCALE = 1e-2
const Q_MAX_LAG            = 50          # lag count for dt = 1.0
const ACF_MAX_TIME         = 50.0        # plot ACFs from t = 0 to 50
const LANGEVIN_DT          = DT_HR       # Langevin integrator step for snapshot spacing
const LANGEVIN_TOTAL_TIME  = 50.0        # integrate over 0–50 for comparison figure
const LANGEVIN_NSTEPS      = Int(round(LANGEVIN_TOTAL_TIME / LANGEVIN_DT))
const LANGEVIN_BURN_IN     = 0
const LANGEVIN_RESOLUTION  = 1
const LANGEVIN_ENSEMBLES   = 1
const Q_MODE_INDEX         = 4
const OBS_NAMES            = [:u0, :energy, :fourier]

#########################
# Helpers
#########################

"""
    load_low_dataset(; resolution=1)

Load the low-resolution KS dataset (from `new_ks.hdf5`) and optionally subsample in time.
This matches the behavior in `test_Q.jl` and returns a *normalized* dataset.
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

Load low-resolution data from `new_ks.hdf5` (normalized) and high-resolution data
from `new_ks_hr.hdf5`, normalized using the low dataset statistics. This mirrors
`test_Q.jl` for the low dataset, while allowing a distinct high-res dataset.
"""
function load_datasets(; resolution_low::Int = 1)
    low = load_low_dataset(; resolution = resolution_low)

    hr_raw = load_hdf5_dataset(HR_DATA_PATH;
                               dataset_key = DATASET_KEY,
                               samples_orientation = :columns,
                               normalize = false)
    hr_norm = apply_stats(hr_raw.data, low.stats)
    hr = NormalizedDataset(hr_norm, low.stats)
    return low, hr
end

extract_trajectory(ds::NormalizedDataset) = begin
    u = ds.data
    nx, _, nt = size(u)
    X = reshape(u, nx, nt)
    return X, u
end

function cluster_trajectory(X::AbstractMatrix; minimum_probability::Real)
    tree_method = Tree(false, minimum_probability)
    partition = StateSpacePartition(X; method = tree_method)
    labels = partition.partitions
    n_states = maximum(labels)

    d = size(X, 1)
    centers = zeros(Float64, d, n_states)
    counts = zeros(Int, n_states)
    for (i, s) in enumerate(labels)
        centers[:, s] .+= X[:, i]
        counts[s] += 1
    end
    for j in 1:n_states
        counts[j] > 0 && (centers[:, j] ./= counts[j])
    end
    return labels, centers, n_states
end

stationary_from_labels(labels::AbstractVector{<:Integer}, n_states::Int) = begin
    counts = zeros(Float64, n_states)
    for s in labels
        counts[s] += 1
    end
    counts ./ sum(counts)
end

function load_score_model()
    contents = BSON.load(MODEL_PATH)
    model = contents[:model]
    sigma = haskey(contents, :trainer_cfg) ? contents[:trainer_cfg].sigma : 0.05
    trainer_cfg = ScoreUNet1D.ScoreTrainerConfig(; sigma = sigma)
    return model, trainer_cfg
end

function evaluate_score_at_centers(model, trainer_cfg, centers::AbstractMatrix, dataset::NormalizedDataset)
    D, n_states = size(centers)
    L = D
    C = 1
    scores_out = zeros(Float32, D, n_states)
    batch_size = min(256, n_states)
    for start_idx in 1:batch_size:n_states
        end_idx = min(start_idx + batch_size - 1, n_states)
        batch_centers = centers[:, start_idx:end_idx]
        batch_input = reshape(Float32.(batch_centers), L, C, :)
        preds = model(batch_input)
        inv_sigma = -one(Float32) / Float32(trainer_cfg.sigma)
        scores = inv_sigma .* preds
        scores_out[:, start_idx:end_idx] = reshape(scores, D, :)
    end
    return scores_out
end

function construct_phi_sigma(Q::AbstractMatrix,
                             centers::AbstractMatrix,
                             π::AbstractVector,
                             score_at_centers::AbstractMatrix;
                             regularization::Real = 1e-6,
                             drift_scale::Real = 1.0)
    D, n_states = size(centers)
    @assert size(Q) == (n_states, n_states)
    @assert length(π) == n_states
    @assert size(score_at_centers) == (D, n_states)

    π_norm = π ./ sum(π)
    M = centers * Q * Diagonal(π_norm) * centers'
    V = centers * Diagonal(π_norm) * score_at_centers'

    @info "Constructing Φ matrix" D = D n_states = n_states rank_V = rank(V)

    Vt = Matrix(V')
    G = Vt * Vt' + regularization * I
    Φ = -(M * Vt') / G
    Φ += regularization * I

    Φ_S = 0.5 * (Φ + Φ')
    chol_L = cholesky(Symmetric(Φ_S), check = true).L
    Σ = Matrix(chol_L)

    return Φ, Σ
end

function ks_observables(u::Array{<:Real,3}; mode_index::Int = 4)
    nx, _, nt = size(u)
    obs_u0      = zeros(Float64, nt)
    obs_energy  = zeros(Float64, nt)
    obs_fourier = zeros(Float64, nt)

    k = clamp(mode_index, 1, nx)
    ω = -2π * (k - 1) / nx
    weights = [cis(ω * (j - 1)) for j in 1:nx]

    @inbounds for t in 1:nt
        ut = view(u, :, 1, t)
        obs_u0[t] = ut[1]
        s_energy = 0.0
        s_fourier = 0.0 + 0.0im
        for j in 1:nx
            val = ut[j]
            s_energy += val * val
            s_fourier += val * weights[j]
        end
        obs_energy[t] = s_energy / nx
        obs_fourier[t] = real(s_fourier)
    end
    return Dict(:u0 => obs_u0, :energy => obs_energy, :fourier => obs_fourier)
end

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
        counts[i] > 0 && (sums[i] /= counts[i])
    end
    return sums
end
#########################
# Generator-based ACF (full Q) – same method as test_Q.jl
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

    πn = π ./ sum(π)
    M = Diagonal(πn)

    μ = dot(g_vals, πn)
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
# Main flow
#########################
@info "Loading datasets..."
low_dataset, hr_dataset = load_datasets()
@info "Datasets loaded" low_size = size(low_dataset.data) hr_size = size(hr_dataset.data)

@info "Loading score model..."
model, trainer_cfg = load_score_model()

# High-resolution path (new_ks_hr.hdf5)
@info "Clustering high-resolution trajectory..."
X_hr, _ = extract_trajectory(hr_dataset)
labels_hr, centers_hr, n_states_hr = cluster_trajectory(X_hr; minimum_probability = Q_MIN_PROB)
Q_hr = generator(reshape(labels_hr, 1, :); dt = DT_HR)
pi_hr = stationary_from_labels(labels_hr, n_states_hr)
score_centers_hr = evaluate_score_at_centers(model, trainer_cfg, centers_hr, hr_dataset)
Φ_hr, Σ_hr = construct_phi_sigma(Q_hr, centers_hr, pi_hr, score_centers_hr;
                                 regularization = 1e-2,
                                 drift_scale = LANGEVIN_DRIFT_SCALE)
eigvals_sym_hr = real.(eigvals(Symmetric(0.5 * (Φ_hr + Φ_hr'))))

# Low-resolution path (new_ks.hdf5)
@info "Clustering low-resolution trajectory..."
X_low, _ = extract_trajectory(low_dataset)
labels_low, centers_low, n_states_low = cluster_trajectory(X_low; minimum_probability = Q_MIN_PROB)
Q_low = generator(reshape(labels_low, 1, :); dt = DT_LOW)
pi_low = stationary_from_labels(labels_low, n_states_low)
score_centers_low = evaluate_score_at_centers(model, trainer_cfg, centers_low, low_dataset)
Φ_low, Σ_low = construct_phi_sigma(Q_low, centers_low, pi_low, score_centers_low;
                                   regularization = 2e-2,
                                   drift_scale = LANGEVIN_DRIFT_SCALE)
eigvals_sym_low = real.(eigvals(Symmetric(0.5 * (Φ_low + Φ_low'))))

#########################
# Figure 1: heatmaps + eigenvalues
#########################
phi_layout = @layout [a b; c]
fig_phi = Plots.plot(layout = phi_layout, size = (1200, 800))

Plots.heatmap!(fig_phi[1], Φ_low;
               title = "Φ (new_ks.hdf5)",
               colormap = :viridis,
               colorbar_title = "Value",
               legend = false)

Plots.heatmap!(fig_phi[2], Φ_hr;
               title = "Φ (new_ks_hr.hdf5)",
               colormap = :viridis,
               colorbar_title = "Value",
               legend = false)

Plots.plot!(fig_phi[3], eigvals_sym_low;
            label = "Sym(Φ) eigenvalues – new_ks.hdf5",
            xlabel = "Index",
            ylabel = "Eigenvalue",
            color = :navy,
            linewidth = 2,
            markershape = :circle,
            markersize = 3,
            legend = :best)
Plots.plot!(fig_phi[3], eigvals_sym_hr;
            label = "Sym(Φ) eigenvalues – new_ks_hr.hdf5",
            color = :firebrick,
            linewidth = 2,
            markershape = :diamond,
            markersize = 3)

phi_fig_path = joinpath(@__DIR__, "phi_comparison_new_ks.png")
Plots.savefig(fig_phi, phi_fig_path)
@info "Φ comparison figure saved" path = phi_fig_path
display(fig_phi)

#########################
# Figure 2: ACFs from data vs Q_low vs Q_hr
#########################
@info "Computing observables for ACFs..."
_, u_hr = extract_trajectory(hr_dataset)
_, u_low = extract_trajectory(low_dataset)
obs_hr = ks_observables(u_hr; mode_index = Q_MODE_INDEX)
obs_low = ks_observables(u_low; mode_index = Q_MODE_INDEX)

acf_data_hr = Dict{Symbol, Vector{Float64}}()
acf_Q_low   = Dict{Symbol, Vector{Float64}}()
acf_Q_hr    = Dict{Symbol, Vector{Float64}}()

hr_max_lag = Int(round(ACF_MAX_TIME / DT_HR))
low_max_lag = Int(round(ACF_MAX_TIME / DT_LOW))

for obs in OBS_NAMES
    acf_data_hr[obs] = empirical_acf(obs_hr[obs], hr_max_lag; normalize = true)

    g_vals_low = cluster_means(obs_low[obs], labels_low, n_states_low)
    acf_Q_low[obs] = generator_acf(Q_low, pi_low, g_vals_low;
                                   dt = DT_LOW, max_lag = low_max_lag, normalize = true)

    g_vals_hr = cluster_means(obs_hr[obs], labels_hr, n_states_hr)
    acf_Q_hr[obs] = generator_acf(Q_hr, pi_hr, g_vals_hr;
                                  dt = DT_HR, max_lag = hr_max_lag, normalize = true)
end

τ_hr = collect(0:hr_max_lag) .* DT_HR
τ_low = collect(0:low_max_lag) .* DT_LOW
fig_acf = Plots.plot(layout = (1, 3), size = (1400, 400))

labels_pretty = Dict(:u0 => "u(0)", :energy => "Energy", :fourier => "Fourier mode")

for (idx, obs) in enumerate(OBS_NAMES)
    Plots.plot!(fig_acf[idx], τ_hr, acf_data_hr[obs];
                label = "Data (hr, dt=$(DT_HR))",
                color = :black,
                linewidth = 3)
    Plots.plot!(fig_acf[idx], τ_hr, acf_Q_hr[obs];
                label = "Q_hr (dt=$(DT_HR))",
                color = :blue,
                linewidth = 2,
                linestyle = :dash)
    Plots.plot!(fig_acf[idx], τ_low, acf_Q_low[obs];
                label = "Q_low (dt=$(DT_LOW))",
                color = :red,
                linewidth = 2,
                linestyle = :dot)
    Plots.xlabel!(fig_acf[idx], "Time")
    Plots.ylabel!(fig_acf[idx], "ACF")
    Plots.title!(fig_acf[idx], "ACF: $(get(labels_pretty, obs, string(obs)))")
end

acf_fig_path = joinpath(@__DIR__, "acf_comparison_new_ks.png")
Plots.savefig(fig_acf, acf_fig_path)
@info "ACF comparison figure saved" path = acf_fig_path
display(fig_acf)

#########################
# Figure 3: Langevin trajectories and ACFs for Φ_low vs Φ_hr vs data
#########################

@info "Preparing Langevin integrations for Φ_low and Φ_hr..."
L_hr = size(hr_dataset.data, 1)
C_hr = size(hr_dataset.data, 2)
D_hr = L_hr * C_hr
@assert D_hr == size(Φ_low, 1) == size(Φ_hr, 1) "Dimension mismatch between Φ and data"

n_ens = LANGEVIN_ENSEMBLES
x0 = Matrix{Float32}(undef, D_hr, n_ens)
for i in 1:n_ens
    idx = rand(1:size(hr_dataset.data, 3))
    x0[:, i] = reshape(hr_dataset.data[:, :, idx], D_hr)
end

device_str = "cpu"
score_wrapper = ScoreUNet1D.ScoreWrapper(model, Float32(trainer_cfg.sigma), L_hr, C_hr, D_hr)
integrator = build_snapshot_integrator(score_wrapper; device = device_str)

data_min_hr = Float32(minimum(hr_dataset.data))
data_max_hr = Float32(maximum(hr_dataset.data))
boundary = (data_min_hr, data_max_hr)

@info "Integrating Langevin SDE with Φ_low and Φ_hr" dt = 0.01 steps = 100_000 burn_in = LANGEVIN_BURN_IN ensembles = n_ens

traj_state_low = integrator(x0, Φ_low, Σ_low;
                            dt        = 0.01,
                            n_steps   = 250_000,
                            burn_in   = LANGEVIN_BURN_IN,
                            resolution =100,
                            boundary   = boundary,
                            progress   = true)

traj_state_hr = integrator(x0, Φ_hr, Σ_hr;
                           dt        = 0.01,
                           n_steps   = 250_000,
                           burn_in   = LANGEVIN_BURN_IN,
                           resolution =100,
                           boundary   = boundary,
                           progress   = true)

@info "Langevin integrations complete" size_low = size(traj_state_low) size_hr = size(traj_state_hr)

flattened_low = reshape(traj_state_low, D_hr, :)
flattened_hr  = reshape(traj_state_hr,  D_hr, :)
traj_low_sim  = reshape(flattened_low, L_hr, C_hr, :)
traj_hr_sim   = reshape(flattened_hr,  L_hr, C_hr, :)

# Time-series comparison for u(1,t)
T_obs = size(hr_dataset.data, 3)
T_sim_low = size(traj_low_sim, 3)
T_sim_hr  = size(traj_hr_sim, 3)
T_common = min(T_obs, T_sim_low, T_sim_hr)
T_common > 0 || error("Not enough samples for Langevin trajectory comparison")

sim_series_low = vec(traj_low_sim[1, 1, 1:T_common])
sim_series_hr  = vec(traj_hr_sim[1, 1, 1:T_common])
obs_series     = vec(hr_dataset.data[1, 1, 1:T_common])

t_axis = collect(0:T_common-1) .* LANGEVIN_DT

traj_plot = Plots.plot(t_axis, obs_series;
                       xlabel = "Time",
                       ylabel = "u(1,t)",
                       title = "Sample trajectories (data vs Langevin)",
                       label = "Data",
                       color = :black,
                       linewidth = 2)
Plots.plot!(traj_plot, t_axis, sim_series_low;
            label = "Langevin Φ_low",
            color = :red,
            linewidth = 2,
            linestyle = :dash)
Plots.plot!(traj_plot, t_axis, sim_series_hr;
            label = "Langevin Φ_hr",
            color = :blue,
            linewidth = 2,
            linestyle = :dot)

# ACF comparison using average_mode_acf
max_lag_acf = Int(round(ACF_MAX_TIME / LANGEVIN_DT))
max_lag_acf = min(max_lag_acf, T_common - 1)

obs_matrix     = reshape(hr_dataset.data[:, :, 1:T_common], L_hr * C_hr, T_common)
sim_low_matrix = reshape(traj_low_sim[:, :, 1:T_common],    L_hr * C_hr, T_common)
sim_hr_matrix  = reshape(traj_hr_sim[:, :, 1:T_common],     L_hr * C_hr, T_common)

acf_obs     = ScoreUNet1D.average_mode_acf(obs_matrix, max_lag_acf)
acf_sim_low = ScoreUNet1D.average_mode_acf(sim_low_matrix, max_lag_acf)
acf_sim_hr  = ScoreUNet1D.average_mode_acf(sim_hr_matrix, max_lag_acf)

lags = collect(0:max_lag_acf) .* LANGEVIN_DT

acf_plot = Plots.plot(lags, acf_obs;
                      xlabel = "Time",
                      ylabel = "ACF",
                      title = "ACF comparison (data vs Langevin)",
                      label = "Data",
                      color = :black,
                      linewidth = 2)
Plots.plot!(acf_plot, lags, acf_sim_low;
            label = "Langevin Φ_low",
            color = :red,
            linewidth = 2,
            linestyle = :dash)
Plots.plot!(acf_plot, lags, acf_sim_hr;
            label = "Langevin Φ_hr",
            color = :blue,
            linewidth = 2,
            linestyle = :dot)

fig_langevin = Plots.plot(traj_plot, acf_plot; layout = (2, 1), size = (900, 600))
langevin_fig_path = joinpath(@__DIR__, "langevin_comparison_new_ks.png")
Plots.savefig(fig_langevin, langevin_fig_path)
@info "Langevin comparison figure saved" path = langevin_fig_path
display(fig_langevin)
