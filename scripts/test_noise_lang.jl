#########################
# Langevin model from Q matrix with V = -I (Stein's identity)
# Computes Φ and Σ matrices, plots ACF comparison and heatmaps
#########################

# nohup julia --project=. scripts/test_noise_lang.jl > test_noise_lang.log 2>&1 &

using LinearAlgebra
using Statistics
using Flux
using BSON
using HDF5
using Plots

using MarkovChainHammer
using MarkovChainHammer.TransitionMatrix: generator

using ScoreUNet1D

using StateSpacePartitions
using StateSpacePartitions.Trees

#########################
# Configuration
#########################
PROJECT_ROOT  = dirname(@__DIR__)
DATA_PATH     = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
MODEL_PATH    = joinpath(PROJECT_ROOT, "scripts", "model.bson")
DATASET_KEY   = "timeseries"
RESOLUTION    = 1        # Use every RESOLUTION-th snapshot
Q_MIN_PROB    = 1e-4     # Minimum probability for clustering
DT_ORIGINAL   = 1      # Time step in original dataset
DT            = DT_ORIGINAL * RESOLUTION  # Effective time step after subsampling
MAX_LAG       = 50  # Number of ACF evaluations
LAG_RES       = Int(1 / DT_ORIGINAL)  # Resolution (in samples) between evaluated lags
MODE_INDEX    = 4        # Fourier mode index for observable
REGULARIZATION = 5.0e-4     # Regularization for Cholesky
PHI_SIGMA_PATH = joinpath(PROJECT_ROOT, "data", "phi_sigma.hdf5")
V_DATA_RESOLUTION = 10   # Subsample stride when estimating V_data to reduce cost

@assert MAX_LAG > 0 "MAX_LAG must be positive"
@assert LAG_RES > 0 "LAG_RES must be at least 1"
LAG_POINTS      = collect(0:LAG_RES:((MAX_LAG - 1) * LAG_RES))
LAG_INDICES     = LAG_POINTS .+ 1
MAX_LAG_INDEX   = LAG_POINTS[end]
τs              = LAG_POINTS .* DT
#########################
# Load dataset
#########################

@info "Loading dataset..."
dataset = ScoreUNet1D.load_hdf5_dataset(DATA_PATH;
                                        dataset_key = DATASET_KEY,
                                        samples_orientation = :columns)
data_full = dataset.data  # (L, C, B_full)
data_clean = data_full[:, :, 1:RESOLUTION:end]  # Subsample
L, C, B = size(data_clean)
D = L * C

@info "Dataset loaded" L=L C=C B=B D=D resolution=RESOLUTION

#########################
# Load score model (for sigma only)
#########################
@info "Loading score model..."
contents = BSON.load(MODEL_PATH)
model = contents[:model]
sigma = haskey(contents, :trainer_cfg) ? contents[:trainer_cfg].sigma : 0.1f0
sigma = Float32(sigma)

Flux.testmode!(model)
@info "Model loaded" sigma=sigma

#########################
# Add noise to create perturbed data y = x + σε
#########################
@info "Creating perturbed dataset (y = x + σε)..."
noise = randn(Float32, size(data_clean)...)
data_noisy = data_clean .+ sigma .* noise

# Reshape to trajectory matrix (D × T) and 3D for observables
X_noisy = reshape(data_noisy, D, B)
u_noisy = reshape(data_noisy, L, C, B)

@info "Perturbed trajectory created" size=size(X_noisy)

#########################
# Cluster perturbed trajectory
#########################
@info "Clustering perturbed trajectory..."

tree_method = Tree(false, Q_MIN_PROB)
partition = StateSpacePartition(X_noisy; method = tree_method)

labels = partition.partitions
n_states = maximum(labels)
@info "Clustering complete" n_states=n_states

# Compute cluster centers (parallelized)
centers = zeros(Float64, D, n_states)
counts = zeros(Int, n_states)

for (i, s) in enumerate(labels)
    @inbounds centers[:, s] .+= X_noisy[:, i]
    @inbounds counts[s] += 1
end

for j in 1:n_states
    if counts[j] > 0
        centers[:, j] ./= counts[j]
    end
end

@info "Cluster centers computed" n_states=n_states

#########################
# Compute stationary distribution
#########################
pi_vec = zeros(Float64, n_states)
for s in labels
    pi_vec[s] += 1.0
end
pi_vec ./= sum(pi_vec)

@info "Stationary distribution computed" sum_pi=sum(pi_vec)

#########################
# Build generator Q from clustering
#########################
##
@info "Building generator Q..."
mc = reshape(labels, 1, :)
Q_raw = generator(mc; dt = DT)
@info "Generator Q built" size=size(Q_raw)

#########################
# KS Observables (parallelized)
#########################
@info "Computing KS observables..."

function ks_observables(u::Array{<:Real,3}; mode_index::Int = 4)
    nx, _, nt = size(u)
    
    obs_u0      = zeros(Float64, nt)
    obs_energy  = zeros(Float64, nt)
    obs_fourier = zeros(Float64, nt)
    
    # Precompute Fourier weights
    k = clamp(mode_index, 1, nx)
    ω = -2π * (k - 1) / nx
    weights = [cis(ω * (j - 1)) for j in 1:nx]
    
    Threads.@threads for t in 1:nt
        ut = @view u[:, 1, t]
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
    
    return Dict(:u0 => obs_u0, :energy => obs_energy, :fourier => obs_fourier)
end

obs_dict = ks_observables(u_noisy; mode_index = MODE_INDEX)
obs_names = [:u0, :energy, :fourier]

@info "Observables computed"

#########################
# Empirical ACF
#########################
function empirical_acf(x::AbstractVector, max_lag::Int; normalize::Bool = false)
    x̄ = mean(x)
    xc = x .- x̄
    var0 = dot(xc, xc) / max(length(xc) - 1, 1)
    
    acf = zeros(Float64, max_lag + 1)
    for lag in 0:max_lag
        n = length(xc) - lag
        n <= 1 && (acf[lag + 1] = NaN; continue)
        acf[lag + 1] = dot(@view(xc[1:n]), @view(xc[1+lag:lag+n])) / max(n - 1, 1)
    end
    return normalize && var0 > 0 ? acf ./ var0 : acf
end

function sample_acf_subset(acf_full::AbstractVector, indices::AbstractVector{<:Integer})
    @assert maximum(indices) <= length(acf_full) "requested lag exceeds computed range"
    sampled = acf_full[indices]
    @assert length(sampled) == MAX_LAG "expected $(MAX_LAG) points, got $(length(sampled))"
    return sampled
end

function lag_indices_from_time(time_points::AbstractVector{<:Real}, dt::Real)
    points = round.(Int, time_points ./ dt)
    @assert all(points .>= 0) "time points must be non-negative"
    return points .+ 1, maximum(points)
end

@info "Computing empirical ACFs..."
acf_data = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    acf_full = empirical_acf(obs_dict[obs], MAX_LAG_INDEX; normalize = true)
    acf_data[obs] = sample_acf_subset(acf_full, LAG_INDICES)
end

#########################
# Generator-based ACF
#########################
function cluster_means(g_time::AbstractVector, labels::AbstractVector{<:Integer}, n_states::Int)
    sums = zeros(Float64, n_states)
    counts = zeros(Int, n_states)
    
    @inbounds for (g, s) in zip(g_time, labels)
        sums[s] += g
        counts[s] += 1
    end
    
    @inbounds for i in 1:n_states
        counts[i] > 0 && (sums[i] /= counts[i])
    end
    return sums
end

function generator_acf(Q::AbstractMatrix, π::AbstractVector, g_vals::AbstractVector;
                       dt::Real = 1.0, max_lag::Int = 200, normalize::Bool = false)
    n = size(Q, 1)
    π_norm = π ./ sum(π)
    
    # Eigendecomposition for small Q
    if n <= 1000
        eig = eigen(Matrix(Q))
        V, Vinv, λ = eig.vectors, inv(eig.vectors), eig.values
        
        μ = dot(g_vals, π_norm)
        g̃ = g_vals .- μ
        v0 = Diagonal(π_norm) * g̃
        v0̂ = Vinv * v0
        
        acf = zeros(ComplexF64, max_lag + 1)
        for (k, lag) in enumerate(0:max_lag)
            t = lag * dt
            ŵ = exp.(λ .* t) .* v0̂
            w = V * ŵ
            acf[k] = dot(g̃, w)
        end
        acf_real = real.(acf)
    else
        # Direct method for large Q: evolve with Q (not Qᵀ)
        μ = dot(g_vals, π_norm)
        g_centered = g_vals .- μ
        v0 = Diagonal(π_norm) * g_centered
        
        Qmat = Matrix(Q)
        expQdt = exp(Qmat .* dt)
        
        acf_real = zeros(Float64, max_lag + 1)
        acf_real[1] = dot(g_centered, v0)
        
        w = Vector{Float64}(v0)
        for lag in 1:max_lag
            w = expQdt * w
            acf_real[lag + 1] = real(dot(g_centered, w))
        end
    end
    
    return normalize ? acf_real ./ acf_real[1] : acf_real
end

alpha = 1.0 # Scaling factor for generator, takes into account errors due to finite number of cluster.
Q = alpha .* Q_raw  # Scale generator if needed

@info "Computing generator-based ACFs..."
acf_Q = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    g_vals = cluster_means(obs_dict[obs], labels, n_states)
    acf_full = generator_acf(Q, pi_vec, g_vals; dt = DT, max_lag = MAX_LAG_INDEX, normalize = true)
    acf_Q[obs] = sample_acf_subset(acf_full, LAG_INDICES)
end

#########################
# ACF: Observations vs Generator Q
#########################
@info "Plotting observation vs Q ACFs..."

obs_labels = Dict(
    :u0      => "u(0,t)",
    :energy  => "(1/L)∫u²dx",
    :fourier => "Re û(4k)"
)

plt_obs_vs_Q = Plots.plot(layout = (3, 1), size = (800, 900), legend = :topright)

for (row, obs) in enumerate(obs_names)
    Plots.plot!(plt_obs_vs_Q, τs, acf_data[obs];
                subplot = row,
                label = "Observation",
                color = :black,
                linewidth = 2.5,
                xlabel = row == 3 ? "Time lag τ" : "",
                ylabel = "ACF of $(obs_labels[obs])",
                title = row == 1 ? "ACF: Observation vs Generator Q" : "")
    
    Plots.plot!(plt_obs_vs_Q, τs, acf_Q[obs];
                subplot = row,
                label = "Generator Q",
                color = :red,
                linewidth = 2.0,
                linestyle = :dash)
end

display(plt_obs_vs_Q)

acf_obs_q_path = joinpath(@__DIR__, "acf_obs_vs_Q.png")
Plots.savefig(plt_obs_vs_Q, acf_obs_q_path)
@info "Observation vs Q ACF figure saved" path=acf_obs_q_path
##

#########################
# Compute M matrix from Q
# M_{ij} = Σ_{n,m} x_j^n π_n x_i^m Q_{mn}
# M = centers * Q * Diagonal(π) * centers'
#########################
@info "Computing M matrix from Q..."

pi_norm = pi_vec ./ sum(pi_vec)
M = centers * Q * Diagonal(pi_norm) * centers'

@info "M matrix computed" size=size(M)

#########################
# Estimate V = E[s(y) yᵀ] on noisy data (Stein matrix)
# This is copied from scripts/test_noise.jl
#########################
@info "Evaluating score at cluster centers (for V_cluster)..."

score_at_centers = zeros(Float32, D, n_states)
batch_size = min(256, n_states)

for start_idx in 1:batch_size:n_states
    end_idx = min(start_idx + batch_size - 1, n_states)
    batch_centers = centers[:, start_idx:end_idx]
    batch_input = reshape(Float32.(batch_centers), L, C, :)

    # Score: s(y) = -model(y) / σ
    preds = model(batch_input)
    scores = -preds ./ sigma

    score_at_centers[:, start_idx:end_idx] = reshape(scores, D, :)
end

@info "Score at centers evaluated" size=size(score_at_centers)

#########################
# Construct V matrix from clustering
# V = E[s(y) y^T] = Σ_n π_n s(y^n) [y^n]^T
# where y^n are cluster centers on noisy data
#########################
@info "Constructing V matrix from clustering..."

V_cluster = Float64.(score_at_centers) * Diagonal(pi_vec) * centers'

@info "V matrix from clustering constructed" size=size(V_cluster)

#########################
# Also compute V_data directly (no clustering) for comparison/use
#########################
@info "Computing V_data directly (no clustering) for comparison..."

batch_size_data = 512
data_indices = collect(1:V_DATA_RESOLUTION:B)
n_batches_data = cld(length(data_indices), batch_size_data)
n_threads = Threads.nthreads()

# Thread-local accumulators
V_locals = [zeros(Float64, D, D) for _ in 1:n_threads]
counts_local = zeros(Int, n_threads)

Threads.@threads for batch_idx in 1:n_batches_data
    tid = Threads.threadid()
    start_idx = (batch_idx - 1) * batch_size_data + 1
    end_idx = min(batch_idx * batch_size_data, length(data_indices))
    batch_selection = data_indices[start_idx:end_idx]
    
    batch_noisy = data_noisy[:, :, batch_selection]

    # Evaluate score: s(y) = -model(y) / σ
    scores = ScoreUNet1D.score_from_model(model, batch_noisy, sigma)

    S_flat = Float64.(reshape(scores, D, :))
    Y_flat = Float64.(reshape(batch_noisy, D, :))

    V_locals[tid] .+= S_flat * Y_flat'
    counts_local[tid] += size(Y_flat, 2)
end

# Reduce
V_data = zeros(Float64, D, D)
total_samples = 0
for tid in 1:n_threads
    V_data .+= V_locals[tid]
    total_samples += counts_local[tid]
end
V_data ./= total_samples

plot(real.(eigvals(V_cluster)))
heatmap(V_data')

#########################
# Construct Φ using V_data (M = Φ V)
#########################
@info "Constructing Φ matrix using V_data..."

# Stabilize inversion of V_data in case it is ill-conditioned
Φ = M / V_data #+ REGULARIZATION * I # solves M = Φ * V_data

heatmap(Φ)

# eigvals(V_data)

@info "Φ matrix constructed" size=size(Φ) reg=REGULARIZATION

#########################
# Extract Σ from symmetric part of Φ
# Φ_S = (Φ + Φ')/2 = Σ Σ^T
#########################
@info "Extracting Σ from Cholesky of symmetric part..."

Φ_S = 0.5 * (Φ + Φ')
Φ_A = 0.5 * (Φ - Φ')

# eigvals(Φ_S)
@info "Φ decomposition" norm_symmetric=norm(Φ_S) norm_antisymmetric=norm(Φ_A)

# Check eigenvalues
eigvals_S = eigvals(Symmetric(Φ_S))
min_eig = minimum(eigvals_S)
@info "Eigenvalues of Φ_S" min=min_eig max=maximum(eigvals_S)

# Cholesky decomposition. Use the lower factor so that Σ Σᵀ = Φ_S.
chol_result = cholesky(Symmetric(Φ_S); check = true)
Σ = LowerTriangular(chol_result.L)

@info "Σ matrix extracted" size=size(Σ)

@info "Saving Φ and Σ to HDF5" path=PHI_SIGMA_PATH
h5open(PHI_SIGMA_PATH, "w") do h5
    write(h5, "Phi", Φ)
    write(h5, "Sigma", Matrix(Σ))
end
@info "Φ and Σ saved"

#########################
# Figure 2: Φ and Σ Heatmaps
#########################
@info "Plotting Φ and Σ heatmaps..."

plt_Phi = Plots.heatmap(Φ,
                        title = "Drift matrix Φ (from M / V)",
                        xlabel = "j", ylabel = "i",
                        color = :RdBu,
                        colorbar_title = "Value")

plt_Sigma = Plots.heatmap(Σ,
                          title = "Diffusion matrix Σ (from Cholesky of Φ_S)",
                          xlabel = "j", ylabel = "i",
                          color = :viridis,
                          colorbar_title = "Value")

plt_matrices = Plots.plot(plt_Phi, plt_Sigma, layout = (1, 2), size = (1200, 500))
display(plt_matrices)

matrices_path = joinpath(@__DIR__, "phi_sigma_matrices.png")
Plots.savefig(plt_matrices, matrices_path)
@info "Φ and Σ figure saved" path=matrices_path

#########################
# Summary statistics
#########################
@info "=== Summary ===" n_states=n_states D=D
@info "Φ statistics" norm=norm(Φ) trace=tr(Φ) min_diag=minimum(diag(Φ)) max_diag=maximum(diag(Φ))
@info "Σ statistics" norm=norm(Σ) trace=tr(Σ) min_diag=minimum(diag(Σ)) max_diag=maximum(diag(Σ))
##
#########################
# Langevin Integration Configuration
#########################
LANGEVIN_DT          = 0.005     # Integration time step
LANGEVIN_N_STEPS     = 2_000_000  # Total integration steps
LANGEVIN_BURN_IN     = 1000     # Burn-in steps to discard
LANGEVIN_RESOLUTION  = 200      # Save every N steps (effective dt = LANGEVIN_DT * LANGEVIN_RESOLUTION)
LANGEVIN_N_ENSEMBLES = 1        # Number of independent trajectories
USE_GPU              = false    # Use GPU for integration

#########################
# Prepare Langevin integration
#########################
@info "Preparing Langevin integration..."

# Create score wrapper
score_wrapper = ScoreUNet1D.ScoreWrapper(model, sigma, L, C, D)
integrator = ScoreUNet1D.build_snapshot_integrator(score_wrapper; device = USE_GPU ? "gpu" : "cpu")

# Initial condition from noisy data
x0 = Matrix{Float32}(undef, D, LANGEVIN_N_ENSEMBLES)
for i in 1:LANGEVIN_N_ENSEMBLES
    idx = rand(1:B)
    x0[:, i] = reshape(data_noisy[:, :, idx], D)
end

# Boundary for clamping (optional, from data range)
data_min = Float32(minimum(data_noisy))
data_max = Float32(maximum(data_noisy))
boundary = (-40, 40)

@info "Langevin integration setup" dt=LANGEVIN_DT n_steps=LANGEVIN_N_STEPS resolution=LANGEVIN_RESOLUTION effective_dt=LANGEVIN_DT*LANGEVIN_RESOLUTION

#########################
# Run Langevin integration
#########################
@info "Running Langevin integration with Φ and Σ..."

Identity = Matrix{Float32}(I, D, D)

@time langevin_raw = integrator(x0, Float32.(2.0*Φ), Float32.(sqrt(2.0)*Σ);
                                dt         = LANGEVIN_DT,
                                n_steps    = LANGEVIN_N_STEPS,
                                burn_in    = LANGEVIN_BURN_IN,
                                resolution = LANGEVIN_RESOLUTION,
                                boundary   = boundary,
                                progress   = true,
                                progress_desc = "Langevin integration")

@info "Langevin integration complete" raw_size=size(langevin_raw)

# Reshape to (L, C, T_langevin)
langevin_traj = reshape(langevin_raw, L, C, :)
T_langevin = size(langevin_traj, 3)
@info "Langevin trajectory reshaped" size=size(langevin_traj)

#########################
# Compute observables from Langevin trajectory
#########################
@info "Computing observables from Langevin trajectory..."

obs_langevin = ks_observables(langevin_traj; mode_index = MODE_INDEX)

@info "Langevin observables computed"

#########################
# Compute ACFs from Langevin trajectory
#########################
@info "Computing ACFs from Langevin trajectory..."

# Effective time step for Langevin trajectory
DT_LANGEVIN_EFF = LANGEVIN_DT * LANGEVIN_RESOLUTION
lag_indices_langevin, max_lag_langevin = lag_indices_from_time(τs, DT_LANGEVIN_EFF)
τs_langevin = τs

acf_langevin = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    acf_full = empirical_acf(obs_langevin[obs], max_lag_langevin; normalize = true)
    acf_langevin[obs] = sample_acf_subset(acf_full, lag_indices_langevin)
end

@info "Langevin ACFs computed"
##
#########################
# ACFs from unperturbed data (data_clean)
#########################
@info "Computing ACFs from unperturbed data (data_clean)..."

u_clean = reshape(data_clean, L, C, :)
obs_clean = ks_observables(u_clean; mode_index = MODE_INDEX)

acf_clean = Dict{Symbol, Vector{Float64}}()
for obs in obs_names
    acf_full = empirical_acf(obs_clean[obs], MAX_LAG_INDEX; normalize = true)
    acf_clean[obs] = sample_acf_subset(acf_full, LAG_INDICES)
end

@info "Unperturbed ACFs computed"

#########################
# Figure 4: All three ACFs together
#########################
@info "Plotting all ACF comparisons together..."

plt_all_acf = Plots.plot(layout = (3, 1), size = (800, 900), legend = :topright)

for (row, obs) in enumerate(obs_names)
    # Data ACF
    Plots.plot!(plt_all_acf, τs, acf_data[obs];
                subplot = row,
                label = "Perturbed Data",
                color = :black,
                linewidth = 2.5,
                xlabel = row == 3 ? "Time lag τ" : "",
                ylabel = "ACF of $(obs_labels[obs])",
                title = row == 1 ? "ACF Comparison: Data vs Q vs Langevin vs Clean" : "")
    
    # Unperturbed data ACF
    Plots.plot!(plt_all_acf, τs, acf_clean[obs];
                subplot = row,
                label = "Unperturbed Data",
                color = :green,
                linewidth = 2.0,
                linestyle = :dashdot)
    
    # Generator Q ACF
    Plots.plot!(plt_all_acf, τs, acf_Q[obs];
                subplot = row,
                label = "Generator Q",
                color = :red,
                linewidth = 2.0,
                linestyle = :dash)
    
    # Langevin ACF
    Plots.plot!(plt_all_acf, τs_langevin, acf_langevin[obs];
                subplot = row,
                label = "Langevin (Φ, Σ)",
                color = :blue,
                linewidth = 2.0,
                linestyle = :dot)
end

display(plt_all_acf)

all_acf_path = joinpath(@__DIR__, "acf_all_comparison.png")
Plots.savefig(plt_all_acf, all_acf_path)
@info "All ACF comparison figure saved" path=all_acf_path
