# ==============================================================================
# test_Q.jl: Self-contained evaluation of Q, Phi, Sigma for KS
# ==============================================================================
# This script is designed for line-by-line execution (shift+control+enter).
# It inlines all necessary logic from the ScoreUNet1D package.
# ==============================================================================

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"  # headless GR
end
using LinearAlgebra
using SparseArrays
using Statistics
using BSON
using HDF5
using TOML
using Plots
using ProgressMeter
using StateSpacePartitions
using StateSpacePartitions.Trees
using ScoreUNet1D
using Functors
using NNlib
const Threads = Base.Threads

# ------------------------------------------------------------------------------
# 1. Configuration and Paths
# ------------------------------------------------------------------------------

const CONFIG_PATH = joinpath(@__DIR__, "phi_sigma_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

config = TOML.parsefile(CONFIG_PATH)

paths_cfg = get(config, "paths", Dict{String,Any}())
est_cfg = get(config, "estimation", Dict{String,Any}())

# Resolve paths
data_path = joinpath(PROJECT_ROOT, get(paths_cfg, "data_path", "data/KS/new_ks.hdf5"))
model_path = joinpath(PROJECT_ROOT, get(paths_cfg, "model_path", "scripts/KS/trained_model.bson"))
dataset_key = get(paths_cfg, "dataset_key", "timeseries")
dataset_orientation = Symbol(get(paths_cfg, "dataset_orientation", "columns"))

# Estimation parameters
sigma_val = Float32(get(est_cfg, "sigma", 0.1))
resolution = Int(get(est_cfg, "resolution", 1))
q_min_prob = Float64(get(est_cfg, "q_min_prob", 1e-4))
dt_original = Float64(get(est_cfg, "dt_original", 1.0))
regularization = Float64(get(est_cfg, "regularization", 5e-4))
v_data_resolution = Int(get(est_cfg, "v_data_resolution", 10))

# ------------------------------------------------------------------------------
# 2. Inlined Functions (from src/evaluation/PhiSigmaEstimator.jl and src/training/Trainer.jl)
# ------------------------------------------------------------------------------

function sparse_generator(labels::AbstractVector{<:Integer}, n_states::Int; dt::Real)
    T = length(labels)
    transition_counts = Dict{Tuple{Int,Int},Int}()
    state_times = zeros(Int, n_states)

    @inbounds for t in 1:(T-1)
        i = labels[t]
        j = labels[t+1]
        state_times[i] += 1
        if i != j
            key = (i, j)
            transition_counts[key] = get(transition_counts, key, 0) + 1
        end
    end
    state_times[labels[T]] += 1

    I = Vector{Int}(undef, length(transition_counts) + n_states)
    J = Vector{Int}(undef, length(transition_counts) + n_states)
    V = Vector{Float64}(undef, length(transition_counts) + n_states)

    row_sums = zeros(Float64, n_states)
    idx = 1
    for ((i, j), count) in transition_counts
        rate = state_times[i] > 0 ? count / (state_times[i] * dt) : 0.0
        I[idx] = i
        J[idx] = j
        V[idx] = rate
        row_sums[i] += rate
        idx += 1
    end

    for i in 1:n_states
        I[idx] = i
        J[idx] = i
        V[idx] = -row_sums[i]
        idx += 1
    end

    return sparse(I, J, V, n_states, n_states)
end

function compute_centers_and_pi(X_noisy::AbstractMatrix, labels::AbstractVector{<:Integer}, n_states::Int)
    D, T = size(X_noisy)
    centers = zeros(Float64, D, n_states)
    counts = zeros(Int, n_states)
    @inbounds for idx in 1:T
        s = labels[idx]
        @views centers[:, s] .+= X_noisy[:, idx]
        counts[s] += 1
    end
    @inbounds for j in 1:n_states
        counts[j] > 0 && (@views centers[:, j] ./= counts[j])
    end
    pi_vec = counts ./ sum(counts)
    return centers, pi_vec
end

function compute_M(centers::AbstractMatrix, Q::AbstractMatrix, π::AbstractVector)
    π_norm = π ./ sum(π)
    # Matches `ScoreUNet1D.PhiSigmaEstimator.compute_M`
    CQ = centers * Q
    CQ_scaled = CQ * Diagonal(π_norm)
    return CQ_scaled * centers'
end

function score_from_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return preds
end

function compute_V_data(data_noisy::Array{Float32,3},
    model,
    sigma::Real;
    v_data_resolution::Int=10,
    batch_size::Int=512)
    # Keep this in sync with `ScoreUNet1D.PhiSigmaEstimator.compute_V_data`
    L, C, B = size(data_noisy)
    D = L * C
    data_indices = collect(1:v_data_resolution:B)
    n_batches = cld(length(data_indices), batch_size)
    nthreads = Threads.nthreads()
    V_locals = [zeros(Float64, D, D) for _ in 1:nthreads]
    counts_local = zeros(Int, nthreads)

    Threads.@threads for batch_idx in 1:n_batches
        tid = Threads.threadid()
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, length(data_indices))
        batch_selection = data_indices[start_idx:end_idx]
        batch_noisy = data_noisy[:, :, batch_selection]

        scores = score_from_model(model, batch_noisy, sigma)
        S_flat = Float64.(reshape(scores, D, :))
        Y_flat = Float64.(reshape(batch_noisy, D, :))

        V_locals[tid] .+= S_flat * Y_flat'
        counts_local[tid] += size(Y_flat, 2)
    end

    V_data = zeros(Float64, D, D)
    total = 0
    for tid in 1:nthreads
        V_data .+= V_locals[tid]
        total += counts_local[tid]
    end
    V_data ./= total
    return V_data
end

# ------------------------------------------------------------------------------
# 3. Main Logic
# ------------------------------------------------------------------------------

# 3.1 Load Model
println("Loading model from $model_path")
model_contents = BSON.load(model_path)
model = model_contents[:model]

# 3.2 Load Data
println("Loading data from $data_path")
data_clean = h5open(data_path, "r") do file
    raw = read(file, dataset_key)
    # Convert to (L, C, T) format
    if dataset_orientation == :columns
        # Data is (L, T) -> reshape to (L, 1, T)
        if ndims(raw) == 2
            reshape(Float32.(raw), size(raw, 1), 1, size(raw, 2))
        else
            Float32.(raw)
        end
    else
        # Data is (T, L) -> transpose and reshape to (L, 1, T)
        if ndims(raw) == 2
            reshape(Float32.(permutedims(raw)), size(raw, 2), 1, size(raw, 1))
        else
            Float32.(permutedims(raw, (2, 3, 1)))
        end
    end
end

data_clean = data_clean[:, :, 1:resolution:end]
L, C, B = size(data_clean)
D = L * C
dt = dt_original * resolution

# 3.3 Perturb and Cluster
println("Perturbing data and clustering...")
noise = randn(Float32, size(data_clean)...)
data_noisy = data_clean .+ sigma_val .* noise
X_noisy = reshape(data_noisy, D, B)

partition = StateSpacePartition(X_noisy; method=Tree(false, q_min_prob))
labels = partition.partitions
n_states = maximum(labels)

# 3.4 Build Q matrix
println("Building sparse Q matrix (n_states = $n_states)...")
Q = sparse_generator(labels, n_states; dt=dt)

Q_S = 0.5 * (Q + Q')
Q_A = 0.5 * (Q - Q')
norm_Q_sq = norm(Q)^2
println("Q explained by symmetric part: ", round(100 * norm(Q_S)^2 / norm_Q_sq, digits=2), "%")
println("Q explained by antisymmetric part: ", round(100 * norm(Q_A)^2 / norm_Q_sq, digits=2), "%")
##
# 3.5 Compute Stationary distribution and centers
println("Computing centers and pi...")
centers, pi_vec = compute_centers_and_pi(X_noisy, labels, n_states)

# 3.6 Compute M matrix
println("Computing M matrix...")
# `sparse_generator` builds a ROW-generator (rows sum to zero, `Q[i,j]` is the rate i→j).
# The derivation in `text.txt` uses the COLUMN-generator convention `ρ̇ = Qρ`, so we
# convert via transpose: `Q_col = Q_row'`.
Q_col = Q'
M = compute_M(centers, Q_col, pi_vec)

# 3.7 Compute Stein matrix V_data
println("Computing V_data...")
V_data = compute_V_data(data_noisy, model, sigma_val; v_data_resolution=v_data_resolution)

eigvals(M)
eigvals(V_data)


# 3.8 Solve for Phi
println("Solving for Phi (M = Phi * V_data)...")
Phi = M / V_data

# 3.9 Decompose Phi
Phi_S = 0.5 * (Phi + Phi')
Phi_A = 0.5 * (Phi - Phi')

# 3.10 Compute Sigma (from symmetric part)
println("Computing Sigma via Cholesky...")
# Regularize for Cholesky
eigvals_S = eigvals(Symmetric(Phi_S))
min_eig = minimum(eigvals_S)
Phi_S_reg = copy(Phi_S)
if min_eig <= 0
    shift = abs(min_eig) + max(regularization, 1e-8)
    Phi_S_reg += shift * I
end
chol = cholesky(Symmetric(Phi_S_reg))
Sigma = Matrix(chol.L)

# ------------------------------------------------------------------------------
# 4. Results Inspection
# ------------------------------------------------------------------------------

println("\n--- RESULTS ---")
println("Phi size: ", size(Phi))
println("Sigma size: ", size(Sigma))
println("Phi_S norm: ", norm(Phi_S))
println("Phi_A norm: ", norm(Phi_A))
norm_Phi_sq = norm(Phi)^2
println("Phi explained by symmetric part: ", round(100 * norm(Phi_S)^2 / norm_Phi_sq, digits=2), "%")
println("Phi explained by antisymmetric part: ", round(100 * norm(Phi_A)^2 / norm_Phi_sq, digits=2), "%")

# Plot heatmaps
p1 = heatmap(Phi, title="Phi", aspect_ratio=:equal, c=:RdBu)
p2 = heatmap(Phi_S, title="Phi_S", aspect_ratio=:equal, c=:RdBu)
p3 = heatmap(Phi_A, title="Phi_A", aspect_ratio=:equal, c=:RdBu)
p4 = heatmap(Sigma, title="Sigma", aspect_ratio=:equal, c=:viridis)
p5 = heatmap(V_data, title="V (Stein)", aspect_ratio=:equal, c=:RdBu)

layout = @layout [grid(2, 2) b{0.5w}]
final_plot = plot(p1, p2, p3, p4, p5, layout=(2, 3), size=(1200, 800))
display(final_plot)

println("Done.")

##

# --- Q convention sanity check (row vs column generator) ---
row_sums = vec(sum(Q, dims=2))          # should be ≈ 0 for row-generator (rates i→j stored in Q[i,j])
col_sums = vec(sum(Q, dims=1))          # should be ≈ 0 for column-generator (rates k→j stored in Q[j,k])

println("||sum(Q,dims=2)||_∞ = ", norm(row_sums, Inf))
println("||sum(Q,dims=1)||_∞ = ", norm(col_sums, Inf))

π = pi_vec ./ sum(pi_vec)
println("||Q*π||_∞  = ", norm(Q * π, Inf))    # small if column-generator
println("||Q'*π||_∞ = ", norm(Q' * π, Inf))   # small if row-generator (π'Q=0)

tol = 1e-10
if norm(row_sums, Inf) < tol && norm(col_sums, Inf) > tol
    println("Q is a ROW-generator ⇒ use M = centers * Q' * Diagonal(π) * centers'")
elseif norm(col_sums, Inf) < tol && norm(row_sums, Inf) > tol
    println("Q is a COLUMN-generator ⇒ use M = centers * Q * Diagonal(π) * centers'")
else
    println("Ambiguous (both/neither near 0); inspect row/col sums and tol.")
end
