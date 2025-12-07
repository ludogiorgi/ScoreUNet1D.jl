#########################
# Estimate V = E[s(y) y^T] from perturbed data and clustering
# Should equal -I by Stein's identity when y ~ p_σ
#########################

using LinearAlgebra
using Statistics
using Flux
using BSON
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
RESOLUTION    = 10      # Use every RESOLUTION-th snapshot
Q_MIN_PROB    = 1e-4    # Minimum probability for clustering
DT            = 1.0     # Time step between snapshots (after subsampling)

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
# Load score model
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

# Reshape to trajectory matrix (D × T)
X_noisy = reshape(data_noisy, D, B)
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

# Compute cluster centers
centers = zeros(Float64, D, n_states)
counts = zeros(Int, n_states)

for (i, s) in enumerate(labels)
    centers[:, s] .+= X_noisy[:, i]
    counts[s] += 1
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
@info "Building generator Q..."
mc = reshape(labels, 1, :)
Q = generator(mc; dt = DT)
@info "Generator Q built" size=size(Q)

#########################
# Evaluate score at cluster centers (on noisy centers!)
#########################
@info "Evaluating score at cluster centers..."

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

# V_{i,k} = Σ_n π_n [s(y^n)]_i [y^n]_k
# Matrix form: V = score_at_centers * Diagonal(π) * centers'
# where score_at_centers is (D × n_states) and centers is (D × n_states)
V_cluster = Float64.(score_at_centers) * Diagonal(pi_vec) * centers'

@info "V matrix from clustering constructed" size=size(V_cluster)

#########################
# Report V_cluster results
#########################
diag_vals_cluster = diag(V_cluster)
diag_mean_cluster = mean(diag_vals_cluster)
diag_std_cluster = std(diag_vals_cluster)
off_diag_cluster = V_cluster - Diagonal(diag_vals_cluster)
off_diag_mean_cluster = mean(abs.(off_diag_cluster))

@info "V_cluster diagonal statistics (expected: -1.0)" mean=diag_mean_cluster std=diag_std_cluster min=minimum(diag_vals_cluster) max=maximum(diag_vals_cluster)
@info "V_cluster off-diagonal statistics (expected: ~0)" mean_abs=off_diag_mean_cluster

println("\n=== V_cluster diagonal (should be ≈ -1.0) ===")
println("Mean: ", round(diag_mean_cluster, digits=4))
println("Std:  ", round(diag_std_cluster, digits=4))
println("Range: [", round(minimum(diag_vals_cluster), digits=4), ", ", round(maximum(diag_vals_cluster), digits=4), "]")
println("\n=== V_cluster off-diagonal mean |V_ij| (should be ≈ 0) ===")
println("Mean |off-diag|: ", round(off_diag_mean_cluster, digits=6))

#########################
# Plot V_cluster heatmap
#########################
@info "Plotting V_cluster heatmap..."
plt_V = Plots.heatmap(V_cluster, 
                      title = "V from clustering on noisy data (should be -I)",
                      xlabel = "j", ylabel = "i",
                      colorbar_title = "Value",
                      color = :RdBu,
                      clims = (-1.5, 0.5))
display(plt_V)

# Save figure
fig_path = joinpath(@__DIR__, "V_cluster_noisy.png")
Plots.savefig(plt_V, fig_path)
@info "V_cluster heatmap saved" path=fig_path
##
#########################
# Also compute V_data directly (no clustering) for comparison
#########################
@info "Computing V_data directly (no clustering) for comparison..."

batch_size_data = 512
n_batches_data = cld(B, batch_size_data)
n_threads = Threads.nthreads()

# Thread-local accumulators
V_locals = [zeros(Float64, D, D) for _ in 1:n_threads]
counts_local = zeros(Int, n_threads)

Threads.@threads for batch_idx in 1:n_batches_data
    tid = Threads.threadid()
    start_idx = (batch_idx - 1) * batch_size_data + 1
    end_idx = min(batch_idx * batch_size_data, B)
    
    # Use the same noisy data
    batch_noisy = data_noisy[:, :, start_idx:end_idx]
    
    # Evaluate score: s(y) = -model(y) / σ
    scores = ScoreUNet1D.score_from_model(model, batch_noisy, sigma)
    
    # Outer product: s(y) y^T
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

#########################
# Report V_data results
#########################
diag_vals_data = diag(V_data)
diag_mean_data = mean(diag_vals_data)
diag_std_data = std(diag_vals_data)
off_diag_data = V_data - Diagonal(diag_vals_data)
off_diag_mean_data = mean(abs.(off_diag_data))

@info "V_data diagonal statistics (expected: -1.0)" mean=diag_mean_data std=diag_std_data min=minimum(diag_vals_data) max=maximum(diag_vals_data)
@info "V_data off-diagonal statistics (expected: ~0)" mean_abs=off_diag_mean_data

println("\n=== V_data (direct) diagonal (should be ≈ -1.0) ===")
println("Mean: ", round(diag_mean_data, digits=4))
println("Std:  ", round(diag_std_data, digits=4))
println("Range: [", round(minimum(diag_vals_data), digits=4), ", ", round(maximum(diag_vals_data), digits=4), "]")

#########################
# Plot comparison
#########################
@info "Plotting comparison..."
plt_data = Plots.heatmap(V_data, 
                         title = "V_data (direct, should be -I)",
                         xlabel = "j", ylabel = "i",
                         colorbar_title = "Value",
                         color = :RdBu,
                         clims = (-1.5, 0.5))

plt_comparison = Plots.plot(plt_V, plt_data, layout = (1, 2), size = (1200, 500))
display(plt_comparison)

comparison_path = joinpath(@__DIR__, "V_comparison_cluster_vs_direct.png")
Plots.savefig(plt_comparison, comparison_path)
@info "Comparison figure saved" path=comparison_path