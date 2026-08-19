# ==============================================================================
# test_integrate.jl: Self-contained Langevin integration for KS
# ==============================================================================
# This script is designed for line-by-line execution (shift+return).
# You can modify Phi and Sigma directly in the code.
# ==============================================================================

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"  # headless GR
end

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using LinearAlgebra
using Statistics
using Random
using BSON
using HDF5
using TOML
using Plots
using ProgressMeter
using ScoreUNet1D
using Flux

# nohup julia scripts/KS/test_integrate.jl > output.log 2>&1 &


# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------

const CONFIG_PATH = joinpath(@__DIR__, "integrate_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

config = TOML.parsefile(CONFIG_PATH)

paths_cfg = get(config, "paths", Dict{String,Any}())
langevin_cfg = get(config, "langevin", Dict{String,Any}())
phi_sigma_cfg = get(config, "phi_sigma", Dict{String,Any}())
device_cfg = get(config, "device", Dict{String,Any}())

# Device settings
device = get(device_cfg, "device", "cpu")

# Paths
data_path = joinpath(PROJECT_ROOT, get(paths_cfg, "data_path", "data/KS/new_ks.hdf5"))
model_path = joinpath(PROJECT_ROOT, get(paths_cfg, "model_path", "scripts/KS/trained_model.bson"))
phi_sigma_path = joinpath(PROJECT_ROOT, get(phi_sigma_cfg, "path", "scripts/KS/phi_sigma.hdf5"))
dataset_key = get(paths_cfg, "dataset_key", "timeseries")
dataset_orientation = Symbol(get(paths_cfg, "dataset_orientation", "columns"))

# Langevin parameters
dt = Float64(get(langevin_cfg, "dt", 0.005))
resolution = Int(get(langevin_cfg, "resolution", 200))
n_steps = Int(get(langevin_cfg, "n_steps", 100_000))
burn_in = Int(get(langevin_cfg, "burn_in", 10_000))
n_ensembles = Int(get(langevin_cfg, "n_ensembles", 1))
seed = Int(get(langevin_cfg, "seed", 2025))

# Boundary (optional)
boundary_raw = get(langevin_cfg, "boundary", nothing)
boundary = if boundary_raw !== nothing && length(boundary_raw) == 2
    (Float64(boundary_raw[1]), Float64(boundary_raw[2]))
else
    nothing
end

# ------------------------------------------------------------------------------
# 2. Load Model
# ------------------------------------------------------------------------------

println("Loading model from $model_path")
model_contents = BSON.load(model_path)
model = Flux.cpu(model_contents[:model])
sigma_model = haskey(model_contents, :trainer_cfg) ? Float32(model_contents[:trainer_cfg].sigma) : 0.1f0
Flux.testmode!(model)
println("Model loaded, sigma = $sigma_model")

# ------------------------------------------------------------------------------
# 3. Load Data
# ------------------------------------------------------------------------------

println("Loading data from $data_path")
data_clean = h5open(data_path, "r") do file
    raw = read(file, dataset_key)
    if dataset_orientation == :columns
        if ndims(raw) == 2
            reshape(Float32.(raw), size(raw, 1), 1, size(raw, 2))
        else
            Float32.(raw)
        end
    else
        if ndims(raw) == 2
            reshape(Float32.(permutedims(raw)), size(raw, 2), 1, size(raw, 1))
        else
            Float32.(permutedims(raw, (2, 3, 1)))
        end
    end
end

L, C, B = size(data_clean)
D = L * C
println("Data loaded: L=$L, C=$C, B=$B, D=$D")

# ------------------------------------------------------------------------------
# 4. Load or Define Phi and Sigma
# ------------------------------------------------------------------------------

# Option A: Load from file
println("Loading Phi/Sigma from $phi_sigma_path")
phi_sigma_data = h5open(phi_sigma_path, "r") do h5
    alpha = haskey(h5, "Alpha") ? read(h5, "Alpha") : 1.0
    Phi_raw = read(h5, "Phi")
    Sigma_raw = read(h5, "Sigma")
    (alpha=alpha, Phi=Phi_raw, Sigma=Sigma_raw)
end

alpha = phi_sigma_data.alpha
Phi = Float32.(alpha .* phi_sigma_data.Phi)
Sigma = Float32.(sqrt(alpha) .* phi_sigma_data.Sigma)

# Phi_S = 0.5 * (Phi + Phi')
# Phi_A = 0.5 * (Phi - Phi')
# eigvals(Phi_S)
# Sigma = Matrix(cholesky(Phi_S).U)

##
# Option B: Use identity matrices (uncomment to use instead)
# Phi = Matrix{Float32}(I, D, D)
# Sigma = Matrix{Float32}(I, D, D)

##
# Option C: Custom Phi and Sigma (modify as needed)
# You can modify Phi and Sigma here before integration:
# Phi = ...  # Your custom drift matrix (D x D)
# Sigma = ... # Your custom diffusion matrix (D x D)

println("Phi size: ", size(Phi))
println("Sigma size: ", size(Sigma))

# ------------------------------------------------------------------------------
# 5. Build Integrator
# ------------------------------------------------------------------------------

println("Building integrator with device=$device...")
integrator = ScoreUNet1D.build_snapshot_integrator(
    ScoreUNet1D.ScoreWrapper(model, sigma_model, L, C, D);
    device=device
)

# ------------------------------------------------------------------------------
# 6. Set Initial Conditions
# ------------------------------------------------------------------------------

rng = MersenneTwister(seed)
x0 = Matrix{Float32}(undef, D, n_ensembles)
@inbounds for i in 1:n_ensembles
    idx = rand(rng, 1:B)
    x0[:, i] = reshape(@view(data_clean[:, :, idx]), D)
end
println("Initial conditions set for $n_ensembles ensemble(s)")

# ------------------------------------------------------------------------------
# 7. Integrate Langevin Dynamics
# ------------------------------------------------------------------------------

println("Starting Langevin integration...")
println("  dt = $dt, resolution = $resolution, n_steps = $n_steps, burn_in = $burn_in")

traj_state = integrator(x0, Phi, Sigma;
    dt=dt, n_steps=n_steps, burn_in=burn_in,
    resolution=resolution, boundary=boundary,
    progress=true, progress_desc="Langevin")

traj = Array(traj_state)
T_snap = size(traj, 2)
traj_tensor = reshape(traj, L, C, T_snap, :)

println("Integration complete: $(T_snap) snapshots, $(n_ensembles) ensemble(s)")

# ------------------------------------------------------------------------------
# 8. Visualize Results
# ------------------------------------------------------------------------------

# Check for NaN/Inf in trajectory
n_nan = count(isnan, traj_tensor)
n_inf = count(isinf, traj_tensor)
println("Trajectory stats: min=$(minimum(traj_tensor)), max=$(maximum(traj_tensor)), NaN=$n_nan, Inf=$n_inf")

# Heatmap of trajectory
n_show = min(500, T_snap)
p1 = heatmap(traj_tensor[:, 1, 1:n_show, 1]',
    title="Langevin Trajectory (first $n_show steps)",
    xlabel="Space", ylabel="Time",
    c=:RdBu, aspect_ratio=:auto)

# Heatmap of reference data
p2 = heatmap(data_clean[:, 1, 1:n_show]',
    title="Reference Data (first $n_show steps)",
    xlabel="Space", ylabel="Time",
    c=:RdBu, aspect_ratio=:auto)

# PDF comparison (filter out NaN/Inf)
langevin_flat = filter(isfinite, vec(traj_tensor))
data_flat = vec(data_clean[:, :, 1:min(size(data_clean, 3), T_snap * n_ensembles)])

if length(langevin_flat) > 0
    p3 = histogram(langevin_flat, bins=100, alpha=0.5, label="Langevin", normalize=:pdf)
    histogram!(p3, data_flat, bins=100, alpha=0.5, label="Data", normalize=:pdf)
    title!(p3, "PDF Comparison")
else
    p3 = plot(title="PDF Comparison (no valid data)", legend=false)
    annotate!(p3, 0.5, 0.5, text("Trajectory has no finite values!", :red, 12))
end

# ACF comparison (averaged over dimensions and ensembles)
using ScoreUNet1D.PhiSigmaEstimator: average_acf_ensemble, average_acf_3d

acf_max_lag = min(100, T_snap - 1, size(data_clean, 3) - 1)
dt_effective = dt * resolution

acf_langevin = average_acf_ensemble(traj_tensor, acf_max_lag)
acf_data = average_acf_3d(data_clean, acf_max_lag)

lags_langevin = collect(0:acf_max_lag) .* dt_effective
lags_data = collect(0:acf_max_lag) .* dt_effective  # Assume same dt for data

p4 = plot(lags_data, acf_data;
    xlabel="Time lag", ylabel="ACF",
    title="Average ACF Comparison",
    label="Data", color=:black, linewidth=2.5)
plot!(p4, lags_langevin, acf_langevin;
    label="Langevin", color=:firebrick, linewidth=2.0, linestyle=:dash)
hline!(p4, [0.0]; color=:gray, linestyle=:dot, linewidth=1.0, label=nothing)
ylims!(p4, -0.15, 1.05)

# Combine plots
final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
display(final_plot)

println("\nDone.")

# ------------------------------------------------------------------------------
# 9. Optional: Save Trajectory
# ------------------------------------------------------------------------------

# Uncomment to save:
# output_path = joinpath(@__DIR__, "test_trajectory.hdf5")
# h5open(output_path, "w") do h5
#     write(h5, "trajectory", traj_tensor)
#     write(h5, "Phi", Phi)
#     write(h5, "Sigma", Sigma)
#     write(h5, "dt", dt)
#     write(h5, "resolution", resolution)
# end
# println("Saved to $output_path")
