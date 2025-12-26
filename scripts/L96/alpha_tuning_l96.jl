# ==============================================================================
# Alpha Tuning L96: Optimize alpha Parameter for Phi/Sigma Scaling
# ==============================================================================
#
# Tunes the alpha parameter by minimizing RMSE between data and Langevin ACFs.
# Phi_scaled = alpha * Phi, Sigma_scaled = sqrt(alpha) * Sigma
#
# Usage:
#   julia --project=. scripts/L96/alpha_tuning_l96.jl
#   nohup julia --project=. scripts/L96/alpha_tuning_l96.jl > alpha_tuning.log 2>&1 &
#
# ==============================================================================

using LinearAlgebra
using Random
using Statistics
using Flux
using BSON
using HDF5
using Plots
using Optim

using ScoreUNet1D
using ScoreUNet1D: ScoreWrapper, build_snapshot_integrator
import ScoreUNet1D.EnsembleIntegrator: CPU_STATE_CACHE, GPU_STATE_CACHE
import ScoreUNet1D.PhiSigmaEstimator: average_component_acf_data

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const CONFIG_PATH = joinpath(SCRIPT_DIR, "alpha_params.toml")

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

function clear_state_cache!()
    empty!(CPU_STATE_CACHE)
    empty!(GPU_STATE_CACHE)
    return nothing
end

function mean_acf_langevin(traj::Array{Float32,3}, max_lag::Int)
    D, T_snap, n_ens = size(traj)
    acf_accum = zeros(Float64, max_lag + 1)
    count = 0
    for e in 1:n_ens
        slice = traj[:, :, e]
        acf_e = average_component_acf_data(slice, max_lag)
        acf_accum .+= acf_e
        count += 1
    end
    return acf_accum ./ max(count, 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading configuration" path=CONFIG_PATH
config = load_config(CONFIG_PATH)

# Extract sections
paths_cfg = get(config, "paths", Dict{String,Any}())
langevin_cfg = get(config, "langevin", Dict{String,Any}())
optim_cfg = get(config, "optimization", Dict{String,Any}())

# Paths
data_path = resolve_path(get(paths_cfg, "data_path", "data/L96/new_l96.hdf5"), PROJECT_ROOT)
model_path = resolve_path(get(paths_cfg, "model_path", "scripts/L96/trained_model.bson"), PROJECT_ROOT)
phi_sigma_path = resolve_path(get(paths_cfg, "phi_sigma_path", "scripts/L96/phi_sigma.hdf5"), PROJECT_ROOT)
output_dir = resolve_path(get(paths_cfg, "output_dir", joinpath(SCRIPT_DIR)), PROJECT_ROOT)
dataset_key = get(paths_cfg, "dataset_key", "timeseries")

# Langevin parameters
dt = Float64(get(langevin_cfg, "dt", 0.0025))
n_steps = Int(get(langevin_cfg, "n_steps", 100_000))
burn_in = Int(get(langevin_cfg, "burn_in", 25_000))
resolution = Int(get(langevin_cfg, "resolution", 400))
n_ensembles = Int(get(langevin_cfg, "n_ensembles", 256))
use_gpu = get(langevin_cfg, "use_gpu", true)

# ACF parameters
dt_original = Float64(get(optim_cfg, "dt_original", 0.01))
max_lag = Int(get(optim_cfg, "max_lag", 100))
lag_res = Int(get(optim_cfg, "lag_res", 1))

# Optimization bounds
alpha_lower = Float64(get(optim_cfg, "alpha_lower", 0.1))
alpha_upper = Float64(get(optim_cfg, "alpha_upper", 5.0))
max_evals = Int(get(optim_cfg, "max_evals", 12))

# ─────────────────────────────────────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading dataset" path=data_path
dataset = load_hdf5_dataset(data_path; dataset_key=dataset_key, samples_orientation=:columns)
data_clean = dataset.data
L, C, B = size(data_clean)
D = L * C

@info "Dataset loaded" size=size(data_clean)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading model" path=model_path
model_contents = BSON.load(model_path)
model = model_contents[:model]
sigma = haskey(model_contents, :trainer_cfg) ? Float32(model_contents[:trainer_cfg].sigma) : 0.1f0
Flux.testmode!(model)

@info "Model loaded" sigma=sigma

# ─────────────────────────────────────────────────────────────────────────────
# Load Phi and Sigma
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading Phi/Sigma" path=phi_sigma_path
@assert isfile(phi_sigma_path) "Phi/Sigma not found: $phi_sigma_path"

_, Phi, Sigma, _ = load_phi_sigma(phi_sigma_path)
@info "Phi/Sigma loaded" size_Phi=size(Phi) size_Sigma=size(Sigma)

# ─────────────────────────────────────────────────────────────────────────────
# Prepare for Optimization
# ─────────────────────────────────────────────────────────────────────────────

# Perturb data for consistency
rng = MersenneTwister(1234)
data_noisy = data_clean .+ sigma .* randn!(rng, similar(data_clean, Float32))

# Compute target ACF from data
lag_points = collect(0:lag_res:((max_lag - 1) * lag_res))
lag_indices = lag_points .+ 1
max_lag_idx = lag_points[end]
taus = lag_points .* dt_original

mean_acf_data_full = average_component_acf_data(reshape(data_noisy, D, :), max_lag_idx)
mean_acf_data = mean_acf_data_full[lag_indices]

@info "Target ACF computed" n_lags=length(mean_acf_data)

# Build integrator and initial conditions
device_str = use_gpu ? "gpu" : "cpu"
integrator = build_snapshot_integrator(ScoreWrapper(model, sigma, L, C, D); device=device_str)

x0 = Matrix{Float32}(undef, D, n_ensembles)
for i in 1:n_ensembles
    idx = rand(rng, 1:B)
    x0[:, i] = reshape(data_noisy[:, :, idx], D)
end

# ─────────────────────────────────────────────────────────────────────────────
# Objective Function
# ─────────────────────────────────────────────────────────────────────────────

function evaluate_alpha(alpha::Real)
    Random.seed!(1234)
    clear_state_cache!()

    Phi_scaled = Float32.(alpha .* Phi)
    Sigma_scaled = Float32.(sqrt(alpha) .* Sigma)

    traj_state = integrator(x0, Phi_scaled, Sigma_scaled;
                            dt=dt, n_steps=n_steps, burn_in=burn_in,
                            resolution=resolution, boundary=nothing,
                            progress=false, progress_desc="Alpha tuning")
    traj = Array(traj_state)
    mean_acf_full = mean_acf_langevin(traj, max_lag_idx)
    return mean_acf_full[lag_indices]
end

function objective(alpha::Real)
    alpha <= 0 && return Inf
    acf_langevin = evaluate_alpha(alpha)
    rmse = sqrt(mean((acf_langevin .- mean_acf_data) .^ 2))
    @info "Eval" alpha=round(alpha, digits=4) rmse=round(rmse, digits=6)
    return rmse
end

# Track history
alpha_history = Float64[]
rmse_history = Float64[]

function logging_objective(alpha::Real)
    val = objective(alpha)
    push!(alpha_history, Float64(alpha))
    push!(rmse_history, Float64(val))
    return val
end

# ─────────────────────────────────────────────────────────────────────────────
# Run Optimization
# ─────────────────────────────────────────────────────────────────────────────

@info "Starting optimization" alpha_lower alpha_upper max_evals

result = optimize(logging_objective, alpha_lower, alpha_upper, Optim.Brent();
    maxevals=max_evals)

best_alpha = Optim.minimizer(result)
best_rmse = Optim.minimum(result)

@info "Optimization complete" best_alpha=best_alpha best_rmse=best_rmse

# ─────────────────────────────────────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────────────────────────────────────

mkpath(output_dir)

output_path = joinpath(output_dir, "alpha_tuning_results.toml")
open(output_path, "w") do io
    println(io, "best_alpha = $best_alpha")
    println(io, "best_rmse = $best_rmse")
    println(io, "alpha_history = $(alpha_history)")
    println(io, "rmse_history = $(rmse_history)")
end

@info "Results saved" path=output_path

# Plot history
plot_path = joinpath(output_dir, "alpha_tuning_history.png")
p = plot(alpha_history, rmse_history;
    xlabel="alpha", ylabel="RMSE",
    title="Alpha Tuning History", marker=:circle, legend=false)
savefig(p, plot_path)
@info "History plot saved" path=plot_path
