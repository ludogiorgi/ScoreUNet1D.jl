# ==============================================================================
# Alpha Tuning KS: Optimize α Parameter for Φ/Σ Scaling
# ==============================================================================
#
# Tunes the α parameter by minimizing RMSE between data and Langevin ACFs.
# Φ_scaled = α * Φ, Σ_scaled = √α * Σ
#
# Usage:
#   julia --project=. scripts/KS/alpha_tuning_ks.jl
#   nohup julia --project=. scripts/KS/alpha_tuning_ks.jl > alpha_tuning.log 2>&1 &
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

@info "Loading configuration" path = CONFIG_PATH
config = load_config(CONFIG_PATH)

# Extract sections
paths_cfg = get(config, "paths", Dict{String,Any}())
langevin_cfg = get(config, "langevin", Dict{String,Any}())
optim_cfg = get(config, "optimization", Dict{String,Any}())

# Paths
data_path = resolve_path(get(paths_cfg, "data_path", "data/KS/new_ks.hdf5"), PROJECT_ROOT)
model_path = resolve_path(get(paths_cfg, "model_path", "runs/KS/trained_model.bson"), PROJECT_ROOT)
phi_sigma_path = resolve_path(get(paths_cfg, "phi_sigma_path", "data/KS/models/model1/phi_sigma.hdf5"), PROJECT_ROOT)
output_dir = resolve_path(get(paths_cfg, "output_dir", joinpath(SCRIPT_DIR)), PROJECT_ROOT)
dataset_key = get(paths_cfg, "dataset_key", "timeseries")
ensure_dir(output_dir)

# Langevin parameters
dt = Float64(get(langevin_cfg, "dt", 0.0025))
n_steps = Int(get(langevin_cfg, "n_steps", 100_000))
burn_in = Int(get(langevin_cfg, "burn_in", 25_000))
resolution = Int(get(langevin_cfg, "resolution", 400))
n_ensembles = Int(get(langevin_cfg, "n_ensembles", 256))
use_gpu = get(langevin_cfg, "use_gpu", true)

b_min = get(langevin_cfg, "boundary_min", nothing)
b_max = get(langevin_cfg, "boundary_max", nothing)
boundary_cfg = (b_min === nothing || b_max === nothing) ? nothing : (Float64(b_min), Float64(b_max))


# ACF parameters
dt_original = Float64(get(optim_cfg, "dt_original", 1.0))
max_lag = Int(get(optim_cfg, "max_lag", 100))
lag_res = Int(get(optim_cfg, "lag_res", 1))

# Optimization bounds
alpha_lower = Float64(get(optim_cfg, "alpha_lower", 0.1))
alpha_upper = Float64(get(optim_cfg, "alpha_upper", 5.0))
max_evals = Int(get(optim_cfg, "max_evals", 12))

# ─────────────────────────────────────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading dataset" path = data_path
dataset = load_hdf5_dataset(data_path; dataset_key=dataset_key, samples_orientation=:columns)
data_clean = dataset.data
L, C, B = size(data_clean)
D = L * C

@info "Dataset loaded" size = size(data_clean)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading model" path = model_path
model_contents = BSON.load(model_path)
model = model_contents[:model]
sigma = haskey(model_contents, :trainer_cfg) ? Float32(model_contents[:trainer_cfg].sigma) : 0.1f0
Flux.testmode!(model)

@info "Model loaded" sigma = sigma

# ─────────────────────────────────────────────────────────────────────────────
# Load Φ and Σ
# ─────────────────────────────────────────────────────────────────────────────

@info "Loading Φ/Σ" path = phi_sigma_path
@assert isfile(phi_sigma_path) "Phi/Sigma not found: $phi_sigma_path"

_, Φ, Σ, _ = load_phi_sigma(phi_sigma_path)
@info "Φ/Σ loaded" size_Φ = size(Φ) size_Σ = size(Σ)

# ─────────────────────────────────────────────────────────────────────────────
# Prepare for Optimization
# ─────────────────────────────────────────────────────────────────────────────

# Perturb data for consistency
rng = MersenneTwister(1234)
data_noisy = data_clean .+ sigma .* randn!(rng, similar(data_clean, Float32))

# Compute target ACF from data
lag_points = collect(0:lag_res:((max_lag-1)*lag_res))
lag_indices = lag_points .+ 1
max_lag_idx = lag_points[end]
τs = lag_points .* dt_original

mean_acf_data_full = average_component_acf_data(reshape(data_noisy, D, :), max_lag_idx)
mean_acf_data = mean_acf_data_full[lag_indices]

@info "Target ACF computed" n_lags = length(mean_acf_data)

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

    Φ_scaled = Float32.(alpha .* Φ)
    Σ_scaled = Float32.(sqrt(alpha) .* Σ)

    traj_state = integrator(x0, Φ_scaled, Σ_scaled;
        dt=dt, n_steps=n_steps, burn_in=burn_in,
        resolution=resolution, boundary=boundary_cfg,
        progress=false, progress_desc="Alpha tuning")
    traj = Array(traj_state)
    mean_acf_full = mean_acf_langevin(traj, max_lag_idx)
    return mean_acf_full[lag_indices]
end

function objective(alpha::Real)
    alpha <= 0 && return Inf
    acf_langevin = evaluate_alpha(alpha)
    rmse = sqrt(mean((acf_langevin .- mean_acf_data) .^ 2))
    @info "Eval" alpha = round(alpha, digits=4) rmse = round(rmse, digits=6)
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

@info "Starting alpha optimization" bounds = (alpha_lower, alpha_upper) max_evals = max_evals

opt_res = optimize(logging_objective, alpha_lower, alpha_upper, Brent(); iterations=max_evals)
best_alpha = Optim.minimizer(opt_res)
best_rmse = Optim.minimum(opt_res)

@info "Optimization complete" best_alpha = best_alpha best_rmse = best_rmse n_evals = Optim.iterations(opt_res)

# Final evaluation for plotting
best_acf = evaluate_alpha(best_alpha)

# ─────────────────────────────────────────────────────────────────────────────
# Generate Plots
# ─────────────────────────────────────────────────────────────────────────────

@info "Generating plots"

# ACF comparison
plt_acf = Plots.plot(τs, mean_acf_data;
    label="Data (avg ACF)", color=:black, linewidth=2.5,
    xlabel="Time lag τ", ylabel="Average ACF",
    title="ACF: Data vs Langevin (α tuned)")
Plots.plot!(plt_acf, τs, best_acf;
    label="Langevin (α=$(round(best_alpha, digits=3)))",
    color=:blue, linewidth=2.0, linestyle=:dash)

acf_plot_path = joinpath(output_dir, "acf_tuned.png")
Plots.savefig(plt_acf, acf_plot_path)

# Alpha vs RMSE history
plt_history = Plots.plot(alpha_history, rmse_history;
    seriestype=:scatter, markershape=:circle, markercolor=:red,
    line=(:dash, :gray),
    xlabel="α", ylabel="RMSE (ACF)",
    title="Alpha optimization history")

history_plot_path = joinpath(output_dir, "alpha_vs_rmse.png")
Plots.savefig(plt_history, history_plot_path)

# ─────────────────────────────────────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────────────────────────────────────

@info "Saving tuned α to Φ/Σ file" path = phi_sigma_path

h5open(phi_sigma_path, "r+") do h5
    haskey(h5, "Alpha") && HDF5.delete_object(h5, "Alpha")
    write(h5, "Alpha", best_alpha)
end

@info "Alpha tuning complete" alpha = best_alpha rmse = best_rmse acf_plot = acf_plot_path history_plot = history_plot_path
