#########################
# Refactored pipeline: Φ, Σ from data via wrapper
#########################

# nohup julia --project=. scripts/alpha_tuning.jl > alpha_tuning.log 2>&1 &

using LinearAlgebra
using Random
using Base.Threads
using Flux
using BSON
using HDF5
using Plots
using Statistics
using Optim
import ScoreUNet1D.EnsembleIntegrator: CPU_STATE_CACHE, GPU_STATE_CACHE
import ScoreUNet1D.PhiSigmaEstimator: average_component_acf_data
using ScoreUNet1D
#########################
# Configuration (mirrors test_noise_lang.jl defaults)
#########################
const PROJECT_ROOT  = dirname(@__DIR__)
const DATA_PATH     = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
const DATASET_KEY   = "timeseries"

# Name of the folder inside data/models containing model.bson and phi_sigma.hdf5
const MODEL_FOLDER  = "model2"  # change as needed
const MODEL_DIR     = joinpath(PROJECT_ROOT, "data", "models", MODEL_FOLDER)
const MODEL_PATH    = joinpath(MODEL_DIR, "model.bson")
const PHI_SIGMA_PATH = joinpath(MODEL_DIR, "phi_sigma.hdf5")

RESOLUTION    = 1
Q_MIN_PROB    = 1e-4
DT_ORIGINAL   = 1.0
MAX_LAG       = 100
LAG_RES       = Int(1 / DT_ORIGINAL)
REGULARIZATION = 5.0e-4
V_DATA_RESOLUTION = 10
ALPHA_LOWER = 0.1
ALPHA_UPPER = 5.0
MAX_ALPHA_EVALS = 12  # total objective evaluations budget

#########################
# Load dataset
#########################
@info "Loading dataset..." path=DATA_PATH
dataset = ScoreUNet1D.load_hdf5_dataset(DATA_PATH;
                                        dataset_key = DATASET_KEY,
                                        samples_orientation = :columns)
data_clean = dataset.data
@info "Dataset loaded" size=size(data_clean)

#########################
# Load score model
#########################
@info "Loading score model..." path=MODEL_PATH
contents = BSON.load(MODEL_PATH)
model = contents[:model]
sigma = haskey(contents, :trainer_cfg) ? Float32(contents[:trainer_cfg].sigma) : 0.1f0
Flux.testmode!(model)
@info "Model loaded" sigma=sigma

#########################
# Load Φ and Σ from the selected model folder
#########################
@info "Loading Φ and Σ from model folder" path=PHI_SIGMA_PATH
isfile(PHI_SIGMA_PATH) ||
    error("Phi/Sigma file not found at $(PHI_SIGMA_PATH). Run the estimation pipeline to generate it first.")

α_init = nothing
Φ = nothing
Σ = nothing

h5open(PHI_SIGMA_PATH, "r") do h5
    @assert haskey(h5, "Phi") "Dataset 'Phi' not found in $PHI_SIGMA_PATH"
    @assert haskey(h5, "Sigma") "Dataset 'Sigma' not found in $PHI_SIGMA_PATH"
    global Φ = read(h5, "Phi")
    global Σ = read(h5, "Sigma")
    global α_init = haskey(h5, "Alpha") ? read(h5, "Alpha") : nothing
end

@info "Φ and Σ loaded" size_Φ=size(Φ) size_Σ=size(Σ) alpha_initial=α_init
##
#########################
# Search over α to minimize mean-ACF RMSE
#########################
@info "Searching for optimal alpha to minimize mean ACF RMSE..."

L, C, B = size(data_clean)
D = L * C

LANGEVIN_DT          = 0.0025
LANGEVIN_N_STEPS     = 100_000
LANGEVIN_BURN_IN     = 25_000
LANGEVIN_RESOLUTION  = 400
LANGEVIN_N_ENSEMBLES = 256
USE_GPU              = true

# Fix randomness for fair comparisons across alphas
rng = MersenneTwister(1234)
data_noisy = data_clean .+ sigma .* randn!(rng, similar(data_clean, Float32))

lag_points = collect(0:LAG_RES:((MAX_LAG - 1) * LAG_RES))
lag_indices = lag_points .+ 1
max_lag_idx = lag_points[end]
τs = lag_points .* DT_ORIGINAL

mean_acf_data_full = average_component_acf_data(reshape(data_noisy, D, :), max_lag_idx)
mean_acf_data = mean_acf_data_full[lag_indices]

# Shared integrator and initial condition
score_wrapper = ScoreUNet1D.ScoreWrapper(model, sigma, L, C, D)
integrator = ScoreUNet1D.build_snapshot_integrator(score_wrapper; device = USE_GPU ? "gpu" : "cpu")
x0 = Matrix{Float32}(undef, D, LANGEVIN_N_ENSEMBLES)
for i in 1:LANGEVIN_N_ENSEMBLES
    idx = rand(rng, 1:B)
    x0[:, i] = reshape(data_noisy[:, :, idx], D)
end

function mean_acf_langevin(traj::Array{Float32,3}, L::Int, C::Int, n_ens::Int, max_lag::Int)
    # traj is (D, T_snap, n_ens)
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

function evaluate_alpha(alpha::Real,
                        Φ::AbstractMatrix,
                        Σ::AbstractMatrix,
                        integrator,
                        x0::AbstractMatrix,
                        max_lag_idx::Int,
                        lag_indices::Vector{Int})
    # Reset RNG so noise is identical across alphas
    Random.seed!(1234)
    # Ensure Phi/Sigma updates take effect by resetting cached integrator state
    empty!(CPU_STATE_CACHE)
    empty!(GPU_STATE_CACHE)
    Φ_scaled = alpha .* Φ
    Σ_scaled = sqrt(alpha) .* Σ
    traj_state = integrator(x0, Float32.(Φ_scaled), Float32.(Σ_scaled);
                            dt         = LANGEVIN_DT,
                            n_steps    = LANGEVIN_N_STEPS,
                            burn_in    = LANGEVIN_BURN_IN,
                            resolution = LANGEVIN_RESOLUTION,
                            boundary   = nothing,
                            progress   = false,
                            progress_desc = "Langevin GPU (alpha tuning)")
    traj = Array(traj_state)  # (D, snapshots, ensembles)
    mean_acf_full = mean_acf_langevin(traj, L, C, LANGEVIN_N_ENSEMBLES, max_lag_idx)
    return mean_acf_full[lag_indices]
end

function objective(alpha::Real)
    alpha <= 0 && return Inf
    acf_langevin = evaluate_alpha(alpha, Φ, Σ, integrator, x0, max_lag_idx, lag_indices)
    rmse = sqrt(mean((acf_langevin .- mean_acf_data) .^ 2))
    @info "Optim eval" alpha=alpha rmse=rmse
    return rmse
end

alpha_history = Float64[]
rmse_history = Float64[]
function logging_objective(alpha::Real)
    val = objective(alpha)
    push!(alpha_history, Float64(alpha))
    push!(rmse_history, Float64(val))
    return val
end

opt_res = optimize(logging_objective, ALPHA_LOWER, ALPHA_UPPER, Brent(); iterations = MAX_ALPHA_EVALS)
best_alpha = Optim.minimizer(opt_res)
best_rmse = Optim.minimum(opt_res)
best_acf = evaluate_alpha(best_alpha, Φ, Σ, integrator, x0, max_lag_idx, lag_indices)

@info "Best alpha search complete" best_alpha=best_alpha best_rmse=best_rmse total_evals=Optim.iterations(opt_res)

# Plot comparison (mean ACF only)
plt = Plots.plot(τs, mean_acf_data;
                 label = "Perturbed Data (mean ACF)",
                 color = :black, linewidth = 2.5,
                 xlabel = "Time lag τ", ylabel = "Average ACF",
                 title = "Average ACF: Data vs Langevin (α tuned)")
Plots.plot!(plt, τs, best_acf;
            label = "Langevin (α=$(round(best_alpha, digits=3)))",
            color = :blue, linewidth = 2.0, linestyle = :dash)

acf_plot_path = joinpath(@__DIR__, "acf_mean_data_vs_langevin_alpha.png")
Plots.savefig(plt, acf_plot_path)
@info "Mean ACF comparison saved" path=acf_plot_path best_alpha=best_alpha best_rmse=best_rmse

# Plot alpha vs RMSE history
plt_alpha_rmse = Plots.plot(alpha_history, rmse_history;
                            seriestype = :scatter,
                            markershape = :circle,
                            markercolor = :red,
                            line = (:dash, :gray),
                            xlabel = "alpha",
                            ylabel = "RMSE (mean ACF)",
                            title = "Alpha search history")
alpha_rmse_plot_path = joinpath(@__DIR__, "alpha_vs_rmse.png")
Plots.savefig(plt_alpha_rmse, alpha_rmse_plot_path)
@info "Alpha vs RMSE history saved" path=alpha_rmse_plot_path
##
#########################
# Persist results
#########################
@info "Saving tuned α (and Φ, Σ) to model folder" path=PHI_SIGMA_PATH
h5open(PHI_SIGMA_PATH, "r+") do h5
    write(h5, "Alpha", best_alpha)
    write(h5, "Phi", Φ)
    write(h5, "Sigma", Σ)
end
@info "Saved α, Φ, and Σ" path=PHI_SIGMA_PATH
