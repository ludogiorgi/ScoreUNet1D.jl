# ==============================================================================
# Estimate Phi/Sigma: Drift and Diffusion Matrix Estimation for KS
# ==============================================================================
#
# Usage:
#   nohup julia --project=. scripts/KS/estimate_phi_sigma_ks.jl > scripts/KS/phi_sigma.log 2>&1 &
#
# This script estimates the drift matrix Φ and diffusion factor Σ from the
# KS dataset using the trained score network. The matrices are saved to HDF5.
#
# ==============================================================================

using ScoreUNet1D
using BSON
using HDF5
using TOML
using Plots
using LinearAlgebra

const CONFIG_PATH = joinpath(@__DIR__, "phi_sigma_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# Load configuration
config = TOML.parsefile(CONFIG_PATH)

paths_cfg = get(config, "paths", Dict{String,Any}())
est_cfg = get(config, "estimation", Dict{String,Any}())
output_cfg = get(config, "output", Dict{String,Any}())
run_cfg = get(config, "run", Dict{String,Any}())

verbose = get(run_cfg, "verbose", true)

# Resolve paths
data_path = joinpath(PROJECT_ROOT, get(paths_cfg, "data_path", "data/KS/new_ks.hdf5"))
model_path = joinpath(PROJECT_ROOT, get(paths_cfg, "model_path", "scripts/KS/trained_model.bson"))
output_path = joinpath(PROJECT_ROOT, get(paths_cfg, "output_path", "scripts/KS/phi_sigma.hdf5"))
dataset_key = get(paths_cfg, "dataset_key", "timeseries")
dataset_orientation = Symbol(get(paths_cfg, "dataset_orientation", "columns"))

# Estimation parameters
sigma = Float32(get(est_cfg, "sigma", 0.1))
resolution = Int(get(est_cfg, "resolution", 1))
q_min_prob = Float64(get(est_cfg, "q_min_prob", 1e-4))
dt_original = Float64(get(est_cfg, "dt_original", 1.0))
max_lag = Int(get(est_cfg, "max_lag", 50))
regularization = Float64(get(est_cfg, "regularization", 5e-4))
v_data_resolution = Int(get(est_cfg, "v_data_resolution", 10))
finite_dt_correction = Bool(get(est_cfg, "finite_dt_correction", false))
finite_dt_correction_mode_raw = get(est_cfg, "finite_dt_correction_mode", "per_state")
finite_dt_correction_mode = finite_dt_correction_mode_raw isa Symbol ?
                            finite_dt_correction_mode_raw :
                            Symbol(lowercase(String(finite_dt_correction_mode_raw)))

# Output options
save_matrices = get(output_cfg, "save_matrices", true)
plot_acf = get(output_cfg, "plot_acf", true)
acf_plot_path = joinpath(PROJECT_ROOT, get(output_cfg, "acf_plot_path", "figures/KS/phi_sigma_acf.png"))
plot_acf && ensure_dir(dirname(acf_plot_path))

verbose && @info "Loading model" model_path
model_contents = BSON.load(model_path)
model = model_contents[:model]

sigma_model = haskey(model_contents, :trainer_cfg) ? Float32(model_contents[:trainer_cfg].sigma) : sigma
sigma_model != sigma && @warn "Config sigma differs from model training sigma; using model sigma" sigma_config=sigma sigma_model=sigma_model
sigma = sigma_model

verbose && @info "Loading dataset (normalized, as in training)" data_path dataset_key dataset_orientation
dataset = load_hdf5_dataset(data_path; dataset_key=dataset_key, samples_orientation=dataset_orientation)
data_clean = dataset.data
verbose && @info "Dataset loaded" size = size(data_clean)

verbose && @info "Estimating Phi and Sigma matrices" sigma resolution q_min_prob dt_original finite_dt_correction finite_dt_correction_mode

Φ, Σ, info = estimate_phi_sigma(
    data_clean,
    model,
    sigma;
    resolution=resolution,
    q_min_prob=q_min_prob,
    dt_original=dt_original,
    finite_dt_correction=finite_dt_correction,
    finite_dt_correction_mode=finite_dt_correction_mode,
    max_lag=max_lag,
    regularization=regularization,
    v_data_resolution=v_data_resolution,
    plot_mean_acf=plot_acf,
    plot_path=plot_acf ? acf_plot_path : nothing
)

verbose && @info "Estimation complete" Φ_size = size(Φ) Σ_size = size(Σ)

if save_matrices
    verbose && @info "Saving matrices to HDF5" output_path
    save_phi_sigma(output_path, Φ, Σ;
        alpha=1.0,
        Q=info[:Q],
        pi_vec=info[:pi_vec],
        dt=info[:dt],
        V_data=info[:V_data])
    @info "Phi/Sigma matrices saved" path = output_path
end

@info "Phi/Sigma estimation complete"
plot_acf && @info "ACF comparison plot saved" path = acf_plot_path

# ==============================================================================
# Professional Heatmap Figure: Φ, Φ_S, Φ_A, Σ, V_data
# ==============================================================================

verbose && @info "Generating matrix heatmap figure..."

# Compute symmetric and antisymmetric parts
Φ_S = 0.5 * (Φ + Φ')      # Symmetric part
Φ_A = 0.5 * (Φ - Φ')      # Antisymmetric part
V_data = info[:V_data]

# Calculate percentage of Φ explained by symmetric and antisymmetric parts
# Using squared Frobenius norms: ||Φ||_F² = ||Φ_S||_F² + ||Φ_A||_F² (orthogonal decomposition)
norm_Φ_sq = norm(Φ)^2
norm_Φ_S_sq = norm(Φ_S)^2
norm_Φ_A_sq = norm(Φ_A)^2
pct_symmetric = (norm_Φ_S_sq / norm_Φ_sq) * 100
pct_antisymmetric = (norm_Φ_A_sq / norm_Φ_sq) * 100

@info "Φ decomposition (squared Frobenius norms)" norm_Φ² = round(norm_Φ_sq, digits=4) norm_Φ_S² = round(norm_Φ_S_sq, digits=4) norm_Φ_A² = round(norm_Φ_A_sq, digits=4)
@info "Percentage of Φ variance explained" symmetric = round(pct_symmetric, digits=2) antisymmetric = round(pct_antisymmetric, digits=2)

# Determine individual color limits for each matrix
clim_Φ = (-maximum(abs.(Φ)), maximum(abs.(Φ)))
clim_Φ_S = (-maximum(abs.(Φ_S)), maximum(abs.(Φ_S)))
clim_Φ_A = (-maximum(abs.(Φ_A)), maximum(abs.(Φ_A)))
clim_V = (-maximum(abs.(V_data)), maximum(abs.(V_data)))

# Plot settings
plot_font = "Computer Modern"
title_size = 11
label_size = 9
tick_size = 7

# Create 5-panel figure (2 rows × 3 columns, last spot empty or used for colorbar)
p1 = heatmap(Φ;
    title="Φ (Drift Matrix)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_Φ,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p2 = heatmap(Φ_S;
    title="Φₛ (Symmetric Part)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_Φ_S,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p3 = heatmap(Φ_A;
    title="Φₐ (Antisymmetric Part)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_Φ_A,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p4 = heatmap(Σ;
    title="Σ (Diffusion Factor)",
    xlabel="Column", ylabel="Row",
    c=:viridis,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p5 = heatmap(V_data;
    title="V (Stein Matrix from Data)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_V,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

# Combine into figure
fig = plot(p1, p2, p3, p4, p5;
    layout=(2, 3),
    size=(1400, 900),
    margin=5Plots.mm,
    plot_title="Estimated Matrices from Score Network",
    plot_titlefontsize=14
)

# Save figure
heatmap_path = joinpath(PROJECT_ROOT, get(output_cfg, "heatmap_plot_path", "figures/KS/phi_sigma_heatmaps.png"))
ensure_dir(dirname(heatmap_path))
savefig(fig, heatmap_path)
@info "Matrix heatmaps saved" path = heatmap_path


eigvals(V_data)
