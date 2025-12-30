# ==============================================================================
# Estimate Phi/Sigma: Drift and Diffusion Matrix Estimation for L96
# ==============================================================================
#
# Usage:
#   nohup julia --project=. scripts/L96/estimate_phi_sigma_l96.jl > scripts/L96/phi_sigma.log 2>&1 &
#
# This script estimates the drift matrix Phi and diffusion factor Sigma from the
# L96 dataset using the trained score network. The matrices are saved to HDF5.
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
data_path = joinpath(PROJECT_ROOT, get(paths_cfg, "data_path", "data/L96/new_l96.hdf5"))
model_path = joinpath(PROJECT_ROOT, get(paths_cfg, "model_path", "scripts/L96/trained_model.bson"))
output_path = joinpath(PROJECT_ROOT, get(paths_cfg, "output_path", "scripts/L96/phi_sigma.hdf5"))
dataset_key = get(paths_cfg, "dataset_key", "timeseries")
dataset_orientation = Symbol(get(paths_cfg, "dataset_orientation", "columns"))

# Estimation parameters
sigma = Float32(get(est_cfg, "sigma", 0.1))
resolution = Int(get(est_cfg, "resolution", 1))
q_min_prob = Float64(get(est_cfg, "q_min_prob", 1e-4))
dt_original = Float64(get(est_cfg, "dt_original", 0.01))
max_lag = Int(get(est_cfg, "max_lag", 50))
regularization = Float64(get(est_cfg, "regularization", 5e-4))
v_data_resolution = Int(get(est_cfg, "v_data_resolution", 10))

# Output options
save_matrices = get(output_cfg, "save_matrices", true)
plot_acf = get(output_cfg, "plot_acf", true)
acf_plot_path = joinpath(PROJECT_ROOT, get(output_cfg, "acf_plot_path", "figures/L96/phi_sigma_acf.png"))
plot_acf && ensure_dir(dirname(acf_plot_path))

verbose && @info "Loading model" model_path
model_contents = BSON.load(model_path)
model = model_contents[:model]

verbose && @info "Loading dataset" data_path dataset_key
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

verbose && @info "Dataset loaded" size = size(data_clean)

verbose && @info "Estimating Phi and Sigma matrices" sigma resolution q_min_prob

Phi, Sigma, info = estimate_phi_sigma(
    data_clean,
    model,
    sigma;
    resolution=resolution,
    q_min_prob=q_min_prob,
    dt_original=dt_original,
    max_lag=max_lag,
    regularization=regularization,
    v_data_resolution=v_data_resolution,
    plot_mean_acf=plot_acf,
    plot_path=plot_acf ? acf_plot_path : nothing
)

verbose && @info "Estimation complete" Phi_size = size(Phi) Sigma_size = size(Sigma)

if save_matrices
    verbose && @info "Saving matrices to HDF5" output_path
    h5open(output_path, "w") do file
        write(file, "Alpha", 1.0)  # Default alpha for integration
        write(file, "Phi", Phi)
        write(file, "Sigma", Sigma)
        write(file, "dt", info[:dt])
    end
    @info "Phi/Sigma matrices saved" path = output_path
end

@info "Phi/Sigma estimation complete"
plot_acf && @info "ACF comparison plot saved" path = acf_plot_path

# ==============================================================================
# Professional Heatmap Figure: Phi, Phi_S, Phi_A, Sigma, V_data
# ==============================================================================

verbose && @info "Generating matrix heatmap figure..."

# Compute symmetric and antisymmetric parts
Phi_S = 0.5 * (Phi + Phi')      # Symmetric part
Phi_A = 0.5 * (Phi - Phi')      # Antisymmetric part
V_data = info[:V_data]

# Calculate percentage of Phi explained by symmetric and antisymmetric parts
norm_Phi_sq = norm(Phi)^2
norm_Phi_S_sq = norm(Phi_S)^2
norm_Phi_A_sq = norm(Phi_A)^2
pct_symmetric = (norm_Phi_S_sq / norm_Phi_sq) * 100
pct_antisymmetric = (norm_Phi_A_sq / norm_Phi_sq) * 100

@info "Phi decomposition (squared Frobenius norms)" norm_Phi_sq = round(norm_Phi_sq, digits=4) norm_Phi_S_sq = round(norm_Phi_S_sq, digits=4) norm_Phi_A_sq = round(norm_Phi_A_sq, digits=4)
@info "Percentage of Phi variance explained" symmetric = round(pct_symmetric, digits=2) antisymmetric = round(pct_antisymmetric, digits=2)

# Determine individual color limits for each matrix
clim_Phi = (-maximum(abs.(Phi)), maximum(abs.(Phi)))
clim_Phi_S = (-maximum(abs.(Phi_S)), maximum(abs.(Phi_S)))
clim_Phi_A = (-maximum(abs.(Phi_A)), maximum(abs.(Phi_A)))
clim_V = (-maximum(abs.(V_data)), maximum(abs.(V_data)))

# Plot settings
plot_font = "Computer Modern"
title_size = 11
label_size = 9
tick_size = 7

# Create 5-panel figure (2 rows x 3 columns, last spot empty or used for colorbar)
p1 = heatmap(Phi;
    title="Phi (Drift Matrix)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_Phi,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p2 = heatmap(Phi_S;
    title="Phi_S (Symmetric Part)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_Phi_S,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p3 = heatmap(Phi_A;
    title="Phi_A (Antisymmetric Part)",
    xlabel="Column", ylabel="Row",
    c=cgrad(:RdBu, rev=true), clims=clim_Phi_A,
    aspect_ratio=:equal, framestyle=:box,
    titlefontsize=title_size, labelfontsize=label_size, tickfontsize=tick_size,
    colorbar_title="Value"
)

p4 = heatmap(Sigma;
    title="Sigma (Diffusion Factor)",
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
heatmap_path = joinpath(PROJECT_ROOT, get(output_cfg, "heatmap_plot_path", "figures/L96/phi_sigma_heatmaps.png"))
ensure_dir(dirname(heatmap_path))
savefig(fig, heatmap_path)
@info "Matrix heatmaps saved" path = heatmap_path

eigvals(Phi_S)
