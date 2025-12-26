# ==============================================================================
# Plot Publication KS: Generate Publication-Ready Figures
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/KS/plot_publication_ks.jl
#
# Requires:
#   - plot_data/KS/trajectory_identity.hdf5 (from integrate_ks.jl with mode=identity)
#   - plot_data/KS/trajectory.hdf5 (from integrate_ks.jl with mode=file)
#
# Also generates (when available):
#   - scripts/KS/phi_sigma_heatmaps.png (Φ, Φ_S, Φ_A, Σ, V_data) from the
#     Phi/Sigma estimator output specified in `scripts/KS/phi_sigma_params.toml`.
#
# ==============================================================================

using ScoreUNet1D
using TOML
using LinearAlgebra
using Random
using Flux
using CairoMakie

const CONFIG_PATH = joinpath(@__DIR__, "plot_params.toml")
const PHI_SIGMA_CONFIG_PATH = joinpath(@__DIR__, "phi_sigma_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

paths = plot_publication_figure(CONFIG_PATH; project_root=PROJECT_ROOT)

@info "Publication figure generated"
@info "  Main figure: $(paths.main_figure)"
@info "  Horizontal figure: $(paths.horizontal_figure)"
@info "  Matrices row figure: $(paths.matrices_figure)"


function overwrite_publication_matrices_row_from_estimator(plot_config_path::AbstractString,
	phi_sigma_config_path::AbstractString;
	project_root::AbstractString)

	isfile(plot_config_path) || (@warn "Plot config not found; skipping overwrite" plot_config_path; return nothing)
	isfile(phi_sigma_config_path) || (@warn "Phi/Sigma config not found; skipping overwrite" phi_sigma_config_path; return nothing)

	plot_cfg = TOML.parsefile(plot_config_path)
	paths_cfg_plot = get(plot_cfg, "paths", Dict{String,Any}())
	figure_cfg = get(plot_cfg, "figure", Dict{String,Any}())
	output_dir = joinpath(project_root, get(paths_cfg_plot, "output_dir", "plot_data/KS"))

	# Figure settings from plot_params.toml (same as PlotRunner.jl)
	font_family = get(figure_cfg, "font_family", "TeX Gyre Heros")
	title_size = Int(get(figure_cfg, "title_size", 28))
	label_size = Int(get(figure_cfg, "label_size", 20))
	tick_size = Int(get(figure_cfg, "tick_size", 16))
	colormap_heatmap = Symbol(get(figure_cfg, "colormap_heatmap", "balance"))

	cfg = TOML.parsefile(phi_sigma_config_path)
	paths_cfg = get(cfg, "paths", Dict{String,Any}())
	est_cfg = get(cfg, "estimation", Dict{String,Any}())

	data_path = joinpath(project_root, get(paths_cfg, "data_path", "data/KS/new_ks.hdf5"))
	model_path = joinpath(project_root, get(paths_cfg, "model_path", "scripts/KS/trained_model.bson"))
	phi_sigma_path = joinpath(project_root, get(paths_cfg, "output_path", "scripts/KS/phi_sigma.hdf5"))
	dataset_key = get(paths_cfg, "dataset_key", "timeseries")
	dataset_orientation = Symbol(get(paths_cfg, "dataset_orientation", "columns"))

	v_data_resolution = Int(get(est_cfg, "v_data_resolution", 10))
	sigma_cfg = Float32(get(est_cfg, "sigma", 0.1))

	if !isfile(phi_sigma_path)
		@warn "Phi/Sigma HDF5 not found; cannot overwrite matrices row" phi_sigma_path
		return nothing
	end

	ensure_dir(output_dir)
	out_path = joinpath(output_dir, "publication_matrices_row.png")

	@info "Overwriting publication matrices row from estimator Φ/Σ" out_path phi_sigma_path

	_, Φ, Σ, aux = load_phi_sigma(phi_sigma_path)
	Φ_S = 0.5 .* (Φ .+ Φ')
	Φ_A = 0.5 .* (Φ .- Φ')

	# Prefer the exact V_data saved by the estimator (if available).
	V_data = get(aux, :V_data, nothing)
	if V_data === nothing
		# Fall back to recomputing V_data in the same spirit as `estimate_phi_sigma_ks.jl`.
		if !(isfile(data_path) && isfile(model_path))
			@warn "Missing data/model needed to compute V_data; cannot overwrite matrices row" data_path model_path
			return nothing
		end

		dataset = load_hdf5_dataset(data_path; dataset_key=dataset_key, samples_orientation=dataset_orientation)
		data_clean = dataset.data

		model, _, trainer_cfg = load_model(model_path)
		model = Flux.cpu(model)
		sigma = trainer_cfg === nothing ? sigma_cfg : Float32(trainer_cfg.sigma)

		noise = randn(Float32, size(data_clean)...)
		data_noisy = data_clean .+ sigma .* noise
		V_data = ScoreUNet1D.PhiSigmaEstimator.compute_V_data(
			data_noisy,
			model,
			sigma;
			v_data_resolution=v_data_resolution,
			batch_size=512,
		)
	end

	# Shared colormap and shared (symmetric) colorrange across all panels (same as PlotRunner.jl)
	clim_all = 0.0
	for M in (Φ, Φ_S, Φ_A, Σ, V_data)
		clim_all = max(clim_all, maximum(abs, M))
	end
	clim_all = clim_all == 0 ? 1e-12 : clim_all

	fig_m = Figure(size=(3000, 520), font=font_family)

	ax1 = Axis(fig_m[1, 1]; title="Φ", xlabel="col", ylabel="row",
		titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
		xticklabelsize=tick_size, yticklabelsize=tick_size)
	heatmap!(ax1, Φ; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

	ax2 = Axis(fig_m[1, 2]; title="Φ_S", xlabel="col", ylabel="row",
		titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
		xticklabelsize=tick_size, yticklabelsize=tick_size)
	heatmap!(ax2, Φ_S; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

	ax3 = Axis(fig_m[1, 3]; title="Φ_A", xlabel="col", ylabel="row",
		titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
		xticklabelsize=tick_size, yticklabelsize=tick_size)
	heatmap!(ax3, Φ_A; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

	ax4 = Axis(fig_m[1, 4]; title="Σ", xlabel="col", ylabel="row",
		titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
		xticklabelsize=tick_size, yticklabelsize=tick_size)
	heatmap!(ax4, Σ; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

	ax5 = Axis(fig_m[1, 5]; title="<s(x)x^T>", xlabel="col", ylabel="row",
		titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
		xticklabelsize=tick_size, yticklabelsize=tick_size)
	hm_v = heatmap!(ax5, V_data; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

	Colorbar(fig_m[1, 6], hm_v; label="value", width=12, labelsize=label_size, ticklabelsize=tick_size)

	save(out_path, fig_m; px_per_unit=1)
	@info "Publication matrices row overwritten" path = out_path
	return out_path
end


row_path = overwrite_publication_matrices_row_from_estimator(CONFIG_PATH, PHI_SIGMA_CONFIG_PATH; project_root=PROJECT_ROOT)
row_path === nothing || @info "  Matrices row (estimator): $row_path"
