# ==============================================================================
# Plot Data L96: Visualize Raw L96 HDF5 Data
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/L96/plot_data_l96.jl
#
# This script reads new_l96.hdf5 and generates:
#   - Heatmap of the first 1000 snapshots
#   - Univariate PDF averaged over all 36 modes
#   - Autocorrelation function (ACF)
#   - Bivariate PDFs for delays 1, 2, and 3
#
# ==============================================================================

using CairoMakie
using HDF5
using KernelDensity
using Statistics

# Import functions from src
using ScoreUNet1D: load_hdf5_dataset
using ScoreUNet1D.PhiSigmaEstimator: empirical_acf, average_acf_3d
using ScoreUNet1D.RunnerUtils: ensure_dir
using ScoreUNet1D.PlotRunner: averaged_univariate_pdf, pair_kde, quantile_bounds

CairoMakie.activate!()

# ----------------------------
# Configuration
# ----------------------------
const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "L96")
const DATA_PATH = joinpath(DATA_DIR, "new_l96.hdf5")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "figures", "L96")
const DATASET_KEY = "timeseries"

const HEATMAP_SNAPSHOTS = 1000
const PDF_NBINS = 256
const PDF_STRIDE = 1
const ACF_MAX_LAG = 500
const BIV_OFFSETS = [1, 2, 3]
const BIV_NPOINTS = 160
const BIV_STRIDE = 10

# Figure settings
const FONT_FAMILY = "TeX Gyre Heros"
const TITLE_SIZE = 24
const LABEL_SIZE = 18
const TICK_SIZE = 14
const LEGEND_SIZE = 14
const COLORMAP_HEATMAP = :balance
const COLORMAP_BIVARIATE = :vik

# ----------------------------
# Main Script
# ----------------------------
function main()
    @info "Loading L96 data" path = DATA_PATH
    @assert isfile(DATA_PATH) "Data file not found: $DATA_PATH"

    # Load data - shape is (K, T) where K=36, T=n_snapshots
    data_raw = h5open(DATA_PATH, "r") do h5
        read(h5, DATASET_KEY)
    end

    K, T = size(data_raw)
    @info "Data loaded" K T

    # Reshape to (L, C, T) format expected by plotting functions
    # L=K=36, C=1 (single channel)
    data = reshape(Float64.(data_raw), K, 1, T)
    L, C, _ = size(data)

    # Compute statistics
    @info "Computing statistics"

    # Heatmap data
    n_hm = min(HEATMAP_SNAPSHOTS, T)
    heatmap_data = data_raw[:, 1:n_hm]  # (K, n_hm)
    heat_range = maximum(abs, heatmap_data)
    heat_range = heat_range == 0 ? 1.0 : heat_range

    # Univariate PDF bounds
    value_bounds = quantile_bounds(data; stride=20, probs=(0.001, 0.999))

    # PDF
    pdf_x, pdf_density = averaged_univariate_pdf(data;
        nbins=PDF_NBINS, stride=PDF_STRIDE, bounds=value_bounds)

    # ACF
    acf_data = average_acf_3d(data, ACF_MAX_LAG)
    lags = collect(0:ACF_MAX_LAG)

    # Bivariate PDFs
    bivariate_specs = Dict{Int,NamedTuple}()
    for offset in BIV_OFFSETS
        xg, yg, dens = pair_kde(data, offset;
            bounds=value_bounds, npoints=BIV_NPOINTS, stride=BIV_STRIDE)
        bivariate_specs[offset] = (; x=xg, y=yg, density=dens)
    end

    # Generate figure
    @info "Generating figure"

    # Layout: 6 rows (1 heatmap, 1 PDF+ACF, 3 bivariate = 5 rows)
    n_biv_rows = length(BIV_OFFSETS)
    fig = Figure(size=(1200, 300 + 250 + 250 * n_biv_rows), font=FONT_FAMILY)

    # Row 1: Heatmap
    ax_hm = Axis(fig[1, 1:2];
        title="L96 Slow Variables (first $n_hm snapshots)",
        xlabel="Time (snapshot index)",
        ylabel="Mode k",
        titlesize=TITLE_SIZE, xlabelsize=LABEL_SIZE, ylabelsize=LABEL_SIZE,
        xticklabelsize=TICK_SIZE, yticklabelsize=TICK_SIZE)
    hm = heatmap!(ax_hm, 1:n_hm, 1:K, heatmap_data';
        colormap=COLORMAP_HEATMAP, colorrange=(-heat_range, heat_range))
    Colorbar(fig[1, 3], hm; label="X[k]", width=12, labelsize=LABEL_SIZE, ticklabelsize=TICK_SIZE)

    # Row 2: PDF and ACF
    ax_pdf = Axis(fig[2, 1];
        title="Averaged Univariate PDF",
        xlabel="Value",
        ylabel="Density",
        titlesize=TITLE_SIZE, xlabelsize=LABEL_SIZE, ylabelsize=LABEL_SIZE,
        xticklabelsize=TICK_SIZE, yticklabelsize=TICK_SIZE)
    lines!(ax_pdf, pdf_x, pdf_density; color=:navy, linewidth=2.5)
    ylims!(ax_pdf, 0, maximum(pdf_density) * 1.05)

    ax_acf = Axis(fig[2, 2];
        title="Averaged ACF",
        xlabel="Lag (snapshots)",
        ylabel="ACF",
        titlesize=TITLE_SIZE, xlabelsize=LABEL_SIZE, ylabelsize=LABEL_SIZE,
        xticklabelsize=TICK_SIZE, yticklabelsize=TICK_SIZE)
    lines!(ax_acf, lags, acf_data; color=:navy, linewidth=2.5)
    hlines!(ax_acf, [0.0]; color=:gray, linestyle=:dot, linewidth=1.5)
    ylims!(ax_acf, -0.2, 1.05)

    # Rows 3+: Bivariate PDFs
    biv_offsets_sorted = sort(collect(keys(bivariate_specs)))
    for (row_idx, offset) in enumerate(biv_offsets_sorted)
        spec = bivariate_specs[offset]
        density_max = maximum(spec.density)
        density_max = density_max == 0 ? 1e-9 : density_max

        ax_biv = Axis(fig[row_idx+2, 1:2];
            title="Bivariate PDF: x[i] vs x[i+$offset]",
            xlabel="x[i]",
            ylabel="x[i+$offset]",
            titlesize=TITLE_SIZE, xlabelsize=LABEL_SIZE, ylabelsize=LABEL_SIZE,
            xticklabelsize=TICK_SIZE, yticklabelsize=TICK_SIZE)
        hm_biv = heatmap!(ax_biv, spec.x, spec.y, spec.density;
            colormap=COLORMAP_BIVARIATE, colorrange=(0, density_max))
        Colorbar(fig[row_idx+2, 3], hm_biv;
            label="Density", width=12, labelsize=LABEL_SIZE, ticklabelsize=TICK_SIZE)
    end

    # Save figure
    ensure_dir(OUTPUT_DIR)
    output_path = joinpath(OUTPUT_DIR, "l96_data_overview.png")
    save(output_path, fig; px_per_unit=2)

    @info "Figure saved" path = output_path
    return output_path
end

main()
