#!/usr/bin/env julia

# Build Figure 1 by mixing two stored Langevin runs:
# - Identity run supplies PDF and bivariate statistics.
# - Phi/Sigma run supplies snapshots and ACFs.
# Both runs are read from HDF5 files in plot_data.

using CairoMakie
using LinearAlgebra
using Random
using Statistics
using KernelDensity
using HDF5
using ScoreUNet1D

CairoMakie.activate!()

const PROJECT_ROOT = dirname(@__DIR__)
const IDENTITY_H5 = joinpath(PROJECT_ROOT, "plot_data", "ks_lang_identity.hdf5")
const PHI_SIGMA_H5 = joinpath(PROJECT_ROOT, "plot_data", "ks_lang.hdf5")
const FIG_PATH = joinpath(PROJECT_ROOT, "plot_data", "publication_langevin_vs_data_mixed.png")

# Analysis defaults (aligned with run_lungevin.jl)
const PDF_QUANTILE_LOW = 0.001
const PDF_QUANTILE_HIGH = 0.999
const PDF_QUANTILE_STRIDE = 20
const VALUE_SEED = 2032
const SMOOTH_W = 2

############################
# Helper utilities
############################
function quantile_bounds(tensors::AbstractArray...;
                         stride::Int = 20,
                         probs::Tuple{Float64,Float64} = (0.001, 0.999),
                         seed::Int = 0)
    rng = MersenneTwister(seed)
    buffer = Float64[]
    for tensor in tensors
        flat = vec(tensor)
        step = max(1, stride)
        if length(flat) <= 2_000_000
            append!(buffer, flat[1:step:end])
        else
            ns = min(div(length(flat), step), 2_000_000)
            idxs = rand(rng, 1:length(flat), ns)
            append!(buffer, flat[idxs])
        end
    end
    qvals = quantile(buffer, [probs[1], probs[2]])
    low, high = Float64(qvals[1]), Float64(qvals[2])
    low == high && ((low -= 1e-3); (high += 1e-3))
    return (low, high)
end

function averaged_univariate_pdf(tensor::AbstractArray{<:Real,3};
                                 nbins::Int,
                                 stride::Int,
                                 bounds::Tuple{Float64,Float64})
    slice = @view tensor[:, :, 1:stride:end]
    grid = range(bounds[1], bounds[2]; length = nbins)
    kd = kde(vec(slice), grid)
    return kd.x, kd.density
end

function pair_kde(tensor::AbstractArray{<:Real,3},
                  offset::Int;
                  bounds::Tuple{Float64,Float64},
                  npoints::Int,
                  stride::Int)
    L, C, B = size(tensor)
    @assert offset < L "Offset $offset exceeds length $L"
    n = (L - offset) * C * cld(B, stride)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    idx = 1
    @inbounds for b in 1:stride:B, c in 1:C, i in 1:(L - offset)
        xs[idx] = tensor[i, c, b]
        ys[idx] = tensor[i + offset, c, b]
        idx += 1
    end
    xs = xs[1:idx-1]
    ys = ys[1:idx-1]
    grid = range(bounds[1], bounds[2]; length = npoints)
    kd = kde((xs, ys), (grid, grid))
    return kd.x, kd.y, kd.density
end

smooth_vec(v; w::Int = 2) = begin
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w)
        hi = min(n, i + w)
        out[i] = mean(@view v[lo:hi])
    end
    return out
end

function load_dataset(path::AbstractString, dataset_key::String, orientation::Symbol)
    ScoreUNet1D.load_hdf5_dataset(path;
                                  dataset_key = dataset_key,
                                  samples_orientation = orientation)
end

############################
# Load HDF5 payloads
############################
h5_id = h5open(IDENTITY_H5, "r")
h5_phi = h5open(PHI_SIGMA_H5, "r")

traj_id_raw = read(h5_id, "trajectory")
traj_phi_raw = read(h5_phi, "trajectory")

params_id = h5_id["langevin_params"]
params_phi = h5_phi["langevin_params"]

pdf_stride = Int(read(params_id, "pdf_stride"))
pdf_nbins = Int(read(params_id, "pdf_nbins"))
acf_max_lag = Int(read(params_phi, "acf_max_lag"))
biv_offsets = collect(Int.(read(params_id, "biv_offsets")))
biv_npoints = Int(read(params_id, "biv_npoints"))
biv_stride = Int(read(params_id, "biv_stride"))
heatmap_snapshots = Int(read(params_phi, "heatmap_snapshots"))

dt_main = read(params_phi, "dt")
res_main = read(params_phi, "resolution")
dt_langevin = dt_main * res_main

data_path = String(read(h5_phi, "data_path"))
data_path_hr = String(read(h5_phi, "data_path_hr"))
dataset_key = String(read(h5_phi, "dataset_key"))
dataset_orientation = Symbol(read(h5_phi, "dataset_orientation"))

close(h5_id)
close(h5_phi)

############################
# Prepare tensors and data
############################
L, C, T_id, E_id = size(traj_id_raw)
L_phi, C_phi, T_phi, E_phi = size(traj_phi_raw)
@assert L == L_phi "Length mismatch between runs"
@assert C == C_phi "Channel mismatch between runs"

traj_id = Array(traj_id_raw)
traj_phi = Array(traj_phi_raw)

langevin_tensor_id = reshape(traj_id, L, C, T_id, E_id)
langevin_tensor_phi = reshape(traj_phi, L, C, T_phi, E_phi)

dataset = load_dataset(data_path, dataset_key, dataset_orientation)
data_clean = dataset.data
dt_data = h5open(data_path_hr, "r") do h5
    haskey(h5, "dt") ? read(h5, "dt") : 1.0
end

############################
# Data products
############################
n_hm = min(heatmap_snapshots, size(langevin_tensor_phi, 3))
heatmap_langevin = Array(langevin_tensor_phi[:, 1, 1:n_hm, 1])
heatmap_data = Array(@view data_clean[:, 1, 1:n_hm])

value_bounds = quantile_bounds(data_clean, reshape(langevin_tensor_id, L, C, :);
                               stride = PDF_QUANTILE_STRIDE,
                               probs = (PDF_QUANTILE_LOW, PDF_QUANTILE_HIGH),
                               seed = VALUE_SEED)
pdf_x, pdf_langevin = averaged_univariate_pdf(reshape(langevin_tensor_id, L, C, :);
                                              nbins = pdf_nbins,
                                              stride = pdf_stride,
                                              bounds = value_bounds)
_, pdf_data = averaged_univariate_pdf(data_clean;
                                      nbins = pdf_nbins,
                                      stride = pdf_stride,
                                      bounds = value_bounds)

acf_langevin = ScoreUNet1D.PhiSigmaEstimator.average_component_acf_data(
    reshape(langevin_tensor_phi, L * C, :), acf_max_lag)
acf_data = ScoreUNet1D.PhiSigmaEstimator.average_component_acf_data(
    reshape(data_clean, L * C, :), acf_max_lag)

lags_data = collect(0:acf_max_lag) .* dt_data
lags_langevin = collect(0:acf_max_lag) .* dt_langevin

bivariate_specs = Dict{Int,NamedTuple}()
for offset in biv_offsets
    xg, yg, dens_lang = pair_kde(reshape(langevin_tensor_id, L, C, :), offset;
                                 bounds = value_bounds,
                                 npoints = biv_npoints,
                                 stride = biv_stride)
    _, _, dens_data = pair_kde(data_clean, offset;
                               bounds = value_bounds,
                               npoints = biv_npoints,
                               stride = biv_stride)
    bivariate_specs[offset] = (; x = xg, y = yg,
                                langevin = dens_lang,
                                data = dens_data)
end

############################
# Derived quantities
############################
heat_range = maximum(abs, vcat(abs.(heatmap_langevin[:]), abs.(heatmap_data[:])))
heat_range = heat_range == 0 ? 1.0 : heat_range
value_bounds_plot = (-heat_range, heat_range)

pdf_bounds = (minimum(pdf_x), maximum(pdf_x))

acf_langevin_sm = smooth_vec(acf_langevin; w = SMOOTH_W)
acf_data_sm = smooth_vec(acf_data; w = SMOOTH_W)
!isempty(acf_langevin_sm) && (acf_langevin_sm[1] = acf_langevin[1])
!isempty(acf_data_sm) && (acf_data_sm[1] = acf_data[1])

acf_tmax = 100.0
mask_data = lags_data .<= acf_tmax
mask_lang = lags_langevin .<= acf_tmax
lags_data_plot = lags_data[mask_data]
lags_lang_plot = lags_langevin[mask_lang]
acf_data_plot = acf_data_sm[mask_data]
acf_langevin_plot = acf_langevin_sm[mask_lang]

biv_offsets_sorted = sort(collect(keys(bivariate_specs)))

############################
# Figure 1
############################
font_family = "TeX Gyre Heros"
title_size = 30
label_size = 22
tick_size = 18
legend_size = 18
cb_label_size = 20
HEATMAP_COLORMAP = :balance

fig1 = Figure(size = (1500, 2100), font = font_family)

ax_hm_lang = Axis(fig1[1, 1]; title = "Langevin snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode",
                  titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
hm_lang = heatmap!(ax_hm_lang, 1:n_hm, 1:L, heatmap_langevin';
                   colormap = HEATMAP_COLORMAP, colorrange = value_bounds_plot)
ax_hm_data = Axis(fig1[1, 2]; title = "Data snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode",
                  titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
hm_data = heatmap!(ax_hm_data, 1:n_hm, 1:L, heatmap_data';
                   colormap = HEATMAP_COLORMAP, colorrange = value_bounds_plot)
Colorbar(fig1[1, 3], hm_data; label = "value", width = 10, labelsize = cb_label_size, ticklabelsize = tick_size)

ax_pdf = Axis(fig1[2, 1]; title = "Averaged univariate PDF", xlabel = "value", ylabel = "density",
              titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
lines!(ax_pdf, pdf_x, pdf_data; color = :black, linewidth = 3.0, label = "Data")
lines!(ax_pdf, pdf_x, pdf_langevin; color = :firebrick, linewidth = 3.0, linestyle = :dash, label = "Langevin")
ylims!(ax_pdf, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
xlims!(ax_pdf, pdf_bounds...)
axislegend(ax_pdf, position = :rt, framevisible = false, labelsize = legend_size)

ax_acf = Axis(fig1[2, 2]; title = "Average ACF", xlabel = "lag (time units)", ylabel = "ACF",
              titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
lines!(ax_acf, lags_data_plot, acf_data_plot; color = :black, linewidth = 3.0, label = "Data")
lines!(ax_acf, lags_lang_plot, acf_langevin_plot; color = :firebrick, linewidth = 3.0, linestyle = :dash, label = "Langevin")
hlines!(ax_acf, [0.0]; color = :gray, linestyle = :dot, linewidth = 1.5)
xlims!(ax_acf, 0, 100)
ylims!(ax_acf, -0.1, 1.05)
axislegend(ax_acf, position = :rt, framevisible = false, labelsize = legend_size)

for (row, offset) in enumerate(biv_offsets_sorted)
    spec = bivariate_specs[offset]
    density_max = max(maximum(spec.langevin), maximum(spec.data))
    density_max = density_max == 0 ? 1e-9 : density_max
    ax_l = Axis(fig1[row + 2, 1];
                title = "Langevin (Identity), offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]")
    hm_l = heatmap!(ax_l, spec.x, spec.y, spec.langevin;
                    colormap = HEATMAP_COLORMAP, colorrange = (0, density_max))

    ax_r = Axis(fig1[row + 2, 2];
                title = "Data, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]")
    hm_r = heatmap!(ax_r, spec.x, spec.y, spec.data;
                    colormap = HEATMAP_COLORMAP, colorrange = (0, density_max))
    Colorbar(fig1[row + 2, 3], hm_r; label = "density", width = 10)
end

mkpath(dirname(FIG_PATH))
save(FIG_PATH, fig1; px_per_unit = 1)

# Alternative layout: 2 rows x 4 columns (row 1: snapshots/PDF/ACF, row 2: bivariates)
fig2 = Figure(size = (1800, 900), font = font_family)

ax_hm_lang2 = Axis(fig2[1, 1]; title = "Langevin snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode",
                   titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
hm_lang2 = heatmap!(ax_hm_lang2, 1:n_hm, 1:L, heatmap_langevin';
                    colormap = HEATMAP_COLORMAP, colorrange = value_bounds_plot)
ax_hm_data2 = Axis(fig2[1, 2]; title = "Data snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode",
                   titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
hm_data2 = heatmap!(ax_hm_data2, 1:n_hm, 1:L, heatmap_data';
                    colormap = HEATMAP_COLORMAP, colorrange = value_bounds_plot)

ax_pdf2 = Axis(fig2[1, 3]; title = "PDF", xlabel = "value", ylabel = "density",
               titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
lines!(ax_pdf2, pdf_x, pdf_data; color = :black, linewidth = 2.5, label = "Data")
lines!(ax_pdf2, pdf_x, pdf_langevin; color = :firebrick, linewidth = 2.5, linestyle = :dash, label = "Langevin")
ylims!(ax_pdf2, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
xlims!(ax_pdf2, pdf_bounds...)
axislegend(ax_pdf2, position = :rt, framevisible = false, labelsize = legend_size)

ax_acf2 = Axis(fig2[1, 4]; title = "ACF", xlabel = "lag (time units)", ylabel = "ACF",
               titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
lines!(ax_acf2, lags_data_plot, acf_data_plot; color = :black, linewidth = 2.5, label = "Data")
lines!(ax_acf2, lags_lang_plot, acf_langevin_plot; color = :firebrick, linewidth = 2.5, linestyle = :dash, label = "Langevin")
hlines!(ax_acf2, [0.0]; color = :gray, linestyle = :dot, linewidth = 1.3)
xlims!(ax_acf2, 0, 100)
ylims!(ax_acf2, -0.1, 1.05)
axislegend(ax_acf2, position = :rt, framevisible = false, labelsize = legend_size)

# Row 2: bivariates with shared columns (col1 Langevin, col2 Data), additional colorbars on cols 3–4
gl_biv_lang = fig2[2, 1] = GridLayout(tellwidth = false)
gl_biv_data = fig2[2, 2] = GridLayout(tellwidth = false)

biv_hm_lang = Heatmap[]
biv_hm_data = Heatmap[]
for (j, offset) in enumerate(biv_offsets_sorted)
    axl = Axis(gl_biv_lang[1, j]; title = "Lang offset=$offset", xlabel = "x[i]", ylabel = "x[i+$offset]",
               titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
    hm = heatmap!(axl, bivariate_specs[offset].x, bivariate_specs[offset].y, bivariate_specs[offset].langevin;
                  colormap = HEATMAP_COLORMAP, colorrange = (0, maximum(bivariate_specs[offset].langevin)))
    push!(biv_hm_lang, hm)

    axd = Axis(gl_biv_data[1, j]; title = "Data offset=$offset", xlabel = "x[i]", ylabel = "x[i+$offset]",
               titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
    hm_d = heatmap!(axd, bivariate_specs[offset].x, bivariate_specs[offset].y, bivariate_specs[offset].data;
                    colormap = HEATMAP_COLORMAP, colorrange = (0, maximum(bivariate_specs[offset].data)))
    push!(biv_hm_data, hm_d)
end

# Colorbars for heatmaps and last bivariate maps
Colorbar(fig2[2, 3], hm_data2; label = "value", width = 12, labelsize = cb_label_size, ticklabelsize = tick_size)
!isempty(biv_hm_data) && Colorbar(fig2[2, 4], last(biv_hm_data); label = "density", width = 12, labelsize = cb_label_size, ticklabelsize = tick_size)

fig2_path = replace(FIG_PATH, ".png" => "_grid.png")
save(fig2_path, fig2; px_per_unit = 1)

# 2-row x 4-column layout with uniformly sized panels
fig3 = Figure(size = (2300, 950), font = font_family)

# Row 1: snapshots, snapshots, PDF, ACF
ax3_hm_lang = Axis(fig3[1, 1]; title = "Langevin snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode",
                   titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
hm3_lang = heatmap!(ax3_hm_lang, 1:n_hm, 1:L, heatmap_langevin'; colormap = HEATMAP_COLORMAP, colorrange = value_bounds_plot)

ax3_hm_data = Axis(fig3[1, 2]; title = "Data snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode",
                   titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
hm3_data = heatmap!(ax3_hm_data, 1:n_hm, 1:L, heatmap_data'; colormap = HEATMAP_COLORMAP, colorrange = value_bounds_plot)

Colorbar(fig3[1, 3], hm3_data; label = "value", width = 12, labelsize = cb_label_size, ticklabelsize = tick_size)

ax3_pdf = Axis(fig3[1, 4]; title = "PDF", xlabel = "value", ylabel = "PDF",
               titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
lines!(ax3_pdf, pdf_x, pdf_data; color = :black, linewidth = 2.5, label = "Data")
lines!(ax3_pdf, pdf_x, pdf_langevin; color = :firebrick, linewidth = 2.5, linestyle = :dash, label = "Langevin")
ylims!(ax3_pdf, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
xlims!(ax3_pdf, pdf_bounds...)
axislegend(ax3_pdf, position = :rt, framevisible = false, labelsize = legend_size)

ax3_acf = Axis(fig3[1, 5]; title = "ACF", xlabel = "lag (time units)", ylabel = "ACF",
               titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
lines!(ax3_acf, lags_data_plot, acf_data_plot; color = :black, linewidth = 2.5, label = "Data")
lines!(ax3_acf, lags_lang_plot, acf_langevin_plot; color = :firebrick, linewidth = 2.5, linestyle = :dash, label = "Langevin")
hlines!(ax3_acf, [0.0]; color = :gray, linestyle = :dot, linewidth = 1.3)
xlims!(ax3_acf, 0, 100)
ylims!(ax3_acf, -0.1, 1.05)
axislegend(ax3_acf, position = :rt, framevisible = false, labelsize = legend_size)

# Row 2: two bivariate offsets, Langevin/Data side by side with shared colorbar
offsets_for_grid = biv_offsets_sorted[1:min(end, 2)]
density_global = maximum([maximum(bivariate_specs[o].langevin) for o in offsets_for_grid] +
                         [maximum(bivariate_specs[o].data) for o in offsets_for_grid])
density_global = density_global == 0 ? 1e-9 : density_global
last_hm_b = Ref{Any}(nothing)
for (k, offset) in enumerate(offsets_for_grid)
    base_col = (k - 1) * 3
    col_lang = 1 + base_col
    col_data = 2 + base_col
    spec = bivariate_specs[offset]

    ax_l = Axis(fig3[2, col_lang];
                title = "Langevin, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]",
                titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
    hm_b_l = heatmap!(ax_l, spec.x, spec.y, spec.langevin;
                      colormap = HEATMAP_COLORMAP, colorrange = (0, density_global))

    ax_d = Axis(fig3[2, col_data];
                title = "Data, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]",
                titlesize = title_size, xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_size, yticklabelsize = tick_size)
    hm_b_d = heatmap!(ax_d, spec.x, spec.y, spec.data;
                      colormap = HEATMAP_COLORMAP, colorrange = (0, density_global))
    last_hm_b[] = hm_b_d
end

# Shared colorbars
Colorbar(fig3[1, 3], hm3_data; label = "value", width = 12, labelsize = cb_label_size, ticklabelsize = tick_size)
last_hm_b[] === nothing || Colorbar(fig3[2, 6], last_hm_b[]; label = "density", width = 12, labelsize = cb_label_size, ticklabelsize = tick_size)

fig3_path = replace(FIG_PATH, ".png" => "_grid_4x2.png")
save(fig3_path, fig3; px_per_unit = 1)

@info "Mixed Langevin figures saved" FIG_PATH fig2_path fig3_path
