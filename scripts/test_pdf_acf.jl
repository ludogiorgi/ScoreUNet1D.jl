#!/usr/bin/env julia

# CairoMakie publication-grade visualizations for Langevin vs. data
# The code is organized so each block can be executed independently in the REPL.

# nohup julia --project=. scripts/publication_figures.jl > publication_figures.log 2>&1 &


using CairoMakie
using LinearAlgebra
using Random
using Statistics
using KernelDensity
using StatsBase
using HDF5
using BSON
using Flux
using ScoreUNet1D
import ScoreUNet1D.EnsembleIntegrator: GPU_STATE_CACHE

CairoMakie.activate!()

############################
# Paths and static settings
############################
const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
const DATA_PATH_HR = joinpath(PROJECT_ROOT, "data", "new_ks_hr.hdf5")
const MODEL_PATH = joinpath(PROJECT_ROOT, "scripts", "model.bson")
const PHI_SIGMA_PATH = joinpath(PROJECT_ROOT, "data", "phi_sigma.hdf5")
const FIG_CACHE_PATH = joinpath(PROJECT_ROOT, "runs", "publication_figures_payload.hdf5")
const LANGEVIN_SAVE_PATH = joinpath(PROJECT_ROOT, "data", "ks_lang.hdf5")
const FIG1_PATH = joinpath(@__DIR__, "publication_langevin_vs_data.png")
const FIG2_PATH = joinpath(@__DIR__, "publication_operators.png")
const DATASET_KEY = "timeseries"
# Toggle to reuse cached payload (no integration when true)
const USE_SAVED_PAYLOAD = true

# Langevin integration parameters (GPU only)
const LANGEVIN_DT_GPU = 5.0e-3
const LANGEVIN_RESOLUTION_GPU = 200                  # snapshots every 0.1 units
const LANGEVIN_EFFECTIVE_DT_GPU = LANGEVIN_DT_GPU * LANGEVIN_RESOLUTION_GPU
const LANGEVIN_N_STEPS_GPU = 100_000              # total integration steps
const LANGEVIN_BURN_IN_GPU = 10_000
const LANGEVIN_N_ENSEMBLES_GPU = 256

const LANGEVIN_BOUNDARY = nothing

# Analysis / plotting knobs
const PDF_STRIDE = 25                              # temporal stride for 1D KDEs
const PDF_NBINS = 256
const ACF_MAX_LAG_DATA = 100                       # lags (dt=1 for data)
const ACF_STRIDE = 1                               # temporal stride to reduce cost
const BIV_OFFSETS = (1, 2, 3)
const BIV_NPOINTS = 160
const BIV_STRIDE = 10
const HEATMAP_SNAPSHOTS = 1_000
const V_DATA_RESOLUTION = 120                      # subsampling for V_data estimate
const RNG_SEED = 2025

############################
# Helper utilities
############################
function load_phi_sigma(path::AbstractString)
    h5open(path, "r") do h5
        alpha = read(h5, "Alpha")
        Phi = read(h5, "Phi")
        Sigma = read(h5, "Sigma")
        aux = Dict{Symbol,Any}()
        haskey(h5, "Q") && (aux[:Q] = read(h5, "Q"))
        haskey(h5, "pi_vec") && (aux[:pi_vec] = read(h5, "pi_vec"))
        return alpha, Phi, Sigma, aux
    end
end

function select_dataset(path::AbstractString, dataset_key::String, target_dim::Int)
    for orient in (:columns, :rows)
        ds = ScoreUNet1D.load_hdf5_dataset(path;
                                           dataset_key = dataset_key,
                                           samples_orientation = orient)
        if prod(size(ds.data)[1:2]) == target_dim
            return ds, orient
        end
    end
    error("Could not match dataset layout to Φ dimension = $target_dim")
end

function normalize_hr_dataset(path::AbstractString,
                              dataset_key::String,
                              orientation::Symbol,
                              stats::ScoreUNet1D.DataStats)
    raw = ScoreUNet1D.load_hdf5_dataset(path;
                                        dataset_key = dataset_key,
                                        samples_orientation = orientation,
                                        normalize = false)
    aligned = ScoreUNet1D.apply_stats(raw.data, stats)
    return aligned
end

function quantile_bounds(tensors::AbstractArray...;
                         stride::Int = 20,
                         probs::Tuple{Float64,Float64} = (0.001, 0.999))
    rng = MersenneTwister(RNG_SEED + 7)
    buffer = Float64[]
    for tensor in tensors
        flat = vec(tensor)
        step = max(1, stride)
        if length(flat) <= 2_000_000
            append!(buffer, flat[1:step:end])
        else
            ns = min(length(flat) ÷ step, 2_000_000)
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

function average_acf_tensor(tensor::AbstractArray{<:Real,3};
                            max_lag::Int,
                            stride::Int)
    slice = @view tensor[:, :, 1:stride:end]
    flat = reshape(slice, :, size(slice, 3))
    return ScoreUNet1D.PhiSigmaEstimator.average_component_acf_data(flat, max_lag)
end

function average_acf_over_ensembles(tensor::AbstractArray{<:Real,4};
                                    max_lag::Int,
                                    stride::Int)
    L, C, _, n_ens = size(tensor)
    acf_sum = zeros(Float64, max_lag + 1)
    count = 0
    for e in 1:n_ens
        slice = @view tensor[:, :, 1:stride:end, e]
        flat = reshape(slice, L * C, size(slice, 3))
        acf_e = ScoreUNet1D.PhiSigmaEstimator.average_component_acf_data(flat, max_lag)
        acf_sum .+= acf_e
        count += 1
    end
    return acf_sum ./ max(count, 1)
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

function save_payload(path::AbstractString;
                      Phi,
                      Sigma,
                      Phi_S,
                      Phi_A,
                      alpha,
                      V_data,
                      heatmap_langevin,
                      heatmap_data,
                      pdf_x,
                      pdf_langevin,
                      pdf_data,
                      acf_lags_data,
                      acf_lags_langevin,
                      acf_langevin,
                      acf_data,
                      bivariate_specs::Dict{Int,NamedTuple})
    mkpath(dirname(path))
    h5open(path, "w") do h5
        write(h5, "Phi", Phi)
        write(h5, "Sigma", Sigma)
        write(h5, "Phi_S", Phi_S)
        write(h5, "Phi_A", Phi_A)
        write(h5, "Alpha", alpha)
        write(h5, "V_data", V_data)

        grp_heat = create_group(h5, "heatmaps")
        write(grp_heat, "langevin", heatmap_langevin)
        write(grp_heat, "data_hr", heatmap_data)

        grp_uni = create_group(h5, "univariate")
        write(grp_uni, "x", collect(pdf_x))
        write(grp_uni, "langevin", pdf_langevin)
        write(grp_uni, "data", pdf_data)

        grp_acf = create_group(h5, "acf")
        write(grp_acf, "lags_data", acf_lags_data)
        write(grp_acf, "lags_langevin", acf_lags_langevin)
        write(grp_acf, "langevin", acf_langevin)
        write(grp_acf, "data", acf_data)

        grp_bi = create_group(h5, "bivariate")
        for (offset, spec) in sort(collect(bivariate_specs); by = first)
            g = create_group(grp_bi, "offset_$offset")
            write(g, "x", collect(spec.x))
            write(g, "y", collect(spec.y))
            write(g, "langevin", spec.langevin)
            write(g, "data", spec.data)
        end
    end
    return path
end

function load_payload(path::AbstractString=FIG_CACHE_PATH)
    h5open(path, "r") do h5
        bivar = Dict{Int,NamedTuple}()
        if haskey(h5, "bivariate")
            for name in keys(h5["bivariate"])
                offset = parse(Int, split(String(name), "_")[end])
                g = h5["bivariate/$name"]
                bivar[offset] = (; x = read(g, "x"),
                                   y = read(g, "y"),
                                   langevin = read(g, "langevin"),
                                   data = read(g, "data"))
            end
        end
        return (
            Phi = read(h5, "Phi"),
            Sigma = read(h5, "Sigma"),
            Phi_S = read(h5, "Phi_S"),
            Phi_A = read(h5, "Phi_A"),
            Alpha = read(h5, "Alpha"),
            V_data = read(h5, "V_data"),
            heatmap_langevin = read(h5, "heatmaps/langevin"),
            heatmap_data = read(h5, "heatmaps/data_hr"),
            pdf_x = read(h5, "univariate/x"),
            pdf_langevin = read(h5, "univariate/langevin"),
            pdf_data = read(h5, "univariate/data"),
            acf_lags_data = read(h5, "acf/lags_data"),
            acf_lags_langevin = read(h5, "acf/lags_langevin"),
            acf_langevin = read(h5, "acf/langevin"),
            acf_data = read(h5, "acf/data"),
            bivariate = bivar
        )
    end
end

############################
# Load payload or regenerate
############################
if USE_SAVED_PAYLOAD && isfile(FIG_CACHE_PATH)
    payload = load_payload(FIG_CACHE_PATH)
    Phi_scaled = payload.Phi
    Sigma_scaled = payload.Sigma
    Phi_S = payload.Phi_S
    Phi_A = payload.Phi_A
    alpha_raw = payload.Alpha
    V_data = payload.V_data
    heatmap_langevin = payload.heatmap_langevin
    heatmap_data = payload.heatmap_data
    pdf_x = payload.pdf_x
    pdf_langevin = payload.pdf_langevin
    pdf_data = payload.pdf_data
    lags_data = payload.acf_lags_data
    lags_langevin = payload.acf_lags_langevin
    acf_langevin = payload.acf_langevin
    acf_data = payload.acf_data
    bivariate_specs = payload.bivariate
else
    ############################
    # Load model and operators
    ############################
    alpha_raw, Phi_raw, Sigma_raw, _ = load_phi_sigma(PHI_SIGMA_PATH)
    Phi_scaled = alpha_raw .* Phi_raw
    Sigma_scaled = sqrt(alpha_raw) .* Sigma_raw
    Phi_S = 0.5 .* (Phi_scaled + Phi_scaled')
    Phi_A = 0.5 .* (Phi_scaled - Phi_scaled')

    model_artifacts = BSON.load(MODEL_PATH)
    model = Flux.cpu(model_artifacts[:model])
    sigma_model = haskey(model_artifacts, :trainer_cfg) ? Float32(model_artifacts[:trainer_cfg].sigma) : 0.1f0
    Flux.testmode!(model)

    ############################
    # Datasets (main + high-res)
    ############################
    dataset, data_orientation = select_dataset(DATA_PATH, DATASET_KEY, size(Phi_scaled, 1))
    data_clean = dataset.data
    data_stats = dataset.stats

    data_hr = normalize_hr_dataset(DATA_PATH_HR, DATASET_KEY, data_orientation, data_stats)
    dt_data = h5open(DATA_PATH_HR, "r") do h5
        haskey(h5, "dt") ? read(h5, "dt") : 1.0
    end

    L, C, B = size(data_clean)
    D = L * C
    acf_max_time = ACF_MAX_LAG_DATA * dt_data

    ############################
    # Compute V_data (Stein matrix)
    ############################
    rng = MersenneTwister(RNG_SEED)
    data_noisy = data_clean .+ sigma_model .* randn!(rng, similar(data_clean))
    V_data = ScoreUNet1D.PhiSigmaEstimator.compute_V_data(
        data_noisy,
        model,
        sigma_model;
        v_data_resolution = V_DATA_RESOLUTION,
        batch_size = 512
    )

    ############################
    # Langevin integration (GPU only)
    ############################
    score_wrapper_gpu = ScoreUNet1D.ScoreWrapper(model, sigma_model, L, C, D)
    integrator_gpu = ScoreUNet1D.build_snapshot_integrator(score_wrapper_gpu; device = "gpu")
    empty!(GPU_STATE_CACHE)

    x0_gpu = Matrix{Float32}(undef, D, LANGEVIN_N_ENSEMBLES_GPU)
    @inbounds for j in 1:LANGEVIN_N_ENSEMBLES_GPU
        idx = rand(rng, 1:B)
        x0_gpu[:, j] = reshape(@view(data_clean[:, :, idx]), D)
    end

    @assert LANGEVIN_EFFECTIVE_DT_GPU > 0 "GPU effective dt must be positive"

    traj_state_gpu = integrator_gpu(x0_gpu, Float32.(Phi_scaled), Float32.(Sigma_scaled);
                                    dt = LANGEVIN_DT_GPU,
                                    n_steps = LANGEVIN_N_STEPS_GPU,
                                    burn_in = LANGEVIN_BURN_IN_GPU,
                                    resolution = LANGEVIN_RESOLUTION_GPU,
                                    boundary = LANGEVIN_BOUNDARY,
                                    progress = false,
                                    progress_desc = "Langevin GPU (dt=$(LANGEVIN_DT_GPU))")
    traj_gpu = Array(traj_state_gpu)

    T_snap_gpu = size(traj_gpu, 2)
    traj_tensor_gpu = reshape(traj_gpu, L, C, T_snap_gpu, :)
    langevin_tensor_pdf = reshape(traj_tensor_gpu, L, C, :)
    langevin_tensor_acf = traj_tensor_gpu

    ############################
    # Persist full GPU Langevin trajectory
    ############################
    h5open(LANGEVIN_SAVE_PATH, "w") do h5
        # GPU trajectory and metadata (kept under legacy key "trajectory")
        write(h5, "trajectory", traj_tensor_gpu)  # (L, C, snapshots, ensembles)
        write(h5, "trajectory_gpu", traj_tensor_gpu)
        write(h5, "dt_gpu", Float64(LANGEVIN_DT_GPU))
        write(h5, "resolution_gpu", Int(LANGEVIN_RESOLUTION_GPU))
        write(h5, "dt_effective_gpu", Float64(LANGEVIN_EFFECTIVE_DT_GPU))
        write(h5, "n_steps_gpu", Int(LANGEVIN_N_STEPS_GPU))
        write(h5, "burn_in_gpu", Int(LANGEVIN_BURN_IN_GPU))
        write(h5, "n_ensembles_gpu", Int(LANGEVIN_N_ENSEMBLES_GPU))
    end

    ############################
    # Data products for figures
    ############################
    n_hm = min(HEATMAP_SNAPSHOTS, size(traj_tensor_gpu, 3))
    heatmap_langevin = Array(traj_tensor_gpu[:, 1, 1:n_hm, 1])
    heatmap_data = Array(@view data_hr[:, 1, 1:n_hm])

    value_bounds = quantile_bounds(data_clean, langevin_tensor_pdf; stride = 50)
    pdf_x, pdf_langevin = averaged_univariate_pdf(langevin_tensor_pdf;
                                                  nbins = PDF_NBINS,
                                                  stride = PDF_STRIDE,
                                                  bounds = value_bounds)
    _, pdf_data = averaged_univariate_pdf(data_clean;
                                          nbins = PDF_NBINS,
                                          stride = PDF_STRIDE,
                                          bounds = value_bounds)

    downsample_factor = max(1, round(Int, 1.0 / LANGEVIN_EFFECTIVE_DT_GPU))
    langevin_down = @view langevin_tensor_acf[:, :, 1:downsample_factor:end, :]

    # ACF: compute per-ensemble/per-dimension and average across all of them
    acf_langevin = average_acf_over_ensembles(langevin_down; max_lag = ACF_MAX_LAG_DATA, stride = ACF_STRIDE)
    acf_data = average_acf_tensor(data_noisy; max_lag = ACF_MAX_LAG_DATA, stride = ACF_STRIDE)
    lags_data = collect(0:ACF_MAX_LAG_DATA) .* dt_data
    dt_langevin = LANGEVIN_EFFECTIVE_DT_GPU * downsample_factor * ACF_STRIDE
    lags_langevin = collect(0:ACF_MAX_LAG_DATA) .* dt_langevin

    bivariate_specs = Dict{Int,NamedTuple}()
    for offset in BIV_OFFSETS
        xg, yg, dens_lang = pair_kde(langevin_tensor_pdf, offset;
                                     bounds = value_bounds,
                                     npoints = BIV_NPOINTS,
                                     stride = BIV_STRIDE)
        _, _, dens_data = pair_kde(data_clean, offset;
                                   bounds = value_bounds,
                                   npoints = BIV_NPOINTS,
                                   stride = BIV_STRIDE)
        bivariate_specs[offset] = (; x = xg, y = yg,
                                    langevin = dens_lang,
                                    data = dens_data)
    end

    ############################
    # Persist payload for reuse
    ############################
    save_payload(FIG_CACHE_PATH;
                 Phi = Phi_scaled,
                 Sigma = Sigma_scaled,
                 Phi_S = Phi_S,
                 Phi_A = Phi_A,
                 alpha = alpha_raw,
                 V_data = V_data,
                 heatmap_langevin = heatmap_langevin,
                 heatmap_data = heatmap_data,
                 pdf_x = pdf_x,
                 pdf_langevin = pdf_langevin,
                 pdf_data = pdf_data,
                 acf_lags_data = lags_data,
                 acf_lags_langevin = lags_langevin,
                 acf_langevin = acf_langevin,
                 acf_data = acf_data,
                 bivariate_specs = bivariate_specs)
end

############################
# Derived quantities (no simulation)
############################
L = size(heatmap_langevin, 1)
heat_range = maximum(abs, vcat(abs.(heatmap_langevin[:]), abs.(heatmap_data[:])))
heat_range = heat_range == 0 ? 1.0 : heat_range
value_bounds = (-heat_range, heat_range)

# Grid bounds for bivariate plots fall back to pdf_x when present
pdf_bounds = (minimum(pdf_x), maximum(pdf_x))

dt_data = length(lags_data) > 1 ? (lags_data[2] - lags_data[1]) : 1.0
dt_langevin = length(lags_langevin) > 1 ? (lags_langevin[2] - lags_langevin[1]) : 1.0

smooth_vec(v; w::Int=5) = begin
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w)
        hi = min(n, i + w)
        out[i] = mean(@view v[lo:hi])
    end
    out
end
acf_langevin_sm = smooth_vec(acf_langevin; w = 2)
acf_data_sm = smooth_vec(acf_data; w = 2)

# Preserve normalized value at lag 0 (ACF(0) ≈ 1)
if !isempty(acf_langevin_sm)
    acf_langevin_sm[1] = acf_langevin[1]
end
if !isempty(acf_data_sm)
    acf_data_sm[1] = acf_data[1]
end

# Align ACF ranges to 0..100 time units (already matched dt=1 for both)
acf_tmax = 100.0
mask_data = lags_data .<= acf_tmax
mask_lang = lags_langevin .<= acf_tmax
lags_data_plot = lags_data[mask_data]
lags_lang_plot = lags_langevin[mask_lang]
acf_data_plot = acf_data_sm[mask_data]
acf_langevin_plot = acf_langevin_sm[mask_lang]

biv_offsets_sorted = sort(collect(keys(bivariate_specs)))

############################
# Figure 1: Langevin vs. data
############################
fig1 = Figure(size = (1500, 2100), font = "TeX Gyre Heros")

ax_hm_lang = Axis(fig1[1, 1]; title = "Langevin snapshots", xlabel = "t (first $(HEATMAP_SNAPSHOTS))", ylabel = "mode")
hm_lang = heatmap!(ax_hm_lang, 1:HEATMAP_SNAPSHOTS, 1:L, heatmap_langevin';
                   colormap = :batlow, colorrange = value_bounds)
ax_hm_data = Axis(fig1[1, 2]; title = "Data (new_ks_hr)", xlabel = "t (first $(HEATMAP_SNAPSHOTS))", ylabel = "mode")
hm_data = heatmap!(ax_hm_data, 1:HEATMAP_SNAPSHOTS, 1:L, heatmap_data';
                   colormap = :batlow, colorrange = value_bounds)
Colorbar(fig1[1, 3], hm_data; label = "value", width = 10)

ax_pdf = Axis(fig1[2, 1]; title = "Averaged univariate PDF", xlabel = "value", ylabel = "density")
lines!(ax_pdf, pdf_x, pdf_data; color = :black, linewidth = 3.0, label = "data")
lines!(ax_pdf, pdf_x, pdf_langevin; color = :firebrick, linewidth = 3.0, linestyle = :dash, label = "Langevin")
ylims!(ax_pdf, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
xlims!(ax_pdf, pdf_bounds...)
axislegend(ax_pdf, position = :rt, framevisible = false)

ax_acf = Axis(fig1[2, 2]; title = "Average ACF", xlabel = "lag (time units)", ylabel = "ACF")
lines!(ax_acf, lags_data_plot, acf_data_plot; color = :black, linewidth = 3.0, label = "data (Δt=$(round(dt_data, digits=3)))")
lines!(ax_acf, lags_lang_plot, acf_langevin_plot; color = :dodgerblue4, linewidth = 3.0, linestyle = :dash, label = "Langevin (Δt=$(round(dt_langevin, digits=3)))")
hlines!(ax_acf, [0.0]; color = :gray, linestyle = :dot, linewidth = 1.5)
xlims!(ax_acf, 0, 100)
ylims!(ax_acf, -0.1, 1.05)
axislegend(ax_acf, position = :rt, framevisible = false)

for (row, offset) in enumerate(biv_offsets_sorted)
    spec = bivariate_specs[offset]
    density_max = max(maximum(spec.langevin), maximum(spec.data))
    density_max = density_max == 0 ? 1e-9 : density_max
    ax_l = Axis(fig1[row + 2, 1];
                title = "Langevin, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]")
    hm_l = heatmap!(ax_l, spec.x, spec.y, spec.langevin;
                    colormap = :vik, colorrange = (0, density_max))

    ax_r = Axis(fig1[row + 2, 2];
                title = "Data, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]")
    hm_r = heatmap!(ax_r, spec.x, spec.y, spec.data;
                    colormap = :vik, colorrange = (0, density_max))
    Colorbar(fig1[row + 2, 3], hm_r; label = "density", width = 10)
end

save(FIG1_PATH, fig1; px_per_unit = 1)

############################
# Figure 2: Operators
############################
fig2 = Figure(size = (2200, 520), font = "TeX Gyre Heros")

global_range = maximum(abs, vcat(abs.(vec(Phi_scaled)),
                                 abs.(vec(Phi_S)),
                                 abs.(vec(Phi_A)),
                                 abs.(vec(Sigma_scaled)),
                                 abs.(vec(V_data))))
global_range = global_range == 0 ? 1.0 : global_range
crange = (-global_range, global_range)
palette = :balance

mats = [
    (Phi_scaled, "Φ"),
    (Phi_S, "Φ_S (symmetric)"),
    (Phi_A, "Φ_A (antisymmetric)"),
    (Sigma_scaled, "Σ"),
    (V_data, "V_data"),
]

last_hm_ref = Ref{Any}(nothing)
for (i, (mat, title)) in enumerate(mats)
    ax = Axis(fig2[1, i];
              title = title,
              xlabel = "index",
              ylabel = "index",
              aspect = DataAspect())
    hm = heatmap!(ax, 1:size(mat, 2), 1:size(mat, 1), mat;
                  colormap = palette, colorrange = crange)
    hidespines!(ax, :t, :r)
    last_hm_ref[] = hm
end
last_hm = last_hm_ref[]
last_hm === nothing && error("Heatmap not created; cannot build colorbar")
Colorbar(fig2[1, length(mats) + 1], last_hm; label = "value", width = 18)

save(FIG2_PATH, fig2; px_per_unit = 1)

############################
# Reload-only workflow
############################
# To regenerate figures without rerunning the integration:
# payload = load_payload()
# (use `payload.*` fields in the plotting sections above)
