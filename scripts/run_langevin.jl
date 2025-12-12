#!/usr/bin/env julia

# Generic Langevin runner driven by a TOML config (langevin_params.toml).
# - Reads all parameters (Phi/Sigma mode, devices, seeds, dt/resolution, paths).
# - Performs a single Langevin integration and uses it for every panel.
# - Writes trajectories and metadata to HDF5.
# - Builds Figure 1 (heatmaps, PDF, ACF, bivariate KDEs).
# No code is executed when this file is created; run manually via julia --project=. scripts/run_lungevin.jl

# nohup julia --project=. scripts/run_langevin.jl > run_langevin.log 2>&1 &

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
using CUDA
using TOML
import ScoreUNet1D.EnsembleIntegrator: CPU_STATE_CACHE, GPU_STATE_CACHE

CairoMakie.activate!()

const PROJECT_ROOT = dirname(@__DIR__)
const CONFIG_PATH = joinpath(@__DIR__, "langevin_params.toml")

############################
# Helper utilities
############################
rel_to_root(path::AbstractString) = isabspath(path) ? path : joinpath(PROJECT_ROOT, path)

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
    error("Could not match dataset layout to Phi dimension = $target_dim")
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

build_integrator(model, sigma_model, L, C, D, device::String) =
    ScoreUNet1D.build_snapshot_integrator(
        ScoreUNet1D.ScoreWrapper(model, sigma_model, L, C, D);
        device = device
    )

function clear_state_cache!(device::String)
    device == "cpu" && empty!(CPU_STATE_CACHE)
    device == "gpu" && empty!(GPU_STATE_CACHE)
    return nothing
end

function maybe_switch_gpu!(device::String, gpu_id::Int)
    if device == "gpu"
        devices = collect(CUDA.devices())
        @assert gpu_id >= 0 "GPU id must be nonnegative"
        n_dev = length(devices)
        @assert gpu_id + 1 <= n_dev "Requested GPU id $gpu_id not available; found $n_dev devices"
        CUDA.device!(devices[gpu_id + 1])  # ids here are 0-based constants; CUDA.devices is 1-based
    end
    return nothing
end

function normalize_boundary(boundary)
    boundary === nothing && return nothing
    boundary === false && return nothing
    boundary === true && error("Boundary cannot be `true`; provide (min, max) or set to false/nothing")
    boundary isa Tuple && length(boundary) == 2 && return boundary
    boundary isa AbstractVector && length(boundary) == 2 && return (boundary[1], boundary[2])
    error("Unsupported boundary specification; use (min, max) tuple or set to false/nothing")
end

function smooth_vec(v; w::Int = 5)
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w)
        hi = min(n, i + w)
        out[i] = mean(@view v[lo:hi])
    end
    return out
end

function save_outputs(path::AbstractString;
                      traj,
                      phi_sigma_mode::Symbol,
                      Phi_used,
                      Sigma_used,
                      model_path::AbstractString,
                      phi_sigma_path::Union{Nothing,String},
                      data_path::AbstractString,
                      data_path_hr::AbstractString,
                      dataset_key::String,
                      dataset_orientation::Symbol,
                      boundary,
                      dt::Float64,
                      resolution::Int,
                      n_steps::Int,
                      burn_in::Int,
                      n_ensembles::Int,
                      device::String,
                      gpu_id::Int,
                      base_seed::Int,
                      run_seed::Int,
                      pdf_stride::Int,
                      pdf_nbins::Int,
                      acf_max_lag::Int,
                      biv_offsets,
                      biv_npoints::Int,
                      biv_stride::Int,
                      heatmap_snapshots::Int,
                      boundary_enabled::Bool)
    mkpath(dirname(path))
    h5open(path, "w") do h5
        write(h5, "trajectory", traj)
        write(h5, "trajectory_single", traj)
        write(h5, "Phi", Phi_used)
        write(h5, "Sigma", Sigma_used)
        write(h5, "phi_sigma_mode", String(phi_sigma_mode))
        write(h5, "model_path", model_path)
        phi_sigma_path !== nothing && write(h5, "phi_sigma_path", phi_sigma_path)
        write(h5, "config_path", CONFIG_PATH)
        write(h5, "data_path", data_path)
        write(h5, "data_path_hr", data_path_hr)
        write(h5, "dataset_key", dataset_key)
        write(h5, "dataset_orientation", String(dataset_orientation))

        params = create_group(h5, "langevin_params")
        write(params, "dt", dt)
        write(params, "resolution", Int(resolution))
        write(params, "dt_effective", dt * resolution)
        write(params, "n_steps", Int(n_steps))
        write(params, "burn_in", Int(burn_in))
        write(params, "n_ensembles", Int(n_ensembles))
        write(params, "device", device)
        write(params, "gpu_id", Int(gpu_id))
        write(params, "base_seed", Int(base_seed))
        write(params, "run_seed", Int(run_seed))
        write(params, "pdf_stride", Int(pdf_stride))
        write(params, "pdf_nbins", Int(pdf_nbins))
        write(params, "acf_max_lag", Int(acf_max_lag))
        write(params, "biv_offsets", collect(Int.(biv_offsets)))
        write(params, "biv_npoints", Int(biv_npoints))
        write(params, "biv_stride", Int(biv_stride))
        write(params, "heatmap_snapshots", Int(heatmap_snapshots))
        write(params, "boundary_enabled", boundary_enabled)
        boundary_enabled && write(params, "boundary", collect(Float64.(boundary)))
    end
    return path
end

############################
# Load configuration
############################
cfg = TOML.parsefile(CONFIG_PATH)

paths_cfg = get(cfg, "paths", Dict())
data_path = rel_to_root(get(paths_cfg, "data_path", "data/new_ks.hdf5"))
data_path_hr = rel_to_root(get(paths_cfg, "data_path_hr", "data/new_ks_hr.hdf5"))
dataset_key = get(paths_cfg, "dataset_key", "timeseries")
dataset_orientation_str = lowercase(get(paths_cfg, "dataset_orientation", "columns"))
@assert dataset_orientation_str in ("rows", "columns") "dataset_orientation must be rows or columns"
dataset_orientation = dataset_orientation_str == "rows" ? :rows : :columns
model_folder = get(paths_cfg, "model_folder", "model1")
model_dir = rel_to_root(get(paths_cfg, "model_dir", joinpath("data", "models", model_folder)))
model_filename = get(paths_cfg, "model_filename", "model.bson")
phi_sigma_filename = get(paths_cfg, "phi_sigma_filename", "phi_sigma.hdf5")
output_dir = rel_to_root(get(paths_cfg, "output_dir", "plot_data"))
figure_filename = get(paths_cfg, "figure_filename", "publication_langevin_vs_data.png")
timeseries_filename = get(paths_cfg, "timeseries_filename", "ks_lang.hdf5")

phi_cfg = get(cfg, "phi_sigma", Dict())
phi_sigma_mode_str = lowercase(get(phi_cfg, "mode", "file"))
@assert phi_sigma_mode_str in ("identity", "file") "phi_sigma.mode must be identity or file"
phi_sigma_mode = phi_sigma_mode_str == "identity" ? :identity : :file

device_cfg = get(cfg, "devices", Dict())
device_main = lowercase(get(device_cfg, "pdf_device", get(device_cfg, "device", "cpu")))
gpu_id_main = Int(get(device_cfg, "pdf_gpu_id", get(device_cfg, "gpu_id", 0)))

rng_cfg = get(cfg, "rng", Dict())
base_seed = Int(get(rng_cfg, "base_seed", 2025))
run_seed = Int(get(rng_cfg, "run_seed_offset", 2) + base_seed)
value_seed = Int(get(rng_cfg, "value_bounds_seed", base_seed + 7))

langevin_cfg = get(cfg, "langevin", get(cfg, "langevin_pdf", Dict()))
dt_main = Float64(get(langevin_cfg, "dt", 2.5e-3))
res_main = Int(get(langevin_cfg, "resolution", 400))
n_steps_main = Int(get(langevin_cfg, "n_steps", 10_000))
burn_in_main = Int(get(langevin_cfg, "burn_in", 2_000))
n_ens_main = Int(get(langevin_cfg, "n_ensembles", 1))

analysis_cfg = get(cfg, "analysis", Dict())
pdf_stride = Int(get(analysis_cfg, "pdf_stride", 25))
pdf_nbins = Int(get(analysis_cfg, "pdf_nbins", 256))
acf_max_lag = Int(get(analysis_cfg, "acf_max_lag", 100))
biv_offsets = get(analysis_cfg, "biv_offsets", [1, 2, 3])
biv_npoints = Int(get(analysis_cfg, "biv_npoints", 160))
biv_stride = Int(get(analysis_cfg, "biv_stride", 10))
heatmap_snapshots = Int(get(analysis_cfg, "heatmap_snapshots", 1_000))
quantile_low = Float64(get(analysis_cfg, "value_quantile_low", 0.001))
quantile_high = Float64(get(analysis_cfg, "value_quantile_high", 0.999))
quantile_stride = Int(get(analysis_cfg, "value_quantile_stride", 20))

boundary_cfg = get(cfg, "boundary", Dict())
boundary_enabled = get(boundary_cfg, "enabled", false)
boundary = boundary_enabled ? (boundary_cfg["min"], boundary_cfg["max"]) : nothing
boundary = normalize_boundary(boundary)

############################
# Paths and models
############################
model_path = rel_to_root(joinpath(model_dir, model_filename))
phi_sigma_path = rel_to_root(joinpath(model_dir, phi_sigma_filename))
fig_path = joinpath(output_dir, figure_filename)
timeseries_path = joinpath(output_dir, timeseries_filename)

@assert device_main in ("cpu", "gpu") "device must be cpu or gpu"
@assert dt_main * res_main > 0 "effective dt must be positive"
@assert isfile(model_path) "model file not found at $(model_path)"
phi_sigma_mode == :file && @assert isfile(phi_sigma_path) "phi_sigma file not found at $(phi_sigma_path)"

############################
# Load model and phi/sigma
############################
model_artifacts = BSON.load(model_path)
model = Flux.cpu(model_artifacts[:model])
sigma_model = haskey(model_artifacts, :trainer_cfg) ? Float32(model_artifacts[:trainer_cfg].sigma) : 0.1f0
Flux.testmode!(model)

alpha_raw = 1.0
Phi_scaled = nothing
Sigma_scaled = nothing
dataset = nothing
data_orientation = dataset_orientation

if phi_sigma_mode == :file
    alpha_raw, Phi_raw, Sigma_raw, _ = load_phi_sigma(phi_sigma_path)
    Phi_scaled = alpha_raw .* Phi_raw
    Sigma_scaled = sqrt(alpha_raw) .* Sigma_raw
    dataset, data_orientation = select_dataset(data_path, dataset_key, size(Phi_scaled, 1))
else
    dataset = ScoreUNet1D.load_hdf5_dataset(data_path;
                                            dataset_key = dataset_key,
                                            samples_orientation = dataset_orientation)
end

data_clean = dataset.data
data_stats = dataset.stats
dt_data = h5open(data_path_hr, "r") do h5
    haskey(h5, "dt") ? read(h5, "dt") : 1.0
end

L, C, B = size(data_clean)
D = L * C

if phi_sigma_mode == :identity
    Phi_scaled = Matrix{Float32}(I, D, D)
    Sigma_scaled = Matrix{Float32}(I, D, D)
else
    Phi_scaled = Float32.(Phi_scaled)
    Sigma_scaled = Float32.(Sigma_scaled)
end

############################
# Langevin integration
############################
rng_run = MersenneTwister(run_seed)

clear_state_cache!("cpu"); clear_state_cache!("gpu")
maybe_switch_gpu!(device_main, gpu_id_main)
integrator = build_integrator(model, sigma_model, L, C, D, device_main)

x0 = Matrix{Float32}(undef, D, n_ens_main)
@inbounds for i in 1:n_ens_main
    idx = rand(rng_run, 1:B)
    x0[:, i] = reshape(@view(data_clean[:, :, idx]), D)
end

boundary_eff = normalize_boundary(boundary)

traj_state = integrator(x0, Float32.(Phi_scaled), Float32.(Sigma_scaled);
                        dt = dt_main,
                        n_steps = n_steps_main,
                        burn_in = burn_in_main,
                        resolution = res_main,
                        boundary = boundary_eff,
                        progress = false,
                        progress_desc = "Langevin run ($phi_sigma_mode)")
traj = Array(traj_state)

T_snap = size(traj, 2)
traj_tensor = reshape(traj, L, C, T_snap, :)
langevin_tensor = reshape(traj_tensor, L, C, :)

############################
# Persist trajectories and params
############################
save_outputs(timeseries_path;
             traj = traj_tensor,
             phi_sigma_mode = phi_sigma_mode,
             Phi_used = Float32.(Phi_scaled),
             Sigma_used = Float32.(Sigma_scaled),
             model_path = model_path,
             phi_sigma_path = phi_sigma_mode == :file ? phi_sigma_path : nothing,
             data_path = data_path,
             data_path_hr = data_path_hr,
             dataset_key = dataset_key,
             dataset_orientation = data_orientation,
             boundary = boundary_eff,
             dt = dt_main,
             resolution = res_main,
             n_steps = n_steps_main,
             burn_in = burn_in_main,
             n_ensembles = n_ens_main,
             device = device_main,
             gpu_id = gpu_id_main,
             base_seed = base_seed,
             run_seed = run_seed,
             pdf_stride = pdf_stride,
             pdf_nbins = pdf_nbins,
             acf_max_lag = acf_max_lag,
             biv_offsets = biv_offsets,
             biv_npoints = biv_npoints,
             biv_stride = biv_stride,
             heatmap_snapshots = heatmap_snapshots,
             boundary_enabled = boundary_eff !== nothing)

############################
# Data products for Figure 1
############################
n_hm = min(heatmap_snapshots, size(traj_tensor, 3))
heatmap_langevin = Array(traj_tensor[:, 1, 1:n_hm, 1])
heatmap_data = Array(@view data_clean[:, 1, 1:n_hm])

value_bounds = quantile_bounds(data_clean, langevin_tensor;
                               stride = quantile_stride,
                               probs = (quantile_low, quantile_high),
                               seed = value_seed)
pdf_x, pdf_langevin = averaged_univariate_pdf(langevin_tensor;
                                              nbins = pdf_nbins,
                                              stride = pdf_stride,
                                              bounds = value_bounds)
_, pdf_data = averaged_univariate_pdf(data_clean;
                                      nbins = pdf_nbins,
                                      stride = pdf_stride,
                                      bounds = value_bounds)

acf_langevin = ScoreUNet1D.PhiSigmaEstimator.average_component_acf_data(
    reshape(langevin_tensor, D, :), acf_max_lag)
acf_data = ScoreUNet1D.PhiSigmaEstimator.average_component_acf_data(
    reshape(data_clean, D, :), acf_max_lag)
lags_data = collect(0:acf_max_lag) .* dt_data
lags_langevin = collect(0:acf_max_lag) .* (dt_main * res_main)

bivariate_specs = Dict{Int,NamedTuple}()
for offset in biv_offsets
    xg, yg, dens_lang = pair_kde(langevin_tensor, offset;
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
# Derived quantities (no simulation)
############################
heat_range = maximum(abs, vcat(abs.(heatmap_langevin[:]), abs.(heatmap_data[:])))
heat_range = heat_range == 0 ? 1.0 : heat_range
value_bounds_plot = (-heat_range, heat_range)

pdf_bounds = (minimum(pdf_x), maximum(pdf_x))

dt_langevin = length(lags_langevin) > 1 ? (lags_langevin[2] - lags_langevin[1]) : 1.0

acf_langevin_sm = smooth_vec(acf_langevin; w = 2)
acf_data_sm = smooth_vec(acf_data; w = 2)
if !isempty(acf_langevin_sm)
    acf_langevin_sm[1] = acf_langevin[1]
end
if !isempty(acf_data_sm)
    acf_data_sm[1] = acf_data[1]
end

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

ax_hm_lang = Axis(fig1[1, 1]; title = "Langevin snapshots", xlabel = "t (first $(n_hm))", ylabel = "mode")
hm_lang = heatmap!(ax_hm_lang, 1:n_hm, 1:L, heatmap_langevin';
                   colormap = :batlow, colorrange = value_bounds_plot)
ax_hm_data = Axis(fig1[1, 2]; title = "Data (high-res)", xlabel = "t (first $(n_hm))", ylabel = "mode")
hm_data = heatmap!(ax_hm_data, 1:n_hm, 1:L, heatmap_data';
                   colormap = :batlow, colorrange = value_bounds_plot)
Colorbar(fig1[1, 3], hm_data; label = "value", width = 10)

ax_pdf = Axis(fig1[2, 1]; title = "Averaged univariate PDF", xlabel = "value", ylabel = "density")
lines!(ax_pdf, pdf_x, pdf_data; color = :black, linewidth = 3.0, label = "data")
lines!(ax_pdf, pdf_x, pdf_langevin; color = :firebrick, linewidth = 3.0, linestyle = :dash, label = "Langevin")
ylims!(ax_pdf, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
xlims!(ax_pdf, pdf_bounds...)
axislegend(ax_pdf, position = :rt, framevisible = false)

ax_acf = Axis(fig1[2, 2]; title = "Average ACF", xlabel = "lag (time units)", ylabel = "ACF")
lines!(ax_acf, lags_data_plot, acf_data_plot; color = :black, linewidth = 3.0, label = "data (dt=$(round(dt_data, digits=3)))")
lines!(ax_acf, lags_lang_plot, acf_langevin_plot; color = :dodgerblue4, linewidth = 3.0, linestyle = :dash, label = "Langevin (dt=$(round(dt_langevin, digits=3)))")
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

mkpath(dirname(fig_path))
save(fig_path, fig1; px_per_unit = 1)

@info "Langevin run complete" fig_path timeseries_path phi_sigma_mode device_main
