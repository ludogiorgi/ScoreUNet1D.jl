"""
    IntegrateRunner

Generic Langevin integration wrapper. Can be used for any dynamical system.
"""
module IntegrateRunner

using BSON
using CUDA
using Flux
using HDF5
using KernelDensity
using LinearAlgebra
using Random
using Statistics
using StatsBase
using TOML

using ..ScoreUNet1D: NormalizedDataset, load_hdf5_dataset, sample_length, num_channels,
    ScoreWrapper, build_snapshot_integrator
using ..EnsembleIntegrator: CPU_STATE_CACHE, GPU_STATE_CACHE
using ..PhiSigmaEstimator: average_acf_ensemble, average_acf_3d
using ..RunnerUtils: load_config, resolve_path, ensure_dir, symbol_from_string,
    load_phi_sigma, timed, verbose_log

export IntegrationResult, integrate_langevin

#─────────────────────────────────────────────────────────────────────────────
# Result Struct
#─────────────────────────────────────────────────────────────────────────────

"""
    IntegrationResult

Result of Langevin integration.

# Fields
- `trajectory`: (L, C, T, E) trajectory tensor
- `Phi`: Drift matrix used
- `Sigma`: Diffusion matrix used
- `phi_sigma_mode`: :identity or :file
- `statistics`: Dict with PDF, ACF, bivariate specs
- `output_path`: Path to saved HDF5 file
- `figure_path`: Path to generated figure (if any)
"""
struct IntegrationResult
    trajectory::Array{Float32}
    Phi::Matrix{Float32}
    Sigma::Matrix{Float32}
    phi_sigma_mode::Symbol
    statistics::Dict{Symbol,Any}
    output_path::String
    figure_path::Union{Nothing,String}
end

#─────────────────────────────────────────────────────────────────────────────
# Helper Functions
#─────────────────────────────────────────────────────────────────────────────

function clear_state_cache!()
    empty!(CPU_STATE_CACHE)
    empty!(GPU_STATE_CACHE)
    return nothing
end

function select_dataset_orientation(path::AbstractString, dataset_key::String, target_dim::Int)
    for orient in (:columns, :rows)
        ds = load_hdf5_dataset(path; dataset_key=dataset_key, samples_orientation=orient)
        if prod(size(ds.data)[1:2]) == target_dim
            return ds, orient
        end
    end
    error("Could not match dataset to Phi dimension = $target_dim")
end

function quantile_bounds(tensors::AbstractArray...; stride::Int=20,
    probs::Tuple{Float64,Float64}=(0.001, 0.999), seed::Int=0)
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

function averaged_univariate_pdf(tensor::AbstractArray{<:Real,3}; nbins::Int,
    stride::Int, bounds::Tuple{Float64,Float64})
    slice = @view tensor[:, :, 1:stride:end]
    grid = range(bounds[1], bounds[2]; length=nbins)
    kd = kde(vec(slice), grid)
    return kd.x, kd.density
end

function pair_kde(tensor::AbstractArray{<:Real,3}, offset::Int;
    bounds::Tuple{Float64,Float64}, npoints::Int, stride::Int)
    L, C, B = size(tensor)
    @assert offset < L "Offset $offset exceeds length $L"
    n = (L - offset) * C * cld(B, stride)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    idx = 1
    @inbounds for b in 1:stride:B, c in 1:C, i in 1:(L-offset)
        xs[idx] = tensor[i, c, b]
        ys[idx] = tensor[i+offset, c, b]
        idx += 1
    end
    xs = xs[1:idx-1]
    ys = ys[1:idx-1]
    grid = range(bounds[1], bounds[2]; length=npoints)
    kd = kde((xs, ys), (grid, grid))
    return kd.x, kd.y, kd.density
end

function smooth_vec(v; w::Int=5)
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w)
        hi = min(n, i + w)
        out[i] = mean(@view v[lo:hi])
    end
    return out
end

function save_trajectory(path::AbstractString; traj, phi_sigma_mode::Symbol,
    Phi, Sigma, model_path::AbstractString, data_path::AbstractString,
    data_path_hr::AbstractString, dataset_key::String,
    dataset_orientation::Symbol, dt::Float64, resolution::Int,
    n_steps::Int, burn_in::Int, n_ensembles::Int, device::String,
    config_path::AbstractString, pdf_stride::Int, pdf_nbins::Int,
    acf_max_lag::Int, biv_offsets, biv_npoints::Int, biv_stride::Int,
    heatmap_snapshots::Int)
    mkpath(dirname(path))
    h5open(path, "w") do h5
        write(h5, "trajectory", traj)
        write(h5, "Phi", Matrix{Float32}(Phi))
        write(h5, "Sigma", Matrix{Float32}(Sigma))
        write(h5, "phi_sigma_mode", String(phi_sigma_mode))
        write(h5, "model_path", model_path)
        write(h5, "config_path", config_path)
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
        write(params, "pdf_stride", Int(pdf_stride))
        write(params, "pdf_nbins", Int(pdf_nbins))
        write(params, "acf_max_lag", Int(acf_max_lag))
        write(params, "biv_offsets", collect(Int.(biv_offsets)))
        write(params, "biv_npoints", Int(biv_npoints))
        write(params, "biv_stride", Int(biv_stride))
        write(params, "heatmap_snapshots", Int(heatmap_snapshots))
    end
    return path
end

#─────────────────────────────────────────────────────────────────────────────
# Main Function
#─────────────────────────────────────────────────────────────────────────────

"""
    integrate_langevin(config_path; project_root=nothing, generate_figure=true) -> IntegrationResult

Integrate Langevin dynamics using configuration from a TOML file.

# Arguments
- `config_path`: Path to TOML configuration file
- `project_root=nothing`: Project root for resolving paths (defaults to parent of config dir)
- `generate_figure=true`: Whether to generate comparison figure

# Returns
- `IntegrationResult` with trajectory and statistics

# Config Sections
- `[paths]`: data_path, model_path, output_dir, etc.
- `[phi_sigma]`: mode ("identity" or "file"), path
- `[device]`: device, gpu_id
- `[langevin]`: dt, resolution, n_steps, burn_in, n_ensembles
- `[analysis]`: pdf_stride, pdf_nbins, acf_max_lag, biv_offsets, etc.
"""
function integrate_langevin(config_path::AbstractString;
    project_root::Union{Nothing,AbstractString}=nothing,
    generate_figure::Bool=true)

    # Determine project root
    if project_root === nothing
        project_root = dirname(dirname(abspath(config_path)))
    end

    @info "Loading integration configuration" config = config_path
    config = load_config(config_path)

    # Extract sections
    paths_cfg = get(config, "paths", Dict{String,Any}())
    phi_sigma_cfg = get(config, "phi_sigma", Dict{String,Any}())
    device_cfg = get(config, "device", Dict{String,Any}())
    langevin_cfg = get(config, "langevin", Dict{String,Any}())
    analysis_cfg = get(config, "analysis", Dict{String,Any}())

    # Paths
    data_path = resolve_path(get(paths_cfg, "data_path", "data/new_ks.hdf5"), project_root)
    data_path_hr = resolve_path(get(paths_cfg, "data_path_hr", data_path), project_root)
    model_path = resolve_path(get(paths_cfg, "model_path", "runs/trained_model.bson"), project_root)
    output_dir = resolve_path(get(paths_cfg, "output_dir", "plot_data"), project_root)
    dataset_key = get(paths_cfg, "dataset_key", "timeseries")
    dataset_orientation = symbol_from_string(get(paths_cfg, "dataset_orientation", "columns"))

    # Phi/Sigma mode
    phi_sigma_mode_str = lowercase(get(phi_sigma_cfg, "mode", "identity"))
    @assert phi_sigma_mode_str in ("identity", "file") "phi_sigma.mode must be 'identity' or 'file'"
    phi_sigma_mode = Symbol(phi_sigma_mode_str)
    phi_sigma_path = phi_sigma_mode == :file ?
                     resolve_path(get(phi_sigma_cfg, "path", "phi_sigma.hdf5"), project_root) : nothing

    # Device
    device_main = lowercase(get(device_cfg, "device", "gpu"))
    gpu_id = Int(get(device_cfg, "gpu_id", 0))

    # Langevin parameters
    dt_main = Float64(get(langevin_cfg, "dt", 0.005))
    resolution = Int(get(langevin_cfg, "resolution", 200))
    n_steps = Int(get(langevin_cfg, "n_steps", 100_000))
    burn_in = Int(get(langevin_cfg, "burn_in", 25_000))
    n_ensembles = Int(get(langevin_cfg, "n_ensembles", 256))
    seed = Int(get(langevin_cfg, "seed", 2025))

    # Boundary enforcement (optional)
    boundary_raw = get(langevin_cfg, "boundary", nothing)
    boundary = if boundary_raw !== nothing && length(boundary_raw) == 2
        (Float64(boundary_raw[1]), Float64(boundary_raw[2]))
    else
        nothing
    end

    # Analysis parameters
    pdf_stride = Int(get(analysis_cfg, "pdf_stride", 25))
    pdf_nbins = Int(get(analysis_cfg, "pdf_nbins", 256))
    acf_max_lag = Int(get(analysis_cfg, "acf_max_lag", 100))
    biv_offsets = Int.(get(analysis_cfg, "biv_offsets", [1, 2, 3]))
    biv_npoints = Int(get(analysis_cfg, "biv_npoints", 160))
    biv_stride = Int(get(analysis_cfg, "biv_stride", 10))
    heatmap_snapshots = Int(get(analysis_cfg, "heatmap_snapshots", 1000))

    # Output filename
    output_filename = phi_sigma_mode == :identity ? "trajectory_identity.hdf5" : "trajectory.hdf5"
    output_path = joinpath(output_dir, output_filename)

    # ─────────────────────────────────────────────────────────────────────────
    # Load Model
    # ─────────────────────────────────────────────────────────────────────────

    @info "Loading model" path = model_path
    @assert isfile(model_path) "Model not found: $model_path"

    model_contents = BSON.load(model_path)
    model = Flux.cpu(model_contents[:model])
    sigma_model = haskey(model_contents, :trainer_cfg) ? Float32(model_contents[:trainer_cfg].sigma) : 0.1f0
    Flux.testmode!(model)

    @info "Model loaded" sigma = sigma_model

    # ─────────────────────────────────────────────────────────────────────────
    # Load Data and Phi/Sigma
    # ─────────────────────────────────────────────────────────────────────────

    local dataset, data_orient, Phi, Sigma

    if phi_sigma_mode == :file
        @info "Loading Φ/Σ from file" path = phi_sigma_path
        @assert isfile(phi_sigma_path) "Phi/Sigma file not found: $phi_sigma_path"
        alpha, Phi_raw, Sigma_raw, _ = load_phi_sigma(phi_sigma_path)
        Phi = Float32.(alpha .* Phi_raw)
        Sigma = Float32.(sqrt(alpha) .* Sigma_raw)
        dataset, data_orient = select_dataset_orientation(data_path, dataset_key, size(Phi, 1))
    else
        @info "Using identity matrices for Φ and Σ"
        dataset = load_hdf5_dataset(data_path; dataset_key=dataset_key, samples_orientation=dataset_orientation)
        data_orient = dataset_orientation
        L, C, _ = size(dataset.data)
        D = L * C
        Phi = Matrix{Float32}(I, D, D)
        Sigma = Matrix{Float32}(I, D, D)
    end

    data_clean = dataset.data
    L, C, B = size(data_clean)
    D = L * C

    dt_data = isfile(data_path_hr) ?
              h5open(h5 -> haskey(h5, "dt") ? read(h5, "dt") : 1.0, data_path_hr, "r") : 1.0

    @info "Data loaded" size = size(data_clean) L = L C = C D = D

    # ─────────────────────────────────────────────────────────────────────────
    # Langevin Integration
    # ─────────────────────────────────────────────────────────────────────────

    @info "Starting Langevin integration" mode = phi_sigma_mode dt = dt_main steps = n_steps ensembles = n_ensembles

    rng_run = MersenneTwister(seed)
    clear_state_cache!()

    if device_main == "gpu" && CUDA.functional()
        devices = collect(CUDA.devices())
        @assert gpu_id + 1 <= length(devices) "GPU $gpu_id not available"
        CUDA.device!(devices[gpu_id+1])
    end

    integrator = build_snapshot_integrator(ScoreWrapper(model, sigma_model, L, C, D); device=device_main)

    x0 = Matrix{Float32}(undef, D, n_ensembles)
    @inbounds for i in 1:n_ensembles
        idx = rand(rng_run, 1:B)
        x0[:, i] = reshape(@view(data_clean[:, :, idx]), D)
    end

    traj_state = integrator(x0, Phi, Sigma;
        dt=dt_main, n_steps=n_steps, burn_in=burn_in,
        resolution=resolution, boundary=boundary,
        progress=true, progress_desc="Langevin ($phi_sigma_mode)")
    traj = Array(traj_state)

    T_snap = size(traj, 2)
    traj_tensor = reshape(traj, L, C, T_snap, :)
    langevin_tensor = reshape(traj_tensor, L, C, :)

    @info "Integration complete" snapshots = T_snap total_samples = size(langevin_tensor, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Save Trajectory
    # ─────────────────────────────────────────────────────────────────────────

    @info "Saving trajectory" path = output_path
    ensure_dir(dirname(output_path))

    save_trajectory(output_path;
        traj=traj_tensor,
        phi_sigma_mode=phi_sigma_mode,
        Phi=Phi, Sigma=Sigma,
        model_path=model_path,
        data_path=data_path,
        data_path_hr=data_path_hr,
        dataset_key=dataset_key,
        dataset_orientation=data_orient,
        dt=dt_main,
        resolution=resolution,
        n_steps=n_steps,
        burn_in=burn_in,
        n_ensembles=n_ensembles,
        device=device_main,
        config_path=config_path,
        pdf_stride=pdf_stride,
        pdf_nbins=pdf_nbins,
        acf_max_lag=acf_max_lag,
        biv_offsets=biv_offsets,
        biv_npoints=biv_npoints,
        biv_stride=biv_stride,
        heatmap_snapshots=heatmap_snapshots)

    # ─────────────────────────────────────────────────────────────────────────
    # Compute Statistics
    # ─────────────────────────────────────────────────────────────────────────

    @info "Computing statistics"

    value_bounds = quantile_bounds(data_clean, langevin_tensor; stride=20, probs=(0.001, 0.999), seed=seed + 7)

    pdf_x, pdf_langevin = averaged_univariate_pdf(langevin_tensor; nbins=pdf_nbins, stride=pdf_stride, bounds=value_bounds)
    _, pdf_data = averaged_univariate_pdf(data_clean; nbins=pdf_nbins, stride=pdf_stride, bounds=value_bounds)

    # Compute ACF properly: per-ensemble per-dimension, then average
    acf_langevin = average_acf_ensemble(traj_tensor, acf_max_lag)
    acf_data = average_acf_3d(data_clean, acf_max_lag)

    lags_data = collect(0:acf_max_lag) .* dt_data
    lags_langevin = collect(0:acf_max_lag) .* (dt_main * resolution)

    bivariate_specs = Dict{Int,NamedTuple}()
    for offset in biv_offsets
        xg, yg, dens_lang = pair_kde(langevin_tensor, offset; bounds=value_bounds, npoints=biv_npoints, stride=biv_stride)
        _, _, dens_data = pair_kde(data_clean, offset; bounds=value_bounds, npoints=biv_npoints, stride=biv_stride)
        bivariate_specs[offset] = (; x=xg, y=yg, langevin=dens_lang, data=dens_data)
    end

    n_hm = min(heatmap_snapshots, size(traj_tensor, 3))
    heatmap_langevin = Array(traj_tensor[:, 1, 1:n_hm, 1])
    heatmap_data = Array(@view data_clean[:, 1, 1:n_hm])

    statistics = Dict{Symbol,Any}(
        :value_bounds => value_bounds,
        :pdf_x => pdf_x,
        :pdf_langevin => pdf_langevin,
        :pdf_data => pdf_data,
        :acf_langevin => acf_langevin,
        :acf_data => acf_data,
        :lags_data => lags_data,
        :lags_langevin => lags_langevin,
        :bivariate_specs => bivariate_specs,
        :heatmap_langevin => heatmap_langevin,
        :heatmap_data => heatmap_data,
        :L => L,
        :dt_data => dt_data,
        :dt_langevin => dt_main * resolution
    )

    @info "Langevin integration complete" output = output_path mode = phi_sigma_mode

    return IntegrationResult(
        traj_tensor,
        Phi,
        Sigma,
        phi_sigma_mode,
        statistics,
        output_path,
        nothing  # Figure path set by separate plotting function
    )
end

end # module
