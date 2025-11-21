#!/usr/bin/env julia

using BSON
using CairoMakie
using Dates
using Flux
using Printf
using Random
using ScoreUNet1D
using ScoreUNet1D: NormalizedDataset, sample_length, num_channels,
    MomentMatchingConfig, DriftDiffusionEstimate,
    CorrelationConfig, CorrelationInfo,
    compute_drift_diffusion, compute_correlation_info
using KernelDensity
using TOML

CairoMakie.activate!()

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))
const PARAMETERS_PATH = joinpath(SCRIPT_DIR, "parameters.toml")
const DEFAULT_LAG_OFFSETS = (1, 2, 3)
const HEATMAP_BINS = 192

time_in_seconds(ns) = float(ns) / 1e9

function verbose_log(verbose::Bool, message::AbstractString; kwargs...)
    verbose || return
    @info message kwargs...
end

function timed(label::AbstractString, verbose::Bool, f::Function)
    verbose_log(verbose, "$label started")
    t0 = time_ns()
    result = f()
    elapsed = time_in_seconds(time_ns() - t0)
    verbose_log(verbose, "$label finished"; seconds=elapsed)
    return result
end

timed(f::Function, label::AbstractString, verbose::Bool) = timed(label, verbose, f)
"""
    subset_dataset(dataset, nmax; seed=0)

Returns a dataset with at most `nmax` samples (drawn without replacement) to keep
training manageable while preserving the global normalization statistics.
"""
function subset_dataset(dataset::NormalizedDataset, nmax::Int; seed::Int=0)
    total = length(dataset)
    nmax <= 0 && return dataset
    total <= nmax && return dataset
    rng = MersenneTwister(seed)
    idxs = collect(1:total)
    Random.shuffle!(rng, idxs)
    idxs = idxs[1:nmax]
    return NormalizedDataset(dataset.data[:, :, idxs], dataset.stats)
end

resolve_path(path::AbstractString) =
    isabspath(path) ? path : normpath(joinpath(PROJECT_ROOT, path))

function load_parameters(path::AbstractString=PARAMETERS_PATH)
    isfile(path) || error("Parameters file not found at $path")
    return TOML.parsefile(path)
end

function symbol_from_string(value::AbstractString)
    return Symbol(lowercase(value))
end

function activation_from_string(name::AbstractString)
    lname = lowercase(name)
    lname == "swish" && return Flux.swish
    lname == "gelu" && return Flux.gelu
    lname == "relu" && return Flux.relu
    lname == "tanh" && return tanh
    lname == "identity" && return identity
    lname == "softplus" && return Flux.softplus
    error("Unsupported activation: $name")
end

function final_activation_from_string(name::AbstractString)
    lname = lowercase(name)
    lname == "identity" && return identity
    lname == "tanh" && return tanh
    lname == "relu" && return Flux.relu
    lname == "swish" && return Flux.swish
    error("Unsupported final activation: $name")
end

function vector_to_int_tuple(values)
    ints = Int.(values)
    return Tuple(ints)
end

function mode_from_value(value)
    if value isa Integer
        return Int(value)
    elseif value isa AbstractString
        return symbol_from_string(value)
    elseif value isa Symbol
        return value
    else
        error("Unsupported mode specification: $value")
    end
end

function build_model_config(params::Dict{String,Any})
    activation = activation_from_string(get(params, "activation", "swish"))
    final_activation = final_activation_from_string(get(params, "final_activation", "identity"))
    cfg = ScoreUNetConfig(
        in_channels = Int(get(params, "in_channels", 1)),
        base_channels = Int(get(params, "base_channels", 32)),
        channel_multipliers = Int.(get(params, "channel_multipliers", [1, 2, 4])),
        kernel_size = Int(get(params, "kernel_size", 5)),
        periodic = get(params, "periodic", false),
        activation = activation,
        final_activation = final_activation,
    )
    init_seed = Int(get(params, "init_seed", 314159))
    return cfg, init_seed
end

function build_trainer_config(params::Dict{String,Any})
    max_steps = get(params, "max_steps_per_epoch", nothing)
    max_steps = max_steps === nothing ? nothing : Int(max_steps)
    cfg = ScoreTrainerConfig(
        batch_size = Int(get(params, "batch_size", 32)),
        epochs = Int(get(params, "epochs", 10)),
        lr = Float64(get(params, "lr", 1e-3)),
        sigma = Float32(get(params, "sigma", 0.05)),
        shuffle = get(params, "shuffle", true),
        seed = Int(get(params, "seed", 42)),
        progress = get(params, "progress", true),
        max_steps_per_epoch = max_steps,
    )
    return cfg
end

function build_langevin_config(params::Dict{String,Any}, trainer_cfg::ScoreTrainerConfig, data_dt::Float64)
    boundary_val = get(params, "boundary", [-10.0, 10.0])
    boundary_tuple = boundary_val === nothing ? nothing :
        (Float64(boundary_val[1]), Float64(boundary_val[2]))
    resolution_val = Int(get(params, "resolution", 10))
    sample_dt_param = Float64(get(params, "sample_dt", get(params, "dt", data_dt)))
    resolution_val <= 0 && error("resolution must be positive")
    sample_dt_param <= 0 && error("sample_dt must be positive")
    integrator_dt = sample_dt_param / resolution_val
    cfg = LangevinConfig(
        dt = integrator_dt,
        sample_dt = sample_dt_param,
        nsteps = Int(get(params, "nsteps", 50_000)),
        resolution = resolution_val,
        n_ensembles = Int(get(params, "n_ensembles", 64)),
        burn_in = Int(get(params, "burn_in", 5_000)),
        nbins = Int(get(params, "nbins", 128)),
        sigma = trainer_cfg.sigma,
        seed = Int(get(params, "seed", 21)),
        mode = mode_from_value(get(params, "mode", "all")),
        boundary = boundary_tuple,
    )
    return cfg
end

function build_output_config(params::Dict{String,Any})
    run_root = resolve_path(get(params, "run_root", "runs"))
    model_repo = resolve_path(get(params, "model_repository_path", "scripts/trained_model.bson"))
    run_model_name = get(params, "run_model_filename", "model.bson")
    offsets = haskey(params, "lag_offsets") ?
        vector_to_int_tuple(params["lag_offsets"]) :
        DEFAULT_LAG_OFFSETS
    return (run_root=run_root,
            model_repository_path=model_repo,
            run_model_filename=run_model_name,
            lag_offsets=offsets)
end

function create_run_directory(run_root::AbstractString)
    mkpath(run_root)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    slug = Random.randstring(4)
    run_dir = joinpath(run_root, "run_$(timestamp)_$(slug)")
    mkpath(run_dir)
    return run_dir
end

function instantiate_model(cfg::ScoreUNetConfig, seed::Int)
    Random.seed!(seed)
    model = build_unet(cfg)
    Random.seed!()
    return model
end

function save_model(path::AbstractString, model, cfg::ScoreUNetConfig)
    mkpath(dirname(path))
    BSON.@save path model cfg
    return path
end

function load_saved_model(path::AbstractString)
    contents = BSON.load(path)
    model = contents[:model]
    cfg = haskey(contents, :cfg) ? contents[:cfg] : nothing
    return model, cfg
end

function save_run_metadata(run_dir::AbstractString,
                           params::Dict{String,Any},
                           history::TrainingHistory,
                           kl_epochs::Vector{Int},
                           kl_history::Vector{Float64},
                           result::LangevinResult;
                           data_path::AbstractString,
                           total_samples::Int,
                           train_samples::Int,
                           subset_seed::Int,
                           model_seed::Int,
                           configuration_path::AbstractString,
                           training_performed::Bool,
                           training_plot::Union{Nothing,String},
                           comparison_plot::String,
                           model_repo_path::String,
                           run_model_path::String,
                           moment_matching::Union{Nothing,Dict}=nothing,
                           correlation_metadata::Union{Nothing,Dict}=nothing)
    run_info = Dict(
        "timestamp" => string(Dates.now()),
        "output_dir" => run_dir,
        "data_path" => data_path,
        "total_samples" => total_samples,
        "train_samples" => train_samples,
        "train_subset_seed" => subset_seed,
        "model_seed" => model_seed,
        "trained" => training_performed,
        "epochs" => length(history.epoch_losses),
        "final_loss" => isempty(history.epoch_losses) ? NaN : Float64(history.epoch_losses[end]),
        "kl_epochs" => kl_epochs,
        "kl_history" => kl_history,
        "final_kl" => result.kl_divergence,
        "parameters_file" => configuration_path,
        "model_repository_path" => model_repo_path,
        "run_model_path" => run_model_path,
        "comparison_plot" => comparison_plot,
    )
    training_plot !== nothing && (run_info["training_plot"] = training_plot)
    moment_matching !== nothing && (run_info["moment_matching"] = moment_matching)
    correlation_metadata !== nothing && (run_info["correlation"] = correlation_metadata)
    payload = Dict(
        "parameters" => params,
        "run" => run_info,
    )
    config_path = joinpath(run_dir, "run_config.toml")
    open(config_path, "w") do io
        TOML.print(io, payload)
    end
    return config_path
end

function save_training_plot(history::TrainingHistory,
                            kl_epochs::Vector{Int},
                            kl_values::Vector{Float64},
                            path::AbstractString)
    fig = Figure(size=(900, 720))
    ax_loss = Axis(fig[1, 1];
                   xlabel="Epoch",
                   ylabel="Loss",
                   title="Training loss vs epoch")
    epochs = collect(1:length(history.epoch_losses))
    losses = Float64.(history.epoch_losses)
    lines!(ax_loss, epochs, losses; color=:teal, linewidth=2)
    scatter!(ax_loss, epochs, losses; markersize=8, color=:black)

    ax_kl = Axis(fig[2, 1];
                 xlabel="Epoch",
                 ylabel="KL divergence",
                 title="Langevin KL vs epoch")
    if !isempty(kl_values)
        lines!(ax_kl, kl_epochs, kl_values; color=:purple, linewidth=2)
        scatter!(ax_kl, kl_epochs, kl_values; markersize=8, color=:black)
    end

    save(path, fig; px_per_unit=1)
    return path
end

function reshape_langevin_samples(result::LangevinResult, dataset::NormalizedDataset)
    L = sample_length(dataset)
    C = num_channels(dataset)
    dim, T, E = size(result.trajectory)
    dim == L * C || error("Langevin trajectory dimension $dim does not match dataset layout $(L*C)")
    reshaped = reshape(result.trajectory, L, C, T, E)
    return reshape(reshaped, L, C, T * E)
end

function pair_ranges(tensor::Array{Float32,3}, j::Int)
    L, C, B = size(tensor)
    L > j || error("Lag j=$j exceeds spatial length L=$L")
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf
    @inbounds for b in 1:B, c in 1:C, i in 1:(L - j)
        x = tensor[i, c, b]
        y = tensor[i + j, c, b]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    end
    return (Float64(xmin), Float64(xmax)), (Float64(ymin), Float64(ymax))
end

function ensure_bounds(bounds::Tuple{Float64,Float64})
    a, b = bounds
    if !isfinite(a) || !isfinite(b)
        return (-1.0, 1.0)
    elseif a == b
        δ = max(abs(a), 1.0) * 1e-3
        return (a - δ, b + δ)
    elseif a > b
        return (b, a)
    else
        return (a, b)
    end
end

edges_from_bounds(bounds::Tuple{Float64,Float64}, nbins::Int) =
    collect(range(bounds[1], bounds[2]; length=nbins + 1))

midpoints(edges::Vector{Float64}) = (edges[1:end-1] .+ edges[2:end]) ./ 2

function pair_samples_tensor(tensor::Array{Float32,3}, j::Int)
    L, C, B = size(tensor)
    n = (L - j) * C * B
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    idx = 1
    @inbounds for b in 1:B, c in 1:C, i in 1:(L - j)
        xs[idx] = Float64(tensor[i, c, b])
        ys[idx] = Float64(tensor[i + j, c, b])
        idx += 1
    end
    return xs, ys
end

function kde_heatmap_tensor(tensor::Array{Float32,3}, j::Int,
                            bounds::Tuple{Float64,Float64}, npoints::Int)
    xs, ys = pair_samples_tensor(tensor, j)
    xgrid = range(bounds[1], bounds[2]; length=npoints)
    ygrid = range(bounds[1], bounds[2]; length=npoints)
    kd = kde((xs, ys), (xgrid, ygrid))
    return kd.x, kd.y, kd.density
end

function compute_heatmap_specs(sim_tensor::Array{Float32,3},
                               obs_tensor::Array{Float32,3},
                               offsets::Tuple{Vararg{Int}};
                               nbins::Int=64,
                               bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    specs = NamedTuple[]
    for j in offsets
        local_bounds = bounds
        if local_bounds === nothing
            sim_x, sim_y = pair_ranges(sim_tensor, j)
            obs_x, obs_y = pair_ranges(obs_tensor, j)
            low = min(sim_x[1], obs_x[1])
            high = max(sim_x[2], obs_x[2])
            local_bounds = ensure_bounds((low, high))
        end
        sx, sy, sim_counts = kde_heatmap_tensor(sim_tensor, j, local_bounds, nbins)
        _, _, obs_counts = kde_heatmap_tensor(obs_tensor, j, local_bounds, nbins)
        push!(specs, (
            j=j,
            simulated=(sx, sy, sim_counts),
            observed=(sx, sy, obs_counts),
        ))
    end
    return specs
end

function save_comparison_figure(result::LangevinResult,
                                sim_tensor::Array{Float32,3},
                                obs_tensor::Array{Float32,3},
                                langevin_cfg::LangevinConfig,
                                path::AbstractString;
                                offsets::Tuple{Vararg{Int}}=DEFAULT_LAG_OFFSETS,
                                value_bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    specs = compute_heatmap_specs(sim_tensor, obs_tensor, offsets;
                                  bounds=value_bounds,
                                  nbins=HEATMAP_BINS)
    fig = Figure(size=(1400, 1800))
    mode_label = string(langevin_cfg.mode)
    pdf_axis = Axis(fig[1, 1];
                    xlabel="Value",
                    ylabel="PDF",
                    title=@sprintf("Averaged PDFs (mode=%s, KL=%.3e)", mode_label, result.kl_divergence))
    lines!(pdf_axis, result.bin_centers, result.observed_pdf;
           color=:navy, linewidth=2.0, label="Observed")
    lines!(pdf_axis, result.bin_centers, result.simulated_pdf;
           color=:firebrick, linewidth=2.0, label="Langevin")
    if value_bounds !== nothing
        xlims!(pdf_axis, value_bounds...)
    end
    axislegend(pdf_axis, position=:rb)

    acf_axis = Axis(fig[1, 2];
                    xlabel="Time",
                    ylabel="Average ACF",
                    title=@sprintf("Autocorrelation (τ_dec≈%.2f)", something(result.decorrelation_time, 0.0)))
    obs_time = result.observed_time
    sim_time = result.simulated_time
    if result.observed_acf !== nothing && obs_time !== nothing
        lines!(acf_axis, obs_time, result.observed_acf;
               color=:navy, linewidth=2.0, label="Observed ACF")
    end
    if result.simulated_acf !== nothing && sim_time !== nothing
        lines!(acf_axis, sim_time, result.simulated_acf;
               color=:firebrick, linewidth=2.0, label="Langevin ACF")
    end
    limit_time = nothing
    if obs_time !== nothing && sim_time !== nothing
        limit_time = min(obs_time[end], sim_time[end])
    elseif obs_time !== nothing
        limit_time = obs_time[end]
    elseif sim_time !== nothing
        limit_time = sim_time[end]
    end
    limit_time !== nothing && xlims!(acf_axis, 0, limit_time)
    axislegend(acf_axis, position=:rt)

    for (row, spec) in enumerate(specs)
        lx = Axis(fig[row + 1, 1];
                  xlabel="x[i]",
                  ylabel="x[i+$(spec.j)]",
                  title=@sprintf("Langevin: j = %d", spec.j))
        rx = Axis(fig[row + 1, 2];
                  xlabel="x[i]",
                  ylabel="x[i+$(spec.j)]",
                  title=@sprintf("Data: j = %d", spec.j))
        sx, sy, sim_counts = spec.simulated
        dx, dy, obs_counts = spec.observed
        heatmap!(lx, sx, sy, sim_counts; colormap=:plasma)
        heatmap!(rx, dx, dy, obs_counts; colormap=:plasma)
        if value_bounds !== nothing
            xlims!(lx, value_bounds...)
            ylims!(lx, value_bounds...)
            xlims!(rx, value_bounds...)
            ylims!(rx, value_bounds...)
        end
    end
    save(path, fig; px_per_unit=1)
    return path
end

function copy_parameters_file(run_dir::AbstractString, params_path::AbstractString)
    dest = joinpath(run_dir, basename(params_path))
    cp(params_path, dest; force=true)
    return dest
end

function run_training_and_monitor!(model,
                                   pdf_dataset::NormalizedDataset,
                                   train_data::NormalizedDataset,
                                   trainer_cfg::ScoreTrainerConfig,
                                   langevin_cfg::LangevinConfig,
                                   mm_cfg::MomentMatchingConfig,
                                   corr_info::CorrelationInfo,
                                   eval_interval::Int,
                                   verbose::Bool,
                                   mm_dataset::NormalizedDataset,
                                   phi_enabled::Bool)
    interval = max(eval_interval, 0)
    kl_epochs = Int[]
    kl_values = Float64[]
    last_result = Ref{Union{Nothing,LangevinResult}}(nothing)
    last_mm = Ref{Union{Nothing,DriftDiffusionEstimate}}(nothing)

    function evaluate!(epoch, current_model, label_suffix)
        if phi_enabled
            mm = timed("Moment matching ($label_suffix)", verbose) do
                compute_drift_diffusion(current_model, mm_dataset, trainer_cfg, mm_cfg, corr_info;
                                        apply_correction=true, verbose=verbose)
            end
            langevin_cfg.phi = mm.phi
            langevin_cfg.diffusion = mm.sigma
            last_mm[] = mm
        else
            langevin_cfg.phi = nothing
            langevin_cfg.diffusion = nothing
            last_mm[] = nothing
        end
        res = timed("Langevin integration ($label_suffix)", verbose) do
            run_langevin(current_model, pdf_dataset, langevin_cfg, corr_info)
        end
        push!(kl_epochs, epoch)
        push!(kl_values, res.kl_divergence)
        last_result[] = res
        @info "Epoch $(epoch) KL divergence" value=res.kl_divergence
    end

    epoch_callback = (epoch, m, epoch_time) -> begin
        verbose_log(verbose, "Epoch $epoch training completed"; seconds=epoch_time)
        if interval > 0 && epoch % interval == 0
            evaluate!(epoch, m, "epoch $epoch")
        end
    end

    history = timed("Training loop", verbose) do
        train!(model, train_data, trainer_cfg;
               epoch_callback=epoch_callback)
    end
    if isempty(kl_epochs) || kl_epochs[end] != trainer_cfg.epochs
        evaluate!(trainer_cfg.epochs, model, "epoch $(trainer_cfg.epochs)")
    end
    final_result = last_result[]
    return history, kl_epochs, kl_values, final_result, last_mm[]
end

function main()
    params = load_parameters()
    data_params = get(params, "data", Dict{String,Any}())
    model_params = get(params, "model", Dict{String,Any}())
    training_params = get(params, "training", Dict{String,Any}())
    langevin_params = get(params, "langevin", Dict{String,Any}())
    output_params = get(params, "output", Dict{String,Any}())
    run_params = get(params, "run", Dict{String,Any}())
    verbose = get(run_params, "verbose", false)

    data_path = resolve_path(get(data_params, "path", "data/new_ks.hdf5"))
    dataset_key = get(data_params, "dataset_key", nothing)
    dataset_key = dataset_key === "" ? nothing : dataset_key
    samples_orientation = symbol_from_string(get(data_params, "samples_orientation", "rows"))
    train_samples = Int(get(data_params, "train_samples", 0))
    subset_seed = Int(get(data_params, "subset_seed", 0))
    data_dt = Float64(get(data_params, "dt", 1.0))

    dataset = timed("Loading dataset", verbose) do
        load_hdf5_dataset(data_path;
                          dataset_key=dataset_key,
                          samples_orientation=samples_orientation)
    end
    @info "Loaded dataset" size=size(dataset.data)
    raw_min = Float64(minimum(dataset.data))
    raw_max = Float64(maximum(dataset.data))
    value_bounds = ensure_bounds((raw_min, raw_max))
    hr_dataset = dataset
    hr_data_path = data_path

    train_data = timed("Preparing training subset", verbose) do
        subset_dataset(dataset, train_samples; seed=subset_seed)
    end
    @info "Training subset" size=size(train_data.data) nmax=train_samples

    model_cfg, model_seed = build_model_config(model_params)
    trainer_cfg = build_trainer_config(training_params)
    langevin_cfg = build_langevin_config(langevin_params, trainer_cfg, data_dt)
    output_cfg = build_output_config(output_params)
    eval_interval = Int(get(training_params, "langevin_eval_interval", 1))
    mm_params = get(params, "moment_matching", Dict{String,Any}())
    phi_enabled = get(mm_params, "phi", true)
    mm_cfg = MomentMatchingConfig(
        dt = Float64(get(mm_params, "dt", data_dt)),
        max_samples = Int(get(mm_params, "max_samples", 4096)),
        stride = Int(get(mm_params, "stride", 10)),
        batch_size = Int(get(mm_params, "batch_size", 256)),
        min_eig = Float64(get(mm_params, "min_eig", 1e-6)),
        seed = Int(get(mm_params, "seed", 0)),
    )
    corr_params = get(params, "correlation", Dict{String,Any}())
    corr_cfg = CorrelationConfig(
        dt = Float64(get(corr_params, "dt", data_dt)),
        max_lag = Int(get(corr_params, "max_lag", 512)),
        stride = Int(get(corr_params, "stride", 1)),
        threshold = Float64(get(corr_params, "threshold", exp(-1))),
        multiple = Float64(get(corr_params, "multiple", 3.0)),
    )
    corr_info = timed("Computing data autocorrelation", verbose) do
        compute_correlation_info(dataset, corr_cfg)
    end

    model_repo_path = output_cfg.model_repository_path
    run_root = output_cfg.run_root
    run_model_filename = output_cfg.run_model_filename
    lag_offsets = output_cfg.lag_offsets

    run_dir = create_run_directory(run_root)
    params_copy = timed("Copying parameters file", verbose) do
        copy_parameters_file(run_dir, PARAMETERS_PATH)
    end

    model_exists = isfile(model_repo_path)
    if model_exists
        @info "Loading pretrained model" path=model_repo_path
        model, saved_cfg = load_saved_model(model_repo_path)
        saved_cfg isa ScoreUNetConfig && (model_cfg = saved_cfg)
    else
        model = instantiate_model(model_cfg, model_seed)
    end

    history = TrainingHistory(Float32[], Float32[])
    kl_epochs = Int[]
    kl_history = Float64[]
    training_performed = false
    result = nothing
    mm_result = nothing

    if model_exists
        if phi_enabled
            mm_result = timed("Moment matching (pretrained model)", verbose) do
                compute_drift_diffusion(model, hr_dataset, trainer_cfg, mm_cfg, corr_info;
                                        apply_correction=true, verbose=verbose)
            end
            langevin_cfg.phi = mm_result.phi
            langevin_cfg.diffusion = mm_result.sigma
        else
            langevin_cfg.phi = nothing
            langevin_cfg.diffusion = nothing
        end
        result = timed("Langevin integration (pretrained model)", verbose) do
            run_langevin(model, dataset, langevin_cfg, corr_info)
        end
    else
        training_performed = true
        history, kl_epochs, kl_history, result, mm_result = run_training_and_monitor!(
            model, dataset, train_data, trainer_cfg, langevin_cfg,
            mm_cfg, corr_info, eval_interval, verbose, dataset, phi_enabled)
        timed("Saving reusable model checkpoint", verbose) do
            save_model(model_repo_path, model, model_cfg)
        end
    end

    if phi_enabled && mm_result === nothing
        error("Moment matching estimation missing")
    end

    run_model_path = joinpath(run_dir, run_model_filename)
    timed("Saving run-specific model checkpoint", verbose) do
        save_model(run_model_path, model, model_cfg)
    end
    mm_path = nothing
    if phi_enabled && mm_result !== nothing
        mm_path = joinpath(run_dir, "moment_matching.bson")
        timed("Saving moment matching matrices", verbose) do
            BSON.@save mm_path phi=mm_result.phi sigma=mm_result.sigma samples=mm_result.samples
        end
    end

    training_plot = nothing
    if training_performed
        training_plot = timed("Saving training metrics figure", verbose) do
            save_training_plot(history, kl_epochs, kl_history,
                               joinpath(run_dir, "training_metrics.png"))
        end
    end

    sim_tensor = reshape_langevin_samples(result, dataset)
    comparison_plot = timed("Generating PDF comparison figure", verbose) do
        save_comparison_figure(result, sim_tensor, dataset.data, langevin_cfg,
                               joinpath(run_dir, "comparison.png");
                               offsets=lag_offsets,
                               value_bounds=value_bounds)
    end

    mm_metadata = phi_enabled && mm_result !== nothing ? Dict(
        "samples_used" => mm_result.samples,
        "dt" => mm_cfg.dt,
        "max_samples" => mm_cfg.max_samples,
        "stride" => mm_cfg.stride,
        "batch_size" => mm_cfg.batch_size,
        "min_eig" => mm_cfg.min_eig,
        "moment_matching_path" => mm_path,
        "moment_matching_dataset" => hr_data_path,
    ) : nothing
    corr_metadata = Dict(
        "decorrelation_time" => corr_info.decorrelation_time,
        "time_window" => corr_info.time[end],
        "stride" => corr_cfg.stride,
        "max_lag" => corr_cfg.max_lag,
        "threshold" => corr_cfg.threshold,
        "correlation_dataset" => hr_data_path,
    )

    config_path = timed("Writing run metadata", verbose) do
        save_run_metadata(run_dir, params, history, kl_epochs, kl_history, result;
                          data_path=data_path,
                          total_samples=length(dataset),
                          train_samples=length(train_data),
                          subset_seed=subset_seed,
                          model_seed=model_seed,
                          configuration_path=params_copy,
                          training_performed=training_performed,
                          training_plot=training_plot,
                          comparison_plot=comparison_plot,
                          model_repo_path=model_repo_path,
                          run_model_path=run_model_path,
                          moment_matching=mm_metadata,
                          correlation_metadata=corr_metadata)
    end

    @printf("KL divergence: %.6e\n", result.kl_divergence)
    @info "Run artifacts saved" dir=run_dir training_plot=training_plot comparison_plot=comparison_plot config=config_path model=run_model_path
    return result
end

result = main()
