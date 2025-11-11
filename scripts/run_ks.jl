#!/usr/bin/env julia

using CairoMakie
using Dates
using Flux
using Printf
using Random
using ScoreUNet1D
using ScoreUNet1D: NormalizedDataset, sample_length, num_channels
using TOML

CairoMakie.activate!()

const DEFAULT_RUN_ROOT = "runs"
const LAG_OFFSETS = (1, 2, 3)

"""
    subset_dataset(dataset, nmax; seed=0)

Returns a dataset with at most `nmax` samples (drawn without replacement) to keep
training manageable while preserving the global normalization statistics.
"""
function subset_dataset(dataset::NormalizedDataset, nmax::Int; seed::Int=0)
    total = length(dataset)
    total <= nmax && return dataset
    rng = MersenneTwister(seed)
    idxs = collect(1:total)
    Random.shuffle!(rng, idxs)
    idxs = idxs[1:nmax]
    return NormalizedDataset(dataset.data[:, :, idxs], dataset.stats)
end

function configure_model()
    init_seed = parse(Int, get(ENV, "KS_MODEL_SEED", "314159"))
    Random.seed!(init_seed)
    cfg = ScoreUNetConfig(
        in_channels=1,
        base_channels=24,
        channel_multipliers=[1, 2, 4],
        kernel_size=5,
        periodic=true,
        activation=Flux.swish,
        final_activation=identity,
    )
    model = build_unet(cfg)
    Random.seed!()  # re-seed global RNG from system entropy
    return cfg, model, init_seed
end

function configure_trainer()
    cfg = ScoreTrainerConfig(
        batch_size=128,
        epochs=5,
        lr=1e-3,
        sigma=0.05f0,
        shuffle=true,
        progress=false,
        max_steps_per_epoch=250,
    )
    return cfg
end

function configure_langevin(trainer_cfg::ScoreTrainerConfig)
    cfg = LangevinConfig(
        dt=5e-3,
        nsteps=9_000,
        resolution=20,
        n_ensembles=96,
        burn_in=2_500,
        nbins=384,
        sigma=trainer_cfg.sigma,
        mode=:all,
    )
    return cfg
end

function create_run_directory(root::AbstractString=DEFAULT_RUN_ROOT)
    mkpath(root)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    slug = Random.randstring(4)
    run_dir = joinpath(root, "run_$(timestamp)_$(slug)")
    mkpath(run_dir)
    return run_dir
end

model_config_dict(cfg::ScoreUNetConfig, init_seed::Int) = Dict(
    "in_channels" => cfg.in_channels,
    "base_channels" => cfg.base_channels,
    "channel_multipliers" => cfg.channel_multipliers,
    "kernel_size" => cfg.kernel_size,
    "periodic" => cfg.periodic,
    "activation" => string(cfg.activation),
    "final_activation" => string(cfg.final_activation),
    "init_seed" => init_seed,
)

function trainer_config_dict(cfg::ScoreTrainerConfig)
    dict = Dict(
        "batch_size" => cfg.batch_size,
        "epochs" => cfg.epochs,
        "lr" => cfg.lr,
        "sigma" => Float64(cfg.sigma),
        "shuffle" => cfg.shuffle,
        "seed" => cfg.seed,
        "progress" => cfg.progress,
    )
    cfg.max_steps_per_epoch !== nothing && (dict["max_steps_per_epoch"] = cfg.max_steps_per_epoch)
    return dict
end

langevin_config_dict(cfg::LangevinConfig) = Dict(
    "dt" => cfg.dt,
    "nsteps" => cfg.nsteps,
    "resolution" => cfg.resolution,
    "n_ensembles" => cfg.n_ensembles,
    "burn_in" => cfg.burn_in,
    "nbins" => cfg.nbins,
    "sigma" => Float64(cfg.sigma),
    "seed" => cfg.seed,
    "mode" => string(cfg.mode),
)

function save_run_metadata(run_dir::AbstractString,
                           model_cfg::ScoreUNetConfig,
                           model_seed::Int,
                           trainer_cfg::ScoreTrainerConfig,
                           langevin_cfg::LangevinConfig;
                           data_path::AbstractString,
                           total_samples::Int,
                           train_samples::Int,
                           subset_seed::Int,
                           history::TrainingHistory,
                           result::LangevinResult)
    payload = Dict(
        "run" => Dict(
            "timestamp" => string(Dates.now()),
            "output_dir" => run_dir,
            "data_path" => data_path,
            "total_samples" => total_samples,
            "train_samples" => train_samples,
            "train_subset_seed" => subset_seed,
            "epochs" => length(history.epoch_losses),
            "final_loss" => isempty(history.epoch_losses) ? NaN : Float64(history.epoch_losses[end]),
            "kl_divergence" => result.kl_divergence,
        ),
        "model" => model_config_dict(model_cfg, model_seed),
        "trainer" => trainer_config_dict(trainer_cfg),
        "langevin" => langevin_config_dict(langevin_cfg),
    )
    config_path = joinpath(run_dir, "run_config.toml")
    open(config_path, "w") do io
        TOML.print(io, payload)
    end
    return config_path
end

function save_training_plot(history::TrainingHistory, path::AbstractString)
    fig = Figure(size=(900, 360))
    ax = Axis(fig[1, 1];
              xlabel="Epoch",
              ylabel="Loss",
              title="Training loss vs epoch")
    epochs = collect(1:length(history.epoch_losses))
    losses = Float64.(history.epoch_losses)
    lines!(ax, epochs, losses; color=:teal, linewidth=2)
    scatter!(ax, epochs, losses; markersize=8, color=:black)
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

function histogram2d_from_tensor(tensor::Array{Float32,3}, j::Int,
                                 x_edges::Vector{Float64}, y_edges::Vector{Float64})
    L, C, B = size(tensor)
    nx = length(x_edges) - 1
    ny = length(y_edges) - 1
    counts = zeros(Float64, nx, ny)
    @inbounds for b in 1:B, c in 1:C, i in 1:(L - j)
        x = tensor[i, c, b]
        y = tensor[i + j, c, b]
        xi = clamp(searchsortedlast(x_edges, x), 1, nx)
        yi = clamp(searchsortedlast(y_edges, y), 1, ny)
        counts[xi, yi] += 1
    end
    total = sum(counts)
    total > 0 && (counts ./= total)
    return midpoints(x_edges), midpoints(y_edges), counts
end

function compute_heatmap_specs(sim_tensor::Array{Float32,3},
                               obs_tensor::Array{Float32,3},
                               offsets::Tuple{Vararg{Int}};
                               nbins::Int=64)
    specs = NamedTuple[]
    for j in offsets
        sim_x, sim_y = pair_ranges(sim_tensor, j)
        obs_x, obs_y = pair_ranges(obs_tensor, j)
        bounds_x = ensure_bounds((min(sim_x[1], obs_x[1]), max(sim_x[2], obs_x[2])))
        bounds_y = ensure_bounds((min(sim_y[1], obs_y[1]), max(sim_y[2], obs_y[2])))
        x_edges = edges_from_bounds(bounds_x, nbins)
        y_edges = edges_from_bounds(bounds_y, nbins)
        push!(specs, (
            j=j,
            simulated=histogram2d_from_tensor(sim_tensor, j, x_edges, y_edges),
            observed=histogram2d_from_tensor(obs_tensor, j, x_edges, y_edges),
        ))
    end
    return specs
end

function save_pdf_diagnostic(result::LangevinResult,
                             sim_tensor::Array{Float32,3},
                             obs_tensor::Array{Float32,3},
                             langevin_cfg::LangevinConfig,
                             path::AbstractString;
                             offsets::Tuple{Vararg{Int}}=LAG_OFFSETS)
    specs = compute_heatmap_specs(sim_tensor, obs_tensor, offsets)
    fig = Figure(size=(1200, 1600))
    mode_label = string(langevin_cfg.mode)
    top_axis = Axis(fig[1, 1:2];
                    xlabel="Value",
                    ylabel="PDF",
                    title=@sprintf("Averaged PDFs (mode=%s, KL=%.3e)", mode_label, result.kl_divergence))
    lines!(top_axis, result.bin_centers, result.observed_pdf;
           color=:navy, linewidth=2.0, label="Observed")
    lines!(top_axis, result.bin_centers, result.simulated_pdf;
           color=:firebrick, linewidth=2.0, label="Langevin")
    axislegend(top_axis, position=:rb)

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
    end
    save(path, fig; px_per_unit=1)
    return path
end

function main()
    data_path = get(ENV, "KS_DATA_PATH", "data/new_ks.hdf5")
    dataset = load_hdf5_dataset(data_path;
                                dataset_key="timeseries",
                                samples_orientation=:columns)
    @info "Loaded dataset" size=size(dataset.data)

    nmax = parse(Int, get(ENV, "KS_TRAIN_SAMPLES", "100000"))
    subset_seed = 2025
    train_data = subset_dataset(dataset, nmax; seed=subset_seed)
    @info "Training subset" size=size(train_data.data) nmax=nmax

    model_cfg, model, model_seed = configure_model()
    trainer_cfg = configure_trainer()
    history = train!(model, train_data, trainer_cfg)
    @info "Training complete" epochs=length(history.epoch_losses) last_loss=history.epoch_losses[end]

    langevin_cfg = configure_langevin(trainer_cfg)
    result = run_langevin(model, dataset, langevin_cfg)
    @printf("KL divergence: %.6e\n", result.kl_divergence)

    run_root = get(ENV, "KS_RUN_ROOT", DEFAULT_RUN_ROOT)
    run_dir = create_run_directory(run_root)
    training_plot = save_training_plot(history, joinpath(run_dir, "training_loss.png"))
    sim_tensor = reshape_langevin_samples(result, dataset)
    pdf_plot = save_pdf_diagnostic(result, sim_tensor, dataset.data, langevin_cfg,
                                   joinpath(run_dir, "pdf_comparison.png"))
    config_path = save_run_metadata(run_dir, model_cfg, model_seed, trainer_cfg, langevin_cfg;
                                    data_path=data_path,
                                    total_samples=length(dataset),
                                    train_samples=length(train_data),
                                    subset_seed=subset_seed,
                                    history=history,
                                    result=result)

    @info "Run artifacts saved" dir=run_dir training_plot=training_plot pdf_plot=pdf_plot config=config_path
    return result
end

result = main()
