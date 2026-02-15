"""
    TrainRunner

Generic training wrapper for score networks. Can be used for any dynamical system.
"""
module TrainRunner

using BSON
using Dates
using Flux
using LinearAlgebra
using Random
using TOML

using ..ScoreUNet1D: ScoreUNetConfig, ScoreTrainerConfig, TrainingHistory,
    NormalizedDataset, build_unet, train!, load_hdf5_dataset, sample_length,
    ExecutionDevice, CPUDevice, GPUDevice, select_device, move_model, is_gpu,
    gpu_count, activate_device!

using ..RunnerUtils: load_config, resolve_path, ensure_dir, activation_from_string,
    symbol_from_string, timed, verbose_log

export TrainingResult, train_score_network

#─────────────────────────────────────────────────────────────────────────────
# Result Struct
#─────────────────────────────────────────────────────────────────────────────

"""
    TrainingResult

Result of training a score network.

# Fields
- `model`: Trained Flux model (on CPU)
- `model_config`: ScoreUNetConfig used
- `trainer_config`: ScoreTrainerConfig used
- `history`: TrainingHistory with loss curves
- `model_path`: Path where model was saved
- `dataset_size`: Number of samples in dataset
- `train_size`: Number of samples used for training
"""
struct TrainingResult
    model::Any
    model_config::ScoreUNetConfig
    trainer_config::ScoreTrainerConfig
    history::TrainingHistory
    model_path::String
    dataset_size::Int
    train_size::Int
end

#─────────────────────────────────────────────────────────────────────────────
# Helper Functions
#─────────────────────────────────────────────────────────────────────────────

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

function build_model_config(params::Dict{String,Any})
    activation = activation_from_string(get(params, "activation", "swish"))
    final_act = activation_from_string(get(params, "final_activation", "identity"))
    in_channels = Int(get(params, "in_channels", 1))
    out_channels = Int(get(params, "out_channels", in_channels))
    cfg = ScoreUNetConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=Int(get(params, "base_channels", 32)),
        channel_multipliers=Int.(get(params, "channel_multipliers", [1, 2, 4])),
        kernel_size=Int(get(params, "kernel_size", 5)),
        periodic=get(params, "periodic", false),
        activation=activation,
        final_activation=final_act,
    )
    init_seed = Int(get(params, "init_seed", 314159))
    return cfg, init_seed
end

function build_trainer_config(params::Dict{String,Any})
    max_steps = get(params, "max_steps_per_epoch", nothing)
    max_steps = max_steps === nothing ? nothing : Int(max_steps)
    cfg = ScoreTrainerConfig(
        batch_size=Int(get(params, "batch_size", 32)),
        epochs=Int(get(params, "epochs", 10)),
        lr=Float64(get(params, "lr", 1e-3)),
        sigma=Float32(get(params, "sigma", 0.05)),
        shuffle=get(params, "shuffle", true),
        seed=Int(get(params, "seed", 42)),
        progress=get(params, "progress", true),
        max_steps_per_epoch=max_steps,
        accumulation_steps=Int(get(params, "accumulation_steps", 1)),
        use_lr_schedule=get(params, "use_lr_schedule", false),
        warmup_steps=Int(get(params, "warmup_steps", 500)),
        min_lr_factor=Float64(get(params, "min_lr_factor", 0.1)),
        epoch_subset_size=Int(get(params, "epoch_subset_size", 0)),
    )
    return cfg
end

function instantiate_model(cfg::ScoreUNetConfig, seed::Int)
    Random.seed!(seed)
    model = build_unet(cfg)
    Random.seed!()
    return model
end

function adjust_config_for_data(model_config::ScoreUNetConfig, L::Int)
    max_levels = max(1, floor(Int, log2(max(L, 2))))
    if length(model_config.channel_multipliers) > max_levels
        @warn "Reducing UNet depth" requested = length(model_config.channel_multipliers) max = max_levels
        return ScoreUNetConfig(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            base_channels=model_config.base_channels,
            channel_multipliers=model_config.channel_multipliers[1:max_levels],
            kernel_size=model_config.kernel_size,
            periodic=model_config.periodic,
            activation=model_config.activation,
            final_activation=model_config.final_activation,
        )
    end
    return model_config
end

#─────────────────────────────────────────────────────────────────────────────
# Main Function
#─────────────────────────────────────────────────────────────────────────────

"""
    train_score_network(config_path; model=nothing, project_root=nothing) -> TrainingResult

Train a score network using configuration from a TOML file.

# Arguments
- `config_path`: Path to TOML configuration file
- `model=nothing`: Optional pretrained model to continue training
- `project_root=nothing`: Project root for resolving relative paths (defaults to parent of config dir)

# Returns
- `TrainingResult` with trained model and metadata

# Config Sections
- `[data]`: path, dataset_key, samples_orientation, train_samples, subset_seed
- `[model]`: in_channels, base_channels, channel_multipliers, kernel_size, etc.
- `[training]`: epochs, batch_size, lr, sigma, etc.
- `[output]`: model_path
- `[run]`: device, verbose
"""
function train_score_network(config_path::AbstractString;
    model=nothing,
    project_root::Union{Nothing,AbstractString}=nothing)

    # Determine project root
    if project_root === nothing
        project_root = dirname(dirname(abspath(config_path)))
    end

    @info "Loading training configuration" config = config_path
    config = load_config(config_path)

    # Extract sections
    data_cfg = get(config, "data", Dict{String,Any}())
    model_cfg = get(config, "model", Dict{String,Any}())
    train_cfg = get(config, "training", Dict{String,Any}())
    output_cfg = get(config, "output", Dict{String,Any}())
    run_cfg = get(config, "run", Dict{String,Any}())

    verbose = get(run_cfg, "verbose", true)

    # Device setup
    device_name = uppercase(String(get(run_cfg, "device", "CPU")))
    device = select_device(device_name)
    activate_device!(device)
    @info "Device configured" device = device_name gpus = is_gpu(device) ? gpu_count(device) : 0

    # Load dataset
    data_path = resolve_path(get(data_cfg, "path", "data/new_ks.hdf5"), project_root)
    dataset_key = get(data_cfg, "dataset_key", "timeseries")
    samples_orientation = symbol_from_string(get(data_cfg, "samples_orientation", "columns"))

    dataset = timed("Loading dataset", verbose) do
        load_hdf5_dataset(data_path;
            dataset_key=dataset_key === "" ? nothing : dataset_key,
            samples_orientation=samples_orientation)
    end
    @info "Dataset loaded" size = size(dataset.data) path = data_path

    # Note: Per-epoch subset sampling is now handled by epoch_subset_size in [training]
    # The train_samples/subset_seed in [data] are deprecated (kept for backwards compatibility)
    train_data = dataset

    # Build model config
    model_config, model_seed = build_model_config(model_cfg)
    L = sample_length(dataset)
    model_config = adjust_config_for_data(model_config, L)

    # Model path
    model_path = resolve_path(get(output_cfg, "model_path", "runs/trained_model.bson"), project_root)
    force_retrain = get(train_cfg, "force_retrain", true)

    # Get or create model
    if model !== nothing
        @info "Continuing training with provided model"
        current_model = model
    elseif !force_retrain && isfile(model_path)
        @info "Loading pretrained model" path = model_path
        contents = BSON.load(model_path)
        current_model = contents[:model]
        haskey(contents, :cfg) && (model_config = contents[:cfg])
    else
        @info "Creating new model" seed = model_seed
        current_model = instantiate_model(model_config, model_seed)
    end

    current_model = move_model(current_model, device)

    # Training
    trainer_config = build_trainer_config(train_cfg)

    @info "Starting training" epochs = trainer_config.epochs batch_size = trainer_config.batch_size sigma = trainer_config.sigma

    history = timed("Training", verbose) do
        train!(current_model, train_data, trainer_config; device=device)
    end

    @info "Training complete" final_loss = history.epoch_losses[end] epochs = length(history.epoch_losses)

    # Save model
    model_cpu = is_gpu(device) ? move_model(current_model, CPUDevice()) : current_model
    ensure_dir(dirname(model_path))

    timed("Saving model", verbose) do
        BSON.@save model_path model = model_cpu cfg = model_config trainer_cfg = trainer_config
    end

    @info "Model saved" path = model_path

    return TrainingResult(
        model_cpu,
        model_config,
        trainer_config,
        history,
        model_path,
        length(dataset),
        length(train_data)
    )
end

end # module
