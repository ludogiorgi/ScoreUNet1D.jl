# Standard command (from repository root):
# julia --project=. scripts/arnold/train_unet.jl --params scripts/arnold/parameters_train.toml
# Nohup command:
# nohup julia --project=. scripts/arnold/train_unet.jl --params scripts/arnold/parameters_train.toml > scripts/arnold/nohup_train_unet.log 2>&1 &

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using Flux
using Functors
using HDF5
using Printf
using Random
using ScoreUNet1D
using Statistics
using TOML

include(joinpath(@__DIR__, "lib", "ArnoldCommon.jl"))
using .ArnoldCommon

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_train.toml")
    run_dir = ""
    pipeline_mode = false

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = args[i + 1]
            i += 2
        elseif a == "--run-dir"
            i == length(args) && error("--run-dir expects a value")
            run_dir = args[i + 1]
            i += 2
        elseif a == "--pipeline-mode"
            i == length(args) && error("--pipeline-mode expects true/false")
            pipeline_mode = parse_bool(args[i + 1])
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return (params_path=abspath(params_path), run_dir=run_dir, pipeline_mode=pipeline_mode)
end

function require_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

function as_int(tbl::Dict{String,Any}, key::String, default)
    return Int(get(tbl, key, default))
end

function as_float(tbl::Dict{String,Any}, key::String, default)
    return Float64(get(tbl, key, default))
end

function as_string(tbl::Dict{String,Any}, key::String, default)
    return String(get(tbl, key, default))
end

function as_bool(tbl::Dict{String,Any}, key::String, default)
    return parse_bool(get(tbl, key, default))
end

function as_int_vec(tbl::Dict{String,Any}, key::String, default)
    return Int.(collect(get(tbl, key, default)))
end

function load_config(path::AbstractString)
    isfile(path) || error("Train parameter file not found: $path")
    doc = TOML.parsefile(path)

    train = require_table(doc, "train")
    paths = require_table(doc, "paths")
    data = haskey(doc, "data") ? Dict{String,Any}(doc["data"]) : Dict{String,Any}()

    loss_weight = if haskey(train, "loss_weight")
        as_float(train, "loss_weight", 1.0)
    else
        as_float(train, "loss_x_weight", 1.0)
    end

    cfg = Dict{String,Any}(
        "paths.data_params" => abspath(as_string(paths, "data_params", "scripts/arnold/parameters_data.toml")),
        "data.dataset_role" => as_string(data, "dataset_role", "train_stochastic"),
        # populated from central data params after loading it in main()
        "data.dataset_path" => "",
        "data.dataset_key" => "",
        "data.dt" => NaN,
        "data.save_every" => 1,
        "data.target_spacing" => as_float(data, "target_spacing", 1.0),

        "train.device" => as_string(train, "device", "GPU:0"),
        "train.seed" => as_int(train, "seed", 42),
        "train.epochs" => as_int(train, "epochs", 120),
        "train.batch_size" => as_int(train, "batch_size", 512),
        "train.lr" => as_float(train, "lr", 8e-4),
        "train.sigma" => Float32(as_float(train, "sigma", 0.05)),
        "train.base_channels" => as_int(train, "base_channels", 64),
        "train.channel_multipliers" => as_int_vec(train, "channel_multipliers", [1, 2, 4]),
        "train.norm_type" => lowercase(as_string(train, "norm_type", "group")),
        "train.norm_groups" => as_int(train, "norm_groups", 8),
        "train.progress" => as_bool(train, "progress", false),
        "train.use_lr_schedule" => as_bool(train, "use_lr_schedule", true),
        "train.warmup_steps" => as_int(train, "warmup_steps", 500),
        "train.min_lr_factor" => as_float(train, "min_lr_factor", 0.1),
        "train.ema_enabled" => as_bool(train, "ema_enabled", true),
        "train.ema_decay" => Float32(as_float(train, "ema_decay", 0.999)),
        "train.ema_use_for_eval" => as_bool(train, "ema_use_for_eval", true),
        "train.checkpoint_every" => as_int(train, "checkpoint_every", 10),
        "train.save_state_every" => as_int(train, "save_state_every", 10),
        "train.resume_enabled" => as_bool(train, "resume_enabled", false),
        "train.resume_state_path" => as_string(train, "resume_state_path", ""),
        "train.normalization_mode" => lowercase(as_string(train, "normalization_mode", "per_channel")),
        "train.loss_weight" => Float32(loss_weight),
        "train.loss_mean_weight" => Float32(as_float(train, "loss_mean_weight", 0.0)),
        "train.loss_cov_weight" => Float32(as_float(train, "loss_cov_weight", 0.0)),
    )

    cfg["data.dataset_role"] in ArnoldCommon.ARNOLD_DATASET_ROLES || error("data.dataset_role must be one of $(join(ArnoldCommon.ARNOLD_DATASET_ROLES, ", "))")
    cfg["train.epochs"] >= 1 || error("train.epochs must be >= 1")
    cfg["train.batch_size"] >= 1 || error("train.batch_size must be >= 1")
    cfg["train.norm_type"] in ("batch", "group") || error("train.norm_type must be batch or group")

    return cfg, doc
end

function ensure_train_dataset!(cfg::Dict{String,Any}, data_cfg::Dict{String,Any})
    role = cfg["data.dataset_role"]
    info = ensure_arnold_dataset_role!(data_cfg, role)
    cfg["data.dataset_path"] = info["path"]
    cfg["data.dataset_key"] = info["key"]
    cfg["data.dt"] = data_cfg["twoscale.dt"]
    cfg["data.save_every"] = data_cfg["datasets.$role.save_every"]
    if haskey(data_cfg, "datasets.$role.target_spacing")
        cfg["data.target_spacing"] = max(cfg["data.target_spacing"], data_cfg["datasets.$role.target_spacing"])
    end
    return info["path"]
end

function load_train_tensor(cfg::Dict{String,Any})
    path = cfg["data.dataset_path"]
    key = cfg["data.dataset_key"]
    raw = h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        Float32.(read(h5[key]))
    end

    N, K = size(raw)
    spacing = dataset_time_spacing(cfg["data.dt"], cfg["data.save_every"])
    target_spacing = cfg["data.target_spacing"]
    stride = max(1, Int(round(target_spacing / max(spacing, eps(Float64)))))
    if stride > 1
        raw = raw[1:stride:end, :]
        @info "Applied additional decorrelation stride to training snapshots" stride = stride spacing_original = spacing spacing_effective = spacing * stride samples_after = size(raw, 1)
    else
        @info "Using raw training snapshots spacing" spacing = spacing samples = N
    end

    Neff = size(raw, 1)
    tensor = permutedims(reshape(raw, Neff, 1, K), (3, 2, 1))
    return Array{Float32,3}(tensor)
end

function normalize_tensor(tensor::Array{Float32,3}, mode::AbstractString)
    mode == "per_channel" || mode == "split_xy" || error("Unsupported normalization mode '$mode'")

    K, C, _ = size(tensor)
    C == 1 || error("Arnold train script expects one-channel X tensor, got C=$C")

    mu = Float32(mean(tensor))
    sd = Float32(std(tensor) + eps(Float32))

    mean_mat = fill(mu, C, K)
    std_mat = fill(sd, C, K)
    stats = DataStats(mean_mat, std_mat)

    mean_lc = permutedims(mean_mat, (2, 1))
    std_lc = permutedims(std_mat, (2, 1))
    normalized = (tensor .- reshape(mean_lc, K, C, 1)) ./ reshape(std_lc, K, C, 1)
    return Array{Float32,3}(normalized), stats
end

function parse_norm_type(raw::AbstractString)
    s = lowercase(strip(raw))
    if s == "batch"
        return :batch
    elseif s == "group"
        return :group
    end
    error("Unsupported norm type '$raw'")
end

function parse_channel_multipliers(v)
    vals = Int.(collect(v))
    isempty(vals) && error("channel_multipliers cannot be empty")
    all(>(0), vals) || error("channel_multipliers must be positive")
    return vals
end

function tree_to_cpu(tree)
    return Functors.fmap(tree) do x
        x isa AbstractArray ? Array(x) : x
    end
end

function tree_to_device(tree, device::ExecutionDevice)
    if device isa ScoreUNet1D.GPUDevice
        return Functors.fmap(tree) do x
            x isa AbstractArray ? move_array(Array(x), device) : x
        end
    end
    return tree_to_cpu(tree)
end

function maybe_resume_payload(path::AbstractString)
    p = strip(path)
    isempty(p) && return nothing
    isfile(p) || error("Resume state file not found: $p")
    payload = BSON.load(p)
    haskey(payload, :epoch) || error("Resume state missing :epoch")
    return payload
end

function resume_model(payload)
    haskey(payload, :model_raw) && return payload[:model_raw]
    haskey(payload, :model_raw_cpu) && return payload[:model_raw_cpu]
    haskey(payload, :model) && return payload[:model]
    error("Resume payload missing model")
end

function resume_opt_state(payload)
    haskey(payload, :opt_state) && return payload[:opt_state]
    haskey(payload, :opt_state_cpu) && return payload[:opt_state_cpu]
    error("Resume payload missing optimizer state")
end

function save_training_state(path::AbstractString;
    epoch::Int,
    global_step::Int,
    model_raw_cpu,
    ema_model_cpu,
    opt_state_cpu,
    rng::MersenneTwister,
    thread_rngs::Vector{MersenneTwister},
    cfg,
    trainer_cfg,
    stats::DataStats,
    epoch_times::Vector{Float64},
    history_epoch_losses::Vector{Float32},
    history_batch_losses::Vector{Float32})
    mkpath(dirname(path))
    BSON.@save path epoch global_step model_raw_cpu ema_model_cpu opt_state_cpu rng thread_rngs cfg trainer_cfg stats epoch_times history_epoch_losses history_batch_losses
    return path
end

function write_training_metrics(path::AbstractString,
    epoch_losses::Vector{Float64},
    epoch_times::Vector{Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# epoch,epoch_loss,epoch_time_sec")
        for i in eachindex(epoch_losses)
            t = i <= length(epoch_times) ? epoch_times[i] : NaN
            println(io, "$(i),$(epoch_losses[i]),$(t)")
        end
    end
    return path
end

function resolve_paths(cfg::Dict{String,Any}, run_dir::AbstractString)
    if !isempty(strip(run_dir))
        run_abs = abspath(run_dir)
        model_dir = ArnoldCommon.ensure_dir(joinpath(run_abs, "model"))
        fig_dir = ArnoldCommon.ensure_dir(joinpath(run_abs, "figures", "training"))
        checkpoint_dir = ArnoldCommon.ensure_dir(joinpath(model_dir, "checkpoints"))
        return Dict{String,String}(
            "run_dir" => run_abs,
            "model_path" => joinpath(model_dir, "score_model.bson"),
            "raw_model_path" => joinpath(model_dir, "score_model_raw.bson"),
            "state_path" => joinpath(model_dir, "training_state_latest.bson"),
            "checkpoint_dir" => checkpoint_dir,
            "metrics_csv" => joinpath(fig_dir, "training_metrics.csv"),
        )
    end

    # Standalone defaults from train params [paths]
    root = dirname(dirname(cfg["paths.data_params"]))
    run_default = ArnoldCommon.ensure_dir(joinpath(root, "standalone_train"))
    model_dir = ArnoldCommon.ensure_dir(joinpath(run_default, "model"))
    fig_dir = ArnoldCommon.ensure_dir(joinpath(run_default, "figures"))
    checkpoint_dir = ArnoldCommon.ensure_dir(joinpath(model_dir, "checkpoints"))
    return Dict{String,String}(
        "run_dir" => run_default,
        "model_path" => joinpath(model_dir, "score_model.bson"),
        "raw_model_path" => joinpath(model_dir, "score_model_raw.bson"),
        "state_path" => joinpath(model_dir, "training_state_latest.bson"),
        "checkpoint_dir" => checkpoint_dir,
        "metrics_csv" => joinpath(fig_dir, "training_metrics.csv"),
    )
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg, raw_doc = load_config(parsed.params_path)
    data_cfg, _ = load_data_config(cfg["paths.data_params"])

    paths = resolve_paths(cfg, parsed.run_dir)
    checkpoint_every = cfg["train.checkpoint_every"]
    save_state_every = cfg["train.save_state_every"]

    ensure_train_dataset!(cfg, data_cfg)
    tensor = load_train_tensor(cfg)
    K, C, B = size(tensor)
    @info "Loaded Arnold train tensor" size = size(tensor) path = cfg["data.dataset_path"]

    tensor_norm, stats = normalize_tensor(tensor, cfg["train.normalization_mode"])
    dataset = NormalizedDataset(tensor_norm, stats)

    multipliers = parse_channel_multipliers(cfg["train.channel_multipliers"])
    norm_type = parse_norm_type(cfg["train.norm_type"])

    model_cfg = ScoreUNetConfig(
        in_channels=C,
        out_channels=C,
        base_channels=cfg["train.base_channels"],
        channel_multipliers=multipliers,
        kernel_size=5,
        periodic=true,
        norm_type=norm_type,
        norm_groups=cfg["train.norm_groups"],
    )

    resume_path = ""
    if cfg["train.resume_enabled"]
        resume_path = cfg["train.resume_state_path"]
        isempty(strip(resume_path)) && (resume_path = paths["state_path"])
        resume_path = abspath(resume_path)
    end
    resume_payload = maybe_resume_payload(resume_path)

    Random.seed!(cfg["train.seed"])
    model = build_unet(model_cfg)

    resume_epoch = 0
    resume_global_step = 0
    resume_rng = MersenneTwister(cfg["train.seed"])
    resume_thread_rngs = ScoreUNet1D.seed_thread_rngs(cfg["train.seed"])
    history_epoch_seed = Float32[]
    history_batch_seed = Float32[]
    epoch_times = Float64[]

    if resume_payload !== nothing
        model = resume_model(resume_payload)
        resume_epoch = Int(resume_payload[:epoch])
        resume_global_step = Int(get(resume_payload, :global_step, 0))
        resume_rng = haskey(resume_payload, :rng) ? copy(resume_payload[:rng]) : MersenneTwister(cfg["train.seed"] + resume_epoch)
        resume_thread_rngs = haskey(resume_payload, :thread_rngs) ? [copy(r) for r in resume_payload[:thread_rngs]] : ScoreUNet1D.seed_thread_rngs(cfg["train.seed"] + resume_epoch)
        history_epoch_seed = haskey(resume_payload, :history_epoch_losses) ? Float32.(resume_payload[:history_epoch_losses]) : Float32[]
        history_batch_seed = haskey(resume_payload, :history_batch_losses) ? Float32.(resume_payload[:history_batch_losses]) : Float32[]
        epoch_times = haskey(resume_payload, :epoch_times) ? Float64.(resume_payload[:epoch_times]) : Float64[]
        @info "Resuming Arnold training" state = resume_path resume_epoch = resume_epoch
    end

    trainer_cfg = ScoreTrainerConfig(
        batch_size=cfg["train.batch_size"],
        epochs=cfg["train.epochs"],
        lr=cfg["train.lr"],
        sigma=cfg["train.sigma"],
        seed=cfg["train.seed"],
        progress=false,
        use_lr_schedule=cfg["train.use_lr_schedule"],
        warmup_steps=cfg["train.warmup_steps"],
        min_lr_factor=cfg["train.min_lr_factor"],
        x_loss_weight=cfg["train.loss_weight"],
        y_loss_weight=cfg["train.loss_weight"],
        mean_match_weight=cfg["train.loss_mean_weight"],
        cov_match_weight=cfg["train.loss_cov_weight"],
    )

    device = try
        select_device(cfg["train.device"])
    catch err
        @warn "Training device unavailable; falling back to CPU" requested = cfg["train.device"] error = sprint(showerror, err)
        ScoreUNet1D.CPUDevice()
    end
    activate_device!(device)
    model = move_model(model, device)

    steps_per_epoch = cld(B, cfg["train.batch_size"])
    ema_decay_epoch = Float32(cfg["train.ema_decay"] ^ steps_per_epoch)
    ema_model_cpu = if cfg["train.ema_enabled"]
        if resume_payload !== nothing && haskey(resume_payload, :ema_model_cpu)
            resume_payload[:ema_model_cpu]
        else
            move_model(model, ScoreUNet1D.CPUDevice())
        end
    else
        nothing
    end
    ema_model_cpu !== nothing && Flux.testmode!(ema_model_cpu)

    initial_state = if resume_payload === nothing
        nothing
    else
        opt_state_resume = tree_to_device(resume_opt_state(resume_payload), device)
        TrainingState(
            epoch=resume_epoch,
            global_step=resume_global_step,
            opt_state=opt_state_resume,
            epoch_losses=copy(history_epoch_seed),
            batch_losses=copy(history_batch_seed),
            rng=copy(resume_rng),
            thread_rngs=[copy(r) for r in resume_thread_rngs],
        )
    end

    latest_model_cpu = nothing
    latest_opt_state_cpu = resume_payload === nothing ? nothing : resume_opt_state(resume_payload)
    latest_global_step = resume_global_step
    latest_rng = copy(resume_rng)
    latest_thread_rngs = [copy(r) for r in resume_thread_rngs]

    epoch_callback = function(epoch::Int, model_epoch, epoch_time::Real)
        push!(epoch_times, Float64(epoch_time))
        Flux.testmode!(model_epoch)
        model_cpu_epoch = move_model(model_epoch, ScoreUNet1D.CPUDevice())
        Flux.testmode!(model_cpu_epoch)
        latest_model_cpu = model_cpu_epoch

        if ema_model_cpu !== nothing
            Functors.fmap(ema_model_cpu, model_cpu_epoch) do a, b
                if a isa AbstractArray && b isa AbstractArray
                    @. a = ema_decay_epoch * a + Float32(1 - ema_decay_epoch) * b
                end
                return a
            end
            Flux.testmode!(ema_model_cpu)
        end

        if checkpoint_every > 0 && epoch % checkpoint_every == 0
            ckpt_path = joinpath(paths["checkpoint_dir"], @sprintf("epoch_%04d.bson", epoch))
            if cfg["train.ema_enabled"] && cfg["train.ema_use_for_eval"] && ema_model_cpu !== nothing
                BSON.@save ckpt_path model = ema_model_cpu cfg = model_cfg trainer_cfg stats epoch
            else
                BSON.@save ckpt_path model = model_cpu_epoch cfg = model_cfg trainer_cfg stats epoch
            end
            @info "Saved Arnold checkpoint" epoch = epoch path = ckpt_path
        end

        Flux.trainmode!(model_epoch)
        return nothing
    end

    state_callback = function(state::TrainingState)
        latest_global_step = Int(state.global_step)
        latest_rng = copy(state.rng)
        latest_thread_rngs = [copy(r) for r in state.thread_rngs]
        latest_opt_state_cpu = tree_to_cpu(state.opt_state)

        persist_now = (state.epoch == cfg["train.epochs"]) || (save_state_every > 0 && state.epoch % save_state_every == 0)
        persist_now || return nothing

        model_raw_cpu = latest_model_cpu === nothing ? move_model(model, ScoreUNet1D.CPUDevice()) : latest_model_cpu
        save_training_state(
            paths["state_path"];
            epoch=Int(state.epoch),
            global_step=Int(state.global_step),
            model_raw_cpu=model_raw_cpu,
            ema_model_cpu=ema_model_cpu,
            opt_state_cpu=latest_opt_state_cpu,
            rng=latest_rng,
            thread_rngs=latest_thread_rngs,
            cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            stats=stats,
            epoch_times=epoch_times,
            history_epoch_losses=state.epoch_losses,
            history_batch_losses=state.batch_losses,
        )
        return nothing
    end

    @info "Starting Arnold UNet training" device = cfg["train.device"] epochs = cfg["train.epochs"] batch_size = cfg["train.batch_size"] sigma = cfg["train.sigma"]
    history = train!(
        model,
        dataset,
        trainer_cfg;
        device=device,
        epoch_callback=epoch_callback,
        state_callback=state_callback,
        initial_state=initial_state,
    )

    model_cpu_raw = is_gpu(device) ? move_model(model, ScoreUNet1D.CPUDevice()) : model
    Flux.testmode!(model_cpu_raw)
    model_cpu = if cfg["train.ema_enabled"] && cfg["train.ema_use_for_eval"] && ema_model_cpu !== nothing
        ema_model_cpu
    else
        model_cpu_raw
    end
    Flux.testmode!(model_cpu)

    latest_opt_state_cpu === nothing && (latest_opt_state_cpu = tree_to_cpu(Flux.setup(Flux.Optimisers.Adam(trainer_cfg.lr), model)))

    save_training_state(
        paths["state_path"];
        epoch=length(history.epoch_losses),
        global_step=latest_global_step,
        model_raw_cpu=model_cpu_raw,
        ema_model_cpu=ema_model_cpu,
        opt_state_cpu=latest_opt_state_cpu,
        rng=latest_rng,
        thread_rngs=latest_thread_rngs,
        cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        stats=stats,
        epoch_times=epoch_times,
        history_epoch_losses=history.epoch_losses,
        history_batch_losses=history.batch_losses,
    )

    epoch_losses_f64 = Float64.(history.epoch_losses)
    write_training_metrics(paths["metrics_csv"], epoch_losses_f64, epoch_times)

    if cfg["train.ema_enabled"] && cfg["train.ema_use_for_eval"] && ema_model_cpu !== nothing
        BSON.@save paths["raw_model_path"] model = model_cpu_raw cfg = model_cfg trainer_cfg stats history epoch_times training_metrics_path = paths["metrics_csv"]
    end
    BSON.@save paths["model_path"] model = model_cpu cfg = model_cfg trainer_cfg stats history epoch_times training_metrics_path = paths["metrics_csv"]

    final_loss = isempty(history.epoch_losses) ? NaN : Float64(history.epoch_losses[end])
    @info "Arnold UNet training complete" model = paths["model_path"] state = paths["state_path"] final_loss = final_loss epochs = length(history.epoch_losses)

    if parsed.pipeline_mode && !isempty(strip(parsed.run_dir))
        params_copy = joinpath(parsed.run_dir, "params", "parameters_train_used.toml")
        mkpath(dirname(params_copy))
        open(params_copy, "w") do io
            TOML.print(io, raw_doc)
        end
    end

    println("model_path=$(paths["model_path"])")
    println("state_path=$(paths["state_path"])")
    println("checkpoint_dir=$(paths["checkpoint_dir"])")
    println("training_metrics=$(paths["metrics_csv"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
