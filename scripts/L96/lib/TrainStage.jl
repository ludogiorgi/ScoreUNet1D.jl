module L96TrainStage

using Printf

function _append_log(log_path::AbstractString, msg::AbstractString)
    open(log_path, "a") do io
        println(io, msg)
    end
end

function _run_logged(cmd::Cmd, log_path::AbstractString)
    run(pipeline(pipeline(cmd; stderr=stdout), `tee -a $log_path`))
end

function run_training!(params::Dict{String,Any}, dirs::Dict{String,String}; base_dir::AbstractString=pwd())
    log_path = joinpath(dirs["logs"], "pipeline.log")

    data_path = abspath(joinpath(base_dir, params["data.path"]))
    model_path = joinpath(dirs["model"], "score_model.bson")
    train_diag_dir = dirs["figures_training"]
    tmp_root = joinpath(dirs["logs"], "_tmp")
    checkpoint_dir = joinpath(tmp_root, "checkpoints")
    train_cfg_path = joinpath(tmp_root, "train_stage_config.toml")
    mkpath(tmp_root)

    eval_every = params["train.kl_eval.enabled"] ? params["train.kl_eval.kl_eval_interval_epochs"] : 0
    ckpt_every = if eval_every > 0
        # Langevin is evaluated from checkpoints; enforce interval consistency.
        eval_every
    elseif params["output.save_checkpoints"]
        params["output.checkpoint_every"]
    else
        0
    end

    _append_log(log_path, "[train] start")
    _append_log(log_path, "[train] data_path=$data_path")
    _append_log(log_path, "[train] model_path=$model_path")

    env = copy(ENV)
    env["L96_RUN_DIR"] = dirs["run"]
    env["L96_DATA_PATH"] = data_path
    env["L96_DATASET_KEY"] = params["data.dataset_key"]

    env["L96_MODEL_PATH"] = model_path
    env["L96_TRAIN_DIAG_DIR"] = train_diag_dir
    env["L96_CHECKPOINT_DIR"] = checkpoint_dir
    env["L96_TRAIN_CONFIG_PATH"] = train_cfg_path
    env["L96_PIPELINE_MODE"] = "true"

    env["L96_TRAIN_DEVICE"] = params["train.device"]
    env["L96_TRAIN_SEED"] = string(params["run.seed"])
    env["L96_BATCH_SIZE"] = string(params["train.batch_size"])
    env["L96_EPOCHS"] = string(params["train.num_training_epochs"])
    env["L96_LR"] = string(params["train.lr"])
    env["L96_TRAIN_NOISE_SIGMA"] = string(params["train.sigma"])
    env["L96_BASE_CHANNELS"] = string(params["train.base_channels"])
    env["L96_CHANNEL_MULTIPLIERS"] = join(string.(params["train.channel_multipliers"]), ",")
    env["L96_NORMALIZATION_MODE"] = params["data.normalization_mode"]
    env["L96_PROGRESS"] = params["train.progress"] ? "true" : "false"

    # KL per-epoch eval is done by the pipeline itself using full Langevin outputs.
    env["L96_EVAL_KL_EVERY"] = "0"
    env["L96_SAVE_CHECKPOINT_EVERY"] = string(ckpt_every)

    cmd = setenv(`julia --project=. scripts/L96/lib/train_unet.jl`, env)
    _run_logged(cmd, log_path)

    isfile(model_path) || error("Training failed: model not found at $model_path")

    metrics_csv = joinpath(train_diag_dir, "training_metrics.csv")
    isfile(metrics_csv) || error("Training failed: training metrics CSV not found at $metrics_csv")

    _append_log(log_path, "[train] done")
    return Dict{String,Any}(
        "model_path" => model_path,
        "checkpoint_dir" => checkpoint_dir,
        "training_metrics_csv" => metrics_csv,
        "train_config_path" => train_cfg_path,
        "tmp_root" => tmp_root,
        "eval_interval" => eval_every,
    )
end

function list_checkpoints(checkpoint_dir::AbstractString)
    if !isdir(checkpoint_dir)
        return Tuple{Int,String}[]
    end

    out = Tuple{Int,String}[]
    for name in readdir(checkpoint_dir)
        m = match(r"^epoch_(\d+)\.bson$", name)
        m === nothing && continue
        epoch = parse(Int, m.captures[1])
        push!(out, (epoch, joinpath(checkpoint_dir, name)))
    end
    sort!(out; by=x -> x[1])
    return out
end

end # module
