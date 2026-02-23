module L96TrainStage

using Printf

function _append_log(log_path::AbstractString, msg::AbstractString)
    open(log_path, "a") do io
        println(io, msg)
    end
end

function _resolve_resume_state_path(params::Dict{String,Any}; base_dir::AbstractString=pwd())
    enabled = get(params, "train.resume.enabled", false)
    enabled || return ""

    state_path = strip(String(get(params, "train.resume.state_path", "")))
    if isempty(state_path)
        src_run = strip(String(get(params, "train.resume.source_run_dir", "")))
        isempty(src_run) && error("train.resume.enabled=true requires train.resume.source_run_dir or train.resume.state_path")
        state_path = joinpath(src_run, "model", "training_state_latest.bson")
    end
    if !isabspath(state_path)
        state_path = abspath(joinpath(base_dir, state_path))
    end
    isfile(state_path) || error("Resume state file not found: $state_path")
    return state_path
end

function _wait_file_stable(path::AbstractString; checks::Int=3, sleep_seconds::Float64=0.5)
    prev_size = -1
    stable = 0
    while stable < checks
        if isfile(path)
            sz = filesize(path)
            if sz > 0 && sz == prev_size
                stable += 1
            else
                stable = 0
            end
            prev_size = sz
        else
            stable = 0
            prev_size = -1
        end
        sleep(sleep_seconds)
    end
    return path
end

function _process_status_str(proc)
    parts = String[]
    hasfield(typeof(proc), :exitcode) && push!(parts, "exitcode=$(getfield(proc, :exitcode))")
    if hasfield(typeof(proc), :termsignal)
        sig = getfield(proc, :termsignal)
        sig != 0 && push!(parts, "termsignal=$(sig)")
    end
    if hasfield(typeof(proc), :running)
        push!(parts, "running=$(getfield(proc, :running))")
    end
    return isempty(parts) ? repr(proc) : join(parts, ", ")
end

function launch_training!(params::Dict{String,Any}, dirs::Dict{String,String}; base_dir::AbstractString=pwd())
    log_path = joinpath(dirs["logs"], "pipeline.log")

    data_path = abspath(joinpath(base_dir, params["data.path"]))
    model_path = joinpath(dirs["model"], "score_model.bson")
    train_state_path = joinpath(dirs["model"], "training_state_latest.bson")
    train_diag_dir = dirs["figures_training"]
    tmp_root = joinpath(dirs["logs"], "_tmp")
    checkpoint_dir = joinpath(tmp_root, "checkpoints")
    train_cfg_path = joinpath(tmp_root, "train_stage_config.toml")
    mkpath(tmp_root)

    resume_state_path = _resolve_resume_state_path(params; base_dir=base_dir)

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
    !isempty(resume_state_path) && _append_log(log_path, "[train] resume_state_path=$resume_state_path")

    env = copy(ENV)
    env["L96_RUN_DIR"] = dirs["run"]
    env["L96_DATA_PATH"] = data_path
    env["L96_DATASET_KEY"] = params["data.dataset_key"]

    env["L96_MODEL_PATH"] = model_path
    env["L96_TRAIN_STATE_PATH"] = train_state_path
    env["L96_TRAIN_DIAG_DIR"] = train_diag_dir
    env["L96_CHECKPOINT_DIR"] = checkpoint_dir
    env["L96_TRAIN_CONFIG_PATH"] = train_cfg_path
    env["L96_PIPELINE_MODE"] = "true"
    !isempty(resume_state_path) && (env["L96_RESUME_STATE_PATH"] = resume_state_path)

    env["L96_TRAIN_DEVICE"] = params["train.device"]
    env["L96_TRAIN_SEED"] = string(params["run.seed"])
    env["L96_BATCH_SIZE"] = string(params["train.batch_size"])
    env["L96_EPOCHS"] = string(params["train.num_training_epochs"])
    env["L96_LR"] = string(params["train.lr"])
    env["L96_TRAIN_NOISE_SIGMA"] = string(params["train.sigma"])
    env["L96_BASE_CHANNELS"] = string(params["train.base_channels"])
    env["L96_CHANNEL_MULTIPLIERS"] = join(string.(params["train.channel_multipliers"]), ",")
    env["L96_MODEL_ARCH"] = params["train.model_arch"]
    env["L96_NORMALIZATION_MODE"] = params["data.normalization_mode"]
    env["L96_PROGRESS"] = params["train.progress"] ? "true" : "false"
    env["L96_USE_LR_SCHEDULE"] = params["train.use_lr_schedule"] ? "true" : "false"
    env["L96_WARMUP_STEPS"] = string(params["train.warmup_steps"])
    env["L96_MIN_LR_FACTOR"] = string(params["train.min_lr_factor"])
    env["L96_NORM_TYPE"] = params["train.norm_type"]
    env["L96_NORM_GROUPS"] = string(params["train.norm_groups"])
    env["L96_EMA_ENABLED"] = params["train.ema.enabled"] ? "true" : "false"
    env["L96_EMA_DECAY"] = string(params["train.ema.decay"])
    env["L96_EMA_USE_FOR_EVAL"] = params["train.ema.use_for_eval"] ? "true" : "false"
    env["L96_LOSS_X_WEIGHT"] = string(params["train.loss.x_weight"])
    env["L96_LOSS_Y_WEIGHT"] = string(params["train.loss.y_weight"])
    env["L96_LOSS_MEAN_WEIGHT"] = string(params["train.loss.mean_weight"])
    env["L96_LOSS_COV_WEIGHT"] = string(params["train.loss.cov_weight"])

    # KL per-epoch eval is done by the pipeline itself using full Langevin outputs.
    env["L96_EVAL_KL_EVERY"] = "0"
    env["L96_SAVE_CHECKPOINT_EVERY"] = string(ckpt_every)

    cmd = setenv(`julia --project=. scripts/L96/lib/train_unet.jl`, env)
    io = open(log_path, "a")
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    _append_log(log_path, "[train] launched")

    return Dict{String,Any}(
        "process" => proc,
        "process_log_io" => io,
        "model_path" => model_path,
        "train_state_path" => train_state_path,
        "checkpoint_dir" => checkpoint_dir,
        "training_metrics_csv" => joinpath(train_diag_dir, "training_metrics.csv"),
        "train_config_path" => train_cfg_path,
        "tmp_root" => tmp_root,
        "eval_interval" => eval_every,
        "resume_state_path" => resume_state_path,
        "log_path" => log_path,
    )
end

function wait_training!(training::Dict{String,Any})
    proc = training["process"]
    io = training["process_log_io"]
    log_path = String(training["log_path"])

    wait(proc)
    close(io)
    proc_status = _process_status_str(proc)
    if !success(proc)
        _append_log(log_path, "[train] failed status=$proc_status")
        error("Training subprocess failed (status: $proc_status). See log: $log_path")
    end

    model_path = String(training["model_path"])
    isfile(model_path) || error("Training failed: model not found at $model_path")
    metrics_csv = String(training["training_metrics_csv"])
    isfile(metrics_csv) || error("Training failed: training metrics CSV not found at $metrics_csv")
    train_state_path = String(training["train_state_path"])
    isfile(train_state_path) || error("Training failed: state file not found at $train_state_path")

    _append_log(log_path, "[train] done status=$proc_status")
    return training
end

function run_training!(params::Dict{String,Any}, dirs::Dict{String,String}; base_dir::AbstractString=pwd())
    training = launch_training!(params, dirs; base_dir=base_dir)
    wait_training!(training)
    return Dict{String,Any}(
        "model_path" => training["model_path"],
        "train_state_path" => training["train_state_path"],
        "checkpoint_dir" => training["checkpoint_dir"],
        "training_metrics_csv" => training["training_metrics_csv"],
        "train_config_path" => training["train_config_path"],
        "tmp_root" => training["tmp_root"],
        "eval_interval" => training["eval_interval"],
        "resume_state_path" => training["resume_state_path"],
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

function wait_for_checkpoint!(training::Dict{String,Any}, epoch::Int; poll_seconds::Float64=2.0)
    checkpoint_dir = String(training["checkpoint_dir"])
    proc = training["process"]

    while true
        for (ep, path) in list_checkpoints(checkpoint_dir)
            ep == epoch || continue
            _wait_file_stable(path)
            return path
        end

        if Base.process_exited(proc)
            wait(proc)
            for (ep, path) in list_checkpoints(checkpoint_dir)
                ep == epoch || continue
                _wait_file_stable(path)
                return path
            end
            error("Training process ended before checkpoint epoch=$epoch was produced.")
        end

        sleep(poll_seconds)
    end
end

end # module
