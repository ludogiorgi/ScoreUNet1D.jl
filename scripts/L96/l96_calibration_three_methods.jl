#!/usr/bin/env julia

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

include("l96_parameter_jacobians.jl")

using HDF5
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics
using TOML

const CAL_DEFAULT_OUTPUT_DIR = "scripts/L96/calibration_outputs/run_021_three_method_calibration"

const CAL_TARGET_NSAMPLES = 300_000
const CAL_TARGET_START_INDEX = 50_001

const CAL_EVAL_NSAMPLES = 100_000
const CAL_EVAL_BURN = 4_000
const CAL_TRIAL_NSAMPLES = 40_000
const CAL_TRIAL_BURN = 2_000

const CAL_MAX_ITERS_DEFAULT = 4
const CAL_GFDT_TMAX = 0.0
const CAL_OBJ_IMPROVE_EPS = 1e-5
const CAL_LM_GROWTH = 10.0
const CAL_LM_MAX_TRIES = 2
const CAL_LS_MAX_TRIES = 4

const CAL_PARAM_LOWER = [6.0, 0.3, 6.0, 6.0]
const CAL_PARAM_UPPER = [14.0, 2.0, 14.0, 14.0]
const CAL_PARAM_NAMES = ["F", "h", "c", "b"]

const CAL_METHOD_ORDER = [:fd, :gfdt_unet, :gfdt_gaussian]
const CAL_METHOD_LABEL = Dict(
    :fd => "FD",
    :gfdt_unet => "GFDT+UNet",
    :gfdt_gaussian => "GFDT+Gaussian",
)
const CAL_METHOD_COLOR = Dict(
    :fd => :dodgerblue3,
    :gfdt_unet => :orangered3,
    :gfdt_gaussian => :darkgreen,
)

Base.@kwdef struct MethodConfig
    damping::Float64
    lambda_reg::Float64
    step_cap::Vector{Float64}
    seed_offset::Int
end

const CAL_METHOD_CONFIG = Dict(
    :fd => MethodConfig(
        damping=1.0,
        lambda_reg=1e-2,
        step_cap=[0.8, 0.15, 0.8, 0.8],
        seed_offset=100_000,
    ),
    :gfdt_unet => MethodConfig(
        damping=0.3,
        lambda_reg=5.0,
        step_cap=[0.35, 0.07, 0.35, 0.35],
        seed_offset=200_000,
    ),
    :gfdt_gaussian => MethodConfig(
        damping=0.3,
        lambda_reg=5.0,
        step_cap=[0.35, 0.07, 0.35, 0.35],
        seed_offset=300_000,
    ),
)

Base.@kwdef struct ScoreTrainConfig
    device::String = "GPU:0"
    dataset_key::String = "timeseries"
    normalization_mode::String = "split_xy"
    batch_size::Int = 128
    epochs::Int = 20
    lr::Float64 = 8e-4
    sigma::Float32 = 0.1f0
    base_channels::Int = 32
    channel_multipliers::Vector{Int} = [1, 2]
    progress::Bool = false
    use_lr_schedule::Bool = true
    warmup_steps::Int = 500
    min_lr_factor::Float64 = 0.1
    norm_type::String = "batch"
    norm_groups::Int = 0
    ema_enabled::Bool = true
    ema_decay::Float32 = 0.999f0
    ema_use_for_eval::Bool = true
    x_loss_weight::Float32 = 1.0f0
    y_loss_weight::Float32 = 1.0f0
end

function with_overrides(cfg::ScoreTrainConfig; epochs::Union{Nothing,Int}=nothing, device::Union{Nothing,String}=nothing)
    return ScoreTrainConfig(
        device=(device === nothing ? cfg.device : device),
        dataset_key=cfg.dataset_key,
        normalization_mode=cfg.normalization_mode,
        batch_size=cfg.batch_size,
        epochs=(epochs === nothing ? cfg.epochs : epochs),
        lr=cfg.lr,
        sigma=cfg.sigma,
        base_channels=cfg.base_channels,
        channel_multipliers=copy(cfg.channel_multipliers),
        progress=cfg.progress,
        use_lr_schedule=cfg.use_lr_schedule,
        warmup_steps=cfg.warmup_steps,
        min_lr_factor=cfg.min_lr_factor,
        norm_type=cfg.norm_type,
        norm_groups=cfg.norm_groups,
        ema_enabled=cfg.ema_enabled,
        ema_decay=cfg.ema_decay,
        ema_use_for_eval=cfg.ema_use_for_eval,
        x_loss_weight=cfg.x_loss_weight,
        y_loss_weight=cfg.y_loss_weight,
    )
end

function parse_cli(args::Vector{String})
    out = Dict{String,Any}(
        "run_dir" => DEFAULT_RUN_DIR,
        "integration_toml" => DEFAULT_OBS_INTEGRATION_TOML,
        "output_dir" => CAL_DEFAULT_OUTPUT_DIR,
        "max_iters" => CAL_MAX_ITERS_DEFAULT,
        "init_scale" => 0.8,
        "score_train_epochs" => nothing,
        "score_train_device" => nothing,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--run-dir"
            i == length(args) && error("--run-dir expects a value")
            out["run_dir"] = args[i + 1]
            i += 2
        elseif a == "--integration-toml"
            i == length(args) && error("--integration-toml expects a value")
            out["integration_toml"] = args[i + 1]
            i += 2
        elseif a == "--output-dir"
            i == length(args) && error("--output-dir expects a value")
            out["output_dir"] = args[i + 1]
            i += 2
        elseif a == "--max-iters"
            i == length(args) && error("--max-iters expects a value")
            out["max_iters"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--init-scale"
            i == length(args) && error("--init-scale expects a value")
            out["init_scale"] = parse(Float64, args[i + 1])
            i += 2
        elseif a == "--score-train-epochs"
            i == length(args) && error("--score-train-epochs expects a value")
            out["score_train_epochs"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--score-train-device"
            i == length(args) && error("--score-train-device expects a value")
            out["score_train_device"] = args[i + 1]
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return out
end

function get_nested(d::Dict{String,Any}, keys::Vector{String}, default)
    cur = d
    for k in keys
        if !(cur isa AbstractDict) || !haskey(cur, k)
            return default
        end
        cur = cur[k]
    end
    return cur
end

function parse_int_vec(v, default::Vector{Int})
    if v isa AbstractVector
        out = Int[]
        for e in v
            push!(out, Int(e))
        end
        return isempty(out) ? copy(default) : out
    end
    return copy(default)
end

function load_score_train_config(run_dir::AbstractString, cli::Dict{String,Any})
    summary_path = joinpath(run_dir, "metrics", "run_summary.toml")
    if !isfile(summary_path)
        cfg = ScoreTrainConfig()
        if cli["score_train_epochs"] !== nothing
            cfg = with_overrides(cfg; epochs=Int(cli["score_train_epochs"]))
        end
        if cli["score_train_device"] !== nothing
            cfg = with_overrides(cfg; device=String(cli["score_train_device"]))
        end
        return cfg
    end

    summary = TOML.parsefile(summary_path)
    raw_train = get_nested(summary, ["raw_parameters", "train"], Dict{String,Any}())
    raw_data = get_nested(summary, ["raw_parameters", "data"], Dict{String,Any}())
    rep_opt = get_nested(summary, ["reproducibility", "optimizer"], Dict{String,Any}())
    rep_model = get_nested(summary, ["reproducibility", "model"], Dict{String,Any}())
    rep_ema = get_nested(summary, ["reproducibility", "ema"], Dict{String,Any}())

    epochs_default = Int(get(rep_opt, "epochs", get(raw_train, "num_training_epochs", 20)))
    device_default = String(get(raw_train, "device", "GPU:0"))
    dataset_key = String(get(raw_data, "dataset_key", "timeseries"))
    normalization_mode = String(get(raw_data, "normalization_mode", get(rep_model, "normalization_mode", "split_xy")))

    cfg = ScoreTrainConfig(
        device=device_default,
        dataset_key=dataset_key,
        normalization_mode=normalization_mode,
        batch_size=Int(get(rep_opt, "batch_size", get(raw_train, "batch_size", 128))),
        epochs=epochs_default,
        lr=Float64(get(rep_opt, "learning_rate", get(raw_train, "lr", 8e-4))),
        sigma=Float32(get(rep_opt, "sigma", get(raw_train, "sigma", 0.1))),
        base_channels=Int(get(rep_model, "base_channels", get(raw_train, "base_channels", 32))),
        channel_multipliers=parse_int_vec(get(rep_model, "channel_multipliers", get(raw_train, "channel_multipliers", [1, 2])), [1, 2]),
        progress=Bool(get(raw_train, "progress", false)),
        use_lr_schedule=Bool(get(rep_opt, "use_lr_schedule", get(raw_train, "use_lr_schedule", true))),
        warmup_steps=Int(get(rep_opt, "warmup_steps", get(raw_train, "warmup_steps", 500))),
        min_lr_factor=Float64(get(rep_opt, "min_lr_factor", get(raw_train, "min_lr_factor", 0.1))),
        norm_type=String(get(rep_model, "norm_type", get(raw_train, "norm_type", "batch"))),
        norm_groups=Int(get(rep_model, "norm_groups", get(raw_train, "norm_groups", 0))),
        ema_enabled=Bool(get(rep_ema, "enabled", get(get(raw_train, "ema", Dict{String,Any}()), "enabled", true))),
        ema_decay=Float32(get(rep_ema, "decay", get(get(raw_train, "ema", Dict{String,Any}()), "decay", 0.999))),
        ema_use_for_eval=Bool(get(rep_ema, "use_for_eval", get(get(raw_train, "ema", Dict{String,Any}()), "use_for_eval", true))),
        x_loss_weight=Float32(get(rep_opt, "loss_x_weight", get(get(raw_train, "loss", Dict{String,Any}()), "x_weight", 1.0))),
        y_loss_weight=Float32(get(rep_opt, "loss_y_weight", get(get(raw_train, "loss", Dict{String,Any}()), "y_weight", 1.0))),
    )

    if cli["score_train_epochs"] !== nothing
        cfg = with_overrides(cfg; epochs=Int(cli["score_train_epochs"]))
    end
    if cli["score_train_device"] !== nothing
        cfg = with_overrides(cfg; device=String(cli["score_train_device"]))
    end
    return cfg
end

function tensor_snapshot_to_xy(tensor::Array{Float64,3}, J::Int)
    K, C, N = size(tensor)
    C == J + 1 || error("channel mismatch in tensor_snapshot_to_xy")
    N >= 1 || error("tensor must contain at least one snapshot")
    x0 = copy(vec(@view tensor[:, 1, 1]))
    y0 = Matrix{Float64}(undef, J, K)
    @inbounds for k in 1:K, j in 1:J
        y0[j, k] = tensor[k, j + 1, 1]
    end
    return x0, y0
end

function simulate_l96_tensor(θ::NTuple{4,Float64},
                             x0_init::Vector{Float64},
                             y0_init::Matrix{Float64},
                             cfg::L96Config;
                             nsamples::Int,
                             burn_snapshots::Int,
                             rng_seed::Int)
    K = cfg.K
    J = cfg.J
    x = copy(x0_init)
    y = copy(y0_init)
    ws = make_l96_workspace(K, J)
    rng = MersenneTwister(rng_seed)
    C = J + 1
    tensor = Array{Float64}(undef, K, C, nsamples)

    total = burn_snapshots + nsamples
    save_every = cfg.save_every
    @inbounds for n in 1:total
        for _ in 1:save_every
            rk4_step_l96!(x, y, cfg.dt, ws, θ)
            add_process_noise!(x, y, rng, cfg.process_noise_sigma, cfg.dt)
        end
        if n > burn_snapshots
            s = n - burn_snapshots
            tensor[:, 1, s] .= x
            for k in 1:K, j in 1:J
                tensor[k, j + 1, s] = y[j, k]
            end
        end
    end
    return tensor
end

function aggregate_observable_series(A::Array{Float64,2}, K::Int)
    m, N = size(A)
    if m == 5
        return A
    end
    m == 5 * K || error("aggregate_observable_series expects 5 or 5K rows")
    out = zeros(Float64, 5, N)
    @inbounds for t in 1:5
        for k in 1:K
            out[t, :] .+= @view A[5 * (k - 1) + t, :]
        end
        out[t, :] ./= K
    end
    return out
end

function aggregate_jacobian_rows(S::Matrix{Float64}, K::Int)
    m, p = size(S)
    if m == 5
        return copy(S)
    end
    m == 5 * K || error("aggregate_jacobian_rows expects 5 or 5K rows")
    out = zeros(Float64, 5, p)
    @inbounds for t in 1:5
        for k in 1:K
            out[t, :] .+= @view S[5 * (k - 1) + t, :]
        end
        out[t, :] ./= K
    end
    return out
end

function compute_target_stats(cfg::L96Config)
    tensor = load_observation_subset(cfg; nsamples=CAL_TARGET_NSAMPLES, start_index=CAL_TARGET_START_INDEX)
    A = compute_observables(tensor)
    Aagg = aggregate_observable_series(A, cfg.K)
    target_mean = vec(mean(Aagg; dims=2))
    target_std = vec(std(Aagg; dims=2))
    floor_std = max(quantile(target_std, 0.2), 1e-8)
    wdiag = 1.0 ./ (target_std .^ 2 .+ floor_std ^ 2)
    return target_mean, wdiag
end

@inline function weighted_objective(resid::Vector{Float64}, wdiag::Vector{Float64})
    return sqrt(mean((sqrt.(wdiag) .* resid) .^ 2))
end

function solve_weighted_step(S::Matrix{Float64},
                             resid::Vector{Float64},
                             wdiag::Vector{Float64};
                             lambda::Float64)
    sqrtw = sqrt.(wdiag)
    Sw = S .* reshape(sqrtw, :, 1)
    rw = resid .* sqrtw
    M = Sw' * Sw
    @inbounds for i in 1:size(M, 1)
        M[i, i] += lambda
    end
    rhs = Sw' * rw
    Δ = M \ rhs
    c = try
        cond(M)
    catch
        NaN
    end
    return Δ, c
end

function bounded_update(θ::Vector{Float64},
                        Δ::Vector{Float64};
                        damping::Float64,
                        step_cap::Vector{Float64})
    δ = -damping .* Δ
    δ = clamp.(δ, -step_cap, step_cap)
    θnew = θ .+ δ
    θnew = clamp.(θnew, CAL_PARAM_LOWER, CAL_PARAM_UPPER)
    return θnew
end

function evaluate_state_and_stats(θ::Vector{Float64},
                                  x0_init::Vector{Float64},
                                  y0_init::Matrix{Float64},
                                  cfg::L96Config;
                                  nsamples::Int,
                                  burn_snapshots::Int,
                                  rng_seed::Int)
    θt = (θ[1], θ[2], θ[3], θ[4])
    tensor = simulate_l96_tensor(θt, x0_init, y0_init, cfg;
                                 nsamples=nsamples,
                                 burn_snapshots=burn_snapshots,
                                 rng_seed=rng_seed)
    A = compute_observables(tensor)
    Aagg = aggregate_observable_series(A, cfg.K)
    G = vec(mean(Aagg; dims=2))
    return tensor, A, G
end

function write_tensor_hdf5(path::AbstractString,
                           tensor::Array{Float64,3},
                           cfg::L96Config;
                           dataset_key::AbstractString)
    mkpath(dirname(path))
    raw = permutedims(Float32.(tensor), (3, 2, 1))  # (N, C, K)
    h5open(path, "w") do h5
        ds = create_dataset(h5, dataset_key, datatype(Float32), dataspace(size(raw)))
        write(ds, raw)
        attrs = attributes(ds)
        attrs["K"] = Int(cfg.K)
        attrs["J"] = Int(cfg.J)
        attrs["F"] = Float64(cfg.F)
        attrs["h"] = Float64(cfg.h)
        attrs["c"] = Float64(cfg.c)
        attrs["b"] = Float64(cfg.b)
        attrs["dt"] = Float64(cfg.dt)
        attrs["save_every"] = Int(cfg.save_every)
        attrs["process_noise_sigma"] = Float64(cfg.process_noise_sigma)
    end
    return path
end

function read_last_train_loss(metrics_path::AbstractString)
    isfile(metrics_path) || return NaN
    last = NaN
    for ln in eachline(metrics_path)
        s = strip(ln)
        isempty(s) && continue
        startswith(s, "#") && continue
        parts = split(s, ",")
        length(parts) >= 2 || continue
        ep = tryparse(Int, parts[1])
        lv = tryparse(Float64, parts[2])
        if ep !== nothing && lv !== nothing
            last = lv
        end
    end
    return last
end

function train_unet_on_tensor(tensor::Array{Float64,3},
                              cfg::L96Config,
                              train_cfg::ScoreTrainConfig,
                              out_dir::AbstractString;
                              method::Symbol,
                              iter::Int,
                              seed::Int)
    iter_dir = joinpath(out_dir, String(method), @sprintf("iter_%02d", iter))
    data_dir = joinpath(iter_dir, "data")
    model_dir = joinpath(iter_dir, "model")
    diag_dir = joinpath(iter_dir, "diagnostics")
    ckpt_dir = joinpath(iter_dir, "checkpoints")
    logs_dir = joinpath(iter_dir, "logs")
    mkpath(data_dir)
    mkpath(model_dir)
    mkpath(diag_dir)
    mkpath(ckpt_dir)
    mkpath(logs_dir)

    data_path = write_tensor_hdf5(joinpath(data_dir, "timeseries.hdf5"), tensor, cfg; dataset_key=train_cfg.dataset_key)
    model_path = joinpath(model_dir, "score_model.bson")
    train_log_path = joinpath(logs_dir, "train_unet.log")

    env = copy(ENV)
    env["L96_PIPELINE_MODE"] = "true"
    env["L96_RUN_DIR"] = iter_dir
    env["L96_DATA_PATH"] = data_path
    env["L96_DATASET_KEY"] = train_cfg.dataset_key
    env["L96_MODEL_PATH"] = model_path
    env["L96_TRAIN_DIAG_DIR"] = diag_dir
    env["L96_CHECKPOINT_DIR"] = ckpt_dir
    env["L96_TRAIN_CONFIG_PATH"] = ""

    env["L96_TRAIN_DEVICE"] = train_cfg.device
    env["L96_TRAIN_SEED"] = string(seed)
    env["L96_BATCH_SIZE"] = string(train_cfg.batch_size)
    env["L96_EPOCHS"] = string(train_cfg.epochs)
    env["L96_LR"] = string(train_cfg.lr)
    env["L96_TRAIN_NOISE_SIGMA"] = string(train_cfg.sigma)
    env["L96_BASE_CHANNELS"] = string(train_cfg.base_channels)
    env["L96_CHANNEL_MULTIPLIERS"] = join(string.(train_cfg.channel_multipliers), ",")
    env["L96_NORMALIZATION_MODE"] = train_cfg.normalization_mode
    env["L96_PROGRESS"] = train_cfg.progress ? "true" : "false"
    env["L96_USE_LR_SCHEDULE"] = train_cfg.use_lr_schedule ? "true" : "false"
    env["L96_WARMUP_STEPS"] = string(train_cfg.warmup_steps)
    env["L96_MIN_LR_FACTOR"] = string(train_cfg.min_lr_factor)
    env["L96_NORM_TYPE"] = train_cfg.norm_type
    env["L96_NORM_GROUPS"] = string(train_cfg.norm_groups)
    env["L96_EMA_ENABLED"] = train_cfg.ema_enabled ? "true" : "false"
    env["L96_EMA_DECAY"] = string(train_cfg.ema_decay)
    env["L96_EMA_USE_FOR_EVAL"] = train_cfg.ema_use_for_eval ? "true" : "false"
    env["L96_LOSS_X_WEIGHT"] = string(train_cfg.x_loss_weight)
    env["L96_LOSS_Y_WEIGHT"] = string(train_cfg.y_loss_weight)

    # Requested behavior: train score only, no Langevin/KL checks during retraining.
    env["L96_EVAL_LOSS_EVERY"] = "0"
    env["L96_EVAL_KL_EVERY"] = "0"
    env["L96_SAVE_CHECKPOINT_EVERY"] = "0"

    cmd = setenv(`julia --project=. scripts/L96/lib/train_unet.jl`, env)
    open(train_log_path, "w") do io
        run(pipeline(cmd; stdout=io, stderr=io))
    end

    isfile(model_path) || error("UNet retraining failed: checkpoint not found at $model_path")
    metrics_path = joinpath(diag_dir, "training_metrics.csv")
    train_loss = read_last_train_loss(metrics_path)

    return Dict(
        "model_path" => model_path,
        "metrics_path" => metrics_path,
        "log_path" => train_log_path,
        "data_path" => data_path,
        "train_loss" => train_loss,
        "iter_dir" => iter_dir,
    )
end

function method_jacobian(method::Symbol,
                         θ::Vector{Float64},
                         tensor::Array{Float64,3},
                         A::Array{Float64,2},
                         cfg::L96Config,
                         iter::Int,
                         method_seed_offset::Int;
                         unet_checkpoint_path::Union{Nothing,AbstractString}=nothing)
    if method == :fd
        xfd, yfd = tensor_snapshot_to_xy(tensor, cfg.J)
        S_fd = finite_difference_jacobian_l96((θ[1], θ[2], θ[3], θ[4]), xfd, yfd, cfg;
                                              h_rel=FD_H_REL,
                                              h_abs=collect(Float64.(FD_H_ABS)),
                                              burn_snapshots=2_000,
                                              nsamples=60_000,
                                              n_rep=1,
                                              seed_base=FD_SEED_BASE + method_seed_offset + 100_000 * iter)
        return aggregate_jacobian_rows(S_fd, cfg.K)
    end

    θt = (θ[1], θ[2], θ[3], θ[4])
    Δt_obs = cfg.dt * cfg.save_every

    if method == :gfdt_unet
        unet_checkpoint_path === nothing && error("GFDT+UNet requires per-iteration trained checkpoint")
        G_unet = compute_G_unet(tensor, String(unet_checkpoint_path), θt;
                                batch_size=SCORE_BATCH_SIZE,
                                device_pref=SCORE_DEVICE_PREF)
        S = build_gfdt_jacobian(A, G_unet, Δt_obs, CAL_GFDT_TMAX; mean_center=true)
        return aggregate_jacobian_rows(S, cfg.K)
    elseif method == :gfdt_gaussian
        G_gauss = compute_G_gaussian(tensor, θt)
        S = build_gfdt_jacobian(A, G_gauss, Δt_obs, CAL_GFDT_TMAX; mean_center=true)
        return aggregate_jacobian_rows(S, cfg.K)
    else
        error("Unsupported method: $method")
    end
end

function run_calibration_method(method::Symbol,
                                θ0::Vector{Float64},
                                θ_true::Vector{Float64},
                                target_mean::Vector{Float64},
                                wdiag::Vector{Float64},
                                x0_init::Vector{Float64},
                                y0_init::Matrix{Float64},
                                cfg::L96Config,
                                max_iters::Int,
                                score_train_cfg::ScoreTrainConfig,
                                score_retrain_root::AbstractString)
    mcfg = CAL_METHOD_CONFIG[method]
    θ = copy(θ0)
    θ_path = [copy(θ)]
    rows = NamedTuple[]

    @info "Starting calibration loop" method=method init=θ0 max_iters=max_iters damping=mcfg.damping lambda=mcfg.lambda_reg

    for it in 1:max_iters
        tensor, A, G = evaluate_state_and_stats(θ, x0_init, y0_init, cfg;
                                                nsamples=CAL_EVAL_NSAMPLES,
                                                burn_snapshots=CAL_EVAL_BURN,
                                                rng_seed=mcfg.seed_offset + 10_000 + it)
        resid = G .- target_mean
        obj = weighted_objective(resid, wdiag)

        unet_checkpoint = nothing
        score_train_loss = NaN
        if method == :gfdt_unet
            @info "Retraining UNet score for current iterate" method=method iter=it theta=θ
            train_out = train_unet_on_tensor(tensor, cfg, score_train_cfg, score_retrain_root;
                                             method=method,
                                             iter=it,
                                             seed=mcfg.seed_offset + 500_000 + it)
            unet_checkpoint = String(train_out["model_path"])
            score_train_loss = Float64(train_out["train_loss"])
            @info "UNet retraining complete" iter=it checkpoint=unet_checkpoint train_loss=score_train_loss
        end

        S = method_jacobian(method, θ, tensor, A, cfg, it, mcfg.seed_offset;
                            unet_checkpoint_path=unet_checkpoint)

        accepted = false
        θ_next = copy(θ)
        obj_next = obj
        condM = NaN
        used_damping = 0.0
        λ = mcfg.lambda_reg

        for _ in 1:CAL_LM_MAX_TRIES
            Δθ, condM = solve_weighted_step(S, resid, wdiag; lambda=λ)
            α = mcfg.damping
            for ls in 1:CAL_LS_MAX_TRIES
                θ_try = bounded_update(θ, Δθ; damping=α, step_cap=mcfg.step_cap)
                _, _, G_try = evaluate_state_and_stats(θ_try, x0_init, y0_init, cfg;
                                                       nsamples=CAL_TRIAL_NSAMPLES,
                                                       burn_snapshots=CAL_TRIAL_BURN,
                                                       rng_seed=mcfg.seed_offset + 100_000 + 100 * it + ls)
                obj_try = weighted_objective(G_try .- target_mean, wdiag)
                if obj_try <= obj - CAL_OBJ_IMPROVE_EPS * max(obj, 1e-3)
                    θ_next = θ_try
                    obj_next = obj_try
                    accepted = true
                    used_damping = α
                    break
                end
                α *= 0.5
            end
            accepted && break
            λ *= CAL_LM_GROWTH
        end

        rel_err_next = norm((θ_next .- θ_true) ./ θ_true) / sqrt(length(θ_true))
        push!(rows, (
            iter=it,
            obj=obj,
            obj_next=obj_next,
            damping=used_damping,
            cond=condM,
            theta=copy(θ),
            theta_next=copy(θ_next),
            rel_param_err_next=rel_err_next,
            score_model_path=(unet_checkpoint === nothing ? "" : unet_checkpoint),
            score_train_loss=score_train_loss,
        ))
        θ = θ_next
        push!(θ_path, copy(θ))
        @info "Calibration step" method=method iter=it obj=obj obj_next=obj_next θ=θ rel_param_err=rel_err_next cond=condM accepted=accepted
    end

    return θ, rows, θ_path
end

function write_history_csv(path::AbstractString, rows::Vector{NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "iter,obj,obj_next,damping,cond,theta_F,theta_h,theta_c,theta_b,next_F,next_h,next_c,next_b,rel_param_err,score_train_loss,score_model_path")
        for r in rows
            @printf(io, "%d,%.12e,%.12e,%.6f,%.6e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%s\n",
                    r.iter, r.obj, r.obj_next, r.damping, r.cond,
                    r.theta[1], r.theta[2], r.theta[3], r.theta[4],
                    r.theta_next[1], r.theta_next[2], r.theta_next[3], r.theta_next[4],
                    r.rel_param_err_next,
                    r.score_train_loss,
                    isempty(r.score_model_path) ? "" : "\"" * r.score_model_path * "\"")
        end
    end
    return path
end

function write_parameter_paths_csv(path::AbstractString,
                                   method_paths::Dict{Symbol,Vector{Vector{Float64}}},
                                   method_rows::Dict{Symbol,Vector{NamedTuple}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,iter,F,h,c,b,obj,obj_next")
        for method in CAL_METHOD_ORDER
            θpath = method_paths[method]
            rows = method_rows[method]
            for i in eachindex(θpath)
                it = i - 1
                obj = it == 0 ? NaN : rows[it].obj
                obj_next = it == 0 ? NaN : rows[it].obj_next
                θ = θpath[i]
                @printf(io, "%s,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                        String(method), it, θ[1], θ[2], θ[3], θ[4], obj, obj_next)
            end
        end
    end
    return path
end

function make_parameter_trajectory_figure(path::AbstractString,
                                          method_paths::Dict{Symbol,Vector{Vector{Float64}}},
                                          θ_true::Vector{Float64},
                                          max_iters::Int)
    default(fontfamily="Computer Modern", dpi=180)
    fig = plot(layout=(2, 2), size=(1500, 920))
    steps = collect(0:max_iters)

    for ip in 1:4
        plot!(fig[ip];
              title=@sprintf("%s Calibration Trajectory", CAL_PARAM_NAMES[ip]),
              xlabel="Newton calibration step",
              ylabel=CAL_PARAM_NAMES[ip],
              legend=(ip == 1 ? :best : false),
              lw=2)

        hline!(fig[ip], [θ_true[ip]];
               linestyle=:dash,
               linewidth=2.5,
               color=:black,
               label=(ip == 1 ? "Target" : ""))

        for method in CAL_METHOD_ORDER
            vals = [θ[ip] for θ in method_paths[method]]
            plot!(fig[ip], steps[1:length(vals)], vals;
                  marker=:circle,
                  markersize=4,
                  linewidth=2.5,
                  color=CAL_METHOD_COLOR[method],
                  label=(ip == 1 ? CAL_METHOD_LABEL[method] : ""))
        end
    end

    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function write_summary_md(path::AbstractString,
                          θ_true::Vector{Float64},
                          θ0::Vector{Float64},
                          method_final::Dict{Symbol,Vector{Float64}},
                          method_rows::Dict{Symbol,Vector{NamedTuple}},
                          figure_path::AbstractString,
                          cfg::L96Config,
                          score_train_cfg::ScoreTrainConfig,
                          score_retrain_root::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# L96 Three-Method Calibration")
        println(io)
        println(io, "- true parameters `(F,h,c,b)` = `", θ_true, "`")
        println(io, "- initial guess `(F,h,c,b)` = `", θ0, "`")
        println(io, "- L96 config: `K=", cfg.K, "`, `J=", cfg.J, "`, `dt=", cfg.dt, "`, `save_every=", cfg.save_every, "`, `process_noise_sigma=", cfg.process_noise_sigma, "`")
        println(io, "- figure = `", abspath(figure_path), "`")
        println(io, "- per-iteration UNet retraining dir = `", abspath(score_retrain_root), "`")
        println(io)
        println(io, "## UNet Retraining Config (used at each GFDT+UNet iterate)")
        println(io)
        println(io, "- device: `", score_train_cfg.device, "`")
        println(io, "- epochs: `", score_train_cfg.epochs, "`")
        println(io, "- batch_size: `", score_train_cfg.batch_size, "`")
        println(io, "- lr: `", score_train_cfg.lr, "`")
        println(io, "- train_sigma: `", Float64(score_train_cfg.sigma), "`")
        println(io, "- base_channels: `", score_train_cfg.base_channels, "`")
        println(io, "- channel_multipliers: `", score_train_cfg.channel_multipliers, "`")
        println(io, "- norm_type: `", score_train_cfg.norm_type, "`")
        println(io, "- normalization_mode: `", score_train_cfg.normalization_mode, "`")
        println(io)
        println(io, "| Method | F | h | c | b | rel L2 error |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for method in CAL_METHOD_ORDER
            θ = method_final[method]
            err = norm((θ .- θ_true) ./ θ_true) / sqrt(length(θ_true))
            println(io, "| ", CAL_METHOD_LABEL[method], " | ",
                    @sprintf("%.6f", θ[1]), " | ",
                    @sprintf("%.6f", θ[2]), " | ",
                    @sprintf("%.6f", θ[3]), " | ",
                    @sprintf("%.6f", θ[4]), " | ",
                    @sprintf("%.6e", err), " |")
        end
        println(io)
        for method in CAL_METHOD_ORDER
            rows = method_rows[method]
            if !isempty(rows)
                println(io, "- ", CAL_METHOD_LABEL[method], ": obj `",
                        @sprintf("%.6e", rows[1].obj), "` -> `",
                        @sprintf("%.6e", rows[end].obj_next), "`")
            end
        end
    end
    return path
end

function main(args=ARGS)
    cli = parse_cli(args)
    run_dir = abspath(String(cli["run_dir"]))
    integration_toml = abspath(String(cli["integration_toml"]))
    out_dir = abspath(String(cli["output_dir"]))
    max_iters = Int(cli["max_iters"])
    init_scale = Float64(cli["init_scale"])
    mkpath(out_dir)

    cfg = load_l96_config(integration_toml)
    θ_true = [cfg.F, cfg.h, cfg.c, cfg.b]
    θ0 = init_scale .* θ_true

    target_mean, wdiag = compute_target_stats(cfg)
    init_tensor = load_observation_subset(cfg; nsamples=1, start_index=CAL_TARGET_START_INDEX)
    x0_init, y0_init = tensor_snapshot_to_xy(init_tensor, cfg.J)

    score_train_cfg = load_score_train_config(run_dir, cli)
    score_retrain_root = joinpath(out_dir, "score_retraining")
    mkpath(score_retrain_root)

    method_final = Dict{Symbol,Vector{Float64}}()
    method_rows = Dict{Symbol,Vector{NamedTuple}}()
    method_paths = Dict{Symbol,Vector{Vector{Float64}}}()

    for method in CAL_METHOD_ORDER
        θf, rows, θpath = run_calibration_method(method, θ0, θ_true, target_mean, wdiag,
                                                 x0_init, y0_init, cfg, max_iters,
                                                 score_train_cfg, score_retrain_root)
        method_final[method] = θf
        method_rows[method] = rows
        method_paths[method] = θpath
        write_history_csv(joinpath(out_dir, @sprintf("calibration_history_%s.csv", String(method))), rows)
    end

    paths_csv = write_parameter_paths_csv(joinpath(out_dir, "parameter_paths.csv"), method_paths, method_rows)
    fig_path = make_parameter_trajectory_figure(joinpath(out_dir, "parameter_trajectories_three_methods.png"),
                                                method_paths, θ_true, max_iters)
    summary_md = write_summary_md(joinpath(out_dir, "calibration_three_methods_report.md"),
                                  θ_true, θ0, method_final, method_rows, fig_path, cfg,
                                  score_train_cfg, score_retrain_root)

    summary_toml = joinpath(out_dir, "calibration_three_methods_summary.toml")
    summary_doc = Dict(
        "true_parameters" => Dict("F" => θ_true[1], "h" => θ_true[2], "c" => θ_true[3], "b" => θ_true[4]),
        "initial_guess" => Dict("F" => θ0[1], "h" => θ0[2], "c" => θ0[3], "b" => θ0[4]),
        "score_retraining" => Dict(
            "root" => abspath(score_retrain_root),
            "device" => score_train_cfg.device,
            "epochs" => score_train_cfg.epochs,
            "batch_size" => score_train_cfg.batch_size,
            "lr" => score_train_cfg.lr,
            "sigma" => Float64(score_train_cfg.sigma),
        ),
        "methods" => Dict(
            String(method) => Dict(
                "F" => method_final[method][1],
                "h" => method_final[method][2],
                "c" => method_final[method][3],
                "b" => method_final[method][4],
                "rel_l2_error" => norm((method_final[method] .- θ_true) ./ θ_true) / sqrt(4),
            ) for method in CAL_METHOD_ORDER
        ),
        "paths_csv" => abspath(paths_csv),
        "figure_path" => abspath(fig_path),
        "report_path" => abspath(summary_md),
        "max_iters" => max_iters,
        "init_scale" => init_scale,
    )
    open(summary_toml, "w") do io
        TOML.print(io, summary_doc)
    end

    println("Saved:")
    println("  - ", paths_csv)
    println("  - ", fig_path)
    println("  - ", summary_md)
    println("  - ", summary_toml)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
