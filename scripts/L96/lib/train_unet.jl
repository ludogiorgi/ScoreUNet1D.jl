if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using Flux
using HDF5
using Random
using ScoreUNet1D
using Statistics
using StatsBase
using Printf
using ProgressMeter

include(joinpath(@__DIR__, "run_layout.jl"))
using .L96RunLayout

const PIPELINE_MODE = lowercase(get(ENV, "L96_PIPELINE_MODE", "false")) == "true"
const RUN_DIR = L96RunLayout.pick_stage_run_dir(@__DIR__)
const TRAIN_PATHS = L96RunLayout.default_train_paths(RUN_DIR)

const DATA_PATH = get(ENV, "L96_DATA_PATH", L96RunLayout.default_data_path(RUN_DIR))
const DATASET_KEY = get(ENV, "L96_DATASET_KEY", "timeseries")
const MODEL_PATH = get(ENV, "L96_MODEL_PATH", TRAIN_PATHS.model_path)
const TRAIN_DIAG_DIR = get(ENV, "L96_TRAIN_DIAG_DIR", TRAIN_PATHS.diagnostics_dir)
const CHECKPOINT_DIR = get(ENV, "L96_CHECKPOINT_DIR", TRAIN_PATHS.checkpoints_dir)
const TRAIN_CONFIG_PATH = get(ENV, "L96_TRAIN_CONFIG_PATH", PIPELINE_MODE ? "" : L96RunLayout.default_config_path(RUN_DIR, "train_unet"))

const DEVICE_NAME = get(ENV, "L96_TRAIN_DEVICE", "GPU:0")
const TRAIN_SEED = parse(Int, get(ENV, "L96_TRAIN_SEED", "42"))

const BATCH_SIZE = parse(Int, get(ENV, "L96_BATCH_SIZE", "256"))
const EPOCHS = parse(Int, get(ENV, "L96_EPOCHS", "30"))
const LR = parse(Float64, get(ENV, "L96_LR", "8e-4"))
const SIGMA = parse(Float32, get(ENV, "L96_TRAIN_NOISE_SIGMA", "0.08"))
const BASE_CHANNELS = parse(Int, get(ENV, "L96_BASE_CHANNELS", "32"))
const CHANNEL_MULTIPLIERS_RAW = get(ENV, "L96_CHANNEL_MULTIPLIERS", "1,2,4")
const NORMALIZATION_MODE = lowercase(get(ENV, "L96_NORMALIZATION_MODE", "split_xy"))
const MODEL_ARCH = lowercase(get(ENV, "L96_MODEL_ARCH", "schneider_dualstream"))
const PROGRESS = lowercase(get(ENV, "L96_PROGRESS", "false")) == "true"
const USE_LR_SCHEDULE = lowercase(get(ENV, "L96_USE_LR_SCHEDULE", "true")) == "true"
const WARMUP_STEPS = parse(Int, get(ENV, "L96_WARMUP_STEPS", "500"))
const MIN_LR_FACTOR = parse(Float64, get(ENV, "L96_MIN_LR_FACTOR", "0.1"))
const NORM_TYPE = lowercase(get(ENV, "L96_NORM_TYPE", "group"))
const NORM_GROUPS = parse(Int, get(ENV, "L96_NORM_GROUPS", "0"))
const EMA_ENABLED = lowercase(get(ENV, "L96_EMA_ENABLED", "true")) == "true"
const EMA_DECAY = parse(Float32, get(ENV, "L96_EMA_DECAY", "0.999"))
const EMA_USE_FOR_EVAL = lowercase(get(ENV, "L96_EMA_USE_FOR_EVAL", "true")) == "true"
const LOSS_X_WEIGHT = parse(Float32, get(ENV, "L96_LOSS_X_WEIGHT", "1.0"))
const LOSS_Y_WEIGHT = parse(Float32, get(ENV, "L96_LOSS_Y_WEIGHT", "1.0"))

const EVAL_NUM_SAMPLES = parse(Int, get(ENV, "L96_EVAL_NUM_SAMPLES", "2048"))
const EVAL_BATCH_SIZE = parse(Int, get(ENV, "L96_EVAL_BATCH_SIZE", "256"))
const EVAL_LOSS_EVERY = parse(Int, get(ENV, "L96_EVAL_LOSS_EVERY", "1"))
const EVAL_KL_EVERY = parse(Int, get(ENV, "L96_EVAL_KL_EVERY", "0"))
const SAVE_CHECKPOINT_EVERY = parse(Int, get(ENV, "L96_SAVE_CHECKPOINT_EVERY", "0"))

const EVAL_KL_DT = parse(Float64, get(ENV, "L96_EVAL_KL_DT", "0.0005"))
const EVAL_KL_STEPS = parse(Int, get(ENV, "L96_EVAL_KL_STEPS", "4000"))
const EVAL_KL_RESOLUTION = parse(Int, get(ENV, "L96_EVAL_KL_RESOLUTION", "100"))
const EVAL_KL_BURN_IN = parse(Int, get(ENV, "L96_EVAL_KL_BURN_IN", "1000"))
const EVAL_KL_ENSEMBLES = parse(Int, get(ENV, "L96_EVAL_KL_ENSEMBLES", "8"))
const EVAL_KL_BINS = parse(Int, get(ENV, "L96_EVAL_KL_BINS", "64"))
const EVAL_KL_LOW_Q = parse(Float64, get(ENV, "L96_EVAL_KL_LOW_Q", "0.001"))
const EVAL_KL_HIGH_Q = parse(Float64, get(ENV, "L96_EVAL_KL_HIGH_Q", "0.999"))
const EVAL_KL_BOUNDARY = lowercase(get(ENV, "L96_EVAL_KL_BOUNDARY", "true")) == "true"
const EVAL_KL_SEED_BASE = parse(Int, get(ENV, "L96_EVAL_KL_SEED_BASE", "1000"))

function parse_norm_type(raw::AbstractString)
    s = lowercase(strip(raw))
    if s == "batch"
        return :batch
    elseif s == "group"
        return :group
    end
    error("Unsupported L96_NORM_TYPE='$raw'. Use 'batch' or 'group'.")
end

function parse_model_arch(raw::AbstractString)
    s = lowercase(strip(raw))
    if s == "schneider_dualstream"
        return :schneider_dualstream
    elseif s == "multichannel_unet"
        return :multichannel_unet
    end
    error("Unsupported L96_MODEL_ARCH='$raw'. Use 'schneider_dualstream' or 'multichannel_unet'.")
end

function parse_channel_multipliers(raw::AbstractString)
    vals = Int[]
    for part in split(raw, ",")
        s = strip(part)
        isempty(s) && continue
        v = parse(Int, s)
        v > 0 || error("Channel multipliers must be positive: got $v")
        push!(vals, v)
    end
    isempty(vals) && error("No valid channel multipliers parsed from '$raw'")
    return vals
end

function load_tensor(path::AbstractString)
    raw = h5open(path, "r") do h5
        read(h5, DATASET_KEY)
    end
    tensor = permutedims(raw, (3, 2, 1))
    return Array{Float32,3}(tensor)
end

function split_channel_normalize(tensor::Array{Float32,3})
    L, C, _ = size(tensor)
    C >= 2 || error("Expected at least 2 channels (x + y*), got $C")

    μx = Float32(mean(@view tensor[:, 1, :]))
    σx = Float32(std(@view tensor[:, 1, :]) + eps(Float32))

    μy = Float32(mean(@view tensor[:, 2:C, :]))
    σy = Float32(std(@view tensor[:, 2:C, :]) + eps(Float32))

    mean_mat = Array{Float32}(undef, C, L)
    std_mat = Array{Float32}(undef, C, L)
    @inbounds begin
        mean_mat[1, :] .= μx
        mean_mat[2:C, :] .= μy
        std_mat[1, :] .= σx
        std_mat[2:C, :] .= σy
    end

    stats = DataStats(mean_mat, std_mat)
    mean_lc = permutedims(mean_mat, (2, 1))
    std_lc = permutedims(std_mat, (2, 1))

    normalized = (tensor .- reshape(mean_lc, L, C, 1)) ./ reshape(std_lc, L, C, 1)
    return Array{Float32,3}(normalized), stats
end

function per_channel_normalize(tensor::Array{Float32,3})
    L, C, _ = size(tensor)
    mean_mat = Array{Float32}(undef, C, L)
    std_mat = Array{Float32}(undef, C, L)
    @inbounds for c in 1:C
        μc = Float32(mean(@view tensor[:, c, :]))
        σc = Float32(std(@view tensor[:, c, :]) + eps(Float32))
        mean_mat[c, :] .= μc
        std_mat[c, :] .= σc
    end

    stats = DataStats(mean_mat, std_mat)
    mean_lc = permutedims(mean_mat, (2, 1))
    std_lc = permutedims(std_mat, (2, 1))
    normalized = (tensor .- reshape(mean_lc, L, C, 1)) ./ reshape(std_lc, L, C, 1)
    return Array{Float32,3}(normalized), stats
end

function normalize_tensor(tensor::Array{Float32,3}, mode::AbstractString)
    if mode == "split_xy"
        return split_channel_normalize(tensor)
    elseif mode == "per_channel"
        return per_channel_normalize(tensor)
    else
        error("Unsupported normalization mode '$mode'. Use 'split_xy' or 'per_channel'.")
    end
end

function build_eval_tensor(tensor::Array{Float32,3}, nsamples::Int, seed::Int)
    _, _, B = size(tensor)
    n = min(max(nsamples, 1), B)
    rng = MersenneTwister(seed)
    idx = sample(rng, 1:B, n; replace=false)
    return Array{Float32,3}(@view tensor[:, :, idx])
end

function with_padding(lo::Float64, hi::Float64)
    if lo == hi
        δ = max(abs(lo), 1.0) * 1e-3
        return lo - δ, hi + δ
    end
    return lo, hi
end

function histogram_prob(samples::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, samples, edges)
    probs = Float64.(hist.weights)
    probs .+= eps(Float64)
    probs ./= sum(probs)
    return probs
end

@inline function discrete_kl(p::Vector{Float64}, q::Vector{Float64})
    return sum(@. p * log(p / q))
end

function avg_modewise_kl(truth::Array{Float32,3}, generated::Array{Float32,3};
                         nbins::Int=64, low_q::Float64=0.001, high_q::Float64=0.999)
    L, C, _ = size(truth)
    acc = 0.0
    n = L * C
    @inbounds for l in 1:L, c in 1:C
        tvals = Float64.(vec(@view truth[l, c, :]))
        gvals = Float64.(vec(@view generated[l, c, :]))
        combined = vcat(tvals, gvals)
        lo = quantile(combined, low_q)
        hi = quantile(combined, high_q)
        lo, hi = with_padding(lo, hi)
        edges = collect(range(lo, hi; length=nbins + 1))
        p = histogram_prob(tvals, edges)
        q = histogram_prob(gvals, edges)
        acc += discrete_kl(p, q)
    end
    return acc / n
end

function eval_denoise_loss(model,
                           eval_tensor::Array{Float32,3},
                           sigma::Float32,
                           device::ExecutionDevice;
                           batch_size::Int=256,
                           seed::Int=0)
    Flux.testmode!(model)
    rng = MersenneTwister(seed)
    L, C, N = size(eval_tensor)
    total = 0.0
    count = 0

    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        batch_cpu = Array{Float32,3}(undef, L, C, b)
        @views batch_cpu .= eval_tensor[:, :, start:stop]
        noise_cpu = randn(rng, Float32, L, C, b)

        if is_gpu(device)
            batch = move_array(batch_cpu, device)
            noise = move_array(noise_cpu, device)
            noisy = batch .+ sigma .* noise
            pred = model(noisy)
            loss_b = Float64(Flux.cpu(Flux.Losses.mse(pred, noise)))
        else
            noisy = batch_cpu .+ sigma .* noise_cpu
            pred = model(noisy)
            loss_b = Float64(Flux.Losses.mse(pred, noise_cpu))
        end

        total += loss_b * b
        count += b
    end

    return total / max(count, 1)
end

function eval_epoch_kl(model,
                       eval_dataset::NormalizedDataset,
                       sigma::Float32,
                       device::ExecutionDevice;
                       seed::Int=0)
    Flux.testmode!(model)
    cfg = LangevinConfig(
        dt=EVAL_KL_DT,
        sample_dt=EVAL_KL_DT * EVAL_KL_RESOLUTION,
        nsteps=EVAL_KL_STEPS,
        burn_in=EVAL_KL_BURN_IN,
        resolution=EVAL_KL_RESOLUTION,
        n_ensembles=EVAL_KL_ENSEMBLES,
        nbins=EVAL_KL_BINS,
        sigma=sigma,
        seed=seed,
        mode=:all,
        boundary=EVAL_KL_BOUNDARY ? (-10.0, 10.0) : nothing,
        progress=false,
    )
    result = run_langevin(model, eval_dataset, cfg; device=device)

    truth = eval_dataset.data
    L, C, _ = size(truth)
    traj4 = reshape(result.trajectory, L, C, :, size(result.trajectory, 3))
    generated = reshape(permutedims(traj4, (1, 2, 4, 3)), L, C, :)
    mode_kl = avg_modewise_kl(truth, generated;
                              nbins=EVAL_KL_BINS,
                              low_q=EVAL_KL_LOW_Q,
                              high_q=EVAL_KL_HIGH_Q)
    return mode_kl, Float64(result.kl_divergence)
end

function write_training_metrics(path::AbstractString,
                                epoch_losses::Vector{Float64},
                                epoch_times::Vector{Float64},
                                eval_epochs::Vector{Int},
                                eval_losses::Vector{Float64},
                                kl_epochs::Vector{Int},
                                mode_kl::Vector{Float64},
                                global_kl::Vector{Float64})
    open(path, "w") do io
        println(io, "# epoch,epoch_loss,epoch_time_sec")
        for i in eachindex(epoch_losses)
            t = i <= length(epoch_times) ? epoch_times[i] : NaN
            println(io, "$(i),$(epoch_losses[i]),$(t)")
        end
        println(io, "# eval_epoch,eval_denoise_loss")
        for i in eachindex(eval_epochs)
            println(io, "$(eval_epochs[i]),$(eval_losses[i])")
        end
        println(io, "# kl_epoch,avg_mode_kl,global_kl")
        for i in eachindex(kl_epochs)
            println(io, "$(kl_epochs[i]),$(mode_kl[i]),$(global_kl[i])")
        end
    end
    return path
end

function save_training_config(run_dir::AbstractString,
                              tensor_shape::NTuple{3,Int},
                              multipliers::Vector{Int},
                              model_arch::Symbol)
    isempty(TRAIN_CONFIG_PATH) && return ""
    steps_per_epoch = cld(tensor_shape[3], BATCH_SIZE)
    ema_decay_epoch = Float64(EMA_DECAY ^ steps_per_epoch)

    cfg = Dict{String,Any}(
        "stage" => "train_unet",
        "run_dir" => run_dir,
        "paths" => Dict(
            "data_path" => abspath(DATA_PATH),
            "model_path" => abspath(MODEL_PATH),
            "train_diag_dir" => abspath(TRAIN_DIAG_DIR),
            "checkpoint_dir" => abspath(CHECKPOINT_DIR),
        ),
        "data" => Dict(
            "dataset_key" => DATASET_KEY,
            "tensor_shape_lcb" => [tensor_shape[1], tensor_shape[2], tensor_shape[3]],
            "normalization_mode" => NORMALIZATION_MODE,
        ),
        "model" => Dict(
            "architecture" => String(model_arch),
            "base_channels" => BASE_CHANNELS,
            "channel_multipliers" => multipliers,
            "periodic" => true,
            "norm_type" => NORM_TYPE,
            "norm_groups" => NORM_GROUPS,
        ),
        "trainer" => Dict(
            "device" => DEVICE_NAME,
            "seed" => TRAIN_SEED,
            "batch_size" => BATCH_SIZE,
            "epochs" => EPOCHS,
            "lr" => LR,
            "sigma" => Float64(SIGMA),
            "progress" => PROGRESS,
            "use_lr_schedule" => USE_LR_SCHEDULE,
            "warmup_steps" => WARMUP_STEPS,
            "min_lr_factor" => MIN_LR_FACTOR,
            "loss_x_weight" => Float64(LOSS_X_WEIGHT),
            "loss_y_weight" => Float64(LOSS_Y_WEIGHT),
            "ema_enabled" => EMA_ENABLED,
            "ema_decay_per_step" => Float64(EMA_DECAY),
            "ema_decay_effective_per_epoch" => ema_decay_epoch,
            "ema_use_for_eval" => EMA_USE_FOR_EVAL,
        ),
        "eval" => Dict(
            "eval_num_samples" => EVAL_NUM_SAMPLES,
            "eval_dataset_mode" => "train_full",
            "eval_batch_size" => EVAL_BATCH_SIZE,
            "eval_loss_every" => EVAL_LOSS_EVERY,
            "eval_kl_every" => EVAL_KL_EVERY,
            "save_checkpoint_every" => SAVE_CHECKPOINT_EVERY,
            "eval_kl_dt" => EVAL_KL_DT,
            "eval_kl_steps" => EVAL_KL_STEPS,
            "eval_kl_resolution" => EVAL_KL_RESOLUTION,
            "eval_kl_burn_in" => EVAL_KL_BURN_IN,
            "eval_kl_ensembles" => EVAL_KL_ENSEMBLES,
            "eval_kl_bins" => EVAL_KL_BINS,
            "eval_kl_low_q" => EVAL_KL_LOW_Q,
            "eval_kl_high_q" => EVAL_KL_HIGH_Q,
            "eval_kl_boundary" => EVAL_KL_BOUNDARY,
            "eval_kl_seed_base" => EVAL_KL_SEED_BASE,
        ),
    )
    return L96RunLayout.write_toml_file(TRAIN_CONFIG_PATH, cfg)
end

function main()
    isfile(DATA_PATH) || error("Dataset not found at $DATA_PATH. Run generate_data.jl first.")
    ensure_dir(dirname(MODEL_PATH))
    ensure_dir(TRAIN_DIAG_DIR)
    if !PIPELINE_MODE
        L96RunLayout.ensure_runs_readme!(L96RunLayout.default_runs_root(@__DIR__))
    end

    tensor = load_tensor(DATA_PATH)
    L, C, B = size(tensor)
    @info "Loaded L96 tensor" size = size(tensor) run_dir = RUN_DIR

    tensor_norm, stats = normalize_tensor(tensor, NORMALIZATION_MODE)
    dataset = NormalizedDataset(tensor_norm, stats)
    eval_tensor = tensor_norm
    eval_dataset = dataset
    multipliers = parse_channel_multipliers(CHANNEL_MULTIPLIERS_RAW)
    model_arch = parse_model_arch(MODEL_ARCH)
    train_config_path = save_training_config(RUN_DIR, size(tensor), multipliers, model_arch)
    norm_symbol = parse_norm_type(NORM_TYPE)
    steps_per_epoch = cld(B, BATCH_SIZE)
    # EMA decay is commonly chosen as a per-step factor; convert to an equivalent
    # per-epoch factor because this stage updates EMA once per epoch.
    ema_decay_epoch = Float32(EMA_DECAY ^ steps_per_epoch)

    Random.seed!(TRAIN_SEED)
    cfg = nothing
    model = nothing
    if model_arch == :schneider_dualstream
        C >= 2 || error("Need at least 2 channels (x + fast channels) for schneider_dualstream, got $C")
        J_fast = C - 1
        sch_cfg = L96SchneiderScoreConfig(
            K=L,
            J=J_fast,
            slow_base_channels=max(8, fld(BASE_CHANNELS, 2)),
            fast_base_channels=BASE_CHANNELS,
            slow_channel_multipliers=multipliers,
            fast_channel_multipliers=multipliers,
            kernel_size=5,
            norm_type=norm_symbol,
            norm_groups=NORM_GROUPS,
        )
        cfg = sch_cfg
        model = build_l96_schneider_legacy_model(sch_cfg)
    else
        vanilla_cfg = ScoreUNetConfig(
            in_channels=C,
            out_channels=C,
            base_channels=BASE_CHANNELS,
            channel_multipliers=multipliers,
            periodic=true,
            norm_type=norm_symbol,
            norm_groups=NORM_GROUPS,
        )
        cfg = vanilla_cfg
        model = build_unet(vanilla_cfg)
    end

    trainer_cfg = ScoreTrainerConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        sigma=SIGMA,
        seed=TRAIN_SEED,
        progress=false,
        use_lr_schedule=USE_LR_SCHEDULE,
        warmup_steps=WARMUP_STEPS,
        min_lr_factor=MIN_LR_FACTOR,
        x_loss_weight=LOSS_X_WEIGHT,
        y_loss_weight=LOSS_Y_WEIGHT,
    )

    device = select_device(DEVICE_NAME)
    device isa ScoreUNet1D.GPUDevice || error("L96 training must run on GPU. Set L96_TRAIN_DEVICE to a GPU target (e.g., GPU:0).")
    activate_device!(device)
    model = move_model(model, device)

    epoch_times = Float64[]
    eval_epochs = Int[]
    eval_losses = Float64[]
    kl_epochs = Int[]
    kl_mode_vals = Float64[]
    kl_global_vals = Float64[]
    epoch_progress = Progress(EPOCHS; desc="Training epochs")
    ema_model_cpu = EMA_ENABLED ? move_model(model, ScoreUNet1D.CPUDevice()) : nothing
    if ema_model_cpu !== nothing
        Flux.testmode!(ema_model_cpu)
    end

    epoch_callback = function(epoch::Int, model_epoch, epoch_time::Real)
        push!(epoch_times, Float64(epoch_time))
        Flux.testmode!(model_epoch)
        model_cpu_epoch = move_model(model_epoch, ScoreUNet1D.CPUDevice())
        Flux.testmode!(model_cpu_epoch)

        if ema_model_cpu !== nothing
            ema_model_cpu = Flux.fmap(ema_model_cpu, model_cpu_epoch) do a, b
                if a isa AbstractArray && b isa AbstractArray
                    return ema_decay_epoch .* a .+ Float32(1 - ema_decay_epoch) .* b
                end
                return a
            end
            Flux.testmode!(ema_model_cpu)
        end

        if EVAL_LOSS_EVERY > 0 && epoch % EVAL_LOSS_EVERY == 0
            eval_loss = eval_denoise_loss(model_epoch, eval_tensor, SIGMA, device;
                                          batch_size=EVAL_BATCH_SIZE,
                                          seed=TRAIN_SEED + epoch)
            push!(eval_epochs, epoch)
            push!(eval_losses, eval_loss)
            @info "Eval denoising loss" epoch = epoch eval_loss = eval_loss
        end

        if EVAL_KL_EVERY > 0 && epoch % EVAL_KL_EVERY == 0
            mode_kl, global_kl = eval_epoch_kl(model_epoch, eval_dataset, SIGMA, device;
                                               seed=EVAL_KL_SEED_BASE + epoch)
            push!(kl_epochs, epoch)
            push!(kl_mode_vals, mode_kl)
            push!(kl_global_vals, global_kl)
            @info "Eval KL" epoch = epoch avg_mode_kl = mode_kl global_kl = global_kl
        end

        if SAVE_CHECKPOINT_EVERY > 0 && epoch % SAVE_CHECKPOINT_EVERY == 0
            ensure_dir(CHECKPOINT_DIR)
            ckpt_path = joinpath(CHECKPOINT_DIR, @sprintf("epoch_%03d.bson", epoch))
            if EMA_ENABLED && EMA_USE_FOR_EVAL && ema_model_cpu !== nothing
                raw_ckpt = joinpath(CHECKPOINT_DIR, @sprintf("epoch_%03d_raw.bson", epoch))
                BSON.@save raw_ckpt model = model_cpu_epoch cfg trainer_cfg stats epoch
                BSON.@save ckpt_path model = ema_model_cpu cfg trainer_cfg stats epoch
                @info "Saved epoch checkpoint" path = ckpt_path raw = raw_ckpt
            elseif EMA_ENABLED && ema_model_cpu !== nothing
                ema_ckpt = joinpath(CHECKPOINT_DIR, @sprintf("epoch_%03d_ema.bson", epoch))
                BSON.@save ckpt_path model = model_cpu_epoch cfg trainer_cfg stats epoch
                BSON.@save ema_ckpt model = ema_model_cpu cfg trainer_cfg stats epoch
                @info "Saved epoch checkpoint" path = ckpt_path ema = ema_ckpt
            else
                BSON.@save ckpt_path model = model_cpu_epoch cfg trainer_cfg stats epoch
                @info "Saved epoch checkpoint" path = ckpt_path
            end
        end

        epoch_progress !== nothing && ProgressMeter.next!(epoch_progress; showvalues=[(:epoch, epoch), (:epoch_time_sec, round(Float64(epoch_time); digits=2))])

        Flux.trainmode!(model_epoch)
        return nothing
    end

    @info "Starting L96 training" device = DEVICE_NAME epochs = EPOCHS batch_size = BATCH_SIZE sigma = SIGMA multipliers = multipliers architecture = String(model_arch) normalization = NORMALIZATION_MODE ema_decay_per_step = Float64(EMA_DECAY) ema_decay_effective_per_epoch = Float64(ema_decay_epoch) run_dir = RUN_DIR config = (isempty(train_config_path) ? "disabled_in_pipeline_mode" : train_config_path)
    history = train!(model, dataset, trainer_cfg; device=device, epoch_callback=epoch_callback)
    epoch_progress !== nothing && ProgressMeter.finish!(epoch_progress)

    model_cpu_raw = is_gpu(device) ? move_model(model, ScoreUNet1D.CPUDevice()) : model
    Flux.testmode!(model_cpu_raw)
    model_cpu = if EMA_ENABLED && EMA_USE_FOR_EVAL && ema_model_cpu !== nothing
        ema_model_cpu
    else
        model_cpu_raw
    end
    Flux.testmode!(model_cpu)

    training_metrics_path = joinpath(TRAIN_DIAG_DIR, "training_metrics.csv")
    epoch_losses_f64 = Float64.(history.epoch_losses)
    write_training_metrics(training_metrics_path,
                           epoch_losses_f64,
                           epoch_times,
                           eval_epochs,
                           eval_losses,
                           kl_epochs,
                           kl_mode_vals,
                           kl_global_vals)

    normalization_mode = NORMALIZATION_MODE
    run_dir = RUN_DIR
    if EMA_ENABLED && EMA_USE_FOR_EVAL
        raw_model_path = joinpath(dirname(MODEL_PATH), "score_model_raw.bson")
        BSON.@save raw_model_path model = model_cpu_raw cfg trainer_cfg stats history epoch_times eval_epochs eval_losses kl_epochs kl_mode_vals kl_global_vals training_metrics_path normalization_mode run_dir train_config_path
    end
    BSON.@save MODEL_PATH model = model_cpu cfg trainer_cfg stats history epoch_times eval_epochs eval_losses kl_epochs kl_mode_vals kl_global_vals training_metrics_path normalization_mode run_dir train_config_path

    train_metrics_for_manifest = Dict{String,Any}(
        "final_train_loss" => Float64(history.epoch_losses[end]),
        "best_train_loss" => minimum(Float64.(history.epoch_losses)),
    )
    if !isempty(eval_losses)
        train_metrics_for_manifest["best_eval_denoise_loss"] = minimum(eval_losses)
        train_metrics_for_manifest["last_eval_denoise_loss"] = eval_losses[end]
    end
    if !isempty(kl_mode_vals)
        train_metrics_for_manifest["best_eval_avg_mode_kl"] = minimum(kl_mode_vals)
        train_metrics_for_manifest["last_eval_avg_mode_kl"] = kl_mode_vals[end]
    end

    train_artifacts = Dict{String,Any}(
        "train_config" => abspath(train_config_path),
        "training_metrics_csv" => abspath(training_metrics_path),
    )

    if !PIPELINE_MODE
        manifest_path = L96RunLayout.update_run_manifest!(
            RUN_DIR;
            stage="train_unet",
            parameters=Dict(
                "train_noise_sigma" => Float64(SIGMA),
                "train_seed" => TRAIN_SEED,
                "batch_size" => BATCH_SIZE,
                "epochs" => EPOCHS,
                "learning_rate" => LR,
                "base_channels" => BASE_CHANNELS,
                "channel_multipliers" => multipliers,
                "architecture" => String(model_arch),
                "normalization_mode" => NORMALIZATION_MODE,
                "norm_type" => NORM_TYPE,
                "norm_groups" => NORM_GROUPS,
                "device" => DEVICE_NAME,
                "use_lr_schedule" => USE_LR_SCHEDULE,
                "warmup_steps" => WARMUP_STEPS,
                "min_lr_factor" => MIN_LR_FACTOR,
                "x_loss_weight" => Float64(LOSS_X_WEIGHT),
                "y_loss_weight" => Float64(LOSS_Y_WEIGHT),
                "ema_enabled" => EMA_ENABLED,
                "ema_decay_per_step" => Float64(EMA_DECAY),
                "ema_decay_effective_per_epoch" => Float64(ema_decay_epoch),
                "ema_use_for_eval" => EMA_USE_FOR_EVAL,
            ),
            paths=Dict(
                "data_path" => abspath(DATA_PATH),
                "model_path" => abspath(MODEL_PATH),
                "train_figures_dir" => abspath(TRAIN_DIAG_DIR),
                "checkpoint_dir" => abspath(CHECKPOINT_DIR),
            ),
            artifacts=train_artifacts,
            metrics=train_metrics_for_manifest,
            notes=Dict(
                "tensor_shape_lcb" => [L, C, B],
                "eval_kl_every" => EVAL_KL_EVERY,
                "save_checkpoint_every" => SAVE_CHECKPOINT_EVERY,
            ),
        )
        summary_path = L96RunLayout.write_run_summary!(RUN_DIR)
        index_paths = L96RunLayout.refresh_runs_index!(L96RunLayout.default_runs_root(@__DIR__))
        compat_links = L96RunLayout.update_compat_links!(
            @__DIR__;
            model_path=MODEL_PATH,
            checkpoints_dir=CHECKPOINT_DIR,
        )

        L96RunLayout.write_latest_run!(@__DIR__, RUN_DIR)
        @info "Saved L96 model" path = MODEL_PATH final_loss = history.epoch_losses[end] samples = B L = L C = C run_dir = RUN_DIR manifest = manifest_path summary = summary_path compat_links = compat_links
        @info "Saved training diagnostics" dir = TRAIN_DIAG_DIR metrics = training_metrics_path config = train_config_path run_index = index_paths.index
    else
        @info "Saved L96 model" path = MODEL_PATH final_loss = history.epoch_losses[end] samples = B L = L C = C run_dir = RUN_DIR
        @info "Saved training diagnostics" dir = TRAIN_DIAG_DIR metrics = training_metrics_path
    end
end

main()
