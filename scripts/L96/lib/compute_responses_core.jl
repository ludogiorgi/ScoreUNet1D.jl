#!/usr/bin/env julia
# Standard command (preferred wrapper):
# julia --threads auto --project=. scripts/L96/compute_responses.jl --params scripts/L96/parameters_responses.toml
# Nohup command:
# nohup julia --threads auto --project=. scripts/L96/compute_responses.jl --params scripts/L96/parameters_responses.toml > scripts/L96/nohup_compute_responses.log 2>&1 &
#
# Core implementation for:
# - scripts/L96/compute_responses.jl
# - scripts/L96/generate_reference_responses.jl

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using FFTW
using Flux
using HDF5
using LinearAlgebra
using Plots
using Printf
using ProgressMeter
using Random
using ScoreUNet1D
using SHA
using Statistics
using TOML
using Base.Threads
using Dates

const RESP_DEFAULT_RUN_DIR = "scripts/L96/runs_J10/run_026"
const RESP_DEFAULT_CHECKPOINT = "scripts/L96/runs_J10/run_026/model/score_model_epoch_0020.bson"
const RESP_DEFAULT_INTEGRATION_TOML = "scripts/L96/observations/J10/integration_params.toml"
const RESP_DEFAULT_OUTPUT_ROOT = "scripts/L96/responses_results"
const RESP_DEFAULT_PARAMS_TOML = "scripts/L96/parameters_responses.toml"

const RESP_DEFAULT_GFDT_NSAMPLES = 150_000
const RESP_DEFAULT_GFDT_START_INDEX = 50_001
const RESP_DEFAULT_TMAX = 6.0
const RESP_PLOT_TMAX = 2.0
const RESP_PLOT_NPOINTS = 301
const RESP_DEFAULT_SCORE_BATCH_SIZE = 2048
const RESP_DEFAULT_SCORE_DEVICE = "auto"
const RESP_DEFAULT_SCORE_FORWARD_MODE = "train"
const RESP_DEFAULT_AUTO_GPU_MAX_MEM_FRACTION = 0.25
const RESP_DEFAULT_AUTO_GPU_MAX_UTILIZATION = 20.0
const RESP_DEFAULT_AUTO_GPU_MIN_FREE_MB = 2048.0
const RESP_DEFAULT_DATASET_ATTR_SYNC_MODE = "override" # one of: off|warn|error|override

const RESP_DEFAULT_NUM_ENSEMBLES = 384
const RESP_DEFAULT_NUM_START_INDEX = 80_001
const RESP_DEFAULT_NUM_SEED_BASE = 920_000
const RESP_DEFAULT_NUM_METHOD = "finite_difference"

const RESP_DEFAULT_H_REL = 5e-3
const RESP_DEFAULT_H_ABS = [1e-2, 1e-3, 1e-2, 1e-2]
const RESP_DEFAULT_CORR_RIDGE = 1e-8
const RESP_DEFAULT_CORR_MAX_DEVIATION = 1.5
const RESP_DEFAULT_MEAN_CENTER = true
const RESP_DEFAULT_REFERENCE_CACHE_ROOT = "scripts/L96/reference_responses_cache"
const RESP_DEFAULT_REFERENCE_GFDT_NSAMPLES = 200_000
const RESP_DEFAULT_REFERENCE_GFDT_START_INDEX = 50_001
const RESP_DEFAULT_REFERENCE_NUM_ENSEMBLES = 16_384
const RESP_DEFAULT_REFERENCE_NUM_START_INDEX = 80_001
const RESP_DEFAULT_REFERENCE_NUM_METHOD = "tangent"
const RESP_DEFAULT_REFERENCE_NUM_SEED_BASE = 1_920_000
const RESP_DEFAULT_REFERENCE_TMAX = RESP_PLOT_TMAX

const RESP_OBS_LABELS = [
    "phi1 = <X>",
    "phi2 = <X^2>",
    "phi3 = <X*Ybar>",
    "phi4 = <Y^2>",
    "phi5 = <X_k X_(k-1)>",
]
const PARAM_NAMES = ["F", "h", "c", "b"]
const OBS_GLOBAL_NAMES = [
    "phi1_mean_x",
    "phi2_mean_x2",
    "phi3_mean_x_ybar",
    "phi4_mean_y2",
    "phi5_mean_x_xm1",
]

struct L96Config
    K::Int
    J::Int
    F::Float64
    h::Float64
    c::Float64
    b::Float64
    dt::Float64
    save_every::Int
    process_noise_sigma::Float64
    stochastic_x_noise::Bool
    dataset_path::String
    dataset_key::String
end

mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

function load_l96_config(integration_toml_path::AbstractString)
    isfile(integration_toml_path) || error("Integration config not found: $integration_toml_path")
    doc = TOML.parsefile(integration_toml_path)
    integ = get(doc, "integration", Dict{String,Any}())
    dset = get(doc, "dataset", Dict{String,Any}())
    path = String(get(dset, "path", ""))
    key = String(get(dset, "key", "timeseries"))
    isempty(path) && error("dataset.path missing in $integration_toml_path")

    return L96Config(
        Int(get(integ, "K", 36)),
        Int(get(integ, "J", 10)),
        Float64(get(integ, "F", 10.0)),
        Float64(get(integ, "h", 1.0)),
        Float64(get(integ, "c", 10.0)),
        Float64(get(integ, "b", 10.0)),
        Float64(get(integ, "dt", 0.005)),
        Int(get(integ, "save_every", 10)),
        Float64(get(integ, "process_noise_sigma", 0.03)),
        Bool(get(integ, "stochastic_x_noise", false)),
        path,
        key,
    )
end

function pick_best_checkpoint(run_dir::AbstractString)
    run_summary_path = joinpath(run_dir, "metrics", "run_summary.toml")
    isfile(run_summary_path) || error("run_summary.toml not found at $run_summary_path")
    summary = TOML.parsefile(run_summary_path)

    eval_tbl = get(summary, "evaluation", Dict{String,Any}())
    best_epoch = Int(get(eval_tbl, "best_epoch", -1))
    best_epoch > 0 || error("best_epoch missing/invalid in $run_summary_path")

    best_name = @sprintf("score_model_epoch_%04d.bson", best_epoch)
    best_path = joinpath(run_dir, "model", best_name)
    if !isfile(best_path)
        best_name3 = @sprintf("score_model_epoch_%03d.bson", best_epoch)
        best_path3 = joinpath(run_dir, "model", best_name3)
        isfile(best_path3) || error("Best checkpoint not found for epoch $best_epoch in $run_dir/model")
        best_path = best_path3
    end
    return (best_epoch=best_epoch, checkpoint_path=best_path)
end

function load_observation_subset(cfg::L96Config;
    nsamples::Int,
    start_index::Int=1,
    subset_label::AbstractString="subset")
    path = cfg.dataset_path
    isfile(path) || error("Observation dataset not found: $path")

    raw = h5open(path, "r") do h5
        haskey(h5, cfg.dataset_key) || error("Dataset key $(cfg.dataset_key) not found in $path")
        ds = h5[cfg.dataset_key]
        n_total = size(ds, 1)
        n_total >= 1 || error("Dataset $(cfg.dataset_key) in $path has no samples")
        nsamples >= 1 || error("nsamples must be >= 1")

        n_use = min(nsamples, n_total)
        max_start = max(1, n_total - n_use + 1)
        start_use = clamp(start_index, 1, max_start)
        stop_use = start_use + n_use - 1

        if start_use != start_index || n_use != nsamples
            @warn "Adjusted observation subset to available dataset bounds" subset = subset_label dataset = path key = cfg.dataset_key requested_start = start_index requested_nsamples = nsamples used_start = start_use used_nsamples = n_use total_samples = n_total
        end

        ds[start_use:stop_use, :, :]
    end

    tensor = permutedims(raw, (3, 2, 1))
    return Float64.(tensor)
end

function compute_global_observables(tensor::Array{Float64,3})
    K, C, N = size(tensor)
    J = C - 1
    J >= 1 || error("Need at least one fast channel")
    A = Array{Float64}(undef, 5, N)
    invK = 1.0 / K
    invJ = 1.0 / J

    @showprogress "Computing global observables phi(t)..." for n in 1:N
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        @inbounds for k in 1:K
            km1 = (k == 1) ? K : (k - 1)
            xk = tensor[k, 1, n]
            xkm1 = tensor[km1, 1, n]
            ysum = 0.0
            y2sum = 0.0
            for j in 1:J
                yjk = tensor[k, j + 1, n]
                ysum += yjk
                y2sum += yjk * yjk
            end
            ybar = ysum * invJ
            y2bar = y2sum * invJ

            s1 += xk
            s2 += xk * xk
            s3 += xk * ybar
            s4 += y2bar
            s5 += xk * xkm1
        end
        A[1, n] = s1 * invK
        A[2, n] = s2 * invK
        A[3, n] = s3 * invK
        A[4, n] = s4 * invK
        A[5, n] = s5 * invK
    end
    return A
end

function select_eval_device(preference::AbstractString)
    try
        d = select_device(preference)
        activate_device!(d)
        return (d, preference)
    catch err
        @warn "Requested device unavailable for score inference; falling back to CPU" requested = preference error = sprint(showerror, err)
        d = CPUDevice()
        activate_device!(d)
        return (d, "CPU")
    end
end

function cholesky_inverse_spd(C::Matrix{Float64}; jitter0::Float64=1e-10, max_tries::Int=8)
    jitter = jitter0
    for _ in 1:max_tries
        try
            F = cholesky(Symmetric(C + jitter * I); check=true)
            return Matrix(F \ I), jitter
        catch err
            if err isa PosDefException
                jitter *= 10
            else
                rethrow(err)
            end
        end
    end
    error("Failed SPD inverse after jitter escalation; last jitter=$jitter")
end

function xcorr_one_sided_unbiased_fft(x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    Kmax::Int)
    n = length(x)
    n == length(y) || error("xcorr inputs must have same length")
    K = min(Kmax, n - 1)

    L = 1
    target = 2 * n - 1
    while L < target
        L <<= 1
    end

    xp = zeros(Float64, L)
    yp = zeros(Float64, L)
    @inbounds xp[1:n] .= x
    @inbounds yp[1:n] .= y
    c = real(ifft(fft(xp) .* conj.(fft(yp))))

    out = Array{Float64}(undef, K + 1)
    @inbounds for k in 0:K
        out[k + 1] = c[k + 1] / (n - k)
    end
    return out
end

function make_l96_workspace(K::Int, J::Int)
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
    )
end

function add_process_noise!(x::Vector{Float64},
    y::Matrix{Float64},
    rng::AbstractRNG,
    sigma::Float64,
    dt::Float64;
    stochastic_x_noise::Bool=false)
    sigma_step = sigma * sqrt(dt)
    sigma_step == 0.0 && return nothing
    @inbounds begin
        if stochastic_x_noise
            for k in eachindex(x)
                x[k] += sigma_step * randn(rng)
            end
        end
        for idx in eachindex(y)
            y[idx] += sigma_step * randn(rng)
        end
    end
    return nothing
end

function accumulate_snapshot_observables!(acc::Vector{Float64},
    x::Vector{Float64},
    y::Matrix{Float64})
    K = length(x)
    J = size(y, 1)
    invK = 1.0 / K
    invJ = 1.0 / J
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    @inbounds for k in 1:K
        km1 = (k == 1) ? K : (k - 1)
        xk = x[k]
        xkm1 = x[km1]
        ysum = 0.0
        y2sum = 0.0
        for j in 1:J
            yjk = y[j, k]
            ysum += yjk
            y2sum += yjk * yjk
        end
        ybar = ysum * invJ
        y2bar = y2sum * invJ
        s1 += xk
        s2 += xk * xk
        s3 += xk * ybar
        s4 += y2bar
        s5 += xk * xkm1
    end
    acc[1] += s1 * invK
    acc[2] += s2 * invK
    acc[3] += s3 * invK
    acc[4] += s4 * invK
    acc[5] += s5 * invK
    return nothing
end

function parse_bool(raw::AbstractString)
    s = lowercase(strip(raw))
    s in ("1", "true", "yes", "y", "on") && return true
    s in ("0", "false", "no", "n", "off") && return false
    error("Could not parse boolean value: $raw")
end

function parse_cli(args::Vector{String})
    out = Dict{String,Any}(
        "params_toml" => RESP_DEFAULT_PARAMS_TOML,
        "methods_override" => "",
        "mode" => "responses",
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a TOML file path")
            out["params_toml"] = args[i+1]
            i += 2
        elseif a == "--methods"
            i == length(args) && error("--methods expects comma-separated values from {gaussian,unet,numerical}")
            out["methods_override"] = args[i+1]
            i += 2
        elseif a == "--mode"
            i == length(args) && error("--mode expects one of {responses,reference}")
            mode = lowercase(strip(args[i+1]))
            mode in ("responses", "reference") || error("Unsupported --mode '$mode' (expected responses|reference)")
            out["mode"] = mode
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return out
end

function parse_method_list(raw::AbstractString)
    s = lowercase(strip(raw))
    isempty(s) && return String[]
    vals = String[]
    for tok in split(s, ",")
        t = strip(tok)
        isempty(t) && continue
        t in ("gaussian", "unet", "numerical") || error("Unknown method '$t' in --methods (expected gaussian,unet,numerical)")
        push!(vals, t)
    end
    return unique(vals)
end

function get_cfg_string(cfg::Dict, key::AbstractString, default::AbstractString)
    haskey(cfg, key) || return default
    return String(cfg[key])
end

function get_cfg_bool(cfg::Dict, key::AbstractString, default::Bool)
    haskey(cfg, key) || return default
    v = cfg[key]
    if v isa Bool
        return v
    elseif v isa AbstractString
        return parse_bool(v)
    else
        error("Expected boolean for key '$key', got $(typeof(v))")
    end
end

function get_cfg_int(cfg::Dict, key::AbstractString, default::Int)
    haskey(cfg, key) || return default
    return Int(cfg[key])
end

function get_cfg_float(cfg::Dict, key::AbstractString, default::Float64)
    haskey(cfg, key) || return default
    return Float64(cfg[key])
end

function get_cfg_float_vec(cfg::Dict, key::AbstractString, default::Vector{Float64})
    haskey(cfg, key) || return copy(default)
    v = cfg[key]
    v isa AbstractVector || error("Expected array for key '$key'")
    out = Float64.(collect(v))
    return out
end

function build_l96_config_from_params(integ::Dict{String,Any},
    dset::Dict{String,Any},
    source::AbstractString)
    path = String(get(dset, "path", ""))
    key = String(get(dset, "key", "timeseries"))
    isempty(path) && error("[dataset].path missing in $source")
    return L96Config(
        Int(get(integ, "K", 36)),
        Int(get(integ, "J", 10)),
        Float64(get(integ, "F", 10.0)),
        Float64(get(integ, "h", 1.0)),
        Float64(get(integ, "c", 10.0)),
        Float64(get(integ, "b", 10.0)),
        Float64(get(integ, "dt", 0.005)),
        Int(get(integ, "save_every", 10)),
        Float64(get(integ, "process_noise_sigma", 0.03)),
        Bool(get(integ, "stochastic_x_noise", false)),
        path,
        key,
    )
end

function read_dataset_integration_attrs(cfg::L96Config)
    isfile(cfg.dataset_path) || error("Dataset file not found for attribute sync: $(cfg.dataset_path)")
    attrs = Dict{String,Any}()
    h5open(cfg.dataset_path, "r") do h5
        haskey(h5, cfg.dataset_key) || error("Dataset key '$(cfg.dataset_key)' not found in $(cfg.dataset_path)")
        dset = h5[cfg.dataset_key]
        ad = attributes(dset)
        for key in ("K", "J", "F", "h", "c", "b", "dt", "save_every", "process_noise_sigma")
            haskey(ad, key) || continue
            attrs[key] = read(ad[key])
        end
    end
    return attrs
end

function sync_l96_config_with_dataset_attrs(cfg::L96Config;
    mode::AbstractString)
    sync_mode = lowercase(strip(mode))
    sync_mode in ("off", "warn", "error", "override") || error("Unsupported dataset attr_sync_mode '$mode' (expected off|warn|error|override)")

    info = Dict{String,Any}(
        "mode" => sync_mode,
        "dataset_path" => cfg.dataset_path,
        "dataset_key" => cfg.dataset_key,
        "applied_override" => false,
        "attrs" => Dict{String,Any}(),
        "mismatches" => Dict{String,Any}(),
    )
    sync_mode == "off" && return cfg, info

    attrs = read_dataset_integration_attrs(cfg)
    info["attrs"] = attrs

    cfg_values = Dict{String,Any}(
        "K" => cfg.K,
        "J" => cfg.J,
        "F" => cfg.F,
        "h" => cfg.h,
        "c" => cfg.c,
        "b" => cfg.b,
        "dt" => cfg.dt,
        "save_every" => cfg.save_every,
        "process_noise_sigma" => cfg.process_noise_sigma,
    )
    int_keys = Set(["K", "J", "save_every"])
    mismatches = Dict{String,Any}()
    for (k, cfg_val) in cfg_values
        haskey(attrs, k) || continue
        raw = attrs[k]
        same = if k in int_keys
            Int(raw) == Int(cfg_val)
        else
            isapprox(Float64(raw), Float64(cfg_val); rtol=1e-12, atol=1e-12)
        end
        if !same
            mismatches[k] = Dict("config" => cfg_val, "dataset" => raw)
        end
    end
    info["mismatches"] = mismatches
    isempty(mismatches) && return cfg, info

    msg = "Integration parameters and dataset attributes differ"
    if sync_mode == "error"
        error("$msg. Set [dataset].attr_sync_mode=\"override\" (recommended) or align [integration] with dataset attrs.")
    elseif sync_mode == "warn"
        @warn msg mismatches = mismatches dataset = cfg.dataset_path
        return cfg, info
    end

    @warn "Overriding integration parameters from dataset attributes to keep GFDT and numerical integration consistent" mismatches = mismatches dataset = cfg.dataset_path
    new_cfg = L96Config(
        haskey(attrs, "K") ? Int(attrs["K"]) : cfg.K,
        haskey(attrs, "J") ? Int(attrs["J"]) : cfg.J,
        haskey(attrs, "F") ? Float64(attrs["F"]) : cfg.F,
        haskey(attrs, "h") ? Float64(attrs["h"]) : cfg.h,
        haskey(attrs, "c") ? Float64(attrs["c"]) : cfg.c,
        haskey(attrs, "b") ? Float64(attrs["b"]) : cfg.b,
        haskey(attrs, "dt") ? Float64(attrs["dt"]) : cfg.dt,
        haskey(attrs, "save_every") ? Int(attrs["save_every"]) : cfg.save_every,
        haskey(attrs, "process_noise_sigma") ? Float64(attrs["process_noise_sigma"]) : cfg.process_noise_sigma,
        cfg.stochastic_x_noise,
        cfg.dataset_path,
        cfg.dataset_key,
    )
    info["applied_override"] = true
    return new_cfg, info
end

function load_response_params(path::AbstractString)
    isfile(path) || error("Response-parameter TOML not found: $path")
    raw = TOML.parsefile(path)

    paths_cfg = get(raw, "paths", Dict{String,Any}())
    gfdt_cfg = get(raw, "gfdt", Dict{String,Any}())
    num_cfg = get(raw, "numerical", Dict{String,Any}())
    corr_cfg = get(raw, "correction", Dict{String,Any}())
    meth_cfg = get(raw, "methods", Dict{String,Any}())
    res_cfg = get(raw, "resources", Dict{String,Any}())
    ref_cfg = get(raw, "reference", Dict{String,Any}())
    integ_cfg = get(raw, "integration", Dict{String,Any}())
    dset_cfg = get(raw, "dataset", Dict{String,Any}())

    methods = Dict(
        "gaussian" => get_cfg_bool(meth_cfg, "gaussian", true),
        "unet" => get_cfg_bool(meth_cfg, "unet", true),
        "numerical" => get_cfg_bool(meth_cfg, "numerical", true),
    )

    # Check if user wants impulse response (default false)
    compute_impulse = get_cfg_bool(meth_cfg, "calculate_impulse_response", false)

    h_abs = get_cfg_float_vec(num_cfg, "h_abs", collect(Float64.(RESP_DEFAULT_H_ABS)))
    length(h_abs) == 4 || error("[numerical].h_abs must contain exactly 4 values ordered as [F,h,c,b]")
    ref_h_abs = get_cfg_float_vec(ref_cfg, "h_abs", collect(Float64.(RESP_DEFAULT_H_ABS)))
    length(ref_h_abs) == 4 || error("[reference].h_abs must contain exactly 4 values ordered as [F,h,c,b]")

    run_prefix = get_cfg_string(paths_cfg, "run_prefix", "run_")
    startswith(run_prefix, "run_") || @warn "Using non-standard run prefix" run_prefix = run_prefix
    ref_num_method = lowercase(get_cfg_string(ref_cfg, "numerical_method", RESP_DEFAULT_REFERENCE_NUM_METHOD))
    ref_num_method in ("tangent", "finite_difference") || error("[reference].numerical_method must be tangent or finite_difference")

    cfg = if !isempty(integ_cfg) || !isempty(dset_cfg)
        build_l96_config_from_params(Dict{String,Any}(integ_cfg), Dict{String,Any}(dset_cfg), path)
    else
        # Backward-compatible fallback for older parameter files.
        integration_toml = get_cfg_string(paths_cfg, "integration_toml", RESP_DEFAULT_INTEGRATION_TOML)
        @warn "Using legacy [paths].integration_toml fallback. Move these settings to [integration] and [dataset] in $path." integration_toml = integration_toml
        load_l96_config(abspath(integration_toml))
    end

    return Dict{String,Any}(
        "params_toml" => abspath(path),
        "compute_impulse" => compute_impulse,
        "run_dir" => get_cfg_string(paths_cfg, "run_dir", RESP_DEFAULT_RUN_DIR),
        "checkpoint_path" => get_cfg_string(paths_cfg, "checkpoint_path", ""),
        "checkpoint_epoch" => get_cfg_int(paths_cfg, "checkpoint_epoch", -1),
        "l96_config" => cfg,
        "output_root" => get_cfg_string(paths_cfg, "output_root", RESP_DEFAULT_OUTPUT_ROOT),
        "run_prefix" => run_prefix,
        "methods" => methods,
        "gfdt_nsamples" => get_cfg_int(gfdt_cfg, "nsamples", RESP_DEFAULT_GFDT_NSAMPLES),
        "gfdt_start_index" => get_cfg_int(gfdt_cfg, "start_index", RESP_DEFAULT_GFDT_START_INDEX),
        "response_tmax" => get_cfg_float(gfdt_cfg, "response_tmax", RESP_DEFAULT_TMAX),
        "score_batch_size" => get_cfg_int(gfdt_cfg, "score_batch_size", RESP_DEFAULT_SCORE_BATCH_SIZE),
        "score_device" => get_cfg_string(gfdt_cfg, "score_device", RESP_DEFAULT_SCORE_DEVICE),
        "score_forward_mode" => get_cfg_string(gfdt_cfg, "score_forward_mode", RESP_DEFAULT_SCORE_FORWARD_MODE),
        "mean_center" => get_cfg_bool(gfdt_cfg, "mean_center", RESP_DEFAULT_MEAN_CENTER),
        "dataset_attr_sync_mode" => lowercase(get_cfg_string(dset_cfg, "attr_sync_mode", RESP_DEFAULT_DATASET_ATTR_SYNC_MODE)),
        "numerical_ensembles" => get_cfg_int(num_cfg, "ensembles", RESP_DEFAULT_NUM_ENSEMBLES),
        "numerical_start_index" => get_cfg_int(num_cfg, "start_index", RESP_DEFAULT_NUM_START_INDEX),
        "numerical_seed_base" => get_cfg_int(num_cfg, "seed_base", RESP_DEFAULT_NUM_SEED_BASE),
        "numerical_method" => get_cfg_string(num_cfg, "method", RESP_DEFAULT_NUM_METHOD),
        "h_rel" => get_cfg_float(num_cfg, "h_rel", RESP_DEFAULT_H_REL),
        "h_abs" => h_abs,
        "correction_ridge" => get_cfg_float(corr_cfg, "ridge", RESP_DEFAULT_CORR_RIDGE),
        "correction_max_deviation" => get_cfg_float(corr_cfg, "max_deviation", RESP_DEFAULT_CORR_MAX_DEVIATION),
        "auto_gpu_max_mem_fraction" => get_cfg_float(res_cfg, "auto_gpu_max_mem_fraction", RESP_DEFAULT_AUTO_GPU_MAX_MEM_FRACTION),
        "auto_gpu_max_utilization" => get_cfg_float(res_cfg, "auto_gpu_max_utilization", RESP_DEFAULT_AUTO_GPU_MAX_UTILIZATION),
        "auto_gpu_min_free_mb" => get_cfg_float(res_cfg, "auto_gpu_min_free_mb", RESP_DEFAULT_AUTO_GPU_MIN_FREE_MB),
        "reference_cache_root" => get_cfg_string(ref_cfg, "cache_root", RESP_DEFAULT_REFERENCE_CACHE_ROOT),
        "reference_force_regenerate" => get_cfg_bool(ref_cfg, "force_regenerate", false),
        "reference_gfdt_nsamples" => get_cfg_int(ref_cfg, "gfdt_nsamples", RESP_DEFAULT_REFERENCE_GFDT_NSAMPLES),
        "reference_gfdt_start_index" => get_cfg_int(ref_cfg, "gfdt_start_index", RESP_DEFAULT_REFERENCE_GFDT_START_INDEX),
        "reference_numerical_ensembles" => get_cfg_int(ref_cfg, "numerical_ensembles", RESP_DEFAULT_REFERENCE_NUM_ENSEMBLES),
        "reference_numerical_start_index" => get_cfg_int(ref_cfg, "numerical_start_index", RESP_DEFAULT_REFERENCE_NUM_START_INDEX),
        "reference_numerical_method" => ref_num_method,
        "reference_h_rel" => get_cfg_float(ref_cfg, "h_rel", RESP_DEFAULT_H_REL),
        "reference_h_abs" => ref_h_abs,
        "reference_numerical_seed_base" => get_cfg_int(ref_cfg, "numerical_seed_base", RESP_DEFAULT_REFERENCE_NUM_SEED_BASE),
        "reference_tmax" => get_cfg_float(ref_cfg, "tmax", RESP_DEFAULT_REFERENCE_TMAX),
        "reference_mean_center" => get_cfg_bool(ref_cfg, "mean_center", RESP_DEFAULT_MEAN_CENTER),
    )
end

function allocate_response_run_dir(root::AbstractString; run_prefix::AbstractString="run_")
    mkpath(root)
    pref_re = replace(run_prefix, r"([\\.^$|()?*+{}\[\]])" => s"\\\1")
    max_id = 0
    for name in readdir(root)
        m = match(Regex("^" * pref_re * "(\\d{3})\$"), name)
        m === nothing && continue
        max_id = max(max_id, parse(Int, m.captures[1]))
    end
    run_id = max_id + 1
    out_dir = joinpath(root, @sprintf("%s%03d", run_prefix, run_id))
    mkpath(out_dir)
    return out_dir, run_id
end

function choose_score_device(preference::AbstractString;
    max_mem_fraction::Float64,
    max_utilization::Float64,
    min_free_mb::Float64)
    pref = lowercase(strip(preference))
    pref in ("", "auto") || return (preference, "user-specified")
    if Sys.which("nvidia-smi") === nothing
        return ("CPU", "auto->cpu (nvidia-smi unavailable)")
    end
    lines = try
        split(chomp(read(`nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`, String)), '\n')
    catch
        String[]
    end
    isempty(lines) && return ("CPU", "auto->cpu (cannot query GPU status)")

    candidates = Tuple{Int,Float64,Float64,Float64,Float64}[]
    # tuple = (idx, used_mb, total_mb, util_pct, score)
    for line in lines
        cols = split(strip(line), ",")
        length(cols) < 4 && continue
        idx = try
            parse(Int, strip(cols[1]))
        catch
            continue
        end
        used = try
            parse(Float64, strip(cols[2]))
        catch
            continue
        end
        total = try
            parse(Float64, strip(cols[3]))
        catch
            continue
        end
        util = try
            parse(Float64, strip(cols[4]))
        catch
            continue
        end
        total > 0 || continue
        free = total - used
        mem_frac = used / total
        if mem_frac <= max_mem_fraction && util <= max_utilization && free >= min_free_mb
            score = mem_frac + util / 100.0
            push!(candidates, (idx, used, total, util, score))
        end
    end

    if isempty(candidates)
        return ("CPU", @sprintf("auto->cpu (no GPU below thresholds mem<=%.2f util<=%.1f free>=%.0fMB)", max_mem_fraction, max_utilization, min_free_mb))
    end
    best = candidates[argmin(last.(candidates))]
    idx = best[1]
    return ("GPU:$idx", @sprintf("auto->GPU:%d (used=%.0fMB total=%.0fMB util=%.1f%%)", idx, best[2], best[3], best[4]))
end

function extract_correction_stats_dict(stats)
    return Dict(
        "ridge" => stats.ridge,
        "solver" => stats.solver,
        "pre_identity_rmse" => stats.pre_identity_rmse,
        "post_identity_rmse" => stats.post_identity_rmse,
        "post_identity_rmse_direct" => stats.post_identity_rmse_direct,
        "post_identity_rmse_pinv" => stats.post_identity_rmse_pinv,
        "max_deviation" => stats.max_deviation,
        "post_deviation" => stats.post_deviation,
        "post_deviation_direct" => stats.post_deviation_direct,
        "post_deviation_pinv" => stats.post_deviation_pinv,
        "cond_regularized" => stats.cond_regularized,
    )
end

function resolve_checkpoint_path(run_dir::AbstractString, checkpoint_override::AbstractString, checkpoint_epoch::Int=-1)
    ckpt = strip(checkpoint_override)
    if !isempty(ckpt)
        path = abspath(ckpt)
        isfile(path) || error("Checkpoint not found: $path")
        return path
    end

    if checkpoint_epoch > 0
        epoch_str = @sprintf("score_model_epoch_%04d.bson", checkpoint_epoch)
        path = joinpath(run_dir, "model", epoch_str)
        if !isfile(path)
            epoch_str3 = @sprintf("score_model_epoch_%03d.bson", checkpoint_epoch)
            path = joinpath(run_dir, "model", epoch_str3)
        end
        isfile(path) || error("Checkpoint not found for epoch $checkpoint_epoch in $run_dir/model")
        return abspath(path)
    end

    epoch20 = joinpath(run_dir, "model", "score_model_epoch_0020.bson")
    if isfile(epoch20)
        return abspath(epoch20)
    end

    if isfile(RESP_DEFAULT_CHECKPOINT) && abspath(run_dir) == abspath(RESP_DEFAULT_RUN_DIR)
        return abspath(RESP_DEFAULT_CHECKPOINT)
    end

    best = pick_best_checkpoint(run_dir)
    return abspath(best.checkpoint_path)
end

function checkpoint_epoch(path::AbstractString)
    m = match(r"epoch[_-]?0*([0-9]+)", lowercase(basename(path)))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

@inline l96_linidx(j::Int, k::Int, J::Int) = (k - 1) * J + j

@inline function l96_jk_from_lin(ell::Int, J::Int)
    kk = (ell - 1) ÷ J + 1
    jj = (ell - 1) % J + 1
    return jj, kk
end

@inline l96_wrap_lin(ell::Int, K::Int, J::Int) = mod1idx(ell, K * J)

function compute_G_from_score_batch!(G::Matrix{Float64},
    tensor::Array{Float64,3},
    score_phys::Array{Float64,3},
    theta::NTuple{4,Float64},
    out_start::Int)
    K, C, B = size(tensor)
    J = C - 1
    _, h, c, b = theta

    @inbounds for ib in 1:B
        gF = 0.0
        sum_ybar_sx = 0.0
        sum_x_sy = 0.0
        sum_uc_sy = 0.0
        sum_adv_sy = 0.0

        for k in 1:K
            xk = tensor[k, 1, ib]
            sx = score_phys[k, 1, ib]
            gF -= sx

            ysum = 0.0
            for j in 1:J
                ysum += tensor[k, j+1, ib]
            end
            ybar = ysum / J
            sum_ybar_sx += ybar * sx

            for j in 1:J
                ell = l96_linidx(j, k, J)
                ellp1 = l96_wrap_lin(ell + 1, K, J)
                ellm1 = l96_wrap_lin(ell - 1, K, J)
                ellp2 = l96_wrap_lin(ell + 2, K, J)
                jp1, kp1 = l96_jk_from_lin(ellp1, J)
                jm1, km1 = l96_jk_from_lin(ellm1, J)
                jp2, kp2 = l96_jk_from_lin(ellp2, J)

                yjm1 = tensor[km1, jm1+1, ib]
                yjp1 = tensor[kp1, jp1+1, ib]
                yjp2 = tensor[kp2, jp2+1, ib]
                yj = tensor[k, j+1, ib]
                sy = score_phys[k, j+1, ib]

                adv = yjp1 * (yjp2 - yjm1)
                sum_x_sy += xk * sy
                sum_uc_sy += (-b * adv - yj + (h / J) * xk) * sy
                sum_adv_sy += adv * sy
            end
        end

        n = out_start + ib - 1

        g1x = gF
        g1y = 0.0
        g1c = 0.0

        g2x = c * sum_ybar_sx
        g2y = -(c / J) * sum_x_sy
        g2c = 0.0

        g3x = h * sum_ybar_sx
        g3y = -sum_uc_sy
        g3c = K * J

        g4x = 0.0
        g4y = c * sum_adv_sy
        g4c = 0.0

        G[1, n] = g1x + g1y + g1c
        G[2, n] = g2x + g2y + g2c
        G[3, n] = g3x + g3y + g3c
        G[4, n] = g4x + g4y + g4c
    end
    return nothing
end

function l96_two_scale_drift_twisted!(dx::AbstractVector{Float64},
    dy::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    y::AbstractMatrix{Float64},
    theta::NTuple{4,Float64},
    K::Int,
    J::Int)
    F, h, c, b = theta
    coupling_scale = h * c / J

    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)
        coupling = coupling_scale * sum(@view y[:, k])
        dx[k] = x[km1] * (x[kp1] - x[km2]) - x[k] + F - coupling
    end

    @inbounds for k in 1:K
        xk_term = coupling_scale * x[k]
        for j in 1:J
            ell = l96_linidx(j, k, J)
            ellp1 = l96_wrap_lin(ell + 1, K, J)
            ellm1 = l96_wrap_lin(ell - 1, K, J)
            ellp2 = l96_wrap_lin(ell + 2, K, J)
            jp1, kp1 = l96_jk_from_lin(ellp1, J)
            jm1, km1 = l96_jk_from_lin(ellm1, J)
            jp2, kp2 = l96_jk_from_lin(ellp2, J)

            y_p1 = y[jp1, kp1]
            y_m1 = y[jm1, km1]
            y_p2 = y[jp2, kp2]
            dy[j, k] = c * b * y_p1 * (y_m1 - y_p2) - c * y[j, k] + xk_term
        end
    end
    return nothing
end

function rk4_step_l96_twisted!(x::Vector{Float64},
    y::Matrix{Float64},
    dt::Float64,
    ws,
    theta::NTuple{4,Float64},
    K::Int,
    J::Int)
    l96_two_scale_drift_twisted!(ws.dx1, ws.dy1, x, y, theta, K, J)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    l96_two_scale_drift_twisted!(ws.dx2, ws.dy2, ws.xtmp, ws.ytmp, theta, K, J)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    l96_two_scale_drift_twisted!(ws.dx3, ws.dy3, ws.xtmp, ws.ytmp, theta, K, J)

    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    l96_two_scale_drift_twisted!(ws.dx4, ws.dy4, ws.xtmp, ws.ytmp, theta, K, J)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    return nothing
end

function build_score_correction(M::Matrix{Float64};
    ridge::Float64,
    max_deviation::Float64)
    D = size(M, 1)
    size(M, 2) == D || error("Correction matrix must be square")
    Iden = Matrix{Float64}(I, D, D)
    pre_err = norm(M .- Iden) / sqrt(Float64(D))

    M_reg = copy(M)
    @inbounds for i in 1:D
        M_reg[i, i] += ridge
    end
    cond_reg = try
        cond(M_reg)
    catch
        NaN
    end

    T_direct = M_reg \ Iden
    dev_direct = norm(T_direct .- Iden) / sqrt(Float64(D))
    post_direct = norm(M * T_direct .- Iden) / sqrt(Float64(D))

    T_pinv = pinv(M_reg)
    dev_pinv = norm(T_pinv .- Iden) / sqrt(Float64(D))
    post_pinv = norm(M * T_pinv .- Iden) / sqrt(Float64(D))

    best_solver = "direct_solve"
    best_T = T_direct
    best_post = post_direct
    best_dev = dev_direct
    if isfinite(post_pinv) && (!isfinite(best_post) || post_pinv < best_post)
        best_solver = "pinv"
        best_T = T_pinv
        best_post = post_pinv
        best_dev = dev_pinv
    end

    if isfinite(best_post) && best_post < pre_err && isfinite(best_dev) && best_dev <= max_deviation
        Tcorr = best_T
        solver_used = best_solver
        post_err = best_post
        post_dev = best_dev
    else
        Tcorr = Iden
        solver_used = "identity_fallback"
        post_err = pre_err
        post_dev = 0.0
    end

    stats = (
        ridge=ridge,
        max_deviation=max_deviation,
        solver=solver_used,
        pre_identity_rmse=pre_err,
        post_identity_rmse=post_err,
        post_identity_rmse_direct=post_direct,
        post_identity_rmse_pinv=post_pinv,
        pre_deviation=0.0,
        post_deviation=post_dev,
        post_deviation_direct=dev_direct,
        post_deviation_pinv=dev_pinv,
        cond_regularized=cond_reg,
    )
    return Tcorr, stats
end

function compute_gaussian_conjugates(tensor::Array{Float64,3},
    theta::NTuple{4,Float64};
    batch_size::Int,
    correction_ridge::Float64,
    correction_max_deviation::Float64)
    K, C, N = size(tensor)
    D = K * C
    Xflat = reshape(tensor, D, N)
    mu = vec(mean(Xflat; dims=2))
    Xc = Xflat .- mu
    Cmat = (Xc * transpose(Xc)) / max(N - 1, 1)
    Cinv, jitter = cholesky_inverse_spd(Matrix(Cmat))

    @info "Gaussian score setup complete" dim = D samples = N covariance_jitter = jitter

    G_raw = zeros(Float64, 4, N)
    M_acc = zeros(Float64, D, D)

    @showprogress "Gaussian pass 1 (raw G + correction matrix)..." for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        idx = start:stop

        batch = Array(@view tensor[:, :, idx])
        x_chunk = reshape(batch, D, b)
        x_chunk_c = x_chunk .- mu
        score_flat = -(Cinv * x_chunk_c)
        score_tensor = reshape(score_flat, K, C, b)

        compute_G_from_score_batch!(G_raw, batch, score_tensor, theta, start)
        BLAS.gemm!('N', 'T', -1.0, x_chunk, score_flat, 1.0, M_acc)
    end

    M_raw = M_acc ./ max(N, 1)
    Tcorr, corr_stats = build_score_correction(
        M_raw;
        ridge=correction_ridge,
        max_deviation=correction_max_deviation,
    )
    At = transpose(Tcorr)

    G_corr = zeros(Float64, 4, N)
    @showprogress "Gaussian pass 2 (corrected G)..." for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        idx = start:stop

        batch = Array(@view tensor[:, :, idx])
        x_chunk = reshape(batch, D, b)
        x_chunk_c = x_chunk .- mu
        score_flat = -(Cinv * x_chunk_c)

        score_corr_flat = Matrix{Float64}(undef, D, b)
        mul!(score_corr_flat, At, score_flat)
        score_corr = reshape(score_corr_flat, K, C, b)
        compute_G_from_score_batch!(G_corr, batch, score_corr, theta, start)
    end

    return (
        G_raw=G_raw,
        G_corr=G_corr,
        correction_stats=corr_stats,
        correction_matrix=Tcorr,
        raw_identity_matrix=M_raw,
    )
end

Base.@kwdef struct UnetInferenceBundle
    model
    device
    device_name::String
    sigma_train::Float32
    mean_lc::Array{Float32,2}
    std_lc::Array{Float32,2}
    std_lc64::Array{Float64,2}
end

function load_unet_inference_bundle(checkpoint_path::AbstractString;
    device_pref::AbstractString,
    forward_mode::AbstractString)
    contents = BSON.load(checkpoint_path)
    haskey(contents, :model) || error("Checkpoint missing :model ($checkpoint_path)")
    haskey(contents, :stats) || error("Checkpoint missing :stats ($checkpoint_path)")
    haskey(contents, :trainer_cfg) || error("Checkpoint missing :trainer_cfg ($checkpoint_path)")

    model = contents[:model]
    stats = contents[:stats]
    trainer_cfg = contents[:trainer_cfg]
    sigma_train = Float32(getproperty(trainer_cfg, :sigma))

    mean_lc = Float32.(permutedims(stats.mean, (2, 1)))
    std_lc = Float32.(permutedims(stats.std, (2, 1)))
    std_lc64 = Float64.(std_lc)

    device, device_name = select_eval_device(device_pref)
    model_dev = move_model(model, device)
    if lowercase(forward_mode) == "train"
        Flux.trainmode!(model_dev)
    else
        Flux.testmode!(model_dev)
    end

    @info "UNet inference bundle loaded" checkpoint = checkpoint_path sigma_train = sigma_train device = device_name forward_mode = forward_mode
    return UnetInferenceBundle(
        model=model_dev,
        device=device,
        device_name=device_name,
        sigma_train=sigma_train,
        mean_lc=mean_lc,
        std_lc=std_lc,
        std_lc64=std_lc64,
    )
end

function compute_unet_score_batch(bundle::UnetInferenceBundle,
    batch_phys::Array{Float64,3})
    K, C, b = size(batch_phys)
    batch_norm_f32 = Array{Float32,3}(undef, K, C, b)
    @inbounds for ib in 1:b, c in 1:C, k in 1:K
        batch_norm_f32[k, c, ib] = (Float32(batch_phys[k, c, ib]) - bundle.mean_lc[k, c]) / bundle.std_lc[k, c]
    end

    score_norm = if is_gpu(bundle.device)
        dev_batch = move_array(batch_norm_f32, bundle.device)
        Array(score_from_model(bundle.model, dev_batch, bundle.sigma_train))
    else
        score_from_model(bundle.model, batch_norm_f32, bundle.sigma_train)
    end

    score_phys = Array{Float64,3}(undef, K, C, b)
    @inbounds for ib in 1:b, c in 1:C, k in 1:K
        score_phys[k, c, ib] = Float64(score_norm[k, c, ib]) / bundle.std_lc64[k, c]
    end
    return score_phys
end

function compute_unet_conjugates(tensor::Array{Float64,3},
    theta::NTuple{4,Float64},
    checkpoint_path::AbstractString;
    batch_size::Int,
    device_pref::AbstractString,
    forward_mode::AbstractString,
    correction_ridge::Float64,
    correction_max_deviation::Float64)
    bundle = load_unet_inference_bundle(checkpoint_path; device_pref=device_pref, forward_mode=forward_mode)
    K, C, N = size(tensor)
    D = K * C
    mean64 = Float64.(bundle.mean_lc)
    std64 = bundle.std_lc64
    invstd64 = similar(std64)
    @. invstd64 = 1.0 / std64

    G_raw = zeros(Float64, 4, N)
    M_acc_norm = zeros(Float64, D, D)

    @showprogress "UNet pass 1 (raw G + correction matrix)..." for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        idx = start:stop

        batch = Array(@view tensor[:, :, idx])
        score_phys = compute_unet_score_batch(bundle, batch)

        compute_G_from_score_batch!(G_raw, batch, score_phys, theta, start)

        z_chunk = Matrix{Float64}(undef, D, b)
        score_norm_flat = Matrix{Float64}(undef, D, b)
        @inbounds for ib in 1:b, c in 1:C, k in 1:K
            d = k + (c - 1) * K
            z_chunk[d, ib] = (batch[k, c, ib] - mean64[k, c]) * invstd64[k, c]
            score_norm_flat[d, ib] = score_phys[k, c, ib] * std64[k, c]
        end
        BLAS.gemm!('N', 'T', -1.0, z_chunk, score_norm_flat, 1.0, M_acc_norm)
    end

    M_raw = M_acc_norm ./ max(N, 1)
    Tcorr, corr_stats = build_score_correction(
        M_raw;
        ridge=correction_ridge,
        max_deviation=correction_max_deviation,
    )
    At = transpose(Tcorr)

    G_corr = zeros(Float64, 4, N)
    @showprogress "UNet pass 2 (corrected G)..." for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        idx = start:stop

        batch = Array(@view tensor[:, :, idx])
        score_phys = compute_unet_score_batch(bundle, batch)
        score_norm_flat = Matrix{Float64}(undef, D, b)
        @inbounds for ib in 1:b, c in 1:C, k in 1:K
            d = k + (c - 1) * K
            score_norm_flat[d, ib] = score_phys[k, c, ib] * std64[k, c]
        end

        score_corr_norm_flat = Matrix{Float64}(undef, D, b)
        mul!(score_corr_norm_flat, At, score_norm_flat)

        score_corr_phys_flat = Matrix{Float64}(undef, D, b)
        @inbounds for ib in 1:b, c in 1:C, k in 1:K
            d = k + (c - 1) * K
            score_corr_phys_flat[d, ib] = score_corr_norm_flat[d, ib] * invstd64[k, c]
        end
        score_corr = reshape(score_corr_phys_flat, K, C, b)
        compute_G_from_score_batch!(G_corr, batch, score_corr, theta, start)
    end

    return (
        G_raw=G_raw,
        G_corr=G_corr,
        correction_stats=corr_stats,
        correction_matrix=Tcorr,
        raw_identity_matrix=M_raw,
    )
end

function build_gfdt_response(A::Matrix{Float64},
    G::Matrix{Float64},
    delta_t::Float64,
    n_lags::Int;
    mean_center::Bool=true)
    m, N = size(A)
    p, N2 = size(G)
    N == N2 || error("A/G time length mismatch")
    n_lags = min(n_lags, N - 1)
    n_lags >= 1 || error("Need at least one lag to build responses")

    Ause = mean_center ? (A .- mean(A; dims=2)) : A
    Guse = mean_center ? (G .- mean(G; dims=2)) : G

    C = zeros(Float64, m, p, n_lags + 1)
    R = zeros(Float64, m, p, n_lags + 1)

    @info "Building GFDT response kernels" observables = m parameters = p samples = N lags = n_lags delta_t = delta_t mean_center = mean_center threads = nthreads()
    Threads.@threads for pair in 1:(m*p)
        i = (pair - 1) ÷ p + 1
        j = (pair - 1) % p + 1

        ai = vec(@view Ause[i, :])
        gj = vec(@view Guse[j, :])
        cpos = xcorr_one_sided_unbiased_fft(ai, gj, n_lags)

        @views C[i, j, :] .= cpos
        R[i, j, 1] = 0.0
        acc = 0.0
        @inbounds for lag in 1:n_lags
            acc += cpos[lag]
            R[i, j, lag+1] = delta_t * acc
        end
    end

    times = collect(0:n_lags) .* delta_t
    # Return both C (correlation) and R (integrated response)
    return C, R, times
end

function snapshot_to_xy!(x0::Vector{Float64},
    y0::Matrix{Float64},
    tensor::Array{Float64,3},
    idx::Int)
    K = length(x0)
    J = size(y0, 1)
    @inbounds for k in 1:K
        x0[k] = tensor[k, 1, idx]
        for j in 1:J
            y0[j, k] = tensor[k, j+1, idx]
        end
    end
    return nothing
end

function simulate_observable_series!(out::Matrix{Float64},
    theta::NTuple{4,Float64},
    x0::Vector{Float64},
    y0::Matrix{Float64},
    cfg::L96Config,
    n_lags::Int,
    ws,
    rng::AbstractRNG,
    x::Vector{Float64},
    y::Matrix{Float64},
    acc::Vector{Float64})
    copyto!(x, x0)
    copyto!(y, y0)

    fill!(acc, 0.0)
    accumulate_snapshot_observables!(acc, x, y)
    @views out[:, 1] .= acc

    @inbounds for lag in 1:n_lags
        for _ in 1:cfg.save_every
            rk4_step_l96_twisted!(x, y, cfg.dt, ws, theta, cfg.K, cfg.J)
            add_process_noise!(x, y, rng, cfg.process_noise_sigma, cfg.dt; stochastic_x_noise=cfg.stochastic_x_noise)
        end
        fill!(acc, 0.0)
        accumulate_snapshot_observables!(acc, x, y)
        @views out[:, lag+1] .= acc
    end
    return nothing
end

function make_l96_tangent_workspace(K::Int, J::Int, P::Int)
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        detax1=zeros(Float64, K, P), detax2=zeros(Float64, K, P), detax3=zeros(Float64, K, P), detax4=zeros(Float64, K, P),
        detay1=zeros(Float64, J, K, P), detay2=zeros(Float64, J, K, P), detay3=zeros(Float64, J, K, P), detay4=zeros(Float64, J, K, P),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
        etax_tmp=zeros(Float64, K, P), etay_tmp=zeros(Float64, J, K, P),
    )
end

function l96_two_scale_drift_tangent_twisted!(dx::Vector{Float64},
    dy::Matrix{Float64},
    detax::Matrix{Float64},
    detay::Array{Float64,3},
    x::Vector{Float64},
    y::Matrix{Float64},
    etax::Matrix{Float64},
    etay::Array{Float64,3},
    theta::NTuple{4,Float64},
    K::Int,
    J::Int)
    _, h, c, b = theta
    P = size(etax, 2)
    coupling_scale = h * c / J
    ch_over_j = c / J
    h_over_j = h / J

    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)

        ysum = 0.0
        for j in 1:J
            ysum += y[j, k]
        end
        dx[k] = x[km1] * (x[kp1] - x[km2]) - x[k] + theta[1] - coupling_scale * ysum

        for ip in 1:P
            etay_sum = 0.0
            for j in 1:J
                etay_sum += etay[j, k, ip]
            end

            upx = if ip == 1
                1.0
            elseif ip == 2
                -ch_over_j * ysum
            elseif ip == 3
                -h_over_j * ysum
            else
                0.0
            end

            detax[k, ip] =
                (x[kp1] - x[km2]) * etax[km1, ip] +
                x[km1] * (etax[kp1, ip] - etax[km2, ip]) -
                etax[k, ip] -
                coupling_scale * etay_sum +
                upx
        end
    end

    @inbounds for k in 1:K
        xk_term = coupling_scale * x[k]
        for j in 1:J
            ell = l96_linidx(j, k, J)
            ellp1 = l96_wrap_lin(ell + 1, K, J)
            ellm1 = l96_wrap_lin(ell - 1, K, J)
            ellp2 = l96_wrap_lin(ell + 2, K, J)
            jp1, kp1 = l96_jk_from_lin(ellp1, J)
            jm1, km1 = l96_jk_from_lin(ellm1, J)
            jp2, kp2 = l96_jk_from_lin(ellp2, J)

            y_p1 = y[jp1, kp1]
            y_m1 = y[jm1, km1]
            y_p2 = y[jp2, kp2]
            y_j = y[j, k]
            adv = y_p1 * (y_m1 - y_p2)

            dy[j, k] = c * b * adv - c * y_j + xk_term

            for ip in 1:P
                etay_p1 = etay[jp1, kp1, ip]
                etay_m1 = etay[jm1, km1, ip]
                etay_p2 = etay[jp2, kp2, ip]
                adv_tan = etay_p1 * (y_m1 - y_p2) + y_p1 * (etay_m1 - etay_p2)

                upy = if ip == 1
                    0.0
                elseif ip == 2
                    ch_over_j * x[k]
                elseif ip == 3
                    b * adv - y_j + h_over_j * x[k]
                else
                    c * adv
                end

                detay[j, k, ip] =
                    c * b * adv_tan -
                    c * etay[j, k, ip] +
                    coupling_scale * etax[k, ip] +
                    upy
            end
        end
    end
    return nothing
end

function rk4_step_l96_tangent_twisted!(x::Vector{Float64},
    y::Matrix{Float64},
    etax::Matrix{Float64},
    etay::Array{Float64,3},
    dt::Float64,
    ws,
    theta::NTuple{4,Float64},
    K::Int,
    J::Int)
    l96_two_scale_drift_tangent_twisted!(ws.dx1, ws.dy1, ws.detax1, ws.detay1, x, y, etax, etay, theta, K, J)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    @. ws.etax_tmp = etax + 0.5 * dt * ws.detax1
    @. ws.etay_tmp = etay + 0.5 * dt * ws.detay1
    l96_two_scale_drift_tangent_twisted!(ws.dx2, ws.dy2, ws.detax2, ws.detay2, ws.xtmp, ws.ytmp, ws.etax_tmp, ws.etay_tmp, theta, K, J)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    @. ws.etax_tmp = etax + 0.5 * dt * ws.detax2
    @. ws.etay_tmp = etay + 0.5 * dt * ws.detay2
    l96_two_scale_drift_tangent_twisted!(ws.dx3, ws.dy3, ws.detax3, ws.detay3, ws.xtmp, ws.ytmp, ws.etax_tmp, ws.etay_tmp, theta, K, J)

    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    @. ws.etax_tmp = etax + dt * ws.detax3
    @. ws.etay_tmp = etay + dt * ws.detay3
    l96_two_scale_drift_tangent_twisted!(ws.dx4, ws.dy4, ws.detax4, ws.detay4, ws.xtmp, ws.ytmp, ws.etax_tmp, ws.etay_tmp, theta, K, J)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    @. etax = etax + (dt / 6.0) * (ws.detax1 + 2.0 * ws.detax2 + 2.0 * ws.detax3 + ws.detax4)
    @. etay = etay + (dt / 6.0) * (ws.detay1 + 2.0 * ws.detay2 + 2.0 * ws.detay3 + ws.detay4)
    return nothing
end

function compute_snapshot_observable_sensitivities!(out::Matrix{Float64},
    x::Vector{Float64},
    y::Matrix{Float64},
    etax::Matrix{Float64},
    etay::Array{Float64,3})
    fill!(out, 0.0)
    K = length(x)
    J = size(y, 1)
    P = size(etax, 2)
    invK = 1.0 / K
    invJ = 1.0 / J
    ey_sum = zeros(Float64, P)
    ey_dot = zeros(Float64, P)

    @inbounds for k in 1:K
        km1 = (k == 1) ? K : (k - 1)
        xk = x[k]
        xkm1 = x[km1]

        fill!(ey_sum, 0.0)
        fill!(ey_dot, 0.0)
        ysum = 0.0
        for j in 1:J
            yjk = y[j, k]
            ysum += yjk
            for ip in 1:P
                e = etay[j, k, ip]
                ey_sum[ip] += e
                ey_dot[ip] += yjk * e
            end
        end
        ybar = ysum * invJ

        for ip in 1:P
            exk = etax[k, ip]
            exkm1 = etax[km1, ip]
            out[1, ip] += exk
            out[2, ip] += 2.0 * xk * exk
            out[3, ip] += exk * ybar + xk * (ey_sum[ip] * invJ)
            out[4, ip] += 2.0 * ey_dot[ip] * invJ
            out[5, ip] += exk * xkm1 + xk * exkm1
        end
    end
    @. out *= invK
    return nothing
end

function simulate_observable_sensitivity_series!(out::Array{Float64,3},
    theta::NTuple{4,Float64},
    x0::Vector{Float64},
    y0::Matrix{Float64},
    cfg::L96Config,
    n_lags::Int,
    ws,
    rng::AbstractRNG,
    x::Vector{Float64},
    y::Matrix{Float64},
    etax::Matrix{Float64},
    etay::Array{Float64,3},
    obs_tmp::Matrix{Float64})
    copyto!(x, x0)
    copyto!(y, y0)
    fill!(etax, 0.0)
    fill!(etay, 0.0)
    fill!(out, 0.0)

    @inbounds for lag in 1:n_lags
        for _ in 1:cfg.save_every
            rk4_step_l96_tangent_twisted!(x, y, etax, etay, cfg.dt, ws, theta, cfg.K, cfg.J)
            add_process_noise!(x, y, rng, cfg.process_noise_sigma, cfg.dt; stochastic_x_noise=cfg.stochastic_x_noise)
        end
        compute_snapshot_observable_sensitivities!(obs_tmp, x, y, etax, etay)
        @views out[:, :, lag+1] .= obs_tmp
    end
    return nothing
end

function compute_numerical_responses_tangent(theta::NTuple{4,Float64},
    cfg::L96Config,
    init_tensor::Array{Float64,3},
    n_lags::Int;
    seed_base::Int)
    _, C, n_ens = size(init_tensor)
    C == cfg.J + 1 || error("Initial tensor channel mismatch")
    P = 4
    partials = [zeros(Float64, 5, P, n_lags + 1) for _ in 1:nthreads()]

    @info "Computing numerical responses (tangent integration)" ensembles = n_ens lags = n_lags threads = nthreads()
    Threads.@threads for ens in 1:n_ens
        tid = threadid()
        part = partials[tid]

        x0 = zeros(Float64, cfg.K)
        y0 = zeros(Float64, cfg.J, cfg.K)
        x = zeros(Float64, cfg.K)
        y = zeros(Float64, cfg.J, cfg.K)
        etax = zeros(Float64, cfg.K, P)
        etay = zeros(Float64, cfg.J, cfg.K, P)
        obs_tmp = zeros(Float64, 5, P)
        out = zeros(Float64, 5, P, n_lags + 1)
        ws = make_l96_tangent_workspace(cfg.K, cfg.J, P)
        rng = MersenneTwister(seed_base + ens)

        snapshot_to_xy!(x0, y0, init_tensor, ens)
        simulate_observable_sensitivity_series!(out, theta, x0, y0, cfg, n_lags, ws, rng, x, y, etax, etay, obs_tmp)
        part .+= out
    end

    responses = zeros(Float64, 5, P, n_lags + 1)
    for part in partials
        responses .+= part
    end
    responses ./= max(n_ens, 1)
    h_used = fill(NaN, 4)
    return responses, h_used
end

function compute_numerical_responses_fd(theta::NTuple{4,Float64},
    cfg::L96Config,
    init_tensor::Array{Float64,3},
    n_lags::Int;
    h_rel::Float64,
    h_abs::Vector{Float64},
    seed_base::Int)
    length(h_abs) == 4 || error("h_abs must have length 4 for [F,h,c,b]")
    _, C, n_ens = size(init_tensor)
    C == cfg.J + 1 || error("Initial tensor channel mismatch")

    responses = zeros(Float64, 5, 4, n_lags + 1)
    h_used = zeros(Float64, 4)
    nth = nthreads()
    @info "Computing numerical responses (central parameter perturbations)" ensembles = n_ens lags = n_lags threads = nth
    partials = [zeros(Float64, 5, n_lags + 1) for _ in 1:nth]
    workspaces = [
        (
            x0=zeros(Float64, cfg.K),
            y0=zeros(Float64, cfg.J, cfg.K),
            x=zeros(Float64, cfg.K),
            y=zeros(Float64, cfg.J, cfg.K),
            acc=zeros(Float64, 5),
            out_p=zeros(Float64, 5, n_lags + 1),
            out_m=zeros(Float64, 5, n_lags + 1),
            ws=make_l96_workspace(cfg.K, cfg.J),
            rng=MersenneTwister(seed_base + 100 * tid),
        ) for tid in 1:nth
    ]

    for ip in 1:4
        h = max(h_abs[ip], h_rel * max(abs(theta[ip]), 1.0))
        h_used[ip] = h
        rel_h = h / max(abs(theta[ip]), 1.0)
        if rel_h > 5e-3
            @warn "Large finite-difference step may bias linear-response estimates" parameter = PARAM_NAMES[ip] h = h relative_step = rel_h recommended_max = 5e-3
        end

        theta_p = collect(theta)
        theta_m = collect(theta)
        theta_p[ip] += h
        theta_m[ip] -= h
        tp = (theta_p[1], theta_p[2], theta_p[3], theta_p[4])
        tm = (theta_m[1], theta_m[2], theta_m[3], theta_m[4])

        for tid in 1:nth
            fill!(partials[tid], 0.0)
        end

        Threads.@threads for ens in 1:n_ens
            tid = threadid()
            part = partials[tid]
            wk = workspaces[tid]

            snapshot_to_xy!(wk.x0, wk.y0, init_tensor, ens)
            seed = seed_base + 1_000_000 * ip + ens

            Random.seed!(wk.rng, seed)
            simulate_observable_series!(wk.out_p, tp, wk.x0, wk.y0, cfg, n_lags, wk.ws, wk.rng, wk.x, wk.y, wk.acc)

            Random.seed!(wk.rng, seed)
            simulate_observable_series!(wk.out_m, tm, wk.x0, wk.y0, cfg, n_lags, wk.ws, wk.rng, wk.x, wk.y, wk.acc)

            @inbounds for m in 1:5, lag in 1:(n_lags+1)
                part[m, lag] += (wk.out_p[m, lag] - wk.out_m[m, lag]) / (2h)
            end
        end

        local_sum = zeros(Float64, 5, n_lags + 1)
        for tid in 1:nth
            local_sum .+= partials[tid]
        end
        local_sum ./= max(n_ens, 1)
        @views responses[:, ip, :] .= local_sum
    end

    return responses, h_used
end

function compute_numerical_responses(theta::NTuple{4,Float64},
    cfg::L96Config,
    init_tensor::Array{Float64,3},
    n_lags::Int;
    method::AbstractString,
    h_rel::Float64,
    h_abs::Vector{Float64},
    seed_base::Int)
    m = lowercase(strip(method))
    if m == "tangent"
        return compute_numerical_responses_tangent(theta, cfg, init_tensor, n_lags; seed_base=seed_base)
    elseif m == "finite_difference"
        return compute_numerical_responses_fd(theta, cfg, init_tensor, n_lags; h_rel=h_rel, h_abs=h_abs, seed_base=seed_base)
    else
        error("Unsupported numerical response method '$method' (expected tangent|finite_difference)")
    end
end

function save_response_figure(path::AbstractString,
    times::Vector{Float64},
    R_gauss::Array{Float64,3},
    R_unet::Array{Float64,3},
    R_num::Array{Float64,3};
    corrected::Bool)
    m, p, nt = size(R_num)
    size(R_gauss, 1) == m || error("R_gauss observable mismatch")
    size(R_unet, 1) == m || error("R_unet observable mismatch")
    size(R_gauss, 2) == p || error("R_gauss parameter mismatch")
    size(R_unet, 2) == p || error("R_unet parameter mismatch")
    size(R_gauss, 3) == nt || error("R_gauss lag mismatch")
    size(R_unet, 3) == nt || error("R_unet lag mismatch")
    length(times) == nt || error("times length mismatch")

    default(fontfamily="Computer Modern", dpi=180, legendfontsize=8, guidefontsize=9, tickfontsize=8, titlefontsize=10)
    lbl_g = corrected ? "GFDT+Gaussian (corrected)" : "GFDT+Gaussian"
    lbl_u = corrected ? "GFDT+UNet (corrected)" : "GFDT+UNet"
    lbl_n = "Numerical integration"

    panels = Vector{Plots.Plot}(undef, m * p)
    for i in 1:m, j in 1:p
        idx = (i - 1) * p + j
        legend_mode = (idx == 1 ? :topright : false)
        title_txt = i == 1 ? "d/d" * PARAM_NAMES[j] : ""
        ylabel_txt = j == 1 ? RESP_OBS_LABELS[i] : ""
        xlabel_txt = i == m ? "time" : ""

        pn = plot(times, vec(@view R_num[i, j, :]);
            color=:dodgerblue3,
            linewidth=2.2,
            label=(idx == 1 ? lbl_n : ""),
            legend=legend_mode,
            title=title_txt,
            ylabel=ylabel_txt,
            xlabel=xlabel_txt)
        plot!(pn, times, vec(@view R_gauss[i, j, :]);
            color=:black,
            linewidth=2.0,
            linestyle=:dash,
            label=(idx == 1 ? lbl_g : ""))
        plot!(pn, times, vec(@view R_unet[i, j, :]);
            color=:orangered3,
            linewidth=2.0,
            label=(idx == 1 ? lbl_u : ""))
        hline!(pn, [0.0]; color=:gray55, linestyle=:dot, linewidth=1.0, label="")
        panels[idx] = pn
    end

    fig = plot(panels...; layout=(m, p), size=(2200, 1700))
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function save_selected_response_figure(path::AbstractString,
    times::Vector{Float64},
    curves::Vector{NamedTuple})
    isempty(curves) && return ""
    R0 = curves[1].data
    m, p, nt = size(R0)
    length(times) == nt || error("times length mismatch")
    for c in curves
        size(c.data) == (m, p, nt) || error("Curve response shape mismatch for $(c.name)")
    end

    default(fontfamily="Computer Modern", dpi=180, legendfontsize=8, guidefontsize=9, tickfontsize=8, titlefontsize=10)
    panels = Vector{Plots.Plot}(undef, m * p)
    for i in 1:m, j in 1:p
        idx = (i - 1) * p + j
        legend_mode = (idx == 1 ? :topright : false)
        title_txt = i == 1 ? "d/d" * PARAM_NAMES[j] : ""
        ylabel_txt = j == 1 ? RESP_OBS_LABELS[i] : ""
        xlabel_txt = i == m ? "time" : ""

        pn = plot(; legend=legend_mode, title=title_txt, ylabel=ylabel_txt, xlabel=xlabel_txt)
        for c in curves
            plot!(pn, times, vec(@view c.data[i, j, :]);
                color=c.color,
                linewidth=2.0,
                linestyle=c.linestyle,
                label=(idx == 1 ? c.label : ""))
        end
        hline!(pn, [0.0]; color=:gray55, linestyle=:dot, linewidth=1.0, label="")
        panels[idx] = pn
    end

    fig = plot(panels...; layout=(m, p), size=(2200, 1700))
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function write_responses_csv(path::AbstractString,
    times::Vector{Float64},
    response_sets::Vector{Tuple{String,Array{Float64,3}}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,observable,param,time,response")
        for (method_name, R) in response_sets
            for i in 1:5, j in 1:4, t in eachindex(times)
                @printf(io, "%s,%s,%s,%.12e,%.12e\n",
                    method_name,
                    OBS_GLOBAL_NAMES[i],
                    PARAM_NAMES[j],
                    times[t],
                    R[i, j, t])
            end
        end
    end
    return path
end

function write_responses_hdf5(path::AbstractString,
    times::Vector{Float64},
    responses::Dict{String,Array{Float64,3}})
    mkpath(dirname(path))
    h5open(path, "w") do h5
        h5["times"] = times
        for (name, R) in responses
            h5[joinpath("responses", name)] = R
        end
    end
    return path
end
function compute_rmse_vs_reference(R_est::Array{Float64,3},
    R_ref::Array{Float64,3})
    size(R_est) == size(R_ref) || error("RMSE inputs must have matching shapes")
    overall = sqrt(mean((R_est .- R_ref) .^ 2))
    per_param = Dict(
        "F" => sqrt(mean((R_est[:, 1, :] .- R_ref[:, 1, :]) .^ 2)),
        "h" => sqrt(mean((R_est[:, 2, :] .- R_ref[:, 2, :]) .^ 2)),
        "c" => sqrt(mean((R_est[:, 3, :] .- R_ref[:, 3, :]) .^ 2)),
        "b" => sqrt(mean((R_est[:, 4, :] .- R_ref[:, 4, :]) .^ 2)),
    )
    return Dict{String,Any}("overall" => overall, "per_param" => per_param)
end

function linear_interpolate_3D_time(R::Array{Float64,3}, t_in::Vector{Float64}, t_out::Vector{Float64})
    M, P, N_in = size(R)
    N_out = length(t_out)
    R_out = zeros(Float64, M, P, N_out)
    for i in 1:M, j in 1:P
        for (k, t) in enumerate(t_out)
            if t <= t_in[1]
                R_out[i, j, k] = R[i, j, 1]
            elseif t >= t_in[end]
                R_out[i, j, k] = R[i, j, end]
            else
                idx = searchsortedlast(t_in, t)
                t0, t1 = t_in[idx], t_in[idx+1]
                w = (t - t0) / (t1 - t0)
                R_out[i, j, k] = R[i, j, idx] + w * (R[i, j, idx+1] - R[i, j, idx])
            end
        end
    end
    return R_out
end

function response_output_times(tmax::Float64)
    tmax_eff = min(max(tmax, 1e-8), RESP_PLOT_TMAX)
    return collect(range(0.0, stop=tmax_eff, length=RESP_PLOT_NPOINTS))
end

function step_to_impulse(R::Array{Float64,3}, times::Vector{Float64})
    m, p, nt = size(R)
    nt == length(times) || error("step_to_impulse time mismatch")
    nt >= 2 || error("Need at least two points to differentiate step response")
    out = zeros(Float64, m, p, nt)
    @inbounds for i in 1:m, j in 1:p
        out[i, j, 1] = (R[i, j, 2] - R[i, j, 1]) / (times[2] - times[1])
        for t in 2:(nt - 1)
            out[i, j, t] = (R[i, j, t + 1] - R[i, j, t - 1]) / (times[t + 1] - times[t - 1])
        end
        out[i, j, nt] = (R[i, j, nt] - R[i, j, nt - 1]) / (times[nt] - times[nt - 1])
    end
    return out
end

function response_signature(cfg::L96Config, params::Dict{String,Any})
    kv = Dict{String,Any}(
        "K" => cfg.K,
        "J" => cfg.J,
        "F" => cfg.F,
        "h" => cfg.h,
        "c" => cfg.c,
        "b" => cfg.b,
        "dt" => cfg.dt,
        "save_every" => cfg.save_every,
        "process_noise_sigma" => cfg.process_noise_sigma,
        "stochastic_x_noise" => cfg.stochastic_x_noise,
        "dataset_path" => abspath(cfg.dataset_path),
        "dataset_key" => cfg.dataset_key,
        "reference_gfdt_nsamples" => Int(params["reference_gfdt_nsamples"]),
        "reference_gfdt_start_index" => Int(params["reference_gfdt_start_index"]),
        "reference_numerical_ensembles" => Int(params["reference_numerical_ensembles"]),
        "reference_numerical_start_index" => Int(params["reference_numerical_start_index"]),
        "reference_numerical_method" => String(params["reference_numerical_method"]),
        "reference_h_rel" => Float64(params["reference_h_rel"]),
        "reference_h_abs" => join(Float64.(params["reference_h_abs"]), ","),
        "reference_numerical_seed_base" => Int(params["reference_numerical_seed_base"]),
        "reference_tmax" => Float64(params["reference_tmax"]),
        "reference_mean_center" => Bool(params["reference_mean_center"]),
        "score_batch_size" => Int(params["score_batch_size"]),
        "correction_ridge" => Float64(params["correction_ridge"]),
        "correction_max_deviation" => Float64(params["correction_max_deviation"]),
    )
    parts = String[]
    for k in sort(collect(keys(kv)))
        push!(parts, string(k) * "=" * repr(kv[k]))
    end
    return join(parts, "\n")
end

function response_cache_path(cfg::L96Config, params::Dict{String,Any})
    signature = response_signature(cfg, params)
    digest = bytes2hex(sha1(signature))
    cache_root = abspath(String(params["reference_cache_root"]))
    mkpath(cache_root)
    return joinpath(cache_root, "reference_" * digest * ".hdf5"), signature
end

function reference_cache_matches(path::AbstractString, signature::AbstractString)
    isfile(path) || return false
    return h5open(path, "r") do h5
        attrs = attributes(h5)
        haskey(attrs, "signature") || return false
        stored = try
            String(read(attrs["signature"]))
        catch
            ""
        end
        return stored == signature
    end
end

function required_dataset_samples(params::Dict{String,Any})
    candidates = Int[
        Int(params["gfdt_start_index"]) + Int(params["gfdt_nsamples"]) + 10,
        Int(params["numerical_start_index"]) + Int(params["numerical_ensembles"]) + 10,
        Int(params["reference_gfdt_start_index"]) + Int(params["reference_gfdt_nsamples"]) + 10,
        Int(params["reference_numerical_start_index"]) + Int(params["reference_numerical_ensembles"]) + 10,
        1,
    ]
    return maximum(candidates)
end

function compute_reference_payload(cfg::L96Config, params::Dict{String,Any})
    theta = (cfg.F, cfg.h, cfg.c, cfg.b)
    delta_t_obs = cfg.dt * cfg.save_every
    tmax_ref = Float64(params["reference_tmax"])
    n_lags_req = max(1, Int(floor(tmax_ref / delta_t_obs)))
    times_out = response_output_times(tmax_ref)

    gfdt_nsamples = Int(params["reference_gfdt_nsamples"])
    gfdt_start = Int(params["reference_gfdt_start_index"])
    tensor = load_observation_subset(
        cfg;
        nsamples=gfdt_nsamples,
        start_index=gfdt_start,
        subset_label="reference_gfdt",
    )
    n_lags = min(n_lags_req, size(tensor, 3) - 1)
    n_lags >= 1 || error("Need at least 2 GFDT samples to build reference responses")
    times_native = collect(0:n_lags) .* delta_t_obs
    A = compute_global_observables(tensor)
    mean_center = Bool(params["reference_mean_center"])

    gauss = compute_gaussian_conjugates(
        tensor,
        theta;
        batch_size=Int(params["score_batch_size"]),
        correction_ridge=Float64(params["correction_ridge"]),
        correction_max_deviation=Float64(params["correction_max_deviation"]),
    )
    C_gauss_raw, R_gauss_raw, _ = build_gfdt_response(A, gauss.G_raw, delta_t_obs, n_lags; mean_center=mean_center)
    C_gauss_corr, R_gauss_corr, _ = build_gfdt_response(A, gauss.G_corr, delta_t_obs, n_lags; mean_center=mean_center)

    num_ensembles = Int(params["reference_numerical_ensembles"])
    num_start = Int(params["reference_numerical_start_index"])
    init_tensor = load_observation_subset(
        cfg;
        nsamples=num_ensembles,
        start_index=num_start,
        subset_label="reference_numerical",
    )
    R_num_step, h_used = compute_numerical_responses(
        theta,
        cfg,
        init_tensor,
        n_lags;
        method=String(params["reference_numerical_method"]),
        h_rel=Float64(params["reference_h_rel"]),
        h_abs=Float64.(params["reference_h_abs"]),
        seed_base=Int(params["reference_numerical_seed_base"]),
    )
    R_num_impulse = step_to_impulse(R_num_step, times_native)

    step_map = Dict{String,Array{Float64,3}}(
        "gfdt_gaussian_raw" => linear_interpolate_3D_time(R_gauss_raw, times_native, times_out),
        "gfdt_gaussian_corrected" => linear_interpolate_3D_time(R_gauss_corr, times_native, times_out),
        "numerical_integration" => linear_interpolate_3D_time(R_num_step, times_native, times_out),
    )
    impulse_map = Dict{String,Array{Float64,3}}(
        "gfdt_gaussian_raw" => linear_interpolate_3D_time(C_gauss_raw, times_native, times_out),
        "gfdt_gaussian_corrected" => linear_interpolate_3D_time(C_gauss_corr, times_native, times_out),
        "numerical_integration" => linear_interpolate_3D_time(R_num_impulse, times_native, times_out),
    )

    metadata = Dict{String,Any}(
        "gfdt_samples_used" => size(tensor, 3),
        "numerical_ensembles_used" => size(init_tensor, 3),
        "numerical_method" => String(params["reference_numerical_method"]),
        "h_used" => Float64.(h_used),
        "n_lags_native" => n_lags,
        "delta_t_obs" => delta_t_obs,
        "times_native_end" => times_native[end],
        "times_out_end" => times_out[end],
    )
    return (times_out=times_out, step_map=step_map, impulse_map=impulse_map, metadata=metadata)
end

function write_reference_cache(path::AbstractString,
    signature::AbstractString,
    payload)
    mkpath(dirname(path))
    h5open(path, "w") do h5
        attrs = attributes(h5)
        attrs["signature"] = String(signature)
        attrs["generated_at_utc"] = string(now(UTC))
        attrs["gfdt_samples_used"] = Int(payload.metadata["gfdt_samples_used"])
        attrs["numerical_ensembles_used"] = Int(payload.metadata["numerical_ensembles_used"])
        attrs["numerical_method"] = String(payload.metadata["numerical_method"])
        attrs["delta_t_obs"] = Float64(payload.metadata["delta_t_obs"])
        attrs["h_used"] = Float64.(payload.metadata["h_used"])

        h5["times"] = payload.times_out
        for (name, arr) in payload.step_map
            h5[joinpath("responses", "heaviside", name)] = arr
        end
        for (name, arr) in payload.impulse_map
            h5[joinpath("responses", "impulse", name)] = arr
        end
    end
    return path
end

function ensure_reference_cache!(cfg::L96Config, params::Dict{String,Any}; force::Bool=false)
    cache_path, signature = response_cache_path(cfg, params)
    if !force && reference_cache_matches(cache_path, signature)
        return Dict{String,Any}("path" => cache_path, "signature" => signature, "generated" => false)
    end
    payload = compute_reference_payload(cfg, params)
    write_reference_cache(cache_path, signature, payload)
    return Dict{String,Any}("path" => cache_path, "signature" => signature, "generated" => true)
end

function load_reference_responses(path::AbstractString,
    signature::AbstractString;
    impulse::Bool)
    isfile(path) || error("Reference response cache not found: $path")
    return h5open(path, "r") do h5
        attrs = attributes(h5)
        haskey(attrs, "signature") || error("Reference cache missing signature attribute: $path")
        stored_sig = String(read(attrs["signature"]))
        stored_sig == signature || error("Reference cache signature mismatch at $path")

        times = Float64.(read(h5["times"]))
        bucket = impulse ? "impulse" : "heaviside"
        group = h5[joinpath("responses", bucket)]
        out = Dict{String,Array{Float64,3}}()
        for key in keys(group)
            out[String(key)] = Float64.(read(group[key]))
        end
        metadata = Dict{String,Any}(
            "gfdt_samples_used" => haskey(attrs, "gfdt_samples_used") ? Int(read(attrs["gfdt_samples_used"])) : -1,
            "numerical_ensembles_used" => haskey(attrs, "numerical_ensembles_used") ? Int(read(attrs["numerical_ensembles_used"])) : -1,
            "numerical_method" => haskey(attrs, "numerical_method") ? String(read(attrs["numerical_method"])) : "",
        )
        return (times=times, responses=out, metadata=metadata)
    end
end

function print_reference_summary(cache_info::Dict{String,Any}, loaded, impulse::Bool)
    println("Reference responses:")
    println("  - cache_path: ", cache_info["path"])
    println("  - regenerated: ", cache_info["generated"])
    println("  - response_type: ", impulse ? "impulse" : "heaviside")
    println("  - gfdt_samples_used: ", loaded.metadata["gfdt_samples_used"])
    println("  - numerical_ensembles_used: ", loaded.metadata["numerical_ensembles_used"])
    println("  - numerical_method: ", loaded.metadata["numerical_method"])
end


function write_summary_toml(path::AbstractString,
    cfg::L96Config,
    checkpoint_path::AbstractString,
    n_lags::Int,
    times::Vector{Float64},
    n_gfdt_samples::Int,
    n_numerical_ensembles::Int,
    numerical_method::AbstractString,
    h_used::Vector{Float64},
    score_forward_mode::AbstractString,
    gauss_corr_stats,
    unet_corr_stats,
    rmse_gauss_raw,
    rmse_gauss_corr,
    rmse_unet_raw,
    rmse_unet_corr,
    fig_raw::AbstractString,
    fig_corr::AbstractString,
    responses_csv::AbstractString,
    responses_h5::AbstractString)
    summary = Dict(
        "l96" => Dict(
            "K" => cfg.K,
            "J" => cfg.J,
            "F" => cfg.F,
            "h" => cfg.h,
            "c" => cfg.c,
            "b" => cfg.b,
            "dt" => cfg.dt,
            "save_every" => cfg.save_every,
            "delta_t_obs" => cfg.dt * cfg.save_every,
            "process_noise_sigma" => cfg.process_noise_sigma,
        ),
        "checkpoint" => Dict(
            "path" => abspath(checkpoint_path),
            "epoch" => checkpoint_epoch(checkpoint_path),
        ),
        "score_model" => Dict(
            "forward_mode" => score_forward_mode,
            "independent_of_method3_tuning" => true,
        ),
        "responses" => Dict(
            "n_lags" => n_lags,
            "tmax_effective" => times[end],
            "gfdt_samples_used" => n_gfdt_samples,
            "numerical_ensembles_used" => n_numerical_ensembles,
            "numerical_method" => numerical_method,
            "finite_difference_h" => Dict(
                "F" => h_used[1],
                "h" => h_used[2],
                "c" => h_used[3],
                "b" => h_used[4],
            ),
        ),
        "score_correction" => Dict(
            "gaussian" => Dict(
                "ridge" => gauss_corr_stats.ridge,
                "solver" => gauss_corr_stats.solver,
                "pre_identity_rmse" => gauss_corr_stats.pre_identity_rmse,
                "post_identity_rmse" => gauss_corr_stats.post_identity_rmse,
                "post_identity_rmse_direct" => gauss_corr_stats.post_identity_rmse_direct,
                "post_identity_rmse_pinv" => gauss_corr_stats.post_identity_rmse_pinv,
                "max_deviation" => gauss_corr_stats.max_deviation,
                "post_deviation" => gauss_corr_stats.post_deviation,
                "post_deviation_direct" => gauss_corr_stats.post_deviation_direct,
                "post_deviation_pinv" => gauss_corr_stats.post_deviation_pinv,
                "cond_regularized" => gauss_corr_stats.cond_regularized,
            ),
            "unet" => Dict(
                "ridge" => unet_corr_stats.ridge,
                "solver" => unet_corr_stats.solver,
                "pre_identity_rmse" => unet_corr_stats.pre_identity_rmse,
                "post_identity_rmse" => unet_corr_stats.post_identity_rmse,
                "post_identity_rmse_direct" => unet_corr_stats.post_identity_rmse_direct,
                "post_identity_rmse_pinv" => unet_corr_stats.post_identity_rmse_pinv,
                "max_deviation" => unet_corr_stats.max_deviation,
                "post_deviation" => unet_corr_stats.post_deviation,
                "post_deviation_direct" => unet_corr_stats.post_deviation_direct,
                "post_deviation_pinv" => unet_corr_stats.post_deviation_pinv,
                "cond_regularized" => unet_corr_stats.cond_regularized,
            ),
        ),
        "rmse_vs_numerical" => Dict(
            "gfdt_gaussian_raw" => Dict(
                "overall" => rmse_gauss_raw["overall"],
                "per_param" => rmse_gauss_raw["per_param"],
            ),
            "gfdt_gaussian_corrected" => Dict(
                "overall" => rmse_gauss_corr["overall"],
                "per_param" => rmse_gauss_corr["per_param"],
            ),
            "gfdt_unet_raw" => Dict(
                "overall" => rmse_unet_raw["overall"],
                "per_param" => rmse_unet_raw["per_param"],
            ),
            "gfdt_unet_corrected" => Dict(
                "overall" => rmse_unet_corr["overall"],
                "per_param" => rmse_unet_corr["per_param"],
            ),
        ),
        "outputs" => Dict(
            "figure_raw" => abspath(fig_raw),
            "figure_corrected" => abspath(fig_corr),
            "responses_csv" => abspath(responses_csv),
            "responses_hdf5" => abspath(responses_h5),
        ),
    )

    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, summary)
    end
    return path
end

function write_summary_toml_selected(path::AbstractString,
    cfg::L96Config,
    run_info::Dict{String,Any},
    methods::Dict{String,Bool},
    response_cfg::Dict{String,Any},
    score_cfg::Dict{String,Any},
    dataset_sync_cfg::Dict{String,Any},
    correction_cfg::Dict{String,Any},
    output_cfg::Dict{String,Any},
    h_used::Vector{Float64},
    score_correction::Dict{String,Any},
    rmse::Dict{String,Any})
    summary = Dict(
        "l96" => Dict(
            "K" => cfg.K,
            "J" => cfg.J,
            "F" => cfg.F,
            "h" => cfg.h,
            "c" => cfg.c,
            "b" => cfg.b,
            "dt" => cfg.dt,
            "save_every" => cfg.save_every,
            "delta_t_obs" => cfg.dt * cfg.save_every,
            "process_noise_sigma" => cfg.process_noise_sigma,
        ),
        "run" => run_info,
        "methods" => methods,
        "responses" => response_cfg,
        "score_model" => score_cfg,
        "dataset_attr_sync" => dataset_sync_cfg,
        "correction" => correction_cfg,
        "finite_difference_h" => Dict(
            "F" => h_used[1],
            "h" => h_used[2],
            "c" => h_used[3],
            "b" => h_used[4],
        ),
        "score_correction" => score_correction,
        "rmse_vs_numerical" => rmse,
        "outputs" => output_cfg,
    )
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, summary)
    end
    return path
end

function main(args=ARGS)
    cli = parse_cli(args)
    mode = String(cli["mode"])
    params_path = abspath(String(cli["params_toml"]))
    params = load_response_params(params_path)

    methods = Dict{String,Bool}(
        "gaussian" => Bool(params["methods"]["gaussian"]),
        "unet" => Bool(params["methods"]["unet"]),
        "numerical" => Bool(params["methods"]["numerical"]),
    )
    selected_methods = parse_method_list(String(cli["methods_override"]))
    if !isempty(selected_methods)
        for k in keys(methods)
            methods[k] = false
        end
        for m in selected_methods
            methods[m] = true
        end
    end
    any(values(methods)) || error("No method enabled. Set at least one of {gaussian, unet, numerical} in parameters TOML (or via --methods).")

    cfg = params["l96_config"]
    if !isfile(cfg.dataset_path)
        @info "Response dataset not found. Generating dataset..." dataset_path = cfg.dataset_path
        env = copy(ENV)
        env["L96_K"] = string(cfg.K)
        env["L96_J"] = string(cfg.J)
        env["L96_F"] = string(cfg.F)
        env["L96_H"] = string(cfg.h)
        env["L96_C"] = string(cfg.c)
        env["L96_B"] = string(cfg.b)
        env["L96_DT"] = string(cfg.dt)
        env["L96_SAVE_EVERY"] = string(cfg.save_every)
        env["L96_NSAMPLES"] = string(required_dataset_samples(params))
        env["L96_PROCESS_NOISE_SIGMA"] = string(cfg.process_noise_sigma)
        env["L96_STOCHASTIC_X_NOISE"] = string(cfg.stochastic_x_noise)
        env["L96_DATA_PATH"] = cfg.dataset_path
        env["L96_DATASET_KEY"] = cfg.dataset_key
        env["L96_PIPELINE_MODE"] = "true"
        run(setenv(`julia --project=. scripts/L96/lib/generate_data.jl`, env))
    end

    dataset_attr_sync_mode = String(params["dataset_attr_sync_mode"])
    cfg, dataset_attr_sync_info = sync_l96_config_with_dataset_attrs(cfg; mode=dataset_attr_sync_mode)

    if mode == "reference"
        cache_info = ensure_reference_cache!(cfg, params; force=Bool(params["reference_force_regenerate"]))
        loaded = load_reference_responses(String(cache_info["path"]), String(cache_info["signature"]); impulse=false)
        print_reference_summary(cache_info, loaded, false)
        return
    end

    run_dir = abspath(String(params["run_dir"]))
    output_root = abspath(String(params["output_root"]))
    run_prefix = String(params["run_prefix"])
    out_dir, run_id = allocate_response_run_dir(output_root; run_prefix=run_prefix)
    run_params = Dict{String,Any}(params)
    pop!(run_params, "l96_config", nothing)
    run_params["methods"] = methods
    run_params["mode"] = mode
    run_params["compute_impulse"] = params["compute_impulse"]
    run_params["checkpoint_epoch"] = Int(params["checkpoint_epoch"])
    run_params["integration"] = Dict(
        "K" => cfg.K,
        "J" => cfg.J,
        "F" => cfg.F,
        "h" => cfg.h,
        "c" => cfg.c,
        "b" => cfg.b,
        "dt" => cfg.dt,
        "save_every" => cfg.save_every,
        "process_noise_sigma" => cfg.process_noise_sigma,
    )
    run_params["dataset"] = Dict(
        "path" => cfg.dataset_path,
        "key" => cfg.dataset_key,
        "attr_sync_mode" => dataset_attr_sync_mode,
    )
    run_params["dataset_attr_sync"] = dataset_attr_sync_info
    run_params["generated_at_utc"] = string(now(UTC))
    run_params["output_dir"] = abspath(out_dir)
    run_params["run_id"] = run_id
    run_params_path = joinpath(out_dir, "parameters_used.toml")
    open(run_params_path, "w") do io
        TOML.print(io, run_params)
    end

    response_tmax_requested = Float64(params["response_tmax"])
    response_tmax = response_tmax_requested
    score_batch_size = Int(params["score_batch_size"])
    score_device_pref = String(params["score_device"])
    score_forward_mode = String(params["score_forward_mode"])
    numerical_method = String(params["reference_numerical_method"])
    h_used = Float64.(params["reference_h_abs"])
    correction_ridge = Float64(params["correction_ridge"])
    correction_max_deviation = Float64(params["correction_max_deviation"])
    mean_center = Bool(params["mean_center"])
    compute_impulse = Bool(params["compute_impulse"])

    theta = (cfg.F, cfg.h, cfg.c, cfg.b)
    delta_t_obs = cfg.dt * cfg.save_every
    n_lags_req = max(1, Int(floor(response_tmax / delta_t_obs)))

    try
        FFTW.set_num_threads(max(1, nthreads()))
    catch
    end

    need_unet = methods["unet"]
    tensor = nothing
    A = nothing
    n_lags = n_lags_req
    n_gfdt_samples_used = 0

    if need_unet
        gfdt_nsamples = Int(params["gfdt_nsamples"])
        gfdt_start_index = Int(params["gfdt_start_index"])
        @info "Loading unperturbed trajectory subset for GFDT responses" dataset = cfg.dataset_path nsamples = gfdt_nsamples start_index = gfdt_start_index
        tensor = load_observation_subset(
            cfg;
            nsamples=gfdt_nsamples,
            start_index=gfdt_start_index,
            subset_label="gfdt",
        )
        K, C, N = size(tensor)
        @info "GFDT tensor loaded" K = K C = C samples = N
        C == cfg.J + 1 || error("Channel mismatch in GFDT tensor")
        K == cfg.K || error("K mismatch in GFDT tensor")
        N >= 2 || error("Need at least 2 samples for GFDT response estimation")
        n_lags = min(n_lags_req, N - 1)
        n_gfdt_samples_used = N
        @info "Computing global observables phi(t)"
        A = compute_global_observables(tensor)
    end

    times = collect(0:n_lags) .* delta_t_obs
    times_out = response_output_times(response_tmax)
    @info "Response lag configuration" requested_tmax = response_tmax_requested used_tmax = response_tmax effective_tmax = times[end] output_tmax = times_out[end] n_lags = n_lags delta_t_obs = delta_t_obs

    score_device_selected = "CPU"
    score_device_reason = "UNet disabled"
    if need_unet
        score_device_selected, score_device_reason = choose_score_device(
            score_device_pref;
            max_mem_fraction=Float64(params["auto_gpu_max_mem_fraction"]),
            max_utilization=Float64(params["auto_gpu_max_utilization"]),
            min_free_mb=Float64(params["auto_gpu_min_free_mb"]),
        )
        @info "UNet device selection" requested = score_device_pref selected = score_device_selected reason = score_device_reason
    end

    R_gauss_raw = nothing
    R_gauss_corr = nothing
    R_unet_raw = nothing
    R_unet_corr = nothing
    R_num = nothing
    gauss_corr_stats = nothing
    unet_corr_stats = nothing
    n_num_ens = 0
    reference_cache_path_used = ""
    reference_cache_generated = false
    checkpoint_path = ""

    if methods["gaussian"] || methods["numerical"]
        cache_info = ensure_reference_cache!(cfg, params; force=Bool(params["reference_force_regenerate"]))
        reference_cache_path_used = String(cache_info["path"])
        reference_cache_generated = Bool(cache_info["generated"])
        loaded = load_reference_responses(reference_cache_path_used, String(cache_info["signature"]); impulse=compute_impulse)
        ref_times = loaded.times
        n_gfdt_samples_used = max(n_gfdt_samples_used, Int(get(loaded.metadata, "gfdt_samples_used", n_gfdt_samples_used)))
        n_num_ens = Int(get(loaded.metadata, "numerical_ensembles_used", n_num_ens))
        if haskey(loaded.responses, "numerical_integration")
            numerical_method = String(get(loaded.metadata, "numerical_method", numerical_method))
        end

        if methods["gaussian"]
            haskey(loaded.responses, "gfdt_gaussian_raw") || error("Reference cache missing gfdt_gaussian_raw response")
            haskey(loaded.responses, "gfdt_gaussian_corrected") || error("Reference cache missing gfdt_gaussian_corrected response")
            R_gauss_raw = loaded.responses["gfdt_gaussian_raw"]
            R_gauss_corr = loaded.responses["gfdt_gaussian_corrected"]
            if ref_times != times_out
                R_gauss_raw = linear_interpolate_3D_time(R_gauss_raw, ref_times, times_out)
                R_gauss_corr = linear_interpolate_3D_time(R_gauss_corr, ref_times, times_out)
            end
        end

        if methods["numerical"]
            haskey(loaded.responses, "numerical_integration") || error("Reference cache missing numerical_integration response")
            R_num = loaded.responses["numerical_integration"]
            if ref_times != times_out
                R_num = linear_interpolate_3D_time(R_num, ref_times, times_out)
            end
        end
    end

    if need_unet
        checkpoint_path = resolve_checkpoint_path(run_dir, String(params["checkpoint_path"]), Int(params["checkpoint_epoch"]))
        @info "Method 2: GFDT + UNet score" checkpoint = checkpoint_path device = score_device_selected
        unet = compute_unet_conjugates(
            tensor,
            theta,
            checkpoint_path;
            batch_size=score_batch_size,
            device_pref=score_device_selected,
            forward_mode=score_forward_mode,
            correction_ridge=correction_ridge,
            correction_max_deviation=correction_max_deviation,
        )
        unet_corr_stats = unet.correction_stats
        C_unet_raw, R_unet_raw_step, _ = build_gfdt_response(A, unet.G_raw, delta_t_obs, n_lags; mean_center=mean_center)
        C_unet_corr, R_unet_corr_step, _ = build_gfdt_response(A, unet.G_corr, delta_t_obs, n_lags; mean_center=mean_center)
        target_unet_raw = compute_impulse ? C_unet_raw : R_unet_raw_step
        target_unet_corr = compute_impulse ? C_unet_corr : R_unet_corr_step
        R_unet_raw = linear_interpolate_3D_time(target_unet_raw, times, times_out)
        R_unet_corr = linear_interpolate_3D_time(target_unet_corr, times, times_out)
    end

    rmse_summary = Dict{String,Any}()
    if R_num !== nothing
        if R_gauss_raw !== nothing
            rmse_summary["gfdt_gaussian_raw"] = compute_rmse_vs_reference(R_gauss_raw, R_num)
            rmse_summary["gfdt_gaussian_corrected"] = compute_rmse_vs_reference(R_gauss_corr, R_num)
        end
        if R_unet_raw !== nothing
            rmse_summary["gfdt_unet_raw"] = compute_rmse_vs_reference(R_unet_raw, R_num)
            rmse_summary["gfdt_unet_corrected"] = compute_rmse_vs_reference(R_unet_corr, R_num)
        end
    end

    curves_raw = NamedTuple[]
    curves_corr = NamedTuple[]
    if R_num !== nothing
        push!(curves_raw, (name="numerical_integration", label="Numerical integration", color=:dodgerblue3, linestyle=:solid, data=R_num))
        push!(curves_corr, (name="numerical_integration", label="Numerical integration", color=:dodgerblue3, linestyle=:solid, data=R_num))
    end
    if R_gauss_raw !== nothing
        push!(curves_raw, (name="gfdt_gaussian_raw", label="GFDT+Gaussian", color=:black, linestyle=:dash, data=R_gauss_raw))
        push!(curves_corr, (name="gfdt_gaussian_corrected", label="GFDT+Gaussian (corrected)", color=:black, linestyle=:dash, data=R_gauss_corr))
    end
    if R_unet_raw !== nothing
        push!(curves_raw, (name="gfdt_unet_raw", label="GFDT+UNet", color=:orangered3, linestyle=:solid, data=R_unet_raw))
        push!(curves_corr, (name="gfdt_unet_corrected", label="GFDT+UNet (corrected)", color=:orangered3, linestyle=:solid, data=R_unet_corr))
    end

    fig_raw = save_selected_response_figure(
        joinpath(out_dir, "responses_5x4_selected_methods_raw.png"),
        times_out,
        curves_raw,
    )
    fig_corr = save_selected_response_figure(
        joinpath(out_dir, "responses_5x4_selected_methods_corrected.png"),
        times_out,
        curves_corr,
    )

    response_sets = Vector{Tuple{String,Array{Float64,3}}}()
    response_map = Dict{String,Array{Float64,3}}()
    if R_gauss_raw !== nothing
        push!(response_sets, ("gfdt_gaussian_raw", R_gauss_raw))
        push!(response_sets, ("gfdt_gaussian_corrected", R_gauss_corr))
        response_map["gfdt_gaussian_raw"] = R_gauss_raw
        response_map["gfdt_gaussian_corrected"] = R_gauss_corr
    end
    if R_unet_raw !== nothing
        push!(response_sets, ("gfdt_unet_raw", R_unet_raw))
        push!(response_sets, ("gfdt_unet_corrected", R_unet_corr))
        response_map["gfdt_unet_raw"] = R_unet_raw
        response_map["gfdt_unet_corrected"] = R_unet_corr
    end
    if R_num !== nothing
        push!(response_sets, ("numerical_integration", R_num))
        response_map["numerical_integration"] = R_num
    end
    responses_csv = write_responses_csv(joinpath(out_dir, "responses_5x4_selected_methods.csv"), times_out, response_sets)
    responses_h5 = write_responses_hdf5(joinpath(out_dir, "responses_5x4_selected_methods.hdf5"), times_out, response_map)

    score_corr_summary = Dict{String,Any}()
    gauss_corr_stats !== nothing && (score_corr_summary["gaussian"] = extract_correction_stats_dict(gauss_corr_stats))
    unet_corr_stats !== nothing && (score_corr_summary["unet"] = extract_correction_stats_dict(unet_corr_stats))

    response_cfg = Dict{String,Any}(
        "n_lags" => (length(times_out) - 1),
        "tmax_requested" => response_tmax_requested,
        "tmax_used" => response_tmax,
        "tmax_effective" => times_out[end],
        "time_points" => length(times_out),
        "gfdt_samples_used" => n_gfdt_samples_used,
        "numerical_ensembles_used" => n_num_ens,
        "numerical_method" => numerical_method,
    )
    score_cfg = Dict{String,Any}(
        "forward_mode" => score_forward_mode,
        "requested_device" => score_device_pref,
        "selected_device" => score_device_selected,
        "selection_reason" => score_device_reason,
        "independent_of_method3_tuning" => true,
        "checkpoint_path" => (methods["unet"] ? abspath(checkpoint_path) : ""),
        "checkpoint_epoch" => (methods["unet"] ? checkpoint_epoch(checkpoint_path) : -1),
    )
    dataset_sync_cfg = Dict{String,Any}(dataset_attr_sync_info)
    correction_cfg = Dict{String,Any}(
        "ridge" => correction_ridge,
        "max_deviation" => correction_max_deviation,
    )
    output_cfg = Dict{String,Any}(
        "figure_raw" => abspath(fig_raw),
        "figure_corrected" => abspath(fig_corr),
        "responses_csv" => abspath(responses_csv),
        "responses_hdf5" => abspath(responses_h5),
        "parameters_used_toml" => abspath(run_params_path),
        "reference_cache_hdf5" => reference_cache_path_used,
        "reference_cache_generated" => reference_cache_generated,
    )
    run_info = Dict{String,Any}(
        "id" => run_id,
        "output_dir" => abspath(out_dir),
        "params_toml" => params_path,
        "generated_at_utc" => string(now(UTC)),
    )
    summary_toml = write_summary_toml_selected(
        joinpath(out_dir, "responses_5x4_summary.toml"),
        cfg,
        run_info,
        methods,
        response_cfg,
        score_cfg,
        dataset_sync_cfg,
        correction_cfg,
        output_cfg,
        h_used,
        score_corr_summary,
        rmse_summary,
    )

    @info "Completed L96 response computation" output_dir = out_dir run_id = run_id methods = methods
    println("Saved:")
    println("  - ", run_params_path)
    println("  - ", fig_raw)
    println("  - ", fig_corr)
    println("  - ", responses_csv)
    println("  - ", responses_h5)
    println("  - ", summary_toml)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
