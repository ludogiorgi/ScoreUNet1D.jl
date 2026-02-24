# Standard command (from repository root):
# julia --threads auto --project=. scripts/arnold/compute_responses.jl --params scripts/arnold/parameters_responses.toml
# Nohup command:
# nohup julia --threads auto --project=. scripts/arnold/compute_responses.jl --params scripts/arnold/parameters_responses.toml > scripts/arnold/nohup_compute_responses.log 2>&1 &

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
using Random
using ScoreUNet1D
using SHA
using Statistics
using TOML
using Dates
using Base.Threads
using Zygote

include(joinpath(@__DIR__, "lib", "ArnoldCommon.jl"))
using .ArnoldCommon

const OBS_NAMES = [
    "phi1_mean_x",
    "phi2_mean_x2",
    "phi3_mean_x_xm1",
    "phi4_mean_x_xm2",
    "phi5_mean_x_xm3",
]

const OBS_LABELS = [
    "phi1=<X_k>",
    "phi2=<X_k^2>",
    "phi3=<X_k X_{k-1}>",
    "phi4=<X_k X_{k-2}>",
    "phi5=<X_k X_{k-3}>",
]

const PARAM_NAMES = ["alpha0", "alpha1", "alpha2", "alpha3", "sigma"]
const N_OBS = length(OBS_NAMES)
const N_PARAM = length(PARAM_NAMES)
const ASYMPTOTIC_FD_NSAMPLES = 15_000_000
const ASYMPTOTIC_FD_SPINUP = 2_000

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_responses.toml")
    checkpoint_override = ""
    output_override = ""
    correction_override = nothing
    response_kind_override = ""

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = args[i+1]
            i += 2
        elseif a == "--checkpoint"
            i == length(args) && error("--checkpoint expects a value")
            checkpoint_override = args[i+1]
            i += 2
        elseif a == "--output-dir"
            i == length(args) && error("--output-dir expects a value")
            output_override = args[i+1]
            i += 2
        elseif a == "--apply-correction"
            i == length(args) && error("--apply-correction expects true/false")
            correction_override = parse_bool(args[i+1])
            i += 2
        elseif a == "--response-kind"
            i == length(args) && error("--response-kind expects heaviside or impulse")
            response_kind_override = lowercase(strip(args[i+1]))
            i += 2
        else
            error("Unknown argument: $a")
        end
    end

    return (
        params_path=abspath(params_path),
        checkpoint_override=checkpoint_override,
        output_override=output_override,
        correction_override=correction_override,
        response_kind_override=response_kind_override,
    )
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_bool(tbl::Dict{String,Any}, key::String, default) = parse_bool(get(tbl, key, default))
as_float_vec(tbl::Dict{String,Any}, key::String, default) = Float64.(collect(get(tbl, key, default)))

function require_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa Dict{String,Any} || error("[$key] must be TOML table")
    return Dict{String,Any}(doc[key])
end

function load_config(path::AbstractString)
    isfile(path) || error("Response parameter file not found: $path")
    doc = TOML.parsefile(path)

    paths = require_table(doc, "paths")
    methods = require_table(doc, "methods")
    gfdt = require_table(doc, "gfdt")
    numerical = require_table(doc, "numerical")
    figures = haskey(doc, "figures") ? Dict{String,Any}(doc["figures"]) : Dict{String,Any}()
    cache = haskey(doc, "cache") ? Dict{String,Any}(doc["cache"]) : Dict{String,Any}()

    data_params_path = abspath(as_str(paths, "data_params", "scripts/arnold/parameters_data.toml"))
    gfdt_role = as_str(paths, "gfdt_dataset_role", "gfdt_stochastic")
    data_cfg, _ = load_data_config(data_params_path)
    gfdt_role in ArnoldCommon.ARNOLD_DATASET_ROLES || error("paths.gfdt_dataset_role must be one of $(join(ArnoldCommon.ARNOLD_DATASET_ROLES, ", "))")

    gfdt_info = ensure_arnold_dataset_role!(data_cfg, gfdt_role)
    closure_theta, closure_meta = resolve_closure_theta(data_cfg)
    obs_alpha0 = data_cfg["closure.alpha0_initial"]
    obs_alpha1 = data_cfg["closure.alpha1_initial"]
    obs_alpha2 = data_cfg["closure.alpha2_initial"]
    obs_alpha3 = data_cfg["closure.alpha3_initial"]
    dt_obs = dataset_role_spacing(data_cfg, gfdt_role)
    save_every = Int(round(dt_obs / data_cfg["twoscale.dt"]))
    save_every = max(save_every, 1)

    cfg = Dict{String,Any}(
        "paths.data_params" => data_params_path,
        "paths.dataset_role" => gfdt_role,
        "paths.dataset_path" => gfdt_info["path"],
        "paths.dataset_key" => gfdt_info["key"],
        "paths.checkpoint_path" => as_str(paths, "checkpoint_path", ""),
        "paths.output_root" => abspath(as_str(paths, "output_root", "scripts/arnold/output/responses")),
        "paths.cache_root" => abspath(as_str(paths, "cache_root", "scripts/arnold/cache/responses")), "integration.K" => data_cfg["twoscale.K"],
        "integration.J" => data_cfg["twoscale.J"],
        "integration.F" => data_cfg["twoscale.F"],
        "integration.h" => data_cfg["twoscale.h"],
        "integration.c" => data_cfg["twoscale.c"],
        "integration.b" => data_cfg["twoscale.b"],
        "integration.dt" => data_cfg["twoscale.dt"],
        "integration.save_every" => save_every,
        "integration.nsamples" => data_cfg["datasets.$gfdt_role.nsamples"],
        "integration.rng_seed" => data_cfg["datasets.$gfdt_role.rng_seed"],
        "integration.process_noise_sigma" => data_cfg["twoscale.process_noise_sigma"],
        "integration.stochastic_x_noise" => data_cfg["twoscale.stochastic_x_noise"], "closure.F" => data_cfg["closure.F"],
        "closure.alpha0" => closure_theta[1],
        "closure.alpha1" => closure_theta[2],
        "closure.alpha2" => closure_theta[3],
        "closure.alpha3" => closure_theta[4],
        "closure.sigma" => closure_theta[5],
        "closure.auto_fit" => data_cfg["closure.auto_fit"],
        "closure.meta" => closure_meta,
        "observables.alpha0_ref" => obs_alpha0,
        "observables.alpha1_ref" => obs_alpha1,
        "observables.alpha2_ref" => obs_alpha2,
        "observables.alpha3_ref" => obs_alpha3,
        "observables.F_ref" => data_cfg["closure.F"], "methods.gaussian" => as_bool(methods, "gaussian", true),
        "methods.unet" => as_bool(methods, "unet", true),
        "methods.numerical" => as_bool(methods, "numerical", true),
        "methods.apply_score_correction" => as_bool(methods, "apply_score_correction", true),
        "methods.response_kind" => lowercase(as_str(methods, "response_kind", "heaviside")), "gfdt.nsamples" => as_int(gfdt, "nsamples", 90_000),
        "gfdt.start_index" => as_int(gfdt, "start_index", 25_001),
        "gfdt.mean_center" => as_bool(gfdt, "mean_center", true),
        "gfdt.batch_size" => as_int(gfdt, "batch_size", 1024),
        "gfdt.score_device" => as_str(gfdt, "score_device", "GPU:1"),
        "gfdt.score_forward_mode" => lowercase(as_str(gfdt, "score_forward_mode", "test")),
        "gfdt.response_tmax" => as_float(gfdt, "response_tmax", 2.0),
        "gfdt.output_points" => as_int(gfdt, "output_points", 301),
        "gfdt.divergence_mode" => lowercase(as_str(gfdt, "divergence_mode", "hutchinson")),
        "gfdt.divergence_eps" => as_float(gfdt, "divergence_eps", 0.03),
        "gfdt.divergence_probes" => as_int(gfdt, "divergence_probes", 2), "numerical.ensembles" => as_int(numerical, "ensembles", 2048),
        "numerical.start_index" => as_int(numerical, "start_index", 100_001),
        "numerical.h_rel" => as_float(numerical, "h_rel", 5e-3),
        "numerical.h_abs" => as_float_vec(numerical, "h_abs", [5e-2, 2e-2, 1e-2, 2e-3, 1e-2]),
        "numerical.seed_base" => as_int(numerical, "seed_base", 1_900_000),
        "numerical.max_abs_state" => as_float(numerical, "max_abs_state", 80.0),
        "numerical.min_valid_fraction" => as_float(numerical, "min_valid_fraction", 0.8),
        "numerical.max_h_shrinks" => as_int(numerical, "max_h_shrinks", 6),
        "numerical.h_shrink_factor" => as_float(numerical, "h_shrink_factor", 0.5), "figures.dpi" => as_int(figures, "dpi", 180),
        "cache.force_regenerate" => as_bool(cache, "force_regenerate", false),
    )

    cfg["methods.response_kind"] in ("heaviside", "impulse") || error("methods.response_kind must be heaviside or impulse")
    cfg["gfdt.score_forward_mode"] in ("train", "test") || error("gfdt.score_forward_mode must be train or test")
    cfg["gfdt.divergence_mode"] in ("hutchinson", "fd_axis", "exact") || error("gfdt.divergence_mode must be hutchinson, fd_axis, or exact")
    length(cfg["numerical.h_abs"]) == 5 || error("numerical.h_abs must have length 5")
    cfg["numerical.max_abs_state"] > 0.0 || error("numerical.max_abs_state must be > 0")
    0.0 <= cfg["numerical.min_valid_fraction"] <= 1.0 || error("numerical.min_valid_fraction must be in [0,1]")
    cfg["numerical.max_h_shrinks"] >= 0 || error("numerical.max_h_shrinks must be >= 0")
    0.0 < cfg["numerical.h_shrink_factor"] < 1.0 || error("numerical.h_shrink_factor must be in (0,1)")

    return cfg, doc
end

function response_dataset_signature(cfg::Dict{String,Any})
    return dict_signature(Dict(
        "path" => cfg["paths.dataset_path"],
        "key" => cfg["paths.dataset_key"],
        "role" => cfg["paths.dataset_role"],
        "data_params" => cfg["paths.data_params"],
    ))
end

function ensure_response_dataset!(cfg::Dict{String,Any})
    data_cfg, _ = load_data_config(cfg["paths.data_params"])
    info = ensure_arnold_dataset_role!(data_cfg, cfg["paths.dataset_role"])
    cfg["paths.dataset_path"] = info["path"]
    cfg["paths.dataset_key"] = info["key"]
    return info["path"]
end

function load_x_matrix(path::AbstractString, key::AbstractString;
    nsamples::Int,
    start_index::Int,
    label::AbstractString)
    isfile(path) || error("Dataset file not found: $path")
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        ds = h5[key]
        n_total = size(ds, 1)
        n_total >= 2 || error("Dataset has too few samples: $n_total")

        n_use = min(max(nsamples, 2), n_total)
        max_start = max(1, n_total - n_use + 1)
        s_use = clamp(start_index, 1, max_start)
        e_use = s_use + n_use - 1
        if n_use != nsamples || s_use != start_index
            @warn "Adjusted subset bounds" subset = label requested_nsamples = nsamples used_nsamples = n_use requested_start = start_index used_start = s_use total = n_total
        end

        raw = Float64.(ds[s_use:e_use, :])
        return permutedims(raw, (2, 1))
    end
end

function closure_theta(cfg::Dict{String,Any})
    return (
        cfg["closure.alpha0"],
        cfg["closure.alpha1"],
        cfg["closure.alpha2"],
        cfg["closure.alpha3"],
        cfg["closure.sigma"],
    )
end

function build_correction(M::Matrix{Float64}; ridge::Float64=1e-8, max_deviation::Float64=1.5)
    D = size(M, 1)
    size(M, 2) == D || error("Correction matrix must be square")
    Iden = Matrix{Float64}(I, D, D)
    pre_err = norm(M .- Iden) / sqrt(Float64(D))

    Mreg = copy(M)
    @inbounds for i in 1:D
        Mreg[i, i] += ridge
    end

    T_direct = Mreg \ Iden
    T_pinv = pinv(Mreg)

    post_direct = norm(M * T_direct .- Iden) / sqrt(Float64(D))
    post_pinv = norm(M * T_pinv .- Iden) / sqrt(Float64(D))
    dev_direct = norm(T_direct .- Iden) / sqrt(Float64(D))
    dev_pinv = norm(T_pinv .- Iden) / sqrt(Float64(D))

    solver = "direct"
    Tbest = T_direct
    post = post_direct
    dev = dev_direct
    if isfinite(post_pinv) && post_pinv < post
        solver = "pinv"
        Tbest = T_pinv
        post = post_pinv
        dev = dev_pinv
    end

    if !isfinite(post) || post > pre_err || dev > max_deviation
        Tbest = Iden
        solver = "identity_fallback"
        post = pre_err
        dev = 0.0
    end

    stats = Dict{String,Any}(
        "ridge" => ridge,
        "max_deviation" => max_deviation,
        "solver" => solver,
        "pre_identity_rmse" => pre_err,
        "post_identity_rmse" => post,
        "post_deviation" => dev,
    )
    return Tbest, stats
end

function compute_conjugates_from_score(
    X::Matrix{Float64},
    score::Matrix{Float64},
    divergence::Vector{Float64},
    sigma_param::Float64)
    K, N = size(X)
    size(score) == (K, N) || error("score shape mismatch")
    length(divergence) == N || error("divergence length mismatch")

    G = zeros(Float64, N_PARAM, N)
    @inbounds for n in 1:N
        ssum = 0.0
        xdot_s = 0.0
        x2dot_s = 0.0
        x3dot_s = 0.0
        sum_x = 0.0
        sum_x2 = 0.0
        score_sq = 0.0
        for k in 1:K
            xk = X[k, n]
            sk = score[k, n]
            x2 = xk * xk
            ssum += sk
            xdot_s += xk * sk
            x2dot_s += x2 * sk
            x3dot_s += x2 * xk * sk
            sum_x += xk
            sum_x2 += x2
            score_sq += sk * sk
        end
        G[1, n] = ssum
        G[2, n] = K + xdot_s
        G[3, n] = 2.0 * sum_x + x2dot_s
        G[4, n] = 3.0 * sum_x2 + x3dot_s
        G[5, n] = sigma_param * (divergence[n] + score_sq)
    end
    return G
end

function gaussian_conjugates(X::Matrix{Float64}, sigma_param::Float64;
    apply_correction::Bool)
    K, N = size(X)
    mu = vec(mean(X; dims=2))
    Xc = X .- mu
    Cmat = (Xc * Xc') / max(N - 1, 1)
    Cinv = inv(Symmetric(Cmat + 1e-10I))

    score_raw = -(Cinv * Xc)
    div_raw = fill(-tr(Cinv), N)

    G_raw = compute_conjugates_from_score(X, score_raw, div_raw, sigma_param)

    zstd = sqrt.(diag(Cmat) .+ 1e-12)
    z = Xc ./ zstd
    score_norm = score_raw .* zstd
    M = -(z * score_norm') / max(N, 1)

    if apply_correction
        Tcorr, corr_stats = build_correction(M)
        score_norm_corr = Tcorr' * score_norm
        score_corr = score_norm_corr ./ zstd
        Jraw = -Cinv
        Jcorr = Diagonal(1.0 ./ zstd) * (Tcorr' * (Diagonal(zstd) * Jraw * Diagonal(zstd))) * Diagonal(1.0 ./ zstd)
        div_corr = fill(tr(Jcorr), N)
        G_corr = compute_conjugates_from_score(X, score_corr, div_corr, sigma_param)
        return (
            G_raw=G_raw,
            G_corr=G_corr,
            correction_stats=corr_stats,
            correction_matrix=Tcorr,
            raw_identity_matrix=M,
        )
    end

    return (
        G_raw=G_raw,
        G_corr=G_raw,
        correction_stats=Dict{String,Any}("solver" => "disabled"),
        correction_matrix=Matrix{Float64}(I, K, K),
        raw_identity_matrix=M,
    )
end

Base.@kwdef struct UnetBundle
    model
    stats::DataStats
    sigma_train::Float32
    device
    device_name::String
    forward_mode::String
end

function resolve_device(name::AbstractString)
    try
        dev = select_device(name)
        activate_device!(dev)
        return dev, name
    catch err
        @warn "Requested score device unavailable; using CPU" requested = name error = sprint(showerror, err)
        dev = ScoreUNet1D.CPUDevice()
        activate_device!(dev)
        return dev, "CPU"
    end
end

function load_unet_bundle(checkpoint_path::AbstractString, device_pref::AbstractString, forward_mode::AbstractString)
    isfile(checkpoint_path) || error("Checkpoint not found: $checkpoint_path")
    contents = BSON.load(checkpoint_path)
    haskey(contents, :model) || error("Checkpoint missing :model")
    haskey(contents, :stats) || error("Checkpoint missing :stats")
    haskey(contents, :trainer_cfg) || error("Checkpoint missing :trainer_cfg")

    model = contents[:model]
    stats = contents[:stats]
    sigma_train = Float32(getproperty(contents[:trainer_cfg], :sigma))

    device, device_name = resolve_device(device_pref)
    model_dev = move_model(model, device)
    if lowercase(forward_mode) == "train"
        Flux.trainmode!(model_dev)
    else
        Flux.testmode!(model_dev)
    end

    return UnetBundle(
        model=model_dev,
        stats=stats,
        sigma_train=sigma_train,
        device=device,
        device_name=device_name,
        forward_mode=lowercase(forward_mode),
    )
end

function unet_score_phys_batch(bundle::UnetBundle, Xbatch::Matrix{Float64})
    K, B = size(Xbatch)
    C = 1
    mean_lc = permutedims(bundle.stats.mean, (2, 1))
    std_lc = permutedims(bundle.stats.std, (2, 1))

    xb = Array{Float32,3}(undef, K, C, B)
    @inbounds for b in 1:B, k in 1:K
        xb[k, 1, b] = (Float32(Xbatch[k, b]) - mean_lc[k, 1]) / std_lc[k, 1]
    end

    score_norm = if is_gpu(bundle.device)
        dev_batch = move_array(xb, bundle.device)
        Array(score_from_model(bundle.model, dev_batch, bundle.sigma_train))
    else
        score_from_model(bundle.model, xb, bundle.sigma_train)
    end

    out = Array{Float64}(undef, K, B)
    @inbounds for b in 1:B, k in 1:K
        out[k, b] = Float64(score_norm[k, 1, b]) / Float64(std_lc[k, 1])
    end
    return out
end

function estimate_divergence_hutchinson(
    score_fun::Function,
    X::Matrix{Float64};
    eps::Float64,
    n_probes::Int,
    rng::AbstractRNG)
    K, N = size(X)
    n_probes >= 1 || error("n_probes must be >= 1")
    div = zeros(Float64, N)
    scale = 1.0 / (2eps)

    for _ in 1:n_probes
        # Rademacher distribution: ±1 with equal probability
        R = Float64.(rand(rng, [-1.0, 1.0], K, N))
        Splus = score_fun(X .+ eps .* R)
        Sminus = score_fun(X .- eps .* R)
        D = (Splus .- Sminus) .* scale
        @inbounds for n in 1:N
            div[n] += dot(view(R, :, n), view(D, :, n))
        end
    end

    div ./= n_probes
    return div
end

function estimate_divergence_fd_axis(score_fun::Function, X::Matrix{Float64}; eps::Float64)
    K, N = size(X)
    div = zeros(Float64, N)

    for k in 1:K
        E = zeros(Float64, K, N)
        @inbounds E[k, :] .= 1.0
        Splus = score_fun(X .+ eps .* E)
        Sminus = score_fun(X .- eps .* E)
        @inbounds for n in 1:N
            div[n] += (Splus[k, n] - Sminus[k, n]) / (2eps)
        end
    end

    return div
end

function estimate_divergence_exact(bundle::UnetBundle, Xbatch::Matrix{Float64}, Tcorr_transpose::Union{Nothing,Matrix{Float64}})
    K, B = size(Xbatch)
    div_raw = zeros(Float64, B)
    div_corr = zeros(Float64, B)

    mean_lc = permutedims(bundle.stats.mean, (2, 1))
    std_lc = permutedims(bundle.stats.std, (2, 1))
    sigma_train = bundle.sigma_train
    dev = bundle.device

    xb = Array{Float32,3}(undef, K, 1, B)
    @inbounds for b in 1:B, k in 1:K
        xb[k, 1, b] = (Float32(Xbatch[k, b]) - mean_lc[k, 1]) / std_lc[k, 1]
    end

    y, back = if is_gpu(dev)
        xb_dev = move_array(xb, dev)
        Zygote.pullback(x -> bundle.model(x), xb_dev)
    else
        Zygote.pullback(x -> bundle.model(x), xb)
    end

    for j in 1:K
        dy = zeros(Float32, K, 1, B)
        dy[j, 1, :] .= 1f0
        if is_gpu(dev)
            dy = move_array(dy, dev)
        end
        grad_x_dev = back(dy)[1]
        grad_x = Array(grad_x_dev)

        @inbounds for b in 1:B
            val_raw = grad_x[j, 1, b] * (-1.0 / (sigma_train * std_lc[j, 1]^2))
            div_raw[b] += Float64(val_raw)

            if Tcorr_transpose !== nothing
                for k in 1:K
                    term = (1.0 / std_lc[k, 1]^2) * Tcorr_transpose[k, j] * grad_x[k, 1, b] * (-1.0 / sigma_train)
                    div_corr[b] += Float64(term)
                end
            end
        end
    end
    return div_raw, div_corr
end

function unet_conjugates(
    X::Matrix{Float64},
    sigma_param::Float64,
    checkpoint_path::AbstractString;
    batch_size::Int,
    score_device::String,
    score_forward_mode::String,
    apply_correction::Bool,
    divergence_mode::String,
    divergence_eps::Float64,
    divergence_probes::Int,
    divergence_seed::Int=12345)
    bundle = load_unet_bundle(checkpoint_path, score_device, score_forward_mode)
    K, N = size(X)

    score_raw = zeros(Float64, K, N)
    M_acc = zeros(Float64, K, K)

    mean_lc = permutedims(bundle.stats.mean, (2, 1))
    std_lc = permutedims(bundle.stats.std, (2, 1))
    mu_vec = Float64.(vec(mean_lc[1:K, 1]))
    std_vec = Float64.(vec(std_lc))

    for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        idx = start:stop
        xb = X[:, idx]

        s_phys = unet_score_phys_batch(bundle, xb)
        score_raw[:, idx] .= s_phys

        z = (xb .- mu_vec) ./ std_vec
        s_norm = s_phys .* std_vec
        M_acc .+= -(z * s_norm')
    end

    M_raw = M_acc ./ max(N, 1)
    Tcorr, corr_stats = if apply_correction
        build_correction(M_raw)
    else
        Matrix{Float64}(I, K, K), Dict{String,Any}("solver" => "disabled")
    end

    score_corr = if apply_correction
        score_norm = score_raw .* std_vec
        (Tcorr' * score_norm) ./ std_vec
    else
        score_raw
    end

    score_fun_raw = x -> unet_score_phys_batch(bundle, x)
    score_fun_corr = if apply_correction
        x -> begin
            s = unet_score_phys_batch(bundle, x)
            s_norm = s .* std_vec
            (Tcorr' * s_norm) ./ std_vec
        end
    else
        score_fun_raw
    end

    rng = MersenneTwister(divergence_seed)

    div_raw = zeros(Float64, N)
    div_corr = zeros(Float64, N)

    if divergence_mode == "exact"
        Tcorr_transpose = apply_correction ? Matrix(Tcorr') : nothing
        for start in 1:batch_size:N
            stop = min(start + batch_size - 1, N)
            idx = start:stop
            d_raw, d_corr = estimate_divergence_exact(bundle, X[:, idx], Tcorr_transpose)
            div_raw[idx] .= d_raw
            if apply_correction
                div_corr[idx] .= d_corr
            end
        end
        if !apply_correction
            div_corr .= div_raw
        end
    else
        div_raw = if divergence_mode == "fd_axis"
            estimate_divergence_fd_axis(score_fun_raw, X; eps=divergence_eps)
        else
            estimate_divergence_hutchinson(score_fun_raw, X; eps=divergence_eps, n_probes=divergence_probes, rng=rng)
        end

        div_corr = if apply_correction
            if divergence_mode == "fd_axis"
                estimate_divergence_fd_axis(score_fun_corr, X; eps=divergence_eps)
            else
                estimate_divergence_hutchinson(score_fun_corr, X; eps=divergence_eps, n_probes=divergence_probes, rng=rng)
            end
        else
            div_raw
        end
    end

    G_raw = compute_conjugates_from_score(X, score_raw, div_raw, sigma_param)
    G_corr = compute_conjugates_from_score(X, score_corr, div_corr, sigma_param)

    return (
        G_raw=G_raw,
        G_corr=G_corr,
        correction_stats=corr_stats,
        correction_matrix=Tcorr,
        raw_identity_matrix=M_raw,
        score_device_name=bundle.device_name,
    )
end

function observable_reference(cfg::Dict{String,Any})
    return (
        F_ref=cfg["observables.F_ref"],
        alpha0_ref=cfg["observables.alpha0_ref"],
        alpha1_ref=cfg["observables.alpha1_ref"],
        alpha2_ref=cfg["observables.alpha2_ref"],
        alpha3_ref=cfg["observables.alpha3_ref"],
    )
end

function compute_observables_for_matrix(X::Matrix{Float64}, obs_ref)
    return compute_observables_series(
        X,
        obs_ref.F_ref,
        obs_ref.alpha0_ref,
        obs_ref.alpha1_ref,
        obs_ref.alpha2_ref,
        obs_ref.alpha3_ref,
    )
end

function simulate_observable_series!(
    out::Matrix{Float64},
    x0::Vector{Float64},
    theta::NTuple{5,Float64},
    F::Float64,
    dt::Float64,
    save_every::Int,
    n_lags::Int,
    obs_ref,
    max_abs_state::Float64,
    ws,
    rng::AbstractRNG,
    x::Vector{Float64})
    copyto!(x, x0)
    if !(all(isfinite, x) && maximum(abs, x) <= max_abs_state)
        return false
    end
    out[:, 1] .= compute_observables_x(
        x,
        obs_ref.F_ref,
        obs_ref.alpha0_ref,
        obs_ref.alpha1_ref,
        obs_ref.alpha2_ref,
        obs_ref.alpha3_ref,
    )
    if !all(isfinite, @view(out[:, 1]))
        return false
    end

    a0, a1, a2, a3, sig = theta
    for lag in 1:n_lags
        for _ in 1:save_every
            l96_reduced_step!(x, dt, F, a0, a1, a2, a3, ws)
            add_reduced_noise!(x, rng, sig, dt)
            if !(all(isfinite, x) && maximum(abs, x) <= max_abs_state)
                return false
            end
        end
        out[:, lag+1] .= compute_observables_x(
            x,
            obs_ref.F_ref,
            obs_ref.alpha0_ref,
            obs_ref.alpha1_ref,
            obs_ref.alpha2_ref,
            obs_ref.alpha3_ref,
        )
        if !all(isfinite, @view(out[:, lag+1]))
            return false
        end
    end
    return true
end

function compute_numerical_responses(
    theta::NTuple{5,Float64},
    init_states::Matrix{Float64},
    cfg::Dict{String,Any},
    n_lags::Int)
    K, n_ens = size(init_states)
    h_abs = cfg["numerical.h_abs"]
    h_rel = cfg["numerical.h_rel"]
    seed_base = cfg["numerical.seed_base"]
    max_abs_state = cfg["numerical.max_abs_state"]
    min_valid_fraction = cfg["numerical.min_valid_fraction"]
    max_h_shrinks = cfg["numerical.max_h_shrinks"]
    h_shrink_factor = cfg["numerical.h_shrink_factor"]
    obs_ref = observable_reference(cfg)

    responses = zeros(Float64, N_OBS, N_PARAM, n_lags + 1)
    h_used = zeros(Float64, N_PARAM)
    valid_fraction = zeros(Float64, N_PARAM)
    valid_counts = zeros(Int, N_PARAM)
    h_attempts = zeros(Int, N_PARAM)

    partials = [zeros(Float64, N_OBS, n_lags + 1) for _ in 1:nthreads()]
    valid_locals = zeros(Int, nthreads())
    workspaces = [
        (
            x0=zeros(Float64, K),
            x=zeros(Float64, K),
            out_p=zeros(Float64, N_OBS, n_lags + 1),
            out_m=zeros(Float64, N_OBS, n_lags + 1),
            ws=make_reduced_workspace(K),
            rng=MersenneTwister(seed_base + tid),
        ) for tid in 1:nthreads()
    ]

    for ip in 1:N_PARAM
        h = max(h_abs[ip], h_rel * max(abs(theta[ip]), 1.0))
        accepted = false

        for attempt in 0:max_h_shrinks
            h_attempts[ip] = attempt + 1

            tp = collect(theta)
            tm = collect(theta)
            tp[ip] += h
            tm[ip] -= h
            theta_p = (tp[1], tp[2], tp[3], tp[4], tp[5])
            theta_m = (tm[1], tm[2], tm[3], tm[4], tm[5])

            for tid in 1:nthreads()
                fill!(partials[tid], 0.0)
                valid_locals[tid] = 0
            end

            Threads.@threads for ens in 1:n_ens
                tid = threadid()
                part = partials[tid]
                wk = workspaces[tid]

                @inbounds wk.x0 .= view(init_states, :, ens)
                seed = seed_base + 1_000_000 * ip + ens

                Random.seed!(wk.rng, seed)
                ok_p = simulate_observable_series!(
                    wk.out_p,
                    wk.x0,
                    theta_p,
                    cfg["closure.F"],
                    cfg["integration.dt"],
                    cfg["integration.save_every"],
                    n_lags,
                    obs_ref,
                    max_abs_state,
                    wk.ws,
                    wk.rng,
                    wk.x,
                )

                Random.seed!(wk.rng, seed)
                ok_m = simulate_observable_series!(
                    wk.out_m,
                    wk.x0,
                    theta_m,
                    cfg["closure.F"],
                    cfg["integration.dt"],
                    cfg["integration.save_every"],
                    n_lags,
                    obs_ref,
                    max_abs_state,
                    wk.ws,
                    wk.rng,
                    wk.x,
                )

                if !(ok_p && ok_m)
                    continue
                end

                valid_locals[tid] += 1
                @inbounds for m in 1:N_OBS, lag in 1:(n_lags+1)
                    part[m, lag] += (wk.out_p[m, lag] - wk.out_m[m, lag]) / (2h)
                end
            end

            n_valid = sum(valid_locals)
            frac_valid = n_valid / max(n_ens, 1)

            if n_valid == 0 || frac_valid < min_valid_fraction
                if attempt == max_h_shrinks
                    error("Numerical response failed for parameter $(PARAM_NAMES[ip]): valid_fraction=$(round(frac_valid, digits=4)), h=$(h), attempts=$(attempt + 1)")
                end
                @warn "Retrying numerical response with smaller perturbation step" parameter = PARAM_NAMES[ip] attempt = attempt + 1 h = h valid_fraction = frac_valid
                h *= h_shrink_factor
                continue
            end

            sum_local = zeros(Float64, N_OBS, n_lags + 1)
            for part in partials
                sum_local .+= part
            end
            sum_local ./= n_valid

            if !all(isfinite, sum_local)
                if attempt == max_h_shrinks
                    error("Numerical response remained non-finite for parameter $(PARAM_NAMES[ip]) after $(attempt + 1) attempts")
                end
                @warn "Non-finite averaged numerical response; shrinking perturbation step" parameter = PARAM_NAMES[ip] attempt = attempt + 1 h = h
                h *= h_shrink_factor
                continue
            end

            h_used[ip] = h
            valid_counts[ip] = n_valid
            valid_fraction[ip] = frac_valid
            @views responses[:, ip, :] .= sum_local
            accepted = true
            break
        end

        accepted || error("Failed to compute stable numerical response for parameter $(PARAM_NAMES[ip])")
    end

    return responses, h_used, Dict{String,Any}(
        "valid_counts" => valid_counts,
        "valid_fraction" => valid_fraction,
        "h_attempts" => h_attempts,
    )
end

function compute_steady_state_observables(theta::NTuple{5,Float64},
    cfg::Dict{String,Any},
    n_samples::Int,
    spinup::Int,
    seed::Int)
    n_samples >= 1 || error("n_samples must be >= 1")
    spinup >= 0 || error("spinup must be >= 0")

    K = cfg["integration.K"]
    F = cfg["closure.F"]
    dt = cfg["integration.dt"]
    save_every = cfg["integration.save_every"]
    max_abs_state = cfg["numerical.max_abs_state"]
    obs_ref = observable_reference(cfg)

    a0, a1, a2, a3, sig = theta
    rng = MersenneTwister(seed)
    x = zeros(Float64, K)
    ws = make_reduced_workspace(K)

    accum = zeros(Float64, N_OBS)
    collected = 0
    restarts = 0
    max_restarts = 10_000

    while collected < n_samples
        @inbounds for k in 1:K
            x[k] = F + 0.01 * randn(rng)
        end

        stable = true
        for _ in 1:spinup
            l96_reduced_step!(x, dt, F, a0, a1, a2, a3, ws)
            add_reduced_noise!(x, rng, sig, dt)
            if !(all(isfinite, x) && maximum(abs, x) <= max_abs_state)
                stable = false
                break
            end
        end

        if !stable
            restarts += 1
            if restarts > max_restarts
                error("Steady-state integration exceeded restart budget during spinup (seed=$seed, n_samples=$n_samples)")
            end
            continue
        end

        while collected < n_samples
            for _ in 1:save_every
                l96_reduced_step!(x, dt, F, a0, a1, a2, a3, ws)
                add_reduced_noise!(x, rng, sig, dt)
                if !(all(isfinite, x) && maximum(abs, x) <= max_abs_state)
                    stable = false
                    break
                end
            end
            stable || break

            obs = compute_observables_x(
                x,
                obs_ref.F_ref,
                obs_ref.alpha0_ref,
                obs_ref.alpha1_ref,
                obs_ref.alpha2_ref,
                obs_ref.alpha3_ref,
            )
            if !all(isfinite, obs)
                stable = false
                break
            end
            @inbounds for m in 1:N_OBS
                accum[m] += obs[m]
            end
            collected += 1
        end

        if !stable
            restarts += 1
            if restarts > max_restarts
                error("Steady-state integration exceeded restart budget during averaging (seed=$seed, n_samples=$n_samples)")
            end
        end
    end

    if restarts > 0
        @info "Steady-state integration used restarts" seed = seed restarts = restarts collected = collected
    end
    return accum ./ n_samples
end

function compute_numerical_jacobians(theta::NTuple{5,Float64}, cfg::Dict{String,Any})
    h_abs = cfg["numerical.h_abs"]
    h_rel = cfg["numerical.h_rel"]
    seed_base = cfg["numerical.seed_base"] + 50_000_000

    J = zeros(Float64, N_OBS, N_PARAM)

    Threads.@threads for ip in 1:N_PARAM
        h = max(h_abs[ip], h_rel * max(abs(theta[ip]), 1.0))
        tp = collect(theta)
        tm = collect(theta)
        tp[ip] += h
        tm[ip] -= h
        theta_p = (tp[1], tp[2], tp[3], tp[4], tp[5])
        theta_m = (tm[1], tm[2], tm[3], tm[4], tm[5])

        seed = seed_base + 1000 * ip
        @info "Computing high-precision asymptotic FD Jacobian column" parameter = PARAM_NAMES[ip] h = h n_samples = ASYMPTOTIC_FD_NSAMPLES spinup = ASYMPTOTIC_FD_SPINUP seed = seed
        avg_p = compute_steady_state_observables(theta_p, cfg, ASYMPTOTIC_FD_NSAMPLES, ASYMPTOTIC_FD_SPINUP, seed)
        avg_m = compute_steady_state_observables(theta_m, cfg, ASYMPTOTIC_FD_NSAMPLES, ASYMPTOTIC_FD_SPINUP, seed)
        @views J[:, ip] .= (avg_p .- avg_m) ./ (2h)
    end

    all(isfinite, J) || error("Asymptotic finite-difference Jacobian contains non-finite values")
    return J
end

function compute_rmse_summary(R_est::Array{Float64,3}, R_ref::Array{Float64,3})
    size(R_est) == size(R_ref) || error("RMSE arrays must have same shape")
    out = Dict{String,Any}()
    out["overall"] = finite_rmse(R_est, R_ref)
    per_param = Dict{String,Float64}()
    for (j, name) in enumerate(PARAM_NAMES)
        per_param[name] = finite_rmse(@view(R_est[:, j, :]), @view(R_ref[:, j, :]))
    end
    out["per_param"] = per_param
    return out
end

function asymptotic_window_indices(times::Vector{Float64}; tail_window::Float64=1.0)
    isempty(times) && error("times must be non-empty")
    t_end = times[end]
    t_start = t_end - max(tail_window, 0.0)
    idx = findall(t -> t >= t_start && t <= t_end, times)
    isempty(idx) && return [length(times)]
    return idx
end

function extract_asymptotic_jacobians(times::Vector{Float64}, responses::Array{Float64,3}; tail_window::Float64=1.0)
    size(responses, 3) == length(times) || error("Response time dimension mismatch")
    idx = asymptotic_window_indices(times; tail_window=tail_window)
    return dropdims(mean(@view(responses[:, :, idx]); dims=3), dims=3)
end

function compute_jacobian_distance(num_jacobians::Matrix{Float64},
    model_jacobians::Matrix{Float64};
    epsilon::Float64=1e-8)
    size(num_jacobians) == size(model_jacobians) || error("Jacobian shapes must match")
    numel = length(num_jacobians)
    numel > 0 || error("Jacobian arrays must be non-empty")
    acc = 0.0
    @inbounds for idx in eachindex(num_jacobians, model_jacobians)
        n = num_jacobians[idx]
        m = model_jacobians[idx]
        acc += abs(n - m) / (abs(n) + abs(m) + epsilon)
    end
    return acc / numel
end

function response_signature(cfg::Dict{String,Any})
    return dict_signature(Dict(
        "dataset_path" => cfg["paths.dataset_path"],
        "dataset_key" => cfg["paths.dataset_key"],
        "integration" => Dict(
            "K" => cfg["integration.K"],
            "J" => cfg["integration.J"],
            "F" => cfg["integration.F"],
            "h" => cfg["integration.h"],
            "c" => cfg["integration.c"],
            "b" => cfg["integration.b"],
            "dt" => cfg["integration.dt"],
            "save_every" => cfg["integration.save_every"],
            "rng_seed" => cfg["integration.rng_seed"],
            "nsamples" => cfg["integration.nsamples"],
        ),
        "closure" => Dict(
            "F" => cfg["closure.F"],
            "alpha0" => cfg["closure.alpha0"],
            "alpha1" => cfg["closure.alpha1"],
            "alpha2" => cfg["closure.alpha2"],
            "alpha3" => cfg["closure.alpha3"],
            "sigma" => cfg["closure.sigma"],
            "auto_fit" => cfg["closure.auto_fit"],
        ),
        "observables" => Dict(
            "F_ref" => cfg["observables.F_ref"],
            "alpha0_ref" => cfg["observables.alpha0_ref"],
            "alpha1_ref" => cfg["observables.alpha1_ref"],
            "alpha2_ref" => cfg["observables.alpha2_ref"],
            "alpha3_ref" => cfg["observables.alpha3_ref"],
            "names" => OBS_NAMES,
            "n_observables" => N_OBS,
        ),
        "gfdt" => Dict(
            "nsamples" => cfg["gfdt.nsamples"],
            "start_index" => cfg["gfdt.start_index"],
            "mean_center" => cfg["gfdt.mean_center"],
            "response_tmax" => cfg["gfdt.response_tmax"],
            "output_points" => cfg["gfdt.output_points"],
            "batch_size" => cfg["gfdt.batch_size"],
        ),
        "numerical" => Dict(
            "ensembles" => cfg["numerical.ensembles"],
            "start_index" => cfg["numerical.start_index"],
            "h_rel" => cfg["numerical.h_rel"],
            "h_abs" => cfg["numerical.h_abs"],
            "seed_base" => cfg["numerical.seed_base"],
            "max_abs_state" => cfg["numerical.max_abs_state"],
            "min_valid_fraction" => cfg["numerical.min_valid_fraction"],
            "max_h_shrinks" => cfg["numerical.max_h_shrinks"],
            "h_shrink_factor" => cfg["numerical.h_shrink_factor"],
        ),
    ))
end

function baseline_cache_path(cfg::Dict{String,Any})
    sig = response_signature(cfg)
    digest = bytes2hex(sha1(sig))
    mkpath(cfg["paths.cache_root"])
    return joinpath(cfg["paths.cache_root"], "baseline_" * digest * ".hdf5"), sig
end

function write_baseline_cache(path::AbstractString, signature::AbstractString, payload)
    mkpath(dirname(path))
    h5open(path, "w") do h5
        attrs = attributes(h5)
        attrs["signature"] = signature
        attrs["generated_at"] = string(now())
        attrs["h_used"] = payload.h_used
        if haskey(payload.numerical_diag, "valid_fraction")
            attrs["valid_fraction"] = payload.numerical_diag["valid_fraction"]
        end
        if haskey(payload.numerical_diag, "valid_counts")
            attrs["valid_counts"] = payload.numerical_diag["valid_counts"]
        end
        if haskey(payload.numerical_diag, "h_attempts")
            attrs["h_attempts"] = payload.numerical_diag["h_attempts"]
        end

        h5["times_out"] = payload.times_out
        h5["asymptotic_jacobians"] = payload.asymptotic_jacobians
        for (name, arr) in payload.step_map
            h5[joinpath("responses", "heaviside", name)] = arr
        end
        for (name, arr) in payload.impulse_map
            h5[joinpath("responses", "impulse", name)] = arr
        end
    end
    return path
end

function load_baseline_cache(path::AbstractString, signature::AbstractString; response_kind::String)
    isfile(path) || error("Baseline cache not found: $path")
    return h5open(path, "r") do h5
        attrs = attributes(h5)
        haskey(attrs, "signature") || error("Cache file missing signature")
        stored = String(read(attrs["signature"]))
        stored == signature || error("Cache signature mismatch")

        times = Float64.(read(h5["times_out"]))
        bucket = response_kind == "impulse" ? "impulse" : "heaviside"
        grp = h5[joinpath("responses", bucket)]
        out = Dict{String,Array{Float64,3}}()
        for key in keys(grp)
            out[String(key)] = Float64.(read(grp[key]))
        end

        h_used = haskey(attrs, "h_used") ? Float64.(read(attrs["h_used"])) : fill(NaN, N_PARAM)
        valid_fraction = haskey(attrs, "valid_fraction") ? Float64.(read(attrs["valid_fraction"])) : fill(NaN, N_PARAM)
        valid_counts = haskey(attrs, "valid_counts") ? Int.(round.(Float64.(read(attrs["valid_counts"])))) : fill(0, N_PARAM)
        h_attempts = haskey(attrs, "h_attempts") ? Int.(round.(Float64.(read(attrs["h_attempts"])))) : fill(0, N_PARAM)
        asymptotic_jacobians = haskey(h5, "asymptotic_jacobians") ? Float64.(read(h5["asymptotic_jacobians"])) : nothing
        return (
            times=times,
            responses=out,
            h_used=h_used,
            numerical_diag=Dict{String,Any}(
                "valid_fraction" => valid_fraction,
                "valid_counts" => valid_counts,
                "h_attempts" => h_attempts,
            ),
            asymptotic_jacobians=asymptotic_jacobians,
        )
    end
end

function cache_has_asymptotic_jacobians(path::AbstractString)
    isfile(path) || return false
    return h5open(path, "r") do h5
        haskey(h5, "asymptotic_jacobians")
    end
end

function cache_matches(path::AbstractString, signature::AbstractString)
    isfile(path) || return false
    return h5open(path, "r") do h5
        attrs = attributes(h5)
        haskey(attrs, "signature") || return false
        stored = try
            String(read(attrs["signature"]))
        catch
            return false
        end
        return stored == signature
    end
end

function compute_baseline_payload(cfg::Dict{String,Any})
    theta = closure_theta(cfg)
    dt_obs = cfg["integration.dt"] * cfg["integration.save_every"]
    obs_ref = observable_reference(cfg)

    Xgfdt = load_x_matrix(
        cfg["paths.dataset_path"],
        cfg["paths.dataset_key"];
        nsamples=cfg["gfdt.nsamples"],
        start_index=cfg["gfdt.start_index"],
        label="gfdt",
    )

    n_lags_req = max(1, Int(floor(cfg["gfdt.response_tmax"] / dt_obs)))
    n_lags = min(n_lags_req, size(Xgfdt, 2) - 1)
    n_lags >= 1 || error("Not enough GFDT samples for selected response_tmax")

    A = compute_observables_for_matrix(Xgfdt, obs_ref)
    gauss = gaussian_conjugates(Xgfdt, theta[5]; apply_correction=cfg["methods.apply_score_correction"])

    Cg_raw, Rg_raw_step, times_native = build_gfdt_response(A, gauss.G_raw, dt_obs, n_lags; mean_center=cfg["gfdt.mean_center"])
    Cg_corr, Rg_corr_step, _ = build_gfdt_response(A, gauss.G_corr, dt_obs, n_lags; mean_center=cfg["gfdt.mean_center"])

    Xinit = load_x_matrix(
        cfg["paths.dataset_path"],
        cfg["paths.dataset_key"];
        nsamples=cfg["numerical.ensembles"],
        start_index=cfg["numerical.start_index"],
        label="numerical",
    )

    Rnum_step, h_used, num_diag = compute_numerical_responses(theta, Xinit, cfg, n_lags)
    all(isfinite, Rnum_step) || error("Numerical response contains non-finite values after stabilization")
    Rnum_impulse = step_to_impulse(Rnum_step, times_native)
    asymptotic_jacobians = compute_numerical_jacobians(theta, cfg)

    times_out = response_output_times(cfg["gfdt.response_tmax"]; npoints=cfg["gfdt.output_points"])

    step_map = Dict{String,Array{Float64,3}}(
        "gfdt_gaussian_raw" => linear_interpolate_3d_time(Rg_raw_step, times_native, times_out),
        "gfdt_gaussian_corrected" => linear_interpolate_3d_time(Rg_corr_step, times_native, times_out),
        "numerical_integration" => linear_interpolate_3d_time(Rnum_step, times_native, times_out),
    )

    impulse_map = Dict{String,Array{Float64,3}}(
        "gfdt_gaussian_raw" => linear_interpolate_3d_time(Cg_raw, times_native, times_out),
        "gfdt_gaussian_corrected" => linear_interpolate_3d_time(Cg_corr, times_native, times_out),
        "numerical_integration" => linear_interpolate_3d_time(Rnum_impulse, times_native, times_out),
    )

    return (
        times_out=times_out,
        step_map=step_map,
        impulse_map=impulse_map,
        asymptotic_jacobians=asymptotic_jacobians,
        h_used=h_used,
        numerical_diag=num_diag,
        theta=theta,
        gauss_corr_stats=gauss.correction_stats,
    )
end

function ensure_baseline_cache!(cfg::Dict{String,Any})
    path, sig = baseline_cache_path(cfg)
    force = cfg["cache.force_regenerate"]
    if !force && cache_matches(path, sig) && cache_has_asymptotic_jacobians(path)
        return Dict{String,Any}("path" => path, "signature" => sig, "generated" => false)
    end
    payload = compute_baseline_payload(cfg)
    write_baseline_cache(path, sig, payload)
    return Dict{String,Any}("path" => path, "signature" => sig, "generated" => true)
end

function save_response_figure(path::AbstractString,
    times::Vector{Float64},
    curves::Vector{NamedTuple};
    title_text::String,
    dpi::Int)
    isempty(curves) && return ""

    R0 = curves[1].data
    m, p, nt = size(R0)
    nt == length(times) || error("Curve time dimension mismatch")
    asymptotic_by_curve = [(;
        label=c.label,
        color=c.color,
        linestyle=c.linestyle,
        jacobians=extract_asymptotic_jacobians(times, c.data),
    ) for c in curves]

    default(fontfamily="Computer Modern", dpi=dpi, legendfontsize=8, guidefontsize=9, tickfontsize=8, titlefontsize=10)
    panels = Vector{Plots.Plot}(undef, m * p)

    for i in 1:m, j in 1:p
        idx = (i - 1) * p + j
        legend_mode = idx == 1 ? :topright : false
        title_txt = i == 1 ? "d/d" * PARAM_NAMES[j] : ""
        ylabel_txt = j == 1 ? OBS_LABELS[i] : ""
        xlabel_txt = i == m ? "time" : ""

        pn = plot(; legend=legend_mode, title=title_txt, ylabel=ylabel_txt, xlabel=xlabel_txt)
        for c in curves
            plot!(pn, times, vec(@view c.data[i, j, :]); color=c.color, linestyle=c.linestyle, linewidth=2.0, label=(idx == 1 ? c.label : ""))
        end
        for ac in asymptotic_by_curve
            hline!(pn, [ac.jacobians[i, j]]; color=ac.color, linestyle=ac.linestyle, linewidth=1.8, label=(idx == 1 ? ac.label * " asymptote" : ""))
        end
        plot!(pn; left_margin=12Plots.mm, right_margin=6Plots.mm, top_margin=8Plots.mm, bottom_margin=10Plots.mm)
        panels[idx] = pn
    end

    fig = plot(panels...; layout=(m, p), size=(2550, 1850), left_margin=6Plots.mm, right_margin=6Plots.mm, top_margin=8Plots.mm, bottom_margin=6Plots.mm, plot_title=title_text, plot_titlefontsize=13)
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function write_responses_hdf5(path::AbstractString,
    times::Vector{Float64},
    responses::Dict{String,Array{Float64,3}})
    mkpath(dirname(path))
    h5open(path, "w") do h5
        h5["times"] = times
        for (name, arr) in responses
            h5[joinpath("responses", name)] = arr
        end
    end
    return path
end

function write_responses_csv(path::AbstractString,
    times::Vector{Float64},
    responses::Dict{String,Array{Float64,3}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,observable,param,time,response")
        for name in sort!(collect(keys(responses)))
            R = responses[name]
            for i in 1:N_OBS, j in 1:N_PARAM, t in eachindex(times)
                @printf(io, "%s,%s,%s,%.12e,%.12e\n", name, OBS_NAMES[i], PARAM_NAMES[j], times[t], R[i, j, t])
            end
        end
    end
    return path
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg, raw_doc = load_config(parsed.params_path)

    if !isempty(strip(parsed.response_kind_override))
        cfg["methods.response_kind"] = parsed.response_kind_override
    end
    if parsed.correction_override !== nothing
        cfg["methods.apply_score_correction"] = parsed.correction_override
    end

    ensure_response_dataset!(cfg)

    if !isempty(strip(parsed.output_override))
        cfg["paths.output_root"] = abspath(parsed.output_override)
    end

    out_root = cfg["paths.output_root"]
    mkpath(out_root)
    run_info = next_run_dir(out_root)
    out_dir = run_info.run_dir
    mkpath(out_dir)

    run_cfg_path = joinpath(out_dir, "parameters_used.toml")
    open(run_cfg_path, "w") do io
        TOML.print(io, raw_doc)
    end

    cache_info = ensure_baseline_cache!(cfg)
    baseline = load_baseline_cache(cache_info["path"], cache_info["signature"]; response_kind=cfg["methods.response_kind"])

    times_out = baseline.times
    responses = Dict{String,Array{Float64,3}}()

    if cfg["methods.gaussian"]
        if cfg["methods.apply_score_correction"]
            responses["gfdt_gaussian"] = baseline.responses["gfdt_gaussian_corrected"]
            responses["gfdt_gaussian_raw"] = baseline.responses["gfdt_gaussian_raw"]
            responses["gfdt_gaussian_corrected"] = baseline.responses["gfdt_gaussian_corrected"]
        else
            responses["gfdt_gaussian"] = baseline.responses["gfdt_gaussian_raw"]
            responses["gfdt_gaussian_raw"] = baseline.responses["gfdt_gaussian_raw"]
        end
    end

    if cfg["methods.numerical"]
        responses["numerical_integration"] = baseline.responses["numerical_integration"]
    end

    rmse = Dict{String,Any}()

    checkpoint_path = if isempty(strip(parsed.checkpoint_override))
        strip(cfg["paths.checkpoint_path"])
    else
        strip(parsed.checkpoint_override)
    end

    if cfg["methods.unet"]
        isempty(checkpoint_path) && error("UNet method enabled but no checkpoint provided (set [paths].checkpoint_path or --checkpoint)")
        checkpoint_path = abspath(checkpoint_path)

        Xgfdt = load_x_matrix(
            cfg["paths.dataset_path"],
            cfg["paths.dataset_key"];
            nsamples=cfg["gfdt.nsamples"],
            start_index=cfg["gfdt.start_index"],
            label="gfdt_unet",
        )
        theta = closure_theta(cfg)
        obs_ref = observable_reference(cfg)

        dt_obs = cfg["integration.dt"] * cfg["integration.save_every"]
        n_lags_req = max(1, Int(floor(cfg["gfdt.response_tmax"] / dt_obs)))
        n_lags = min(n_lags_req, size(Xgfdt, 2) - 1)
        A = compute_observables_for_matrix(Xgfdt, obs_ref)

        unet = unet_conjugates(
            Xgfdt,
            theta[5],
            checkpoint_path;
            batch_size=cfg["gfdt.batch_size"],
            score_device=cfg["gfdt.score_device"],
            score_forward_mode=cfg["gfdt.score_forward_mode"],
            apply_correction=cfg["methods.apply_score_correction"],
            divergence_mode=cfg["gfdt.divergence_mode"],
            divergence_eps=cfg["gfdt.divergence_eps"],
            divergence_probes=cfg["gfdt.divergence_probes"],
        )

        C_u_raw, R_u_raw_step, times_native = build_gfdt_response(A, unet.G_raw, dt_obs, n_lags; mean_center=cfg["gfdt.mean_center"])
        C_u_corr, R_u_corr_step, _ = build_gfdt_response(A, unet.G_corr, dt_obs, n_lags; mean_center=cfg["gfdt.mean_center"])

        R_u_raw = if cfg["methods.response_kind"] == "impulse"
            linear_interpolate_3d_time(C_u_raw, times_native, times_out)
        else
            linear_interpolate_3d_time(R_u_raw_step, times_native, times_out)
        end
        R_u_corr = if cfg["methods.response_kind"] == "impulse"
            linear_interpolate_3d_time(C_u_corr, times_native, times_out)
        else
            linear_interpolate_3d_time(R_u_corr_step, times_native, times_out)
        end

        responses["gfdt_unet_raw"] = R_u_raw
        responses["gfdt_unet_corrected"] = R_u_corr
        responses["gfdt_unet"] = cfg["methods.apply_score_correction"] ? R_u_corr : R_u_raw

        if haskey(responses, "numerical_integration")
            rmse["numerics_vs_unet_raw"] = compute_rmse_summary(R_u_raw, responses["numerical_integration"])
            rmse["numerics_vs_unet_corrected"] = compute_rmse_summary(R_u_corr, responses["numerical_integration"])
        end
    end

    if haskey(responses, "numerical_integration") && haskey(responses, "gfdt_gaussian")
        rmse["numerics_vs_gaussian"] = compute_rmse_summary(responses["gfdt_gaussian"], responses["numerical_integration"])
    end

    title_rmse_gauss = haskey(rmse, "numerics_vs_gaussian") ? rmse["numerics_vs_gaussian"]["overall"] : NaN
    title_rmse_unet = if haskey(rmse, "numerics_vs_unet_corrected")
        rmse["numerics_vs_unet_corrected"]["overall"]
    elseif haskey(rmse, "numerics_vs_unet_raw")
        rmse["numerics_vs_unet_raw"]["overall"]
    else
        NaN
    end

    jacobian_asymptotes = Dict{String,Matrix{Float64}}()
    haskey(responses, "numerical_integration") && (jacobian_asymptotes["numerical_integration"] = extract_asymptotic_jacobians(times_out, responses["numerical_integration"]))
    haskey(responses, "gfdt_gaussian") && (jacobian_asymptotes["gfdt_gaussian"] = extract_asymptotic_jacobians(times_out, responses["gfdt_gaussian"]))
    haskey(responses, "gfdt_unet") && (jacobian_asymptotes["gfdt_unet"] = extract_asymptotic_jacobians(times_out, responses["gfdt_unet"]))

    jacobian_distance = Dict{String,Float64}()
    if haskey(jacobian_asymptotes, "numerical_integration") && haskey(jacobian_asymptotes, "gfdt_gaussian")
        jacobian_distance["gaussian_vs_numerical"] = compute_jacobian_distance(
            jacobian_asymptotes["numerical_integration"],
            jacobian_asymptotes["gfdt_gaussian"],
        )
    end
    if haskey(jacobian_asymptotes, "numerical_integration") && haskey(jacobian_asymptotes, "gfdt_unet")
        jacobian_distance["unet_vs_numerical"] = compute_jacobian_distance(
            jacobian_asymptotes["numerical_integration"],
            jacobian_asymptotes["gfdt_unet"],
        )
    end

    title_text = @sprintf("Responses (%s) | avg RMSE num-gauss=%.4f | avg RMSE num-unet=%.4f", cfg["methods.response_kind"], title_rmse_gauss, title_rmse_unet)

    curves = NamedTuple[]
    haskey(responses, "numerical_integration") && push!(curves, (label="Numerical", color=:dodgerblue3, linestyle=:solid, data=responses["numerical_integration"]))
    haskey(responses, "gfdt_gaussian") && push!(curves, (label="GFDT+Gaussian", color=:black, linestyle=:dash, data=responses["gfdt_gaussian"]))
    haskey(responses, "gfdt_unet") && push!(curves, (label="GFDT+UNet", color=:orangered3, linestyle=:solid, data=responses["gfdt_unet"]))

    fig_main = save_response_figure(joinpath(out_dir, "responses_5x5_selected_methods.png"), times_out, curves; title_text=title_text, dpi=cfg["figures.dpi"])

    fig_raw = ""
    if haskey(responses, "gfdt_unet_raw") || haskey(responses, "gfdt_gaussian_raw")
        curves_raw = NamedTuple[]
        haskey(responses, "numerical_integration") && push!(curves_raw, (label="Numerical", color=:dodgerblue3, linestyle=:solid, data=responses["numerical_integration"]))
        haskey(responses, "gfdt_gaussian_raw") && push!(curves_raw, (label="GFDT+Gaussian raw", color=:black, linestyle=:dash, data=responses["gfdt_gaussian_raw"]))
        haskey(responses, "gfdt_unet_raw") && push!(curves_raw, (label="GFDT+UNet raw", color=:orangered3, linestyle=:solid, data=responses["gfdt_unet_raw"]))
        fig_raw = save_response_figure(joinpath(out_dir, "responses_5x5_selected_methods_raw.png"), times_out, curves_raw; title_text=title_text * " (raw)", dpi=cfg["figures.dpi"])
    end

    fig_corr = ""
    if haskey(responses, "gfdt_unet_corrected") || haskey(responses, "gfdt_gaussian_corrected")
        curves_corr = NamedTuple[]
        haskey(responses, "numerical_integration") && push!(curves_corr, (label="Numerical", color=:dodgerblue3, linestyle=:solid, data=responses["numerical_integration"]))
        haskey(responses, "gfdt_gaussian_corrected") && push!(curves_corr, (label="GFDT+Gaussian corrected", color=:black, linestyle=:dash, data=responses["gfdt_gaussian_corrected"]))
        haskey(responses, "gfdt_unet_corrected") && push!(curves_corr, (label="GFDT+UNet corrected", color=:orangered3, linestyle=:solid, data=responses["gfdt_unet_corrected"]))
        fig_corr = save_response_figure(joinpath(out_dir, "responses_5x5_selected_methods_corrected.png"), times_out, curves_corr; title_text=title_text * " (corrected)", dpi=cfg["figures.dpi"])
    end

    h5_path = write_responses_hdf5(joinpath(out_dir, "responses_5x5_selected_methods.hdf5"), times_out, responses)
    csv_path = write_responses_csv(joinpath(out_dir, "responses_5x5_selected_methods.csv"), times_out, responses)

    summary = Dict{String,Any}(
        "run_id" => run_info.run_name,
        "output_dir" => abspath(out_dir),
        "checkpoint_path" => checkpoint_path,
        "response_kind" => cfg["methods.response_kind"],
        "apply_score_correction" => cfg["methods.apply_score_correction"],
        "cache_path" => cache_info["path"],
        "cache_regenerated" => cache_info["generated"],
        "h_used" => baseline.h_used,
        "numerical_diag" => baseline.numerical_diag,
        "rmse" => rmse,
        "jacobian_distance_smape" => Dict{String,Any}(k => v for (k, v) in jacobian_distance),
        "figures" => Dict(
            "main" => abspath(fig_main),
            "raw" => fig_raw,
            "corrected" => fig_corr,
        ),
        "outputs" => Dict(
            "responses_hdf5" => abspath(h5_path),
            "responses_csv" => abspath(csv_path),
            "parameters_used" => abspath(run_cfg_path),
        ),
        "closure_parameters_used" => Dict(
            "F" => cfg["closure.F"],
            "alpha0" => cfg["closure.alpha0"],
            "alpha1" => cfg["closure.alpha1"],
            "alpha2" => cfg["closure.alpha2"],
            "alpha3" => cfg["closure.alpha3"],
            "sigma" => cfg["closure.sigma"],
            "auto_fit" => cfg["closure.auto_fit"],
        ),
        "closure_meta" => cfg["closure.meta"],
        "observable_reference" => Dict(
            "F_ref" => cfg["observables.F_ref"],
            "alpha0_ref" => cfg["observables.alpha0_ref"],
            "alpha1_ref" => cfg["observables.alpha1_ref"],
            "alpha2_ref" => cfg["observables.alpha2_ref"],
            "alpha3_ref" => cfg["observables.alpha3_ref"],
        ),
    )

    summary_path = joinpath(out_dir, "responses_5x5_summary.toml")
    open(summary_path, "w") do io
        TOML.print(io, summary)
    end

    if haskey(jacobian_distance, "gaussian_vs_numerical")
        println(@sprintf("jacobian_smape_gaussian_vs_numerical=%.10f", jacobian_distance["gaussian_vs_numerical"]))
    end
    if haskey(jacobian_distance, "unet_vs_numerical")
        println(@sprintf("jacobian_smape_unet_vs_numerical=%.10f", jacobian_distance["unet_vs_numerical"]))
    end
    println("output_dir=$(abspath(out_dir))")
    println("figure_main=$(abspath(fig_main))")
    println("responses_hdf5=$(abspath(h5_path))")
    println("responses_csv=$(abspath(csv_path))")
    println("summary=$(abspath(summary_path))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
