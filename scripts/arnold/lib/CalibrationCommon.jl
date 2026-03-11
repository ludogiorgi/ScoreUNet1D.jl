module CalibrationCommon

using BSON
using CUDA
using Flux
using Functors
using HDF5
using LinearAlgebra
using Plots
using Printf
using Random
using ScoreUNet1D
using Statistics
using TOML
using Dates

import Main.ArnoldCommon

const HAS_PARAMETER_CALIBRATION = let
    ok = false
    try
        @eval import ParameterCalibration: newton_step, weight_inverse_cov
        ok = true
    catch
        ok = false
    end
    ok
end

export load_calibration_config,
       compute_target_observables,
       generate_iteration_datasets,
       train_iteration_score,
       apply_run_runtime_overrides!,
       compute_iteration_jacobians,
       perform_newton_update,
       check_convergence,
       save_iteration_outputs,
       save_convergence_figure,
       estimate_observables,
       estimate_observables_for_acceptance,
       estimate_observables_for_convergence,
       weighted_observable_residual,
       CalibrationState

const PARAM_NAMES = ["alpha0", "alpha1", "alpha2", "alpha3", "sigma"]
const RUN_OFFSET_SEED_KEYS = (
    "datasets.train_rng_seed_base",
    "training.seed",
    "figures.langevin_seed_base",
)

function observable_names(m::Int)
    m >= 0 || error("Observable lag parameter m must be >= 0")
    out = String["phi1_mean_x", "phi2_mean_x2"]
    for lag in 1:m
        push!(out, "phi$(lag + 2)_mean_x_xp$(lag)")
    end
    return out
end

function observable_labels(m::Int)
    m >= 0 || error("Observable lag parameter m must be >= 0")
    out = String["phi1=<X_k>", "phi2=<X_k^2>"]
    for lag in 1:m
        push!(out, "phi$(lag + 2)=<X_k X_{k+$(lag)}>")
    end
    return out
end

observable_count(m::Int) = m + 2
observable_count(cfg::Dict{String,Any}) = Int(get(cfg, "observables.n", observable_count(Int(get(cfg, "observables.m", 3)))))

Base.@kwdef mutable struct CalibrationState
    iteration::Int = 0
    theta::Vector{Float64} = Float64[]
    theta_history::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    theta_per_method::Dict{String,Vector{Vector{Float64}}} = Dict{String,Vector{Vector{Float64}}}()
    obs_history::Dict{String,Vector{Vector{Float64}}} = Dict{String,Vector{Vector{Float64}}}()
    jacobian_history::Dict{String,Vector{Matrix{Float64}}} = Dict{String,Vector{Matrix{Float64}}}()
    converged::Bool = false
    convergence_metric::Float64 = Inf
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_bool(tbl::Dict{String,Any}, key::String, default) = ArnoldCommon.parse_bool(get(tbl, key, default))
as_int_vec(tbl::Dict{String,Any}, key::String, default) = Int.(collect(get(tbl, key, default)))
as_float_vec(tbl::Dict{String,Any}, key::String, default) = Float64.(collect(get(tbl, key, default)))

function _device_override(env_key::AbstractString, default::AbstractString)
    value = strip(get(ENV, String(env_key), ""))
    return isempty(value) ? String(default) : value
end

function require_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa Dict{String,Any} || error("[$key] must be TOML table")
    return Dict{String,Any}(doc[key])
end

function maybe_table(doc::Dict{String,Any}, key::String)
    if !haskey(doc, key)
        return Dict{String,Any}()
    end
    doc[key] isa Dict{String,Any} || error("[$key] must be TOML table")
    return Dict{String,Any}(doc[key])
end

function maybe_subtable(tbl::Dict{String,Any}, key::String)
    if !haskey(tbl, key)
        return Dict{String,Any}()
    end
    tbl[key] isa Dict{String,Any} || error("[$key] must be TOML table")
    return Dict{String,Any}(tbl[key])
end

function apply_run_runtime_overrides!(cfg::Dict{String,Any}, run_id::Int, run_name::AbstractString)
    Bool(get(cfg, "runtime.run_seed_offset_applied", false)) && return cfg

    cfg["runtime.run_id"] = Int(run_id)
    cfg["runtime.run_name"] = String(run_name)

    seed_stride = Int(get(cfg, "run.seed_stride", 1_000_000_000))
    manual_seed_offset = Int(get(cfg, "run.seed_offset", 0))
    run_seed_offset = manual_seed_offset + max(run_id - 1, 0) * seed_stride
    cfg["runtime.run_seed_offset"] = run_seed_offset

    if run_seed_offset != 0
        for key in RUN_OFFSET_SEED_KEYS
            cfg[key] = Int(cfg[key]) + run_seed_offset
        end
        cfg["numerical.seed_base"] = cfg["responses.finite_difference.seed_base"]
    end

    shared_gpu = strip(get(ENV, "ARNOLD_GPU_DEVICE", ""))
    cfg["training.device"] = _device_override("ARNOLD_TRAIN_DEVICE", isempty(shared_gpu) ? cfg["training.device"] : shared_gpu)
    cfg["responses.score_device"] = _device_override("ARNOLD_SCORE_DEVICE", isempty(shared_gpu) ? cfg["responses.score_device"] : shared_gpu)
    cfg["figures.langevin_device"] = _device_override("ARNOLD_LANGEVIN_DEVICE", isempty(shared_gpu) ? cfg["figures.langevin_device"] : shared_gpu)
    cfg["runtime.run_seed_offset_applied"] = true

    return cfg
end

function require_main_symbol(sym::Symbol)
    isdefined(Main, sym) || error("Missing required symbol in Main: $sym. Ensure compute_responses.jl is included before calling calibration helpers that need it.")
    return getproperty(Main, sym)
end

function reclaim_device_memory!()
    if isdefined(Main, :reclaim_device_memory!)
        getproperty(Main, :reclaim_device_memory!)()
        return nothing
    end
    GC.gc(true)
    try
        CUDA.reclaim()
    catch
    end
    return nothing
end

function normalize_indices(raw::Vector{Int}, nmax::Int, label::String)
    if isempty(raw)
        return collect(1:nmax)
    end
    all(1 .<= raw .<= nmax) || error("$label indices must be in 1:$nmax")
    return sort(unique(raw))
end

function method_order()
    return ["unet", "gaussian", "finite_difference"]
end

function enabled_methods(cfg::Dict{String,Any})
    return [m for m in method_order() if get(cfg, "methods.$m", false)]
end

function obs_ref_tuple(cfg::Dict{String,Any})
    return (
        F_ref=cfg["observables.F_ref"],
        alpha0_ref=cfg["observables.alpha0_ref"],
        alpha1_ref=cfg["observables.alpha1_ref"],
        alpha2_ref=cfg["observables.alpha2_ref"],
        alpha3_ref=cfg["observables.alpha3_ref"],
    )
end

function theta_to_tuple(theta::AbstractVector{<:Real})
    length(theta) == 5 || error("theta must have length 5")
    return (Float64(theta[1]), Float64(theta[2]), Float64(theta[3]), Float64(theta[4]), Float64(theta[5]))
end

function detect_dataset_key(path::AbstractString)
    return h5open(path, "r") do h5
        for key in keys(h5)
            obj = h5[key]
            if obj isa HDF5.Dataset
                return String(key)
            end
        end
        error("No datasets found in $path")
    end
end

function load_x_matrix(path::AbstractString, key::AbstractString, K_expected::Int)
    isfile(path) || error("Dataset file not found: $path")
    key_use = isempty(strip(String(key))) ? detect_dataset_key(path) : String(key)
    raw = h5open(path, "r") do h5
        haskey(h5, key_use) || error("Dataset key '$key_use' not found in $path")
        Float64.(read(h5[key_use]))
    end

    ndims(raw) == 2 || error("Expected 2D dataset at $path/$key_use")
    if size(raw, 2) == K_expected
        return permutedims(raw, (2, 1))
    elseif size(raw, 1) == K_expected
        return raw
    end
    error("Cannot infer orientation for $path/$key_use with shape $(size(raw)) and K=$K_expected")
end

function write_vector_csv(path::AbstractString, names::Vector{String}, vals::Vector{Float64}; prefix::String="")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "name,value")
        for (n, v) in zip(names, vals)
            if isempty(prefix)
                @printf(io, "%s,%.16e\n", n, v)
            else
                @printf(io, "%s%s,%.16e\n", prefix, n, v)
            end
        end
    end
    return path
end

function tensor_from_matrix(X::Matrix{Float64})
    K, N = size(X)
    out = Array{Float64,3}(undef, K, 1, N)
    @inbounds for n in 1:N, k in 1:K
        out[k, 1, n] = X[k, n]
    end
    return out
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

function load_train_tensor(path::AbstractString, key::AbstractString)
    raw = h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        Float32.(read(h5[key]))
    end
    N, K = size(raw)
    tensor = permutedims(reshape(raw, N, 1, K), (3, 2, 1))
    return Array{Float32,3}(tensor)
end

function normalize_tensor(tensor::Array{Float32,3}, mode::AbstractString)
    mode_l = lowercase(strip(mode))
    mode_l in ("per_channel", "per_feature", "split_xy", "l96_grouped") || error("Unsupported normalization mode '$mode'")

    K, C, _ = size(tensor)
    C == 1 || error("Calibration training expects one-channel X tensor, got C=$C")

    if mode_l == "per_feature"
        stats = ScoreUNet1D.compute_stats(tensor; normalize_mode=:per_feature)
        return ScoreUNet1D.apply_stats(tensor, stats), stats
    elseif mode_l == "l96_grouped"
        stats = ScoreUNet1D.compute_stats(tensor; normalize_mode=:l96_grouped)
        return ScoreUNet1D.apply_stats(tensor, stats), stats
    end

    # Match train_unet.jl default behavior.
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

function safe_cuda_reclaim()
    try
        CUDA.reclaim()
    catch
    end
    return nothing
end

function fallback_weight_inverse_cov(A_of_x::Function, X_obs::AbstractMatrix; base_jitter::Real=1e-8, max_tries::Int=6)
    m = length(A_of_x(@view X_obs[:, 1]))
    T = size(X_obs, 2)
    Aseries = Matrix{Float64}(undef, m, T)
    for t in 1:T
        @views Aseries[:, t] .= A_of_x(X_obs[:, t])
    end
    mu = mean(Aseries; dims=2)
    Acenter = Aseries .- mu
    Sigma = Symmetric((Acenter * Acenter') / max(T - 1, 1))

    jitter = float(base_jitter)
    chol = nothing
    for k in 1:max_tries
        try
            chol = cholesky(Symmetric(Matrix(Sigma) + jitter * I), check=true)
            break
        catch
            if k == max_tries
                rethrow()
            end
            jitter *= 10
        end
    end
    chol === nothing && error("Failed to build inverse covariance weight matrix")
    W = chol \ I
    return Symmetric(Matrix(W))
end

function fallback_newton_step(S::AbstractMatrix, W::Union{AbstractMatrix,Symmetric}, Gamma::Symmetric,
    G::AbstractVector, A::AbstractVector; jitter::Real=1e-10, max_tries::Int=6)
    size(S, 1) == length(A) == length(G) || error("Shape mismatch in fallback_newton_step")
    p = size(S, 2)
    size(Gamma, 1) == p == size(Gamma, 2) || error("Regularization shape mismatch")

    M = S' * (W * S)
    M .+= Matrix(Gamma)
    rhs = S' * (W * (G .- A))

    jitter_local = float(jitter)
    theta = nothing
    Mreg = copy(M)
    for k in 1:max_tries
        Mreg .= M
        @inbounds for i in 1:p
            Mreg[i, i] += jitter_local
        end
        try
            F = cholesky(Symmetric(Mreg), check=true)
            theta = F \ rhs
            break
        catch
            if k == max_tries
                theta = Mreg \ rhs
                break
            end
            jitter_local *= 10
        end
    end
    theta === nothing && error("Failed to solve Newton system")

    cval = try
        cond(Mreg)
    catch
        NaN
    end
    return theta, (cond=cval, nrm_rhs=norm(rhs))
end

function weight_inverse_cov_bridge(A_of_x::Function, X_obs::AbstractMatrix)
    if HAS_PARAMETER_CALIBRATION
        return weight_inverse_cov(A_of_x, X_obs)
    end
    return fallback_weight_inverse_cov(A_of_x, X_obs)
end

function newton_step_bridge(S::AbstractMatrix, W::Union{AbstractMatrix,Symmetric}, Gamma::Symmetric,
    G::AbstractVector, A::AbstractVector)
    if HAS_PARAMETER_CALIBRATION
        return newton_step(S, W, Gamma, G, A)
    end
    return fallback_newton_step(S, W, Gamma, G, A)
end

function load_calibration_config(path::String)
    isfile(path) || error("Calibration parameter file not found: $path")
    doc = TOML.parsefile(path)

    paths = require_table(doc, "paths")
    run = maybe_table(doc, "run")
    truth = require_table(doc, "truth")
    initial_theta = require_table(doc, "initial_theta")
    calibration = require_table(doc, "calibration")
    methods = require_table(doc, "methods")
    datasets = require_table(doc, "datasets")
    training = require_table(doc, "training")
    responses = require_table(doc, "responses")
    responses_fd = maybe_subtable(responses, "finite_difference")
    observables = maybe_table(doc, "observables")
    acceptance_ensemble = maybe_table(doc, "acceptance_ensemble")
    observables_ensemble = maybe_table(doc, "observables_ensemble")
    figures = maybe_table(doc, "figures")
    performance = maybe_table(doc, "performance")
    numerical = maybe_table(doc, "numerical")
    twoscale = maybe_table(doc, "twoscale")
    closure = maybe_table(doc, "closure")

    data_params_path = abspath(as_str(paths, "data_params", "scripts/arnold/parameters_data.toml"))
    train_params_path = abspath(as_str(paths, "train_params", "scripts/arnold/parameters_train.toml"))
    responses_params_path = abspath(as_str(paths, "responses_params", "scripts/arnold/parameters_responses.toml"))

    data_cfg, _ = ArnoldCommon.load_data_config(data_params_path)
    train_doc = TOML.parsefile(train_params_path)
    responses_doc = TOML.parsefile(responses_params_path)

    twoscale_K = as_int(twoscale, "K", data_cfg["twoscale.K"])
    twoscale_J = as_int(twoscale, "J", data_cfg["twoscale.J"])
    twoscale_F = as_float(twoscale, "F", data_cfg["twoscale.F"])
    twoscale_h = as_float(twoscale, "h", data_cfg["twoscale.h"])
    twoscale_c = as_float(twoscale, "c", data_cfg["twoscale.c"])
    twoscale_b = as_float(twoscale, "b", data_cfg["twoscale.b"])
    twoscale_dt = as_float(twoscale, "dt", data_cfg["twoscale.dt"])
    twoscale_process_noise_sigma_y = as_float(twoscale, "process_noise_sigma_y", as_float(twoscale, "process_noise_sigma", data_cfg["twoscale.process_noise_sigma_y"]))
    twoscale_process_noise_sigma_x = as_float(twoscale, "process_noise_sigma_x", as_bool(twoscale, "stochastic_x_noise", data_cfg["twoscale.stochastic_x_noise"]) ? as_float(twoscale, "process_noise_sigma", data_cfg["twoscale.process_noise_sigma"]) : data_cfg["twoscale.process_noise_sigma_x"])

    closure_F = as_float(closure, "F", data_cfg["closure.F"])
    closure_alpha0_initial = as_float(closure, "alpha0_initial", as_float(closure, "alpha0", data_cfg["closure.alpha0_initial"]))
    closure_alpha1_initial = as_float(closure, "alpha1_initial", as_float(closure, "alpha1", data_cfg["closure.alpha1_initial"]))
    closure_alpha2_initial = as_float(closure, "alpha2_initial", as_float(closure, "alpha2", data_cfg["closure.alpha2_initial"]))
    closure_alpha3_initial = as_float(closure, "alpha3_initial", as_float(closure, "alpha3", data_cfg["closure.alpha3_initial"]))
    closure_sigma_initial = as_float(closure, "sigma_initial", as_float(closure, "sigma", data_cfg["closure.sigma_initial"]))
    closure_auto_fit = as_bool(closure, "auto_fit", data_cfg["closure.auto_fit"])
    closure_fit_dataset_role = as_str(closure, "fit_dataset_role", data_cfg["closure.fit_dataset_role"])
    closure_fit_start_index = as_int(closure, "fit_start_index", data_cfg["closure.fit_start_index"])
    closure_fit_samples = as_int(closure, "fit_samples", data_cfg["closure.fit_samples"])
    closure_fit_min_samples = as_int(closure, "fit_min_samples", data_cfg["closure.fit_min_samples"])

    closure_fit_meta = Dict{String,Any}()
    if closure_auto_fit
        fit_data_cfg = copy(data_cfg)
        fit_data_cfg["twoscale.K"] = twoscale_K
        fit_data_cfg["twoscale.J"] = twoscale_J
        fit_data_cfg["twoscale.F"] = twoscale_F
        fit_data_cfg["twoscale.h"] = twoscale_h
        fit_data_cfg["twoscale.c"] = twoscale_c
        fit_data_cfg["twoscale.b"] = twoscale_b
        fit_data_cfg["twoscale.dt"] = twoscale_dt
        fit_data_cfg["twoscale.process_noise_sigma"] = twoscale_process_noise_sigma_y
        fit_data_cfg["twoscale.process_noise_sigma_y"] = twoscale_process_noise_sigma_y
        fit_data_cfg["twoscale.process_noise_sigma_x"] = twoscale_process_noise_sigma_x
        fit_data_cfg["twoscale.stochastic_x_noise"] = twoscale_process_noise_sigma_x > 0.0
        fit_data_cfg["closure.F"] = closure_F
        fit_data_cfg["closure.alpha0_initial"] = closure_alpha0_initial
        fit_data_cfg["closure.alpha1_initial"] = closure_alpha1_initial
        fit_data_cfg["closure.alpha2_initial"] = closure_alpha2_initial
        fit_data_cfg["closure.alpha3_initial"] = closure_alpha3_initial
        fit_data_cfg["closure.sigma_initial"] = closure_sigma_initial
        fit_data_cfg["closure.auto_fit"] = true
        fit_data_cfg["closure.fit_dataset_role"] = closure_fit_dataset_role
        fit_data_cfg["closure.fit_start_index"] = closure_fit_start_index
        fit_data_cfg["closure.fit_samples"] = closure_fit_samples
        fit_data_cfg["closure.fit_min_samples"] = closure_fit_min_samples

        theta_fit, closure_fit_meta = ArnoldCommon.resolve_closure_theta(fit_data_cfg)
        closure_alpha0_initial, closure_alpha1_initial, closure_alpha2_initial, closure_alpha3_initial, closure_sigma_initial = theta_fit
    end

    train_tbl = maybe_table(train_doc, "train")
    responses_gfdt_tbl = maybe_table(responses_doc, "gfdt")
    responses_observables_tbl = maybe_table(responses_doc, "observables")
    responses_numerical_tbl = maybe_table(responses_doc, "numerical")
    responses_tail_window = as_float(responses, "tail_window", as_float(responses_gfdt_tbl, "tail_window", 1.0))
    responses_tmax = as_float(responses, "response_tmax", as_float(responses_gfdt_tbl, "response_tmax", 2.0))
    responses_t_end = as_float(responses, "t_end", as_float(responses_gfdt_tbl, "t_end", responses_tmax))
    responses_t_start = as_float(responses, "t_start", as_float(responses_gfdt_tbl, "t_start", max(0.0, responses_t_end - responses_tail_window)))
    obs_m = as_int(observables, "m", as_int(responses_observables_tbl, "m", as_int(responses_gfdt_tbl, "m", 3)))
    obs_m >= 0 || error("observables.m must be >= 0")
    obs_m <= twoscale_K - 1 || error("observables.m must be <= K-1")

    cfg = Dict{String,Any}(
        "paths.params_file" => abspath(path),
        "paths.data_params" => data_params_path,
        "paths.train_params" => train_params_path,
        "paths.responses_params" => responses_params_path,
        "paths.runs_root" => abspath(as_str(paths, "runs_root", "scripts/arnold/runs_calibration")),
        "run.run_id_padding" => as_int(run, "run_id_padding", 3),
        "run.seed_stride" => as_int(run, "seed_stride", 1_000_000_000),
        "run.seed_offset" => as_int(run, "seed_offset", 0),

        "integration.K" => twoscale_K,
        "integration.J" => twoscale_J,
        "integration.F" => twoscale_F,
        "integration.h" => twoscale_h,
        "integration.c" => twoscale_c,
        "integration.b" => twoscale_b,
        "integration.dt" => twoscale_dt,
        "integration.process_noise_sigma" => twoscale_process_noise_sigma_y,
        "integration.process_noise_sigma_y" => twoscale_process_noise_sigma_y,
        "integration.process_noise_sigma_x" => twoscale_process_noise_sigma_x,
        "integration.stochastic_x_noise" => twoscale_process_noise_sigma_x > 0.0,

        "closure.F" => closure_F,
        "closure.alpha0_initial" => closure_alpha0_initial,
        "closure.alpha1_initial" => closure_alpha1_initial,
        "closure.alpha2_initial" => closure_alpha2_initial,
        "closure.alpha3_initial" => closure_alpha3_initial,
        "closure.sigma_initial" => closure_sigma_initial,
        "closure.auto_fit" => closure_auto_fit,
        "closure.fit_dataset_role" => closure_fit_dataset_role,
        "closure.fit_start_index" => closure_fit_start_index,
        "closure.fit_samples" => closure_fit_samples,
        "closure.fit_min_samples" => closure_fit_min_samples,
        "closure.fit_meta" => closure_fit_meta,

        "observables.F_ref" => closure_F,
        "observables.alpha0_ref" => closure_alpha0_initial,
        "observables.alpha1_ref" => closure_alpha1_initial,
        "observables.alpha2_ref" => closure_alpha2_initial,
        "observables.alpha3_ref" => closure_alpha3_initial,
        "observables.m" => obs_m,
        "observables.n" => observable_count(obs_m),

        "truth.source" => lowercase(as_str(truth, "source", "two_scale")),
        "truth.nsamples" => as_int(truth, "nsamples", 200_000),
        "truth.spinup_steps" => as_int(truth, "spinup_steps", 2_000),
        "truth.rng_seed" => as_int(truth, "rng_seed", 101),
        "truth.save_every" => as_int(truth, "save_every", 200),
        "truth.truth_file" => as_str(truth, "truth_file", ""),
        "truth.truth_key" => as_str(truth, "truth_key", ""),

        "initial_theta.alpha0" => closure_auto_fit ? closure_alpha0_initial : as_float(initial_theta, "alpha0", closure_alpha0_initial),
        "initial_theta.alpha1" => closure_auto_fit ? closure_alpha1_initial : as_float(initial_theta, "alpha1", closure_alpha1_initial),
        "initial_theta.alpha2" => closure_auto_fit ? closure_alpha2_initial : as_float(initial_theta, "alpha2", closure_alpha2_initial),
        "initial_theta.alpha3" => closure_auto_fit ? closure_alpha3_initial : as_float(initial_theta, "alpha3", closure_alpha3_initial),
        "initial_theta.sigma" => closure_auto_fit ? closure_sigma_initial : as_float(initial_theta, "sigma", closure_sigma_initial),
        "initial_theta.perturbation_fraction" => as_float(initial_theta, "perturbation_fraction", 0.0),

        "calibration.max_iterations" => as_int(calibration, "max_iterations", 10),
        "calibration.tol_theta" => as_float(calibration, "tol_theta", 1e-5),
        "calibration.tol_obs" => as_float(calibration, "tol_obs", 1e-4),
        "calibration.damping" => as_float(calibration, "damping", 1.0),
        "calibration.line_search" => as_bool(calibration, "line_search", true),
        "calibration.line_search_max" => as_int(calibration, "line_search_max", 5),
        "calibration.regularization_gamma" => as_float(calibration, "regularization_gamma", 1e-4),
        "calibration.lm_max_attempts" => as_int(calibration, "lm_max_attempts", 4),
        "calibration.lm_growth_factor" => as_float(calibration, "lm_growth_factor", 10.0),
        "calibration.min_actual_to_predicted_ratio" => as_float(calibration, "min_actual_to_predicted_ratio", 0.1),
        "calibration.max_relative_step" => as_float(calibration, "max_relative_step", 0.0),
        "calibration.conditioning_step_threshold" => as_float(calibration, "conditioning_step_threshold", 1e4),
        "calibration.conditioned_max_relative_step" => as_float(calibration, "conditioned_max_relative_step", 0.05),
        "calibration.postcheck_rel_tol" => as_float(calibration, "postcheck_rel_tol", 0.05),
        "calibration.postcheck_abs_tol" => as_float(calibration, "postcheck_abs_tol", 1e-3),
        "calibration.convergence_scope" => lowercase(as_str(calibration, "convergence_scope", "primary")),
        "calibration.weight_matrix" => lowercase(as_str(calibration, "weight_matrix", "identity")),
        "calibration.apply_sigma_jacobian_mask" => as_bool(calibration, "apply_sigma_jacobian_mask", true),
        "calibration.freeze_parameters" => as_int_vec(calibration, "freeze_parameters", Int[]),
        "calibration.freeze_steps" => as_int(calibration, "freeze_steps", 0),

        "methods.unet" => as_bool(methods, "unet", true),
        "methods.gaussian" => as_bool(methods, "gaussian", true),
        "methods.finite_difference" => as_bool(methods, "finite_difference", false),

        "datasets.spinup_steps" => data_cfg["datasets.train_stochastic.spinup_steps"],
        "datasets.train_nsamples" => as_int(datasets, "train_nsamples", data_cfg["datasets.train_stochastic.nsamples"]),
        "datasets.train_save_every" => as_int(datasets, "train_save_every", data_cfg["datasets.train_stochastic.save_every"]),
        "datasets.train_rng_seed_base" => as_int(datasets, "train_rng_seed_base", 2_000),
        "datasets.train_ensemble_trajectories" => as_int(datasets, "train_ensemble_trajectories", 1),
        "datasets.train_parallel_trajectories" => as_bool(datasets, "train_parallel_trajectories", true),
        "datasets.gfdt_nsamples" => as_int(datasets, "gfdt_nsamples", data_cfg["datasets.gfdt_stochastic.nsamples"]),
        "datasets.gfdt_save_every" => as_int(datasets, "gfdt_save_every", data_cfg["datasets.gfdt_stochastic.save_every"]),
        "datasets.gfdt_rng_seed_base" => as_int(datasets, "gfdt_rng_seed_base", 3_000),
        "datasets.train_key" => "x_train_stochastic",
        "datasets.gfdt_key" => "x_gfdt_stochastic",

        "training.device" => as_str(training, "device", as_str(train_tbl, "device", "GPU:0")),
        "training.epochs" => as_int(training, "epochs", as_int(train_tbl, "epochs", 80)),
        "training.resume_from_previous" => as_bool(training, "resume_from_previous", true),

        "training.seed" => as_int(train_tbl, "seed", 42),
        "training.batch_size" => as_int(train_tbl, "batch_size", 512),
        "training.lr" => as_float(train_tbl, "lr", 8e-4),
        "training.sigma" => Float32(as_float(train_tbl, "sigma", 0.05)),
        "training.base_channels" => as_int(train_tbl, "base_channels", 32),
        "training.channel_multipliers" => as_int_vec(train_tbl, "channel_multipliers", [1, 2]),
        "training.norm_type" => lowercase(as_str(train_tbl, "norm_type", "group")),
        "training.norm_groups" => as_int(train_tbl, "norm_groups", 8),
        "training.normalization_mode" => lowercase(as_str(train_tbl, "normalization_mode", "per_channel")),
        "training.use_lr_schedule" => as_bool(train_tbl, "use_lr_schedule", true),
        "training.warmup_steps" => as_int(train_tbl, "warmup_steps", 500),
        "training.min_lr_factor" => as_float(train_tbl, "min_lr_factor", 0.1),
        "training.checkpoint_every" => as_int(train_tbl, "checkpoint_every", 10),
        "training.loss_weight" => Float32(as_float(train_tbl, "loss_weight", 1.0)),
        "training.loss_mean_weight" => Float32(as_float(train_tbl, "loss_mean_weight", 0.0)),
        "training.loss_cov_weight" => Float32(as_float(train_tbl, "loss_cov_weight", 0.0)),

        "responses.response_tmax" => responses_tmax,
        "responses.mean_center" => as_bool(responses, "mean_center", true),
        "responses.impulse_tail_debias" => as_bool(responses, "impulse_tail_debias", as_bool(responses_gfdt_tbl, "impulse_tail_debias", false)),
        "responses.impulse_tail_taper" => lowercase(as_str(responses, "impulse_tail_taper", as_str(responses_gfdt_tbl, "impulse_tail_taper", "linear"))),
        "responses.t_start" => responses_t_start,
        "responses.t_end" => responses_t_end,
        "responses.apply_score_correction" => as_bool(responses, "apply_score_correction", true),
        "responses.divergence_mode" => lowercase(as_str(responses, "divergence_mode", as_str(responses_gfdt_tbl, "divergence_mode", "hutchinson"))),
        "responses.divergence_eps" => as_float(responses, "divergence_eps", as_float(responses_gfdt_tbl, "divergence_eps", 0.03)),
        "responses.divergence_probes" => as_int(responses, "divergence_probes", as_int(responses_gfdt_tbl, "divergence_probes", 10)),
        "responses.score_device" => as_str(responses, "score_device", as_str(responses_gfdt_tbl, "score_device", "CPU")),
        "responses.score_forward_mode" => lowercase(as_str(responses_gfdt_tbl, "score_forward_mode", "test")),
        "responses.batch_size" => as_int(responses, "batch_size", as_int(responses_gfdt_tbl, "batch_size", 1024)),

        "responses.finite_difference.nsamples" => as_int(responses_fd, "nsamples", 15_000_000),
        "responses.finite_difference.spinup" => as_int(responses_fd, "spinup", 2_000),
        "responses.finite_difference.h_abs" => as_float_vec(responses_fd, "h_abs", [0.05, 0.02, 0.01, 0.002, 0.05]),
        "responses.finite_difference.h_rel" => as_float(responses_fd, "h_rel", 0.005),
        "responses.finite_difference.seed_base" => as_int(responses_fd, "seed_base", 50_000_000),
        "responses.finite_difference.use_ensemble" => as_bool(responses_fd, "use_ensemble", true),
        "responses.finite_difference.ensemble_trajectories" => as_int(responses_fd, "ensemble_trajectories", 12),
        "responses.finite_difference.samples_per_trajectory" => as_int(responses_fd, "samples_per_trajectory", 120_000),
        "responses.finite_difference.save_every" => as_int(responses_fd, "save_every", as_int(datasets, "gfdt_save_every", data_cfg["datasets.gfdt_stochastic.save_every"])),
        "responses.finite_difference.parallel_trajectories" => as_bool(responses_fd, "parallel_trajectories", true),

        "acceptance.trajectories" => as_int(acceptance_ensemble, "trajectories", 4),
        "acceptance.samples_per_trajectory" => as_int(acceptance_ensemble, "samples_per_trajectory", 4_000),
        "acceptance.save_every" => as_int(acceptance_ensemble, "save_every", as_int(datasets, "gfdt_save_every", data_cfg["datasets.gfdt_stochastic.save_every"])),
        "acceptance.spinup_steps" => as_int(acceptance_ensemble, "spinup_steps", max(500, data_cfg["datasets.train_stochastic.spinup_steps"])),
        "acceptance.seed_base" => as_int(acceptance_ensemble, "seed_base", 80_000_000),
        "acceptance.parallel_trajectories" => as_bool(acceptance_ensemble, "parallel_trajectories", false),
        "acceptance.refine_residual_threshold" => as_float(acceptance_ensemble, "refine_residual_threshold", 1.5),
        "acceptance.refine_conditioning_threshold" => as_float(acceptance_ensemble, "refine_conditioning_threshold", as_float(calibration, "conditioning_step_threshold", 1e4)),
        "acceptance.refined_trajectories" => as_int(acceptance_ensemble, "refined_trajectories", 12),
        "acceptance.refined_samples_per_trajectory" => as_int(acceptance_ensemble, "refined_samples_per_trajectory", 10_000),
        "acceptance.refined_save_every" => as_int(acceptance_ensemble, "refined_save_every", as_int(acceptance_ensemble, "save_every", as_int(datasets, "gfdt_save_every", data_cfg["datasets.gfdt_stochastic.save_every"]))),
        "acceptance.refined_spinup_steps" => as_int(acceptance_ensemble, "refined_spinup_steps", as_int(acceptance_ensemble, "spinup_steps", max(500, data_cfg["datasets.train_stochastic.spinup_steps"]))),
        "acceptance.refined_parallel_trajectories" => as_bool(acceptance_ensemble, "refined_parallel_trajectories", as_bool(acceptance_ensemble, "parallel_trajectories", false)),

        "observables_ensemble.trajectories" => as_int(observables_ensemble, "trajectories", 24),
        "observables_ensemble.samples_per_trajectory" => as_int(observables_ensemble, "samples_per_trajectory", 4_000),
        "observables_ensemble.save_every" => as_int(observables_ensemble, "save_every", as_int(datasets, "gfdt_save_every", data_cfg["datasets.gfdt_stochastic.save_every"])),
        "observables_ensemble.spinup_steps" => as_int(observables_ensemble, "spinup_steps", data_cfg["datasets.train_stochastic.spinup_steps"]),
        "observables_ensemble.seed_base" => as_int(observables_ensemble, "seed_base", 60_000_000),
        "observables_ensemble.parallel_trajectories" => as_bool(observables_ensemble, "parallel_trajectories", true),

        "numerical.max_abs_state" => max(as_float(responses_numerical_tbl, "max_abs_state", 80.0), 1e4),
        "numerical.state_min" => as_float(numerical, "state_min", -200.0),
        "numerical.state_max" => as_float(numerical, "state_max", 200.0),
        "numerical.max_boundary_hits" => as_int(numerical, "max_boundary_hits", 40),
        "numerical.min_valid_fraction" => as_float(responses_numerical_tbl, "min_valid_fraction", 0.8),
        "numerical.max_h_shrinks" => as_int(responses_numerical_tbl, "max_h_shrinks", 6),
        "numerical.thread_chunk_size" => as_int(responses_numerical_tbl, "thread_chunk_size", 256),
        "numerical.h_shrink_factor" => as_float(responses_numerical_tbl, "h_shrink_factor", 0.5),

        "figures.dpi" => as_int(figures, "dpi", 180),
        "figures.save_per_iteration" => as_bool(figures, "save_per_iteration", true),
        "figures.save_convergence" => as_bool(figures, "save_convergence", true),
        "figures.save_langevin_stats" => as_bool(figures, "save_langevin_stats", true),
        "figures.langevin_device" => as_str(figures, "langevin_device", as_str(responses, "score_device", as_str(responses_gfdt_tbl, "score_device", "CPU"))),
        "figures.langevin_dt" => as_float(figures, "langevin_dt", twoscale_dt),
        "figures.langevin_resolution" => as_int(figures, "langevin_resolution", as_int(datasets, "train_save_every", data_cfg["datasets.train_stochastic.save_every"])),
        "figures.langevin_nsteps" => as_int(figures, "langevin_nsteps", 12_000),
        "figures.langevin_burn_in" => as_int(figures, "langevin_burn_in", 2_000),
        "figures.langevin_ensembles" => as_int(figures, "langevin_ensembles", 128),
        "figures.langevin_seed_base" => as_int(figures, "langevin_seed_base", 70_000_000),
        "figures.langevin_use_boundary" => as_bool(figures, "langevin_use_boundary", true),
        "figures.langevin_boundary_min" => as_float(figures, "langevin_boundary_min", as_float(numerical, "state_min", -200.0)),
        "figures.langevin_boundary_max" => as_float(figures, "langevin_boundary_max", as_float(numerical, "state_max", 200.0)),
        "figures.langevin_max_samples" => as_int(figures, "langevin_max_samples", 10_000),
        "figures.langevin_pdf_bins" => as_int(figures, "langevin_pdf_bins", 80),
        "figures.langevin_max_acf_lag" => as_int(figures, "langevin_max_acf_lag", 200),

        "performance.parallel_methods" => as_bool(performance, "parallel_methods", true),
        "performance.max_parallel_methods" => as_int(performance, "max_parallel_methods", 2),
        "performance.parallel_fd_columns" => as_bool(performance, "parallel_fd_columns", true),
        "performance.parallel_line_search" => as_bool(performance, "parallel_line_search", true),
        "performance.max_parallel_line_search" => as_int(performance, "max_parallel_line_search", 3),
        "performance.allow_nested_parallel" => as_bool(performance, "allow_nested_parallel", true),
    )

    cfg["calibration.free_parameters"] = normalize_indices(as_int_vec(calibration, "free_parameters", Int[]), 5, "calibration.free_parameters")
    cfg["calibration.active_observables"] = normalize_indices(as_int_vec(calibration, "active_observables", Int[]), cfg["observables.n"], "calibration.active_observables")
    freeze_params_raw = collect(get(calibration, "freeze_parameters", Int[]))
    cfg["calibration.freeze_parameters"] = isempty(freeze_params_raw) ? Int[] : normalize_indices(
        Int.(freeze_params_raw),
        5,
        "calibration.freeze_parameters",
    )

    cfg["integration.save_every"] = cfg["datasets.gfdt_save_every"]
    cfg["closure.alpha0"] = cfg["initial_theta.alpha0"]
    cfg["closure.alpha1"] = cfg["initial_theta.alpha1"]
    cfg["closure.alpha2"] = cfg["initial_theta.alpha2"]
    cfg["closure.alpha3"] = cfg["initial_theta.alpha3"]
    cfg["closure.sigma"] = cfg["initial_theta.sigma"]

    cfg["numerical.h_abs"] = copy(cfg["responses.finite_difference.h_abs"])
    cfg["numerical.h_rel"] = cfg["responses.finite_difference.h_rel"]
    cfg["numerical.seed_base"] = cfg["responses.finite_difference.seed_base"]

    cfg["calibration.line_search_samples"] = min(100_000, cfg["datasets.gfdt_nsamples"])
    cfg["calibration.line_search_spinup"] = max(500, cfg["datasets.spinup_steps"])

    cfg["truth.source"] in ("two_scale", "file") || error("truth.source must be 'two_scale' or 'file'")
    if cfg["truth.source"] == "file"
        isempty(strip(cfg["truth.truth_file"])) && error("truth.truth_file is required when truth.source='file'")
        cfg["truth.truth_file"] = abspath(cfg["truth.truth_file"])
    end

    cfg["calibration.max_iterations"] >= 1 || error("calibration.max_iterations must be >= 1")
    cfg["calibration.tol_theta"] > 0 || error("calibration.tol_theta must be > 0")
    cfg["calibration.tol_obs"] > 0 || error("calibration.tol_obs must be > 0")
    cfg["calibration.damping"] > 0 || error("calibration.damping must be > 0")
    cfg["calibration.line_search_max"] >= 0 || error("calibration.line_search_max must be >= 0")
    cfg["calibration.regularization_gamma"] >= 0 || error("calibration.regularization_gamma must be >= 0")
    cfg["calibration.lm_max_attempts"] >= 1 || error("calibration.lm_max_attempts must be >= 1")
    cfg["calibration.lm_growth_factor"] > 1 || error("calibration.lm_growth_factor must be > 1")
    cfg["calibration.min_actual_to_predicted_ratio"] >= 0 || error("calibration.min_actual_to_predicted_ratio must be >= 0")
    cfg["calibration.conditioning_step_threshold"] > 0 || error("calibration.conditioning_step_threshold must be > 0")
    cfg["calibration.postcheck_rel_tol"] >= 0 || error("calibration.postcheck_rel_tol must be >= 0")
    cfg["calibration.postcheck_abs_tol"] >= 0 || error("calibration.postcheck_abs_tol must be >= 0")
    cfg["calibration.convergence_scope"] in ("primary", "all") || error("calibration.convergence_scope must be 'primary' or 'all'")
    cfg["calibration.weight_matrix"] in ("identity", "inverse_cov", "diagonal") || error("calibration.weight_matrix must be identity, inverse_cov, or diagonal")
    cfg["calibration.freeze_steps"] >= 0 || error("calibration.freeze_steps must be >= 0")

    freeze_params = cfg["calibration.freeze_parameters"]
    free_params = cfg["calibration.free_parameters"]
    active_obs = cfg["calibration.active_observables"]
    all(p -> p in free_params, freeze_params) || error("calibration.freeze_parameters must be a subset of calibration.free_parameters")
    if cfg["calibration.freeze_steps"] > 0
        length(freeze_params) < length(active_obs) || error("calibration.freeze_parameters freezes too many parameters for the selected active observables; at least one active observable must remain after dropping one per frozen parameter")
    end

    cfg["integration.K"] >= 2 || error("twoscale.K must be >= 2")
    cfg["integration.J"] >= 1 || error("twoscale.J must be >= 1")
    cfg["integration.dt"] > 0 || error("twoscale.dt must be > 0")

    !isempty(enabled_methods(cfg)) || error("At least one Jacobian method must be enabled")

    cfg["datasets.train_nsamples"] >= 2 || error("datasets.train_nsamples must be >= 2")
    cfg["datasets.train_ensemble_trajectories"] >= 1 || error("datasets.train_ensemble_trajectories must be >= 1")
    cfg["datasets.gfdt_nsamples"] >= 2 || error("datasets.gfdt_nsamples must be >= 2")
    cfg["datasets.train_save_every"] >= 1 || error("datasets.train_save_every must be >= 1")
    cfg["datasets.gfdt_save_every"] >= 1 || error("datasets.gfdt_save_every must be >= 1")

    cfg["training.epochs"] >= 1 || error("training.epochs must be >= 1")
    cfg["training.batch_size"] >= 1 || error("training.batch_size must be >= 1")
    cfg["training.norm_type"] in ("batch", "group") || error("training.norm_type must be batch or group")
    cfg["training.checkpoint_every"] >= 0 || error("training.checkpoint_every must be >= 0")

    cfg["responses.response_tmax"] > 0 || error("responses.response_tmax must be > 0")
    cfg["responses.divergence_mode"] in ("hutchinson", "fd_axis", "exact") || error("responses.divergence_mode must be one of hutchinson|fd_axis|exact")
    cfg["responses.impulse_tail_taper"] in ("none", "hard", "linear") || error("responses.impulse_tail_taper must be one of none|hard|linear")
    cfg["responses.t_start"] >= 0 || error("responses.t_start must be >= 0")
    cfg["responses.t_end"] > cfg["responses.t_start"] || error("responses.t_end must be > responses.t_start")
    cfg["responses.t_end"] <= cfg["responses.response_tmax"] + 1e-12 || error("responses.t_end must be <= responses.response_tmax")
    cfg["responses.batch_size"] >= 1 || error("responses.batch_size must be >= 1")

    length(cfg["responses.finite_difference.h_abs"]) == 5 || error("responses.finite_difference.h_abs must have length 5")
    cfg["responses.finite_difference.h_rel"] > 0 || error("responses.finite_difference.h_rel must be > 0")
    cfg["responses.finite_difference.ensemble_trajectories"] >= 1 || error("responses.finite_difference.ensemble_trajectories must be >= 1")
    cfg["responses.finite_difference.samples_per_trajectory"] >= 1 || error("responses.finite_difference.samples_per_trajectory must be >= 1")
    cfg["responses.finite_difference.save_every"] >= 1 || error("responses.finite_difference.save_every must be >= 1")

    cfg["acceptance.trajectories"] >= 1 || error("acceptance_ensemble.trajectories must be >= 1")
    cfg["acceptance.samples_per_trajectory"] >= 1 || error("acceptance_ensemble.samples_per_trajectory must be >= 1")
    cfg["acceptance.save_every"] >= 1 || error("acceptance_ensemble.save_every must be >= 1")
    cfg["acceptance.spinup_steps"] >= 0 || error("acceptance_ensemble.spinup_steps must be >= 0")
    cfg["acceptance.refine_residual_threshold"] >= 0 || error("acceptance_ensemble.refine_residual_threshold must be >= 0")
    cfg["acceptance.refine_conditioning_threshold"] >= 0 || error("acceptance_ensemble.refine_conditioning_threshold must be >= 0")
    cfg["acceptance.refined_trajectories"] >= 1 || error("acceptance_ensemble.refined_trajectories must be >= 1")
    cfg["acceptance.refined_samples_per_trajectory"] >= 1 || error("acceptance_ensemble.refined_samples_per_trajectory must be >= 1")
    cfg["acceptance.refined_save_every"] >= 1 || error("acceptance_ensemble.refined_save_every must be >= 1")
    cfg["acceptance.refined_spinup_steps"] >= 0 || error("acceptance_ensemble.refined_spinup_steps must be >= 0")

    cfg["observables_ensemble.trajectories"] >= 1 || error("observables_ensemble.trajectories must be >= 1")
    cfg["observables_ensemble.samples_per_trajectory"] >= 1 || error("observables_ensemble.samples_per_trajectory must be >= 1")
    cfg["observables_ensemble.save_every"] >= 1 || error("observables_ensemble.save_every must be >= 1")
    cfg["observables_ensemble.spinup_steps"] >= 0 || error("observables_ensemble.spinup_steps must be >= 0")

    cfg["numerical.max_abs_state"] > 0 || error("numerical.max_abs_state must be > 0")
    cfg["numerical.state_min"] <= cfg["numerical.state_max"] || error("numerical.state_min must be <= numerical.state_max")
    cfg["numerical.max_boundary_hits"] >= 1 || error("numerical.max_boundary_hits must be >= 1")
    0.0 <= cfg["numerical.min_valid_fraction"] <= 1.0 || error("numerical.min_valid_fraction must be in [0,1]")
    cfg["numerical.max_h_shrinks"] >= 0 || error("numerical.max_h_shrinks must be >= 0")
    cfg["numerical.thread_chunk_size"] >= 1 || error("numerical.thread_chunk_size must be >= 1")
    0.0 < cfg["numerical.h_shrink_factor"] < 1.0 || error("numerical.h_shrink_factor must be in (0,1)")

    cfg["performance.max_parallel_methods"] >= 1 || error("performance.max_parallel_methods must be >= 1")
    cfg["performance.max_parallel_line_search"] >= 1 || error("performance.max_parallel_line_search must be >= 1")
    cfg["run.run_id_padding"] >= 1 || error("run.run_id_padding must be >= 1")
    cfg["run.seed_stride"] >= 0 || error("run.seed_stride must be >= 0")

    cfg["figures.langevin_dt"] > 0 || error("figures.langevin_dt must be > 0")
    cfg["figures.langevin_resolution"] >= 1 || error("figures.langevin_resolution must be >= 1")
    cfg["figures.langevin_nsteps"] > cfg["figures.langevin_burn_in"] || error("figures.langevin_nsteps must be > figures.langevin_burn_in")
    cfg["figures.langevin_ensembles"] >= 1 || error("figures.langevin_ensembles must be >= 1")
    cfg["figures.langevin_boundary_min"] <= cfg["figures.langevin_boundary_max"] || error("figures.langevin_boundary_min must be <= figures.langevin_boundary_max")
    cfg["figures.langevin_max_samples"] >= 2 || error("figures.langevin_max_samples must be >= 2")
    cfg["figures.langevin_pdf_bins"] >= 8 || error("figures.langevin_pdf_bins must be >= 8")
    cfg["figures.langevin_max_acf_lag"] >= 1 || error("figures.langevin_max_acf_lag must be >= 1")

    cfg["runtime.A_target"] = nothing
    cfg["runtime.current_observable_series"] = nothing
    cfg["runtime.current_method"] = ""
    cfg["runtime.primary_observables"] = nothing
    cfg["runtime.iteration_diagnostics"] = Dict{String,Any}()
    cfg["runtime.truth_matrix"] = nothing
    cfg["runtime.current_gfdt_path"] = ""
    cfg["runtime.current_train_path"] = ""
    cfg["runtime.current_checkpoint_path"] = ""
    cfg["runtime.current_langevin_method"] = ""
    cfg["runtime.iteration"] = 0
    cfg["runtime.method_parallel_active"] = false
    cfg["runtime.previous_observable_residual"] = Inf
    cfg["runtime.previous_observable_residual_unweighted"] = Inf
    cfg["runtime.run_id"] = 0
    cfg["runtime.run_name"] = ""
    cfg["runtime.run_seed_offset"] = 0
    cfg["runtime.run_seed_offset_applied"] = false

    return cfg
end

function normalize_with_stats(tensor::Array{Float32,3}, stats::DataStats)
    K, C, _ = size(tensor)
    mean_lc = permutedims(stats.mean, (2, 1))
    std_lc = permutedims(stats.std, (2, 1))
    out = (tensor .- reshape(mean_lc, K, C, 1)) ./ reshape(std_lc, K, C, 1)
    return Array{Float32,3}(out)
end

function denormalize_with_stats(tensor::Array{Float32,3}, stats::DataStats)
    K, C, _ = size(tensor)
    mean_lc = permutedims(stats.mean, (2, 1))
    std_lc = permutedims(stats.std, (2, 1))
    out = tensor .* reshape(std_lc, K, C, 1) .+ reshape(mean_lc, K, C, 1)
    return Array{Float32,3}(out)
end

function resolve_langevin_device(name::AbstractString)
    try
        device = select_device(name)
        activate_device!(device)
        return device, String(name)
    catch err
        @warn "Requested Langevin diagnostic device unavailable; using CPU" requested = name error = sprint(showerror, err)
        device = ScoreUNet1D.CPUDevice()
        activate_device!(device)
        return device, "CPU"
    end
end

function langevin_result_to_tensor(result, K::Int, C::Int)
    traj = result.trajectory
    ndims(traj) in (2, 3, 4) || error("Unexpected Langevin trajectory rank: $(ndims(traj))")
    size(traj, 1) == K || error("Langevin trajectory K mismatch")

    if ndims(traj) == 2
        size(traj, 2) >= 1 || error("Langevin trajectory has no time samples")
        return reshape(traj, K, 1, size(traj, 2))
    elseif ndims(traj) == 3
        # Match run_langevin.jl convention: trajectory is typically K x T x E.
        traj4 = reshape(traj, K, C, :, size(traj, 3))
        return reshape(permutedims(traj4, (1, 2, 4, 3)), K, C, :)
    end

    # K x C x T x E
    size(traj, 2) == C || error("Langevin trajectory channel mismatch")
    traj_perm = permutedims(traj, (1, 2, 4, 3))
    return reshape(traj_perm, K, C, :)
end

function save_score_langevin_iteration_figure(path::AbstractString,
    cfg::Dict{String,Any},
    train_data_path::AbstractString,
    checkpoint_path::AbstractString,
    iteration::Int;
    archive_path::Union{Nothing,AbstractString}=nothing,
    summary_path::Union{Nothing,AbstractString}=nothing)
    isdefined(Main, :ArnoldStatsPlots) || return ""
    sp = getproperty(Main, :ArnoldStatsPlots)

    train_path_s = String(strip(String(train_data_path)))
    ckpt_path_s = String(strip(String(checkpoint_path)))
    isempty(train_path_s) && return ""
    isempty(ckpt_path_s) && return ""
    isfile(train_path_s) || error("Training dataset not found for Langevin diagnostic: $train_path_s")
    isfile(ckpt_path_s) || error("Checkpoint not found for Langevin diagnostic: $ckpt_path_s")

    payload = BSON.load(ckpt_path_s)
    haskey(payload, :model) || error("Checkpoint missing :model at $ckpt_path_s")
    haskey(payload, :trainer_cfg) || error("Checkpoint missing :trainer_cfg at $ckpt_path_s")
    haskey(payload, :stats) || error("Checkpoint missing :stats at $ckpt_path_s")

    model = payload[:model]
    trainer_cfg = payload[:trainer_cfg]
    stats = payload[:stats]

    tensor_obs_raw = load_train_tensor(train_path_s, cfg["datasets.train_key"])
    tensor_obs_norm = normalize_with_stats(tensor_obs_raw, stats)
    dataset = NormalizedDataset(tensor_obs_norm, stats)

    device, device_name = resolve_langevin_device(cfg["figures.langevin_device"])
    model = move_model(model, is_gpu(device) ? device : ScoreUNet1D.CPUDevice())

    sigma_train = Float32(getproperty(trainer_cfg, :sigma))
    lg_cfg = LangevinConfig(
        dt=cfg["figures.langevin_dt"],
        sample_dt=cfg["figures.langevin_dt"] * cfg["figures.langevin_resolution"],
        nsteps=cfg["figures.langevin_nsteps"],
        burn_in=cfg["figures.langevin_burn_in"],
        resolution=cfg["figures.langevin_resolution"],
        n_ensembles=cfg["figures.langevin_ensembles"],
        nbins=cfg["figures.langevin_pdf_bins"],
        sigma=sigma_train,
        seed=cfg["figures.langevin_seed_base"] + iteration,
        mode=:all,
        boundary=cfg["figures.langevin_use_boundary"] ? (cfg["figures.langevin_boundary_min"], cfg["figures.langevin_boundary_max"]) : nothing,
        progress=false,
    )

    @info "Running score-Langevin diagnostics for calibration iteration" iteration device = device_name nsteps = cfg["figures.langevin_nsteps"] burn_in = cfg["figures.langevin_burn_in"] ensembles = cfg["figures.langevin_ensembles"] resolution = cfg["figures.langevin_resolution"]
    result = run_langevin(model, dataset, lg_cfg; device=device)

    K, C, _ = size(tensor_obs_raw)
    tensor_gen_norm = langevin_result_to_tensor(result, K, C)
    tensor_gen_raw = denormalize_with_stats(tensor_gen_norm, stats)
    n_use = min(size(tensor_obs_raw, 3), size(tensor_gen_raw, 3), Int(cfg["figures.langevin_max_samples"]))
    n_use >= 2 || error("Langevin diagnostic has too few samples: $n_use")

    obs_eval = tensor_obs_raw[:, :, 1:n_use]
    gen_eval = tensor_gen_raw[:, :, 1:n_use]
    obs_eval_norm = tensor_obs_norm[:, :, 1:n_use]
    gen_eval_norm = tensor_gen_norm[:, :, 1:n_use]
    kl_mode, js_mode = sp.modewise_metrics(
        obs_eval,
        gen_eval;
        nbins=cfg["figures.langevin_pdf_bins"],
        low_q=0.001,
        high_q=0.999,
    )

    fig_path = sp.save_stats_figure_acf(
        path,
        obs_eval,
        gen_eval,
        kl_mode,
        js_mode,
        cfg["figures.langevin_pdf_bins"];
        max_lag=cfg["figures.langevin_max_acf_lag"],
        obs_dt=Float64(cfg["datasets.train_save_every"]) * Float64(cfg["integration.dt"]),
        gen_dt=Float64(lg_cfg.sample_dt),
        obs_label="Train stochastic",
        gen_label="Score Langevin",
    )

    avg_mode_kl = mean(kl_mode)
    avg_mode_js = mean(js_mode)

    if archive_path !== nothing
        archive_path_s = abspath(String(archive_path))
        mkpath(dirname(archive_path_s))
        h5open(archive_path_s, "w") do h5
            h5["observed/raw"] = obs_eval
            h5["generated/raw"] = gen_eval
            h5["observed/normalized"] = obs_eval_norm
            h5["generated/normalized"] = gen_eval_norm
            h5["metrics/kl_mode"] = kl_mode
            h5["metrics/js_mode"] = js_mode
            h5["stats/mean"] = Float32.(stats.mean)
            h5["stats/std"] = Float32.(stats.std)
            attrs = HDF5.attributes(h5)
            attrs["iteration"] = iteration
            attrs["n_use"] = n_use
            attrs["device"] = device_name
            attrs["train_data_path"] = train_path_s
            attrs["checkpoint_path"] = ckpt_path_s
        end
    end

    if summary_path !== nothing
        summary_path_s = abspath(String(summary_path))
        mkpath(dirname(summary_path_s))
        doc = Dict{String,Any}(
            "iteration" => iteration,
            "device" => device_name,
            "n_use" => n_use,
            "avg_mode_kl" => avg_mode_kl,
            "avg_mode_js" => avg_mode_js,
            "train_data_path" => train_path_s,
            "checkpoint_path" => ckpt_path_s,
            "figure_path" => abspath(fig_path),
            "archive_path" => archive_path === nothing ? "" : abspath(String(archive_path)),
        )
        open(summary_path_s, "w") do io
            TOML.print(io, doc)
        end
    end

    reclaim_device_memory!()
    safe_cuda_reclaim()
    return fig_path
end

function build_calibration_truth_data_cfg(cfg::Dict{String,Any})
    data_cfg, _ = ArnoldCommon.load_data_config(cfg["paths.data_params"])
    data_cfg["twoscale.K"] = Int(cfg["integration.K"])
    data_cfg["twoscale.J"] = Int(cfg["integration.J"])
    data_cfg["twoscale.F"] = Float64(cfg["integration.F"])
    data_cfg["twoscale.h"] = Float64(cfg["integration.h"])
    data_cfg["twoscale.c"] = Float64(cfg["integration.c"])
    data_cfg["twoscale.b"] = Float64(cfg["integration.b"])
    data_cfg["twoscale.dt"] = Float64(cfg["integration.dt"])
    data_cfg["twoscale.process_noise_sigma"] = Float64(cfg["integration.process_noise_sigma_y"])
    data_cfg["twoscale.process_noise_sigma_y"] = Float64(cfg["integration.process_noise_sigma_y"])
    data_cfg["twoscale.process_noise_sigma_x"] = Float64(cfg["integration.process_noise_sigma_x"])
    data_cfg["twoscale.stochastic_x_noise"] = Bool(cfg["integration.process_noise_sigma_x"] > 0)

    data_cfg["closure.F"] = Float64(cfg["closure.F"])
    data_cfg["closure.alpha0_initial"] = Float64(cfg["closure.alpha0_initial"])
    data_cfg["closure.alpha1_initial"] = Float64(cfg["closure.alpha1_initial"])
    data_cfg["closure.alpha2_initial"] = Float64(cfg["closure.alpha2_initial"])
    data_cfg["closure.alpha3_initial"] = Float64(cfg["closure.alpha3_initial"])
    data_cfg["closure.sigma_initial"] = Float64(cfg["closure.sigma_initial"])
    data_cfg["closure.auto_fit"] = Bool(cfg["closure.auto_fit"])

    data_cfg["datasets.two_scale_observed.spinup_steps"] = Int(cfg["truth.spinup_steps"])
    data_cfg["datasets.two_scale_observed.save_every"] = Int(cfg["truth.save_every"])
    data_cfg["datasets.two_scale_observed.nsamples"] = Int(cfg["truth.nsamples"])
    data_cfg["datasets.two_scale_observed.rng_seed"] = Int(cfg["truth.rng_seed"])
    data_cfg["datasets.two_scale_observed.target_spacing"] = ArnoldCommon.dataset_time_spacing(
        data_cfg["twoscale.dt"],
        data_cfg["datasets.two_scale_observed.save_every"],
    )

    return data_cfg
end

function maybe_save_truth_artifacts!(cfg::Dict{String,Any}, X::Matrix{Float64}, A_target::Vector{Float64})
    if !haskey(cfg, "runtime.truth_dir")
        return nothing
    end
    truth_dir = String(cfg["runtime.truth_dir"])
    isempty(strip(truth_dir)) && return nothing

    mkpath(truth_dir)
    target_csv = joinpath(truth_dir, "target_observables.csv")
    write_vector_csv(target_csv, observable_names(Int(cfg["observables.m"])), A_target)

    traj_path = joinpath(truth_dir, "truth_trajectory.hdf5")
    attrs = Dict{String,Any}(
        "source" => cfg["truth.source"],
        "nsamples" => size(X, 2),
        "K" => size(X, 1),
        "generated_at" => string(now()),
    )
    ArnoldCommon.save_x_dataset(traj_path, "x_truth", permutedims(Float32.(X), (2, 1)), attrs)
    return nothing
end

function compute_target_observables(cfg::Dict)
    obs_ref = obs_ref_tuple(cfg)
    K = cfg["integration.K"]
    obs_m = Int(cfg["observables.m"])

    X = if cfg["truth.source"] == "two_scale"
        # Reuse cached Arnold dataset when signatures already match; this avoids
        # regenerating the expensive two-scale trajectory on every calibration run.
        truth_data_cfg = build_calibration_truth_data_cfg(cfg)
        ArnoldCommon.ensure_arnold_dataset_role!(truth_data_cfg, "two_scale_observed")
        ArnoldCommon.load_role_x_matrix(
            truth_data_cfg,
            "two_scale_observed";
            nsamples=cfg["truth.nsamples"],
            start_index=1,
            label="truth",
        )
    else
        load_x_matrix(cfg["truth.truth_file"], cfg["truth.truth_key"], K)
    end

    A = ArnoldCommon.compute_observables_series(
        X,
        obs_ref.F_ref,
        obs_ref.alpha0_ref,
        obs_ref.alpha1_ref,
        obs_ref.alpha2_ref,
        obs_ref.alpha3_ref,
        obs_m,
    )
    A_target = vec(mean(A; dims=2))
    all(isfinite, A_target) || error("Target observables contain non-finite values")

    cfg["runtime.truth_matrix"] = X
    cfg["runtime.A_target"] = A_target
    maybe_save_truth_artifacts!(cfg, X, A_target)

    return A_target
end

function generate_iteration_datasets(cfg::Dict, theta::NTuple{5,Float64}, iteration::Int, run_dir::String;
    generate_train::Bool=true)
    iter_data_dir = joinpath(run_dir, @sprintf("iter_%03d", iteration), "data")
    mkpath(iter_data_dir)

    a0, a1, a2, a3, sig = theta

    train_seed = cfg["datasets.train_rng_seed_base"] + iteration
    gfdt_seed = cfg["datasets.gfdt_rng_seed_base"] + iteration

    train_total = Int(cfg["datasets.train_nsamples"])
    train_ntraj = Int(cfg["datasets.train_ensemble_trajectories"])
    train_parallel = Bool(cfg["datasets.train_parallel_trajectories"]) && train_ntraj > 1 && Base.Threads.nthreads() > 1

    gfdt_data = ArnoldCommon.generate_reduced_x_timeseries(
        K=cfg["integration.K"],
        F=cfg["closure.F"],
        alpha0=a0,
        alpha1=a1,
        alpha2=a2,
        alpha3=a3,
        sigma=sig,
        dt=cfg["integration.dt"],
        spinup_steps=cfg["datasets.spinup_steps"],
        save_every=cfg["datasets.gfdt_save_every"],
        nsamples=cfg["datasets.gfdt_nsamples"],
        rng_seed=gfdt_seed,
        max_abs_state=1e4,
        max_restarts=40,
        state_min=cfg["numerical.state_min"],
        state_max=cfg["numerical.state_max"],
        max_boundary_hits=cfg["numerical.max_boundary_hits"],
    )

    train_path = ""
    gfdt_path = joinpath(iter_data_dir, "gfdt_stochastic.hdf5")

    common_attrs = Dict{String,Any}(
        "generated_at" => string(now()),
        "K" => cfg["integration.K"],
        "dt" => cfg["integration.dt"],
        "F" => cfg["closure.F"],
        "alpha0" => a0,
        "alpha1" => a1,
        "alpha2" => a2,
        "alpha3" => a3,
        "sigma" => sig,
    )

    attrs_gfdt = copy(common_attrs)
    attrs_gfdt["save_every"] = cfg["datasets.gfdt_save_every"]
    attrs_gfdt["nsamples"] = cfg["datasets.gfdt_nsamples"]
    attrs_gfdt["spinup_steps"] = cfg["datasets.spinup_steps"]
    attrs_gfdt["rng_seed"] = gfdt_seed
    attrs_gfdt["role"] = "gfdt_stochastic"

    if generate_train
        base = div(train_total, train_ntraj)
        remn = mod(train_total, train_ntraj)
        counts = [base + (i <= remn ? 1 : 0) for i in 1:train_ntraj]
        train_chunks = Vector{Matrix{Float32}}(undef, train_ntraj)
        train_errors = fill("", train_ntraj)

        build_chunk! = function (i::Int)
            ns = counts[i]
            if ns <= 0
                train_chunks[i] = zeros(Float32, 0, cfg["integration.K"])
                train_errors[i] = ""
                return nothing
            end
            try
                chunk_seed = train_seed + 100_000 * i
                train_chunks[i] = ArnoldCommon.generate_reduced_x_timeseries(
                    K=cfg["integration.K"],
                    F=cfg["closure.F"],
                    alpha0=a0,
                    alpha1=a1,
                    alpha2=a2,
                    alpha3=a3,
                    sigma=sig,
                    dt=cfg["integration.dt"],
                    spinup_steps=cfg["datasets.spinup_steps"],
                    save_every=cfg["datasets.train_save_every"],
                    nsamples=ns,
                    rng_seed=chunk_seed,
                    max_abs_state=1e4,
                    max_restarts=40,
                    state_min=cfg["numerical.state_min"],
                    state_max=cfg["numerical.state_max"],
                    max_boundary_hits=cfg["numerical.max_boundary_hits"],
                )
                train_errors[i] = ""
            catch err
                train_errors[i] = sprint(showerror, err)
                train_chunks[i] = zeros(Float32, 0, cfg["integration.K"])
            end
            return nothing
        end

        if train_parallel
            Base.Threads.@threads for i in 1:train_ntraj
                build_chunk!(i)
            end
        else
            for i in 1:train_ntraj
                build_chunk!(i)
            end
        end

        for i in 1:train_ntraj
            isempty(train_errors[i]) || error("Training trajectory chunk $(i) failed: $(train_errors[i])")
        end

        train_data = Array{Float32}(undef, train_total, cfg["integration.K"])
        pos = 1
        for i in 1:train_ntraj
            chunk = train_chunks[i]
            nrows = size(chunk, 1)
            if nrows > 0
                @views train_data[pos:(pos + nrows - 1), :] .= chunk
                pos += nrows
            end
        end
        pos == train_total + 1 || error("Training dataset assembly mismatch: expected $(train_total) rows, got $(pos - 1)")

        train_path = joinpath(iter_data_dir, "train_stochastic.hdf5")
        attrs_train = copy(common_attrs)
        attrs_train["save_every"] = cfg["datasets.train_save_every"]
        attrs_train["nsamples"] = cfg["datasets.train_nsamples"]
        attrs_train["spinup_steps"] = cfg["datasets.spinup_steps"]
        attrs_train["rng_seed"] = train_seed
        attrs_train["ensemble_trajectories"] = train_ntraj
        attrs_train["parallel_trajectories"] = train_parallel
        attrs_train["role"] = "train_stochastic"
        ArnoldCommon.save_x_dataset(train_path, cfg["datasets.train_key"], train_data, attrs_train)
    end

    ArnoldCommon.save_x_dataset(gfdt_path, cfg["datasets.gfdt_key"], gfdt_data, attrs_gfdt)

    return train_path, gfdt_path
end

function train_iteration_score(cfg::Dict, train_data_path::String, iteration::Int, run_dir::String; prev_checkpoint=nothing)
    iter_dir = joinpath(run_dir, @sprintf("iter_%03d", iteration))
    model_dir = joinpath(iter_dir, "model")
    checkpoint_dir = joinpath(model_dir, "checkpoints")
    mkpath(checkpoint_dir)

    reclaim_device_memory!()

    train_tensor = load_train_tensor(train_data_path, cfg["datasets.train_key"])
    train_tensor_norm, stats = normalize_tensor(train_tensor, cfg["training.normalization_mode"])
    dataset = NormalizedDataset(train_tensor_norm, stats)

    model_cfg = ScoreUNetConfig(
        in_channels=1,
        out_channels=1,
        base_channels=cfg["training.base_channels"],
        channel_multipliers=parse_channel_multipliers(cfg["training.channel_multipliers"]),
        kernel_size=5,
        periodic=true,
        norm_type=parse_norm_type(cfg["training.norm_type"]),
        norm_groups=cfg["training.norm_groups"],
    )

    model = nothing
    if cfg["training.resume_from_previous"] && prev_checkpoint !== nothing && isfile(prev_checkpoint)
        payload = BSON.load(prev_checkpoint)
        if haskey(payload, :model)
            model = payload[:model]
        else
            @warn "Previous checkpoint missing :model; starting from fresh initialization" path = prev_checkpoint
        end
    end
    model === nothing && (model = build_unet(model_cfg))

    trainer_cfg = ScoreTrainerConfig(
        batch_size=cfg["training.batch_size"],
        epochs=cfg["training.epochs"],
        lr=cfg["training.lr"],
        sigma=cfg["training.sigma"],
        seed=cfg["training.seed"] + iteration,
        progress=false,
        use_lr_schedule=cfg["training.use_lr_schedule"],
        warmup_steps=cfg["training.warmup_steps"],
        min_lr_factor=cfg["training.min_lr_factor"],
        x_loss_weight=cfg["training.loss_weight"],
        y_loss_weight=cfg["training.loss_weight"],
        mean_match_weight=cfg["training.loss_mean_weight"],
        cov_match_weight=cfg["training.loss_cov_weight"],
    )

    device = try
        select_device(cfg["training.device"])
    catch err
        @warn "Training device unavailable; falling back to CPU" requested = cfg["training.device"] error = sprint(showerror, err)
        ScoreUNet1D.CPUDevice()
    end
    activate_device!(device)
    model = move_model(model, device)

    checkpoint_every = cfg["training.checkpoint_every"]
    epoch_callback = function(epoch::Int, model_epoch, _epoch_time::Real)
        checkpoint_every > 0 || return nothing
        if epoch % checkpoint_every != 0
            return nothing
        end
        ckpt_path = joinpath(checkpoint_dir, @sprintf("epoch_%04d.bson", epoch))
        model_cpu = move_model(model_epoch, ScoreUNet1D.CPUDevice())
        Flux.testmode!(model_cpu)
        BSON.@save ckpt_path model=model_cpu cfg=model_cfg trainer_cfg stats epoch
        return nothing
    end

    @info "Training iteration score model" iteration train_data_path epochs = cfg["training.epochs"] batch_size = cfg["training.batch_size"]
    history = train!(
        model,
        dataset,
        trainer_cfg;
        device=device,
        epoch_callback=epoch_callback,
    )

    model_cpu = is_gpu(device) ? move_model(model, ScoreUNet1D.CPUDevice()) : model
    Flux.testmode!(model_cpu)

    final_path = joinpath(model_dir, "final_checkpoint.bson")
    BSON.@save final_path model=model_cpu cfg=model_cfg trainer_cfg stats epoch=length(history.epoch_losses)

    reclaim_device_memory!()
    safe_cuda_reclaim()

    return final_path
end

function response_n_lags(cfg::Dict, n_samples::Int)
    dt_obs = cfg["integration.dt"] * cfg["datasets.gfdt_save_every"]
    n_lags_req = max(1, Int(floor(cfg["responses.response_tmax"] / dt_obs)))
    n_lags = min(n_lags_req, n_samples - 1)
    n_lags >= 1 || error("Not enough GFDT samples for selected response_tmax")
    return n_lags, dt_obs
end

function compute_fd_jacobian_asymptotic(theta::NTuple{5,Float64}, cfg::Dict{String,Any})
    compute_steady_state_observables = require_main_symbol(:compute_steady_state_observables)
    n_obs = observable_count(cfg)

    h_abs = Float64.(cfg["responses.finite_difference.h_abs"])
    h_rel = Float64(cfg["responses.finite_difference.h_rel"])
    n_samples = Int(cfg["responses.finite_difference.nsamples"])
    spinup = Int(cfg["responses.finite_difference.spinup"])
    seed_base = Int(cfg["responses.finite_difference.seed_base"])
    h_shrink_factor = Float64(cfg["numerical.h_shrink_factor"])
    max_h_shrinks = Int(cfg["numerical.max_h_shrinks"])
    use_fd_ensemble = Bool(cfg["responses.finite_difference.use_ensemble"])
    fd_trajectories = Int(cfg["responses.finite_difference.ensemble_trajectories"])
    fd_samples_per_traj = Int(cfg["responses.finite_difference.samples_per_trajectory"])
    fd_save_every = Int(cfg["responses.finite_difference.save_every"])
    fd_parallel_traj = Bool(cfg["responses.finite_difference.parallel_trajectories"])
    allow_nested_parallel = Bool(get(cfg, "performance.allow_nested_parallel", true))
    parallel_fd_columns = Bool(cfg["performance.parallel_fd_columns"])
    nested_parallel_active = Bool(get(cfg, "runtime.method_parallel_active", false))

    length(h_abs) == 5 || error("responses.finite_difference.h_abs must have length 5")
    n_samples >= 1 || error("responses.finite_difference.nsamples must be >= 1")
    spinup >= 0 || error("responses.finite_difference.spinup must be >= 0")
    h_rel > 0 || error("responses.finite_difference.h_rel must be > 0")

    J = zeros(Float64, n_obs, 5)
    iter_id = Int(get(cfg, "runtime.iteration", 0))
    if use_fd_ensemble
        h_current = [max(h_abs[ip], h_rel * max(abs(theta[ip]), 1.0)) for ip in 1:5]
        attempts = zeros(Int, 5)
        pending = collect(1:5)
        do_parallel = (parallel_fd_columns || fd_parallel_traj) && Base.Threads.nthreads() > 1
        do_parallel = do_parallel && (allow_nested_parallel || !nested_parallel_active)

        while !isempty(pending)
            n_pending = length(pending)
            obs_plus = Array{Float64,3}(undef, n_obs, fd_trajectories, n_pending)
            obs_minus = Array{Float64,3}(undef, n_obs, fd_trajectories, n_pending)
            task_errors = fill("", n_pending, fd_trajectories)

            for pidx in 1:n_pending
                ip = pending[pidx]
                attempts[ip] += 1
                seed = seed_base + 1_000_000 * ip + 100_000 * (attempts[ip] - 1) + 10_000_000 * iter_id
                @info "Computing calibration asymptotic FD Jacobian column" parameter = PARAM_NAMES[ip] h = h_current[ip] n_samples = fd_trajectories * fd_samples_per_traj spinup seed attempt = attempts[ip]
            end

            run_pair! = function (pidx::Int, trj::Int)
                ip = pending[pidx]
                h = h_current[ip]
                tp = collect(theta)
                tm = collect(theta)
                tp[ip] += h
                tm[ip] -= h
                theta_p = (tp[1], tp[2], tp[3], tp[4], tp[5])
                theta_m = (tm[1], tm[2], tm[3], tm[4], tm[5])
                seed = seed_base + 1_000_000 * ip + 100_000 * (attempts[ip] - 1) + 10_000_000 * iter_id + 11 + 10_000 * trj
                local_cfg = copy(cfg)
                local_cfg["integration.save_every"] = fd_save_every
                try
                    obs_p = compute_steady_state_observables(theta_p, local_cfg, fd_samples_per_traj, spinup, seed)
                    obs_m = compute_steady_state_observables(theta_m, local_cfg, fd_samples_per_traj, spinup, seed)
                    @views obs_plus[:, trj, pidx] .= Float64.(obs_p)
                    @views obs_minus[:, trj, pidx] .= Float64.(obs_m)
                catch err
                    task_errors[pidx, trj] = sprint(showerror, err)
                end
                return nothing
            end

            ntasks = n_pending * fd_trajectories
            if do_parallel && ntasks > 1
                Base.Threads.@threads for task_idx in 1:ntasks
                    pidx = 1 + (task_idx - 1) ÷ fd_trajectories
                    trj = 1 + (task_idx - 1) % fd_trajectories
                    run_pair!(pidx, trj)
                end
            else
                for pidx in 1:n_pending
                    for trj in 1:fd_trajectories
                        run_pair!(pidx, trj)
                    end
                end
            end

            next_pending = Int[]
            for pidx in 1:n_pending
                ip = pending[pidx]
                err = findfirst(msg -> !isempty(msg), view(task_errors, pidx, :))
                if err === nothing
                    avg_p = vec(mean(@view obs_plus[:, :, pidx]; dims=2))
                    avg_m = vec(mean(@view obs_minus[:, :, pidx]; dims=2))
                    col = (avg_p .- avg_m) ./ (2 * h_current[ip])
                    all(isfinite, col) || error("FD column has non-finite entries")
                    @views J[:, ip] .= col
                else
                    reason = task_errors[pidx, err]
                    if attempts[ip] >= max_h_shrinks + 1
                        error("Asymptotic FD Jacobian failed for parameter $(PARAM_NAMES[ip]) after $(max_h_shrinks + 1) attempts. Last error: $reason")
                    end
                    @warn "Retrying calibration asymptotic FD Jacobian with smaller h" parameter = PARAM_NAMES[ip] h = h_current[ip] attempt = attempts[ip] reason = reason
                    h_current[ip] *= h_shrink_factor
                    push!(next_pending, ip)
                end
            end
            pending = next_pending
        end
    else
        col_errors = fill("", 5)

        compute_column! = function (ip::Int)
            h = max(h_abs[ip], h_rel * max(abs(theta[ip]), 1.0))
            accepted = false
            last_err = ""
            for attempt in 0:max_h_shrinks
                tp = collect(theta)
                tm = collect(theta)
                tp[ip] += h
                tm[ip] -= h
                theta_p = (tp[1], tp[2], tp[3], tp[4], tp[5])
                theta_m = (tm[1], tm[2], tm[3], tm[4], tm[5])

                seed = seed_base + 1_000_000 * ip + 100_000 * attempt + 10_000_000 * iter_id
                @info "Computing calibration asymptotic FD Jacobian column" parameter = PARAM_NAMES[ip] h n_samples spinup seed attempt = attempt + 1
                try
                    avg_p = compute_steady_state_observables(theta_p, cfg, n_samples, spinup, seed)
                    avg_m = compute_steady_state_observables(theta_m, cfg, n_samples, spinup, seed)
                    col = (avg_p .- avg_m) ./ (2h)
                    all(isfinite, col) || error("FD column has non-finite entries")
                    @views J[:, ip] .= col
                    accepted = true
                    break
                catch err
                    last_err = sprint(showerror, err)
                    if attempt == max_h_shrinks
                        break
                    end
                    @warn "Retrying calibration asymptotic FD Jacobian with smaller h" parameter = PARAM_NAMES[ip] h attempt = attempt + 1 reason = last_err
                    h *= h_shrink_factor
                end
            end

            if accepted
                col_errors[ip] = ""
            else
                col_errors[ip] = "Asymptotic FD Jacobian failed for parameter $(PARAM_NAMES[ip]) after $(max_h_shrinks + 1) attempts. Last error: $last_err"
            end
            return nothing
        end

        if parallel_fd_columns && (allow_nested_parallel || !nested_parallel_active)
            Base.Threads.@threads for ip in 1:5
                compute_column!(ip)
            end
        else
            for ip in 1:5
                compute_column!(ip)
            end
        end

        for ip in 1:5
            isempty(col_errors[ip]) || error(col_errors[ip])
        end
    end

    all(isfinite, J) || error("Asymptotic finite-difference Jacobian contains non-finite values")
    return J
end

function apply_sigma_surgical_mask!(S::Matrix{Float64}; row_var::Int=2, col_sigma::Int=5)
    n_obs, n_param = size(S)
    col_sigma in 1:n_param || error("Sigma column index out of bounds: col_sigma=$(col_sigma), n_param=$(n_param)")
    row_var in 1:n_obs || error("Variance row index out of bounds: row_var=$(row_var), n_obs=$(n_obs)")
    @inbounds for j in 1:n_obs
        if j != row_var
            S[j, col_sigma] = 0.0
        end
    end
    return S
end

function compute_iteration_jacobians(cfg, theta, gfdt_data_path, checkpoint_path, iteration, run_dir)
    X = load_x_matrix(gfdt_data_path, cfg["datasets.gfdt_key"], cfg["integration.K"])
    obs_ref = obs_ref_tuple(cfg)
    obs_m = Int(cfg["observables.m"])
    n_obs = observable_count(cfg)

    A = ArnoldCommon.compute_observables_series(
        X,
        obs_ref.F_ref,
        obs_ref.alpha0_ref,
        obs_ref.alpha1_ref,
        obs_ref.alpha2_ref,
        obs_ref.alpha3_ref,
        obs_m,
    )
    G_obs = vec(mean(A; dims=2))
    all(isfinite, G_obs) || error("Observable averages contain non-finite values")

    cfg["runtime.current_observable_series"] = A
    cfg["runtime.current_gfdt_path"] = gfdt_data_path

    n_lags, dt_obs = response_n_lags(cfg, size(X, 2))
    apply_sigma_mask = Bool(cfg["calibration.apply_sigma_jacobian_mask"])

    out = Dict{String,NamedTuple{(:S,:G,:A,:times,:R_step,:C),Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Vector{Float64},Array{Float64,3},Array{Float64,3}}}}()

    if cfg["methods.unet"]
        checkpoint_path === nothing && error("UNet Jacobian requested but checkpoint_path is nothing")
        isfile(checkpoint_path) || error("UNet checkpoint not found: $checkpoint_path")
        unet_conjugates = require_main_symbol(:unet_conjugates)
        extract_asymptotic_jacobians = require_main_symbol(:extract_asymptotic_jacobians)

        unet = unet_conjugates(
            X,
            theta[5],
            checkpoint_path;
            batch_size=cfg["responses.batch_size"],
            score_device=cfg["responses.score_device"],
            score_forward_mode=cfg["responses.score_forward_mode"],
            apply_correction=cfg["responses.apply_score_correction"],
            divergence_mode=cfg["responses.divergence_mode"],
            divergence_eps=cfg["responses.divergence_eps"],
            divergence_probes=cfg["responses.divergence_probes"],
            divergence_seed=cfg["datasets.gfdt_rng_seed_base"] + 10_000 + iteration,
        )

        G_conj = cfg["responses.apply_score_correction"] ? unet.G_corr : unet.G_raw
        C, R_step, times = ArnoldCommon.build_gfdt_response(
            A,
            G_conj,
            dt_obs,
            n_lags;
            mean_center=cfg["responses.mean_center"],
            impulse_tail_debias=cfg["responses.impulse_tail_debias"],
            t_start=cfg["responses.t_start"],
            t_end=cfg["responses.t_end"],
            tail_taper=cfg["responses.impulse_tail_taper"],
        )
        S = extract_asymptotic_jacobians(times, R_step; t_start=cfg["responses.t_start"], t_end=cfg["responses.t_end"])
        if apply_sigma_mask
            apply_sigma_surgical_mask!(S; row_var=2, col_sigma=5)
        end
        all(isfinite, S) || error("UNet Jacobian contains non-finite values")

        out["unet"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.gaussian"]
        gaussian_conjugates = require_main_symbol(:gaussian_conjugates)
        extract_asymptotic_jacobians = require_main_symbol(:extract_asymptotic_jacobians)

        gauss = gaussian_conjugates(
            X,
            theta[5];
            apply_correction=cfg["responses.apply_score_correction"],
        )

        G_conj = cfg["responses.apply_score_correction"] ? gauss.G_corr : gauss.G_raw
        C, R_step, times = ArnoldCommon.build_gfdt_response(
            A,
            G_conj,
            dt_obs,
            n_lags;
            mean_center=cfg["responses.mean_center"],
            impulse_tail_debias=cfg["responses.impulse_tail_debias"],
            t_start=cfg["responses.t_start"],
            t_end=cfg["responses.t_end"],
            tail_taper=cfg["responses.impulse_tail_taper"],
        )
        S = extract_asymptotic_jacobians(times, R_step; t_start=cfg["responses.t_start"], t_end=cfg["responses.t_end"])
        if apply_sigma_mask
            apply_sigma_surgical_mask!(S; row_var=2, col_sigma=5)
        end
        all(isfinite, S) || error("Gaussian Jacobian contains non-finite values")

        out["gaussian"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.finite_difference"]
        Sfd = compute_fd_jacobian_asymptotic(theta, cfg)
        all(isfinite, Sfd) || error("Finite-difference Jacobian contains non-finite values")
        out["finite_difference"] = (
            S=Sfd,
            G=G_obs,
            A=A,
            times=Float64[],
            R_step=zeros(Float64, n_obs, 5, 0),
            C=zeros(Float64, n_obs, 5, 0),
        )
    end

    return out
end

function diagonal_weight(A_subseries::Matrix{Float64})
    vars = vec(var(A_subseries; dims=2, corrected=true))
    vars = max.(vars, 1e-10)
    return Diagonal(1.0 ./ vars)
end

function inverse_cov_weight(A_subseries::Matrix{Float64}; base_jitter::Float64=1e-10, max_tries::Int=8)
    n_obs, T = size(A_subseries)
    T >= 2 || error("Need at least two samples to build inverse-covariance weight matrix")

    mu = mean(A_subseries; dims=2)
    A_centered = A_subseries .- mu
    Sigma_obs = Symmetric((A_centered * A_centered') / max(T - 1, 1))

    jitter = base_jitter
    F = nothing
    for attempt in 1:max_tries
        try
            Sigma_reg = Symmetric(Matrix(Sigma_obs) + jitter * I)
            F = cholesky(Sigma_reg; check=true)
            break
        catch err
            if attempt == max_tries
                error("Failed to factorize observable covariance after $max_tries attempts: $(sprint(showerror, err))")
            end
            jitter *= 10.0
        end
    end
    F === nothing && error("Failed to construct inverse-covariance weight matrix")

    B = F \ Matrix{Float64}(I, n_obs, n_obs)
    return Symmetric(B)
end

function build_weight_matrix(cfg::Dict{String,Any}, active_idx::Vector{Int})
    mode = cfg["calibration.weight_matrix"]
    nobs = length(active_idx)

    if mode == "identity"
        return Matrix{Float64}(I, nobs, nobs)
    end

    A_all = get(cfg, "runtime.current_observable_series", nothing)
    if !(A_all isa Matrix{Float64})
        @warn "Observable series unavailable for requested weight matrix mode; falling back to identity" mode
        return Matrix{Float64}(I, nobs, nobs)
    end

    A_sub = @view A_all[active_idx, :]
    if mode == "diagonal"
        return Matrix(diagonal_weight(Matrix(A_sub)))
    elseif mode == "inverse_cov"
        try
            return Matrix(inverse_cov_weight(Matrix(A_sub)))
        catch err
            @warn "Inverse-covariance weight construction failed; falling back to identity" error = sprint(showerror, err)
            return Matrix{Float64}(I, nobs, nobs)
        end
    end

    error("Unsupported calibration.weight_matrix mode '$mode'")
end

function weighted_residual_norm(W::AbstractMatrix, r::AbstractVector{<:Real})
    v = Float64.(r)
    q = dot(v, W * v)
    q = isfinite(q) ? max(q, 0.0) : Inf
    return sqrt(q)
end

function weighted_observable_residual(cfg::Dict{String,Any},
    G::AbstractVector{<:Real},
    A_target::AbstractVector{<:Real};
    active_idx::Union{Nothing,Vector{Int}}=nothing,
    W::Union{Nothing,AbstractMatrix}=nothing)
    idx = active_idx === nothing ? cfg["calibration.active_observables"] : active_idx
    r = Float64.(G[idx] .- A_target[idx])
    W_use = W === nothing ? build_weight_matrix(cfg, idx) : Matrix{Float64}(W)
    return weighted_residual_norm(W_use, r)
end

function estimate_observables(theta_candidate::Vector{Float64}, cfg::Dict{String,Any})
    return estimate_observables_ensemble(
        theta_candidate,
        cfg;
        trajectories=Int(cfg["observables_ensemble.trajectories"]),
        samples_per_trajectory=Int(cfg["observables_ensemble.samples_per_trajectory"]),
        save_every=Int(cfg["observables_ensemble.save_every"]),
        spinup_steps=Int(cfg["observables_ensemble.spinup_steps"]),
        seed_base=Int(cfg["observables_ensemble.seed_base"]),
        parallel_trajectories=Bool(cfg["observables_ensemble.parallel_trajectories"]),
    )
end

function acceptance_method_seed_offset(cfg::Dict{String,Any})
    return 0
end

function resolve_acceptance_ensemble_settings(cfg::Dict{String,Any};
    reference_residual::Real=Inf,
    cond_sub::Real=NaN)
    return (
        trajectories=Int(cfg["observables_ensemble.trajectories"]),
        samples_per_trajectory=Int(cfg["observables_ensemble.samples_per_trajectory"]),
        save_every=Int(cfg["observables_ensemble.save_every"]),
        spinup_steps=Int(cfg["observables_ensemble.spinup_steps"]),
        seed_base=Int(cfg["observables_ensemble.seed_base"]),
        parallel_trajectories=Bool(cfg["observables_ensemble.parallel_trajectories"]),
        refined=false,
        refine_reason="shared_observable_estimator",
        reference_residual=Float64(reference_residual),
        conditioning_reference=Float64(cond_sub),
    )
end

function estimate_observables_ensemble(theta_candidate::Vector{Float64}, cfg::Dict{String,Any};
    trajectories::Int,
    samples_per_trajectory::Int,
    save_every::Int,
    spinup_steps::Int,
    seed_base::Int,
    parallel_trajectories::Bool,
    seed_offset::Int=0,
    force_serial::Bool=false)
    compute_steady_state_observables = require_main_symbol(:compute_steady_state_observables)
    theta_tuple = theta_to_tuple(theta_candidate)
    n_obs = observable_count(cfg)

    iter_id = Int(get(cfg, "runtime.iteration", 0))
    obs_mat = Matrix{Float64}(undef, n_obs, trajectories)
    traj_errors = fill("", trajectories)

    nested_parallel_active = Bool(get(cfg, "runtime.method_parallel_active", false))
    allow_nested_parallel = Bool(get(cfg, "performance.allow_nested_parallel", true))
    do_parallel = parallel_trajectories && !force_serial && Base.Threads.nthreads() > 1
    do_parallel = do_parallel && (allow_nested_parallel || !nested_parallel_active)

    run_traj! = function (trj::Int)
        seed = seed_base + seed_offset + 1_000_000 * iter_id + 10_000 * trj
        local_cfg = copy(cfg)
        local_cfg["integration.save_every"] = save_every
        try
            obs = compute_steady_state_observables(theta_tuple, local_cfg, samples_per_trajectory, spinup_steps, seed)
            @views obs_mat[:, trj] .= Float64.(obs)
            traj_errors[trj] = ""
        catch err
            traj_errors[trj] = sprint(showerror, err)
        end
        return nothing
    end

    if do_parallel
        Base.Threads.@threads for trj in 1:trajectories
            run_traj!(trj)
        end
    else
        for trj in 1:trajectories
            run_traj!(trj)
        end
    end

    for trj in 1:trajectories
        isempty(traj_errors[trj]) || error("Observable-ensemble trajectory $(trj) failed: $(traj_errors[trj])")
    end

    return vec(mean(obs_mat; dims=2))
end

function estimate_observables_for_convergence(theta_candidate::Vector{Float64}, cfg::Dict{String,Any}; seed_offset::Int=0)
    return estimate_observables_ensemble(
        theta_candidate,
        cfg;
        trajectories=Int(cfg["observables_ensemble.trajectories"]),
        samples_per_trajectory=Int(cfg["observables_ensemble.samples_per_trajectory"]),
        save_every=Int(cfg["observables_ensemble.save_every"]),
        spinup_steps=Int(cfg["observables_ensemble.spinup_steps"]),
        seed_base=Int(cfg["observables_ensemble.seed_base"]),
        parallel_trajectories=Bool(cfg["observables_ensemble.parallel_trajectories"]),
        seed_offset=seed_offset,
    )
end

function estimate_observables_for_acceptance(theta_candidate::Vector{Float64}, cfg::Dict{String,Any};
    seed_offset::Int=0,
    force_serial::Bool=false,
    settings=nothing)
    acc = settings === nothing ? resolve_acceptance_ensemble_settings(cfg) : settings
    method_offset = acceptance_method_seed_offset(cfg)
    return estimate_observables_ensemble(
        theta_candidate,
        cfg;
        trajectories=Int(acc.trajectories),
        samples_per_trajectory=Int(acc.samples_per_trajectory),
        save_every=Int(acc.save_every),
        spinup_steps=Int(acc.spinup_steps),
        seed_base=Int(acc.seed_base),
        parallel_trajectories=Bool(acc.parallel_trajectories),
        seed_offset=method_offset + seed_offset,
        force_serial=force_serial,
    )
end

function relative_step_norm(delta::AbstractVector{<:Real}, theta_ref::AbstractVector{<:Real})
    return norm(Float64.(delta)) / max(norm(Float64.(theta_ref)), eps())
end

function active_relative_step_cap(cfg::Dict{String,Any}, cond_sub::Real)
    cap = Inf
    max_rel = Float64(get(cfg, "calibration.max_relative_step", 0.0))
    if max_rel > 0
        cap = min(cap, max_rel)
    end
    cond_threshold = Float64(get(cfg, "calibration.conditioning_step_threshold", Inf))
    cond_cap = Float64(get(cfg, "calibration.conditioned_max_relative_step", 0.0))
    if isfinite(Float64(cond_sub)) && Float64(cond_sub) >= cond_threshold && cond_cap > 0
        cap = min(cap, cond_cap)
    end
    return cap
end

function theta_candidate_is_stable(theta_candidate::Vector{Float64}, cfg::Dict{String,Any})
    iter_id = Int(get(cfg, "runtime.iteration", 0))
    test_iter = iter_id + 1

    max_abs_state = max(Float64(cfg["numerical.max_abs_state"]), 1e4)
    spinup_steps = Int(cfg["datasets.spinup_steps"])

    specs = (
        (
            nsamples=min(2_000, Int(cfg["datasets.train_nsamples"])),
            save_every=Int(cfg["datasets.train_save_every"]),
            seed=Int(cfg["datasets.train_rng_seed_base"]) + test_iter,
        ),
        (
            nsamples=min(20_000, Int(cfg["datasets.gfdt_nsamples"])),
            save_every=Int(cfg["datasets.gfdt_save_every"]),
            seed=Int(cfg["datasets.gfdt_rng_seed_base"]) + test_iter,
        ),
    )

    for spec in specs
        try
            ArnoldCommon.generate_reduced_x_timeseries(
                K=cfg["integration.K"],
                F=cfg["closure.F"],
                alpha0=theta_candidate[1],
                alpha1=theta_candidate[2],
                alpha2=theta_candidate[3],
                alpha3=theta_candidate[4],
                sigma=max(theta_candidate[5], 1e-8),
                dt=cfg["integration.dt"],
                spinup_steps=spinup_steps,
                save_every=spec.save_every,
                nsamples=spec.nsamples,
                rng_seed=spec.seed,
                max_abs_state=max_abs_state,
                max_restarts=12,
                state_min=cfg["numerical.state_min"],
                state_max=cfg["numerical.state_max"],
                max_boundary_hits=max(3, min(12, Int(cfg["numerical.max_boundary_hits"]))),
            )
        catch
            return false
        end
    end
    return true
end

function perform_newton_update(S, G, A_target, theta, cfg)
    free_idx = cfg["calibration.free_parameters"]
    active_idx = cfg["calibration.active_observables"]

    S_sub = Matrix{Float64}(S[active_idx, free_idx])
    G_sub = Float64.(G[active_idx])
    A_sub = Float64.(A_target[active_idx])

    W_sub = build_weight_matrix(cfg, active_idx)

    nfree = length(free_idx)
    theta_current = Float64.(theta)
    gamma_requested = Float64(cfg["calibration.regularization_gamma"])
    gamma_floor = max(gamma_requested, 1e-8)
    damping_requested = Float64(cfg["calibration.damping"])
    lm_max_attempts = Int(cfg["calibration.lm_max_attempts"])
    lm_growth_factor = Float64(cfg["calibration.lm_growth_factor"])
    min_rho = Float64(cfg["calibration.min_actual_to_predicted_ratio"])

    cond_sub = try
        cond(S_sub)
    catch
        NaN
    end

    line_search_used = Bool(cfg["calibration.line_search"])
    line_search_max = line_search_used ? Int(cfg["calibration.line_search_max"]) : 0
    nested_parallel_active = Bool(get(cfg, "runtime.method_parallel_active", false))
    allow_nested_parallel = Bool(get(cfg, "performance.allow_nested_parallel", true))
    requested_parallel_line_search = Bool(get(cfg, "performance.parallel_line_search", true))

    previous_obs_residual = Float64(get(cfg, "runtime.previous_observable_residual", Inf))
    acceptance_settings = resolve_acceptance_ensemble_settings(
        cfg;
        reference_residual=previous_obs_residual,
        cond_sub=cond_sub,
    )
    acceptance_parallel_eval = Bool(acceptance_settings.parallel_trajectories) &&
        Int(acceptance_settings.trajectories) > 1 &&
        Base.Threads.nthreads() > 1
    do_parallel_line_search = requested_parallel_line_search && line_search_used && line_search_max > 0
    do_parallel_line_search = do_parallel_line_search && (allow_nested_parallel || !nested_parallel_active)
    do_parallel_line_search = do_parallel_line_search && !acceptance_parallel_eval
    force_serial_acceptance = false
    acceptance_fallback_used = false
    acceptance_current = try
        estimate_observables_for_acceptance(
            theta_current,
            cfg;
            force_serial=force_serial_acceptance,
            settings=acceptance_settings,
        )
    catch err
        acceptance_fallback_used = true
        @warn "Acceptance-ensemble estimate failed at current theta; falling back to GFDT observables" method = get(cfg, "runtime.current_method", "") error = sprint(showerror, err)
        vec(Float64.(G))
    end

    residual_before = weighted_residual_norm(W_sub, acceptance_current[active_idx] .- A_sub)
    residual_before_linear = weighted_residual_norm(W_sub, G_sub .- A_sub)
    residual_before_unweighted = norm(acceptance_current[active_idx] .- A_sub)

    theta_best = copy(theta_current)
    applied_step = zeros(Float64, 5)
    residual_after = residual_before
    residual_after_predicted = residual_before
    line_search_scale = 0.0
    line_search_accepted = false
    stability_checked = false
    stability_accepted = false
    raw_diag = (cond=NaN, nrm_rhs=NaN)
    correction_full = zeros(Float64, 5)
    correction_full_raw = zeros(Float64, 5)
    lm_attempts_used = 0
    gamma_effective = gamma_requested
    predicted_reduction = 0.0
    actual_reduction = 0.0
    actual_to_predicted_ratio = -Inf
    step_cap_triggered = false
    step_cap_factor = 1.0
    damping_effective = damping_requested
    best_candidate_residual = Inf

    for lm_attempt in 1:lm_max_attempts
        lm_attempts_used = lm_attempt
        gamma_effective = gamma_requested > 0 ? gamma_requested * (lm_growth_factor ^ (lm_attempt - 1)) :
            gamma_floor * (lm_growth_factor ^ (lm_attempt - 1))
        Gamma_sub = Symmetric(Matrix{Float64}(I, nfree, nfree) * gamma_effective)
        correction_sub, raw_diag = newton_step_bridge(S_sub, W_sub, Gamma_sub, G_sub, A_sub)

        fill!(correction_full_raw, 0.0)
        correction_full_raw[free_idx] .= correction_sub

        damping_effective = damping_requested
        step_cap_triggered = false
        step_cap_factor = 1.0
        rel_cap = active_relative_step_cap(cfg, cond_sub)
        base_relative_step = relative_step_norm(damping_requested .* correction_full_raw, theta_current)
        if isfinite(rel_cap) && base_relative_step > rel_cap
            step_cap_triggered = true
            step_cap_factor = rel_cap / max(base_relative_step, eps())
            damping_effective *= step_cap_factor
        end

        scales = [damping_effective / (2.0 ^ h) for h in 0:line_search_max]
        linear_delta_obs = S_sub * correction_sub

        evaluate_candidate = function (scale::Float64)
            try
                theta_try = theta_current .- scale .* correction_full_raw
                obs_try = estimate_observables_for_acceptance(
                    theta_try,
                    cfg;
                    force_serial=force_serial_acceptance,
                    settings=acceptance_settings,
                )
                res_try = weighted_residual_norm(W_sub, obs_try[active_idx] .- A_sub)
                pred_obs = acceptance_current[active_idx] .- scale .* linear_delta_obs
                pred_res = weighted_residual_norm(W_sub, pred_obs .- A_sub)
                pred_red = residual_before - pred_res
                act_red = residual_before - res_try
                need_stability = act_red > 0
                stable_try = true
                if need_stability
                    stable_try = theta_candidate_is_stable(theta_try, cfg)
                end
                rho = pred_red > eps() ? act_red / pred_red : -Inf
                accepted = act_red > 0 && pred_red > eps() && rho >= min_rho && stable_try
                return (
                    scale=scale,
                    theta_try=theta_try,
                    obs_try=obs_try,
                    res_try=res_try,
                    pred_res=pred_res,
                    pred_red=pred_red,
                    act_red=act_red,
                    rho=rho,
                    need_stability=need_stability,
                    stable_try=stable_try,
                    accepted=accepted,
                    err="",
                )
            catch err
                return (
                    scale=scale,
                    theta_try=copy(theta_current),
                    obs_try=copy(acceptance_current),
                    res_try=Inf,
                    pred_res=Inf,
                    pred_red=-Inf,
                    act_red=-Inf,
                    rho=-Inf,
                    need_stability=false,
                    stable_try=false,
                    accepted=false,
                    err=sprint(showerror, err),
                )
            end
        end

        results = Vector{Any}(undef, length(scales))
        if do_parallel_line_search
            max_parallel = min(length(scales), max(1, Int(get(cfg, "performance.max_parallel_line_search", 3))), Base.Threads.nthreads())
            sem = Base.Semaphore(max_parallel)
            @sync for idx in eachindex(scales)
                scale = scales[idx]
                Base.Threads.@spawn begin
                    Base.acquire(sem)
                    try
                        results[idx] = evaluate_candidate(scale)
                    finally
                        Base.release(sem)
                    end
                end
            end
        else
            for idx in eachindex(scales)
                results[idx] = evaluate_candidate(scales[idx])
            end
        end

        accepted_results = Any[]
        local_best_residual = Inf
        for r in results
            if !isempty(r.err)
                @warn "Acceptance evaluation failed for candidate theta; trying stronger regularization" scale = r.scale gamma = gamma_effective error = r.err
                continue
            end
            if r.need_stability
                stability_checked = true
            end
            if r.need_stability && !r.stable_try
                @warn "Rejecting Newton candidate: unstable stochastic rollout" scale = r.scale gamma = gamma_effective res_try = r.res_try
                continue
            end
            local_best_residual = min(local_best_residual, r.res_try)
            if r.accepted
                push!(accepted_results, r)
            end
        end
        best_candidate_residual = min(best_candidate_residual, local_best_residual)

        if !isempty(accepted_results)
            best_idx = argmin([r.res_try for r in accepted_results])
            chosen = accepted_results[best_idx]
            theta_best .= chosen.theta_try
            applied_step .= chosen.scale .* correction_full_raw
            correction_full .= correction_full_raw
            residual_after = chosen.res_try
            residual_after_predicted = chosen.pred_res
            predicted_reduction = chosen.pred_red
            actual_reduction = chosen.act_red
            actual_to_predicted_ratio = chosen.rho
            line_search_scale = chosen.scale
            line_search_accepted = true
            stability_accepted = true
            break
        end
    end

    if !line_search_accepted
        theta_best .= theta_current
        applied_step .= 0.0
        correction_full .= correction_full_raw
        residual_after = residual_before
        residual_after_predicted = residual_before
        predicted_reduction = 0.0
        actual_reduction = 0.0
        actual_to_predicted_ratio = -Inf
        line_search_scale = 0.0
        stability_accepted = false
    end

    diagnostics = Dict{String,Any}(
        "raw_cond" => get(raw_diag, :cond, NaN),
        "raw_rhs_norm" => get(raw_diag, :nrm_rhs, NaN),
        "residual_before" => residual_before,
        "residual_before_linear" => residual_before_linear,
        "residual_before_unweighted" => residual_before_unweighted,
        "residual_after" => residual_after,
        "residual_after_predicted" => residual_after_predicted,
        "line_search_used" => line_search_used,
        "parallel_line_search_requested" => requested_parallel_line_search,
        "parallel_line_search_used" => do_parallel_line_search,
        "line_search_accepted" => line_search_accepted,
        "line_search_scale" => line_search_scale,
        "stability_checked" => stability_checked,
        "stability_accepted" => stability_accepted,
        "lm_attempts_used" => lm_attempts_used,
        "gamma_requested" => gamma_requested,
        "gamma_effective" => gamma_effective,
        "predicted_reduction" => predicted_reduction,
        "actual_reduction" => actual_reduction,
        "actual_to_predicted_ratio" => actual_to_predicted_ratio,
        "acceptance_fallback_used" => acceptance_fallback_used,
        "acceptance_trajectories" => Int(acceptance_settings.trajectories),
        "acceptance_samples_per_trajectory" => Int(acceptance_settings.samples_per_trajectory),
        "acceptance_save_every" => Int(acceptance_settings.save_every),
        "acceptance_spinup_steps" => Int(acceptance_settings.spinup_steps),
        "acceptance_refined" => Bool(acceptance_settings.refined),
        "acceptance_refine_reason" => String(acceptance_settings.refine_reason),
        "acceptance_reference_residual" => Float64(acceptance_settings.reference_residual),
        "acceptance_conditioning_reference" => Float64(acceptance_settings.conditioning_reference),
        "acceptance_parallel_trajectories" => Bool(acceptance_settings.parallel_trajectories),
        "acceptance_parallel_eval" => acceptance_parallel_eval,
        "damping_requested" => damping_requested,
        "damping_effective" => damping_effective,
        "relative_step_cap_active" => isfinite(active_relative_step_cap(cfg, cond_sub)),
        "relative_step_cap" => isfinite(active_relative_step_cap(cfg, cond_sub)) ? active_relative_step_cap(cfg, cond_sub) : 0.0,
        "relative_step_cap_triggered" => step_cap_triggered,
        "relative_step_cap_factor" => step_cap_factor,
        "free_parameters" => free_idx,
        "active_observables" => active_idx,
        "weight_matrix_mode" => cfg["calibration.weight_matrix"],
        "correction_full" => copy(correction_full),
        "correction_full_raw" => copy(correction_full_raw),
        "applied_step" => copy(applied_step),
        "active_subproblem_cond" => cond_sub,
        "best_candidate_residual" => best_candidate_residual,
    )

    return theta_best, applied_step, diagnostics
end

function check_convergence(state::CalibrationState, cfg::Dict)
    tol_theta = cfg["calibration.tol_theta"]
    tol_obs = cfg["calibration.tol_obs"]

    theta_ok = state.convergence_metric < tol_theta

    obs_ok = false
    A_target = get(cfg, "runtime.A_target", nothing)
    G_primary = get(cfg, "runtime.primary_observables", nothing)
    if A_target isa Vector{Float64} && G_primary isa Vector{Float64}
        active_idx = cfg["calibration.active_observables"]
        obs_residual = weighted_observable_residual(cfg, G_primary, A_target; active_idx=active_idx)
        cfg["runtime.last_obs_residual"] = obs_residual
        obs_ok = obs_residual < tol_obs
    else
        cfg["runtime.last_obs_residual"] = Inf
    end

    return theta_ok || obs_ok
end

function style_for_method(method::String)
    if method == "unet"
        return (label="UNet", color=:orangered3, linestyle=:solid, marker=:circle, markersize=5)
    elseif method == "gaussian"
        return (label="Gaussian", color=:black, linestyle=:dash, marker=:utriangle, markersize=5)
    elseif method == "finite_difference"
        return (label="Finite-difference", color=:dodgerblue3, linestyle=:solid, marker=:diamond, markersize=5)
    end
    return (label=method, color=:gray30, linestyle=:solid, marker=:auto, markersize=4)
end

function save_response_iteration_archive(path::AbstractString,
    cfg::Dict{String,Any},
    jacobians,
    observables::Dict{String,Vector{Float64}})
    obs_names = observable_names(Int(cfg["observables.m"]))
    obs_labels = observable_labels(Int(cfg["observables.m"]))
    active_idx = Vector{Int}(get(cfg, "runtime.iteration_active_observables", cfg["calibration.active_observables"]))
    free_idx = Vector{Int}(get(cfg, "runtime.iteration_free_parameters", cfg["calibration.free_parameters"]))
    A_target = get(cfg, "runtime.A_target", nothing)

    mkpath(dirname(path))
    h5open(path, "w") do h5
        h5["meta/observable_names"] = obs_names
        h5["meta/observable_labels"] = obs_labels
        h5["meta/param_names"] = PARAM_NAMES
        h5["meta/active_observables"] = active_idx
        h5["meta/free_parameters"] = free_idx
        if A_target isa Vector{Float64}
            h5["target/observables"] = A_target
        end

        for method in sort(collect(keys(jacobians)))
            jr = jacobians[method]
            current_obs = get(observables, method, jr.G)

            h5[joinpath("jacobians", method)] = jr.S
            h5[joinpath("observables", method, "gfdt_mean")] = jr.G
            h5[joinpath("observables", method, "committed")] = current_obs
            h5[joinpath("responses", method, "times")] = jr.times
            h5[joinpath("responses", method, "R_step")] = jr.R_step
            h5[joinpath("responses", method, "C")] = jr.C
            h5[joinpath("selected_jacobians", method)] = jr.S[active_idx, :]
            h5[joinpath("selected_observables", method)] = current_obs[active_idx]
            if size(jr.R_step, 3) == 0
                h5[joinpath("selected_responses", method, "times")] = jr.times
                h5[joinpath("selected_responses", method, "R_step")] = zeros(Float64, length(active_idx), size(jr.S, 2), 0)
                h5[joinpath("selected_responses", method, "C")] = zeros(Float64, length(active_idx), size(jr.S, 2), 0)
            else
                h5[joinpath("selected_responses", method, "times")] = jr.times
                h5[joinpath("selected_responses", method, "R_step")] = jr.R_step[active_idx, :, :]
                h5[joinpath("selected_responses", method, "C")] = jr.C[active_idx, :, :]
            end
        end
    end
    return path
end

function save_response_iteration_figure(path::AbstractString, cfg::Dict, jacobians)
    save_response_figure = require_main_symbol(:save_response_figure)
    active_idx = Vector{Int}(get(cfg, "runtime.iteration_active_observables", cfg["calibration.active_observables"]))
    isempty(active_idx) && return ""
    obs_labels_all = observable_labels(Int(cfg["observables.m"]))
    selected_obs_labels = [obs_labels_all[i] for i in active_idx]

    curves = NamedTuple[]
    times = nothing
    for method in method_order()
        haskey(jacobians, method) || continue
        jr = jacobians[method]
        if length(jr.times) == 0 || size(jr.R_step, 3) == 0
            continue
        end
        style = style_for_method(method)
        if times === nothing
            times = jr.times
        end
        push!(curves, (
            method_key=method,
            label=style.label,
            color=style.color,
            linestyle=style.linestyle,
            data=jr.R_step[active_idx, :, :],
        ))
    end

    isempty(curves) && return ""

    asymptotic_curves = NamedTuple[]
    for method in method_order()
        haskey(jacobians, method) || continue
        style = style_for_method(method)
        push!(asymptotic_curves, (
            label=style.label,
            color=style.color,
            linestyle=style.linestyle,
            jacobians=jacobians[method].S[active_idx, :],
        ))
    end

    # Guarantee that finite-difference asymptotes are present whenever available,
    # even though finite differences have no time-dependent GFDT curve.
    if haskey(jacobians, "finite_difference") && !any(c -> c.label == "Finite-difference", asymptotic_curves)
        style = style_for_method("finite_difference")
        push!(asymptotic_curves, (
            label=style.label,
            color=style.color,
            linestyle=style.linestyle,
            jacobians=jacobians["finite_difference"].S[active_idx, :],
        ))
    end

    title_text = if length(active_idx) == observable_count(cfg)
        "Calibration iteration responses"
    else
        "Calibration iteration responses (tuned observables)"
    end

    return save_response_figure(
        path,
        times,
        curves;
        asymptotic_curves=asymptotic_curves,
        title_text=title_text,
        dpi=cfg["figures.dpi"],
        observable_row_labels=selected_obs_labels,
    )
end

function save_stats_iteration_figure(path::AbstractString, cfg::Dict)
    isdefined(Main, :ArnoldStatsPlots) || return ""
    sp = getproperty(Main, :ArnoldStatsPlots)

    X_truth = get(cfg, "runtime.truth_matrix", nothing)
    X_truth isa Matrix{Float64} || return ""

    gfdt_path = get(cfg, "runtime.current_gfdt_path", "")
    isempty(strip(gfdt_path)) && return ""
    isfile(gfdt_path) || return ""

    X_model = load_x_matrix(gfdt_path, cfg["datasets.gfdt_key"], cfg["integration.K"])

    n_truth = min(size(X_truth, 2), 50_000)
    n_model = min(size(X_model, 2), 50_000)
    obs_tensor = tensor_from_matrix(X_truth[:, 1:n_truth])
    gen_tensor = tensor_from_matrix(X_model[:, 1:n_model])
    truth_dt = Float64(cfg["truth.save_every"]) * Float64(cfg["integration.dt"])
    model_dt = Float64(cfg["datasets.gfdt_save_every"]) * Float64(cfg["integration.dt"])

    kl_mode, js_mode = sp.modewise_metrics(obs_tensor, gen_tensor; nbins=80, low_q=0.001, high_q=0.999)
    return sp.save_stats_figure_acf(
        path,
        obs_tensor,
        gen_tensor,
        kl_mode,
        js_mode,
        80;
        max_lag=200,
        obs_dt=truth_dt,
        gen_dt=model_dt,
        obs_label="Truth",
        gen_label="Current model",
    )
end

function save_iteration_outputs(state, cfg, run_dir, iteration, jacobians, observables)
    iter_dir = joinpath(run_dir, @sprintf("iter_%03d", iteration))
    results_dir = joinpath(iter_dir, "results")
    figures_dir = joinpath(iter_dir, "figures")
    mkpath(results_dir)
    mkpath(figures_dir)

    params_path = joinpath(results_dir, "parameters.toml")
    params_doc = Dict{String,Any}(
        "iteration" => iteration,
        "theta" => Dict(
            "alpha0" => state.theta[1],
            "alpha1" => state.theta[2],
            "alpha2" => state.theta[3],
            "alpha3" => state.theta[4],
            "sigma" => state.theta[5],
        ),
        "theta_history_length" => length(state.theta_history),
    )
    open(params_path, "w") do io
        TOML.print(io, params_doc)
    end

    jac_path = joinpath(results_dir, "jacobians.hdf5")
    save_response_iteration_archive(jac_path, cfg, jacobians, observables)

    obs_path = joinpath(results_dir, "observables.csv")
    A_target = get(cfg, "runtime.A_target", nothing)
    obs_names = observable_names(Int(cfg["observables.m"]))
    open(obs_path, "w") do io
        println(io, "method,observable,value")
        for method in sort(collect(keys(observables)))
            vals = observables[method]
            for i in 1:min(length(vals), length(obs_names))
                @printf(io, "%s,%s,%.16e\n", method, obs_names[i], vals[i])
            end
        end
        if A_target isa Vector{Float64}
            for i in 1:min(length(A_target), length(obs_names))
                @printf(io, "target,%s,%.16e\n", obs_names[i], A_target[i])
            end
        end
    end

    diag_path = joinpath(results_dir, "diagnostics.toml")
    diag = get(cfg, "runtime.iteration_diagnostics", Dict{String,Any}())
    open(diag_path, "w") do io
        TOML.print(io, diag)
    end

    if cfg["figures.save_per_iteration"]
        response_fig = joinpath(figures_dir, @sprintf("responses_%dx5.png", observable_count(cfg)))
        try
            save_response_iteration_figure(response_fig, cfg, jacobians)
        catch err
            @warn "Failed to save response figure" path = response_fig error = sprint(showerror, err)
        end

        stats_fig = joinpath(figures_dir, "stats_comparison.png")
        try
            save_stats_iteration_figure(stats_fig, cfg)
        catch err
            @warn "Failed to save stats comparison figure" path = stats_fig error = sprint(showerror, err)
        end

        if cfg["figures.save_langevin_stats"]
            score_langevin_fig = joinpath(figures_dir, "figB_train_vs_score_langevin_4x2.png")
            score_langevin_archive = joinpath(results_dir, "score_langevin_samples.hdf5")
            score_langevin_summary = joinpath(results_dir, "score_langevin_summary.toml")
            train_data_path = get(cfg, "runtime.current_train_path", "")
            checkpoint_path = get(cfg, "runtime.current_checkpoint_path", "")
            try
                fig_saved = save_score_langevin_iteration_figure(
                    score_langevin_fig,
                    cfg,
                    train_data_path,
                    checkpoint_path,
                    iteration,
                    archive_path=score_langevin_archive,
                    summary_path=score_langevin_summary,
                )
                if !isempty(fig_saved)
                    @info "Saved per-iteration score-Langevin stats figure" iteration path = fig_saved method = get(cfg, "runtime.current_langevin_method", "")
                end
            catch err
                @warn "Failed to save score-Langevin stats figure" iteration path = score_langevin_fig error = sprint(showerror, err)
            end
        end
    end

    return results_dir
end

function true_theta_reference(cfg::Dict{String,Any})
    return [
        cfg["closure.alpha0_initial"],
        cfg["closure.alpha1_initial"],
        cfg["closure.alpha2_initial"],
        cfg["closure.alpha3_initial"],
        cfg["closure.sigma_initial"],
    ]
end

function save_convergence_figure(state, cfg, run_dir)
    cfg["figures.save_convergence"] || return ""

    free_idx = cfg["calibration.free_parameters"]
    active_idx = cfg["calibration.active_observables"]
    methods = enabled_methods(cfg)

    ncols = max(length(free_idx), length(active_idx))
    ncols >= 1 || error("Need at least one free parameter or active observable for convergence plot")

    default(fontfamily="Computer Modern", dpi=cfg["figures.dpi"], legendfontsize=8, guidefontsize=9, tickfontsize=8, titlefontsize=10)

    panels = Vector{Any}(undef, 2 * ncols)
    theta_ref = true_theta_reference(cfg)
    A_target = get(cfg, "runtime.A_target", nothing)
    obs_names = observable_names(Int(cfg["observables.m"]))

    for col in 1:ncols
        if col <= length(free_idx)
            pidx = free_idx[col]
            pn = plot(; title="$(PARAM_NAMES[pidx])", xlabel="iteration", ylabel="value", legend=(col == 1 ? :best : false))
            for method in methods
                style = style_for_method(method)
                path = get(state.theta_per_method, method, Vector{Vector{Float64}}())
                isempty(path) && continue
                ys = Float64[v[pidx] for v in path]
                xs = collect(0:(length(ys) - 1))
                plot!(pn, xs, ys; color=style.color, linestyle=style.linestyle, linewidth=2.0,
                      marker=style.marker, markersize=style.markersize, markerstrokewidth=0.5,
                      label=(col == 1 ? style.label : ""))
            end
            hline!(pn, [theta_ref[pidx]]; color=:gray40, linestyle=:dash, linewidth=1.5, label=(col == 1 ? "reference" : ""))
            panels[col] = pn
        else
            panels[col] = plot(; axis=false, grid=false)
        end
    end

    for col in 1:ncols
        idx = ncols + col
        if col <= length(active_idx)
            oidx = active_idx[col]
            obs_title = oidx <= length(obs_names) ? obs_names[oidx] : "obs_$oidx"
            pn = plot(; title=obs_title, xlabel="iteration", ylabel="observable", legend=(col == 1 ? :best : false))
            for method in methods
                style = style_for_method(method)
                obs_hist = get(state.obs_history, method, Vector{Vector{Float64}}())
                isempty(obs_hist) && continue
                ys = Float64[v[oidx] for v in obs_hist]
                xs = collect(0:(length(ys) - 1))
                plot!(pn, xs, ys; color=style.color, linestyle=style.linestyle, linewidth=2.0,
                      marker=style.marker, markersize=style.markersize, markerstrokewidth=0.5,
                      label=(col == 1 ? style.label : ""))
            end
            if A_target isa Vector{Float64}
                hline!(pn, [A_target[oidx]]; color=:gray40, linestyle=:dash, linewidth=1.5, label=(col == 1 ? "target" : ""))
            end
            panels[idx] = pn
        else
            panels[idx] = plot(; axis=false, grid=false)
        end
    end

    fig = plot(
        panels...;
        layout=(2, ncols),
        size=(max(1200, 380 * ncols), 900),
        left_margin=6Plots.mm,
        right_margin=6Plots.mm,
        top_margin=8Plots.mm,
        bottom_margin=6Plots.mm,
        plot_title="Calibration convergence",
        plot_titlefontsize=13,
    )

    out_path = joinpath(run_dir, "convergence.png")
    mkpath(dirname(out_path))
    savefig(fig, out_path)
    return out_path
end

end
