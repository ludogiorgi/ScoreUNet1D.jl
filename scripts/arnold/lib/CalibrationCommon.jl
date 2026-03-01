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
       compute_iteration_jacobians,
       perform_newton_update,
       check_convergence,
       save_iteration_outputs,
       save_convergence_figure,
       estimate_observables,
    estimate_observables_for_convergence,
       CalibrationState

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
    truth = require_table(doc, "truth")
    initial_theta = require_table(doc, "initial_theta")
    calibration = require_table(doc, "calibration")
    methods = require_table(doc, "methods")
    datasets = require_table(doc, "datasets")
    training = require_table(doc, "training")
    responses = require_table(doc, "responses")
    responses_fd = maybe_subtable(responses, "finite_difference")
    observables_ensemble = maybe_table(doc, "observables_ensemble")
    figures = maybe_table(doc, "figures")
    performance = maybe_table(doc, "performance")

    data_params_path = abspath(as_str(paths, "data_params", "scripts/arnold/parameters_data.toml"))
    train_params_path = abspath(as_str(paths, "train_params", "scripts/arnold/parameters_train.toml"))
    responses_params_path = abspath(as_str(paths, "responses_params", "scripts/arnold/parameters_responses.toml"))

    data_cfg, _ = ArnoldCommon.load_data_config(data_params_path)
    train_doc = TOML.parsefile(train_params_path)
    responses_doc = TOML.parsefile(responses_params_path)

    train_tbl = maybe_table(train_doc, "train")
    responses_gfdt_tbl = maybe_table(responses_doc, "gfdt")
    responses_numerical_tbl = maybe_table(responses_doc, "numerical")

    cfg = Dict{String,Any}(
        "paths.params_file" => abspath(path),
        "paths.data_params" => data_params_path,
        "paths.train_params" => train_params_path,
        "paths.responses_params" => responses_params_path,
        "paths.runs_root" => abspath(as_str(paths, "runs_root", "scripts/arnold/runs_calibration")),

        "integration.K" => data_cfg["twoscale.K"],
        "integration.J" => data_cfg["twoscale.J"],
        "integration.F" => data_cfg["twoscale.F"],
        "integration.h" => data_cfg["twoscale.h"],
        "integration.c" => data_cfg["twoscale.c"],
        "integration.b" => data_cfg["twoscale.b"],
        "integration.dt" => data_cfg["twoscale.dt"],

        "closure.F" => data_cfg["closure.F"],
        "closure.alpha0_initial" => data_cfg["closure.alpha0_initial"],
        "closure.alpha1_initial" => data_cfg["closure.alpha1_initial"],
        "closure.alpha2_initial" => data_cfg["closure.alpha2_initial"],
        "closure.alpha3_initial" => data_cfg["closure.alpha3_initial"],
        "closure.sigma_initial" => data_cfg["closure.sigma_initial"],
        "closure.auto_fit" => data_cfg["closure.auto_fit"],

        "observables.F_ref" => data_cfg["closure.F"],
        "observables.alpha0_ref" => data_cfg["closure.alpha0_initial"],
        "observables.alpha1_ref" => data_cfg["closure.alpha1_initial"],
        "observables.alpha2_ref" => data_cfg["closure.alpha2_initial"],
        "observables.alpha3_ref" => data_cfg["closure.alpha3_initial"],

        "truth.source" => lowercase(as_str(truth, "source", "two_scale")),
        "truth.nsamples" => as_int(truth, "nsamples", 200_000),
        "truth.spinup_steps" => as_int(truth, "spinup_steps", 2_000),
        "truth.rng_seed" => as_int(truth, "rng_seed", 101),
        "truth.save_every" => as_int(truth, "save_every", 200),
        "truth.truth_file" => as_str(truth, "truth_file", ""),
        "truth.truth_key" => as_str(truth, "truth_key", ""),

        "initial_theta.alpha0" => as_float(initial_theta, "alpha0", data_cfg["closure.alpha0_initial"]),
        "initial_theta.alpha1" => as_float(initial_theta, "alpha1", data_cfg["closure.alpha1_initial"]),
        "initial_theta.alpha2" => as_float(initial_theta, "alpha2", data_cfg["closure.alpha2_initial"]),
        "initial_theta.alpha3" => as_float(initial_theta, "alpha3", data_cfg["closure.alpha3_initial"]),
        "initial_theta.sigma" => as_float(initial_theta, "sigma", data_cfg["closure.sigma_initial"]),
        "initial_theta.perturbation_fraction" => as_float(initial_theta, "perturbation_fraction", 0.0),

        "calibration.max_iterations" => as_int(calibration, "max_iterations", 10),
        "calibration.tol_theta" => as_float(calibration, "tol_theta", 1e-5),
        "calibration.tol_obs" => as_float(calibration, "tol_obs", 1e-4),
        "calibration.damping" => as_float(calibration, "damping", 1.0),
        "calibration.line_search" => as_bool(calibration, "line_search", true),
        "calibration.line_search_max" => as_int(calibration, "line_search_max", 5),
        "calibration.regularization_gamma" => as_float(calibration, "regularization_gamma", 1e-4),
        "calibration.weight_matrix" => lowercase(as_str(calibration, "weight_matrix", "identity")),

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

        "responses.response_tmax" => as_float(responses, "response_tmax", as_float(responses_gfdt_tbl, "response_tmax", 2.0)),
        "responses.mean_center" => as_bool(responses, "mean_center", true),
        "responses.apply_score_correction" => as_bool(responses, "apply_score_correction", true),
        "responses.divergence_mode" => lowercase(as_str(responses, "divergence_mode", as_str(responses_gfdt_tbl, "divergence_mode", "hutchinson"))),
        "responses.divergence_eps" => as_float(responses, "divergence_eps", as_float(responses_gfdt_tbl, "divergence_eps", 0.03)),
        "responses.divergence_probes" => as_int(responses, "divergence_probes", as_int(responses_gfdt_tbl, "divergence_probes", 10)),
        "responses.score_device" => as_str(responses, "score_device", as_str(responses_gfdt_tbl, "score_device", "CPU")),
        "responses.score_forward_mode" => lowercase(as_str(responses_gfdt_tbl, "score_forward_mode", "test")),
        "responses.batch_size" => as_int(responses, "batch_size", as_int(responses_gfdt_tbl, "batch_size", 1024)),
        "responses.tail_window" => as_float(responses, "tail_window", 1.0),

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

        "observables_ensemble.trajectories" => as_int(observables_ensemble, "trajectories", 24),
        "observables_ensemble.samples_per_trajectory" => as_int(observables_ensemble, "samples_per_trajectory", 4_000),
        "observables_ensemble.save_every" => as_int(observables_ensemble, "save_every", as_int(datasets, "gfdt_save_every", data_cfg["datasets.gfdt_stochastic.save_every"])),
        "observables_ensemble.spinup_steps" => as_int(observables_ensemble, "spinup_steps", data_cfg["datasets.train_stochastic.spinup_steps"]),
        "observables_ensemble.seed_base" => as_int(observables_ensemble, "seed_base", 60_000_000),
        "observables_ensemble.parallel_trajectories" => as_bool(observables_ensemble, "parallel_trajectories", true),

        "numerical.max_abs_state" => max(as_float(responses_numerical_tbl, "max_abs_state", 80.0), 1e4),
        "numerical.min_valid_fraction" => as_float(responses_numerical_tbl, "min_valid_fraction", 0.8),
        "numerical.max_h_shrinks" => as_int(responses_numerical_tbl, "max_h_shrinks", 6),
        "numerical.thread_chunk_size" => as_int(responses_numerical_tbl, "thread_chunk_size", 256),
        "numerical.h_shrink_factor" => as_float(responses_numerical_tbl, "h_shrink_factor", 0.5),

        "figures.dpi" => as_int(figures, "dpi", 180),
        "figures.save_per_iteration" => as_bool(figures, "save_per_iteration", true),
        "figures.save_convergence" => as_bool(figures, "save_convergence", true),

        "performance.parallel_methods" => as_bool(performance, "parallel_methods", true),
        "performance.parallel_fd_columns" => as_bool(performance, "parallel_fd_columns", true),
        "performance.parallel_line_search" => as_bool(performance, "parallel_line_search", true),
        "performance.max_parallel_line_search" => as_int(performance, "max_parallel_line_search", 3),
        "performance.allow_nested_parallel" => as_bool(performance, "allow_nested_parallel", true),
    )

    cfg["calibration.free_parameters"] = normalize_indices(as_int_vec(calibration, "free_parameters", Int[]), 5, "calibration.free_parameters")
    cfg["calibration.active_observables"] = normalize_indices(as_int_vec(calibration, "active_observables", Int[]), 5, "calibration.active_observables")

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
    cfg["calibration.weight_matrix"] in ("identity", "inverse_cov", "diagonal") || error("calibration.weight_matrix must be identity, inverse_cov, or diagonal")

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
    cfg["responses.batch_size"] >= 1 || error("responses.batch_size must be >= 1")
    cfg["responses.tail_window"] >= 0 || error("responses.tail_window must be >= 0")

    length(cfg["responses.finite_difference.h_abs"]) == 5 || error("responses.finite_difference.h_abs must have length 5")
    cfg["responses.finite_difference.h_rel"] > 0 || error("responses.finite_difference.h_rel must be > 0")
    cfg["responses.finite_difference.ensemble_trajectories"] >= 1 || error("responses.finite_difference.ensemble_trajectories must be >= 1")
    cfg["responses.finite_difference.samples_per_trajectory"] >= 1 || error("responses.finite_difference.samples_per_trajectory must be >= 1")
    cfg["responses.finite_difference.save_every"] >= 1 || error("responses.finite_difference.save_every must be >= 1")

    cfg["observables_ensemble.trajectories"] >= 1 || error("observables_ensemble.trajectories must be >= 1")
    cfg["observables_ensemble.samples_per_trajectory"] >= 1 || error("observables_ensemble.samples_per_trajectory must be >= 1")
    cfg["observables_ensemble.save_every"] >= 1 || error("observables_ensemble.save_every must be >= 1")
    cfg["observables_ensemble.spinup_steps"] >= 0 || error("observables_ensemble.spinup_steps must be >= 0")

    cfg["numerical.max_abs_state"] > 0 || error("numerical.max_abs_state must be > 0")
    0.0 <= cfg["numerical.min_valid_fraction"] <= 1.0 || error("numerical.min_valid_fraction must be in [0,1]")
    cfg["numerical.max_h_shrinks"] >= 0 || error("numerical.max_h_shrinks must be >= 0")
    cfg["numerical.thread_chunk_size"] >= 1 || error("numerical.thread_chunk_size must be >= 1")
    0.0 < cfg["numerical.h_shrink_factor"] < 1.0 || error("numerical.h_shrink_factor must be in (0,1)")

    cfg["performance.max_parallel_line_search"] >= 1 || error("performance.max_parallel_line_search must be >= 1")

    cfg["runtime.A_target"] = nothing
    cfg["runtime.current_observable_series"] = nothing
    cfg["runtime.current_method"] = ""
    cfg["runtime.primary_observables"] = nothing
    cfg["runtime.iteration_diagnostics"] = Dict{String,Any}()
    cfg["runtime.truth_matrix"] = nothing
    cfg["runtime.current_gfdt_path"] = ""
    cfg["runtime.iteration"] = 0
    cfg["runtime.method_parallel_active"] = false

    return cfg
end

function maybe_save_truth_artifacts!(cfg::Dict{String,Any}, X::Matrix{Float64}, A_target::Vector{Float64})
    if !haskey(cfg, "runtime.truth_dir")
        return nothing
    end
    truth_dir = String(cfg["runtime.truth_dir"])
    isempty(strip(truth_dir)) && return nothing

    mkpath(truth_dir)
    target_csv = joinpath(truth_dir, "target_observables.csv")
    write_vector_csv(target_csv, OBS_NAMES, A_target)

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

    X = if cfg["truth.source"] == "two_scale"
        # Reuse cached Arnold dataset when signatures already match; this avoids
        # regenerating the expensive two-scale trajectory on every calibration run.
        data_cfg, _ = ArnoldCommon.load_data_config(cfg["paths.data_params"])
        ArnoldCommon.ensure_arnold_dataset_role!(data_cfg, "two_scale_observed")
        ArnoldCommon.load_role_x_matrix(
            data_cfg,
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
    )
    A_target = vec(mean(A; dims=2))
    all(isfinite, A_target) || error("Target observables contain non-finite values")

    cfg["runtime.truth_matrix"] = X
    cfg["runtime.A_target"] = A_target
    maybe_save_truth_artifacts!(cfg, X, A_target)

    return A_target
end

function generate_iteration_datasets(cfg::Dict, theta::NTuple{5,Float64}, iteration::Int, run_dir::String)
    iter_data_dir = joinpath(run_dir, @sprintf("iter_%03d", iteration), "data")
    mkpath(iter_data_dir)

    a0, a1, a2, a3, sig = theta

    train_seed = cfg["datasets.train_rng_seed_base"] + iteration
    gfdt_seed = cfg["datasets.gfdt_rng_seed_base"] + iteration

    train_total = Int(cfg["datasets.train_nsamples"])
    train_ntraj = Int(cfg["datasets.train_ensemble_trajectories"])
    train_parallel = Bool(cfg["datasets.train_parallel_trajectories"]) && train_ntraj > 1 && Base.Threads.nthreads() > 1

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
    )

    train_path = joinpath(iter_data_dir, "train_stochastic.hdf5")
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

    attrs_train = copy(common_attrs)
    attrs_train["save_every"] = cfg["datasets.train_save_every"]
    attrs_train["nsamples"] = cfg["datasets.train_nsamples"]
    attrs_train["spinup_steps"] = cfg["datasets.spinup_steps"]
    attrs_train["rng_seed"] = train_seed
    attrs_train["ensemble_trajectories"] = train_ntraj
    attrs_train["parallel_trajectories"] = train_parallel
    attrs_train["role"] = "train_stochastic"

    attrs_gfdt = copy(common_attrs)
    attrs_gfdt["save_every"] = cfg["datasets.gfdt_save_every"]
    attrs_gfdt["nsamples"] = cfg["datasets.gfdt_nsamples"]
    attrs_gfdt["spinup_steps"] = cfg["datasets.spinup_steps"]
    attrs_gfdt["rng_seed"] = gfdt_seed
    attrs_gfdt["role"] = "gfdt_stochastic"

    ArnoldCommon.save_x_dataset(train_path, cfg["datasets.train_key"], train_data, attrs_train)
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

    length(h_abs) == 5 || error("responses.finite_difference.h_abs must have length 5")
    n_samples >= 1 || error("responses.finite_difference.nsamples must be >= 1")
    spinup >= 0 || error("responses.finite_difference.spinup must be >= 0")
    h_rel > 0 || error("responses.finite_difference.h_rel must be > 0")

    J = zeros(Float64, 5, 5)
    iter_id = Int(get(cfg, "runtime.iteration", 0))
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
                if use_fd_ensemble
                    force_serial_inner = parallel_fd_columns && !allow_nested_parallel
                    avg_p = estimate_observables_ensemble(
                        Float64[theta_p[1], theta_p[2], theta_p[3], theta_p[4], theta_p[5]],
                        cfg;
                        trajectories=fd_trajectories,
                        samples_per_trajectory=fd_samples_per_traj,
                        save_every=fd_save_every,
                        spinup_steps=spinup,
                        seed_base=seed,
                        parallel_trajectories=fd_parallel_traj,
                        seed_offset=11,
                        force_serial=force_serial_inner,
                    )
                    avg_m = estimate_observables_ensemble(
                        Float64[theta_m[1], theta_m[2], theta_m[3], theta_m[4], theta_m[5]],
                        cfg;
                        trajectories=fd_trajectories,
                        samples_per_trajectory=fd_samples_per_traj,
                        save_every=fd_save_every,
                        spinup_steps=spinup,
                        seed_base=seed,
                        parallel_trajectories=fd_parallel_traj,
                        seed_offset=29,
                        force_serial=force_serial_inner,
                    )
                else
                    avg_p = compute_steady_state_observables(theta_p, cfg, n_samples, spinup, seed)
                    avg_m = compute_steady_state_observables(theta_m, cfg, n_samples, spinup, seed)
                end
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

    if parallel_fd_columns
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

    all(isfinite, J) || error("Asymptotic finite-difference Jacobian contains non-finite values")
    return J
end

function compute_iteration_jacobians(cfg, theta, gfdt_data_path, checkpoint_path, iteration, run_dir)
    X = load_x_matrix(gfdt_data_path, cfg["datasets.gfdt_key"], cfg["integration.K"])
    obs_ref = obs_ref_tuple(cfg)

    A = ArnoldCommon.compute_observables_series(
        X,
        obs_ref.F_ref,
        obs_ref.alpha0_ref,
        obs_ref.alpha1_ref,
        obs_ref.alpha2_ref,
        obs_ref.alpha3_ref,
    )
    G_obs = vec(mean(A; dims=2))
    all(isfinite, G_obs) || error("Observable averages contain non-finite values")

    cfg["runtime.current_observable_series"] = A
    cfg["runtime.current_gfdt_path"] = gfdt_data_path

    n_lags, dt_obs = response_n_lags(cfg, size(X, 2))

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
        C, R_step, times = ArnoldCommon.build_gfdt_response(A, G_conj, dt_obs, n_lags; mean_center=cfg["responses.mean_center"])
        S = extract_asymptotic_jacobians(times, R_step; tail_window=cfg["responses.tail_window"])
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
        C, R_step, times = ArnoldCommon.build_gfdt_response(A, G_conj, dt_obs, n_lags; mean_center=cfg["responses.mean_center"])
        S = extract_asymptotic_jacobians(times, R_step; tail_window=cfg["responses.tail_window"])
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
            R_step=zeros(Float64, 5, 5, 0),
            C=zeros(Float64, 5, 5, 0),
        )
    end

    return out
end

function diagonal_weight(A_subseries::Matrix{Float64})
    vars = vec(var(A_subseries; dims=2, corrected=true))
    vars = max.(vars, 1e-10)
    return Diagonal(1.0 ./ vars)
end

function inverse_cov_weight(A_subseries::Matrix{Float64})
    A_of_x = x -> Float64.(collect(x))
    return weight_inverse_cov_bridge(A_of_x, A_subseries)
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

function estimate_observables(theta_candidate::Vector{Float64}, cfg::Dict{String,Any})
    compute_steady_state_observables = require_main_symbol(:compute_steady_state_observables)
    # Use next-iteration GFDT seed to align line-search acceptance with the
    # actual dataset-generation stability constraints.
    seed = cfg["datasets.gfdt_rng_seed_base"] + Int(get(cfg, "runtime.iteration", 0)) + 1
    obs = compute_steady_state_observables(
        theta_to_tuple(theta_candidate),
        cfg,
        cfg["calibration.line_search_samples"],
        cfg["calibration.line_search_spinup"],
        seed,
    )
    return vec(Float64.(obs))
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

    iter_id = Int(get(cfg, "runtime.iteration", 0))
    obs_mat = Matrix{Float64}(undef, 5, trajectories)
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
                F=cfg["integration.F"],
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

    gamma = cfg["calibration.regularization_gamma"]
    nfree = length(free_idx)
    Gamma_sub = Symmetric(Matrix{Float64}(I, nfree, nfree) * gamma)

    correction_sub, raw_diag = newton_step_bridge(S_sub, W_sub, Gamma_sub, G_sub, A_sub)

    correction_full = zeros(Float64, 5)
    correction_full[free_idx] .= correction_sub

    theta_current = Float64.(theta)
    damping = cfg["calibration.damping"]

    residual_before = norm(G_sub .- A_sub)
    theta_best = theta_current .- damping .* correction_full
    applied_step = damping .* correction_full
    residual_after = residual_before
    line_search_scale = damping
    line_search_used = false
    line_search_accepted = false
    stability_checked = false
    stability_accepted = true

    if cfg["calibration.line_search"]
        line_search_used = true
        line_search_max = cfg["calibration.line_search_max"]
        best_residual = Inf
        best_theta = copy(theta_best)
        best_step = copy(applied_step)
        best_scale = damping

        nested_parallel_active = Bool(get(cfg, "runtime.method_parallel_active", false))
        allow_nested_parallel = Bool(get(cfg, "performance.allow_nested_parallel", true))
        do_parallel_line_search = Bool(get(cfg, "performance.parallel_line_search", true))
        do_parallel_line_search = do_parallel_line_search && (allow_nested_parallel || !nested_parallel_active)

        if do_parallel_line_search && line_search_max > 0
            scales = [damping / (2.0 ^ h) for h in 0:line_search_max]
            max_parallel = min(length(scales), max(1, Int(get(cfg, "performance.max_parallel_line_search", 3))), Base.Threads.nthreads())
            sem = Base.Semaphore(max_parallel)
            results = Vector{Any}(undef, length(scales))

            @sync for idx in eachindex(scales)
                scale = scales[idx]
                Base.Threads.@spawn begin
                    Base.acquire(sem)
                    try
                        theta_try = theta_current .- scale .* correction_full
                        G_try = estimate_observables(theta_try, cfg)
                        res_try = norm(G_try[active_idx] .- A_sub)
                        need_stability = res_try < residual_before
                        stable_try = true
                        if need_stability
                            stable_try = theta_candidate_is_stable(theta_try, cfg)
                        end
                        results[idx] = (
                            scale=scale,
                            theta_try=theta_try,
                            res_try=res_try,
                            need_stability=need_stability,
                            stable_try=stable_try,
                            err="",
                        )
                    catch err
                        results[idx] = (
                            scale=scale,
                            theta_try=copy(theta_current),
                            res_try=Inf,
                            need_stability=false,
                            stable_try=false,
                            err=sprint(showerror, err),
                        )
                    finally
                        Base.release(sem)
                    end
                end
            end

            for idx in eachindex(scales)
                r = results[idx]
                if !isempty(r.err)
                    @warn "Line search evaluation failed for candidate theta; trying smaller step" scale = r.scale error = r.err
                    continue
                end
                if r.need_stability
                    stability_checked = true
                end
                if r.need_stability && !r.stable_try
                    @warn "Rejecting Newton candidate: unstable stochastic rollout" scale = r.scale res_try = r.res_try
                    continue
                end
                if r.res_try < best_residual
                    best_residual = r.res_try
                    best_theta .= r.theta_try
                    best_step .= r.scale .* correction_full
                    best_scale = r.scale
                end
                if r.res_try < residual_before
                    line_search_accepted = true
                    stability_accepted = true
                    theta_best .= r.theta_try
                    applied_step .= r.scale .* correction_full
                    residual_after = r.res_try
                    line_search_scale = r.scale
                    break
                end
            end
        else
            for halvings in 0:line_search_max
                scale = damping / (2.0 ^ halvings)
                theta_try = theta_current .- scale .* correction_full
                try
                    G_try = estimate_observables(theta_try, cfg)
                    res_try = norm(G_try[active_idx] .- A_sub)
                    need_stability = res_try < best_residual || res_try < residual_before
                    stable_try = true
                    if need_stability
                        stability_checked = true
                        stable_try = theta_candidate_is_stable(theta_try, cfg)
                    end
                    if !stable_try
                        @warn "Rejecting Newton candidate: unstable stochastic rollout" scale res_try
                        continue
                    end
                    if res_try < best_residual
                        best_residual = res_try
                        best_theta .= theta_try
                        best_step .= scale .* correction_full
                        best_scale = scale
                    end
                    if res_try < residual_before
                        line_search_accepted = true
                        stability_accepted = true
                        theta_best .= theta_try
                        applied_step .= scale .* correction_full
                        residual_after = res_try
                        line_search_scale = scale
                        break
                    end
                catch err
                    @warn "Line search evaluation failed for candidate theta; trying smaller step" scale error = sprint(showerror, err)
                end
            end
        end

        if !line_search_accepted
            if isfinite(best_residual) && best_residual < residual_before
                theta_best .= best_theta
                applied_step .= best_step
                residual_after = best_residual
                line_search_scale = best_scale
                stability_accepted = true
            else
                # Reject the update if no candidate improved residual.
                theta_best .= theta_current
                applied_step .= 0.0
                residual_after = residual_before
                line_search_scale = 0.0
                stability_accepted = false
            end
        end
    end

    diagnostics = Dict{String,Any}(
        "raw_cond" => get(raw_diag, :cond, NaN),
        "raw_rhs_norm" => get(raw_diag, :nrm_rhs, NaN),
        "residual_before" => residual_before,
        "residual_after" => residual_after,
        "line_search_used" => line_search_used,
        "line_search_accepted" => line_search_accepted,
        "line_search_scale" => line_search_scale,
        "stability_checked" => stability_checked,
        "stability_accepted" => stability_accepted,
        "free_parameters" => free_idx,
        "active_observables" => active_idx,
        "weight_matrix_mode" => cfg["calibration.weight_matrix"],
        "correction_full" => copy(correction_full),
        "applied_step" => copy(applied_step),
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
        obs_residual = norm(G_primary[active_idx] .- A_target[active_idx])
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

function save_response_iteration_figure(path::AbstractString, cfg::Dict, jacobians)
    save_response_figure = require_main_symbol(:save_response_figure)

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
            data=jr.R_step,
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
            jacobians=jacobians[method].S,
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
            jacobians=jacobians["finite_difference"].S,
        ))
    end

    return save_response_figure(
        path,
        times,
        curves;
        asymptotic_curves=asymptotic_curves,
        title_text="Calibration iteration responses",
        dpi=cfg["figures.dpi"],
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

    kl_mode, js_mode = sp.modewise_metrics(obs_tensor, gen_tensor; nbins=80, low_q=0.001, high_q=0.999)
    return sp.save_stats_figure_acf(
        path,
        obs_tensor,
        gen_tensor,
        kl_mode,
        js_mode,
        80;
        max_lag=200,
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
    h5open(jac_path, "w") do h5
        for method in sort(collect(keys(jacobians)))
            h5[joinpath("jacobians", method)] = jacobians[method].S
        end
    end

    obs_path = joinpath(results_dir, "observables.csv")
    A_target = get(cfg, "runtime.A_target", nothing)
    open(obs_path, "w") do io
        println(io, "method,observable,value")
        for method in sort(collect(keys(observables)))
            vals = observables[method]
            for i in 1:min(length(vals), length(OBS_NAMES))
                @printf(io, "%s,%s,%.16e\n", method, OBS_NAMES[i], vals[i])
            end
        end
        if A_target isa Vector{Float64}
            for i in 1:min(length(A_target), length(OBS_NAMES))
                @printf(io, "target,%s,%.16e\n", OBS_NAMES[i], A_target[i])
            end
        end
    end

    diag_path = joinpath(results_dir, "diagnostics.toml")
    diag = get(cfg, "runtime.iteration_diagnostics", Dict{String,Any}())
    open(diag_path, "w") do io
        TOML.print(io, diag)
    end

    if cfg["figures.save_per_iteration"]
        response_fig = joinpath(figures_dir, "responses_5x5.png")
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
            pn = plot(; title="$(OBS_NAMES[oidx])", xlabel="iteration", ylabel="observable", legend=(col == 1 ? :best : false))
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
