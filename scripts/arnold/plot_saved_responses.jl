#!/usr/bin/env julia
# Compute response functions from scratch for a saved calibration iteration.
#
# Example:
#   julia --threads auto --project=. scripts/arnold/plot_saved_responses.jl \
#       --params scripts/arnold/parameters_saved_responses.toml

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using HDF5
using Printf
using TOML

include(joinpath(@__DIR__, "run_calibration.jl"))

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_saved_responses.toml")

    i = 1
    while i <= length(args)
        a = strip(args[i])
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = strip(args[i + 1])
            i += 2
        elseif startswith(a, "--params=")
            params_path = strip(split(a, "=", limit=2)[2])
            i += 1
        else
            error("Unknown argument: $a")
        end
    end

    return (params_path=abspath(params_path),)
end

function maybe_table(doc::Dict{String,Any}, key::String)
    if !haskey(doc, key)
        return Dict{String,Any}()
    end
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_bool(tbl::Dict{String,Any}, key::String, default) = Bool(get(tbl, key, default))
as_str_vec(tbl::Dict{String,Any}, key::String, default) = String.(collect(get(tbl, key, default)))
as_float_vec(tbl::Dict{String,Any}, key::String, default) = Float64.(collect(get(tbl, key, default)))

function load_params(path::String)
    isfile(path) || error("Parameters file not found: $path")
    doc = TOML.parsefile(path)

    paths = maybe_table(doc, "paths")
    run_tbl = maybe_table(doc, "run")
    methods = maybe_table(doc, "methods")
    datasets = maybe_table(doc, "datasets")
    responses = maybe_table(doc, "responses")
    numerical = maybe_table(doc, "numerical")
    observables = maybe_table(doc, "observables")
    figures = maybe_table(doc, "figures")

    target_dir = strip(as_str(paths, "target_dir", ""))
    runs_root = abspath(as_str(paths, "runs_root", "scripts/arnold/runs_calibration"))
    calibration_config = strip(as_str(paths, "calibration_config", ""))
    output_subdir = as_str(paths, "output_subdir", "saved_response_analysis")
    output_name = as_str(paths, "output_name", "responses_recomputed")

    iteration = as_int(run_tbl, "iteration", 0)
    auto_latest_iteration = as_bool(run_tbl, "auto_latest_iteration", true)

    response_methods = lowercase.(as_str_vec(methods, "response_methods", ["numerical_integration", "gaussian", "unet", "finite_difference"]))
    response_kind = lowercase(as_str(responses, "response_kind", "heaviside"))
    response_kind in ("heaviside", "impulse") || error("responses.response_kind must be heaviside or impulse")

    return (
        raw_doc=doc,
        target_dir=target_dir,
        runs_root=runs_root,
        calibration_config=isempty(calibration_config) ? "" : abspath(calibration_config),
        output_subdir=output_subdir,
        output_name=output_name,
        iteration=iteration,
        auto_latest_iteration=auto_latest_iteration,
        methods=Dict(
            "unet" => as_bool(methods, "unet", true),
            "gaussian" => as_bool(methods, "gaussian", true),
            "finite_difference" => as_bool(methods, "finite_difference", true),
            "numerical_integration" => as_bool(methods, "numerical_integration", true),
            "response_methods" => response_methods,
        ),
        datasets=Dict(
            "gfdt_nsamples" => haskey(datasets, "gfdt_nsamples") ? as_int(datasets, "gfdt_nsamples", 250_000) : nothing,
            "gfdt_save_every" => haskey(datasets, "gfdt_save_every") ? as_int(datasets, "gfdt_save_every", 2) : nothing,
            "gfdt_rng_seed_base" => haskey(datasets, "gfdt_rng_seed_base") ? as_int(datasets, "gfdt_rng_seed_base", 3_000) : nothing,
            "spinup_steps" => haskey(datasets, "spinup_steps") ? as_int(datasets, "spinup_steps", 50_000) : nothing,
            "gfdt_ensemble_trajectories" => haskey(datasets, "gfdt_ensemble_trajectories") ? as_int(datasets, "gfdt_ensemble_trajectories", 1) : nothing,
            "gfdt_parallel_trajectories" => haskey(datasets, "gfdt_parallel_trajectories") ? as_bool(datasets, "gfdt_parallel_trajectories", true) : nothing,
            "max_abs_state" => haskey(datasets, "max_abs_state") ? as_float(datasets, "max_abs_state", 1.0e4) : nothing,
            "max_restarts" => haskey(datasets, "max_restarts") ? as_int(datasets, "max_restarts", 40) : nothing,
            "state_min" => haskey(datasets, "state_min") ? as_float(datasets, "state_min", -1.0e300) : nothing,
            "state_max" => haskey(datasets, "state_max") ? as_float(datasets, "state_max", 1.0e300) : nothing,
            "max_boundary_hits" => haskey(datasets, "max_boundary_hits") ? as_int(datasets, "max_boundary_hits", 40) : nothing,
        ),
        responses=Dict(
            "response_kind" => response_kind,
            "response_tmax" => haskey(responses, "response_tmax") ? as_float(responses, "response_tmax", 5.0) : nothing,
            "t_start" => haskey(responses, "t_start") ? as_float(responses, "t_start", 2.0) : nothing,
            "t_end" => haskey(responses, "t_end") ? as_float(responses, "t_end", 5.0) : nothing,
            "mean_center" => get(responses, "mean_center", nothing),
            "impulse_tail_debias" => get(responses, "impulse_tail_debias", nothing),
            "impulse_tail_taper" => haskey(responses, "impulse_tail_taper") ? lowercase(as_str(responses, "impulse_tail_taper", "hard")) : nothing,
            "apply_score_correction" => get(responses, "apply_score_correction", nothing),
            "divergence_mode" => haskey(responses, "divergence_mode") ? lowercase(as_str(responses, "divergence_mode", "hutchinson")) : nothing,
            "divergence_eps" => haskey(responses, "divergence_eps") ? as_float(responses, "divergence_eps", 0.03) : nothing,
            "divergence_probes" => haskey(responses, "divergence_probes") ? as_int(responses, "divergence_probes", 10) : nothing,
            "score_device" => haskey(responses, "score_device") ? as_str(responses, "score_device", "CPU") : nothing,
            "score_forward_mode" => haskey(responses, "score_forward_mode") ? lowercase(as_str(responses, "score_forward_mode", "test")) : nothing,
            "batch_size" => haskey(responses, "batch_size") ? as_int(responses, "batch_size", 1024) : nothing,
        ),
        numerical=Dict(
            "nsamples" => haskey(numerical, "nsamples") ? as_int(numerical, "nsamples", 120_000) : nothing,
            "spinup" => haskey(numerical, "spinup") ? as_int(numerical, "spinup", 2_000) : nothing,
            "h_abs" => haskey(numerical, "h_abs") ? as_float_vec(numerical, "h_abs", [0.05, 0.02, 0.01, 0.002, 0.05]) : nothing,
            "h_rel" => haskey(numerical, "h_rel") ? as_float(numerical, "h_rel", 0.005) : nothing,
            "seed_base" => haskey(numerical, "seed_base") ? as_int(numerical, "seed_base", 50_000_000) : nothing,
            "use_ensemble" => haskey(numerical, "use_ensemble") ? as_bool(numerical, "use_ensemble", true) : nothing,
            "ensemble_trajectories" => haskey(numerical, "ensemble_trajectories") ? as_int(numerical, "ensemble_trajectories", 36) : nothing,
            "init_ensembles" => haskey(numerical, "init_ensembles") ? as_int(numerical, "init_ensembles", 36) : nothing,
            "samples_per_trajectory" => haskey(numerical, "samples_per_trajectory") ? as_int(numerical, "samples_per_trajectory", 6_000) : nothing,
            "save_every" => haskey(numerical, "save_every") ? as_int(numerical, "save_every", 200) : nothing,
            "parallel_trajectories" => haskey(numerical, "parallel_trajectories") ? as_bool(numerical, "parallel_trajectories", true) : nothing,
            "max_abs_state" => haskey(numerical, "max_abs_state") ? as_float(numerical, "max_abs_state", 80.0) : nothing,
            "state_min" => haskey(numerical, "state_min") ? as_float(numerical, "state_min", -Inf) : nothing,
            "state_max" => haskey(numerical, "state_max") ? as_float(numerical, "state_max", Inf) : nothing,
            "max_boundary_hits" => haskey(numerical, "max_boundary_hits") ? as_int(numerical, "max_boundary_hits", 40) : nothing,
            "min_valid_fraction" => haskey(numerical, "min_valid_fraction") ? as_float(numerical, "min_valid_fraction", 0.8) : nothing,
            "max_h_shrinks" => haskey(numerical, "max_h_shrinks") ? as_int(numerical, "max_h_shrinks", 6) : nothing,
            "thread_chunk_size" => haskey(numerical, "thread_chunk_size") ? as_int(numerical, "thread_chunk_size", 256) : nothing,
            "h_shrink_factor" => haskey(numerical, "h_shrink_factor") ? as_float(numerical, "h_shrink_factor", 0.5) : nothing,
        ),
        observables=Dict(
            "active_names" => lowercase.(as_str_vec(observables, "active_names", String[])),
        ),
        figures=Dict(
            "dpi" => haskey(figures, "dpi") ? as_int(figures, "dpi", 180) : nothing,
        ),
    )
end

function latest_run_dir(root::AbstractString)
    isdir(root) || error("Calibration runs root does not exist: $root")
    runs = filter(name -> startswith(name, "run_") && isdir(joinpath(root, name)), readdir(root))
    isempty(runs) && error("No calibration run directories found under $root")
    sort!(runs)
    return joinpath(root, runs[end])
end

function find_run_dir_from_path(path::String)
    cur = abspath(path)
    for _ in 1:12
        cfg_path = joinpath(cur, "config", "parameters_calibration.toml")
        if isfile(cfg_path)
            return cur
        end
        parent = dirname(cur)
        parent == cur && break
        cur = parent
    end
    error("Could not locate calibration run root from: $path")
end

function iter_number(name::AbstractString)
    startswith(name, "iter_") || return typemax(Int)
    n = tryparse(Int, split(String(name), "_")[end])
    n === nothing && return typemax(Int)
    return n
end

function latest_iteration_name(method_root::AbstractString)
    isdir(method_root) || error("method_unet directory not found: $method_root")
    iter_names = [name for name in readdir(method_root) if startswith(name, "iter_") && isdir(joinpath(method_root, name))]
    isempty(iter_names) && error("No iter_* directories found under $method_root")
    sort!(iter_names; by=iter_number)
    return iter_names[end]
end

function resolve_iteration_target(parsed)
    target_dir = isempty(parsed.target_dir) ? latest_run_dir(parsed.runs_root) : abspath(parsed.target_dir)
    isdir(target_dir) || error("Target directory not found: $target_dir")

    base = basename(target_dir)
    if startswith(base, "iter_")
        iter_name = base
        if basename(dirname(target_dir)) == "method_unet"
            run_dir = find_run_dir_from_path(target_dir)
            method_iter_dir = target_dir
            return run_dir, iter_name, method_iter_dir
        end
        run_dir = find_run_dir_from_path(target_dir)
        method_iter_dir = joinpath(run_dir, "method_unet", iter_name)
        isdir(method_iter_dir) || error("method_unet iteration directory not found: $method_iter_dir")
        return run_dir, iter_name, method_iter_dir
    end

    run_dir = find_run_dir_from_path(target_dir)
    method_root = base == "method_unet" ? target_dir : joinpath(run_dir, "method_unet")
    isdir(method_root) || error("method_unet directory not found: $method_root")

    iter_name = if parsed.iteration > 0
        @sprintf("iter_%03d", parsed.iteration)
    elseif parsed.auto_latest_iteration
        latest_iteration_name(method_root)
    else
        error("Set [run].iteration > 0 or enable auto_latest_iteration")
    end

    method_iter_dir = joinpath(method_root, iter_name)
    isdir(method_iter_dir) || error("Iteration directory not found: $method_iter_dir")
    return run_dir, iter_name, method_iter_dir
end

function read_theta_from_dataset(path::AbstractString, key::AbstractString)
    isfile(path) || error("Dataset file not found: $path")
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        attrs = HDF5.attributes(h5[key])
        return Float64[
            read(attrs["alpha0"]),
            read(attrs["alpha1"]),
            read(attrs["alpha2"]),
            read(attrs["alpha3"]),
            read(attrs["sigma"]),
        ]
    end
end

function resolve_theta_source_path(run_dir::AbstractString, iter_name::AbstractString, method_iter_dir::AbstractString)
    direct_candidates = String[
        joinpath(method_iter_dir, "data", "gfdt_stochastic.hdf5"),
        joinpath(run_dir, iter_name, "data", "gfdt_stochastic.hdf5"),
    ]
    for path in direct_candidates
        isfile(path) && return path
    end

    shared_root = joinpath(run_dir, iter_name, "shared")
    if isdir(shared_root)
        matches = String[]
        for (root, _, files) in walkdir(shared_root)
            "gfdt_stochastic.hdf5" in files || continue
            push!(matches, joinpath(root, "gfdt_stochastic.hdf5"))
        end
        if !isempty(matches)
            sort!(matches)
            return matches[1]
        end
    end

    searched = vcat(direct_candidates, [joinpath(shared_root, "**", "gfdt_stochastic.hdf5")])
    error("Missing saved iteration dataset for theta recovery. Searched: $(join(searched, ", "))")
end

function apply_optional_override!(cfg::Dict{String,Any}, key::String, value)
    value === nothing && return
    cfg[key] = value
    return nothing
end

function apply_overrides!(cfg::Dict{String,Any}, parsed)
    cfg["methods.unet"] = parsed.methods["unet"]
    cfg["methods.gaussian"] = parsed.methods["gaussian"]
    cfg["methods.finite_difference"] = parsed.methods["finite_difference"]
    cfg["methods.numerical_integration"] = parsed.methods["numerical_integration"]
    cfg["figures.response_methods"] = copy(parsed.methods["response_methods"])

    apply_optional_override!(cfg, "datasets.gfdt_nsamples", parsed.datasets["gfdt_nsamples"])
    apply_optional_override!(cfg, "datasets.gfdt_save_every", parsed.datasets["gfdt_save_every"])
    apply_optional_override!(cfg, "datasets.gfdt_rng_seed_base", parsed.datasets["gfdt_rng_seed_base"])
    apply_optional_override!(cfg, "datasets.spinup_steps", parsed.datasets["spinup_steps"])
    apply_optional_override!(cfg, "saved_responses.gfdt_ensemble_trajectories", parsed.datasets["gfdt_ensemble_trajectories"])
    apply_optional_override!(cfg, "saved_responses.gfdt_parallel_trajectories", parsed.datasets["gfdt_parallel_trajectories"])
    apply_optional_override!(cfg, "saved_responses.dataset_max_abs_state", parsed.datasets["max_abs_state"])
    apply_optional_override!(cfg, "saved_responses.dataset_max_restarts", parsed.datasets["max_restarts"])
    apply_optional_override!(cfg, "saved_responses.dataset_state_min", parsed.datasets["state_min"])
    apply_optional_override!(cfg, "saved_responses.dataset_state_max", parsed.datasets["state_max"])
    apply_optional_override!(cfg, "saved_responses.dataset_max_boundary_hits", parsed.datasets["max_boundary_hits"])
    cfg["integration.save_every"] = cfg["datasets.gfdt_save_every"]

    apply_optional_override!(cfg, "responses.response_tmax", parsed.responses["response_tmax"])
    apply_optional_override!(cfg, "responses.t_start", parsed.responses["t_start"])
    apply_optional_override!(cfg, "responses.t_end", parsed.responses["t_end"])
    apply_optional_override!(cfg, "responses.mean_center", parsed.responses["mean_center"])
    apply_optional_override!(cfg, "responses.impulse_tail_debias", parsed.responses["impulse_tail_debias"])
    apply_optional_override!(cfg, "responses.impulse_tail_taper", parsed.responses["impulse_tail_taper"])
    apply_optional_override!(cfg, "responses.apply_score_correction", parsed.responses["apply_score_correction"])
    apply_optional_override!(cfg, "responses.divergence_mode", parsed.responses["divergence_mode"])
    apply_optional_override!(cfg, "responses.divergence_eps", parsed.responses["divergence_eps"])
    apply_optional_override!(cfg, "responses.divergence_probes", parsed.responses["divergence_probes"])
    apply_optional_override!(cfg, "responses.score_device", parsed.responses["score_device"])
    apply_optional_override!(cfg, "responses.score_forward_mode", parsed.responses["score_forward_mode"])
    apply_optional_override!(cfg, "responses.batch_size", parsed.responses["batch_size"])

    apply_optional_override!(cfg, "responses.finite_difference.nsamples", parsed.numerical["nsamples"])
    apply_optional_override!(cfg, "responses.finite_difference.spinup", parsed.numerical["spinup"])
    apply_optional_override!(cfg, "responses.finite_difference.h_abs", parsed.numerical["h_abs"])
    apply_optional_override!(cfg, "responses.finite_difference.h_rel", parsed.numerical["h_rel"])
    apply_optional_override!(cfg, "responses.finite_difference.seed_base", parsed.numerical["seed_base"])
    apply_optional_override!(cfg, "responses.finite_difference.use_ensemble", parsed.numerical["use_ensemble"])
    apply_optional_override!(cfg, "responses.finite_difference.ensemble_trajectories", parsed.numerical["ensemble_trajectories"])
    apply_optional_override!(cfg, "responses.finite_difference.samples_per_trajectory", parsed.numerical["samples_per_trajectory"])
    apply_optional_override!(cfg, "responses.finite_difference.save_every", parsed.numerical["save_every"])
    apply_optional_override!(cfg, "responses.finite_difference.parallel_trajectories", parsed.numerical["parallel_trajectories"])

    apply_optional_override!(cfg, "figures.dpi", parsed.figures["dpi"])

    cfg["numerical.h_abs"] = copy(cfg["responses.finite_difference.h_abs"])
    cfg["numerical.h_rel"] = cfg["responses.finite_difference.h_rel"]
    cfg["numerical.seed_base"] = cfg["responses.finite_difference.seed_base"]
    apply_optional_override!(cfg, "saved_responses.numerical_init_ensembles", parsed.numerical["init_ensembles"])
    apply_optional_override!(cfg, "numerical.max_abs_state", parsed.numerical["max_abs_state"])
    apply_optional_override!(cfg, "numerical.state_min", parsed.numerical["state_min"])
    apply_optional_override!(cfg, "numerical.state_max", parsed.numerical["state_max"])
    apply_optional_override!(cfg, "numerical.max_boundary_hits", parsed.numerical["max_boundary_hits"])
    apply_optional_override!(cfg, "numerical.min_valid_fraction", parsed.numerical["min_valid_fraction"])
    apply_optional_override!(cfg, "numerical.max_h_shrinks", parsed.numerical["max_h_shrinks"])
    apply_optional_override!(cfg, "numerical.thread_chunk_size", parsed.numerical["thread_chunk_size"])
    apply_optional_override!(cfg, "numerical.h_shrink_factor", parsed.numerical["h_shrink_factor"])

    active_names = parsed.observables["active_names"]
    isempty(active_names) && error("observables.active_names must be set in parameters_saved_responses.toml for this standalone script")
    all_names = CalibrationCommon.observable_names(cfg)
    name_to_idx = Dict(lowercase(name) => i for (i, name) in enumerate(all_names))
    available_names_str = join(all_names, ", ")
    active_idx = Int[]
    for name in active_names
        haskey(name_to_idx, name) || error("Unknown observable name '$name'. Available observables: $available_names_str")
        push!(active_idx, name_to_idx[name])
    end
    cfg["saved_responses.active_observables"] = active_idx

    return cfg
end

function generate_gfdt_dataset(cfg::Dict{String,Any}, theta::NTuple{5,Float64}, iteration::Int, out_dir::String)
    data = ArnoldCommon.generate_reduced_x_timeseries_ensemble(
        K=cfg["integration.K"],
        F=cfg["closure.F"],
        alpha0=theta[1],
        alpha1=theta[2],
        alpha2=theta[3],
        alpha3=theta[4],
        sigma=theta[5],
        dt=cfg["integration.dt"],
        spinup_steps=Int(cfg["datasets.spinup_steps"]),
        save_every=Int(cfg["datasets.gfdt_save_every"]),
        nsamples=Int(cfg["datasets.gfdt_nsamples"]),
        rng_seed=Int(cfg["datasets.gfdt_rng_seed_base"]) + iteration,
        max_abs_state=Float64(get(cfg, "saved_responses.dataset_max_abs_state", 1.0e4)),
        max_restarts=Int(get(cfg, "saved_responses.dataset_max_restarts", 40)),
        state_min=Float64(get(cfg, "saved_responses.dataset_state_min", -1.0e300)),
        state_max=Float64(get(cfg, "saved_responses.dataset_state_max", 1.0e300)),
        max_boundary_hits=Int(get(cfg, "saved_responses.dataset_max_boundary_hits", 40)),
        ensemble_trajectories=Int(get(cfg, "saved_responses.gfdt_ensemble_trajectories", 1)),
        parallel_trajectories=Bool(get(cfg, "saved_responses.gfdt_parallel_trajectories", true)),
    )

    gfdt_path = joinpath(out_dir, "generated_gfdt_stochastic.hdf5")
    attrs = Dict{String,Any}(
        "generated_at" => string(now()),
        "role" => "gfdt_stochastic",
        "K" => cfg["integration.K"],
        "dt" => cfg["integration.dt"],
        "F" => cfg["closure.F"],
        "alpha0" => theta[1],
        "alpha1" => theta[2],
        "alpha2" => theta[3],
        "alpha3" => theta[4],
        "sigma" => theta[5],
        "save_every" => cfg["datasets.gfdt_save_every"],
        "nsamples" => cfg["datasets.gfdt_nsamples"],
        "spinup_steps" => cfg["datasets.spinup_steps"],
        "rng_seed" => Int(cfg["datasets.gfdt_rng_seed_base"]) + iteration,
        "ensemble_trajectories" => Int(get(cfg, "saved_responses.gfdt_ensemble_trajectories", 1)),
    )
    ArnoldCommon.save_x_dataset(gfdt_path, cfg["datasets.gfdt_key"], data, attrs)

    return Float64.(permutedims(data, (2, 1))), gfdt_path
end

function style_label(method::String)
    style = CalibrationCommon.style_for_method(method)
    return (label=style.label, color=style.color, linestyle=style.linestyle)
end

function active_observable_indices(cfg::Dict{String,Any})
    return Vector{Int}(get(cfg, "saved_responses.active_observables", Int[]))
end

function active_observable_labels(cfg::Dict{String,Any})
    all_labels = CalibrationCommon.observable_labels(cfg)
    idx = active_observable_indices(cfg)
    return [all_labels[i] for i in idx]
end

function build_method_curves(method_payloads, response_kind::String, active_idx::Vector{Int})
    curves = NamedTuple[]
    for method in ("numerical_integration", "gaussian", "unet")
        haskey(method_payloads, method) || continue
        payload = method_payloads[method]
        style = style_label(method)
        data_full = response_kind == "impulse" ? payload.C : payload.R_step
        data = data_full[active_idx, :, :]
        push!(curves, (
            method_key=method,
            label=style.label,
            color=style.color,
            linestyle=style.linestyle,
            data=data,
        ))
    end
    return curves
end

function build_asymptotic_curves(method_payloads, plot_methods::Vector{String}, active_idx::Vector{Int})
    curves = NamedTuple[]
    for method in plot_methods
        haskey(method_payloads, method) || continue
        style = style_label(method)
        push!(curves, (
            label=style.label,
            color=style.color,
            linestyle=style.linestyle,
            jacobians=method_payloads[method].S[active_idx, :],
        ))
    end
    return curves
end

function write_response_outputs(path::String, times::Vector{Float64}, method_payloads, response_kind::String, active_idx::Vector{Int})
    mkpath(dirname(path))
    h5open(path, "w") do h5
        h5["times"] = times
        h5["response_kind"] = response_kind
        h5["active_observables"] = active_idx
        for method in sort(collect(keys(method_payloads)))
            jr = method_payloads[method]
            h5[joinpath("responses", method, "R_step")] = jr.R_step
            h5[joinpath("responses", method, "C")] = jr.C
            h5[joinpath("jacobians", method)] = jr.S
            h5[joinpath("observables", method, "mean")] = jr.G
            h5[joinpath("selected_responses", method, "R_step")] = size(jr.R_step, 3) == 0 ? zeros(Float64, length(active_idx), size(jr.S, 2), 0) : jr.R_step[active_idx, :, :]
            h5[joinpath("selected_responses", method, "C")] = size(jr.C, 3) == 0 ? zeros(Float64, length(active_idx), size(jr.S, 2), 0) : jr.C[active_idx, :, :]
            h5[joinpath("selected_jacobians", method)] = jr.S[active_idx, :]
            h5[joinpath("selected_observables", method, "mean")] = jr.G[active_idx]
        end
    end
    return path
end

function compute_saved_run_responses(cfg::Dict{String,Any}, theta::NTuple{5,Float64}, X::Matrix{Float64}, checkpoint_path::Union{Nothing,String}, iteration::Int)
    obs_ref = CalibrationCommon.obs_ref_tuple(cfg)
    A = ArnoldCommon.compute_observables_series(
        X,
        obs_ref.F_ref,
        obs_ref.alpha0_ref,
        obs_ref.alpha1_ref,
        obs_ref.alpha2_ref,
        obs_ref.alpha3_ref,
        cfg["observables.specs"],
    )
    G_obs = vec(mean(A; dims=2))
    cfg["runtime.current_observable_series"] = A
    n_lags, dt_obs = CalibrationCommon.response_n_lags(cfg, size(X, 2))
    times = collect(0:n_lags) .* dt_obs

    payloads = Dict{String,Any}()

    if cfg["methods.gaussian"]
        gauss = gaussian_conjugates(
            X,
            theta[5];
            apply_correction=cfg["responses.apply_score_correction"],
        )
        G_conj = cfg["responses.apply_score_correction"] ? gauss.G_corr : gauss.G_raw
        C, R_step, _ = ArnoldCommon.build_gfdt_response(
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
        payloads["gaussian"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.unet"]
        checkpoint_path === nothing && error("UNet method requested but checkpoint_path is nothing")
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
        C, R_step, _ = ArnoldCommon.build_gfdt_response(
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
        payloads["unet"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.numerical_integration"]
        n_init = min(Int(get(cfg, "saved_responses.numerical_init_ensembles", cfg["responses.finite_difference.ensemble_trajectories"])), size(X, 2))
        Xinit = CalibrationCommon.select_numerical_init_states(X, n_init)
        R_step, _, _ = compute_numerical_responses(theta, Xinit, cfg, n_lags)
        C = ArnoldCommon.step_to_impulse(R_step, times)
        S = extract_asymptotic_jacobians(times, R_step; t_start=cfg["responses.t_start"], t_end=cfg["responses.t_end"])
        payloads["numerical_integration"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.finite_difference"]
        S = CalibrationCommon.compute_fd_jacobian_asymptotic(theta, cfg)
        payloads["finite_difference"] = (
            S=S,
            G=G_obs,
            A=A,
            times=times,
            R_step=zeros(Float64, size(A, 1), 5, 0),
            C=zeros(Float64, size(A, 1), 5, 0),
        )
    end

    return payloads, times
end

function build_params_used_doc(parsed, run_dir::String, iter_name::String, cfg_path::String, gfdt_path::String, checkpoint_path::String, theta::Vector{Float64})
    return Dict(
        "paths" => Dict(
            "target_dir" => run_dir,
            "calibration_config" => cfg_path,
            "output_subdir" => parsed.output_subdir,
            "output_name" => parsed.output_name,
            "gfdt_path" => gfdt_path,
            "checkpoint_path" => checkpoint_path,
        ),
        "run" => Dict(
            "iteration_name" => iter_name,
            "iteration" => iter_number(iter_name),
            "theta_from_dataset" => theta,
        ),
        "methods" => Dict(
            "unet" => parsed.methods["unet"],
            "gaussian" => parsed.methods["gaussian"],
            "finite_difference" => parsed.methods["finite_difference"],
            "numerical_integration" => parsed.methods["numerical_integration"],
            "response_methods" => parsed.methods["response_methods"],
        ),
        "datasets" => parsed.datasets,
        "responses" => parsed.responses,
        "numerical" => parsed.numerical,
        "observables" => parsed.observables,
        "figures" => parsed.figures,
    )
end

function main(args=ARGS)
    cli = parse_args(args)
    parsed = load_params(cli.params_path)

    run_dir, iter_name, method_iter_dir = resolve_iteration_target(parsed)
    cfg_path = isempty(parsed.calibration_config) ? joinpath(run_dir, "config", "parameters_calibration.toml") : parsed.calibration_config
    cfg = load_calibration_config(cfg_path)
    apply_overrides!(cfg, parsed)

    iteration = iter_number(iter_name)
    iteration == typemax(Int) && error("Could not parse iteration number from $iter_name")

    theta_source_path = resolve_theta_source_path(run_dir, iter_name, method_iter_dir)
    checkpoint_path = joinpath(method_iter_dir, "model", "final_checkpoint.bson")
    cfg["methods.unet"] && isfile(checkpoint_path) || !cfg["methods.unet"] || error("Missing UNet checkpoint: $checkpoint_path")

    theta_vec = read_theta_from_dataset(theta_source_path, cfg["datasets.gfdt_key"])
    theta = (theta_vec[1], theta_vec[2], theta_vec[3], theta_vec[4], theta_vec[5])

    cfg["runtime.iteration"] = iteration
    cfg["runtime.current_method"] = "saved_response_analysis"
    cfg["runtime.method_parallel_active"] = false
    cfg["runtime.current_gfdt_path"] = ""
    cfg["runtime.iteration_active_observables"] = copy(active_observable_indices(cfg))

    out_dir = joinpath(run_dir, parsed.output_subdir, iter_name, parsed.output_name)
    mkpath(out_dir)

    X, gfdt_path = generate_gfdt_dataset(cfg, theta, iteration, out_dir)
    cfg["runtime.current_gfdt_path"] = gfdt_path

    method_payloads, times = compute_saved_run_responses(
        cfg,
        theta,
        X,
        cfg["methods.unet"] ? checkpoint_path : nothing,
        iteration,
    )

    active_idx = active_observable_indices(cfg)
    curves = build_method_curves(method_payloads, parsed.responses["response_kind"], active_idx)
    asymptotic_curves = build_asymptotic_curves(method_payloads, Vector{String}(cfg["figures.response_methods"]), active_idx)

    title_text = @sprintf(
        "Responses from saved run (%s) | %s",
        iter_name,
        parsed.responses["response_kind"],
    )

    fig_path = joinpath(out_dir, @sprintf("responses_%s_%dx5.png", parsed.responses["response_kind"], length(active_idx)))
    archive_path = joinpath(out_dir, "responses.hdf5")
    summary_path = joinpath(out_dir, "summary.toml")
    params_used_path = joinpath(out_dir, "parameters_used.toml")

    save_response_figure(
        fig_path,
        times,
        curves;
        asymptotic_curves=asymptotic_curves,
        title_text=title_text,
        dpi=cfg["figures.dpi"],
        observable_row_labels=active_observable_labels(cfg),
    )
    write_response_outputs(archive_path, times, method_payloads, parsed.responses["response_kind"], active_idx)

    open(params_used_path, "w") do io
        TOML.print(io, build_params_used_doc(parsed, run_dir, iter_name, cfg_path, gfdt_path, checkpoint_path, theta_vec))
    end

    summary = Dict{String,Any}(
        "run_dir" => run_dir,
        "iteration" => iteration,
        "iteration_name" => iter_name,
        "method_iter_dir" => method_iter_dir,
        "gfdt_path" => gfdt_path,
        "theta_source_path" => theta_source_path,
        "checkpoint_path" => cfg["methods.unet"] ? checkpoint_path : "",
        "theta" => theta_vec,
        "response_kind" => parsed.responses["response_kind"],
        "active_observables" => active_idx,
        "active_observable_labels" => active_observable_labels(cfg),
        "derived" => Dict(
            "dt_obs" => cfg["integration.dt"] * cfg["datasets.gfdt_save_every"],
            "n_lags" => length(times) - 1,
            "response_total_fine_steps" => (length(times) - 1) * cfg["datasets.gfdt_save_every"],
            "fd_total_samples" => cfg["responses.finite_difference.use_ensemble"] ?
                cfg["responses.finite_difference.ensemble_trajectories"] * cfg["responses.finite_difference.samples_per_trajectory"] :
                cfg["responses.finite_difference.nsamples"],
            "numerical_init_ensembles" => get(cfg, "saved_responses.numerical_init_ensembles", cfg["responses.finite_difference.ensemble_trajectories"]),
        ),
        "methods_enabled" => Dict(
            "unet" => cfg["methods.unet"],
            "gaussian" => cfg["methods.gaussian"],
            "finite_difference" => cfg["methods.finite_difference"],
            "numerical_integration" => cfg["methods.numerical_integration"],
        ),
        "response_methods" => cfg["figures.response_methods"],
        "outputs" => Dict(
            "figure" => fig_path,
            "responses_hdf5" => archive_path,
            "generated_gfdt_hdf5" => gfdt_path,
            "parameters_used" => params_used_path,
        ),
    )
    open(summary_path, "w") do io
        TOML.print(io, summary)
    end

    println("Saved response analysis to $out_dir")
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
