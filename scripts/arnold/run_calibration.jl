#!/usr/bin/env julia
# Standard command (from repository root):
#   julia --threads auto --project=. scripts/arnold/run_calibration.jl --params scripts/arnold/parameters_calibration.toml
#   nohup julia --threads auto --project=. scripts/arnold/run_calibration.jl --params scripts/arnold/parameters_calibration.toml > scripts/arnold/nohup_run_calibration.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup julia --threads auto --project=. scripts/arnold/run_calibration.jl --params scripts/arnold/parameters_calibration.toml > scripts/arnold/nohup_run_calibration_gpu0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup julia --threads auto --project=. scripts/arnold/run_calibration.jl --params scripts/arnold/parameters_calibration.toml > scripts/arnold/nohup_run_calibration_gpu1.log 2>&1 &


if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using CUDA
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

include(joinpath(@__DIR__, "lib", "ArnoldStatsPlots.jl"))
# Includes helper functions such as unet_conjugates(), gaussian_conjugates(),
# compute_numerical_jacobians(), extract_asymptotic_jacobians(), and
# reclaim_device_memory!, and also includes ArnoldCommon.jl.
include(joinpath(@__DIR__, "compute_responses.jl"))
include(joinpath(@__DIR__, "lib", "CalibrationCommon.jl"))
using .ArnoldCommon
using .ArnoldStatsPlots
using .CalibrationCommon

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_calibration.toml")

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

function enabled_methods(cfg::Dict{String,Any})
    out = String[]
    cfg["methods.unet"] && push!(out, "unet")
    cfg["methods.gaussian"] && push!(out, "gaussian")
    cfg["methods.finite_difference"] && push!(out, "finite_difference")
    return out
end

function primary_method(cfg::Dict{String,Any})
    for method in ("unet", "gaussian", "finite_difference")
        if get(cfg, "methods.$method", false)
            return method
        end
    end
    error("No Jacobian method enabled")
end

function observable_names(cfg::Dict{String,Any})
    m = Int(get(cfg, "observables.m", 3))
    m >= 0 || error("observables.m must be >= 0")
    out = String["phi1_mean_x", "phi2_mean_x2"]
    for lag in 1:m
        push!(out, "phi$(lag + 2)_mean_x_xp$(lag)")
    end
    return out
end

function theta_from_cfg(cfg::Dict{String,Any})
    theta = Float64[
        cfg["initial_theta.alpha0"],
        cfg["initial_theta.alpha1"],
        cfg["initial_theta.alpha2"],
        cfg["initial_theta.alpha3"],
        cfg["initial_theta.sigma"],
    ]
    frac = cfg["initial_theta.perturbation_fraction"]
    if frac > 0.0
        rng = Random.MersenneTwister(cfg["truth.rng_seed"] + 999)
        for i in eachindex(theta)
            sign = rand(rng, (-1, 1))
            theta[i] += sign * frac * abs(theta[i])
        end
        @info "Perturbed initial theta" fraction=frac theta
    end
    return theta
end

function iteration_calibration_selection(cfg::Dict{String,Any}, iteration::Int)
    base_free = Vector{Int}(get(cfg, "calibration.base_free_parameters", cfg["calibration.free_parameters"]))
    base_active = Vector{Int}(get(cfg, "calibration.base_active_observables", cfg["calibration.active_observables"]))
    freeze_params = Vector{Int}(get(cfg, "calibration.freeze_parameters", Int[]))
    freeze_steps = Int(get(cfg, "calibration.freeze_steps", 0))

    if iteration <= freeze_steps && !isempty(freeze_params)
        freeze_set = Set(freeze_params)
        free_iter = [p for p in base_free if !(p in freeze_set)]
        n_frozen = length(base_free) - length(free_iter)
        n_drop = min(n_frozen, max(length(base_active) - 1, 0))
        active_iter = n_drop == 0 ? copy(base_active) : base_active[1:(end - n_drop)]

        isempty(free_iter) && error("All parameters were frozen at iteration $iteration; keep at least one free parameter")
        isempty(active_iter) && error("No active observables remain at iteration $iteration after staged observable dropping")

        return (
            free_parameters=free_iter,
            active_observables=active_iter,
            staged=true,
            frozen_count=n_frozen,
            dropped_observables=n_drop,
            freeze_steps=freeze_steps,
        )
    end

    return (
        free_parameters=copy(base_free),
        active_observables=copy(base_active),
        staged=false,
        frozen_count=0,
        dropped_observables=0,
        freeze_steps=freeze_steps,
    )
end

function init_state(theta0::Vector{Float64}, cfg::Dict{String,Any})
    methods = enabled_methods(cfg)
    theta_per_method = Dict{String,Vector{Vector{Float64}}}()
    obs_history = Dict{String,Vector{Vector{Float64}}}()
    jacobian_history = Dict{String,Vector{Matrix{Float64}}}()

    for method in methods
        theta_per_method[method] = [copy(theta0)]
        obs_history[method] = Vector{Vector{Float64}}()
        jacobian_history[method] = Vector{Matrix{Float64}}()
    end

    return CalibrationState(
        iteration=0,
        theta=copy(theta0),
        theta_history=[copy(theta0)],
        theta_per_method=theta_per_method,
        obs_history=obs_history,
        jacobian_history=jacobian_history,
        converged=false,
        convergence_metric=Inf,
    )
end

function copy_configs!(cfg::Dict{String,Any}, params_path::String, run_dir::String)
    cfg_dir = joinpath(run_dir, "config")
    mkpath(cfg_dir)

    cp(params_path, joinpath(cfg_dir, "parameters_calibration.toml"); force=true)
    cp(cfg["paths.data_params"], joinpath(cfg_dir, "parameters_data.toml"); force=true)
    cp(cfg["paths.train_params"], joinpath(cfg_dir, "parameters_train.toml"); force=true)
    cp(cfg["paths.responses_params"], joinpath(cfg_dir, "parameters_responses.toml"); force=true)

    return cfg_dir
end

function append_run_log(run_dir::String, msg::AbstractString)
    ArnoldCommon.append_log(joinpath(run_dir, "calibration_log.txt"), String(msg))
    return nothing
end

function reclaim_runtime_memory!()
    GC.gc(true)
    try
        reclaim_device_memory!()
    catch
        try
            CUDA.reclaim()
        catch
        end
    end
    return nothing
end

function write_target_csv(path::AbstractString, vals::Vector{Float64}, obs_names::Vector{String})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "name,value")
        for (name, v) in zip(obs_names, vals)
            @printf(io, "%s,%.16e\n", name, v)
        end
    end
    return path
end

function write_iter0_observables_csv(path::AbstractString, vals::Vector{Float64}, obs_names::Vector{String}; methods::Vector{String}=String[])
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,observable,value")
        if isempty(methods)
            for (name, v) in zip(obs_names, vals)
                @printf(io, "shared,%s,%.16e\n", name, v)
            end
        else
            for method in methods
                for (name, v) in zip(obs_names, vals)
                    @printf(io, "%s,%s,%.16e\n", method, name, v)
                end
            end
        end
    end
    return path
end

function observable_gap_norm(A_target::Vector{Float64}, G::Vector{Float64})
    length(G) == length(A_target) || error("Observable length mismatch: G has $(length(G)) entries, target has $(length(A_target))")
    return norm(G .- A_target)
end

function push_method_metric!(iterations_by_method::Dict{String,Vector{Int}},
    residuals_by_method::Dict{String,Vector{Float64}},
    method::String,
    iteration::Int,
    value::Real)
    push!(get!(iterations_by_method, method, Int[]), iteration)
    push!(get!(residuals_by_method, method, Float64[]), Float64(value))
    return nothing
end

function save_observable_gap_figure(path::AbstractString,
    iterations_by_method::Dict{String,Vector{Int}},
    residuals_by_method::Dict{String,Vector{Float64}};
    dpi::Int=180)
    methods = [
        method for method in CalibrationCommon.method_order()
        if haskey(iterations_by_method, method) && haskey(residuals_by_method, method) &&
           !isempty(iterations_by_method[method]) && !isempty(residuals_by_method[method])
    ]
    isempty(methods) && return ""

    default(fontfamily="Computer Modern", dpi=dpi, legendfontsize=10, guidefontsize=11, tickfontsize=10, titlefontsize=12)
    p = plot(
        ;
        xlabel="iteration",
        ylabel="||A - G(theta)||_2",
        title="Observable mismatch norm",
        linewidth=2.0,
        legend=:topright,
        size=(1450, 920),
    )

    for method in methods
        xs_raw = iterations_by_method[method]
        ys_raw = residuals_by_method[method]
        length(xs_raw) == length(ys_raw) || error("Observable-gap history length mismatch for method '$method'")
        ord = sortperm(xs_raw)
        xs = Float64.(xs_raw[ord])
        ys = ys_raw[ord]
        style = CalibrationCommon.style_for_method(method)
        plot!(
            p,
            xs,
            ys;
            color=style.color,
            linestyle=style.linestyle,
            linewidth=2.2,
            marker=style.marker,
            markersize=style.markersize,
            markerstrokewidth=0.5,
            label=style.label,
        )
    end

    hline!(p, [0.0]; color=:gray40, linestyle=:dash, linewidth=1.2, label="")

    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function vector_history_to_matrix(history::Vector{Vector{Float64}}, width::Int)
    isempty(history) && return zeros(Float64, 0, width)
    mat = Matrix{Float64}(undef, length(history), width)
    for (i, vals) in enumerate(history)
        length(vals) == width || error("History vector width mismatch: expected $width, got $(length(vals)) at row $i")
        @views mat[i, :] .= vals
    end
    return mat
end

function matrix_history_to_stack(history::Vector{Matrix{Float64}}, nrow::Int, ncol::Int)
    isempty(history) && return zeros(Float64, nrow, ncol, 0)
    stack = Array{Float64,3}(undef, nrow, ncol, length(history))
    for (i, mat) in enumerate(history)
        size(mat) == (nrow, ncol) || error("History matrix size mismatch at slice $i: expected ($(nrow), $(ncol)), got $(size(mat))")
        @views stack[:, :, i] .= mat
    end
    return stack
end

function serialized_metric_value(value)
    if value isa AbstractVector
        return join(string.(value), ";")
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa AbstractFloat
        v = Float64(value)
        return isfinite(v) ? @sprintf("%.16e", v) : string(v)
    elseif value isa Integer
        return string(value)
    elseif value === nothing
        return ""
    end
    return string(value)
end

function csv_escape(value)
    s = replace(serialized_metric_value(value), "\"" => "\"\"")
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * s * "\""
    end
    return s
end

function write_method_metrics_csv(path::AbstractString, rows::Vector{Dict{String,Any}})
    headers = if isempty(rows)
        String["iteration", "method"]
    else
        sort!(collect(union((Set(keys(row)) for row in rows)...)))
    end

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join((csv_escape(get(row, h, "")) for h in headers), ","))
        end
    end
    return path
end

function write_observable_gap_csv(path::AbstractString,
    iterations_by_method::Dict{String,Vector{Int}},
    residuals_by_method::Dict{String,Vector{Float64}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,iteration,observable_gap_norm")
        for method in CalibrationCommon.method_order()
            haskey(iterations_by_method, method) || continue
            haskey(residuals_by_method, method) || continue
            xs = iterations_by_method[method]
            ys = residuals_by_method[method]
            length(xs) == length(ys) || error("Observable-gap CSV history length mismatch for method '$method'")
            ord = sortperm(xs)
            for idx in ord
                @printf(io, "%s,%d,%.16e\n", method, xs[idx], ys[idx])
            end
        end
    end
    return path
end

function nth_or_nan(value, idx::Int)
    value isa AbstractVector || return NaN
    length(value) >= idx || return NaN
    return Float64(value[idx])
end

function build_method_metric_row(iteration::Int,
    method::String,
    method_diag::Dict{String,Any},
    observables_method::Vector{Float64},
    A_target::Vector{Float64};
    is_primary::Bool=false)
    newton = get(method_diag, "newton", Dict{String,Any}())
    active_idx = Int.(collect(get(method_diag, "active_observables_used", Int[])))
    full_gap = observable_gap_norm(A_target, observables_method)
    active_gap = isempty(active_idx) ? full_gap : norm(observables_method[active_idx] .- A_target[active_idx])

    row = Dict{String,Any}(
        "iteration" => iteration,
        "method" => method,
        "is_primary" => is_primary,
        "observable_gap_norm_full" => full_gap,
        "observable_gap_norm_active" => active_gap,
        "relative_step" => Float64(get(method_diag, "relative_step", NaN)),
        "relative_step_proposed" => Float64(get(method_diag, "relative_step_proposed", NaN)),
        "observable_residual" => Float64(get(method_diag, "observable_residual", NaN)),
        "observable_residual_unweighted" => Float64(get(method_diag, "observable_residual_unweighted", NaN)),
        "observable_residual_proposed" => Float64(get(method_diag, "observable_residual_proposed", NaN)),
        "observable_residual_proposed_unweighted" => Float64(get(method_diag, "observable_residual_proposed_unweighted", NaN)),
        "observable_residual_previous" => Float64(get(method_diag, "observable_residual_previous", NaN)),
        "observable_residual_previous_unweighted" => Float64(get(method_diag, "observable_residual_previous_unweighted", NaN)),
        "postcheck_rejected" => Bool(get(method_diag, "postcheck_rejected", false)),
        "jacobian_cond_active_free" => Float64(get(method_diag, "jacobian_cond_active_free", NaN)),
        "line_search_used" => Bool(get(newton, "line_search_used", false)),
        "line_search_accepted" => Bool(get(newton, "line_search_accepted", false)),
        "line_search_scale" => Float64(get(newton, "line_search_scale", NaN)),
        "gamma_requested" => Float64(get(newton, "gamma_requested", NaN)),
        "gamma_effective" => Float64(get(newton, "gamma_effective", NaN)),
        "lm_attempts_used" => Int(get(newton, "lm_attempts_used", 0)),
        "predicted_reduction" => Float64(get(newton, "predicted_reduction", NaN)),
        "actual_reduction" => Float64(get(newton, "actual_reduction", NaN)),
        "actual_to_predicted_ratio" => Float64(get(newton, "actual_to_predicted_ratio", NaN)),
        "best_candidate_residual" => Float64(get(newton, "best_candidate_residual", NaN)),
        "stability_checked" => Bool(get(newton, "stability_checked", false)),
        "stability_accepted" => Bool(get(newton, "stability_accepted", false)),
        "acceptance_refined" => Bool(get(newton, "acceptance_refined", false)),
        "acceptance_refine_reason" => String(get(newton, "acceptance_refine_reason", "")),
        "relative_step_cap_triggered" => Bool(get(newton, "relative_step_cap_triggered", false)),
        "relative_step_cap" => Float64(get(newton, "relative_step_cap", NaN)),
        "relative_step_cap_factor" => Float64(get(newton, "relative_step_cap_factor", NaN)),
        "free_parameters_used" => copy(Int.(collect(get(method_diag, "free_parameters_used", Int[])))),
        "active_observables_used" => copy(active_idx),
        "train_data_path" => String(get(method_diag, "train_data_path", "")),
        "gfdt_data_path" => String(get(method_diag, "gfdt_data_path", "")),
        "checkpoint_path" => String(get(method_diag, "checkpoint_path", "")),
    )

    for (idx, pname) in enumerate(CalibrationCommon.PARAM_NAMES)
        row["theta_before_$pname"] = nth_or_nan(get(method_diag, "theta_before", nothing), idx)
        row["theta_proposed_$pname"] = nth_or_nan(get(method_diag, "theta_proposed", nothing), idx)
        row["theta_committed_$pname"] = nth_or_nan(get(method_diag, "theta_committed", nothing), idx)
        row["delta_theta_proposed_$pname"] = nth_or_nan(get(method_diag, "delta_theta_proposed", nothing), idx)
        row["delta_theta_$pname"] = nth_or_nan(get(method_diag, "delta_theta", nothing), idx)
        row["newton_correction_$pname"] = nth_or_nan(get(newton, "correction_full", nothing), idx)
        row["newton_correction_raw_$pname"] = nth_or_nan(get(newton, "correction_full_raw", nothing), idx)
        row["applied_step_$pname"] = nth_or_nan(get(newton, "applied_step", nothing), idx)
    end

    return row
end

function save_calibration_history_artifacts(run_dir::String,
    state::CalibrationState,
    cfg::Dict{String,Any},
    A_target::Vector{Float64},
    method_metric_rows::Vector{Dict{String,Any}},
    observable_gap_iters_by_method::Dict{String,Vector{Int}},
    observable_gap_vals_by_method::Dict{String,Vector{Float64}})
    history_dir = joinpath(run_dir, "history")
    mkpath(history_dir)

    obs_width = length(A_target)
    theta_width = length(CalibrationCommon.PARAM_NAMES)
    methods = enabled_methods(cfg)

    history_h5 = joinpath(history_dir, "calibration_history.hdf5")
    h5open(history_h5, "w") do h5
        h5["meta/param_names"] = CalibrationCommon.PARAM_NAMES
        h5["meta/observable_names"] = observable_names(cfg)
        h5["meta/enabled_methods"] = methods
        h5["selection/base_free_parameters"] = get(cfg, "calibration.base_free_parameters", cfg["calibration.free_parameters"])
        h5["selection/base_active_observables"] = get(cfg, "calibration.base_active_observables", cfg["calibration.active_observables"])
        h5["target_observables"] = A_target
        h5["primary_theta_history"] = vector_history_to_matrix(state.theta_history, theta_width)

        for method in methods
            theta_hist = get(state.theta_per_method, method, Vector{Vector{Float64}}())
            obs_hist = get(state.obs_history, method, Vector{Vector{Float64}}())
            jac_hist = get(state.jacobian_history, method, Vector{Matrix{Float64}}())

            h5[joinpath("theta_history", method)] = vector_history_to_matrix(theta_hist, theta_width)
            h5[joinpath("observable_history", method)] = vector_history_to_matrix(obs_hist, obs_width)
            h5[joinpath("jacobian_history", method)] = matrix_history_to_stack(jac_hist, obs_width, theta_width)
            h5[joinpath("observable_gap_history", method, "iterations")] = get(observable_gap_iters_by_method, method, Int[])
            h5[joinpath("observable_gap_history", method, "values")] = get(observable_gap_vals_by_method, method, Float64[])
        end
    end

    metrics_csv = write_method_metrics_csv(joinpath(history_dir, "method_iteration_metrics.csv"), method_metric_rows)
    gap_csv = write_observable_gap_csv(joinpath(history_dir, "observable_gap_norm.csv"), observable_gap_iters_by_method, observable_gap_vals_by_method)

    return Dict(
        "history_hdf5" => history_h5,
        "method_metrics_csv" => metrics_csv,
        "observable_gap_csv" => gap_csv,
    )
end

function save_final_summary(state::CalibrationState, cfg::Dict{String,Any}, run_dir::String, A_target::Vector{Float64})
    summary_path = joinpath(run_dir, "summary.toml")

    active_idx = cfg["calibration.active_observables"]
    primary_obs = get(cfg, "runtime.primary_observables", nothing)
    obs_residual = if primary_obs isa Vector{Float64}
        weighted_observable_residual(cfg, primary_obs, A_target; active_idx=active_idx)
    else
        Inf
    end
    obs_names = observable_names(cfg)
    target_obs = Dict{String,Any}()
    for i in 1:min(length(obs_names), length(A_target))
        target_obs[obs_names[i]] = A_target[i]
    end

    artifact_path(path::String) = isfile(path) ? abspath(path) : ""

    doc = Dict{String,Any}(
        "run_dir" => run_dir,
        "run" => Dict(
            "run_id" => get(cfg, "runtime.run_id", 0),
            "run_name" => get(cfg, "runtime.run_name", ""),
            "run_seed_offset" => get(cfg, "runtime.run_seed_offset", 0),
        ),
        "converged" => state.converged,
        "iterations" => state.iteration,
        "convergence_metric" => state.convergence_metric,
        "obs_residual_primary" => obs_residual,
        "final_theta" => Dict(
            "alpha0" => state.theta[1],
            "alpha1" => state.theta[2],
            "alpha2" => state.theta[3],
            "alpha3" => state.theta[4],
            "sigma" => state.theta[5],
        ),
        "active_observables" => cfg["calibration.active_observables"],
        "free_parameters" => cfg["calibration.free_parameters"],
        "methods" => Dict(
            "unet" => cfg["methods.unet"],
            "gaussian" => cfg["methods.gaussian"],
            "finite_difference" => cfg["methods.finite_difference"],
            "primary" => primary_method(cfg),
            "convergence_scope" => get(cfg, "calibration.convergence_scope", "primary"),
        ),
        "devices" => Dict(
            "training" => cfg["training.device"],
            "responses_score" => cfg["responses.score_device"],
            "figures_langevin" => cfg["figures.langevin_device"],
        ),
        "target_observables" => target_obs,
        "artifacts" => Dict(
            "target_observables_csv" => artifact_path(joinpath(run_dir, "truth", "target_observables.csv")),
            "truth_trajectory_hdf5" => artifact_path(joinpath(run_dir, "truth", "truth_trajectory.hdf5")),
            "convergence_figure" => artifact_path(joinpath(run_dir, "convergence.png")),
            "observable_gap_figure" => artifact_path(joinpath(run_dir, "observable_gap_norm.png")),
            "history_hdf5" => artifact_path(joinpath(run_dir, "history", "calibration_history.hdf5")),
            "method_metrics_csv" => artifact_path(joinpath(run_dir, "history", "method_iteration_metrics.csv")),
            "observable_gap_csv" => artifact_path(joinpath(run_dir, "history", "observable_gap_norm.csv")),
        ),
    )

    open(summary_path, "w") do io
        TOML.print(io, doc)
    end
    return summary_path
end

function ensure_iteration_dirs!(iter_dir::String)
    mkpath(joinpath(iter_dir, "data"))
    mkpath(joinpath(iter_dir, "model", "checkpoints"))
    mkpath(joinpath(iter_dir, "results"))
    mkpath(joinpath(iter_dir, "figures"))
    return iter_dir
end

theta_tuple(theta::AbstractVector{<:Real}) = (
    Float64(theta[1]),
    Float64(theta[2]),
    Float64(theta[3]),
    Float64(theta[4]),
    Float64(theta[5]),
)

method_needs_training(method::String) = method == "unet"
method_uses_gpu(method::String) = method == "unet"

function theta_group_label(theta::NTuple{5,Float64})
    payload = join((@sprintf("%.16e", x) for x in theta), ",")
    return "theta_" * bytes2hex(sha1(payload))[1:16]
end

function prepare_iteration_assets!(cfg::Dict{String,Any},
    state::CalibrationState,
    methods_iter::Vector{String},
    iteration::Int,
    run_dir::String,
    method_run_dirs::Dict{String,String},
    prev_checkpoints::Dict{String,Union{Nothing,String}})
    shared_root = joinpath(run_dir, @sprintf("iter_%03d", iteration), "shared")
    mkpath(shared_root)

    method_thetas = Dict{String,Vector{Float64}}()
    grouped_methods = Dict{NTuple{5,Float64},Vector{String}}()
    for method in methods_iter
        theta_vec = copy(state.theta_per_method[method][end])
        method_thetas[method] = theta_vec
        push!(get!(grouped_methods, theta_tuple(theta_vec), String[]), method)
    end

    assets = Dict{String,NamedTuple{(:theta, :theta_tuple, :train_path, :gfdt_path, :checkpoint_path, :method_root, :shared_group),Tuple{Vector{Float64},NTuple{5,Float64},String,String,Union{Nothing,String},String,String}}}()

    for theta_key in sort!(collect(keys(grouped_methods)); by=theta_group_label)
        group_methods = sort!(copy(grouped_methods[theta_key]))
        group_name = theta_group_label(theta_key)
        shared_dir = joinpath(shared_root, group_name)
        need_train = any(method_needs_training, group_methods)
        train_path, gfdt_path = generate_iteration_datasets(
            cfg,
            theta_key,
            iteration,
            shared_dir;
            generate_train=need_train,
        )
        append_run_log(
            run_dir,
            "Iteration $iteration shared-assets [$group_name] methods=$(join(group_methods, ", ")) train_path=$(isempty(train_path) ? "" : train_path) gfdt_path=$gfdt_path",
        )

        checkpoint_path = nothing
        if "unet" in group_methods
            checkpoint_path = train_iteration_score(cfg, train_path, iteration, method_run_dirs["unet"]; prev_checkpoint=prev_checkpoints["unet"])
            prev_checkpoints["unet"] = checkpoint_path
            append_run_log(run_dir, "Iteration $iteration [unet] checkpoint=$checkpoint_path")
        end

        for method in group_methods
            assets[method] = (
                theta=copy(method_thetas[method]),
                theta_tuple=theta_key,
                train_path=method_needs_training(method) ? train_path : "",
                gfdt_path=gfdt_path,
                checkpoint_path=method == "unet" ? checkpoint_path : nothing,
                method_root=method_run_dirs[method],
                shared_group=group_name,
            )
        end
    end

    return assets
end

function execute_iteration_method(method::String,
    asset,
    state::CalibrationState,
    cfg::Dict{String,Any},
    A_target::Vector{Float64},
    iter_selection,
    iteration::Int,
    run_dir::String;
    method_parallel_active::Bool=false)
    method_theta = copy(asset.theta)
    theta_tuple_method = asset.theta_tuple
    train_path = asset.train_path
    gfdt_path = asset.gfdt_path
    checkpoint_path = asset.checkpoint_path

    append_run_log(run_dir, "Iteration $iteration [$method] start theta=$(method_theta)")
    append_run_log(run_dir, "Iteration $iteration [$method] datasets train=$train_path gfdt=$gfdt_path")

    if !method_parallel_active
        reclaim_runtime_memory!()
    end

    method_cfg = copy(cfg)
    method_cfg["methods.unet"] = method == "unet"
    method_cfg["methods.gaussian"] = method == "gaussian"
    method_cfg["methods.finite_difference"] = method == "finite_difference"
    method_cfg["runtime.current_method"] = method
    method_cfg["runtime.method_parallel_active"] = method_parallel_active
    method_cfg["runtime.iteration"] = iteration
    method_cfg["calibration.free_parameters"] = copy(iter_selection.free_parameters)
    method_cfg["calibration.active_observables"] = copy(iter_selection.active_observables)

    jac_method = compute_iteration_jacobians(method_cfg, theta_tuple_method, gfdt_path, checkpoint_path, iteration, asset.method_root)
    haskey(jac_method, method) || error("Missing Jacobian output for method '$method'")
    jr = jac_method[method]

    active_idx = method_cfg["calibration.active_observables"]
    prev_obs_method = isempty(state.obs_history[method]) ? copy(jr.G) : copy(state.obs_history[method][end])
    prev_obs_residual = weighted_observable_residual(
        method_cfg,
        prev_obs_method,
        A_target;
        active_idx=active_idx,
    )
    prev_obs_residual_unweighted = norm(prev_obs_method[active_idx] .- A_target[active_idx])
    method_cfg["runtime.previous_observable_residual"] = prev_obs_residual
    method_cfg["runtime.previous_observable_residual_unweighted"] = prev_obs_residual_unweighted

    theta_proposed_raw, delta_proposed_raw, diag_proposed = perform_newton_update(jr.S, jr.G, A_target, method_theta, method_cfg)

    obs_at_proposed = try
        estimate_observables_for_convergence(theta_proposed_raw, method_cfg)
    catch err
        @warn "Could not estimate observables at proposed theta for method; using current" method err=sprint(showerror, err)
        copy(jr.G)
    end

    cond_sub = try
        cond(jr.S[method_cfg["calibration.active_observables"], method_cfg["calibration.free_parameters"]])
    catch
        NaN
    end

    obs_residual_proposed = weighted_observable_residual(
        method_cfg,
        obs_at_proposed,
        A_target;
        active_idx=active_idx,
    )
    obs_residual_proposed_unweighted = norm(obs_at_proposed[active_idx] .- A_target[active_idx])

    theta_committed = copy(theta_proposed_raw)
    delta_committed = copy(delta_proposed_raw)
    obs_committed = copy(obs_at_proposed)
    obs_residual_method = obs_residual_proposed
    obs_residual_method_unweighted = obs_residual_proposed_unweighted
    rel_step_proposed = norm(delta_proposed_raw) / max(norm(theta_proposed_raw), eps())
    rel_step = rel_step_proposed

    postcheck_rel_tol = Float64(method_cfg["calibration.postcheck_rel_tol"])
    postcheck_abs_tol = Float64(method_cfg["calibration.postcheck_abs_tol"])
    rollback_threshold = prev_obs_residual * (1.0 + postcheck_rel_tol) + postcheck_abs_tol
    postcheck_rejected = obs_residual_proposed > rollback_threshold
    if postcheck_rejected
        theta_committed .= method_theta
        delta_committed .= 0.0
        obs_committed .= prev_obs_method
        obs_residual_method = prev_obs_residual
        obs_residual_method_unweighted = prev_obs_residual_unweighted
        rel_step = 0.0
        append_run_log(
            run_dir,
            "Iteration $iteration [$method] rollback after post-check: proposed_residual=$(obs_residual_proposed) previous_residual=$(prev_obs_residual) threshold=$(rollback_threshold)",
        )
    end

    method_diag = Dict{String,Any}(
        "theta_before" => copy(method_theta),
        "theta_proposed" => copy(theta_proposed_raw),
        "theta_committed" => copy(theta_committed),
        "delta_theta_proposed" => copy(delta_proposed_raw),
        "delta_theta" => copy(delta_committed),
        "free_parameters_used" => copy(method_cfg["calibration.free_parameters"]),
        "active_observables_used" => copy(method_cfg["calibration.active_observables"]),
        "relative_step" => rel_step,
        "relative_step_proposed" => rel_step_proposed,
        "observable_residual" => obs_residual_method,
        "observable_residual_unweighted" => obs_residual_method_unweighted,
        "observable_residual_proposed" => obs_residual_proposed,
        "observable_residual_proposed_unweighted" => obs_residual_proposed_unweighted,
        "observable_residual_previous" => prev_obs_residual,
        "observable_residual_previous_unweighted" => prev_obs_residual_unweighted,
        "postcheck_rejected" => postcheck_rejected,
        "postcheck_rel_tol" => postcheck_rel_tol,
        "postcheck_abs_tol" => postcheck_abs_tol,
        "postcheck_threshold" => rollback_threshold,
        "jacobian_cond_active_free" => cond_sub,
        "newton" => diag_proposed,
        "checkpoint_path" => checkpoint_path === nothing ? "" : checkpoint_path,
        "train_data_path" => train_path,
        "gfdt_data_path" => gfdt_path,
        "shared_group" => asset.shared_group,
    )

    append_run_log(
        run_dir,
        "Iteration $iteration [$method] done theta_new=$(theta_committed) rel_step=$(rel_step) obs_residual=$(obs_residual_method) proposed_obs_residual=$(obs_residual_proposed) postcheck_rejected=$(postcheck_rejected)",
    )

    if !method_parallel_active
        reclaim_runtime_memory!()
    end

    return (
        method=method,
        jacobian=jr,
        observables=obs_committed,
        theta_committed=theta_committed,
        method_diag=method_diag,
    )
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg = load_calibration_config(parsed.params_path)

    run_info = ArnoldCommon.claim_next_run_dir(cfg["paths.runs_root"]; pad=cfg["run.run_id_padding"])
    apply_run_runtime_overrides!(cfg, run_info.run_id, run_info.run_name)
    run_dir = run_info.run_dir
    mkpath(joinpath(run_dir, "truth"))

    copy_configs!(cfg, parsed.params_path, run_dir)
    append_run_log(run_dir, "Calibration run created: $(run_info.run_name)")
    run_seed_offset = cfg["runtime.run_seed_offset"]
    training_device = cfg["training.device"]
    score_device = cfg["responses.score_device"]
    langevin_device = cfg["figures.langevin_device"]
    append_run_log(
        run_dir,
        "Runtime overrides: seed_offset=$run_seed_offset training_device=$training_device score_device=$score_device langevin_device=$langevin_device",
    )

    cfg["runtime.truth_dir"] = joinpath(run_dir, "truth")

    A_target = compute_target_observables(cfg)
    cfg["runtime.A_target"] = A_target
    obs_names = observable_names(cfg)
    write_target_csv(joinpath(run_dir, "truth", "target_observables.csv"), A_target, obs_names)

    @info "Target observables" A_target
    append_run_log(run_dir, "Target observables = $(A_target)")

    theta0 = theta_from_cfg(cfg)
    state = init_state(theta0, cfg)
    cfg["calibration.base_free_parameters"] = copy(cfg["calibration.free_parameters"])
    cfg["calibration.base_active_observables"] = copy(cfg["calibration.active_observables"])
    methods_enabled = enabled_methods(cfg)
    chosen_primary = primary_method(cfg)
    observable_gap_iters_by_method = Dict{String,Vector{Int}}(m => Int[] for m in methods_enabled)
    observable_gap_vals_by_method = Dict{String,Vector{Float64}}(m => Float64[] for m in methods_enabled)
    method_metric_rows = Dict{String,Any}[]
    history_artifacts = Dict{String,String}()
    observable_gap_path = joinpath(run_dir, "observable_gap_norm.png")

    # Seed observable histories at iteration 0 (before any parameter update).
    # This baseline is shared across methods because all start from the same θ₀.
    try
        obs0 = estimate_observables_for_convergence(theta0, cfg)
        cfg["runtime.obs0"] = copy(obs0)
        for method in methods_enabled
            push!(state.obs_history[method], copy(obs0))
            push_method_metric!(observable_gap_iters_by_method, observable_gap_vals_by_method, method, 0, observable_gap_norm(A_target, obs0))
        end
        write_iter0_observables_csv(
            joinpath(run_dir, "truth", "observables_iter0.csv"),
            obs0,
            obs_names;
            methods=methods_enabled,
        )
        fig_obs0 = save_observable_gap_figure(
            observable_gap_path,
            observable_gap_iters_by_method,
            observable_gap_vals_by_method;
            dpi=cfg["figures.dpi"],
        )
        isempty(fig_obs0) || append_run_log(run_dir, "Saved observable-gap figure (iteration 0): $fig_obs0")
        history_artifacts = save_calibration_history_artifacts(
            run_dir,
            state,
            cfg,
            A_target,
            method_metric_rows,
            observable_gap_iters_by_method,
            observable_gap_vals_by_method,
        )
    catch err
        @warn "Failed to compute iteration-0 observables baseline" error = sprint(showerror, err)
    end

    inactive_methods = Set{String}()
    prev_checkpoints = Dict{String,Union{Nothing,String}}(m => nothing for m in methods_enabled)
    method_run_dirs = Dict{String,String}(m => joinpath(run_dir, "method_" * m) for m in methods_enabled)
    for m in methods_enabled
        mkpath(method_run_dirs[m])
    end

    append_run_log(run_dir, "Enabled methods: $(join(methods_enabled, ", "))")
    append_run_log(run_dir, "Primary update method: $chosen_primary")

    for it in 1:cfg["calibration.max_iterations"]
        state.iteration = it
        cfg["runtime.iteration"] = it

        iter_dir = joinpath(run_dir, @sprintf("iter_%03d", it))
        ensure_iteration_dirs!(iter_dir)

        @info "Calibration iteration" iteration = it
        append_run_log(run_dir, "Iteration $it start")

        iter_selection = iteration_calibration_selection(cfg, it)
        cfg["runtime.iteration_free_parameters"] = copy(iter_selection.free_parameters)
        cfg["runtime.iteration_active_observables"] = copy(iter_selection.active_observables)
        cfg["runtime.iteration_staged_freeze"] = iter_selection.staged
        cfg["runtime.iteration_frozen_count"] = iter_selection.frozen_count
        cfg["runtime.iteration_dropped_observables"] = iter_selection.dropped_observables
        if iter_selection.staged
            append_run_log(
                run_dir,
                "Iteration $it staged-freeze active: frozen=$(iter_selection.frozen_count) for first $(iter_selection.freeze_steps) iterations; free_parameters=$(iter_selection.free_parameters); active_observables=$(iter_selection.active_observables)",
            )
        end

        reclaim_runtime_memory!()

        methods_iter = [m for m in enabled_methods(cfg) if !(m in inactive_methods)]
        isempty(methods_iter) && error("No active calibration methods remain at iteration $it")
        assets = prepare_iteration_assets!(cfg, state, methods_iter, it, run_dir, method_run_dirs, prev_checkpoints)
        method_diags = Dict{String,Any}()
        observables = Dict{String,Vector{Float64}}()
        jacobians = Dict{String,NamedTuple{(:S,:G,:A,:times,:R_step,:C),Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Vector{Float64},Array{Float64,3},Array{Float64,3}}}}()

        record_result! = function (result)
            method = result.method
            push!(state.obs_history[method], result.observables)
            push!(state.jacobian_history[method], result.jacobian.S)
            push!(state.theta_per_method[method], copy(result.theta_committed))
            observables[method] = result.observables
            jacobians[method] = result.jacobian
            method_diags[method] = result.method_diag
            return nothing
        end

        run_method = function (method::String; parallel_active::Bool=false)
            try
                return execute_iteration_method(
                    method,
                    assets[method],
                    state,
                    cfg,
                    A_target,
                    iter_selection,
                    it,
                    run_dir;
                    method_parallel_active=parallel_active,
                )
            catch err
                msg = sprint(showerror, err)
                append_run_log(run_dir, "Iteration $it [$method] interrupted due to error: $msg")
                return (failed=true, method=method, error=msg)
            end
        end

        process_outcome! = function (outcome)
            outcome === nothing && return nothing
            if hasproperty(outcome, :failed) && Bool(getproperty(outcome, :failed))
                push!(inactive_methods, outcome.method)
                @warn "Method interrupted and deactivated for remaining iterations" method=outcome.method iteration=it error=outcome.error
                reclaim_runtime_memory!()
                return nothing
            end
            record_result!(outcome)
            return nothing
        end

        gpu_methods = [m for m in methods_iter if method_uses_gpu(m)]
        cpu_methods = [m for m in methods_iter if !(m in gpu_methods)]

        for method in gpu_methods
            result = run_method(method; parallel_active=false)
            process_outcome!(result)
        end

        if !isempty(cpu_methods)
            if Bool(cfg["performance.parallel_methods"]) && length(cpu_methods) > 1
                max_parallel = min(length(cpu_methods), Int(cfg["performance.max_parallel_methods"]))
                sem = Base.Semaphore(max_parallel)
                task_results = Vector{Any}(undef, length(cpu_methods))
                @sync for (idx, method) in enumerate(cpu_methods)
                    Base.Threads.@spawn begin
                        Base.acquire(sem)
                        try
                            task_results[idx] = run_method(method; parallel_active=true)
                        finally
                            Base.release(sem)
                        end
                    end
                end
                for result in task_results
                    process_outcome!(result)
                end
            else
                for method in cpu_methods
                    result = run_method(method; parallel_active=false)
                    process_outcome!(result)
                end
            end
        end

        isempty(method_diags) && error("All calibration methods failed at iteration $it")

        effective_primary = haskey(method_diags, chosen_primary) ? chosen_primary : first(sort(collect(keys(method_diags))))
        haskey(state.theta_per_method, effective_primary) || error("Missing theta path for effective primary method '$effective_primary'")
        state.theta .= state.theta_per_method[effective_primary][end]
        push!(state.theta_history, copy(state.theta))

        primary_diag = method_diags[effective_primary]
        state.convergence_metric = Float64(get(primary_diag, "relative_step", Inf))
        cfg["runtime.primary_observables"] = copy(observables[effective_primary])
        cfg["runtime.current_observable_series"] = copy(jacobians[effective_primary].A)
        cfg["runtime.current_gfdt_path"] = String(get(primary_diag, "gfdt_data_path", ""))
        obs_residual = Float64(get(primary_diag, "observable_residual", Inf))
        obs_gap_full = observable_gap_norm(A_target, observables[effective_primary])
        for method in sort(collect(keys(method_diags)))
            push_method_metric!(
                observable_gap_iters_by_method,
                observable_gap_vals_by_method,
                method,
                it,
                observable_gap_norm(A_target, observables[method]),
            )
            push!(
                method_metric_rows,
                build_method_metric_row(
                    it,
                    method,
                    method_diags[method],
                    observables[method],
                    A_target;
                    is_primary=(method == effective_primary),
                ),
            )
        end

        langevin_method = ""
        langevin_train_path = ""
        langevin_checkpoint_path = ""
        if haskey(method_diags, "unet")
            md_unet = method_diags["unet"]
            unet_train = String(get(md_unet, "train_data_path", ""))
            unet_ckpt = String(get(md_unet, "checkpoint_path", ""))
            if !isempty(strip(unet_train)) && !isempty(strip(unet_ckpt))
                langevin_method = "unet"
                langevin_train_path = unet_train
                langevin_checkpoint_path = unet_ckpt
            end
        end
        if isempty(langevin_checkpoint_path)
            primary_train = String(get(primary_diag, "train_data_path", ""))
            primary_ckpt = String(get(primary_diag, "checkpoint_path", ""))
            if !isempty(strip(primary_train)) && !isempty(strip(primary_ckpt))
                langevin_method = effective_primary
                langevin_train_path = primary_train
                langevin_checkpoint_path = primary_ckpt
            end
        end
        cfg["runtime.current_langevin_method"] = langevin_method
        cfg["runtime.current_train_path"] = langevin_train_path
        cfg["runtime.current_checkpoint_path"] = langevin_checkpoint_path

        cfg["runtime.iteration_diagnostics"] = Dict{String,Any}(
            "iteration" => it,
            "primary_method_requested" => chosen_primary,
            "primary_method" => effective_primary,
            "theta_primary" => copy(state.theta),
            "convergence_metric_primary" => state.convergence_metric,
            "observable_residual_primary" => obs_residual,
            "observable_gap_norm_primary" => obs_gap_full,
            "langevin_figure_method" => langevin_method,
            "langevin_train_data_path" => langevin_train_path,
            "langevin_checkpoint_path" => langevin_checkpoint_path,
            "staged_freeze_active" => iter_selection.staged,
            "iteration_free_parameters" => copy(iter_selection.free_parameters),
            "iteration_active_observables" => copy(iter_selection.active_observables),
            "iteration_frozen_count" => iter_selection.frozen_count,
            "iteration_dropped_observables" => iter_selection.dropped_observables,
            "inactive_methods" => sort(collect(inactive_methods)),
            "per_method" => method_diags,
        )

        save_iteration_outputs(state, cfg, run_dir, it, jacobians, observables)
        try
            iter_conv_fig = save_convergence_figure(state, cfg, run_dir)
            isempty(iter_conv_fig) || append_run_log(run_dir, "Saved convergence figure (through iteration $it): $iter_conv_fig")
        catch err
            @warn "Failed to refresh convergence figure after iteration" iteration = it error = sprint(showerror, err)
            append_run_log(run_dir, "Failed to refresh convergence figure at iteration $it: $(sprint(showerror, err))")
        end
        try
            iter_obs_gap_fig = save_observable_gap_figure(
                observable_gap_path,
                observable_gap_iters_by_method,
                observable_gap_vals_by_method;
                dpi=cfg["figures.dpi"],
            )
            isempty(iter_obs_gap_fig) || append_run_log(run_dir, "Saved observable-gap figure (through iteration $it): $iter_obs_gap_fig")
        catch err
            @warn "Failed to refresh observable-gap figure after iteration" iteration = it error = sprint(showerror, err)
            append_run_log(run_dir, "Failed to refresh observable-gap figure at iteration $it: $(sprint(showerror, err))")
        end
        try
            history_artifacts = save_calibration_history_artifacts(
                run_dir,
                state,
                cfg,
                A_target,
                method_metric_rows,
                observable_gap_iters_by_method,
                observable_gap_vals_by_method,
            )
            append_run_log(run_dir, "Updated history artifacts at iteration $it")
        catch err
            @warn "Failed to refresh calibration history artifacts" iteration = it error = sprint(showerror, err)
            append_run_log(run_dir, "Failed to refresh calibration history artifacts at iteration $it: $(sprint(showerror, err))")
        end

        append_run_log(run_dir, "Iteration $it primary=$effective_primary theta=$(state.theta) rel_step=$(state.convergence_metric) obs_residual=$(obs_residual)")

        tol_theta = cfg["calibration.tol_theta"]
        tol_obs = cfg["calibration.tol_obs"]
        method_converged = Dict{String,Bool}()
        for method in keys(method_diags)
            md = method_diags[method]
            rel = Float64(get(md, "relative_step", Inf))
            res = Float64(get(md, "observable_residual", Inf))
            method_converged[method] = (rel < tol_theta) && (res < tol_obs)
        end
        convergence_scope = String(get(cfg, "calibration.convergence_scope", "primary"))
        all_converged = if convergence_scope == "all"
            all(values(method_converged))
        else
            get(method_converged, effective_primary, false)
        end
        if iter_selection.staged
            all_converged = false
            append_run_log(run_dir, "Iteration $it: convergence check deferred (staged freeze still active)")
        end
        cfg["runtime.method_converged"] = method_converged

        if all_converged
            state.converged = true
            @info "Converged" iteration = it convergence_scope effective_primary method_converged
            append_run_log(run_dir, "Converged at iteration $it (scope=$convergence_scope, primary=$effective_primary)")
            break
        end

        reclaim_runtime_memory!()
    end

    conv_fig = ""
    try
        conv_fig = save_convergence_figure(state, cfg, run_dir)
        isempty(conv_fig) || append_run_log(run_dir, "Saved convergence figure: $conv_fig")
    catch err
        @warn "Failed to save convergence figure" error = sprint(showerror, err)
        append_run_log(run_dir, "Failed to save convergence figure: $(sprint(showerror, err))")
    end
    try
        obs_gap_fig = save_observable_gap_figure(
            observable_gap_path,
            observable_gap_iters_by_method,
            observable_gap_vals_by_method;
            dpi=cfg["figures.dpi"],
        )
        isempty(obs_gap_fig) || append_run_log(run_dir, "Saved observable-gap figure: $obs_gap_fig")
    catch err
        @warn "Failed to save observable-gap figure" error = sprint(showerror, err)
        append_run_log(run_dir, "Failed to save observable-gap figure: $(sprint(showerror, err))")
    end
    try
        history_artifacts = save_calibration_history_artifacts(
            run_dir,
            state,
            cfg,
            A_target,
            method_metric_rows,
            observable_gap_iters_by_method,
            observable_gap_vals_by_method,
        )
        append_run_log(run_dir, "Saved history artifacts: $(history_artifacts)")
    catch err
        @warn "Failed to save calibration history artifacts" error = sprint(showerror, err)
        append_run_log(run_dir, "Failed to save calibration history artifacts: $(sprint(showerror, err))")
    end

    summary_path = save_final_summary(state, cfg, run_dir, A_target)
    append_run_log(run_dir, "Saved final summary: $summary_path")

    @info "Calibration complete" run_dir converged = state.converged iterations = state.iteration summary_path
    return run_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
