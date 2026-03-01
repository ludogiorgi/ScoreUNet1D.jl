#!/usr/bin/env julia
# Standard command (from repository root):
#   julia --threads auto --project=. scripts/arnold/run_calibration.jl --params scripts/arnold/parameters_calibration.toml
#   nohup julia --threads auto --project=. scripts/arnold/run_calibration.jl --params scripts/arnold/parameters_calibration.toml > scripts/arnold/nohup_run_calibration.log 2>&1 &

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

function write_target_csv(path::AbstractString, vals::Vector{Float64})
    obs_names = ["phi1_mean_x", "phi2_mean_x2", "phi3_mean_x_xm1", "phi4_mean_x_xm2", "phi5_mean_x_xm3"]
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "name,value")
        for (name, v) in zip(obs_names, vals)
            @printf(io, "%s,%.16e\n", name, v)
        end
    end
    return path
end

function write_iter0_observables_csv(path::AbstractString, vals::Vector{Float64}; methods::Vector{String}=String[])
    obs_names = ["phi1_mean_x", "phi2_mean_x2", "phi3_mean_x_xm1", "phi4_mean_x_xm2", "phi5_mean_x_xm3"]
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

function save_final_summary(state::CalibrationState, cfg::Dict{String,Any}, run_dir::String, A_target::Vector{Float64})
    summary_path = joinpath(run_dir, "summary.toml")

    active_idx = cfg["calibration.active_observables"]
    primary_obs = get(cfg, "runtime.primary_observables", nothing)
    obs_residual = if primary_obs isa Vector{Float64}
        norm(primary_obs[active_idx] .- A_target[active_idx])
    else
        Inf
    end

    doc = Dict{String,Any}(
        "run_dir" => run_dir,
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
        ),
        "target_observables" => Dict(
            "phi1_mean_x" => A_target[1],
            "phi2_mean_x2" => A_target[2],
            "phi3_mean_x_xm1" => A_target[3],
            "phi4_mean_x_xm2" => A_target[4],
            "phi5_mean_x_xm3" => A_target[5],
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

function main(args=ARGS)
    parsed = parse_args(args)
    cfg = load_calibration_config(parsed.params_path)

    run_info = ArnoldCommon.next_run_dir(cfg["paths.runs_root"])
    run_dir = run_info.run_dir
    mkpath(run_dir)
    mkpath(joinpath(run_dir, "truth"))

    copy_configs!(cfg, parsed.params_path, run_dir)
    append_run_log(run_dir, "Calibration run created: $(run_info.run_name)")

    cfg["runtime.truth_dir"] = joinpath(run_dir, "truth")

    A_target = compute_target_observables(cfg)
    cfg["runtime.A_target"] = A_target
    write_target_csv(joinpath(run_dir, "truth", "target_observables.csv"), A_target)

    @info "Target observables" A_target
    append_run_log(run_dir, "Target observables = $(A_target)")

    theta0 = theta_from_cfg(cfg)
    state = init_state(theta0, cfg)

    # Seed observable histories at iteration 0 (before any parameter update).
    # This baseline is shared across methods because all start from the same θ₀.
    try
        obs0 = estimate_observables_for_convergence(theta0, cfg)
        cfg["runtime.obs0"] = copy(obs0)
        for method in enabled_methods(cfg)
            push!(state.obs_history[method], copy(obs0))
        end
        write_iter0_observables_csv(
            joinpath(run_dir, "truth", "observables_iter0.csv"),
            obs0;
            methods=enabled_methods(cfg),
        )
    catch err
        @warn "Failed to compute iteration-0 observables baseline" error = sprint(showerror, err)
    end

    prev_checkpoint = nothing
    chosen_primary = primary_method(cfg)

    append_run_log(run_dir, "Enabled methods: $(join(enabled_methods(cfg), ", "))")
    append_run_log(run_dir, "Primary update method: $chosen_primary")

    for it in 1:cfg["calibration.max_iterations"]
        state.iteration = it
        cfg["runtime.iteration"] = it

        iter_dir = joinpath(run_dir, @sprintf("iter_%03d", it))
        ensure_iteration_dirs!(iter_dir)

        theta_tuple = (state.theta[1], state.theta[2], state.theta[3], state.theta[4], state.theta[5])
        @info "Calibration iteration" iteration = it theta = state.theta
        append_run_log(run_dir, "Iteration $it start theta=$(state.theta)")

        reclaim_runtime_memory!()

        train_path, gfdt_path = generate_iteration_datasets(cfg, theta_tuple, it, run_dir)
        append_run_log(run_dir, "Iteration $it datasets generated train=$train_path gfdt=$gfdt_path")

        checkpoint_path = nothing
        if cfg["methods.unet"]
            checkpoint_path = train_iteration_score(cfg, train_path, it, run_dir; prev_checkpoint=prev_checkpoint)
            prev_checkpoint = checkpoint_path
            append_run_log(run_dir, "Iteration $it checkpoint=$checkpoint_path")
        end

        reclaim_runtime_memory!()

        jacobians = compute_iteration_jacobians(cfg, theta_tuple, gfdt_path, checkpoint_path, it, run_dir)
        append_run_log(run_dir, "Iteration $it Jacobians computed for methods: $(join(sort(collect(keys(jacobians))), ", "))")

        methods_iter = enabled_methods(cfg)
        method_diags = Dict{String,Any}()
        observables = Dict{String,Vector{Float64}}()
        proposed_updates = Dict{String,NamedTuple{(:theta, :delta, :diag),Tuple{Vector{Float64},Vector{Float64},Dict{String,Any}}}}()

        nmethods = length(methods_iter)
        theta_props = Vector{Vector{Float64}}(undef, nmethods)
        delta_props = Vector{Vector{Float64}}(undef, nmethods)
        diag_props = Vector{Dict{String,Any}}(undef, nmethods)
        obs_props = Vector{Vector{Float64}}(undef, nmethods)
        jac_props = Vector{Matrix{Float64}}(undef, nmethods)
        cond_props = fill(NaN, nmethods)

        do_parallel_methods = cfg["performance.parallel_methods"] && nmethods > 1 && Base.Threads.nthreads() > 1

        compute_method_index! = function (midx::Int)
            method = methods_iter[midx]
            haskey(jacobians, method) || error("Missing Jacobian output for method '$method'")
            jr = jacobians[method]

            method_cfg = copy(cfg)
            method_cfg["runtime.current_method"] = method
            method_cfg["runtime.current_observable_series"] = jr.A
            method_cfg["runtime.method_parallel_active"] = do_parallel_methods

            method_theta = state.theta_per_method[method][end]
            theta_proposed, delta_proposed, diag_proposed = perform_newton_update(jr.S, jr.G, A_target, method_theta, method_cfg)

            obs_at_proposed = try
                estimate_observables_for_convergence(theta_proposed, method_cfg; seed_offset=10_000 * midx)
            catch err
                @warn "Could not estimate observables at proposed theta for method; using current" method err=sprint(showerror, err)
                copy(jr.G)
            end

            cond_sub = try
                cond(jr.S[cfg["calibration.active_observables"], cfg["calibration.free_parameters"]])
            catch
                NaN
            end

            theta_props[midx] = copy(theta_proposed)
            delta_props[midx] = copy(delta_proposed)
            diag_props[midx] = diag_proposed
            obs_props[midx] = obs_at_proposed
            jac_props[midx] = copy(jr.S)
            cond_props[midx] = cond_sub
            return nothing
        end

        if do_parallel_methods
            Base.Threads.@threads for midx in 1:nmethods
                compute_method_index!(midx)
            end
        else
            for midx in 1:nmethods
                compute_method_index!(midx)
            end
        end

        for midx in 1:nmethods
            method = methods_iter[midx]
            theta_proposed = theta_props[midx]
            delta_proposed = delta_props[midx]
            diag_proposed = diag_props[midx]
            obs_at_proposed = obs_props[midx]
            jrS = jac_props[midx]
            cond_sub = cond_props[midx]

            push!(state.obs_history[method], obs_at_proposed)
            push!(state.jacobian_history[method], jrS)
            push!(state.theta_per_method[method], copy(theta_proposed))
            observables[method] = obs_at_proposed
            proposed_updates[method] = (theta=copy(theta_proposed), delta=copy(delta_proposed), diag=diag_proposed)

            method_diags[method] = Dict{String,Any}(
                "theta_proposed" => copy(theta_proposed),
                "delta_theta" => copy(delta_proposed),
                "jacobian_cond_active_free" => cond_sub,
                "newton" => diag_proposed,
            )
        end

        haskey(proposed_updates, chosen_primary) || error("Missing proposed update for primary method '$chosen_primary'")
        chosen_update = proposed_updates[chosen_primary]
        theta_new = copy(chosen_update.theta)
        delta_theta = copy(chosen_update.delta)
        diag_primary = chosen_update.diag
        effective_primary = chosen_primary

        rejected_update = Bool(get(diag_primary, "line_search_used", false)) &&
                          Float64(get(diag_primary, "line_search_scale", 1.0)) == 0.0 &&
                          norm(delta_theta) == 0.0

        if rejected_update
            best_method = ""
            best_residual = Inf
            for method in methods_iter
                haskey(proposed_updates, method) || continue
                upd = proposed_updates[method]
                scale = Float64(get(upd.diag, "line_search_scale", 0.0))
                residual_after = Float64(get(upd.diag, "residual_after", Inf))
                if scale > 0.0 && isfinite(residual_after) && residual_after < best_residual
                    best_method = method
                    best_residual = residual_after
                end
            end
            if !isempty(best_method) && best_method != chosen_primary
                fallback = proposed_updates[best_method]
                theta_new = copy(fallback.theta)
                delta_theta = copy(fallback.delta)
                diag_primary = fallback.diag
                effective_primary = best_method
                rejected_update = false
                append_run_log(run_dir, "Iteration $it primary '$chosen_primary' rejected; falling back to '$best_method'")
                @info "Primary update rejected; using fallback method" requested_primary = chosen_primary fallback_method = best_method residual_after = best_residual
            end
        end

        state.theta .= theta_new
        push!(state.theta_history, copy(theta_new))
        state.convergence_metric = rejected_update ? Inf : (norm(delta_theta) / max(norm(theta_new), eps()))

        jr_primary = jacobians[effective_primary]
        cfg["runtime.primary_observables"] = copy(jr_primary.G)

        active_idx = cfg["calibration.active_observables"]
        obs_residual = norm(jr_primary.G[active_idx] .- A_target[active_idx])

        cfg["runtime.iteration_diagnostics"] = Dict{String,Any}(
            "iteration" => it,
            "primary_method_requested" => chosen_primary,
            "primary_method" => effective_primary,
            "theta_before" => collect(theta_tuple),
            "theta_after" => copy(theta_new),
            "delta_theta_applied" => copy(delta_theta),
            "convergence_metric" => state.convergence_metric,
            "observable_residual" => obs_residual,
            "newton_primary" => diag_primary,
            "per_method" => method_diags,
            "checkpoint_path" => checkpoint_path === nothing ? "" : checkpoint_path,
            "train_data_path" => train_path,
            "gfdt_data_path" => gfdt_path,
        )

        save_iteration_outputs(state, cfg, run_dir, it, jacobians, observables)
        try
            iter_conv_fig = save_convergence_figure(state, cfg, run_dir)
            isempty(iter_conv_fig) || append_run_log(run_dir, "Saved convergence figure (through iteration $it): $iter_conv_fig")
        catch err
            @warn "Failed to refresh convergence figure after iteration" iteration = it error = sprint(showerror, err)
            append_run_log(run_dir, "Failed to refresh convergence figure at iteration $it: $(sprint(showerror, err))")
        end

        append_run_log(
            run_dir,
            "Iteration $it done primary=$effective_primary theta_new=$(theta_new) rel_step=$(state.convergence_metric) obs_residual=$(obs_residual)",
        )

        if check_convergence(state, cfg)
            state.converged = true
            @info "Converged" iteration = it metric = state.convergence_metric obs_residual = get(cfg, "runtime.last_obs_residual", NaN)
            append_run_log(run_dir, "Converged at iteration $it")
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

    summary_path = save_final_summary(state, cfg, run_dir, A_target)
    append_run_log(run_dir, "Saved final summary: $summary_path")

    @info "Calibration complete" run_dir converged = state.converged iterations = state.iteration summary_path
    return run_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
