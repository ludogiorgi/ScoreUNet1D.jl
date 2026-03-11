#!/usr/bin/env julia

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Dates
using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML
using Base.Threads

include(joinpath(@__DIR__, "run_calibration.jl"))

using .ArnoldCommon
using .CalibrationCommon

const DEFAULT_SAMPLE_GRID = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
const DEFAULT_REPLICATES = 6
const DEFAULT_GFDT_SUBSET = 100_000
const DEFAULT_GFDT_START = 450_001
const DEFAULT_FD_TRAJECTORIES = 24
const DEFAULT_FD_SAMPLES = 20_000
const REF_TRAJECTORIES = 32
const REF_SAMPLES_PER_TRAJECTORY = 50_000

timestamp() = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

function logmsg(msg::AbstractString)
    println("[$(timestamp())] $msg")
    flush(stdout)
    return nothing
end

function parse_args(args::Vector{String})
    run_dir = abspath(joinpath(@__DIR__, "runs_calibration", "run_048"))
    output_dir = joinpath(run_dir, "analysis_iter_001")
    score_device = get(ENV, "ARNOLD_SCORE_DEVICE", get(ENV, "ARNOLD_GPU_DEVICE", "GPU:0"))
    sample_grid = copy(DEFAULT_SAMPLE_GRID)
    replicates = DEFAULT_REPLICATES
    gfdt_subset = DEFAULT_GFDT_SUBSET
    gfdt_start = DEFAULT_GFDT_START
    fd_trajectories = DEFAULT_FD_TRAJECTORIES
    fd_samples = DEFAULT_FD_SAMPLES

    i = 1
    while i <= length(args)
        a = strip(args[i])
        if a == "--run-dir"
            i == length(args) && error("--run-dir expects a value")
            run_dir = abspath(strip(args[i + 1]))
            i += 2
        elseif a == "--output-dir"
            i == length(args) && error("--output-dir expects a value")
            output_dir = abspath(strip(args[i + 1]))
            i += 2
        elseif a == "--score-device"
            i == length(args) && error("--score-device expects a value")
            score_device = strip(args[i + 1])
            i += 2
        elseif a == "--replicates"
            i == length(args) && error("--replicates expects a value")
            replicates = parse(Int, strip(args[i + 1]))
            i += 2
        elseif a == "--sample-grid"
            i == length(args) && error("--sample-grid expects comma-separated integers")
            raw = split(strip(args[i + 1]), ",")
            sample_grid = sort(unique(parse.(Int, raw)))
            i += 2
        elseif a == "--gfdt-subset"
            i == length(args) && error("--gfdt-subset expects a value")
            gfdt_subset = parse(Int, strip(args[i + 1]))
            i += 2
        elseif a == "--gfdt-start"
            i == length(args) && error("--gfdt-start expects a value")
            gfdt_start = parse(Int, strip(args[i + 1]))
            i += 2
        elseif a == "--fd-trajectories"
            i == length(args) && error("--fd-trajectories expects a value")
            fd_trajectories = parse(Int, strip(args[i + 1]))
            i += 2
        elseif a == "--fd-samples"
            i == length(args) && error("--fd-samples expects a value")
            fd_samples = parse(Int, strip(args[i + 1]))
            i += 2
        else
            error("Unknown argument: $a")
        end
    end

    replicates >= 1 || error("--replicates must be >= 1")
    all(n -> n >= 1, sample_grid) || error("--sample-grid entries must be >= 1")
    gfdt_subset >= 2 || error("--gfdt-subset must be >= 2")
    gfdt_start >= 1 || error("--gfdt-start must be >= 1")
    fd_trajectories >= 1 || error("--fd-trajectories must be >= 1")
    fd_samples >= 1 || error("--fd-samples must be >= 1")

    return (
        run_dir=run_dir,
        output_dir=output_dir,
        score_device=score_device,
        sample_grid=sample_grid,
        replicates=replicates,
        gfdt_subset=gfdt_subset,
        gfdt_start=gfdt_start,
        fd_trajectories=fd_trajectories,
        fd_samples=fd_samples,
    )
end

function theta_tuple(v::AbstractVector{<:Real})
    length(v) == 5 || error("theta must have length 5")
    return (Float64(v[1]), Float64(v[2]), Float64(v[3]), Float64(v[4]), Float64(v[5]))
end

function load_run48_config(run_dir::String, score_device::String)
    params_path = joinpath(run_dir, "config", "parameters_calibration.toml")
    isfile(params_path) || error("Calibration config not found: $params_path")
    cfg = CalibrationCommon.load_calibration_config(params_path)
    cfg["runtime.iteration"] = 1
    cfg["responses.score_device"] = score_device
    cfg["figures.langevin_device"] = score_device
    return cfg
end

function compute_jacobians_from_subset(
    cfg::Dict{String,Any},
    theta::Vector{Float64},
    gfdt_path::String,
    checkpoint_path::String,
    iteration::Int;
    nsamples::Int,
    start_index::Int,
)
    X = load_x_matrix(
        gfdt_path,
        cfg["datasets.gfdt_key"];
        nsamples=nsamples,
        start_index=start_index,
        label="analysis_gfdt",
    )
    obs_m = Int(cfg["observables.m"])
    n_obs = CalibrationCommon.observable_count(cfg)

    A = ArnoldCommon.compute_observables_series(
        X,
        cfg["observables.F_ref"],
        cfg["observables.alpha0_ref"],
        cfg["observables.alpha1_ref"],
        cfg["observables.alpha2_ref"],
        cfg["observables.alpha3_ref"],
        obs_m,
    )
    G_obs = vec(mean(A; dims=2))
    all(isfinite, G_obs) || error("Observable averages contain non-finite values")

    cfg["runtime.current_observable_series"] = A
    cfg["runtime.current_gfdt_path"] = gfdt_path

    n_lags, dt_obs = CalibrationCommon.response_n_lags(cfg, size(X, 2))
    apply_sigma_mask = Bool(cfg["calibration.apply_sigma_jacobian_mask"])
    out = Dict{String,NamedTuple{(:S,:G,:A,:times,:R_step,:C),Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Vector{Float64},Array{Float64,3},Array{Float64,3}}}}()

    if cfg["methods.unet"]
        isfile(checkpoint_path) || error("UNet checkpoint not found: $checkpoint_path")
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
            CalibrationCommon.apply_sigma_surgical_mask!(S; row_var=2, col_sigma=5)
        end
        out["unet"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.gaussian"]
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
            CalibrationCommon.apply_sigma_surgical_mask!(S; row_var=2, col_sigma=5)
        end
        out["gaussian"] = (S=S, G=G_obs, A=A, times=times, R_step=R_step, C=C)
    end

    if cfg["methods.finite_difference"]
        Sfd = CalibrationCommon.compute_fd_jacobian_asymptotic(theta_tuple(theta), cfg)
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

function compute_linearized_update(S::AbstractMatrix, G::AbstractVector, A_target::AbstractVector, theta0::Vector{Float64}, cfg::Dict{String,Any})
    active_idx = Vector{Int}(cfg["calibration.active_observables"])
    free_idx = Vector{Int}(cfg["calibration.free_parameters"])
    S_sub = Matrix{Float64}(S[active_idx, free_idx])
    G_sub = Float64.(G[active_idx])
    A_sub = Float64.(A_target[active_idx])
    W_sub = CalibrationCommon.build_weight_matrix(cfg, active_idx)
    cond_sub = try
        cond(S_sub)
    catch
        NaN
    end

    gamma = Float64(cfg["calibration.regularization_gamma"])
    gamma_eff = gamma > 0 ? gamma : 1e-8
    Gamma_sub = Symmetric(Matrix{Float64}(I, length(free_idx), length(free_idx)) * gamma_eff)
    correction_sub, raw_diag = CalibrationCommon.newton_step_bridge(S_sub, W_sub, Gamma_sub, G_sub, A_sub)

    correction_full = zeros(Float64, 5)
    correction_full[free_idx] .= correction_sub

    damping_requested = Float64(cfg["calibration.damping"])
    damping_effective = damping_requested
    rel_cap = CalibrationCommon.active_relative_step_cap(cfg, cond_sub)
    rel_step_unclipped = CalibrationCommon.relative_step_norm(damping_requested .* correction_full, theta0)
    step_cap_triggered = false
    step_cap_factor = 1.0
    if isfinite(rel_cap) && rel_step_unclipped > rel_cap
        step_cap_triggered = true
        step_cap_factor = rel_cap / max(rel_step_unclipped, eps())
        damping_effective *= step_cap_factor
    end

    delta = damping_effective .* correction_full
    theta_new = theta0 .- delta
    theta_new[5] = max(theta_new[5], 1e-8)

    linear_delta_obs = S_sub * correction_sub
    predicted_obs = copy(Float64.(G))
    predicted_obs[active_idx] .= G_sub .- damping_effective .* linear_delta_obs

    predicted_residual_active = CalibrationCommon.weighted_observable_residual(
        cfg,
        predicted_obs,
        A_target;
        active_idx=active_idx,
    )
    gfdt_residual_active = CalibrationCommon.weighted_observable_residual(
        cfg,
        G,
        A_target;
        active_idx=active_idx,
    )

    stability_ok = try
        CalibrationCommon.theta_candidate_is_stable(theta_new, cfg)
    catch
        false
    end

    return Dict{String,Any}(
        "theta_new" => theta_new,
        "delta_theta" => delta,
        "correction_full" => correction_full,
        "cond_active_free" => cond_sub,
        "gamma_effective" => gamma_eff,
        "raw_cond" => get(raw_diag, :cond, NaN),
        "raw_rhs_norm" => get(raw_diag, :nrm_rhs, NaN),
        "damping_requested" => damping_requested,
        "damping_effective" => damping_effective,
        "relative_step_unclipped" => rel_step_unclipped,
        "relative_step_applied" => CalibrationCommon.relative_step_norm(delta, theta0),
        "step_cap_triggered" => step_cap_triggered,
        "step_cap_factor" => step_cap_factor,
        "gfdt_residual_active" => gfdt_residual_active,
        "predicted_residual_active" => predicted_residual_active,
        "stable_rollout" => stability_ok,
    )
end

function append_update_to_hdf5!(path::String, method::String, update::Dict{String,Any})
    h5open(path, "r+") do h5
        base = joinpath("updates", method)
        for (key, val) in update
            if val isa AbstractVector
                h5[joinpath(base, key)] = collect(Float64.(val))
            elseif val isa Bool
                h5[joinpath(base, key)] = Int(val)
            elseif val isa Real
                h5[joinpath(base, key)] = Float64(val)
            else
                h5[joinpath(base, key)] = string(val)
            end
        end
    end
    return nothing
end

function estimate_reference_observables(theta::Vector{Float64}, cfg::Dict{String,Any}, method_index::Int)
    return CalibrationCommon.estimate_observables_ensemble(
        theta,
        cfg;
        trajectories=REF_TRAJECTORIES,
        samples_per_trajectory=REF_SAMPLES_PER_TRAJECTORY,
        save_every=Int(cfg["observables_ensemble.save_every"]),
        spinup_steps=Int(cfg["observables_ensemble.spinup_steps"]),
        seed_base=900_000_000 + 100_000_000 * method_index,
        parallel_trajectories=true,
    )
end

function estimate_single_trajectory(theta::Vector{Float64}, cfg::Dict{String,Any}, n_samples::Int, seed::Int)
    local_cfg = copy(cfg)
    local_cfg["integration.save_every"] = Int(cfg["observables_ensemble.save_every"])
    return Float64.(compute_steady_state_observables(
        theta_tuple(theta),
        local_cfg,
        n_samples,
        Int(cfg["observables_ensemble.spinup_steps"]),
        seed,
    ))
end

function summarize_sample_grid(rows::Vector{NamedTuple}, sample_grid::Vector{Int}, tol::Float64)
    summary = Dict{String,Any}()
    min_n_active = nothing
    min_n_full = nothing
    min_n_maxabs = nothing

    for n in sample_grid
        subset = filter(r -> r.n_samples == n, rows)
        isempty(subset) && continue
        max_active = maximum(r.active_l2_error for r in subset)
        max_full = maximum(r.full_l2_error for r in subset)
        max_abs = maximum(r.max_abs_error for r in subset)
        mean_active = mean(r.active_l2_error for r in subset)
        mean_full = mean(r.full_l2_error for r in subset)
        summary[string(n)] = Dict(
            "replicates" => length(subset),
            "mean_active_l2_error" => mean_active,
            "max_active_l2_error" => max_active,
            "mean_full_l2_error" => mean_full,
            "max_full_l2_error" => max_full,
            "max_abs_error" => max_abs,
        )
        if min_n_active === nothing && max_active <= tol
            min_n_active = n
        end
        if min_n_full === nothing && max_full <= tol
            min_n_full = n
        end
        if min_n_maxabs === nothing && max_abs <= tol
            min_n_maxabs = n
        end
    end

    summary["criterion_tol"] = tol
    summary["min_samples_active_l2"] = something(min_n_active, -1)
    summary["min_samples_full_l2"] = something(min_n_full, -1)
    summary["min_samples_max_abs"] = something(min_n_maxabs, -1)
    return summary
end

function write_sample_rows_csv(path::String, rows::Vector{NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,n_samples,replicate,seed,active_l2_error,full_l2_error,max_abs_error,active_gap_to_target,full_gap_to_target")
        for row in rows
            @printf(
                io,
                "%s,%d,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                row.method,
                row.n_samples,
                row.replicate,
                row.seed,
                row.active_l2_error,
                row.full_l2_error,
                row.max_abs_error,
                row.active_gap_to_target,
                row.full_gap_to_target,
            )
        end
    end
    return path
end

function write_updates_toml(path::String, updates::Dict{String,Dict{String,Any}})
    doc = Dict{String,Any}()
    for (method, payload) in sort(collect(updates); by=first)
        doc[method] = Dict{String,Any}(
            "theta_new" => payload["theta_new"],
            "delta_theta" => payload["delta_theta"],
            "cond_active_free" => payload["cond_active_free"],
            "gamma_effective" => payload["gamma_effective"],
            "damping_requested" => payload["damping_requested"],
            "damping_effective" => payload["damping_effective"],
            "relative_step_unclipped" => payload["relative_step_unclipped"],
            "relative_step_applied" => payload["relative_step_applied"],
            "step_cap_triggered" => payload["step_cap_triggered"],
            "step_cap_factor" => payload["step_cap_factor"],
            "gfdt_residual_active" => payload["gfdt_residual_active"],
            "predicted_residual_active" => payload["predicted_residual_active"],
            "stable_rollout" => payload["stable_rollout"],
        )
    end
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return path
end

function write_sample_summary_toml(path::String, summary::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, summary)
    end
    return path
end

function main(args=ARGS)
    parsed = parse_args(args)
    run_dir = parsed.run_dir
    output_dir = parsed.output_dir
    mkpath(output_dir)

    logmsg("Loading run artifacts from $run_dir")
    cfg = load_run48_config(run_dir, parsed.score_device)
    cfg["responses.finite_difference.ensemble_trajectories"] = parsed.fd_trajectories
    cfg["responses.finite_difference.samples_per_trajectory"] = parsed.fd_samples
    theta0 = theta_from_cfg(cfg)
    A_target = CalibrationCommon.compute_target_observables(cfg)
    cfg["runtime.A_target"] = A_target

    gfdt_path = joinpath(run_dir, "method_unet", "iter_001", "data", "gfdt_stochastic.hdf5")
    checkpoint_path = joinpath(run_dir, "method_unet", "iter_001", "model", "final_checkpoint.bson")
    isfile(gfdt_path) || error("GFDT dataset not found: $gfdt_path")
    isfile(checkpoint_path) || error("UNet checkpoint not found: $checkpoint_path")

    logmsg("Computing Jacobians from run_048 checkpoint, GFDT subset $(parsed.gfdt_subset) starting at $(parsed.gfdt_start), and FD ensemble $(parsed.fd_trajectories)x$(parsed.fd_samples)")
    jacobians = compute_jacobians_from_subset(
        cfg,
        theta0,
        gfdt_path,
        checkpoint_path,
        1;
        nsamples=parsed.gfdt_subset,
        start_index=parsed.gfdt_start,
    )

    observables = Dict{String,Vector{Float64}}(method => copy(jr.G) for (method, jr) in jacobians)
    jac_path = joinpath(output_dir, "jacobians_iter001.hdf5")
    CalibrationCommon.save_response_iteration_archive(jac_path, cfg, jacobians, observables)
    h5open(jac_path, "r+") do h5
        h5["meta/theta0"] = theta0
        h5["meta/run_dir"] = run_dir
        h5["meta/gfdt_path"] = gfdt_path
        h5["meta/checkpoint_path"] = checkpoint_path
        h5["meta/generated_at"] = string(now())
    end
    logmsg("Saved Jacobians and response data to $jac_path")

    updates = Dict{String,Dict{String,Any}}()
    h5open(jac_path, "r") do h5
        target_from_file = Float64.(read(h5["target/observables"]))
        for method in sort(collect(keys(jacobians)))
            S = Float64.(read(h5[joinpath("jacobians", method)]))
            G = Float64.(read(h5[joinpath("observables", method, "gfdt_mean")]))
            update = compute_linearized_update(S, G, target_from_file, theta0, cfg)
            updates[method] = update
        end
    end
    for (method, update) in updates
        append_update_to_hdf5!(jac_path, method, update)
    end

    updates_path = joinpath(output_dir, "linearized_updates.toml")
    write_updates_toml(updates_path, updates)
    logmsg("Saved linearized parameter updates to $updates_path")

    sample_rows = NamedTuple[]
    sample_summary = Dict{String,Any}()
    active_idx = Vector{Int}(cfg["calibration.active_observables"])
    tol = Float64(cfg["calibration.tol_obs"])

    for (midx, method) in enumerate(sort(collect(keys(updates))))
        theta_new = Vector{Float64}(updates[method]["theta_new"])
        logmsg("Estimating high-accuracy reference observables for method=$method")
        G_ref = estimate_reference_observables(theta_new, cfg, midx)
        active_gap_ref = CalibrationCommon.weighted_observable_residual(cfg, G_ref, A_target; active_idx=active_idx)
        full_gap_ref = norm(G_ref .- A_target)

        method_rows = NamedTuple[]
        for n_samples in parsed.sample_grid
            logmsg("Sample study method=$method n_samples=$n_samples replicates=$(parsed.replicates)")
            rep_rows = Vector{NamedTuple}(undef, parsed.replicates)
            Threads.@threads for rep in 1:parsed.replicates
                seed = 1_200_000_000 + 100_000_000 * midx + 1_000_000 * rep + n_samples
                G_est = estimate_single_trajectory(theta_new, cfg, n_samples, seed)
                active_l2 = norm(G_est[active_idx] .- G_ref[active_idx])
                full_l2 = norm(G_est .- G_ref)
                max_abs = maximum(abs.(G_est .- G_ref))
                active_gap = CalibrationCommon.weighted_observable_residual(cfg, G_est, A_target; active_idx=active_idx)
                full_gap = norm(G_est .- A_target)
                rep_rows[rep] = (
                    method=method,
                    n_samples=n_samples,
                    replicate=rep,
                    seed=seed,
                    active_l2_error=active_l2,
                    full_l2_error=full_l2,
                    max_abs_error=max_abs,
                    active_gap_to_target=active_gap,
                    full_gap_to_target=full_gap,
                )
            end
            append!(method_rows, rep_rows)
            append!(sample_rows, rep_rows)
        end

        method_summary = summarize_sample_grid(method_rows, parsed.sample_grid, tol)
        method_summary["reference_active_gap_to_target"] = active_gap_ref
        method_summary["reference_full_gap_to_target"] = full_gap_ref
        method_summary["reference_trajectories"] = REF_TRAJECTORIES
        method_summary["reference_samples_per_trajectory"] = REF_SAMPLES_PER_TRAJECTORY
        sample_summary[method] = method_summary

        h5open(jac_path, "r+") do h5
            base = joinpath("sample_study", method)
            h5[joinpath(base, "reference_observables")] = G_ref
            h5[joinpath(base, "reference_active_gap_to_target")] = active_gap_ref
            h5[joinpath(base, "reference_full_gap_to_target")] = full_gap_ref
        end
    end

    rows_path = joinpath(output_dir, "sample_convergence_rows.csv")
    write_sample_rows_csv(rows_path, sample_rows)
    summary_path = joinpath(output_dir, "sample_convergence_summary.toml")
    write_sample_summary_toml(summary_path, sample_summary)

    logmsg("Saved sample-convergence rows to $rows_path")
    logmsg("Saved sample-convergence summary to $summary_path")
    logmsg("Analysis complete")
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
