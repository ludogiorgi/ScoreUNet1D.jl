#!/usr/bin/env julia

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

include("l96_parameter_jacobians.jl")

using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

const DEFAULT_CAL_OUT_DIR = "scripts/L96/calibration_outputs/run_021_calibration"

const TARGET_NSAMPLES = 300_000
const TARGET_START_INDEX = 50_001

const EVAL_NSAMPLES = 100_000
const EVAL_BURN = 4_000
const TRIAL_NSAMPLES = 40_000
const TRIAL_BURN = 2_000

const FD_JAC_NSAMPLES = 60_000
const FD_JAC_BURN = 2_000
const FD_JAC_NREP = 1

const GFDT_TMAX_CAL = 0.0

const FD_MAX_ITERS = 2
const GFDT_MAX_ITERS = 4

const FD_DAMPING = 1.0
const GFDT_DAMPING = 0.3
const FD_LAMBDA_REG = 1e-2
const GFDT_LAMBDA_REG = 5.0
const LM_GROWTH = 10.0
const LM_MAX_TRIES = 2
const LS_MAX_TRIES = 4
const OBJ_IMPROVE_EPS = 1e-5

const FD_STEP_CAP = [0.8, 0.15, 0.8, 0.8]
const GFDT_STEP_CAP = [0.35, 0.07, 0.35, 0.35]
const PARAM_LOWER = [6.0, 0.3, 6.0, 6.0]
const PARAM_UPPER = [14.0, 2.0, 14.0, 14.0]

const THETA_INIT = [9.3, 0.9, 9.3, 10.5]
const PARAM_NAMES = ["F", "h", "c", "b"]

function parse_cli(args::Vector{String})
    out = Dict{String,Any}(
        "run_dir" => DEFAULT_RUN_DIR,
        "integration_toml" => DEFAULT_OBS_INTEGRATION_TOML,
        "output_dir" => DEFAULT_CAL_OUT_DIR,
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
        else
            error("Unknown argument: $a")
        end
    end
    return out
end

function tensor_snapshot_to_xy(tensor::Array{Float64,3}, J::Int)
    K, C, N = size(tensor)
    C == J + 1 || error("channel mismatch in tensor_snapshot_to_xy")
    N >= 1 || error("tensor must have at least one snapshot")
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

function aggregate_observable_vector(v::Vector{Float64}, K::Int)
    if length(v) == 5
        return copy(v)
    end
    length(v) == 5 * K || error("aggregate_observable_vector expects length 5 or 5K")
    out = zeros(Float64, 5)
    @inbounds for t in 1:5
        s = 0.0
        for k in 1:K
            s += v[5 * (k - 1) + t]
        end
        out[t] = s / K
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
    tensor = load_observation_subset(cfg; nsamples=TARGET_NSAMPLES, start_index=TARGET_START_INDEX)
    A = compute_observables(tensor)
    Aagg = aggregate_observable_series(A, cfg.K)
    target_mean = vec(mean(Aagg; dims=2))
    target_std = vec(std(Aagg; dims=2))
    floor_std = max(quantile(target_std, 0.2), 1e-8)
    wdiag = 1.0 ./ (target_std .^ 2 .+ floor_std ^ 2)
    return target_mean, target_std, wdiag
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
    θnew = clamp.(θnew, PARAM_LOWER, PARAM_UPPER)
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
    Gagg = vec(mean(Aagg; dims=2))
    return tensor, A, Gagg
end

function write_history_csv(path::AbstractString, rows::Vector{NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "iter,obj,obj_next,damping,cond,theta_F,theta_h,theta_c,theta_b,next_F,next_h,next_c,next_b,rel_param_err")
        for r in rows
            @printf(io, "%d,%.12e,%.12e,%.6f,%.6e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    r.iter, r.obj, r.obj_next, r.damping, r.cond,
                    r.theta[1], r.theta[2], r.theta[3], r.theta[4],
                    r.theta_next[1], r.theta_next[2], r.theta_next[3], r.theta_next[4],
                    r.rel_param_err_next)
        end
    end
    return path
end

function best_theta_from_history(rows::Vector{NamedTuple}, θ_fallback::Vector{Float64})
    isempty(rows) && return copy(θ_fallback)
    best_idx = 1
    best_obj = rows[1].obj_next
    for i in 2:length(rows)
        oi = rows[i].obj_next
        if oi < best_obj
            best_obj = oi
            best_idx = i
        end
    end
    return copy(rows[best_idx].theta_next)
end

function run_calibration_fd(θ0::Vector{Float64},
                            θ_true::Vector{Float64},
                            target_mean::Vector{Float64},
                            wdiag::Vector{Float64},
                            x0_init::Vector{Float64},
                            y0_init::Matrix{Float64},
                            cfg::L96Config;
                            max_iters::Int=FD_MAX_ITERS,
                            damping::Float64=FD_DAMPING)
    θ = copy(θ0)
    rows = NamedTuple[]
    @info "Starting FD calibration" θ0=θ0 max_iters=max_iters damping=damping
    for it in 1:max_iters
        tensor, _, G = evaluate_state_and_stats(θ, x0_init, y0_init, cfg;
                                                nsamples=EVAL_NSAMPLES,
                                                burn_snapshots=EVAL_BURN,
                                                rng_seed=30_000 + it)
        resid = G .- target_mean
        obj = weighted_objective(resid, wdiag)

        xfd, yfd = tensor_snapshot_to_xy(tensor, cfg.J)
        S_fd = finite_difference_jacobian_l96((θ[1], θ[2], θ[3], θ[4]), xfd, yfd, cfg;
                                              h_rel=FD_H_REL,
                                              h_abs=collect(Float64.(FD_H_ABS)),
                                              burn_snapshots=FD_JAC_BURN,
                                              nsamples=FD_JAC_NSAMPLES,
                                              n_rep=FD_JAC_NREP,
                                              seed_base=FD_SEED_BASE + 100_000 * it)
        S_fd_agg = aggregate_jacobian_rows(S_fd, cfg.K)
        accepted = false
        θ_next = copy(θ)
        obj_next = obj
        condM = NaN
        used_damping = 0.0
        λ = FD_LAMBDA_REG
        trial_seed = 40_000 + it
        for _ in 1:LM_MAX_TRIES
            Δθ, condM = solve_weighted_step(S_fd_agg, resid, wdiag; lambda=λ)
            α = damping
            for _ in 1:LS_MAX_TRIES
                θ_try = bounded_update(θ, Δθ; damping=α, step_cap=collect(Float64.(FD_STEP_CAP)))
                _, _, G_try = evaluate_state_and_stats(θ_try, x0_init, y0_init, cfg;
                                                       nsamples=TRIAL_NSAMPLES,
                                                       burn_snapshots=TRIAL_BURN,
                                                       rng_seed=trial_seed)
                obj_try = weighted_objective(G_try .- target_mean, wdiag)
                if obj_try <= obj - OBJ_IMPROVE_EPS * max(obj, 1e-3)
                    θ_next = θ_try
                    obj_next = obj_try
                    used_damping = α
                    accepted = true
                    break
                end
                α *= 0.5
            end
            accepted && break
            λ *= LM_GROWTH
        end
        if !accepted
            θ_next = copy(θ)
            obj_next = obj
            used_damping = 0.0
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
        ))

        @info "FD iteration" iter=it obj=obj obj_next=obj_next θ=θ θ_next=θ_next rel_param_err_next=rel_err_next cond=condM
        θ = θ_next
    end
    return θ, rows
end

function run_calibration_gfdt_unet(θ0::Vector{Float64},
                                   θ_true::Vector{Float64},
                                   target_mean::Vector{Float64},
                                   wdiag::Vector{Float64},
                                   x0_init::Vector{Float64},
                                   y0_init::Matrix{Float64},
                                   cfg::L96Config,
                                   checkpoint_path::AbstractString;
                                   max_iters::Int=GFDT_MAX_ITERS,
                                   damping::Float64=GFDT_DAMPING)
    θ = copy(θ0)
    rows = NamedTuple[]
    Δt_obs = cfg.dt * cfg.save_every
    @info "Starting GFDT+UNet calibration" θ0=θ0 max_iters=max_iters damping=damping checkpoint=checkpoint_path
    for it in 1:max_iters
        tensor, A, G = evaluate_state_and_stats(θ, x0_init, y0_init, cfg;
                                                nsamples=EVAL_NSAMPLES,
                                                burn_snapshots=EVAL_BURN,
                                                rng_seed=50_000 + it)
        resid = G .- target_mean
        obj = weighted_objective(resid, wdiag)

        G_unet = compute_G_unet(tensor, checkpoint_path, (θ[1], θ[2], θ[3], θ[4]);
                                batch_size=SCORE_BATCH_SIZE,
                                device_pref=SCORE_DEVICE_PREF)
        S_unet = build_gfdt_jacobian(A, G_unet, Δt_obs, GFDT_TMAX_CAL; mean_center=true)
        S_unet_agg = aggregate_jacobian_rows(S_unet, cfg.K)

        accepted = false
        θ_next = copy(θ)
        obj_next = obj
        condM = NaN
        used_damping = 0.0
        λ = GFDT_LAMBDA_REG
        trial_seed = 60_000 + it
        for _ in 1:LM_MAX_TRIES
            Δθ, condM = solve_weighted_step(S_unet_agg, resid, wdiag; lambda=λ)
            α = damping
            for _ in 1:LS_MAX_TRIES
                θ_try = bounded_update(θ, Δθ; damping=α, step_cap=collect(Float64.(GFDT_STEP_CAP)))
                _, _, G_try = evaluate_state_and_stats(θ_try, x0_init, y0_init, cfg;
                                                       nsamples=TRIAL_NSAMPLES,
                                                       burn_snapshots=TRIAL_BURN,
                                                       rng_seed=trial_seed)
                obj_try = weighted_objective(G_try .- target_mean, wdiag)
                if obj_try <= obj - OBJ_IMPROVE_EPS * max(obj, 1e-3)
                    θ_next = θ_try
                    obj_next = obj_try
                    accepted = true
                    used_damping = α
                    break
                end
                α *= 0.5
            end
            accepted && break
            λ *= LM_GROWTH
        end
        if !accepted
            θ_next = copy(θ)
            obj_next = obj
            used_damping = 0.0
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
        ))

        @info "GFDT+UNet iteration" iter=it obj=obj obj_next=obj_next θ=θ θ_next=θ_next rel_param_err_next=rel_err_next cond=condM
        θ = θ_next
    end
    return θ, rows
end

function write_summary_markdown(path::AbstractString,
                                θ_true::Vector{Float64},
                                θ0::Vector{Float64},
                                θ_gfdt0::Vector{Float64},
                                θ_fd::Vector{Float64},
                                θ_gfdt::Vector{Float64},
                                rows_fd::Vector{NamedTuple},
                                rows_gfdt::Vector{NamedTuple},
                                checkpoint_path::AbstractString,
                                cfg::L96Config)
    mkpath(dirname(path))
    fd_err = (θ_fd .- θ_true) ./ θ_true
    gfdt_err = (θ_gfdt .- θ_true) ./ θ_true
    open(path, "w") do io
        println(io, "# L96 Parameter Calibration Report")
        println(io)
        println(io, "- true parameters `(F,h,c,b)` = `", θ_true, "`")
        println(io, "- initial FD guess `(F,h,c,b)` = `", θ0, "`")
        println(io, "- initial GFDT guess `(F,h,c,b)` = `", θ_gfdt0, "`")
        println(io, "- UNet checkpoint = `", abspath(checkpoint_path), "`")
        println(io, "- L96 config: `K=", cfg.K, "`, `J=", cfg.J, "`, `dt=", cfg.dt, "`, `save_every=", cfg.save_every, "`, `process_noise_sigma=", cfg.process_noise_sigma, "`")
        println(io)
        println(io, "## Final Estimates")
        println(io)
        println(io, "| Method | F | h | c | b | rel L2 error |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        println(io, "| FD | ", @sprintf("%.6f", θ_fd[1]), " | ", @sprintf("%.6f", θ_fd[2]), " | ", @sprintf("%.6f", θ_fd[3]), " | ", @sprintf("%.6f", θ_fd[4]), " | ", @sprintf("%.6e", norm(fd_err) / sqrt(4)), " |")
        println(io, "| GFDT+UNet | ", @sprintf("%.6f", θ_gfdt[1]), " | ", @sprintf("%.6f", θ_gfdt[2]), " | ", @sprintf("%.6f", θ_gfdt[3]), " | ", @sprintf("%.6f", θ_gfdt[4]), " | ", @sprintf("%.6e", norm(gfdt_err) / sqrt(4)), " |")
        println(io)
        println(io, "## Relative Errors by Parameter")
        println(io)
        println(io, "| Method | dF/F | dh/h | dc/c | db/b |")
        println(io, "|---|---:|---:|---:|---:|")
        println(io, "| FD | ", @sprintf("%.6e", fd_err[1]), " | ", @sprintf("%.6e", fd_err[2]), " | ", @sprintf("%.6e", fd_err[3]), " | ", @sprintf("%.6e", fd_err[4]), " |")
        println(io, "| GFDT+UNet | ", @sprintf("%.6e", gfdt_err[1]), " | ", @sprintf("%.6e", gfdt_err[2]), " | ", @sprintf("%.6e", gfdt_err[3]), " | ", @sprintf("%.6e", gfdt_err[4]), " |")
        println(io)
        println(io, "## Objective Trace")
        println(io)
        println(io, "- FD iterations: `", length(rows_fd), "`")
        if !isempty(rows_fd)
            println(io, "- FD first/last objective: `", @sprintf("%.6e", rows_fd[1].obj), "` -> `", @sprintf("%.6e", rows_fd[end].obj_next), "`")
        end
        println(io, "- GFDT+UNet iterations: `", length(rows_gfdt), "`")
        if !isempty(rows_gfdt)
            println(io, "- GFDT+UNet first/last objective: `", @sprintf("%.6e", rows_gfdt[1].obj), "` -> `", @sprintf("%.6e", rows_gfdt[end].obj_next), "`")
        end
    end
    return path
end

function main(args=ARGS)
    cli = parse_cli(args)
    run_dir = abspath(String(cli["run_dir"]))
    integration_toml = abspath(String(cli["integration_toml"]))
    out_dir = abspath(String(cli["output_dir"]))
    mkpath(out_dir)

    cfg = load_l96_config(integration_toml)
    θ_true = [cfg.F, cfg.h, cfg.c, cfg.b]
    θ0 = copy(THETA_INIT)

    best = pick_best_checkpoint(run_dir)
    checkpoint_path = best.checkpoint_path

    @info "Preparing calibration targets" dataset=cfg.dataset_path target_nsamples=TARGET_NSAMPLES target_start=TARGET_START_INDEX
    target_mean, _, wdiag = compute_target_stats(cfg)

    # Use a fixed initial state from observations for all model runs.
    init_tensor = load_observation_subset(cfg; nsamples=1, start_index=TARGET_START_INDEX)
    x0_init, y0_init = tensor_snapshot_to_xy(init_tensor, cfg.J)

    θ_fd, rows_fd = run_calibration_fd(θ0, θ_true, target_mean, wdiag, x0_init, y0_init, cfg;
                                       max_iters=FD_MAX_ITERS,
                                       damping=FD_DAMPING)
    θ_gfdt0 = best_theta_from_history(rows_fd, θ_fd)
    @info "Starting GFDT+UNet from best FD estimate" θ_gfdt0=θ_gfdt0
    θ_gfdt, rows_gfdt = run_calibration_gfdt_unet(θ_gfdt0, θ_true, target_mean, wdiag, x0_init, y0_init, cfg, checkpoint_path;
                                                  max_iters=GFDT_MAX_ITERS,
                                                  damping=GFDT_DAMPING)

    fd_csv = write_history_csv(joinpath(out_dir, "calibration_history_fd.csv"), rows_fd)
    gfdt_csv = write_history_csv(joinpath(out_dir, "calibration_history_gfdt_unet.csv"), rows_gfdt)

    summary_md = write_summary_markdown(joinpath(out_dir, "calibration_report.md"),
                                        θ_true,
                                        θ0,
                                        θ_gfdt0,
                                        θ_fd,
                                        θ_gfdt,
                                        rows_fd,
                                        rows_gfdt,
                                        checkpoint_path,
                                        cfg)

    summary_toml = joinpath(out_dir, "calibration_summary.toml")
    summary_doc = Dict(
        "true_parameters" => Dict("F" => θ_true[1], "h" => θ_true[2], "c" => θ_true[3], "b" => θ_true[4]),
        "initial_parameters" => Dict("F" => θ0[1], "h" => θ0[2], "c" => θ0[3], "b" => θ0[4]),
        "gfdt_initial_parameters" => Dict("F" => θ_gfdt0[1], "h" => θ_gfdt0[2], "c" => θ_gfdt0[3], "b" => θ_gfdt0[4]),
        "fd_final" => Dict("F" => θ_fd[1], "h" => θ_fd[2], "c" => θ_fd[3], "b" => θ_fd[4]),
        "gfdt_unet_final" => Dict("F" => θ_gfdt[1], "h" => θ_gfdt[2], "c" => θ_gfdt[3], "b" => θ_gfdt[4]),
        "fd_rel_l2_error" => norm((θ_fd .- θ_true) ./ θ_true) / sqrt(4),
        "gfdt_unet_rel_l2_error" => norm((θ_gfdt .- θ_true) ./ θ_true) / sqrt(4),
        "checkpoint_path" => abspath(checkpoint_path),
        "fd_history_csv" => abspath(fd_csv),
        "gfdt_history_csv" => abspath(gfdt_csv),
        "report_md" => abspath(summary_md),
    )
    open(summary_toml, "w") do io
        TOML.print(io, summary_doc)
    end

    @info "Calibration completed" out_dir=out_dir θ_true=θ_true θ_fd=θ_fd θ_gfdt=θ_gfdt
    println("Saved:")
    println("  - ", fd_csv)
    println("  - ", gfdt_csv)
    println("  - ", summary_md)
    println("  - ", summary_toml)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
