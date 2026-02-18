#!/usr/bin/env julia

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

include("l96_parameter_jacobians.jl")

using LinearAlgebra
using Plots
using Printf
using Random
using Statistics
using TOML

const FD_CAL_DEFAULT_OUTPUT_DIR = "scripts/L96/calibration_results/fd"
const FD_CAL_DEFAULT_INTEGRATION_TOML = "scripts/L96/observations/J10/integration_params.toml"

const TARGET_NSAMPLES = 300_000
const TARGET_START_INDEX = 50_001

const FD_VALIDATE_NSAMPLES = 60_000
const FD_VALIDATE_BURN = 2_500
const FD_VALIDATE_NREP = 4
const FD_VALIDATE_H_REL = 5e-3
const FD_VALIDATE_H_ABS = [1e-2, 1e-3, 1e-2, 1e-2]
const FD_VALIDATE_SEED_BASE = 441_700
const FD_CAL_STEP_SWEEP_FACTORS = [0.5, 1.0, 2.0]
const FD_SWEEP_NREP = 2

const CAL_MAX_ITERS = 8
const CAL_EVAL_NSAMPLES = 50_000
const CAL_EVAL_BURN = 2_000
const CAL_JAC_NSAMPLES = 30_000
const CAL_JAC_BURN = 1_500
const CAL_JAC_NREP = 2
const CAL_H_REL = 5e-3
const CAL_H_ABS = [1e-2, 1e-3, 1e-2, 1e-2]
const CAL_BASE_LAMBDA = 5e-2
const CAL_DAMPING = 1.0
const CAL_LS_FACTORS = [1.0, 0.5, 0.25, 0.125, 0.0625]
const CAL_LM_GROWTH = 10.0
const CAL_LM_MAX_TRIES = 4
const CAL_OBJ_IMPROVE_EPS = 1e-5
const CAL_PARAM_LOWER = [6.0, 0.3, 6.0, 6.0]
const CAL_PARAM_UPPER = [14.0, 2.0, 14.0, 14.0]

const FD_PARAM_NAMES = ["F", "h", "c", "b"]
const FD_OBS_NAMES = ["phi1_mean_x", "phi2_mean_x2", "phi3_mean_x_ybar", "phi4_mean_y2", "phi5_mean_x_xm1"]

function parse_cli(args::Vector{String})
    out = Dict{String,Any}(
        "integration_toml" => FD_CAL_DEFAULT_INTEGRATION_TOML,
        "output_dir" => FD_CAL_DEFAULT_OUTPUT_DIR,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--integration-toml"
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
    C == J + 1 || error("Channel mismatch in tensor_snapshot_to_xy")
    N >= 1 || error("Need at least one snapshot in tensor")
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
    C = J + 1
    x = copy(x0_init)
    y = copy(y0_init)
    ws = make_l96_workspace(K, J)
    rng = MersenneTwister(rng_seed)
    tensor = Array{Float64}(undef, K, C, nsamples)

    total = burn_snapshots + nsamples
    @inbounds for n in 1:total
        for _ in 1:cfg.save_every
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

function evaluate_observable_mean(θ::Vector{Float64},
                                  x0_init::Vector{Float64},
                                  y0_init::Matrix{Float64},
                                  cfg::L96Config;
                                  nsamples::Int,
                                  burn_snapshots::Int,
                                  rng_seed::Int)
    tensor = simulate_l96_tensor((θ[1], θ[2], θ[3], θ[4]), x0_init, y0_init, cfg;
                                 nsamples=nsamples,
                                 burn_snapshots=burn_snapshots,
                                 rng_seed=rng_seed)
    A = compute_global_observables(tensor)
    return vec(mean(A; dims=2)), tensor
end

@inline function weighted_objective(resid::Vector{Float64}, wdiag::Vector{Float64})
    return sqrt(mean((sqrt.(wdiag) .* resid) .^ 2))
end

function solve_weighted_step(J::Matrix{Float64},
                             resid::Vector{Float64},
                             wdiag::Vector{Float64},
                             λ::Float64)
    sqrtw = sqrt.(wdiag)
    Jw = J .* reshape(sqrtw, :, 1)
    rw = resid .* sqrtw
    M = Jw' * Jw
    @inbounds for i in 1:size(M, 1)
        M[i, i] += λ
    end
    rhs = Jw' * rw
    Δ = M \ rhs
    c = try
        cond(M)
    catch
        NaN
    end
    return Δ, c
end

function clamp_theta(θ::Vector{Float64})
    return clamp.(θ, CAL_PARAM_LOWER, CAL_PARAM_UPPER)
end

function fd_entry_stats(S_reps::Vector{Matrix{Float64}})
    nrep = length(S_reps)
    m, p = size(S_reps[1])
    μ = zeros(Float64, m, p)
    σ = zeros(Float64, m, p)
    se = zeros(Float64, m, p)
    @inbounds for i in 1:m, j in 1:p
        vals = [S[i, j] for S in S_reps]
        μ[i, j] = mean(vals)
        σ[i, j] = (nrep > 1 ? std(vals) : 0.0)
        se[i, j] = (nrep > 1 ? σ[i, j] / sqrt(nrep) : 0.0)
    end
    return μ, σ, se
end

function write_fd_matrix_with_uncertainty(path::AbstractString,
                                          S_mean::Matrix{Float64},
                                          S_std::Matrix{Float64},
                                          S_se::Matrix{Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "observable,param,mean,std,se")
        for i in 1:5, j in 1:4
            @printf(io, "%s,%s,%.12e,%.12e,%.12e\n",
                    FD_OBS_NAMES[i], FD_PARAM_NAMES[j], S_mean[i, j], S_std[i, j], S_se[i, j])
        end
    end
    return path
end

function write_fd_replicates(path::AbstractString,
                             S_reps::Vector{Matrix{Float64}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "replicate,observable,param,value")
        for (r, S) in enumerate(S_reps)
            for i in 1:5, j in 1:4
                @printf(io, "%d,%s,%s,%.12e\n", r, FD_OBS_NAMES[i], FD_PARAM_NAMES[j], S[i, j])
            end
        end
    end
    return path
end

function write_fd_step_sweep(path::AbstractString,
                             θ_true::Vector{Float64},
                             x0_init::Vector{Float64},
                             y0_init::Matrix{Float64},
                             cfg::L96Config)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "h_scale,observable,param,value")
        for hs in FD_CAL_STEP_SWEEP_FACTORS
            Sout = finite_difference_jacobian_l96((θ_true[1], θ_true[2], θ_true[3], θ_true[4]),
                                                  x0_init, y0_init, cfg;
                                                  h_rel=FD_VALIDATE_H_REL,
                                                  h_abs=collect(Float64.(FD_VALIDATE_H_ABS)),
                                                  h_scale=hs,
                                                  burn_snapshots=FD_VALIDATE_BURN,
                                                  nsamples=FD_VALIDATE_NSAMPLES,
                                                  n_rep=FD_SWEEP_NREP,
                                                  seed_base=FD_VALIDATE_SEED_BASE + round(Int, 10_000 * hs),
                                                  return_replicates=false)
            for i in 1:5, j in 1:4
                @printf(io, "%.6f,%s,%s,%.12e\n", hs, FD_OBS_NAMES[i], FD_PARAM_NAMES[j], Sout[i, j])
            end
        end
    end
    return path
end

function write_fd_latex_report(path::AbstractString,
                               θ_true::Vector{Float64},
                               cfg::L96Config,
                               S_mean::Matrix{Float64},
                               S_std::Matrix{Float64},
                               S_se::Matrix{Float64},
                               step_sweep_csv::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "\\documentclass[11pt]{article}")
        println(io, "\\usepackage[margin=1in]{geometry}")
        println(io, "\\usepackage{booktabs}")
        println(io, "\\usepackage{siunitx}")
        println(io, "\\begin{document}")
        println(io, "\\section*{L96 FD Jacobian Validation}")
        d = Char(36)
        println(io, "\\noindent True parameters ", d, "(F,h,c,b)", d, " = (", @sprintf("%.3f", θ_true[1]), ", ", @sprintf("%.3f", θ_true[2]), ", ", @sprintf("%.3f", θ_true[3]), ", ", @sprintf("%.3f", θ_true[4]), ").\\\\")
        println(io, "\\noindent Configuration: K=", cfg.K, ", J=", cfg.J, ", dt=", @sprintf("%.4f", cfg.dt), ", save\\_every=", cfg.save_every, ", process\\_noise\\_sigma=", @sprintf("%.4f", cfg.process_noise_sigma), ".\\\\")
        println(io, "\\noindent FD uncertainty estimated from replicate central differences.")
        println(io, "\\subsection*{Jacobian Entries with Uncertainty}")
        println(io, "\\begin{center}")
        println(io, "\\begin{tabular}{llrrr}")
        println(io, "\\toprule")
        println(io, "Observable & Parameter & Mean & Std & SE \\\\")
        println(io, "\\midrule")
        for i in 1:5, j in 1:4
            @printf(io, "%s & %s & %.6e & %.6e & %.6e \\\\\n",
                    FD_OBS_NAMES[i], FD_PARAM_NAMES[j], S_mean[i, j], S_std[i, j], S_se[i, j])
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{center}")
        println(io, "\\subsection*{Step-size Sweep}")
        println(io, "\\noindent The FD step-size sweep data are stored in:\\\\")
        println(io, "\\texttt{", replace(abspath(step_sweep_csv), "_" => "\\_"), "}")
        println(io, "\\end{document}")
    end
    return path
end

function write_calibration_history(path::AbstractString,
                                   rows::Vector{NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "iter,obj,obj_next,damping,lambda,cond,theta_F,theta_h,theta_c,theta_b,next_F,next_h,next_c,next_b")
        for r in rows
            @printf(io, "%d,%.12e,%.12e,%.6f,%.6e,%.6e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    r.iter, r.obj, r.obj_next, r.damping, r.lambda, r.cond,
                    r.theta[1], r.theta[2], r.theta[3], r.theta[4],
                    r.theta_next[1], r.theta_next[2], r.theta_next[3], r.theta_next[4])
        end
    end
    return path
end

function write_observable_history(path::AbstractString,
                                  θ_hist::Vector{Vector{Float64}},
                                  obs_hist::Vector{Vector{Float64}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "iter,F,h,c,b,phi1,phi2,phi3,phi4,phi5")
        for i in eachindex(θ_hist)
            θ = θ_hist[i]
            ϕ = obs_hist[i]
            @printf(io, "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    i - 1, θ[1], θ[2], θ[3], θ[4], ϕ[1], ϕ[2], ϕ[3], ϕ[4], ϕ[5])
        end
    end
    return path
end

function make_9panel_figure(path::AbstractString,
                            θ_hist::Vector{Vector{Float64}},
                            obs_hist::Vector{Vector{Float64}},
                            θ_true::Vector{Float64},
                            target_obs::Vector{Float64})
    mkpath(dirname(path))
    nit = length(θ_hist) - 1
    xiter = collect(0:nit)

    p = plot(layout=(3, 3), size=(1800, 1200), legend=:bottomright)

    for j in 1:4
        y = [θ[j] for θ in θ_hist]
        plot!(p[j], xiter, y; label="estimate", lw=2.5, marker=:circle, color=:dodgerblue3)
        hline!(p[j], [θ_true[j]]; label="target", lw=2.0, ls=:dash, color=:black)
        xlabel!(p[j], "Newton iteration")
        ylabel!(p[j], FD_PARAM_NAMES[j])
        title!(p[j], "Parameter " * FD_PARAM_NAMES[j])
    end

    panel_to_obs = Dict(5 => 1, 6 => 2, 7 => 3, 8 => 4, 9 => 5)
    for panel in 5:9
        iobs = panel_to_obs[panel]
        y = [ϕ[iobs] for ϕ in obs_hist]
        plot!(p[panel], xiter, y; label="predicted", lw=2.5, marker=:circle, color=:orangered3)
        hline!(p[panel], [target_obs[iobs]]; label="target", lw=2.0, ls=:dash, color=:black)
        xlabel!(p[panel], "Newton iteration")
        ylabel!(p[panel], FD_OBS_NAMES[iobs])
        title!(p[panel], "Observable " * string(iobs))
    end

    savefig(p, path)
    return path
end

function compute_target_stats(cfg::L96Config)
    tensor = load_observation_subset(cfg; nsamples=TARGET_NSAMPLES, start_index=TARGET_START_INDEX)
    A = compute_global_observables(tensor)
    target_mean = vec(mean(A; dims=2))
    target_std = vec(std(A; dims=2))
    floor_std = max(quantile(target_std, 0.2), 1e-8)
    wdiag = 1.0 ./ (target_std .^ 2 .+ floor_std ^ 2)
    return target_mean, target_std, wdiag
end

function run_fd_newton_calibration(θ0::Vector{Float64},
                                   θ_true::Vector{Float64},
                                   target_obs::Vector{Float64},
                                   wdiag::Vector{Float64},
                                   x0_init::Vector{Float64},
                                   y0_init::Matrix{Float64},
                                   cfg::L96Config)
    θ = clamp_theta(copy(θ0))
    rows = NamedTuple[]
    θ_hist = Vector{Vector{Float64}}()
    obs_hist = Vector{Vector{Float64}}()
    obj_hist = Float64[]

    obs0, tensor0 = evaluate_observable_mean(θ, x0_init, y0_init, cfg;
                                             nsamples=CAL_EVAL_NSAMPLES,
                                             burn_snapshots=CAL_EVAL_BURN,
                                             rng_seed=810_001)
    push!(θ_hist, copy(θ))
    push!(obs_hist, copy(obs0))
    push!(obj_hist, weighted_objective(obs0 .- target_obs, wdiag))

    converged = false
    for it in 1:CAL_MAX_ITERS
        obs = obs_hist[end]
        resid = obs .- target_obs
        obj = obj_hist[end]

        xfd, yfd = tensor_snapshot_to_xy(tensor0, cfg.J)
        Jout = finite_difference_jacobian_l96((θ[1], θ[2], θ[3], θ[4]), xfd, yfd, cfg;
                                              h_rel=CAL_H_REL,
                                              h_abs=collect(Float64.(CAL_H_ABS)),
                                              h_scale=1.0,
                                              burn_snapshots=CAL_JAC_BURN,
                                              nsamples=CAL_JAC_NSAMPLES,
                                              n_rep=CAL_JAC_NREP,
                                              seed_base=510_000 + 7_000 * it,
                                              return_replicates=true)
        J = Jout.S

        accepted = false
        θ_next = copy(θ)
        obj_next = obj
        obs_next = copy(obs)
        tensor_next = tensor0
        condM = NaN
        used_damping = 0.0
        used_lambda = CAL_BASE_LAMBDA

        λ = CAL_BASE_LAMBDA
        for _ in 1:CAL_LM_MAX_TRIES
            Δ, condM = solve_weighted_step(J, resid, wdiag, λ)
            for α in CAL_LS_FACTORS
                θ_try = clamp_theta(θ .- (CAL_DAMPING * α) .* Δ)
                obs_try, tensor_try = evaluate_observable_mean(θ_try, x0_init, y0_init, cfg;
                                                               nsamples=CAL_EVAL_NSAMPLES,
                                                               burn_snapshots=CAL_EVAL_BURN,
                                                               rng_seed=920_000 + it)
                obj_try = weighted_objective(obs_try .- target_obs, wdiag)
                if obj_try <= obj - CAL_OBJ_IMPROVE_EPS * max(obj, 1e-3)
                    θ_next = θ_try
                    obs_next = obs_try
                    tensor_next = tensor_try
                    obj_next = obj_try
                    used_damping = α
                    used_lambda = λ
                    accepted = true
                    break
                end
            end
            accepted && break
            λ *= CAL_LM_GROWTH
        end

        push!(rows, (
            iter=it,
            obj=obj,
            obj_next=obj_next,
            damping=used_damping,
            lambda=used_lambda,
            cond=condM,
            theta=copy(θ),
            theta_next=copy(θ_next),
        ))

        θ = θ_next
        tensor0 = tensor_next
        push!(θ_hist, copy(θ))
        push!(obs_hist, copy(obs_next))
        push!(obj_hist, obj_next)

        rel_param_err = norm((θ .- θ_true) ./ θ_true) / sqrt(length(θ_true))
        @info "FD Newton iteration" iter=it obj=obj obj_next=obj_next θ=θ rel_param_err=rel_param_err damping=used_damping lambda=used_lambda cond=condM

        if rel_param_err < 2e-2 || obj_next < 5e-3
            converged = true
            break
        end
    end

    return (theta_final=θ, rows=rows, theta_hist=θ_hist, obs_hist=obs_hist, obj_hist=obj_hist, converged=converged)
end

function main(args=ARGS)
    cli = parse_cli(args)
    integration_toml = abspath(String(cli["integration_toml"]))
    out_dir = abspath(String(cli["output_dir"]))
    mkpath(out_dir)

    validate_global_observable_impl!()
    validate_conjugate_implementation!()
    validate_gfdt_sign_ou!()

    cfg = load_l96_config(integration_toml)
    θ_true = [cfg.F, cfg.h, cfg.c, cfg.b]
    θ0 = 1.15 .* θ_true
    θ0 = clamp_theta(θ0)

    @info "Computing target statistics from observation dataset" dataset=cfg.dataset_path nsamples=TARGET_NSAMPLES start_index=TARGET_START_INDEX
    target_obs, target_std, wdiag = compute_target_stats(cfg)

    init_tensor = load_observation_subset(cfg; nsamples=1, start_index=TARGET_START_INDEX)
    x0_init, y0_init = tensor_snapshot_to_xy(init_tensor, cfg.J)

    @info "Validating FD Jacobian at true parameters" nsamples=FD_VALIDATE_NSAMPLES n_rep=FD_VALIDATE_NREP
    fd_true = finite_difference_jacobian_l96((θ_true[1], θ_true[2], θ_true[3], θ_true[4]),
                                             x0_init, y0_init, cfg;
                                             h_rel=FD_VALIDATE_H_REL,
                                             h_abs=collect(Float64.(FD_VALIDATE_H_ABS)),
                                             h_scale=1.0,
                                             burn_snapshots=FD_VALIDATE_BURN,
                                             nsamples=FD_VALIDATE_NSAMPLES,
                                             n_rep=FD_VALIDATE_NREP,
                                             seed_base=FD_VALIDATE_SEED_BASE,
                                             return_replicates=true)
    S_mean, S_std, S_se = fd_entry_stats(fd_true.S_reps)

    fd_mat_csv = write_matrix_csv(joinpath(out_dir, "fd_jacobian_matrix_global5.csv"), S_mean)
    fd_unc_csv = write_fd_matrix_with_uncertainty(joinpath(out_dir, "fd_jacobian_uncertainty_global5.csv"), S_mean, S_std, S_se)
    fd_rep_csv = write_fd_replicates(joinpath(out_dir, "fd_jacobian_replicates_global5.csv"), fd_true.S_reps)
    fd_det_csv = write_fd_determinant_stats(joinpath(out_dir, "fd_jacobian_determinant_stats_global5.csv"), S_mean, fd_true.S_reps)
    fd_sweep_csv = write_fd_step_sweep(joinpath(out_dir, "fd_jacobian_step_sweep_global5.csv"), θ_true, x0_init, y0_init, cfg)
    fd_tex = write_fd_latex_report(joinpath(out_dir, "fd_jacobian_report.tex"), θ_true, cfg, S_mean, S_std, S_se, fd_sweep_csv)

    @info "Running FD-only Newton calibration" initial_theta=θ0 true_theta=θ_true
    cal = run_fd_newton_calibration(θ0, θ_true, target_obs, wdiag, x0_init, y0_init, cfg)

    cal_hist_csv = write_calibration_history(joinpath(out_dir, "fd_calibration_history.csv"), cal.rows)
    cal_obs_csv = write_observable_history(joinpath(out_dir, "fd_observable_history.csv"), cal.theta_hist, cal.obs_hist)
    cal_fig = make_9panel_figure(joinpath(out_dir, "fd_calibration_9panel.png"), cal.theta_hist, cal.obs_hist, θ_true, target_obs)

    rel_param_err = norm((cal.theta_final .- θ_true) ./ θ_true) / sqrt(length(θ_true))
    summary_toml = joinpath(out_dir, "fd_calibration_summary.toml")
    summary_doc = Dict(
        "true_parameters" => Dict("F" => θ_true[1], "h" => θ_true[2], "c" => θ_true[3], "b" => θ_true[4]),
        "initial_parameters" => Dict("F" => θ0[1], "h" => θ0[2], "c" => θ0[3], "b" => θ0[4]),
        "final_parameters" => Dict("F" => cal.theta_final[1], "h" => cal.theta_final[2], "c" => cal.theta_final[3], "b" => cal.theta_final[4]),
        "relative_param_l2_error" => rel_param_err,
        "converged" => cal.converged,
        "num_iterations" => length(cal.theta_hist) - 1,
        "objective_initial" => cal.obj_hist[1],
        "objective_final" => cal.obj_hist[end],
        "target_observables" => Dict(FD_OBS_NAMES[i] => target_obs[i] for i in 1:5),
        "target_std" => Dict(FD_OBS_NAMES[i] => target_std[i] for i in 1:5),
        "fd_jacobian_matrix_csv" => abspath(fd_mat_csv),
        "fd_jacobian_uncertainty_csv" => abspath(fd_unc_csv),
        "fd_jacobian_replicates_csv" => abspath(fd_rep_csv),
        "fd_jacobian_determinant_csv" => abspath(fd_det_csv),
        "fd_jacobian_step_sweep_csv" => abspath(fd_sweep_csv),
        "fd_jacobian_report_tex" => abspath(fd_tex),
        "calibration_history_csv" => abspath(cal_hist_csv),
        "observable_history_csv" => abspath(cal_obs_csv),
        "calibration_figure_png" => abspath(cal_fig),
    )
    open(summary_toml, "w") do io
        TOML.print(io, summary_doc)
    end

    println("Saved:")
    println("  - ", fd_mat_csv)
    println("  - ", fd_unc_csv)
    println("  - ", fd_rep_csv)
    println("  - ", fd_det_csv)
    println("  - ", fd_sweep_csv)
    println("  - ", fd_tex)
    println("  - ", cal_hist_csv)
    println("  - ", cal_obs_csv)
    println("  - ", cal_fig)
    println("  - ", summary_toml)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
