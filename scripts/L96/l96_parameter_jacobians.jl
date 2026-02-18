#!/usr/bin/env julia

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using FFTW
using Flux
using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML
using ProgressMeter
using ScoreUNet1D

"""
Evaluate L96 parameter Jacobians with three methods:
1) finite differences (FD),
2) GFDT + quasi-Gaussian linear score,
3) GFDT + U-Net DSM score (best checkpoint from run_021).

Outputs:
- Jacobian matrices for each method (`5 x 4`, global observables):
  `.../jacobian_*_matrix_global5.csv`
- Comparison metrics:
  `.../jacobian_comparison_metrics_global5.csv`
- LaTeX table:
  `.../jacobian_table_global5.tex`
- Markdown report with key discrepancy diagnostics:
  `.../jacobian_report_global5.md`
"""

const DEFAULT_RUN_DIR = "scripts/L96/runs_J10/run_021"
const DEFAULT_OBS_INTEGRATION_TOML = "scripts/L96/observations/J10/integration_params.toml"
const DEFAULT_OUTPUT_DIR = "scripts/L96/calibration_outputs/run_021_jacobians_global5"

# GFDT trajectory settings (unperturbed trajectory from observation repository).
const GFDT_NSAMPLES = 300_000
const GFDT_START_INDEX = 50_001
const GFDT_TMAX = 2.0
const GFDT_MEAN_CENTER = true
const GFDT_TMAX_SCAN = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0]
const GFDT_TMAX_SELECT_MODE = "fixed"  # one of: "fixed", "det_plateau"

# Optional UNet score-gain correction using only UNet statistics (no FD).
# "mean_zero_g" solves E[G_i]=0 in least squares over i in {F,h,c,b}.
const UNET_GAIN_MODE = "none"   # one of: "none", "mean_zero_g"
const UNET_GAIN_RIDGE = 1e-10

# U-Net score evaluation.
const SCORE_BATCH_SIZE = 2048
const SCORE_DEVICE_PREF = "GPU:0"
const SCORE_FORWARD_MODE = "test"  # one of: "test", "train"

# FD settings.
const FD_NSAMPLES = 120_000
const FD_BURN_SNAPSHOTS = 5_000
const FD_N_REPS = 2
const FD_H_REL = 5e-3
const FD_H_ABS = [1e-2, 1e-3, 1e-2, 1e-2]  # [F,h,c,b]
const FD_SEED_BASE = 240_117
const FD_INIT_SNAPSHOT_INDEX = 1
const FD_STEP_SWEEP_FACTORS = [1.0]

const PARAM_NAMES = ["F", "h", "c", "b"]
const OBS_GLOBAL_NAMES = ["phi1_mean_x", "phi2_mean_x2", "phi3_mean_x_ybar", "phi4_mean_y2", "phi5_mean_x_xm1"]
const OBS_LATEX_NAMES = ["\\phi_1=\\langle X\\rangle", "\\phi_2=\\langle X^2\\rangle", "\\phi_3=\\langle X\\bar{Y}\\rangle", "\\phi_4=\\langle Y^2\\rangle", "\\phi_5=\\langle X_k X_{k-1}\\rangle"]

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
    dataset_path::String
    dataset_key::String
end

mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

function parse_cli(args::Vector{String})
    out = Dict{String,Any}(
        "run_dir" => DEFAULT_RUN_DIR,
        "checkpoint_path" => "",
        "integration_toml" => DEFAULT_OBS_INTEGRATION_TOML,
        "output_dir" => DEFAULT_OUTPUT_DIR,
        "gfdt_tmax" => GFDT_TMAX,
        "gfdt_nsamples" => GFDT_NSAMPLES,
        "gfdt_start_index" => GFDT_START_INDEX,
        "fd_step_scales" => collect(Float64.(FD_STEP_SWEEP_FACTORS)),
        "fd_nsamples" => FD_NSAMPLES,
        "fd_burn_snapshots" => FD_BURN_SNAPSHOTS,
        "fd_n_reps" => FD_N_REPS,
        "fd_h_rel" => FD_H_REL,
        "fd_h_abs" => collect(Float64.(FD_H_ABS)),
        "fd_seed_base" => FD_SEED_BASE,
        "fd_matrix_csv" => "",
        "gfdt_tmax_select_mode" => GFDT_TMAX_SELECT_MODE,
        "unet_gain_mode" => UNET_GAIN_MODE,
        "unet_gain_ridge" => UNET_GAIN_RIDGE,
        "unet_forward_mode" => SCORE_FORWARD_MODE,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--run-dir"
            i == length(args) && error("--run-dir expects a value")
            out["run_dir"] = args[i + 1]
            i += 2
        elseif a == "--checkpoint-path"
            i == length(args) && error("--checkpoint-path expects a value")
            out["checkpoint_path"] = args[i + 1]
            i += 2
        elseif a == "--integration-toml"
            i == length(args) && error("--integration-toml expects a value")
            out["integration_toml"] = args[i + 1]
            i += 2
        elseif a == "--output-dir"
            i == length(args) && error("--output-dir expects a value")
            out["output_dir"] = args[i + 1]
            i += 2
        elseif a == "--gfdt-tmax"
            i == length(args) && error("--gfdt-tmax expects a value")
            out["gfdt_tmax"] = parse(Float64, args[i + 1])
            i += 2
        elseif a == "--gfdt-nsamples"
            i == length(args) && error("--gfdt-nsamples expects a value")
            out["gfdt_nsamples"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--gfdt-start-index"
            i == length(args) && error("--gfdt-start-index expects a value")
            out["gfdt_start_index"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--fd-step-scales"
            i == length(args) && error("--fd-step-scales expects comma-separated values")
            vals = Float64[]
            for tok in split(args[i + 1], ",")
                s = strip(tok)
                isempty(s) && continue
                push!(vals, parse(Float64, s))
            end
            isempty(vals) && error("--fd-step-scales parsed no valid values")
            out["fd_step_scales"] = vals
            i += 2
        elseif a == "--fd-nsamples"
            i == length(args) && error("--fd-nsamples expects an integer")
            out["fd_nsamples"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--fd-burn-snapshots"
            i == length(args) && error("--fd-burn-snapshots expects an integer")
            out["fd_burn_snapshots"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--fd-n-reps"
            i == length(args) && error("--fd-n-reps expects an integer")
            out["fd_n_reps"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--fd-h-rel"
            i == length(args) && error("--fd-h-rel expects a float")
            out["fd_h_rel"] = parse(Float64, args[i + 1])
            i += 2
        elseif a == "--fd-h-abs"
            i == length(args) && error("--fd-h-abs expects 4 comma-separated floats")
            vals = Float64[]
            for tok in split(args[i + 1], ",")
                s = strip(tok)
                isempty(s) && continue
                push!(vals, parse(Float64, s))
            end
            length(vals) == 4 || error("--fd-h-abs expects exactly 4 values for [F,h,c,b]")
            out["fd_h_abs"] = vals
            i += 2
        elseif a == "--fd-seed-base"
            i == length(args) && error("--fd-seed-base expects an integer")
            out["fd_seed_base"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--fd-matrix-csv"
            i == length(args) && error("--fd-matrix-csv expects a value")
            out["fd_matrix_csv"] = args[i + 1]
            i += 2
        elseif a == "--gfdt-tmax-select-mode"
            i == length(args) && error("--gfdt-tmax-select-mode expects one of: fixed|det_plateau")
            mode = lowercase(strip(args[i + 1]))
            mode in ("fixed", "det_plateau") || error("Invalid --gfdt-tmax-select-mode '$mode' (expected fixed|det_plateau)")
            out["gfdt_tmax_select_mode"] = mode
            i += 2
        elseif a == "--unet-gain-mode"
            i == length(args) && error("--unet-gain-mode expects one of: none|mean_zero_g")
            mode = lowercase(strip(args[i + 1]))
            mode in ("none", "mean_zero_g") || error("Invalid --unet-gain-mode '$mode' (expected none|mean_zero_g)")
            out["unet_gain_mode"] = mode
            i += 2
        elseif a == "--unet-gain-ridge"
            i == length(args) && error("--unet-gain-ridge expects a numeric value")
            out["unet_gain_ridge"] = parse(Float64, args[i + 1])
            i += 2
        elseif a == "--unet-forward-mode"
            i == length(args) && error("--unet-forward-mode expects one of: test|train")
            mode = lowercase(strip(args[i + 1]))
            mode in ("test", "train") || error("Invalid --unet-forward-mode '$mode' (expected test|train)")
            out["unet_forward_mode"] = mode
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return out
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
        # Fallback for older naming formats.
        best_name3 = @sprintf("score_model_epoch_%03d.bson", best_epoch)
        best_path3 = joinpath(run_dir, "model", best_name3)
        isfile(best_path3) || error("Best checkpoint not found for epoch $best_epoch in $run_dir/model")
        best_path = best_path3
    end
    return (best_epoch=best_epoch, checkpoint_path=best_path)
end

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
        path,
        key,
    )
end

function load_observation_subset(cfg::L96Config; nsamples::Int, start_index::Int=1)
    path = cfg.dataset_path
    isfile(path) || error("Observation dataset not found: $path")

    raw = h5open(path, "r") do h5
        haskey(h5, cfg.dataset_key) || error("Dataset key $(cfg.dataset_key) not found in $path")
        ds = h5[cfg.dataset_key]
        n_total = size(ds, 1)
        start_index >= 1 || error("start_index must be >= 1")
        stop_index = min(start_index + nsamples - 1, n_total)
        stop_index >= start_index || error("Invalid subset bounds")
        ds[start_index:stop_index, :, :]  # (N, C, K)
    end

    tensor = permutedims(raw, (3, 2, 1))  # (K, C, N)
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

compute_observables(tensor::Array{Float64,3}) = compute_global_observables(tensor)

function validate_global_observable_impl!()
    # Tiny deterministic tensor sanity check:
    # K=3, J=2, C=3, N=1, channels [x,y1,y2].
    tensor = zeros(Float64, 3, 3, 1)
    tensor[:, 1, 1] .= [1.0, 2.0, 3.0]
    tensor[:, 2, 1] .= [2.0, 0.0, 1.0]
    tensor[:, 3, 1] .= [0.0, 2.0, 1.0]
    A = compute_global_observables(tensor)
    ybar = [(2.0 + 0.0) / 2, (0.0 + 2.0) / 2, (1.0 + 1.0) / 2]
    y2bar = [(2.0^2 + 0.0^2) / 2, (0.0^2 + 2.0^2) / 2, (1.0^2 + 1.0^2) / 2]
    expected = zeros(Float64, 5)
    expected[1] = mean([1.0, 2.0, 3.0])
    expected[2] = mean([1.0^2, 2.0^2, 3.0^2])
    expected[3] = mean([1.0 * ybar[1], 2.0 * ybar[2], 3.0 * ybar[3]])
    expected[4] = mean(y2bar)
    expected[5] = mean([1.0 * 3.0, 2.0 * 1.0, 3.0 * 2.0])  # periodic k-1
    ok = all(isapprox.(A[:, 1], expected; atol=1e-12, rtol=1e-12))
    ok || error("Global observable implementation check failed. got=$(A[:, 1]) expected=$expected")
    return nothing
end

function compute_G_from_score!(G::Array{Float64,2},
                               tensor::Array{Float64,3},
                               score_phys::Array{Float64,3},
                               θ::NTuple{4,Float64},
                               out_start::Int;
                               Gx::Union{Nothing,Array{Float64,2}}=nothing,
                               Gy::Union{Nothing,Array{Float64,2}}=nothing,
                               Gconst::Union{Nothing,Array{Float64,2}}=nothing)
    K, C, B = size(tensor)
    J = C - 1
    F, h, c, b = θ
    _ = F

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

            ybar = 0.0
            for j in 1:J
                ybar += tensor[k, j + 1, ib]
            end
            ybar /= J
            sum_ybar_sx += ybar * sx

            for j in 1:J
                jm1 = (j == 1) ? J : j - 1
                jp1 = (j == J) ? 1 : j + 1
                jp2 = (j >= J - 1) ? j + 2 - J : j + 2

                yjm1 = tensor[k, jm1 + 1, ib]
                yjp1 = tensor[k, jp1 + 1, ib]
                yjp2 = tensor[k, jp2 + 1, ib]
                yj = tensor[k, j + 1, ib]
                sy = score_phys[k, j + 1, ib]

                adv = yjp1 * (yjp2 - yjm1)
                sum_x_sy += xk * sy
                sum_uc_sy += (-b * adv - yj + (h / b) * xk) * sy
                sum_adv_sy += adv * sy
            end
        end

        # GFDT conjugates for additive-noise L96.
        # These expressions are matched to the implemented drift in `l96_two_scale_drift!`
        # (which uses coupling scale h*c/b and fast advection scale c*b).
        #   G = -(div(∂θ f) + (∂θ f)·s)
        # with ybar = (1/J)∑_j y_j.
        cj_over_b = c / b
        hj_over_b = h / b
        hcj_over_b2 = (h * c) / (b * b)

        n = out_start + ib - 1

        g1x = gF
        g1y = 0.0
        g1c = 0.0

        g2x = (J * cj_over_b) * sum_ybar_sx
        g2y = -cj_over_b * sum_x_sy
        g2c = 0.0

        g3x = (J * hj_over_b) * sum_ybar_sx
        g3y = -sum_uc_sy
        g3c = K * J

        g4x = -(J * hcj_over_b2) * sum_ybar_sx
        g4y = c * sum_adv_sy + hcj_over_b2 * sum_x_sy
        g4c = 0.0

        G[1, n] = g1x + g1y + g1c
        G[2, n] = g2x + g2y + g2c
        G[3, n] = g3x + g3y + g3c
        G[4, n] = g4x + g4y + g4c

        if Gx !== nothing
            Gx[1, n] = g1x
            Gx[2, n] = g2x
            Gx[3, n] = g3x
            Gx[4, n] = g4x
        end
        if Gy !== nothing
            Gy[1, n] = g1y
            Gy[2, n] = g2y
            Gy[3, n] = g3y
            Gy[4, n] = g4y
        end
        if Gconst !== nothing
            Gconst[1, n] = g1c
            Gconst[2, n] = g2c
            Gconst[3, n] = g3c
            Gconst[4, n] = g4c
        end
    end
    return nothing
end

function select_eval_device(preference::AbstractString)
    try
        d = select_device(preference)
        activate_device!(d)
        return (d, preference)
    catch err
        @warn "Requested device unavailable for score inference; falling back to CPU" requested=preference error=sprint(showerror, err)
        d = CPUDevice()
        activate_device!(d)
        return (d, "CPU")
    end
end

function compute_G_unet(tensor::Array{Float64,3},
                        checkpoint_path::AbstractString,
                        θ::NTuple{4,Float64};
                        batch_size::Int=SCORE_BATCH_SIZE,
                        device_pref::AbstractString=SCORE_DEVICE_PREF,
                        forward_mode::AbstractString=SCORE_FORWARD_MODE,
                        return_components::Bool=false)
    contents = BSON.load(checkpoint_path)
    haskey(contents, :model) || error("Checkpoint missing :model ($checkpoint_path)")
    haskey(contents, :stats) || error("Checkpoint missing :stats ($checkpoint_path)")
    haskey(contents, :trainer_cfg) || error("Checkpoint missing :trainer_cfg ($checkpoint_path)")

    model = contents[:model]
    stats = contents[:stats]
    trainer_cfg = contents[:trainer_cfg]
    sigma_train = Float32(trainer_cfg.sigma)

    mean_lc = Float32.(permutedims(stats.mean, (2, 1)))  # (K, C)
    std_lc = Float32.(permutedims(stats.std, (2, 1)))    # (K, C)
    std_lc64 = Float64.(std_lc)

    K, C, N = size(tensor)
    G = zeros(Float64, 4, N)
    Gx = return_components ? zeros(Float64, 4, N) : nothing
    Gy = return_components ? zeros(Float64, 4, N) : nothing
    Gconst = return_components ? zeros(Float64, 4, N) : nothing

    device, device_name = select_eval_device(device_pref)
    model_dev = move_model(model, device)
    if lowercase(forward_mode) == "train"
        Flux.trainmode!(model_dev)
    else
        Flux.testmode!(model_dev)
    end

    @info "Computing UNet-score GFDT conjugates" checkpoint=checkpoint_path sigma_train=sigma_train device=device_name batches=cld(N, batch_size) forward_mode=forward_mode
    @showprogress "UNet score batches..." for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        idx = start:stop

        batch_phys = @view tensor[:, :, idx]  # Float64
        batch_norm_f32 = Array{Float32,3}(undef, K, C, b)
        @inbounds for ib in 1:b, c in 1:C, k in 1:K
            batch_norm_f32[k, c, ib] = (Float32(batch_phys[k, c, ib]) - mean_lc[k, c]) / std_lc[k, c]
        end

        score_norm = if is_gpu(device)
            dev_batch = move_array(batch_norm_f32, device)
            Array(score_from_model(model_dev, dev_batch, sigma_train))
        else
            score_from_model(model_dev, batch_norm_f32, sigma_train)
        end

        score_phys = Array{Float64,3}(undef, K, C, b)
        @inbounds for ib in 1:b, c in 1:C, k in 1:K
            score_phys[k, c, ib] = Float64(score_norm[k, c, ib]) / std_lc64[k, c]
        end

        compute_G_from_score!(G, Array(batch_phys), score_phys, θ, start; Gx=Gx, Gy=Gy, Gconst=Gconst)
    end
    if return_components
        return (G=G, Gx=Gx, Gy=Gy, Gconst=Gconst)
    end
    return G
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

function compute_G_gaussian(tensor::Array{Float64,3}, θ::NTuple{4,Float64}; batch_size::Int=2000)
    K, C, N = size(tensor)
    D = K * C
    Xflat = reshape(tensor, D, N)
    μ = vec(mean(Xflat; dims=2))
    Xc = Xflat .- μ
    Cmat = (Xc * transpose(Xc)) / max(N - 1, 1)
    Cinv, jitter_used = cholesky_inverse_spd(Matrix(Cmat))
    @info "Gaussian score covariance inversion complete" dim=D jitter=jitter_used

    G = zeros(Float64, 4, N)
    @showprogress "Gaussian-score batches..." for start in 1:batch_size:N
        stop = min(start + batch_size - 1, N)
        b = stop - start + 1
        idx = start:stop

        Xchunk = @view Xflat[:, idx]
        Xchunk_c = Xchunk .- μ
        score_flat = -(Cinv * Xchunk_c)  # D×b
        score_tensor = reshape(score_flat, K, C, b)
        compute_G_from_score!(G, Array(@view tensor[:, :, idx]), score_tensor, θ, start)
    end
    return G
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

function build_gfdt_jacobian(A::Array{Float64,2},
                             G::Array{Float64,2},
                             Δt_obs::Float64,
                             Tmax::Float64;
                             mean_center::Bool=true)
    m, N = size(A)
    p, N2 = size(G)
    N == N2 || error("A and G length mismatch")
    Kmax = min(Int(floor(Tmax / Δt_obs)), N - 1)

    Ause = mean_center ? (A .- mean(A; dims=2)) : A
    Guse = mean_center ? (G .- mean(G; dims=2)) : G

    S = zeros(Float64, m, p)
    @info "Building GFDT Jacobian from correlations" observables=m parameters=p N=N Kmax=Kmax Δt_obs=Δt_obs Tmax=Tmax mean_center=mean_center

    @showprogress "GFDT correlations..." for i in 1:m
        ai = vec(@view Ause[i, :])
        for j in 1:p
            gj = vec(@view Guse[j, :])
            cpos = xcorr_one_sided_unbiased_fft(ai, gj, Kmax)
            S[i, j] = Δt_obs * sum(cpos)
        end
    end
    return S
end

function validate_gfdt_sign_ou!()
    rng = MersenneTwister(90_111)
    N = 120_000
    burn = 20_000
    Δt = 0.01
    λ = 1.0
    ρ = exp(-λ * Δt)
    σstat = 1.0
    noise_std = sqrt(σstat^2 * (1 - ρ^2))

    x_full = zeros(Float64, N + burn)
    for n in 2:(N + burn)
        x_full[n] = ρ * x_full[n - 1] + noise_std * randn(rng)
    end
    x = @view x_full[(burn + 1):end]

    # For dX = (-X + F)dt + sqrt(2)dW, score at F=0 is s=-x and hence G_F=-s=x.
    A = reshape(copy(x), 1, N)
    G = reshape(copy(x), 1, N)
    Tmax = 5.0
    Kmax = Int(floor(Tmax / Δt))
    S = build_gfdt_jacobian(A, G, Δt, Tmax; mean_center=true)
    target = Δt * sum((ρ^k for k in 0:Kmax))
    err = abs(S[1, 1] - target)
    err <= 0.15 || error("GFDT sign sanity check failed (OU test): got=$(S[1,1]) target≈$target")
    return nothing
end

function l96_two_scale_drift!(dx::AbstractVector{Float64},
                              dy::AbstractMatrix{Float64},
                              x::AbstractVector{Float64},
                              y::AbstractMatrix{Float64},
                              θ::NTuple{4,Float64})
    K = length(x)
    J = size(y, 1)
    F, h, c, b = θ
    coupling_scale = h * c / b

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
            jm1 = mod1idx(j - 1, J)
            jp1 = mod1idx(j + 1, J)
            jp2 = mod1idx(j + 2, J)
            dy[j, k] = c * b * y[jp1, k] * (y[jm1, k] - y[jp2, k]) - c * y[j, k] + xk_term
        end
    end
    return nothing
end

function rk4_step_l96!(x::Vector{Float64},
                       y::Matrix{Float64},
                       dt::Float64,
                       ws::NamedTuple,
                       θ::NTuple{4,Float64})
    l96_two_scale_drift!(ws.dx1, ws.dy1, x, y, θ)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    l96_two_scale_drift!(ws.dx2, ws.dy2, ws.xtmp, ws.ytmp, θ)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    l96_two_scale_drift!(ws.dx3, ws.dy3, ws.xtmp, ws.ytmp, θ)

    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    l96_two_scale_drift!(ws.dx4, ws.dy4, ws.xtmp, ws.ytmp, θ)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    return nothing
end

function add_process_noise!(x::Vector{Float64},
                            y::Matrix{Float64},
                            rng::AbstractRNG,
                            sigma::Float64,
                            dt::Float64)
    sigma_step = sigma * sqrt(dt)
    sigma_step == 0.0 && return nothing
    @inbounds begin
        for k in eachindex(x)
            x[k] += sigma_step * randn(rng)
        end
        for idx in eachindex(y)
            y[idx] += sigma_step * randn(rng)
        end
    end
    return nothing
end

function make_l96_workspace(K::Int, J::Int)
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
    )
end

function l96_drift_state_vector(z::Vector{Float64},
                                θ::NTuple{4,Float64},
                                K::Int,
                                J::Int)
    length(z) == K + J * K || error("State length mismatch in drift helper")
    x = copy(@view z[1:K])
    y = reshape(copy(@view z[K + 1:end]), J, K)
    dx = zeros(Float64, K)
    dy = zeros(Float64, J, K)
    l96_two_scale_drift!(dx, dy, x, y, θ)
    return vcat(dx, vec(dy))
end

function validate_conjugate_implementation!()
    rng = MersenneTwister(71_023)
    K = 4
    J = 3
    C = J + 1
    θ = (10.0, 1.0, 10.0, 10.0)

    x = randn(rng, K)
    y = randn(rng, J, K)
    sx = randn(rng, K)
    sy = randn(rng, J, K)

    tensor = zeros(Float64, K, C, 1)
    score = zeros(Float64, K, C, 1)
    @inbounds for k in 1:K
        tensor[k, 1, 1] = x[k]
        score[k, 1, 1] = sx[k]
        for j in 1:J
            tensor[k, j + 1, 1] = y[j, k]
            score[k, j + 1, 1] = sy[j, k]
        end
    end

    G_formula = zeros(Float64, 4, 1)
    compute_G_from_score!(G_formula, tensor, score, θ, 1)
    G_formula_vec = vec(@view G_formula[:, 1])

    z = vcat(x, vec(y))
    s_vec = vcat(sx, vec(sy))
    z_eps = 1e-6
    G_numeric = zeros(Float64, 4)
    for ip in 1:4
        δp = 1e-6 * max(abs(θ[ip]), 1.0)
        θp = ntuple(i -> i == ip ? θ[i] + δp : θ[i], 4)
        θm = ntuple(i -> i == ip ? θ[i] - δp : θ[i], 4)

        function u_of(zz::Vector{Float64})
            fp = l96_drift_state_vector(zz, θp, K, J)
            fm = l96_drift_state_vector(zz, θm, K, J)
            return (fp .- fm) ./ (2δp)
        end

        u0 = u_of(z)
        div_u = 0.0
        for d in eachindex(z)
            zp = copy(z)
            zm = copy(z)
            zp[d] += z_eps
            zm[d] -= z_eps
            up = u_of(zp)
            um = u_of(zm)
            div_u += (up[d] - um[d]) / (2z_eps)
        end
        G_numeric[ip] = -(div_u + dot(u0, s_vec))
    end

    ok = all(isapprox.(G_formula_vec, G_numeric; atol=2e-5, rtol=8e-4))
    ok || error("Conjugate implementation check failed.\nformula=$G_formula_vec\nnumeric=$G_numeric")
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

function simulate_mean_observables_fd(θ::NTuple{4,Float64},
                                      x0::Vector{Float64},
                                      y0::Matrix{Float64},
                                      cfg::L96Config;
                                      burn_snapshots::Int,
                                      nsamples::Int,
                                      rng_seed::Int)
    K = cfg.K
    J = cfg.J
    x = copy(x0)
    y = copy(y0)
    ws = make_l96_workspace(K, J)
    rng = MersenneTwister(rng_seed)
    save_every = cfg.save_every
    total_snapshots = burn_snapshots + nsamples
    m = 5
    acc = zeros(Float64, m)

    for n in 1:total_snapshots
        for _ in 1:save_every
            rk4_step_l96!(x, y, cfg.dt, ws, θ)
            add_process_noise!(x, y, rng, cfg.process_noise_sigma, cfg.dt)
        end
        if n > burn_snapshots
            accumulate_snapshot_observables!(acc, x, y)
        end
    end

    acc ./= max(nsamples, 1)
    return acc
end

function finite_difference_jacobian_l96(base_θ::NTuple{4,Float64},
                                        x0::Vector{Float64},
                                        y0::Matrix{Float64},
                                        cfg::L96Config;
                                        h_rel::Float64=FD_H_REL,
                                        h_abs::Vector{Float64}=FD_H_ABS,
                                        h_scale::Float64=1.0,
                                        burn_snapshots::Int=FD_BURN_SNAPSHOTS,
                                        nsamples::Int=FD_NSAMPLES,
                                        n_rep::Int=FD_N_REPS,
                                        seed_base::Int=FD_SEED_BASE,
                                        return_replicates::Bool=false)
    m = 5
    p = 4
    length(h_abs) == p || error("h_abs must have length $p for parameters [F,h,c,b]")
    S = zeros(Float64, m, p)
    h_used = zeros(Float64, p)
    S_reps = [zeros(Float64, m, p) for _ in 1:max(n_rep, 1)]

    @info "Computing FD Jacobian" nsamples=nsamples burn_snapshots=burn_snapshots n_rep=n_rep h_rel=h_rel h_scale=h_scale
    @showprogress "Finite differences..." for j in 1:p
        θj = base_θ[j]
        h = h_scale * max(h_abs[j], h_rel * max(abs(θj), 1.0))
        h_used[j] = h
        θp = collect(base_θ)
        θm = collect(base_θ)
        θp[j] += h
        θm[j] -= h

        Ap = zeros(Float64, m)
        Am = zeros(Float64, m)
        for rep in 1:n_rep
            seed = seed_base + 100_000 * j + rep
            Ap_rep = simulate_mean_observables_fd((θp[1], θp[2], θp[3], θp[4]), x0, y0, cfg;
                                                  burn_snapshots=burn_snapshots,
                                                  nsamples=nsamples,
                                                  rng_seed=seed)
            Am_rep = simulate_mean_observables_fd((θm[1], θm[2], θm[3], θm[4]), x0, y0, cfg;
                                                  burn_snapshots=burn_snapshots,
                                                  nsamples=nsamples,
                                                  rng_seed=seed)
            drep = (Ap_rep .- Am_rep) ./ (2h)
            @views S_reps[rep][:, j] .= drep
            Ap .+= Ap_rep
            Am .+= Am_rep
        end
        Ap ./= n_rep
        Am ./= n_rep
        @views S[:, j] .= (Ap .- Am) ./ (2h)
    end
    if return_replicates
        return (S=S, S_reps=S_reps, h_used=h_used)
    end
    return S
end

function write_matrix_csv(path::AbstractString, S::Matrix{Float64})
    mkpath(dirname(path))
    size(S, 1) == 5 || error("Expected global-5 observable rows, got $(size(S, 1))")
    open(path, "w") do io
        println(io, "observable,$(join(PARAM_NAMES, ','))")
        for t in 1:5
            @printf(io, "%s,%.12e,%.12e,%.12e,%.12e\n",
                    OBS_GLOBAL_NAMES[t], S[t, 1], S[t, 2], S[t, 3], S[t, 4])
        end
    end
    return path
end

function read_matrix_csv(path::AbstractString)
    isfile(path) || error("FD matrix CSV not found: $path")
    lines = readlines(path)
    length(lines) >= 6 || error("FD matrix CSV appears malformed (need header + 5 rows): $path")
    S = zeros(Float64, 5, 4)
    for i in 1:5
        cols = split(lines[i + 1], ",")
        length(cols) >= 5 || error("Malformed row $(i + 1) in $path")
        for j in 1:4
            S[i, j] = parse(Float64, strip(cols[j + 1]))
        end
    end
    return S
end

function write_fd_replicates_csv(path::AbstractString,
                                 S_reps::Vector{Matrix{Float64}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "replicate,observable,param,value")
        for (rep, S) in enumerate(S_reps)
            for i in 1:5, j in 1:4
                @printf(io, "%d,%s,%s,%.12e\n", rep, OBS_GLOBAL_NAMES[i], PARAM_NAMES[j], S[i, j])
            end
        end
    end
    return path
end

function write_fd_uncertainty_csv(path::AbstractString,
                                  S_mean::Matrix{Float64},
                                  S_reps::Vector{Matrix{Float64}})
    mkpath(dirname(path))
    nrep = length(S_reps)
    open(path, "w") do io
        println(io, "observable,param,mean,std,se")
        for i in 1:5, j in 1:4
            vals = [S[i, j] for S in S_reps]
            μ = S_mean[i, j]
            σ = nrep > 1 ? std(vals) : 0.0
            se = nrep > 1 ? σ / sqrt(nrep) : 0.0
            @printf(io, "%s,%s,%.12e,%.12e,%.12e\n", OBS_GLOBAL_NAMES[i], PARAM_NAMES[j], μ, σ, se)
        end
    end
    return path
end

function write_fd_determinant_stats(path::AbstractString,
                                    S_mean::Matrix{Float64},
                                    S_reps::Vector{Matrix{Float64}})
    mkpath(dirname(path))
    det_mean_matrix = gram_determinant(S_mean)
    det_reps = [gram_determinant(S) for S in S_reps]
    nrep = length(det_reps)
    det_avg = mean(det_reps)
    det_std = nrep > 1 ? std(det_reps) : 0.0
    det_se = nrep > 1 ? det_std / sqrt(nrep) : 0.0

    open(path, "w") do io
        println(io, "stat,value")
        @printf(io, "det_gram_of_mean_matrix,%.12e\n", det_mean_matrix)
        @printf(io, "det_gram_replicate_mean,%.12e\n", det_avg)
        @printf(io, "det_gram_replicate_std,%.12e\n", det_std)
        @printf(io, "det_gram_replicate_se,%.12e\n", det_se)
        for (i, v) in enumerate(det_reps)
            @printf(io, "det_gram_rep_%03d,%.12e\n", i, v)
        end
    end
    return path
end

function write_full_long_table(path::AbstractString,
                               S_fd::Matrix{Float64},
                               S_gauss::Matrix{Float64},
                               S_unet::Matrix{Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "observable,param,fd,gfdt_gaussian,gfdt_unet,abs_diff_unet_fd,rel_diff_unet_fd,abs_diff_gauss_fd,rel_diff_gauss_fd")
        for t in 1:5
            obs = OBS_GLOBAL_NAMES[t]
            for j in 1:4
                fd = S_fd[t, j]
                gq = S_gauss[t, j]
                gu = S_unet[t, j]
                absd_u = abs(gu - fd)
                reld_u = absd_u / max(abs(fd), 1e-10)
                absd_g = abs(gq - fd)
                reld_g = absd_g / max(abs(fd), 1e-10)
                @printf(io, "%s,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                        obs, PARAM_NAMES[j], fd, gq, gu, absd_u, reld_u, absd_g, reld_g)
            end
        end
    end
    return path
end

function write_comparison_metrics(path::AbstractString,
                                  S_fd::Matrix{Float64},
                                  S_gauss::Matrix{Float64},
                                  S_unet::Matrix{Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "observable,param,fd,gfdt_gaussian,gfdt_unet,abs_diff_unet_fd,rel_diff_unet_fd,abs_diff_gauss_fd,rel_diff_gauss_fd")
        for t in 1:5
            obs = OBS_GLOBAL_NAMES[t]
            for j in 1:4
                fd = S_fd[t, j]
                gq = S_gauss[t, j]
                gu = S_unet[t, j]
                absd_u = abs(gu - fd)
                reld_u = absd_u / max(abs(fd), 1e-10)
                absd_g = abs(gq - fd)
                reld_g = absd_g / max(abs(fd), 1e-10)
                @printf(io, "%s,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                        obs, PARAM_NAMES[j], fd, gq, gu, absd_u, reld_u, absd_g, reld_g)
            end
        end
        @printf(io, "GLOBAL,rmse,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                0.0,
                sqrt(mean((S_gauss .- S_fd) .^ 2)),
                sqrt(mean((S_unet .- S_fd) .^ 2)),
                mean(abs.(S_unet .- S_fd)),
                mean(abs.(S_unet .- S_fd) ./ max.(abs.(S_fd), 1e-10)),
                mean(abs.(S_gauss .- S_fd)),
                mean(abs.(S_gauss .- S_fd) ./ max.(abs.(S_fd), 1e-10)))
        @printf(io, "GLOBAL,sum_abs_rel,NaN,NaN,NaN,%.12e,NaN,%.12e,NaN\n",
                sum(abs.(S_unet .- S_fd) ./ max.(abs.(S_fd), 1e-10)),
                sum(abs.(S_gauss .- S_fd) ./ max.(abs.(S_fd), 1e-10)))
        @printf(io, "GLOBAL,corr,NaN,%.12e,%.12e,NaN,NaN,NaN,NaN\n",
                pearson_corr(vec(S_gauss), vec(S_fd)),
                pearson_corr(vec(S_unet), vec(S_fd)))
    end
    return path
end

function pearson_corr(x::Vector{Float64}, y::Vector{Float64})
    μx = mean(x); μy = mean(y)
    xc = x .- μx
    yc = y .- μy
    den = sqrt(sum(abs2, xc) * sum(abs2, yc))
    den > 0 || return NaN
    return dot(xc, yc) / den
end

function compute_lag_scan_metrics(A::Matrix{Float64},
                                  G_unet::Matrix{Float64},
                                  G_gauss::Matrix{Float64},
                                  S_fd::Matrix{Float64},
                                  Δt_obs::Float64,
                                  tmax_values::Vector{Float64})
    rows = NamedTuple[]
    for tmax in tmax_values
        S_u = build_gfdt_jacobian(A, G_unet, Δt_obs, tmax; mean_center=GFDT_MEAN_CENTER)
        S_g = build_gfdt_jacobian(A, G_gauss, Δt_obs, tmax; mean_center=GFDT_MEAN_CENTER)
        rmse_u = sqrt(mean((S_u .- S_fd) .^ 2))
        rmse_g = sqrt(mean((S_g .- S_fd) .^ 2))
        rel_u = mean(abs.(S_u .- S_fd) ./ max.(abs.(S_fd), 1e-10))
        rel_g = mean(abs.(S_g .- S_fd) ./ max.(abs.(S_fd), 1e-10))
        corr_u = pearson_corr(vec(S_u), vec(S_fd))
        corr_g = pearson_corr(vec(S_g), vec(S_fd))
        push!(rows, (;
            tmax=tmax,
            rmse_unet_vs_fd=rmse_u,
            rmse_gauss_vs_fd=rmse_g,
            mean_abs_rel_unet_vs_fd=rel_u,
            mean_abs_rel_gauss_vs_fd=rel_g,
            corr_unet_fd=corr_u,
            corr_gauss_fd=corr_g,
        ))
    end
    return rows
end

function solve_unet_gain_mean_zero(Gx::Matrix{Float64},
                                   Gy::Matrix{Float64},
                                   Gconst::Matrix{Float64};
                                   ridge::Float64=UNET_GAIN_RIDGE)
    mx = vec(mean(Gx; dims=2))
    my = vec(mean(Gy; dims=2))
    mc = vec(mean(Gconst; dims=2))
    M = hcat(mx, my)
    rhs = -mc
    A = transpose(M) * M + ridge * Matrix{Float64}(I, 2, 2)
    b = transpose(M) * rhs
    α = A \ b
    return (alpha_x=Float64(α[1]), alpha_y=Float64(α[2]), residual_norm=norm(M * α - rhs))
end

function gram_determinant(S::Matrix{Float64})
    G = transpose(S) * S
    return det(Symmetric(G))
end

function compute_unet_tmax_scan(A::Matrix{Float64},
                                G_unet::Matrix{Float64},
                                Δt_obs::Float64,
                                tmax_values::Vector{Float64})
    rows = NamedTuple[]
    for tmax in tmax_values
        S_u = build_gfdt_jacobian(A, G_unet, Δt_obs, tmax; mean_center=GFDT_MEAN_CENTER)
        push!(rows, (;
            tmax=tmax,
            det_gram=gram_determinant(S_u),
            fro_norm=norm(S_u),
        ))
    end
    return rows
end

function select_tmax_det_plateau(rows::Vector{<:NamedTuple};
                                 rel_tol::Float64=0.15)
    length(rows) == 0 && error("Empty tmax scan rows")
    if length(rows) == 1
        return rows[1]
    end
    for i in 2:length(rows)
        prev = max(abs(rows[i - 1].det_gram), 1e-14)
        rel = abs(rows[i].det_gram - rows[i - 1].det_gram) / prev
        if rel <= rel_tol
            return rows[i]
        end
    end
    return rows[end]
end

function write_unet_tmax_scan(path::AbstractString,
                              rows::Vector{<:NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "tmax,det_gram,fro_norm")
        for row in rows
            @printf(io, "%.6f,%.12e,%.12e\n",
                    row.tmax,
                    row.det_gram,
                    row.fro_norm)
        end
    end
    return path
end

function write_lag_scan_metrics(path::AbstractString,
                                rows::Vector{<:NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "tmax,rmse_unet_vs_fd,rmse_gauss_vs_fd,mean_abs_rel_unet_vs_fd,mean_abs_rel_gauss_vs_fd,corr_unet_fd,corr_gauss_fd")
        for row in rows
            @printf(io, "%.6f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    row.tmax,
                    row.rmse_unet_vs_fd,
                    row.rmse_gauss_vs_fd,
                    row.mean_abs_rel_unet_vs_fd,
                    row.mean_abs_rel_gauss_vs_fd,
                    row.corr_unet_fd,
                    row.corr_gauss_fd)
        end
    end
    return path
end

function write_latex_table(path::AbstractString,
                           S_fd::Matrix{Float64},
                           S_gauss::Matrix{Float64},
                           S_unet::Matrix{Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "\\begin{table*}[t]")
        println(io, "\\centering")
        println(io, "\\small")
        println(io, "\\caption{L96 global-5 parameter Jacobians (\$\\partial\\langle\\phi_m\\rangle/\\partial\\theta_i\$) at \$(F,h,c,b)=(10,1,10,10)\$.}")
        println(io, "\\label{tab:l96_jacobian_global5}")
        println(io, "\\begin{tabular}{llrrrr}")
        println(io, "\\toprule")
        println(io, "Method & Observable & \$\\partial_F\$ & \$\\partial_h\$ & \$\\partial_c\$ & \$\\partial_b\$ \\\\")
        println(io, "\\midrule")
        for (label, S) in [("Finite differences", S_fd), ("GFDT + Gaussian", S_gauss), ("GFDT + UNet", S_unet)]
            for i in 1:5
                @printf(io, "%s & %s & %.6e & %.6e & %.6e & %.6e \\\\\n",
                        label, OBS_LATEX_NAMES[i], S[i, 1], S[i, 2], S[i, 3], S[i, 4])
            end
            println(io, "\\midrule")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table*}")
    end
    return path
end

function write_report(path::AbstractString,
                      run_dir::AbstractString,
                      checkpoint_path::AbstractString,
                      best_epoch::Int,
                      cfg::L96Config,
                      n_used::Int,
                      S_fd::Matrix{Float64},
                      S_gauss::Matrix{Float64},
                      S_unet::Matrix{Float64},
                      Δt_obs::Float64,
                      gfdt_tmax_requested::Float64,
                      gfdt_tmax_selected::Float64,
                      gfdt_tmax_select_mode::AbstractString,
                      unet_gain_mode::AbstractString,
                      unet_gain_ridge::Float64,
                      unet_alpha_x::Float64,
                      unet_alpha_y::Float64;
                      fd_nsamples::Int=FD_NSAMPLES,
                      fd_burn_snapshots::Int=FD_BURN_SNAPSHOTS,
                      fd_n_reps::Int=FD_N_REPS,
                      fd_h_rel::Float64=FD_H_REL,
                      fd_h_abs::Vector{Float64}=collect(Float64.(FD_H_ABS)),
                      fd_seed_base::Int=FD_SEED_BASE,
                      fd_det_stats::Union{Nothing,NamedTuple}=nothing)
    mkpath(dirname(path))
    m, p = size(S_fd)
    abs_diff_unet = abs.(S_unet .- S_fd)
    rel_diff_unet = abs_diff_unet ./ max.(abs.(S_fd), 1e-10)
    sum_rel_unet = sum(rel_diff_unet)
    abs_diff_gauss = abs.(S_gauss .- S_fd)
    det_fd = gram_determinant(S_fd)
    det_unet = gram_determinant(S_unet)
    det_gauss = gram_determinant(S_gauss)
    det_rel_unet = abs(det_unet - det_fd) / max(abs(det_fd), 1e-14)
    det_rel_gauss = abs(det_gauss - det_fd) / max(abs(det_fd), 1e-14)
    fd_scale = median(abs.(vec(S_fd)))
    robust_den = max(fd_scale, 1e-10)
    rel_diff_unet_robust = abs_diff_unet ./ max.(abs.(S_fd), robust_den)
    rel_diff_gauss_robust = abs_diff_gauss ./ max.(abs.(S_fd), robust_den)

    open(path, "w") do io
        println(io, "# L96 Parameter Jacobian Comparison")
        println(io)
        println(io, "- run_dir: `", abspath(run_dir), "`")
        println(io, "- best_checkpoint: `", abspath(checkpoint_path), "` (epoch ", best_epoch, ")")
        println(io, "- parameters: `(F,h,c,b)=(", cfg.F, ", ", cfg.h, ", ", cfg.c, ", ", cfg.b, ")`")
        println(io, "- K: `", cfg.K, "`, J: `", cfg.J, "`, observables: `5 (global)`")
        println(io, "- GFDT samples used: `", n_used, "`")
        println(io, "- observation Δt: `", Δt_obs, "`")
        println(io, "- GFDT Tmax requested: `", gfdt_tmax_requested, "`")
        println(io, "- GFDT Tmax selection mode: `", gfdt_tmax_select_mode, "`")
        println(io, "- GFDT Tmax selected: `", gfdt_tmax_selected, "`")
        println(io, "- UNet gain mode: `", unet_gain_mode, "`")
        println(io, "- UNet gain ridge: `", @sprintf("%.6e", unet_gain_ridge), "`")
        println(io, "- UNet gain alpha_x: `", @sprintf("%.6e", unet_alpha_x), "`")
        println(io, "- UNet gain alpha_y: `", @sprintf("%.6e", unet_alpha_y), "`")
        println(io, "- FD nsamples/burn_snapshots: `", fd_nsamples, "` / `", fd_burn_snapshots, "`")
        h_abs_fmt = join([@sprintf("%.3e", v) for v in fd_h_abs], ", ")
        println(io, "- FD n_reps: `", fd_n_reps, "`, h_rel: `", @sprintf("%.6e", fd_h_rel), "`, h_abs: `", h_abs_fmt, "`, seed_base: `", fd_seed_base, "`")
        println(io)
        println(io, "## Observable Definitions")
        println(io)
        for (i, name) in enumerate(OBS_GLOBAL_NAMES)
            println(io, "- `", name, "`: row ", i)
        end
        println(io)
        println(io, "## Global Discrepancy (UNet vs FD)")
        println(io)
        println(io, "- det(S_fd' * S_fd): `", @sprintf("%.6e", det_fd), "`")
        println(io, "- det(S_unet' * S_unet): `", @sprintf("%.6e", det_unet), "`")
        println(io, "- det(S_gauss' * S_gauss): `", @sprintf("%.6e", det_gauss), "`")
        println(io, "- det_rel_diff_unet_vs_fd: `", @sprintf("%.6e", det_rel_unet), "`")
        println(io, "- det_rel_diff_gauss_vs_fd: `", @sprintf("%.6e", det_rel_gauss), "`")
        if fd_det_stats !== nothing
            println(io, "- det(FD replicate mean): `", @sprintf("%.6e", fd_det_stats.det_rep_mean), "`")
            println(io, "- det(FD replicate std): `", @sprintf("%.6e", fd_det_stats.det_rep_std), "`")
            println(io, "- det(FD replicate se): `", @sprintf("%.6e", fd_det_stats.det_rep_se), "`")
        end
        println(io, "- mean_abs_diff: `", @sprintf("%.6e", mean(abs_diff_unet)), "`")
        println(io, "- median_abs_diff: `", @sprintf("%.6e", median(vec(abs_diff_unet))), "`")
        println(io, "- rmse: `", @sprintf("%.6e", sqrt(mean((S_unet .- S_fd).^2))), "`")
        println(io, "- mean_abs_rel_diff: `", @sprintf("%.6e", mean(rel_diff_unet)), "`")
        println(io, "- sum_abs_rel_diff: `", @sprintf("%.6e", sum_rel_unet), "`")
        println(io, "- median_abs_rel_diff: `", @sprintf("%.6e", median(vec(rel_diff_unet))), "`")
        println(io, "- robust_rel_floor (median |FD|): `", @sprintf("%.6e", robust_den), "`")
        println(io, "- mean_abs_rel_diff_robust: `", @sprintf("%.6e", mean(rel_diff_unet_robust)), "`")
        println(io)
        println(io, "## Parameter-wise Correlation (UNet vs FD)")
        println(io)
        println(io, "| Parameter | Corr | RMSE |")
        println(io, "|---|---:|---:|")
        for j in 1:p
            c = pearson_corr(vec(S_unet[:, j]), vec(S_fd[:, j]))
            r = sqrt(mean((S_unet[:, j] .- S_fd[:, j]).^2))
            println(io, "| ", PARAM_NAMES[j], " | ", @sprintf("%.6f", c), " | ", @sprintf("%.6e", r), " |")
        end
        println(io)
        println(io, "## Gaussian Baseline (for reference)")
        println(io)
        println(io, "- mean_abs_diff (Gaussian vs FD): `", @sprintf("%.6e", mean(abs_diff_gauss)), "`")
        println(io, "- rmse (Gaussian vs FD): `", @sprintf("%.6e", sqrt(mean((S_gauss .- S_fd).^2))), "`")
        println(io, "- mean_abs_rel_diff_robust (Gaussian vs FD): `", @sprintf("%.6e", mean(rel_diff_gauss_robust)), "`")
        println(io)
        println(io, "## Jacobian Matrices (rows = observables, cols = parameters)")
        println(io)
        println(io, "- FD:")
        for i in 1:m
            println(io, "  - `", OBS_GLOBAL_NAMES[i], "`: [",
                    @sprintf("%.6e, %.6e, %.6e, %.6e", S_fd[i, 1], S_fd[i, 2], S_fd[i, 3], S_fd[i, 4]), "]")
        end
        println(io, "- GFDT+UNet:")
        for i in 1:m
            println(io, "  - `", OBS_GLOBAL_NAMES[i], "`: [",
                    @sprintf("%.6e, %.6e, %.6e, %.6e", S_unet[i, 1], S_unet[i, 2], S_unet[i, 3], S_unet[i, 4]), "]")
        end
        println(io, "- GFDT+Gaussian:")
        for i in 1:m
            println(io, "  - `", OBS_GLOBAL_NAMES[i], "`: [",
                    @sprintf("%.6e, %.6e, %.6e, %.6e", S_gauss[i, 1], S_gauss[i, 2], S_gauss[i, 3], S_gauss[i, 4]), "]")
        end
    end
    return path
end

function main(args=ARGS)
    cli = parse_cli(args)
    run_dir = abspath(String(cli["run_dir"]))
    checkpoint_override = String(cli["checkpoint_path"])
    integration_toml = abspath(String(cli["integration_toml"]))
    out_dir = abspath(String(cli["output_dir"]))
    gfdt_tmax = Float64(cli["gfdt_tmax"])
    gfdt_nsamples = Int(cli["gfdt_nsamples"])
    gfdt_start_index = Int(cli["gfdt_start_index"])
    fd_step_scales = collect(Float64.(cli["fd_step_scales"]))
    fd_nsamples = Int(cli["fd_nsamples"])
    fd_burn_snapshots = Int(cli["fd_burn_snapshots"])
    fd_n_reps = Int(cli["fd_n_reps"])
    fd_h_rel = Float64(cli["fd_h_rel"])
    fd_h_abs = collect(Float64.(cli["fd_h_abs"]))
    fd_seed_base = Int(cli["fd_seed_base"])
    fd_matrix_csv = String(cli["fd_matrix_csv"])
    gfdt_tmax_select_mode = String(cli["gfdt_tmax_select_mode"])
    unet_gain_mode = String(cli["unet_gain_mode"])
    unet_gain_ridge = Float64(cli["unet_gain_ridge"])
    unet_forward_mode = String(cli["unet_forward_mode"])
    mkpath(out_dir)
    validate_global_observable_impl!()
    validate_conjugate_implementation!()
    validate_gfdt_sign_ou!()

    if gfdt_tmax_select_mode != "fixed"
        @warn "Non-fixed GFDT Tmax selection is heuristic and not part of the primary estimator." mode=gfdt_tmax_select_mode
    end
    if lowercase(unet_gain_mode) != "none"
        @warn "UNet gain correction is heuristic and not part of the primary estimator." mode=unet_gain_mode
    end

    cfg = load_l96_config(integration_toml)
    θ = (cfg.F, cfg.h, cfg.c, cfg.b)
    Δt_obs = cfg.dt * cfg.save_every

    checkpoint_path = ""
    best_epoch = -1
    if isempty(strip(checkpoint_override))
        best = pick_best_checkpoint(run_dir)
        checkpoint_path = best.checkpoint_path
        best_epoch = best.best_epoch
    else
        checkpoint_path = abspath(checkpoint_override)
        isfile(checkpoint_path) || error("Checkpoint override not found: $checkpoint_path")
        m = match(r"epoch[_-]?0*([0-9]+)", lowercase(basename(checkpoint_path)))
        best_epoch = m === nothing ? -1 : parse(Int, m.captures[1])
    end

    @info "Loading unperturbed L96 trajectory subset" dataset=cfg.dataset_path dataset_key=cfg.dataset_key nsamples=gfdt_nsamples start_index=gfdt_start_index
    tensor = load_observation_subset(cfg; nsamples=gfdt_nsamples, start_index=gfdt_start_index)
    K, C, N = size(tensor)
    C == cfg.J + 1 || error("Channel mismatch: C=$C, expected $(cfg.J + 1)")
    K == cfg.K || error("K mismatch: K=$K, expected $(cfg.K)")
    @info "Loaded tensor" K=K C=C N=N

    A = compute_global_observables(tensor)

    # U-Net GFDT
    G_unet_data = compute_G_unet(tensor, checkpoint_path, θ;
                                 batch_size=SCORE_BATCH_SIZE,
                                 device_pref=SCORE_DEVICE_PREF,
                                 forward_mode=unet_forward_mode,
                                 return_components=true)
    G_unet_raw = G_unet_data.G
    G_unet_x = G_unet_data.Gx
    G_unet_y = G_unet_data.Gy
    G_unet_const = G_unet_data.Gconst

    # Quasi-Gaussian GFDT
    G_gauss = compute_G_gaussian(tensor, θ)

    # FD Jacobian (load from file or compute).
    S_fd_reps = Matrix{Float64}[]
    fd_det_stats = nothing
    S_fd = if !isempty(strip(fd_matrix_csv))
        fd_path = abspath(fd_matrix_csv)
        @info "Loading precomputed FD Jacobian" path=fd_path
        read_matrix_csv(fd_path)
    else
        x0 = copy(vec(tensor[:, 1, FD_INIT_SNAPSHOT_INDEX]))
        y0 = Matrix{Float64}(undef, cfg.J, cfg.K)
        @inbounds for k in 1:cfg.K, j in 1:cfg.J
            y0[j, k] = tensor[k, j + 1, FD_INIT_SNAPSHOT_INDEX]
        end
        fd_out = finite_difference_jacobian_l96(θ, x0, y0, cfg;
                                       h_rel=fd_h_rel,
                                       h_abs=fd_h_abs,
                                       h_scale=1.0,
                                       burn_snapshots=fd_burn_snapshots,
                                       nsamples=fd_nsamples,
                                       n_rep=fd_n_reps,
                                       seed_base=fd_seed_base,
                                       return_replicates=true)
        S_fd_reps = fd_out.S_reps
        det_vals = [gram_determinant(Sr) for Sr in S_fd_reps]
        nrep = length(det_vals)
        fd_det_stats = (;
            det_rep_mean=mean(det_vals),
            det_rep_std=(nrep > 1 ? std(det_vals) : 0.0),
            det_rep_se=(nrep > 1 ? std(det_vals) / sqrt(nrep) : 0.0),
        )
        fd_out.S
    end

    # Lag-window diagnostics (FD comparison only) and UNet-only automatic tmax selection.
    tscan_vals = sort(unique(vcat(collect(Float64.(GFDT_TMAX_SCAN)), [gfdt_tmax])))
    lag_rows = compute_lag_scan_metrics(A, G_unet_raw, G_gauss, S_fd, Δt_obs, tscan_vals)
    unet_tmax_rows = compute_unet_tmax_scan(A, G_unet_raw, Δt_obs, tscan_vals)
    selected_tmax = gfdt_tmax
    selected_alpha_x = 1.0
    selected_alpha_y = 1.0

    if gfdt_tmax_select_mode == "det_plateau"
        best_row = select_tmax_det_plateau(unet_tmax_rows)
        selected_tmax = best_row.tmax
        @info "Selected GFDT lag window from UNet determinant plateau" mode=gfdt_tmax_select_mode requested_tmax=gfdt_tmax selected_tmax=selected_tmax
    end

    if lowercase(unet_gain_mode) == "mean_zero_g"
        gain = solve_unet_gain_mean_zero(G_unet_x, G_unet_y, G_unet_const; ridge=unet_gain_ridge)
        selected_alpha_x = gain.alpha_x
        selected_alpha_y = gain.alpha_y
        @info "Selected FD-independent UNet gain from E[G]=0" mode=unet_gain_mode alpha_x=selected_alpha_x alpha_y=selected_alpha_y residual_norm=gain.residual_norm
    end

    # Jacobians from GFDT correlations at selected lag window.
    S_unet_raw = build_gfdt_jacobian(A, G_unet_raw, Δt_obs, selected_tmax; mean_center=GFDT_MEAN_CENTER)
    S_unet = if lowercase(unet_gain_mode) == "mean_zero_g"
        Sx = build_gfdt_jacobian(A, G_unet_x, Δt_obs, selected_tmax; mean_center=GFDT_MEAN_CENTER)
        Sy = build_gfdt_jacobian(A, G_unet_y, Δt_obs, selected_tmax; mean_center=GFDT_MEAN_CENTER)
        Sc = build_gfdt_jacobian(A, G_unet_const, Δt_obs, selected_tmax; mean_center=GFDT_MEAN_CENTER)
        Sc .+ selected_alpha_x .* Sx .+ selected_alpha_y .* Sy
    else
        S_unet_raw
    end
    S_gauss = build_gfdt_jacobian(A, G_gauss, Δt_obs, selected_tmax; mean_center=GFDT_MEAN_CENTER)

    # Persist outputs.
    s_fd_csv = write_matrix_csv(joinpath(out_dir, "jacobian_fd_matrix_global5.csv"), S_fd)
    s_g_csv = write_matrix_csv(joinpath(out_dir, "jacobian_gfdt_gaussian_matrix_global5.csv"), S_gauss)
    s_u_csv = write_matrix_csv(joinpath(out_dir, "jacobian_gfdt_unet_matrix_global5.csv"), S_unet)
    s_u_raw_csv = write_matrix_csv(joinpath(out_dir, "jacobian_gfdt_unet_raw_matrix_global5.csv"), S_unet_raw)
    full_csv = write_full_long_table(joinpath(out_dir, "jacobian_full_table_global5.csv"), S_fd, S_gauss, S_unet)
    summary_csv = write_comparison_metrics(joinpath(out_dir, "jacobian_comparison_metrics_global5.csv"), S_fd, S_gauss, S_unet)
    fd_reps_csv = ""
    fd_unc_csv = ""
    fd_det_csv = ""
    if !isempty(S_fd_reps)
        fd_reps_csv = write_fd_replicates_csv(joinpath(out_dir, "jacobian_fd_replicates_global5.csv"), S_fd_reps)
        fd_unc_csv = write_fd_uncertainty_csv(joinpath(out_dir, "jacobian_fd_uncertainty_global5.csv"), S_fd, S_fd_reps)
        fd_det_csv = write_fd_determinant_stats(joinpath(out_dir, "jacobian_fd_determinant_stats_global5.csv"), S_fd, S_fd_reps)
    end
    report_md = write_report(joinpath(out_dir, "jacobian_report_global5.md"),
                             run_dir,
                             checkpoint_path,
                             best_epoch,
                             cfg,
                             N,
                             S_fd,
                             S_gauss,
                             S_unet,
                             Δt_obs,
                             gfdt_tmax,
                             selected_tmax,
                             gfdt_tmax_select_mode,
                             unet_gain_mode,
                             unet_gain_ridge,
                             selected_alpha_x,
                             selected_alpha_y;
                             fd_nsamples=fd_nsamples,
                             fd_burn_snapshots=fd_burn_snapshots,
                             fd_n_reps=fd_n_reps,
                             fd_h_rel=fd_h_rel,
                             fd_h_abs=fd_h_abs,
                             fd_seed_base=fd_seed_base,
                             fd_det_stats=fd_det_stats)
    latex_tbl = write_latex_table(joinpath(out_dir, "jacobian_table_global5.tex"), S_fd, S_gauss, S_unet)
    lag_scan_csv = write_lag_scan_metrics(joinpath(out_dir, "jacobian_lag_scan_global5.csv"), lag_rows)
    unet_tmax_scan_csv = write_unet_tmax_scan(joinpath(out_dir, "jacobian_unet_tmax_scan_global5.csv"), unet_tmax_rows)

    if isempty(strip(fd_matrix_csv)) && any(!=(1.0), fd_step_scales)
        fd_sweep_csv = joinpath(out_dir, "jacobian_fd_step_sweep_global5.csv")
        open(fd_sweep_csv, "w") do io
            println(io, "h_scale,observable,param,value")
            x0 = copy(vec(tensor[:, 1, FD_INIT_SNAPSHOT_INDEX]))
            y0 = Matrix{Float64}(undef, cfg.J, cfg.K)
            @inbounds for k in 1:cfg.K, j in 1:cfg.J
                y0[j, k] = tensor[k, j + 1, FD_INIT_SNAPSHOT_INDEX]
            end
            for hs in sort(unique(fd_step_scales))
                S_fd_h = finite_difference_jacobian_l96(θ, x0, y0, cfg;
                                                        h_rel=fd_h_rel,
                                                        h_abs=fd_h_abs,
                                                        h_scale=hs,
                                                        burn_snapshots=fd_burn_snapshots,
                                                        nsamples=fd_nsamples,
                                                        n_rep=fd_n_reps,
                                                        seed_base=fd_seed_base + round(Int, 1000 * hs))
                for i in 1:5, j in 1:4
                    @printf(io, "%.6f,%s,%s,%.12e\n", hs, OBS_GLOBAL_NAMES[i], PARAM_NAMES[j], S_fd_h[i, j])
                end
            end
        end
    end

    # Console summary.
    unet_rmse = sqrt(mean((S_unet .- S_fd).^2))
    unet_rel = mean(abs.(S_unet .- S_fd) ./ max.(abs.(S_fd), 1e-10))
    gauss_rmse = sqrt(mean((S_gauss .- S_fd).^2))
    @info "Jacobian evaluation completed" output_dir=out_dir best_epoch=best_epoch gfdt_tmax_requested=gfdt_tmax gfdt_tmax_selected=selected_tmax gfdt_tmax_select_mode=gfdt_tmax_select_mode unet_rmse=unet_rmse unet_mean_abs_rel=unet_rel gauss_rmse=gauss_rmse
    println("Saved:")
    println("  - $s_fd_csv")
    println("  - $s_g_csv")
    println("  - $s_u_csv")
    println("  - $s_u_raw_csv")
    println("  - $full_csv")
    println("  - $summary_csv")
    println("  - $report_md")
    println("  - $latex_tbl")
    println("  - $lag_scan_csv")
    println("  - $unet_tmax_scan_csv")
    !isempty(fd_reps_csv) && println("  - $fd_reps_csv")
    !isempty(fd_unc_csv) && println("  - $fd_unc_csv")
    !isempty(fd_det_csv) && println("  - $fd_det_csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
