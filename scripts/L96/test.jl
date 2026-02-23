#!/usr/bin/env julia

# Run command:
# julia --threads auto --project=. test.jl

using BSON
using FFTW
using Flux
using HDF5
using LinearAlgebra
using Plots
using Printf
using ProgressMeter
using Random
using ScoreUNet1D
using Statistics
using TOML
using Base.Threads

const PARAMS_TOML = "scripts/L96/parameters_responses.toml"
const OUTPUT_PLOT_RAW = "scripts/L96/responses_FxFy_impulse_unet_vs_numerical_raw_5x2.png"
const OUTPUT_PLOT_CORR = "scripts/L96/responses_FxFy_impulse_unet_vs_numerical_corrected_5x2.png"
const SCORE_RESPONSE_MODE = :corrected  # :raw, :corrected, or :both
const CORRECTION_FIT_NSAMPLES = 0  # 0 => use all GFDT samples; >0 uses subset for fitting correction T in corrected-only mode

function resolve_score_response_mode(mode)
    m = mode isa Symbol ? mode : Symbol(lowercase(String(mode)))
    m in (:raw, :corrected, :both) || error("Invalid SCORE_RESPONSE_MODE=$mode. Use :raw, :corrected, or :both")
    return m
end

mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1
@inline l96_linidx(j::Int, k::Int, J::Int) = (k - 1) * J + j
@inline function l96_jk_from_lin(ell::Int, J::Int)
    kk = (ell - 1) ÷ J + 1
    jj = (ell - 1) % J + 1
    return jj, kk
end
@inline l96_wrap_lin(ell::Int, K::Int, J::Int) = mod1idx(ell, K * J)

struct L96FxFyConfig
    K::Int
    J::Int
    Fx::Float64
    Fy::Float64
    h::Float64
    c::Float64
    b::Float64
    dt::Float64
    save_every::Int
    process_noise_sigma::Float64
    stochastic_x_noise::Bool
    dataset_path::String
    dataset_key::String
end

function parse_bool_like(v)
    if v isa Bool
        return v
    elseif v isa AbstractString
        s = lowercase(strip(v))
        s in ("1", "true", "yes", "on") && return true
        s in ("0", "false", "no", "off") && return false
    end
    error("Could not parse boolean value: $v")
end

function resolve_checkpoint(run_dir::AbstractString, checkpoint_epoch::Int, checkpoint_path::AbstractString)
    if !isempty(strip(checkpoint_path))
        return abspath(checkpoint_path)
    end
    if checkpoint_epoch > 0
        p4 = joinpath(run_dir, "model", @sprintf("score_model_epoch_%04d.bson", checkpoint_epoch))
        p3 = joinpath(run_dir, "model", @sprintf("score_model_epoch_%03d.bson", checkpoint_epoch))
        isfile(p4) && return abspath(p4)
        isfile(p3) && return abspath(p3)
        @warn "Requested checkpoint epoch not found; falling back to score_model.bson" checkpoint_epoch run_dir
    end
    p = joinpath(run_dir, "model", "score_model.bson")
    isfile(p) || error("Could not resolve checkpoint in $run_dir")
    return abspath(p)
end

function load_config_from_params(path::AbstractString)
    raw = TOML.parsefile(path)

    paths = get(raw, "paths", Dict{String,Any}())
    integ = get(raw, "integration", Dict{String,Any}())
    dset = get(raw, "dataset", Dict{String,Any}())
    gfdt = get(raw, "gfdt", Dict{String,Any}())
    num = get(raw, "numerical", Dict{String,Any}())

    run_dir = abspath(String(get(paths, "run_dir", "scripts/L96/runs_J10/run_033")))
    checkpoint_epoch = Int(get(paths, "checkpoint_epoch", 8))
    checkpoint_path = String(get(paths, "checkpoint_path", ""))
    ckpt = resolve_checkpoint(run_dir, checkpoint_epoch, checkpoint_path)

    cfg = L96FxFyConfig(
        Int(get(integ, "K", 36)),
        Int(get(integ, "J", 10)),
        Float64(get(integ, "F", 10.0)),
        0.0,
        Float64(get(integ, "h", 1.0)),
        Float64(get(integ, "c", 10.0)),
        Float64(get(integ, "b", 10.0)),
        Float64(get(integ, "dt", 0.005)),
        Int(get(integ, "save_every", 10)),
        Float64(get(integ, "process_noise_sigma", 0.03)),
        parse_bool_like(get(integ, "stochastic_x_noise", false)),
        abspath(String(get(dset, "path", "scripts/L96/observations/J10/l96_timeseries.hdf5"))),
        String(get(dset, "key", "timeseries")),
    )

    opts = Dict{String,Any}(
        "gfdt_nsamples" => Int(get(gfdt, "nsamples", 150_000)),
        "gfdt_start" => Int(get(gfdt, "start_index", 50_001)),
        "response_tmax" => Float64(get(gfdt, "response_tmax", 6.0)),
        "score_batch_size" => Int(get(gfdt, "score_batch_size", 2048)),
        "score_device" => String(get(gfdt, "score_device", "auto")),
        "score_forward_mode" => String(get(gfdt, "score_forward_mode", "train")),
        "mean_center" => parse_bool_like(get(gfdt, "mean_center", true)),
        "num_ensembles" => Int(get(num, "ensembles", 8192)),
        "num_start" => Int(get(num, "start_index", 80_001)),
        "num_seed" => Int(get(num, "seed_base", 920_000)),
        "h_rel" => Float64(get(num, "h_rel", 0.01)),
        "h_abs" => Float64.(collect(get(num, "h_abs", [0.05, 0.02, 0.05, 0.05]))),
        "correction_ridge" => Float64(get(get(raw, "correction", Dict{String,Any}()), "ridge", 1e-8)),
        "correction_max_deviation" => Float64(get(get(raw, "correction", Dict{String,Any}()), "max_deviation", 10.0)),
        "checkpoint" => ckpt,
    )

    return cfg, opts
end

function sync_with_dataset_attrs(cfg::L96FxFyConfig)
    isfile(cfg.dataset_path) || error("Dataset not found: $(cfg.dataset_path)")
    attrs = h5open(cfg.dataset_path, "r") do h5
        haskey(h5, cfg.dataset_key) || error("Dataset key $(cfg.dataset_key) missing in $(cfg.dataset_path)")
        ds = h5[cfg.dataset_key]
        a = attributes(ds)
        Dict{String,Any}(k => read(a[k]) for k in keys(a))
    end

    K = haskey(attrs, "K") ? Int(attrs["K"]) : cfg.K
    J = haskey(attrs, "J") ? Int(attrs["J"]) : cfg.J
    dt = haskey(attrs, "dt") ? Float64(attrs["dt"]) : cfg.dt
    save_every = haskey(attrs, "save_every") ? Int(attrs["save_every"]) : cfg.save_every
    process_noise_sigma = haskey(attrs, "process_noise_sigma") ? Float64(attrs["process_noise_sigma"]) : cfg.process_noise_sigma
    stochastic_x_noise = haskey(attrs, "stochastic_x_noise") ? Bool(attrs["stochastic_x_noise"]) : cfg.stochastic_x_noise

    return L96FxFyConfig(
        K,
        J,
        cfg.Fx,
        cfg.Fy,
        cfg.h,
        cfg.c,
        cfg.b,
        dt,
        save_every,
        process_noise_sigma,
        stochastic_x_noise,
        cfg.dataset_path,
        cfg.dataset_key,
    )
end

function load_observation_subset(cfg::L96FxFyConfig; nsamples::Int, start_index::Int, label::AbstractString)
    raw = h5open(cfg.dataset_path, "r") do h5
        ds = h5[cfg.dataset_key]
        n_total = size(ds, 1)
        n_use = min(nsamples, n_total)
        max_start = max(1, n_total - n_use + 1)
        s_use = clamp(start_index, 1, max_start)
        e_use = s_use + n_use - 1
        if n_use != nsamples || s_use != start_index
            @warn "Adjusted subset bounds" label requested_nsamples = nsamples used_nsamples = n_use requested_start = start_index used_start = s_use total = n_total
        end
        ds[s_use:e_use, :, :]
    end
    return Float64.(permutedims(raw, (3, 2, 1)))
end

function compute_global_observables(tensor::Array{Float64,3})
    K, C, N = size(tensor)
    J = C - 1
    A = zeros(Float64, 5, N)
    invK = 1.0 / K
    invJ = 1.0 / J

    @showprogress "Computing 5 observables..." for n in 1:N
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
                y = tensor[k, j + 1, n]
                ysum += y
                y2sum += y * y
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

function xcorr_one_sided_unbiased_fft(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, n_lags::Int)
    n = length(x)
    n == length(y) || error("xcorr input length mismatch")
    K = min(n_lags, n - 1)

    L = 1
    while L < (2 * n - 1)
        L <<= 1
    end

    xp = zeros(Float64, L)
    yp = zeros(Float64, L)
    @inbounds xp[1:n] .= x
    @inbounds yp[1:n] .= y
    c = real(ifft(fft(xp) .* conj.(fft(yp))))

    out = Vector{Float64}(undef, K + 1)
    @inbounds for k in 0:K
        out[k + 1] = c[k + 1] / (n - k)
    end
    return out
end

function build_gfdt_response(A::Matrix{Float64}, G::Matrix{Float64}, delta_t::Float64, n_lags::Int; mean_center::Bool=true)
    m, N = size(A)
    p, N2 = size(G)
    N == N2 || error("A/G length mismatch")
    n_lags = min(n_lags, N - 1)

    Ause = mean_center ? (A .- mean(A; dims=2)) : A
    Guse = mean_center ? (G .- mean(G; dims=2)) : G

    C = zeros(Float64, m, p, n_lags + 1)
    R = zeros(Float64, m, p, n_lags + 1)

    Threads.@threads for pair in 1:(m * p)
        i = (pair - 1) ÷ p + 1
        j = (pair - 1) % p + 1
        cpos = xcorr_one_sided_unbiased_fft(vec(@view Ause[i, :]), vec(@view Guse[j, :]), n_lags)
        @views C[i, j, :] .= cpos

        acc = 0.0
        R[i, j, 1] = 0.0
        @inbounds for lag in 1:n_lags
            acc += 0.5 * (cpos[lag] + cpos[lag + 1])
            R[i, j, lag + 1] = delta_t * acc
        end
    end

    times = collect(0:n_lags) .* delta_t
    return C, R, times
end

function linear_interpolate_3D_time(R::Array{Float64,3}, t_in::Vector{Float64}, t_out::Vector{Float64})
    M, P, Nin = size(R)
    Nin == length(t_in) || error("Time/data mismatch in interpolation")
    Nout = length(t_out)
    Rout = zeros(Float64, M, P, Nout)
    for i in 1:M, j in 1:P
        for (k, t) in enumerate(t_out)
            if t <= t_in[1]
                Rout[i, j, k] = R[i, j, 1]
            elseif t >= t_in[end]
                Rout[i, j, k] = R[i, j, end]
            else
                idx = searchsortedlast(t_in, t)
                t0 = t_in[idx]
                t1 = t_in[idx + 1]
                w = (t - t0) / (t1 - t0)
                Rout[i, j, k] = R[i, j, idx] + w * (R[i, j, idx + 1] - R[i, j, idx])
            end
        end
    end
    return Rout
end

function step_to_impulse(R::Array{Float64,3}, times::Vector{Float64})
    M, P, Nt = size(R)
    Nt == length(times) || error("step_to_impulse time mismatch")
    Nt >= 2 || error("Need at least two points")
    out = zeros(Float64, M, P, Nt)
    @inbounds for i in 1:M, j in 1:P
        out[i, j, 1] = (R[i, j, 2] - R[i, j, 1]) / (times[2] - times[1])
        for t in 2:(Nt - 1)
            out[i, j, t] = (R[i, j, t + 1] - R[i, j, t - 1]) / (times[t + 1] - times[t - 1])
        end
        out[i, j, Nt] = (R[i, j, Nt] - R[i, j, Nt - 1]) / (times[Nt] - times[Nt - 1])
    end
    return out
end

function build_score_correction(M::Matrix{Float64}; ridge::Float64, max_deviation::Float64)
    D = size(M, 1)
    size(M, 2) == D || error("Correction matrix must be square")
    Iden = Matrix{Float64}(I, D, D)
    pre_err = norm(M .- Iden) / sqrt(Float64(D))

    Mreg = copy(M)
    @inbounds for i in 1:D
        Mreg[i, i] += ridge
    end

    Tdirect = Mreg \ Iden
    dev_direct = norm(Tdirect .- Iden) / sqrt(Float64(D))
    post_direct = norm(M * Tdirect .- Iden) / sqrt(Float64(D))

    Tpinv = pinv(Mreg)
    dev_pinv = norm(Tpinv .- Iden) / sqrt(Float64(D))
    post_pinv = norm(M * Tpinv .- Iden) / sqrt(Float64(D))

    bestT = post_pinv < post_direct ? Tpinv : Tdirect
    bestPost = min(post_direct, post_pinv)
    bestDev = post_pinv < post_direct ? dev_pinv : dev_direct

    if isfinite(bestPost) && bestPost < pre_err && isfinite(bestDev) && bestDev <= max_deviation
        return bestT, (solver=post_pinv < post_direct ? "pinv" : "direct", post_identity_rmse=bestPost)
    end
    return Iden, (solver="identity_fallback", post_identity_rmse=pre_err)
end

Base.@kwdef struct UnetBundle
    model
    device
    sigma_train::Float32
    mean_lc::Array{Float32,2}
    std_lc::Array{Float32,2}
    std_lc64::Array{Float64,2}
end

function load_unet_bundle(checkpoint_path::AbstractString, device_pref::AbstractString, forward_mode::AbstractString)
    ckpt = BSON.load(checkpoint_path)
    model = ckpt[:model]
    stats = ckpt[:stats]
    trainer_cfg = ckpt[:trainer_cfg]

    mean_lc = Float32.(permutedims(stats.mean, (2, 1)))
    std_lc = Float32.(permutedims(stats.std, (2, 1)))

    device = try
        d = select_device(device_pref)
        activate_device!(d)
        d
    catch
        d = CPUDevice()
        activate_device!(d)
        d
    end

    model_dev = move_model(model, device)
    lowercase(forward_mode) == "train" ? Flux.trainmode!(model_dev) : Flux.testmode!(model_dev)

    return UnetBundle(
        model=model_dev,
        device=device,
        sigma_train=Float32(getproperty(trainer_cfg, :sigma)),
        mean_lc=mean_lc,
        std_lc=std_lc,
        std_lc64=Float64.(std_lc),
    )
end

function compute_unet_score_batch(bundle::UnetBundle, batch_phys::Array{Float64,3})
    K, C, B = size(batch_phys)
    batch_norm = Array{Float32,3}(undef, K, C, B)
    @inbounds for ib in 1:B, c in 1:C, k in 1:K
        batch_norm[k, c, ib] = (Float32(batch_phys[k, c, ib]) - bundle.mean_lc[k, c]) / bundle.std_lc[k, c]
    end

    score_norm = if is_gpu(bundle.device)
        dev_batch = move_array(batch_norm, bundle.device)
        Array(score_from_model(bundle.model, dev_batch, bundle.sigma_train))
    else
        score_from_model(bundle.model, batch_norm, bundle.sigma_train)
    end

    score_phys = Array{Float64,3}(undef, K, C, B)
    @inbounds for ib in 1:B, c in 1:C, k in 1:K
        score_phys[k, c, ib] = Float64(score_norm[k, c, ib]) / bundle.std_lc64[k, c]
    end
    return score_phys
end

function compute_G_FxFy_from_score_batch!(G::Matrix{Float64}, score_phys::Array{Float64,3}, out_start::Int)
    K, C, B = size(score_phys)
    J = C - 1
    @inbounds for ib in 1:B
        gFx = 0.0
        gFy = 0.0
        for k in 1:K
            gFx -= score_phys[k, 1, ib]
            for j in 1:J
                gFy -= score_phys[k, j + 1, ib]
            end
        end
        n = out_start + ib - 1
        G[1, n] = gFx
        G[2, n] = gFy
    end
    return nothing
end

function compute_unet_conjugates_FxFy(tensor::Array{Float64,3}, checkpoint_path::AbstractString; batch_size::Int, device_pref::AbstractString, forward_mode::AbstractString, correction_ridge::Float64, correction_max_deviation::Float64, compute_raw::Bool=true, compute_corrected::Bool=true, correction_fit_nsamples::Int=0)
    (compute_raw || compute_corrected) || error("At least one of compute_raw/compute_corrected must be true")
    bundle = load_unet_bundle(checkpoint_path, device_pref, forward_mode)
    K, C, N = size(tensor)
    D = K * C
    mean64 = Float64.(bundle.mean_lc)
    std64 = bundle.std_lc64
    invstd64 = 1.0 ./ std64

    G_raw = compute_raw ? zeros(Float64, 2, N) : nothing
    M_acc_norm = compute_corrected ? zeros(Float64, D, D) : nothing

    fit_N = N
    if compute_corrected && !compute_raw && correction_fit_nsamples > 0
        fit_N = min(correction_fit_nsamples, N)
    end

    pass1_desc = compute_raw && compute_corrected ?
        "UNet pass 1 (raw conjugates + correction matrix)..." :
        (compute_corrected ? "UNet pass 1 (correction matrix)..." : "UNet pass 1 (raw conjugates)...")

    @showprogress pass1_desc for start in 1:batch_size:fit_N
        stop = min(start + batch_size - 1, fit_N)
        b = stop - start + 1
        idx = start:stop
        batch = Array(@view tensor[:, :, idx])
        score_phys = compute_unet_score_batch(bundle, batch)

        if compute_raw
            compute_G_FxFy_from_score_batch!(G_raw, score_phys, start)
        end

        if compute_corrected
            z_chunk = Matrix{Float64}(undef, D, b)
            score_norm_flat = Matrix{Float64}(undef, D, b)
            @inbounds for ib in 1:b, c in 1:C, k in 1:K
                d = k + (c - 1) * K
                z_chunk[d, ib] = (batch[k, c, ib] - mean64[k, c]) * invstd64[k, c]
                score_norm_flat[d, ib] = score_phys[k, c, ib] * std64[k, c]
            end
            BLAS.gemm!('N', 'T', -1.0, z_chunk, score_norm_flat, 1.0, M_acc_norm)
        end
    end

    Tcorr = Matrix{Float64}(I, D, D)
    corr_stats = (solver="identity_fallback", post_identity_rmse=NaN)
    At = transpose(Tcorr)
    if compute_corrected
        M_raw = M_acc_norm ./ max(fit_N, 1)
        Tcorr, corr_stats = build_score_correction(M_raw; ridge=correction_ridge, max_deviation=correction_max_deviation)
        At = transpose(Tcorr)
    end

    G_corr = nothing
    if compute_corrected
        G_corr = zeros(Float64, 2, N)
        @showprogress "UNet pass 2 (corrected conjugates)..." for start in 1:batch_size:N
            stop = min(start + batch_size - 1, N)
            b = stop - start + 1
            idx = start:stop
            batch = Array(@view tensor[:, :, idx])
            score_phys = compute_unet_score_batch(bundle, batch)

            score_norm_flat = Matrix{Float64}(undef, D, b)
            @inbounds for ib in 1:b, c in 1:C, k in 1:K
                d = k + (c - 1) * K
                score_norm_flat[d, ib] = score_phys[k, c, ib] * std64[k, c]
            end
            score_corr_norm_flat = Matrix{Float64}(undef, D, b)
            mul!(score_corr_norm_flat, At, score_norm_flat)

            score_corr_phys_flat = Matrix{Float64}(undef, D, b)
            @inbounds for ib in 1:b, c in 1:C, k in 1:K
                d = k + (c - 1) * K
                score_corr_phys_flat[d, ib] = score_corr_norm_flat[d, ib] * invstd64[k, c]
            end
            score_corr = reshape(score_corr_phys_flat, K, C, b)
            compute_G_FxFy_from_score_batch!(G_corr, score_corr, start)
        end
    end

    return (G_raw=G_raw, G_corr=G_corr, correction_stats=corr_stats)
end

function make_l96_workspace(K::Int, J::Int)
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
    )
end

function l96_drift_modified!(dx::Vector{Float64}, dy::Matrix{Float64}, x::Vector{Float64}, y::Matrix{Float64}, cfg::L96FxFyConfig)
    K, J = cfg.K, cfg.J
    Fx, Fy, h, c, b = cfg.Fx, cfg.Fy, cfg.h, cfg.c, cfg.b
    coupling_scale = h * c / J

    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)
        coupling = coupling_scale * sum(@view y[:, k])
        dx[k] = x[km1] * (x[kp1] - x[km2]) - x[k] + Fx - coupling
    end

    @inbounds for k in 1:K
        xk_term = coupling_scale * x[k]
        for j in 1:J
            ell = l96_linidx(j, k, J)
            ellp1 = l96_wrap_lin(ell + 1, K, J)
            ellm1 = l96_wrap_lin(ell - 1, K, J)
            ellp2 = l96_wrap_lin(ell + 2, K, J)
            jp1, kp1 = l96_jk_from_lin(ellp1, J)
            jm1, km1 = l96_jk_from_lin(ellm1, J)
            jp2, kp2 = l96_jk_from_lin(ellp2, J)

            y_p1 = y[jp1, kp1]
            y_m1 = y[jm1, km1]
            y_p2 = y[jp2, kp2]
            dy[j, k] = c * b * y_p1 * (y_m1 - y_p2) - c * y[j, k] + xk_term + Fy
        end
    end
    return nothing
end

function rk4_step_modified!(x::Vector{Float64}, y::Matrix{Float64}, dt::Float64, ws, cfg::L96FxFyConfig)
    l96_drift_modified!(ws.dx1, ws.dy1, x, y, cfg)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    l96_drift_modified!(ws.dx2, ws.dy2, ws.xtmp, ws.ytmp, cfg)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    l96_drift_modified!(ws.dx3, ws.dy3, ws.xtmp, ws.ytmp, cfg)

    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    l96_drift_modified!(ws.dx4, ws.dy4, ws.xtmp, ws.ytmp, cfg)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    return nothing
end

function add_process_noise!(x::Vector{Float64}, y::Matrix{Float64}, rng::AbstractRNG, cfg::L96FxFyConfig)
    σ = cfg.process_noise_sigma * sqrt(cfg.dt)
    σ == 0.0 && return nothing
    @inbounds begin
        if cfg.stochastic_x_noise
            for k in eachindex(x)
                x[k] += σ * randn(rng)
            end
        end
        for idx in eachindex(y)
            y[idx] += σ * randn(rng)
        end
    end
    return nothing
end

function snapshot_to_xy!(x0::Vector{Float64}, y0::Matrix{Float64}, tensor::Array{Float64,3}, idx::Int)
    K = length(x0)
    J = size(y0, 1)
    @inbounds for k in 1:K
        x0[k] = tensor[k, 1, idx]
        for j in 1:J
            y0[j, k] = tensor[k, j + 1, idx]
        end
    end
    return nothing
end

function accumulate_observables!(out::Vector{Float64}, x::Vector{Float64}, y::Matrix{Float64})
    fill!(out, 0.0)
    K = length(x)
    J = size(y, 1)
    invK = 1.0 / K
    invJ = 1.0 / J

    @inbounds for k in 1:K
        km1 = (k == 1) ? K : (k - 1)
        xk = x[k]
        xkm1 = x[km1]
        ysum = 0.0
        y2sum = 0.0
        for j in 1:J
            yy = y[j, k]
            ysum += yy
            y2sum += yy * yy
        end
        ybar = ysum * invJ
        y2bar = y2sum * invJ
        out[1] += xk
        out[2] += xk * xk
        out[3] += xk * ybar
        out[4] += y2bar
        out[5] += xk * xkm1
    end
    @. out *= invK
    return nothing
end

function simulate_observable_series!(out::Matrix{Float64}, cfg::L96FxFyConfig, x0::Vector{Float64}, y0::Matrix{Float64}, n_lags::Int, rng::AbstractRNG, ws)
    x = copy(x0)
    y = copy(y0)
    acc = zeros(Float64, 5)

    accumulate_observables!(acc, x, y)
    @views out[:, 1] .= acc

    @inbounds for lag in 1:n_lags
        for _ in 1:cfg.save_every
            rk4_step_modified!(x, y, cfg.dt, ws, cfg)
            add_process_noise!(x, y, rng, cfg)
        end
        accumulate_observables!(acc, x, y)
        @views out[:, lag + 1] .= acc
    end
    return nothing
end

function compute_numerical_responses_FxFy_fd(cfg_base::L96FxFyConfig, init_tensor::Array{Float64,3}, n_lags::Int; h_rel::Float64, h_abs_Fx::Float64, h_abs_Fy::Float64, seed_base::Int)
    _, C, nens = size(init_tensor)
    C == cfg_base.J + 1 || error("Channel mismatch for initial tensor")

    responses = zeros(Float64, 5, 2, n_lags + 1)

    pert_names = ("Fx", "Fy")
    h_abs = (h_abs_Fx, h_abs_Fy)

    for ip in 1:2
        base_val = ip == 1 ? cfg_base.Fx : cfg_base.Fy
        h = max(h_abs[ip], h_rel * max(abs(base_val), 1.0))
        @info "Numerical FD perturbation" parameter = pert_names[ip] base = base_val step = h

        cfg_p = L96FxFyConfig(cfg_base.K, cfg_base.J,
            ip == 1 ? cfg_base.Fx + h : cfg_base.Fx,
            ip == 2 ? cfg_base.Fy + h : cfg_base.Fy,
            cfg_base.h, cfg_base.c, cfg_base.b,
            cfg_base.dt, cfg_base.save_every, cfg_base.process_noise_sigma, cfg_base.stochastic_x_noise,
            cfg_base.dataset_path, cfg_base.dataset_key)

        cfg_m = L96FxFyConfig(cfg_base.K, cfg_base.J,
            ip == 1 ? cfg_base.Fx - h : cfg_base.Fx,
            ip == 2 ? cfg_base.Fy - h : cfg_base.Fy,
            cfg_base.h, cfg_base.c, cfg_base.b,
            cfg_base.dt, cfg_base.save_every, cfg_base.process_noise_sigma, cfg_base.stochastic_x_noise,
            cfg_base.dataset_path, cfg_base.dataset_key)

        partials = [zeros(Float64, 5, n_lags + 1) for _ in 1:nthreads()]

        @info "Numerical responses loop" parameter = pert_names[ip] ensembles = nens
        Threads.@threads for ens in 1:nens
            tid = threadid()
            part = partials[tid]

            x0 = zeros(Float64, cfg_base.K)
            y0 = zeros(Float64, cfg_base.J, cfg_base.K)
            out_p = zeros(Float64, 5, n_lags + 1)
            out_m = zeros(Float64, 5, n_lags + 1)
            ws_p = make_l96_workspace(cfg_base.K, cfg_base.J)
            ws_m = make_l96_workspace(cfg_base.K, cfg_base.J)
            rng = MersenneTwister(seed_base + 1_000_000 * ip + ens)

            snapshot_to_xy!(x0, y0, init_tensor, ens)

            Random.seed!(rng, seed_base + 1_000_000 * ip + ens)
            simulate_observable_series!(out_p, cfg_p, x0, y0, n_lags, rng, ws_p)

            Random.seed!(rng, seed_base + 1_000_000 * ip + ens)
            simulate_observable_series!(out_m, cfg_m, x0, y0, n_lags, rng, ws_m)

            @inbounds for m in 1:5, lag in 1:(n_lags + 1)
                part[m, lag] += (out_p[m, lag] - out_m[m, lag]) / (2h)
            end
        end

        local_sum = zeros(Float64, 5, n_lags + 1)
        for p in partials
            local_sum .+= p
        end
        local_sum ./= max(nens, 1)
        @views responses[:, ip, :] .= local_sum
    end

    return responses
end

function save_plot_5x2(path::AbstractString, times::Vector{Float64}, R_num::Array{Float64,3}, R_unet::Array{Float64,3}; unet_label::AbstractString)
    labels_obs = [
        "phi1 = <X>",
        "phi2 = <X^2>",
        "phi3 = <X*Ybar>",
        "phi4 = <Y^2>",
        "phi5 = <X_k X_(k-1)>",
    ]
    params = ["Fx", "Fy"]

    default(fontfamily="Computer Modern", dpi=180, legendfontsize=8, guidefontsize=9, tickfontsize=8, titlefontsize=10)
    panels = Vector{Plots.Plot}(undef, 10)

    for i in 1:5, j in 1:2
        idx = (i - 1) * 2 + j
        legend_mode = (idx == 1 ? :topright : false)
        title_txt = i == 1 ? "d/d" * params[j] : ""
        ylabel_txt = j == 1 ? labels_obs[i] : ""
        xlabel_txt = i == 5 ? "time" : ""

        pn = plot(times, vec(@view R_num[i, j, :]);
            color=:dodgerblue3,
            linewidth=2.2,
            label=(idx == 1 ? "Numerical FD (impulse)" : ""),
            legend=legend_mode,
            title=title_txt,
            ylabel=ylabel_txt,
            xlabel=xlabel_txt)
        plot!(pn, times, vec(@view R_unet[i, j, :]);
            color=:orangered3,
            linewidth=2.0,
            label=(idx == 1 ? unet_label : ""))
        hline!(pn, [0.0]; color=:gray55, linestyle=:dot, linewidth=1.0, label="")
        panels[idx] = pn
    end

    fig = plot(panels...; layout=(5, 2), size=(1600, 2000))
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function main()
    mode = resolve_score_response_mode(SCORE_RESPONSE_MODE)
    want_raw = (mode == :raw || mode == :both)
    want_corr = (mode == :corrected || mode == :both)

    cfg, opts = load_config_from_params(PARAMS_TOML)
    cfg = sync_with_dataset_attrs(cfg)

    @info "Modified L96 setup" K = cfg.K J = cfg.J Fx = cfg.Fx Fy = cfg.Fy h = cfg.h c = cfg.c b = cfg.b dt = cfg.dt save_every = cfg.save_every dataset = cfg.dataset_path
    @info "UNet checkpoint" path = opts["checkpoint"]
    @info "UNet response mode" mode = String(mode)
    @info "Correction fit samples" requested = CORRECTION_FIT_NSAMPLES

    delta_t_obs = cfg.dt * cfg.save_every
    n_lags_req = max(1, Int(floor(opts["response_tmax"] / delta_t_obs)))
    if delta_t_obs >= 0.05
        @warn "Observed cadence is coarse for fast-Y response kernels; using trapezoidal GFDT integration and finer numerical save_every override." delta_t_obs c=cfg.c
    end

    tensor_gfdt = load_observation_subset(cfg; nsamples=opts["gfdt_nsamples"], start_index=opts["gfdt_start"], label="gfdt")
    n_lags = min(n_lags_req, size(tensor_gfdt, 3) - 1)
    n_lags >= 1 || error("Need at least 2 GFDT samples")

    A = compute_global_observables(tensor_gfdt)
    unet = compute_unet_conjugates_FxFy(
        tensor_gfdt,
        opts["checkpoint"];
        batch_size=opts["score_batch_size"],
        device_pref=opts["score_device"],
        forward_mode=opts["score_forward_mode"],
        correction_ridge=opts["correction_ridge"],
        correction_max_deviation=opts["correction_max_deviation"],
        compute_raw=want_raw,
        compute_corrected=want_corr,
        correction_fit_nsamples=CORRECTION_FIT_NSAMPLES,
    )
    C_unet_raw = nothing
    C_unet_corr = nothing
    times_gfdt = nothing
    if want_raw
        C_unet_raw, _, times_gfdt = build_gfdt_response(A, unet.G_raw, delta_t_obs, n_lags; mean_center=opts["mean_center"])
    end
    if want_corr
        C_unet_corr, _, times_gfdt_corr = build_gfdt_response(A, unet.G_corr, delta_t_obs, n_lags; mean_center=opts["mean_center"])
        times_gfdt === nothing && (times_gfdt = times_gfdt_corr)
    end

    h_abs = opts["h_abs"]
    h_abs_fx = length(h_abs) >= 1 ? h_abs[1] : 0.05
    h_abs_fy = h_abs_fx

    response_save_every = min(cfg.save_every, 2)
    if response_save_every < cfg.save_every
        @info "Applying finer numerical response cadence for fast-Y quadrature" old_save_every = cfg.save_every new_save_every = response_save_every
    end
    cfg_num = L96FxFyConfig(cfg.K, cfg.J, cfg.Fx, cfg.Fy, cfg.h, cfg.c, cfg.b, cfg.dt, response_save_every, cfg.process_noise_sigma, cfg.stochastic_x_noise, cfg.dataset_path, cfg.dataset_key)
    delta_t_num = cfg_num.dt * cfg_num.save_every
    n_lags_num = max(1, Int(floor(opts["response_tmax"] / delta_t_num)))

    init_tensor = load_observation_subset(cfg_num; nsamples=opts["num_ensembles"], start_index=opts["num_start"], label="numerical")
    R_num = compute_numerical_responses_FxFy_fd(
        cfg_num,
        init_tensor,
        n_lags_num;
        h_rel=opts["h_rel"],
        h_abs_Fx=h_abs_fx,
        h_abs_Fy=h_abs_fy,
        seed_base=opts["num_seed"],
    )
    times_num = collect(0:n_lags_num) .* delta_t_num
    I_num = step_to_impulse(R_num, times_num)

    tmax_eff = min(times_gfdt[end], times_num[end])
    times_out = collect(range(0.0, stop=tmax_eff, length=301))

    I_num_out = linear_interpolate_3D_time(I_num, times_num, times_out)
    if want_raw
        I_unet_raw_out = linear_interpolate_3D_time(C_unet_raw, times_gfdt, times_out)
        out_raw = save_plot_5x2(OUTPUT_PLOT_RAW, times_out, I_num_out, I_unet_raw_out; unet_label="GFDT+UNet raw (impulse)")
        println("Saved raw impulse plot: ", abspath(out_raw))
    end
    if want_corr
        I_unet_corr_out = linear_interpolate_3D_time(C_unet_corr, times_gfdt, times_out)
        out_corr = save_plot_5x2(OUTPUT_PLOT_CORR, times_out, I_num_out, I_unet_corr_out; unet_label="GFDT+UNet corrected (impulse)")
        println("Saved corrected impulse plot: ", abspath(out_corr))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
