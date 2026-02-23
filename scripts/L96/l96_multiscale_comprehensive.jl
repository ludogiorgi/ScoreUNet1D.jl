#!/usr/bin/env julia

# ==============================================================================
# Terminal command to run this file:
# julia --project=. scripts/L96/l96_multiscale_comprehensive.jl
# ==============================================================================

using Plots
using Random
using Statistics
using Printf

# Parameters (matching the currently configured runs/values)
const K = 36
const J = 10
const F = 9.0
const G = 0.0
const h = 1.0
const c = 10.0
const b = 10.0
const dt = 0.005
const process_noise_sigma = 0.0
const stochastic_x_noise = false

# Helpers for indices
@inline mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1
@inline l96_linidx(j::Int, k::Int, J::Int) = (k - 1) * J + j
@inline function l96_jk_from_lin(ell::Int, J::Int)
    kk = (ell - 1) ÷ J + 1
    jj = (ell - 1) % J + 1
    return jj, kk
end
@inline l96_wrap_lin(ell::Int, K::Int, J::Int) = mod1idx(ell, K * J)

function l96_drift!(dx::Vector{Float64}, dy::Matrix{Float64}, x::Vector{Float64}, y::Matrix{Float64})
    coupling_scale = h * c / J

    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)
        coupling = 0.0
        for j in 1:J
            coupling += y[j, k]
        end
        coupling *= coupling_scale
        dx[k] = x[km1] * (x[kp1] - x[km2]) - x[k] + F - coupling
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
            dy[j, k] = c * b * y_p1 * (y_m1 - y_p2) - c * y[j, k] + xk_term + G
        end
    end
    return nothing
end

function rk4_step!(x::Vector{Float64}, y::Matrix{Float64}, dt::Float64, ws)
    l96_drift!(ws.dx1, ws.dy1, x, y)
    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    l96_drift!(ws.dx2, ws.dy2, ws.xtmp, ws.ytmp)
    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    l96_drift!(ws.dx3, ws.dy3, ws.xtmp, ws.ytmp)
    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    l96_drift!(ws.dx4, ws.dy4, ws.xtmp, ws.ytmp)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    return nothing
end

function add_process_noise!(x::Vector{Float64}, y::Matrix{Float64}, rng::AbstractRNG)
    σ = process_noise_sigma * sqrt(dt)
    σ == 0.0 && return nothing
    @inbounds begin
        if stochastic_x_noise
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

function main()
    ws = (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
    )

    rng = MersenneTwister(42)
    x = randn(rng, K)
    y = randn(rng, J, K)

    # Spinup
    println("Spinning up...")
    spinup_steps = 20_000
    for _ in 1:spinup_steps
        rk4_step!(x, y, dt, ws)
        add_process_noise!(x, y, rng)
    end

    # Collection
    println("Collecting data...")
    nsamples = 100_000
    save_every = 10
    total_steps = nsamples * save_every

    X_hist = zeros(Float64, K, nsamples)
    Y_hist = zeros(Float64, J, K, nsamples)

    for i in 1:nsamples
        for _ in 1:save_every
            rk4_step!(x, y, dt, ws)
            add_process_noise!(x, y, rng)
        end
        X_hist[:, i] .= x
        Y_hist[:, :, i] .= y
    end

    Y_hist1 = Y_hist[:, 1, :]
    Y_mean_k = reshape(mean(Y_hist, dims=1), K, nsamples)

    mean_X = mean(X_hist)
    std_X = std(X_hist)
    mean_Y = [mean(view(Y_hist, j, :, :)) for j in 1:J]
    std_Y = [std(view(Y_hist, j, :, :)) for j in 1:J]

    println("Plotting...")
    if get(ENV, "GKSwstype", "") == ""
        ENV["GKSwstype"] = "100"
    end
    default(fontfamily="Computer Modern", dpi=150)

    # 1. Hovmoller X (first 500 samples)
    t_plot = min(500, nsamples)
    t_axis = (1:t_plot) .* (dt * save_every)
    p1 = heatmap(t_axis, 1:K, X_hist[:, 1:t_plot],
        xlabel="Time", ylabel="k", title="Hovmöller: X",
        color=:viridis)

    # 2. Hovmoller Y for k=1 (first 500 samples)
    p2 = heatmap(t_axis, 1:J, Y_hist1[:, 1:t_plot],
        xlabel="Time", ylabel="j", title="Hovmöller: Y (k=1)",
        color=:plasma)

    # 3. Univariate PDF of X
    X_flat = vec(X_hist)
    p3 = histogram(X_flat, normalize=:pdf, bins=100,
        xlabel="X", ylabel="Density", title="PDF of X",
        color=:dodgerblue, alpha=0.7, legend=false)

    # 4. Univariate PDF of Y
    Y_flat = vec(Y_hist1)
    p4 = histogram(Y_flat, normalize=:pdf, bins=100,
        xlabel="Y", ylabel="Density", title="PDF of Y",
        color=:orangered, alpha=0.7, legend=false)

    # 5. Bivariate PDF: X_k vs X_{k+1}
    X_kp1 = circshift(X_hist, (-1, 0))
    Xk_flat = vec(X_hist)
    Xkp1_flat = vec(X_kp1)
    p5 = histogram2d(Xk_flat, Xkp1_flat, bins=(50, 50), normalize=:pdf,
        xlabel="X_k", ylabel="X_{k+1}", title="PDF(X_k, X_{k+1})",
        color=:cividis)

    # 6. Bivariate PDF: Y_k vs Y_{k+1} (using Mean Y_k)
    Ym_kp1 = circshift(Y_mean_k, (-1, 0))
    Ymk_flat = vec(Y_mean_k)
    Ymkp1_flat = vec(Ym_kp1)
    p6 = histogram2d(Ymk_flat, Ymkp1_flat, bins=(50, 50), normalize=:pdf,
        xlabel="Mean Y_k", ylabel="Mean Y_{k+1}", title="PDF(Y_k, Y_{k+1})",
        color=:cividis)

    # 7. Bivariate PDF: X_k vs Y_k
    p7 = histogram2d(Xk_flat, Ymk_flat, bins=(50, 50), normalize=:pdf,
        xlabel="X_k", ylabel="Mean Y_k", title="PDF(X_k, Y_k)",
        color=:cividis)

    # 8. Bivariate PDF: X_k vs Y_{k+1}
    p8 = histogram2d(Xk_flat, Ymkp1_flat, bins=(50, 50), normalize=:pdf,
        xlabel="X_k", ylabel="Mean Y_{k+1}", title="PDF(X_k, Y_{k+1})",
        color=:cividis)

    # 9. Text Panel for Means & Stds
    stat_text = "Statistics (Mean ± Std):\n\n"
    stat_text *= @sprintf("X   : %8.4f ± %8.4f\n", mean_X, std_X)
    for j in 1:J
        stat_text *= @sprintf("Y_%-2d: %8.4f ± %8.4f\n", j, mean_Y[j], std_Y[j])
    end
    p9 = plot(axis=false, grid=false, showaxis=false, xticks=false, yticks=false)
    annotate!(p9, 0.1, 0.5, text(stat_text, 10, :left, :center, family="monospace"))

    fig = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout=(3, 3), size=(1500, 1300), margin=5Plots.mm)

    out_path = "scripts/L96/l96_multiscale_comprehensive.png"
    savefig(fig, out_path)
    println("Saved figure to ", abspath(out_path))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
