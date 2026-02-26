module ArnoldStatsPlots

using Plots
using Printf
using LinearAlgebra
using Statistics
using StatsBase

export modewise_metrics,
    autocorrelation,
    average_mode_acf,
    save_stats_figure_stein,
    save_stats_figure_acf

@inline mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

function with_padding(lo::Float64, hi::Float64)
    if lo == hi
        d = max(abs(lo), 1.0) * 1e-3
        return lo - d, hi + d
    end
    return lo, hi
end

function histogram_prob(samples::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, samples, edges)
    probs = Float64.(hist.weights)
    probs .+= eps(Float64)
    probs ./= sum(probs)
    return probs
end

@inline discrete_kl(p::Vector{Float64}, q::Vector{Float64}) = sum(@. p * log(p / q))

function modewise_metrics(obs::AbstractArray{<:Real,3}, gen::AbstractArray{<:Real,3};
    nbins::Int,
    low_q::Float64,
    high_q::Float64)
    K, C, _ = size(obs)
    C == 1 || error("Expected one channel")
    size(gen, 1) == K || error("Mode count mismatch")
    size(gen, 2) == C || error("Channel count mismatch")
    kl_mode = zeros(Float64, K)
    js_mode = zeros(Float64, K)
    for k in 1:K
        ov = Float64.(vec(@view obs[k, 1, :]))
        gv = Float64.(vec(@view gen[k, 1, :]))
        combo = vcat(ov, gv)
        lo = quantile(combo, low_q)
        hi = quantile(combo, high_q)
        lo, hi = with_padding(lo, hi)
        edges = collect(range(lo, hi; length=nbins + 1))
        p = histogram_prob(ov, edges)
        q = histogram_prob(gv, edges)
        m = 0.5 .* (p .+ q)
        kl_mode[k] = discrete_kl(p, q)
        js_mode[k] = 0.5 * discrete_kl(p, m) + 0.5 * discrete_kl(q, m)
    end
    return kl_mode, js_mode
end

function collect_bivariate_pairs(tensor::AbstractArray{<:Real,3}, lag::Int)
    K, C, T = size(tensor)
    C == 1 || error("Expected one channel")
    lag >= 1 || error("lag must be >= 1")

    xvals = Vector{Float64}(undef, K * T)
    yvals = Vector{Float64}(undef, K * T)
    idx = 1
    for k in 1:K
        kp = mod1idx(k + lag, K)
        @inbounds for t in 1:T
            xvals[idx] = Float64(tensor[k, 1, t])
            yvals[idx] = Float64(tensor[kp, 1, t])
            idx += 1
        end
    end
    return xvals, yvals
end

function bivariate_histogram_density(x::Vector{Float64}, y::Vector{Float64}, edges::Vector{Float64})
    hist = fit(Histogram, (x, y), (edges, edges))
    dens = Float64.(hist.weights)
    dens .+= eps(Float64)
    dens ./= sum(dens)
    return dens
end

function histogram_pdf_curve(samples::Vector{Float64}, edges::Vector{Float64})
    probs = histogram_prob(samples, edges)
    widths = diff(edges)
    all(w -> w > 0.0, widths) || error("Histogram edges must be strictly increasing")
    dens = probs ./ widths
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, dens
end

function contour_levels_from_pair(obs_dens::Matrix{Float64}, gen_dens::Matrix{Float64})
    v = sort(vcat(vec(obs_dens), vec(gen_dens)))
    isempty(v) && return [1.0]
    levels = [quantile(v, q) for q in (0.65, 0.8, 0.9, 0.97)]
    levels = unique(sort(filter(isfinite, levels)))
    isempty(levels) && return [maximum(v)]
    if length(levels) == 1
        l1 = levels[1]
        return [l1, l1 + max(abs(l1) * 0.05, 1e-12)]
    end
    return levels
end

function autocorrelation(series::Vector{Float64}, max_lag::Int)
    n = length(series)
    n <= 1 && return ones(Float64, 1)
    m = min(max(max_lag, 1), n - 1)
    centered = series .- mean(series)
    variance = sum(abs2, centered) / n
    variance <= eps(Float64) && return ones(Float64, m + 1)
    acf = Array{Float64}(undef, m + 1)
    acf[1] = 1.0
    @inbounds for lag in 1:m
        t = n - lag
        acf[lag + 1] = dot(view(centered, 1:t), view(centered, lag + 1:lag + t)) / (t * variance)
    end
    return acf
end

function average_mode_acf(tensor::AbstractArray{<:Real,3}, max_lag::Int)
    K, C, T = size(tensor)
    C == 1 || error("Expected one channel")
    T <= 1 && return ones(Float64, 1)
    lag_used = min(max(max_lag, 1), T - 1)
    acc = zeros(Float64, lag_used + 1)
    for k in 1:K
        acc .+= autocorrelation(Float64.(vec(@view tensor[k, 1, :])), lag_used)
    end
    return acc ./ K
end

function observable_values(tensor::AbstractArray{<:Real,3})
    K, C, T = size(tensor)
    C == 1 || error("Expected one channel")
    T >= 1 || error("Expected at least one timestep")
    vals = zeros(Float64, 5)
    @inbounds for t in 1:T
        for k in 1:K
            km1 = mod1idx(k - 1, K)
            km2 = mod1idx(k - 2, K)
            km3 = mod1idx(k - 3, K)

            xk = Float64(tensor[k, 1, t])
            xk1 = Float64(tensor[km1, 1, t])
            xk2 = Float64(tensor[km2, 1, t])
            xk3 = Float64(tensor[km3, 1, t])
            vals[1] += xk
            vals[2] += xk * xk
            vals[3] += xk * xk1
            vals[4] += xk * xk2
            vals[5] += xk * xk3
        end
    end
    vals ./= (K * T)
    return vals
end

function _stats_first_7_panels(obs::AbstractArray{<:Real,3},
    gen::AbstractArray{<:Real,3},
    kl_mode::Vector{Float64},
    js_mode::Vector{Float64},
    bins::Int;
    obs_label::String="Observed",
    gen_label::String="Generated")
    K = size(obs, 1)
    _, C, _ = size(obs)
    C == 1 || error("Expected one channel")

    obs_vals = Float64.(vec(obs))
    gen_vals = Float64.(vec(gen))

    combo_vals = vcat(obs_vals, gen_vals)
    pdf_lo = quantile(combo_vals, 0.001)
    pdf_hi = quantile(combo_vals, 0.999)
    pdf_lo, pdf_hi = with_padding(pdf_lo, pdf_hi)
    pdf_edges = collect(range(pdf_lo, pdf_hi; length=bins + 1))
    centers_obs, dens_obs = histogram_pdf_curve(obs_vals, pdf_edges)
    centers_gen, dens_gen = histogram_pdf_curve(gen_vals, pdf_edges)

    p_pdf = plot(
        centers_obs,
        dens_obs;
        color=:dodgerblue3,
        linewidth=3.0,
        fillrange=0.0,
        fillalpha=0.14,
        label=obs_label,
        xlabel="X",
        ylabel="PDF",
        title="Univariate PDF",
    )
    plot!(
        p_pdf,
        centers_gen,
        dens_gen;
        color=:tomato3,
        linewidth=3.0,
        linestyle=:dash,
        fillrange=0.0,
        fillalpha=0.10,
        label=gen_label,
    )

    probs = collect(range(0.001, 0.999; length=min(length(obs_vals), length(gen_vals), 6000)))
    q_obs = quantile(obs_vals, probs)
    q_gen = quantile(gen_vals, probs)
    lo = min(minimum(q_obs), minimum(q_gen))
    hi = max(maximum(q_obs), maximum(q_gen))
    lo, hi = with_padding(lo, hi)
    p_qq = scatter(q_obs, q_gen; markersize=2, alpha=0.55, color=:darkslateblue, label="quantiles", xlabel="Observed quantiles", ylabel="Generated quantiles", title="QQ")
    plot!(p_qq, [lo, hi], [lo, hi]; color=:black, linestyle=:dash, label="y=x")

    # Observable panel: two points per observable index, one per compared method.
    obs_obsvals = observable_values(obs)
    gen_obsvals = observable_values(gen)

    obs_idx = collect(1:5)
    p_mom = scatter(
        obs_idx .- 0.08,
        obs_obsvals;
        markersize=6.5,
        marker=:circle,
        color=:dodgerblue3,
        label=obs_label,
        xlabel="Observable index",
        ylabel="Value",
        title="Observable values comparison",
        xticks=(obs_idx, ["phi1", "phi2", "phi3", "phi4", "phi5"]),
    )
    scatter!(
        p_mom,
        obs_idx .+ 0.08,
        gen_obsvals;
        markersize=7.0,
        marker=:diamond,
        color=:tomato3,
        label=gen_label,
    )

    p_kl = bar(1:K, kl_mode; color=:orangered3, label="KL", xlabel="Mode k", ylabel="KL", title=@sprintf("Mode KL/JS (avg KL=%.4f, avg JS=%.4f)", mean(kl_mode), mean(js_mode)))
    plot!(p_kl, 1:K, js_mode; color=:black, marker=:circle, linewidth=2, label="JS")

    p_biv = Vector{Plots.Plot}(undef, 3)
    for (idx_lag, lag) in enumerate((1, 2, 3))
        ox, oy = collect_bivariate_pairs(obs, lag)
        gx, gy = collect_bivariate_pairs(gen, lag)
        combo = vcat(ox, oy, gx, gy)
        lo = quantile(combo, 0.001)
        hi = quantile(combo, 0.999)
        lo, hi = with_padding(lo, hi)
        edges = collect(range(lo, hi; length=bins + 1))
        centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])

        obs_dens = bivariate_histogram_density(ox, oy, edges)
        gen_dens = bivariate_histogram_density(gx, gy, edges)
        levels = contour_levels_from_pair(obs_dens, gen_dens)

        p_pair = contour(
            centers,
            centers,
            transpose(obs_dens);
            levels=levels,
            linewidth=2.0,
            color=:dodgerblue3,
            label=obs_label,
            xlabel="x_k",
            ylabel="x_{k+$lag}",
            title="Bivariate contours: lag $lag",
        )
        contour!(
            p_pair,
            centers,
            centers,
            transpose(gen_dens);
            levels=levels,
            linewidth=2.0,
            color=:tomato3,
            linestyle=:dash,
            label=gen_label,
        )
        p_biv[idx_lag] = p_pair
    end

    return [p_pdf, p_qq, p_mom, p_kl, p_biv[1], p_biv[2], p_biv[3]]
end

function _save_4x2(path::AbstractString, panels::AbstractVector{<:Plots.Plot})
    length(panels) == 8 || error("Expected exactly 8 panels for 4x2 layout")
    for p in panels
        plot!(p; left_margin=12Plots.mm, right_margin=6Plots.mm, top_margin=8Plots.mm, bottom_margin=10Plots.mm)
    end
    fig = plot(
        panels...;
        layout=(4, 2),
        size=(1850, 2500),
        left_margin=6Plots.mm,
        right_margin=6Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
    )
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function save_stats_figure_stein(path::AbstractString,
    obs::AbstractArray{<:Real,3},
    gen::AbstractArray{<:Real,3},
    kl_mode::Vector{Float64},
    js_mode::Vector{Float64},
    bins::Int,
    stein_matrix::Matrix{Float64};
    obs_label::String="Observed",
    gen_label::String="Generated")
    size(stein_matrix, 1) == size(stein_matrix, 2) || error("Stein matrix must be square")
    panels = _stats_first_7_panels(obs, gen, kl_mode, js_mode, bins; obs_label=obs_label, gen_label=gen_label)
    stein_clim = maximum(abs, stein_matrix)
    stein_clim = (isfinite(stein_clim) && stein_clim > 0.0) ? stein_clim : 1e-12
    p_stein = heatmap(
        1:size(stein_matrix, 2),
        1:size(stein_matrix, 1),
        stein_matrix;
        color=:RdBu,
        clims=(-stein_clim, stein_clim),
        xlabel="j",
        ylabel="i",
        title="Stein matrix <s(x)x^T>",
        aspect_ratio=:equal,
    )
    push!(panels, p_stein)
    return _save_4x2(path, panels)
end

function save_stats_figure_acf(path::AbstractString,
    obs::AbstractArray{<:Real,3},
    gen::AbstractArray{<:Real,3},
    kl_mode::Vector{Float64},
    js_mode::Vector{Float64},
    bins::Int;
    max_lag::Int=200,
    obs_label::String="Observed",
    gen_label::String="Generated")
    panels = _stats_first_7_panels(obs, gen, kl_mode, js_mode, bins; obs_label=obs_label, gen_label=gen_label)
    acf_obs = average_mode_acf(obs, max_lag)
    acf_gen = average_mode_acf(gen, max_lag)
    n = min(length(acf_obs), length(acf_gen))
    n = max(n, 1)
    lag = 0:(n - 1)
    p_acf = plot(lag, acf_obs[1:n]; color=:dodgerblue3, linewidth=2, label=obs_label, xlabel="Lag", ylabel="ACF", title="Average ACF over modes")
    plot!(p_acf, lag, acf_gen[1:n]; color=:tomato3, linewidth=2, linestyle=:dash, label=gen_label)
    hline!(p_acf, [0.0]; color=:gray40, linestyle=:dot, label="")
    push!(panels, p_acf)
    return _save_4x2(path, panels)
end

end
