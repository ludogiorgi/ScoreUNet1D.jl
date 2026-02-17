if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using Flux
using HDF5
using KernelDensity
using LinearAlgebra
using Plots
using Random
using ScoreUNet1D
using Statistics
using StatsBase

include(joinpath(@__DIR__, "run_layout.jl"))
using .L96RunLayout

const PIPELINE_MODE = lowercase(get(ENV, "L96_PIPELINE_MODE", "false")) == "true"
const RUN_DIR = L96RunLayout.pick_stage_run_dir(@__DIR__)
const DEFAULT_DATA_PATH = let p = L96RunLayout.default_data_path(RUN_DIR)
    isfile(p) ? p : joinpath(@__DIR__, "l96_timeseries.hdf5")
end
const DEFAULT_MODEL_PATH = let p = L96RunLayout.default_train_paths(RUN_DIR).model_path
    isfile(p) ? p : joinpath(@__DIR__, "trained_model.bson")
end
const EVAL_ROOT = get(ENV, "L96_EVAL_ROOT", PIPELINE_MODE ? "" : L96RunLayout.default_eval_root(RUN_DIR))

const DATA_PATH = get(ENV, "L96_DATA_PATH", DEFAULT_DATA_PATH)
const DATASET_KEY = get(ENV, "L96_DATASET_KEY", "timeseries")
const MODEL_PATH = get(ENV, "L96_MODEL_PATH", DEFAULT_MODEL_PATH)
const FIG_DIR = get(ENV, "L96_FIG_DIR", joinpath(EVAL_ROOT, "figures"))
const METRICS_PATH = get(ENV, "L96_METRICS_PATH", joinpath(EVAL_ROOT, "discrepancy_metrics.txt"))
const EVAL_CONFIG_PATH = get(ENV, "L96_EVAL_CONFIG_PATH", isempty(EVAL_ROOT) ? "" : joinpath(EVAL_ROOT, "eval_config.toml"))

const LANGEVIN_DT = parse(Float64, get(ENV, "L96_LANGEVIN_DT", "0.001"))
const LANGEVIN_STEPS = parse(Int, get(ENV, "L96_LANGEVIN_STEPS", "12000"))
const LANGEVIN_RESOLUTION = parse(Int, get(ENV, "L96_LANGEVIN_RESOLUTION", "100"))
const LANGEVIN_SAMPLE_DT = LANGEVIN_DT * LANGEVIN_RESOLUTION
const LANGEVIN_BURN_IN = parse(Int, get(ENV, "L96_LANGEVIN_BURN_IN", "2000"))
const LANGEVIN_ENSEMBLES = parse(Int, get(ENV, "L96_LANGEVIN_ENSEMBLES", "16"))
const LANGEVIN_PROFILE = lowercase(get(ENV, "L96_LANGEVIN_PROFILE", "full"))
const PDF_BINS = parse(Int, get(ENV, "L96_PDF_BINS", "80"))
const LANGEVIN_SEED = parse(Int, get(ENV, "L96_LANGEVIN_SEED", "7"))
const LANGEVIN_PROGRESS = lowercase(get(ENV, "L96_LANGEVIN_PROGRESS", "false")) == "true"
const LANGEVIN_DEVICE_NAME = get(ENV, "L96_LANGEVIN_DEVICE", "GPU:0")
const MIN_KEPT_SNAPSHOTS_WARN = parse(Int, get(ENV, "L96_MIN_KEPT_SNAPSHOTS_WARN", "800"))

const TARGET_AVG_KL = parse(Float64, get(ENV, "L96_TARGET_AVG_KL", "0.05"))
const KL_LOW_Q = parse(Float64, get(ENV, "L96_KL_LOW_Q", "0.001"))
const KL_HIGH_Q = parse(Float64, get(ENV, "L96_KL_HIGH_Q", "0.999"))
const USE_BOUNDARY = lowercase(get(ENV, "L96_USE_BOUNDARY", "true")) == "true"
const BOUNDARY_MIN = parse(Float64, get(ENV, "L96_BOUNDARY_MIN", "-10.0"))
const BOUNDARY_MAX = parse(Float64, get(ENV, "L96_BOUNDARY_MAX", "10.0"))
const MAX_ACF_LAG = parse(Int, get(ENV, "L96_ACF_MAX_LAG", "150"))
const SCORE_EVAL_SAMPLES = parse(Int, get(ENV, "L96_SCORE_EVAL_SAMPLES", "4096"))
const SCORE_EVAL_BATCH = parse(Int, get(ENV, "L96_SCORE_EVAL_BATCH", "256"))
const SCORE_EVAL_SEED = parse(Int, get(ENV, "L96_SCORE_EVAL_SEED", "123"))
const SCORE_SCATTER_POINTS = parse(Int, get(ENV, "L96_SCORE_SCATTER_POINTS", "12000"))
const SYM_MAX_POINTS = parse(Int, get(ENV, "L96_SYM_MAX_POINTS", "450000"))
const STYLE_DPI = parse(Int, get(ENV, "L96_FIG_DPI", "170"))

function setup_plot_style!()
    default(
        dpi=STYLE_DPI,
        size=(980, 620),
        linewidth=2,
        markerstrokewidth=0.0,
        grid=true,
        gridalpha=0.22,
        framestyle=:box,
        legend=:best,
        legendfontsize=10,
        guidefontsize=12,
        tickfontsize=10,
        titlefontsize=13,
        foreground_color_legend=:black,
        background_color_legend=:white,
    )
    return nothing
end

function load_truth_tensor(path::AbstractString)
    raw = h5open(path, "r") do h5
        read(h5, DATASET_KEY)
    end
    tensor = permutedims(raw, (3, 2, 1))
    return Array{Float32,3}(tensor)
end

function normalize_with_stats(tensor::Array{Float32,3}, stats::DataStats)
    L, C, _ = size(tensor)
    mean_lc = permutedims(stats.mean, (2, 1))
    std_lc = permutedims(stats.std, (2, 1))
    normed = (tensor .- reshape(mean_lc, L, C, 1)) ./ reshape(std_lc, L, C, 1)
    return Array{Float32,3}(normed)
end

function with_padding(lo::Float64, hi::Float64)
    if lo == hi
        δ = max(abs(lo), 1.0) * 1e-3
        return lo - δ, hi + δ
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

@inline function discrete_kl(p::Vector{Float64}, q::Vector{Float64})
    return sum(@. p * log(p / q))
end

function modewise_univariate_metrics(tensor_truth::Array{Float32,3},
                                     tensor_gen::Array{Float32,3};
                                     nbins::Int=80,
                                     bounds_mode::Symbol=:minmax,
                                     low_q::Float64=0.001,
                                     high_q::Float64=0.999)
    L, C, _ = size(tensor_truth)
    kl_mode = Array{Float64}(undef, L, C)
    js_mode = Array{Float64}(undef, L, C)
    l1_mode = Array{Float64}(undef, L, C)

    @inbounds for l in 1:L, c in 1:C
        truth_vals = Float64.(vec(@view tensor_truth[l, c, :]))
        gen_vals = Float64.(vec(@view tensor_gen[l, c, :]))

        if bounds_mode == :quantile
            combined = vcat(truth_vals, gen_vals)
            lo = quantile(combined, low_q)
            hi = quantile(combined, high_q)
        else
            lo = min(minimum(truth_vals), minimum(gen_vals))
            hi = max(maximum(truth_vals), maximum(gen_vals))
        end
        lo, hi = with_padding(lo, hi)
        edges = collect(range(lo, hi; length=nbins + 1))

        p = histogram_prob(truth_vals, edges)
        q = histogram_prob(gen_vals, edges)
        m = 0.5 .* (p .+ q)

        kl_mode[l, c] = discrete_kl(p, q)
        js_mode[l, c] = 0.5 * discrete_kl(p, m) + 0.5 * discrete_kl(q, m)
        l1_mode[l, c] = mean(abs.(p .- q))
    end

    return kl_mode, js_mode, l1_mode
end

function autocorrelation(series::Vector{Float64}, max_lag::Int)
    n = length(series)
    m = min(max(max_lag, 1), n - 1)
    centered = series .- mean(series)
    variance = sum(abs2, centered) / max(n, 1)
    variance <= eps(Float64) && return ones(Float64, m + 1)
    acf = Array{Float64}(undef, m + 1)
    acf[1] = 1.0
    @inbounds for lag in 1:m
        t = n - lag
        acf[lag + 1] = dot(view(centered, 1:t), view(centered, (lag + 1):(lag + t))) / (t * variance)
    end
    return acf
end

function collect_values_channels(tensor::Array{Float32,3}, channels::Vector{Int}; max_points::Int=450_000)
    L, _, T = size(tensor)
    total = L * length(channels) * T
    stride = max(cld(total, max(max_points, 1)), 1)
    vals = Float64[]
    sizehint!(vals, min(total, max_points))
    counter = 0
    @inbounds for t in 1:T, c in channels, l in 1:L
        counter += 1
        if (counter - 1) % stride == 0
            push!(vals, Float64(tensor[l, c, t]))
        end
    end
    return vals
end

function collect_pairs_xx(tensor::Array{Float32,3}; max_points::Int=450_000)
    L, _, T = size(tensor)
    total = L * T
    stride = max(cld(total, max(max_points, 1)), 1)
    x1 = Float64[]
    x2 = Float64[]
    sizehint!(x1, min(total, max_points))
    sizehint!(x2, min(total, max_points))
    counter = 0
    @inbounds for t in 1:T, k in 1:L
        counter += 1
        if (counter - 1) % stride == 0
            kp1 = (k == L) ? 1 : k + 1
            push!(x1, Float64(tensor[k, 1, t]))
            push!(x2, Float64(tensor[kp1, 1, t]))
        end
    end
    return x1, x2
end

function collect_pairs_yy(tensor::Array{Float32,3}; max_points::Int=450_000)
    L, C, T = size(tensor)
    C >= 3 || return Float64[], Float64[]
    total = L * (C - 1) * T
    stride = max(cld(total, max(max_points, 1)), 1)
    y1 = Float64[]
    y2 = Float64[]
    sizehint!(y1, min(total, max_points))
    sizehint!(y2, min(total, max_points))
    counter = 0
    @inbounds for t in 1:T, c in 2:C, k in 1:L
        counter += 1
        if (counter - 1) % stride == 0
            cnext = (c == C) ? 2 : c + 1
            push!(y1, Float64(tensor[k, c, t]))
            push!(y2, Float64(tensor[k, cnext, t]))
        end
    end
    return y1, y2
end

function collect_pairs_xy(tensor::Array{Float32,3}; max_points::Int=450_000)
    L, C, T = size(tensor)
    C >= 2 || return Float64[], Float64[]
    total = L * (C - 1) * T
    stride = max(cld(total, max(max_points, 1)), 1)
    xv = Float64[]
    yv = Float64[]
    sizehint!(xv, min(total, max_points))
    sizehint!(yv, min(total, max_points))
    counter = 0
    @inbounds for t in 1:T, c in 2:C, k in 1:L
        counter += 1
        if (counter - 1) % stride == 0
            push!(xv, Float64(tensor[k, 1, t]))
            push!(yv, Float64(tensor[k, c, t]))
        end
    end
    return xv, yv
end

function average_mode_acf(tensor::Array{Float32,3}, mode::Symbol, max_lag::Int)
    L, C, T = size(tensor)
    mlag = min(max(max_lag, 1), T - 1)
    acc = zeros(Float64, mlag + 1)
    n = 0

    if mode == :x
        @inbounds for k in 1:L
            acf = autocorrelation(Float64.(vec(@view tensor[k, 1, :])), mlag)
            acc .+= acf[1:(mlag + 1)]
            n += 1
        end
    elseif mode == :y
        @inbounds for c in 2:C, k in 1:L
            acf = autocorrelation(Float64.(vec(@view tensor[k, c, :])), mlag)
            acc .+= acf[1:(mlag + 1)]
            n += 1
        end
    else
        error("Unsupported mode for average_mode_acf: $mode")
    end

    n > 0 || return ones(Float64, mlag + 1)
    return acc ./ n
end

function mode_moment_panel(tensor_truth::Array{Float32,3},
                           tensor_gen::Array{Float32,3})
    L, C, _ = size(tensor_truth)

    x_idx = collect(1:L)
    x_mean_t = [mean(@view tensor_truth[k, 1, :]) for k in 1:L]
    x_mean_g = [mean(@view tensor_gen[k, 1, :]) for k in 1:L]
    x_var_t = [var(@view tensor_truth[k, 1, :]) for k in 1:L]
    x_var_g = [var(@view tensor_gen[k, 1, :]) for k in 1:L]

    ny = L * max(C - 1, 1)
    y_idx = collect(1:ny)
    y_mean_t = Vector{Float64}(undef, ny)
    y_mean_g = Vector{Float64}(undef, ny)
    y_var_t = Vector{Float64}(undef, ny)
    y_var_g = Vector{Float64}(undef, ny)
    if C >= 2
        @inbounds for c in 2:C, k in 1:L
            m = (c - 2) * L + k
            y_mean_t[m] = mean(@view tensor_truth[k, c, :])
            y_mean_g[m] = mean(@view tensor_gen[k, c, :])
            y_var_t[m] = var(@view tensor_truth[k, c, :])
            y_var_g[m] = var(@view tensor_gen[k, c, :])
        end
    else
        fill!(y_mean_t, NaN)
        fill!(y_mean_g, NaN)
        fill!(y_var_t, NaN)
        fill!(y_var_g, NaN)
    end

    p = plot(x_idx, x_mean_t;
             label="x mean truth",
             color=:dodgerblue3,
             marker=:circle,
             markersize=3,
             xlabel="x-mode index k",
             ylabel="Moment value",
             title="Mode Moments (x bottom axis, y top axis)")
    plot!(p, x_idx, x_mean_g; label="x mean gen", color=:dodgerblue3, linestyle=:dash, marker=:diamond, markersize=3)
    plot!(p, x_idx, x_var_t; label="x var truth", color=:navy, marker=:circle, markersize=3)
    plot!(p, x_idx, x_var_g; label="x var gen", color=:navy, linestyle=:dash, marker=:diamond, markersize=3)

    pt = twiny(p)
    plot!(pt, y_idx, y_mean_t; label="y mean truth", color=:tomato3, marker=:circle, markersize=2, alpha=0.85)
    plot!(pt, y_idx, y_mean_g; label="y mean gen", color=:tomato3, linestyle=:dash, marker=:diamond, markersize=2, alpha=0.85)
    plot!(pt, y_idx, y_var_t; label="y var truth", color=:darkorange3, marker=:circle, markersize=2, alpha=0.85)
    plot!(pt, y_idx, y_var_g; label="y var gen", color=:darkorange3, linestyle=:dash, marker=:diamond, markersize=2, alpha=0.85)
    xlabel!(pt, "y-mode flattened index (top axis)")
    return p
end

function save_stats_figure_3x3(out_path::AbstractString,
                               p_pdf_x,
                               p_pdf_y,
                               p_joint_xx,
                               p_joint_yy,
                               p_joint_xy,
                               p_qq_x,
                               p_qq_y,
                               p_heat_kl,
                               p_mom)
    for p in (p_pdf_x, p_pdf_y, p_joint_xx, p_joint_yy, p_joint_xy, p_qq_x, p_qq_y, p_heat_kl, p_mom)
        plot!(
            p;
            left_margin=12Plots.mm,
            right_margin=6Plots.mm,
            top_margin=7Plots.mm,
            bottom_margin=10Plots.mm,
            titlefontsize=11,
            guidefontsize=11,
            tickfontsize=9,
            legendfontsize=9,
        )
    end

    fig = plot(
        p_pdf_x, p_pdf_y, p_heat_kl,
        p_joint_xx, p_joint_yy, p_joint_xy,
        p_qq_x, p_qq_y, p_mom;
        layout=grid(3, 3),
        size=(2150, 1900),
        left_margin=6Plots.mm,
        right_margin=6Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
    )
    savefig(fig, out_path)
    return out_path
end

function save_dynamics_figure_3x2(out_path::AbstractString,
                                  tensor_truth::Array{Float32,3},
                                  tensor_gen::Array{Float32,3};
                                  max_lag::Int=150)
    L, C, Tt = size(tensor_truth)
    _, _, Tg = size(tensor_gen)
    C >= 2 || error("Need at least 2 channels (x plus fast y channels) for dynamics figure")
    T = min(Tt, Tg, 300)
    T >= 2 || error("Need at least 2 time samples for dynamics figure")
    Jfast = C - 1

    truth_x = Array{Float64}(undef, L, T)
    gen_x = Array{Float64}(undef, L, T)
    @inbounds begin
        truth_x .= Float64.(tensor_truth[:, 1, 1:T])
        gen_x .= Float64.(tensor_gen[:, 1, 1:T])
    end

    truth_ybar = Array{Float64}(undef, L, T)
    gen_ybar = Array{Float64}(undef, L, T)
    @inbounds begin
        truth_ybar .= dropdims(mean(Float64.(tensor_truth[:, 2:C, 1:T]); dims=2), dims=2)
        gen_ybar .= dropdims(mean(Float64.(tensor_gen[:, 2:C, 1:T]); dims=2), dims=2)
    end
    truth_y1j = Float64.(tensor_truth[1, 2:C, 1:T])
    gen_y1j = Float64.(tensor_gen[1, 2:C, 1:T])

    p1 = heatmap(1:T, 1:L, truth_x;
                 xlabel="Time index",
                 ylabel="Spatial index k",
                 title="Hovmoller x: observed",
                 color=:viridis)
    p2 = heatmap(1:T, 1:L, truth_ybar;
                 xlabel="Time index",
                 ylabel="Spatial index k",
                 title="Hovmoller ybar: observed",
                 color=:viridis)
    p3 = heatmap(1:T, 1:Jfast, truth_y1j;
                 xlabel="Time index",
                 ylabel="Fast index j",
                 title="Hovmoller y_{1,j}: observed (k=1)",
                 color=:viridis)
    p4 = heatmap(1:T, 1:L, gen_x;
                 xlabel="Time index",
                 ylabel="Spatial index k",
                 title="Hovmoller x: generated",
                 color=:plasma)
    p5 = heatmap(1:T, 1:L, gen_ybar;
                 xlabel="Time index",
                 ylabel="Spatial index k",
                 title="Hovmoller ybar: generated",
                 color=:plasma)
    p6 = heatmap(1:T, 1:Jfast, gen_y1j;
                 xlabel="Time index",
                 ylabel="Fast index j",
                 title="Hovmoller y_{1,j}: generated (k=1)",
                 color=:plasma)

    acf_xt = average_mode_acf(tensor_truth[:, :, 1:T], :x, max_lag)
    acf_xg = average_mode_acf(tensor_gen[:, :, 1:T], :x, max_lag)
    acf_yt = average_mode_acf(tensor_truth[:, :, 1:T], :y, max_lag)
    acf_yg = average_mode_acf(tensor_gen[:, :, 1:T], :y, max_lag)

    lag_axis = 0:(length(acf_xt) - 1)
    p7 = plot(lag_axis, acf_xt;
              label="Observed x avg ACF",
              color=:dodgerblue3,
              marker=:circle,
              markersize=3,
              xlabel="Lag",
              ylabel="ACF",
              title="Average ACF over x modes")
    plot!(p7, lag_axis, acf_xg; label="Generated x avg ACF", color=:tomato3, linestyle=:dash, marker=:diamond, markersize=3)
    hline!(p7, [0.0]; label=nothing, color=:black, linestyle=:dot)

    p8 = plot(lag_axis, acf_yt;
              label="Observed y avg ACF",
              color=:dodgerblue3,
              marker=:circle,
              markersize=3,
              xlabel="Lag",
              ylabel="ACF",
              title="Average ACF over y modes")
    plot!(p8, lag_axis, acf_yg; label="Generated y avg ACF", color=:tomato3, linestyle=:dash, marker=:diamond, markersize=3)
    hline!(p8, [0.0]; label=nothing, color=:black, linestyle=:dot)

    t_axis = 1:T
    x1_obs = Float64.(vec(tensor_truth[1, 1, 1:T]))
    x1_gen = Float64.(vec(tensor_gen[1, 1, 1:T]))
    y11_obs = Float64.(vec(tensor_truth[1, 2, 1:T]))
    y11_gen = Float64.(vec(tensor_gen[1, 2, 1:T]))
    p9 = plot(t_axis, x1_obs;
              label="obs x1",
              color=:dodgerblue3,
              linewidth=2,
              xlabel="Time index",
              ylabel="Normalized value",
              title="Timeseries at k=1: x1 and y11")
    plot!(p9, t_axis, x1_gen; label="gen x1", color=:dodgerblue3, linestyle=:dash, linewidth=2)
    plot!(p9, t_axis, y11_obs; label="obs y11", color=:darkorange3, linewidth=2)
    plot!(p9, t_axis, y11_gen; label="gen y11", color=:darkorange3, linestyle=:dash, linewidth=2)

    for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9)
        plot!(
            p;
            left_margin=12Plots.mm,
            right_margin=6Plots.mm,
            top_margin=7Plots.mm,
            bottom_margin=10Plots.mm,
            titlefontsize=11,
            guidefontsize=11,
            tickfontsize=9,
            legendfontsize=9,
        )
    end

    fig = plot(
        p1, p2, p3,
        p4, p5, p6,
        p7, p8, p9;
        layout=(3, 3),
        size=(2650, 1850),
        left_margin=6Plots.mm,
        right_margin=6Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
    )
    savefig(fig, out_path)
    return out_path
end

function make_univariate_panel(truth_vals::Vector{Float64},
                               gen_vals::Vector{Float64};
                               title::AbstractString,
                               xlabel::AbstractString)
    lo = min(minimum(truth_vals), minimum(gen_vals))
    hi = max(maximum(truth_vals), maximum(gen_vals))
    lo, hi = with_padding(lo, hi)
    edges = collect(range(lo, hi; length=PDF_BINS + 1))

    μt, σt = mean(truth_vals), std(truth_vals)
    μg, σg = mean(gen_vals), std(gen_vals)

    p = histogram(truth_vals;
                  bins=edges,
                  normalize=:pdf,
                  alpha=0.45,
                  label="Truth (μ=$(round(μt, digits=3)), σ=$(round(σt, digits=3)))",
                  color=:dodgerblue3,
                  title=title,
                  xlabel=xlabel,
                  ylabel="PDF")
    histogram!(p, gen_vals;
               bins=edges,
               normalize=:pdf,
               alpha=0.45,
               label="Generated (μ=$(round(μg, digits=3)), σ=$(round(σg, digits=3)))",
               color=:tomato3)
    xlims!(p, (lo, hi))
    return p
end

function make_qq_panel(truth_vals::Vector{Float64},
                       gen_vals::Vector{Float64};
                       title::AbstractString)
    n = min(length(truth_vals), length(gen_vals))
    probs = collect(range(0.001, 0.999; length=n))
    qt = quantile(truth_vals, probs)
    qg = quantile(gen_vals, probs)
    lo = min(minimum(qt), minimum(qg))
    hi = max(maximum(qt), maximum(qg))
    lo, hi = with_padding(lo, hi)

    p = scatter(qt, qg;
                label="Quantiles",
                color=:darkslateblue,
                alpha=0.6,
                markersize=3,
                xlabel="Truth quantiles",
                ylabel="Generated quantiles",
                title=title)
    plot!(p, [lo, hi], [lo, hi];
          label="Ideal y=x",
          color=:black,
          linestyle=:dash)
    xlims!(p, (lo, hi))
    ylims!(p, (lo, hi))
    return p
end

function make_joint_kde_panel(x_truth::Vector{Float64},
                              y_truth::Vector{Float64},
                              x_gen::Vector{Float64},
                              y_gen::Vector{Float64};
                              title::AbstractString,
                              xlabel::AbstractString,
                              ylabel::AbstractString)
    xmin = min(minimum(x_truth), minimum(x_gen))
    xmax = max(maximum(x_truth), maximum(x_gen))
    ymin = min(minimum(y_truth), minimum(y_gen))
    ymax = max(maximum(y_truth), maximum(y_gen))
    xmin, xmax = with_padding(xmin, xmax)
    ymin, ymax = with_padding(ymin, ymax)

    kde_truth = kde((x_truth, y_truth); npoints=(130, 130), boundary=((xmin, xmax), (ymin, ymax)))
    kde_gen = kde((x_gen, y_gen); npoints=(130, 130), boundary=((xmin, xmax), (ymin, ymax)))

    p = contour(kde_truth.x, kde_truth.y, kde_truth.density;
                levels=10,
                linewidth=2,
                color=:dodgerblue3,
                colorbar=false,
                label="Truth contours",
                xlabel=xlabel,
                ylabel=ylabel,
                title=title)
    contour!(p, kde_gen.x, kde_gen.y, kde_gen.density;
             levels=10,
             linewidth=2,
             linestyle=:dash,
             color=:tomato3,
             colorbar=false,
             label="Generated contours")
    xlims!(p, (xmin, xmax))
    ylims!(p, (ymin, ymax))
    return p
end

function channel_moment_stats(tensor_truth::Array{Float32,3},
                              tensor_gen::Array{Float32,3})
    _, C, _ = size(tensor_truth)
    truth_mean = [mean(@view tensor_truth[:, c, :]) for c in 1:C]
    gen_mean = [mean(@view tensor_gen[:, c, :]) for c in 1:C]
    truth_std = [std(@view tensor_truth[:, c, :]) for c in 1:C]
    gen_std = [std(@view tensor_gen[:, c, :]) for c in 1:C]
    rel_std_err = abs.((gen_std .- truth_std) ./ (abs.(truth_std) .+ eps(Float64)))
    return truth_mean, gen_mean, truth_std, gen_std, rel_std_err
end

function score_prediction_diagnostics(model,
                                      tensor_truth::Array{Float32,3},
                                      sigma::Float32,
                                      device::ExecutionDevice;
                                      nsamples::Int=4096,
                                      batch_size::Int=256,
                                      seed::Int=123,
                                      scatter_points::Int=10_000)
    Flux.testmode!(model)
    L, C, N = size(tensor_truth)
    nsel = min(max(nsamples, 1), N)
    rng = MersenneTwister(seed)
    idx = sample(rng, 1:N, nsel; replace=false)

    sum_true = 0.0
    sum_pred = 0.0
    sum_true2 = 0.0
    sum_pred2 = 0.0
    sum_true_pred = 0.0
    sum_sq_err = 0.0
    count = 0

    ch_sse = zeros(Float64, C)
    ch_count = zeros(Int, C)

    scatter_true = Float64[]
    scatter_pred = Float64[]

    for start in 1:batch_size:nsel
        stop = min(start + batch_size - 1, nsel)
        batch_idx = idx[start:stop]
        b = length(batch_idx)
        batch_cpu = Array{Float32,3}(undef, L, C, b)
        @views batch_cpu .= tensor_truth[:, :, batch_idx]
        noise_cpu = randn(rng, Float32, L, C, b)

        if is_gpu(device)
            batch = move_array(batch_cpu, device)
            noise = move_array(noise_cpu, device)
            noisy = batch .+ sigma .* noise
            pred_cpu = Array(model(noisy))
        else
            noisy = batch_cpu .+ sigma .* noise_cpu
            pred_cpu = Array(model(noisy))
        end

        truth_vec = Float64.(vec(noise_cpu))
        pred_vec = Float64.(vec(pred_cpu))
        err_vec = pred_vec .- truth_vec

        sum_true += sum(truth_vec)
        sum_pred += sum(pred_vec)
        sum_true2 += sum(abs2, truth_vec)
        sum_pred2 += sum(abs2, pred_vec)
        sum_true_pred += sum(truth_vec .* pred_vec)
        sum_sq_err += sum(abs2, err_vec)
        count += length(truth_vec)

        @inbounds for c in 1:C
            pred_ch = @view pred_cpu[:, c, :]
            noise_ch = @view noise_cpu[:, c, :]
            ch_sse[c] += sum(abs2, pred_ch .- noise_ch)
            ch_count[c] += length(pred_ch)
        end

        if length(scatter_true) < scatter_points
            remaining = scatter_points - length(scatter_true)
            take_n = min(remaining, length(truth_vec))
            append!(scatter_true, truth_vec[1:take_n])
            append!(scatter_pred, pred_vec[1:take_n])
        end
    end

    mse = sum_sq_err / max(count, 1)
    var_true = sum_true2 - (sum_true^2 / max(count, 1))
    var_pred = sum_pred2 - (sum_pred^2 / max(count, 1))
    cov = sum_true_pred - (sum_true * sum_pred / max(count, 1))
    corr = cov / sqrt(max(var_true, eps(Float64)) * max(var_pred, eps(Float64)))
    ch_mse = ch_sse ./ max.(ch_count, 1)

    return (
        mse=mse,
        corr=corr,
        channel_mse=ch_mse,
        scatter_true=scatter_true,
        scatter_pred=scatter_pred,
    )
end

function write_metrics(path::AbstractString, stats::AbstractDict{String,<:Real})
    open(path, "w") do io
        for k in sort!(collect(keys(stats)))
            println(io, "$k=$(Float64(stats[k]))")
        end
    end
    return path
end

function save_eval_config(run_dir::AbstractString,
                          eval_root::AbstractString,
                          effective_device_name::AbstractString,
                          trainer_sigma::Real)
    (PIPELINE_MODE || isempty(EVAL_CONFIG_PATH)) && return ""

    cfg = Dict{String,Any}(
        "stage" => "sample_and_compare",
        "run_dir" => run_dir,
        "paths" => Dict(
            "data_path" => abspath(DATA_PATH),
            "model_path" => abspath(MODEL_PATH),
            "eval_root" => abspath(eval_root),
            "fig_dir" => abspath(FIG_DIR),
            "metrics_path" => abspath(METRICS_PATH),
        ),
        "langevin" => Dict(
            "profile" => LANGEVIN_PROFILE,
            "device_requested" => LANGEVIN_DEVICE_NAME,
            "device_effective" => effective_device_name,
            "dt" => LANGEVIN_DT,
            "sample_dt" => LANGEVIN_SAMPLE_DT,
            "nsteps" => LANGEVIN_STEPS,
            "resolution" => LANGEVIN_RESOLUTION,
            "burn_in" => LANGEVIN_BURN_IN,
            "n_ensembles" => LANGEVIN_ENSEMBLES,
            "sigma" => Float64(trainer_sigma),
            "seed" => LANGEVIN_SEED,
            "progress" => LANGEVIN_PROGRESS,
            "use_boundary" => USE_BOUNDARY,
            "boundary_min" => BOUNDARY_MIN,
            "boundary_max" => BOUNDARY_MAX,
            "min_kept_snapshots_warn" => MIN_KEPT_SNAPSHOTS_WARN,
        ),
        "metrics_config" => Dict(
            "pdf_bins" => PDF_BINS,
            "target_avg_kl" => TARGET_AVG_KL,
            "kl_low_q" => KL_LOW_Q,
            "kl_high_q" => KL_HIGH_Q,
            "acf_max_lag" => MAX_ACF_LAG,
        ),
        "score_diagnostics" => Dict(
            "eval_samples" => SCORE_EVAL_SAMPLES,
            "eval_batch" => SCORE_EVAL_BATCH,
            "eval_seed" => SCORE_EVAL_SEED,
            "scatter_points" => SCORE_SCATTER_POINTS,
        ),
    )
    return L96RunLayout.write_toml_file(EVAL_CONFIG_PATH, cfg)
end

function kept_snapshot_diagnostics(nsteps::Int, burn_in::Int, resolution::Int, ensembles::Int)
    total_snapshots = fld(max(nsteps, 1), max(resolution, 1))
    burn_snapshots = fld(max(burn_in, 0), max(resolution, 1))
    kept_per_ensemble = max(total_snapshots - burn_snapshots, 0)
    total_kept = kept_per_ensemble * max(ensembles, 1)
    return (
        total_snapshots=total_snapshots,
        burn_snapshots=burn_snapshots,
        kept_per_ensemble=kept_per_ensemble,
        total_kept=total_kept,
    )
end

function resolve_langevin_device(name::AbstractString)
    try
        device = select_device(name)
        activate_device!(device)
        return device, name
    catch err
        @warn "Requested Langevin device unavailable; falling back to CPU" requested = name error = sprint(showerror, err)
        device = ScoreUNet1D.CPUDevice()
        activate_device!(device)
        return device, "CPU"
    end
end

function main()
    setup_plot_style!()
    isfile(MODEL_PATH) || error("Model not found at $MODEL_PATH. Run train_unet.jl first.")
    isfile(DATA_PATH) || error("Data not found at $DATA_PATH. Run generate_data.jl first.")
    if !PIPELINE_MODE
        L96RunLayout.ensure_runs_readme!(L96RunLayout.default_runs_root(@__DIR__))
    end

    contents = BSON.load(MODEL_PATH)
    model = contents[:model]
    cfg = contents[:cfg]
    trainer_cfg = contents[:trainer_cfg]
    stats = contents[:stats]

    tensor_truth_raw = load_truth_tensor(DATA_PATH)
    tensor_truth = normalize_with_stats(tensor_truth_raw, stats)
    dataset = NormalizedDataset(tensor_truth, stats)

    device, effective_device_name = resolve_langevin_device(LANGEVIN_DEVICE_NAME)
    model = move_model(model, is_gpu(device) ? device : ScoreUNet1D.CPUDevice())
    Flux.testmode!(model)
    eval_config_path = save_eval_config(RUN_DIR, EVAL_ROOT, effective_device_name, trainer_cfg.sigma)

    langevin_cfg = LangevinConfig(
        dt=LANGEVIN_DT,
        sample_dt=LANGEVIN_SAMPLE_DT,
        nsteps=LANGEVIN_STEPS,
        burn_in=LANGEVIN_BURN_IN,
        resolution=LANGEVIN_RESOLUTION,
        n_ensembles=LANGEVIN_ENSEMBLES,
        nbins=PDF_BINS,
        sigma=trainer_cfg.sigma,
        seed=LANGEVIN_SEED,
        mode=:all,
        boundary=USE_BOUNDARY ? (BOUNDARY_MIN, BOUNDARY_MAX) : nothing,
        progress=LANGEVIN_PROGRESS,
    )

    @info "Running L96 Langevin sampling" device = effective_device_name sigma = trainer_cfg.sigma
    result = run_langevin(model, dataset, langevin_cfg; device=device)
    kept_diag = kept_snapshot_diagnostics(LANGEVIN_STEPS, LANGEVIN_BURN_IN, LANGEVIN_RESOLUTION, LANGEVIN_ENSEMBLES)
    if kept_diag.kept_per_ensemble < MIN_KEPT_SNAPSHOTS_WARN
        @warn "Low retained snapshots per ensemble for Langevin PDF estimate" kept_per_ensemble = kept_diag.kept_per_ensemble threshold = MIN_KEPT_SNAPSHOTS_WARN nsteps = LANGEVIN_STEPS burn_in = LANGEVIN_BURN_IN resolution = LANGEVIN_RESOLUTION
    end

    L, C, _ = size(tensor_truth)
    C >= 2 || error("Expected at least 2 channels (x + y1) for plotting; got $C")
    traj4 = reshape(result.trajectory, L, C, :, size(result.trajectory, 3))
    tensor_gen = reshape(permutedims(traj4, (1, 2, 4, 3)), L, C, :)

    ensure_dir(FIG_DIR)

    # Symmetry-averaged statistics over all relevant modes for tighter estimates.
    truth_x = collect_values_channels(tensor_truth, [1]; max_points=SYM_MAX_POINTS)
    gen_x = collect_values_channels(tensor_gen, [1]; max_points=SYM_MAX_POINTS)
    truth_y1 = collect_values_channels(tensor_truth, collect(2:C); max_points=SYM_MAX_POINTS)
    gen_y1 = collect_values_channels(tensor_gen, collect(2:C); max_points=SYM_MAX_POINTS)

    p_pdf_x = make_univariate_panel(
        truth_x,
        gen_x;
        title="Univariate PDF at x(k=1)",
        xlabel="x(k=1) normalized",
    )
    p_pdf_y = make_univariate_panel(
        truth_y1,
        gen_y1;
        title="Univariate PDF at y1(k=1)",
        xlabel="y1(k=1) normalized",
    )

    p_qq_x = make_qq_panel(
        truth_x,
        gen_x;
        title="QQ Plot: x(k=1)",
    )
    p_qq_y = make_qq_panel(
        truth_y1,
        gen_y1;
        title="QQ Plot: y1(k=1)",
    )

    truth_x1, truth_x2 = collect_pairs_xx(tensor_truth; max_points=SYM_MAX_POINTS)
    gen_x1, gen_x2 = collect_pairs_xx(tensor_gen; max_points=SYM_MAX_POINTS)
    truth_yy1, truth_yy2 = collect_pairs_yy(tensor_truth; max_points=SYM_MAX_POINTS)
    gen_yy1, gen_yy2 = collect_pairs_yy(tensor_gen; max_points=SYM_MAX_POINTS)
    truth_xy1, truth_xy2 = collect_pairs_xy(tensor_truth; max_points=SYM_MAX_POINTS)
    gen_xy1, gen_xy2 = collect_pairs_xy(tensor_gen; max_points=SYM_MAX_POINTS)

    p_joint_xx = make_joint_kde_panel(
        truth_x1,
        truth_x2,
        gen_x1,
        gen_x2;
        title="Joint KDE: (x1, x2)",
        xlabel="x1 normalized",
        ylabel="x2 normalized",
    )

    p_joint_xy = make_joint_kde_panel(
        truth_xy1,
        truth_xy2,
        gen_xy1,
        gen_xy2;
        title="Joint KDE: (x_k, y_{k,i}) symmetry-averaged",
        xlabel="x_k normalized",
        ylabel="y_{k,i} normalized",
    )

    p_joint_yy = make_joint_kde_panel(
        truth_yy1,
        truth_yy2,
        gen_yy1,
        gen_yy2;
        title="Joint KDE: (y_{k,i}, y_{k,i+1}) symmetry-averaged",
        xlabel="y_{k,i} normalized",
        ylabel="y_{k,i+1} normalized",
    )

    kl_mode_raw, js_mode_raw, l1_mode_raw = modewise_univariate_metrics(
        tensor_truth,
        tensor_gen;
        nbins=PDF_BINS,
        bounds_mode=:minmax,
    )
    kl_mode_clip, js_mode_clip, l1_mode_clip = modewise_univariate_metrics(
        tensor_truth,
        tensor_gen;
        nbins=PDF_BINS,
        bounds_mode=:quantile,
        low_q=KL_LOW_Q,
        high_q=KL_HIGH_Q,
    )

    avg_mode_kl_raw = mean(kl_mode_raw)
    max_mode_kl_raw = maximum(kl_mode_raw)
    avg_x_kl_raw = mean(@view kl_mode_raw[:, 1])
    avg_y_kl_raw = mean(@view kl_mode_raw[:, 2:end])
    avg_mode_js_raw = mean(js_mode_raw)
    avg_mode_l1_raw = mean(l1_mode_raw)

    avg_mode_kl_clip = mean(kl_mode_clip)
    max_mode_kl_clip = maximum(kl_mode_clip)
    avg_x_kl_clip = mean(@view kl_mode_clip[:, 1])
    avg_y_kl_clip = mean(@view kl_mode_clip[:, 2:end])
    avg_mode_js_clip = mean(js_mode_clip)
    avg_mode_l1_clip = mean(l1_mode_clip)

    p_heat_kl = heatmap(1:C, 1:L, kl_mode_clip;
                        xlabel="Channel",
                        ylabel="Spatial index k",
                        title="Mode-wise KL(truth || generated, clipped)",
                        color=:viridis)
    truth_mean, gen_mean, truth_std, gen_std, rel_std_err = channel_moment_stats(tensor_truth, tensor_gen)

    score_diag = score_prediction_diagnostics(
        model,
        tensor_truth,
        trainer_cfg.sigma,
        device;
        nsamples=SCORE_EVAL_SAMPLES,
        batch_size=SCORE_EVAL_BATCH,
        seed=SCORE_EVAL_SEED,
        scatter_points=SCORE_SCATTER_POINTS,
    )
    # Requested publication figures:
    p_mom = mode_moment_panel(tensor_truth, tensor_gen)
    figB_path = save_stats_figure_3x3(
        joinpath(FIG_DIR, "figB_stats_3x3.png"),
        p_pdf_x,
        p_pdf_y,
        p_joint_xx,
        p_joint_yy,
        p_joint_xy,
        p_qq_x,
        p_qq_y,
        p_heat_kl,
        p_mom,
    )
    figC_path = save_dynamics_figure_3x2(
        joinpath(FIG_DIR, "figC_dynamics_3x2.png"),
        tensor_truth,
        tensor_gen;
        max_lag=MAX_ACF_LAG,
    )

    metrics = Dict(
        "avg_mode_kl_clipped" => avg_mode_kl_clip,
        "max_mode_kl_clipped" => max_mode_kl_clip,
        "avg_x_channel_kl_clipped" => avg_x_kl_clip,
        "avg_y_channels_kl_clipped" => avg_y_kl_clip,
        "avg_mode_js_clipped" => avg_mode_js_clip,
        "avg_mode_l1_clipped" => avg_mode_l1_clip,
        "avg_mode_kl_raw" => avg_mode_kl_raw,
        "max_mode_kl_raw" => max_mode_kl_raw,
        "avg_x_channel_kl_raw" => avg_x_kl_raw,
        "avg_y_channels_kl_raw" => avg_y_kl_raw,
        "avg_mode_js_raw" => avg_mode_js_raw,
        "avg_mode_l1_raw" => avg_mode_l1_raw,
        "global_kl_from_run_langevin" => Float64(result.kl_divergence),
        "score_pred_mse" => score_diag.mse,
        "score_pred_corr" => score_diag.corr,
        "avg_channel_mean_abs_error" => mean(abs.(gen_mean .- truth_mean)),
        "avg_channel_std_abs_error" => mean(abs.(gen_std .- truth_std)),
        "avg_channel_std_rel_error" => mean(rel_std_err),
        "target_avg_mode_kl" => TARGET_AVG_KL,
        "kl_low_q" => KL_LOW_Q,
        "kl_high_q" => KL_HIGH_Q,
        "langevin_profile_quick0_full1" => LANGEVIN_PROFILE == "quick" ? 0.0 : 1.0,
        "kept_snapshots_per_ensemble" => Float64(kept_diag.kept_per_ensemble),
        "total_kept_snapshots_all_ensembles" => Float64(kept_diag.total_kept),
        "burn_in_snapshots_per_ensemble" => Float64(kept_diag.burn_snapshots),
        "min_kept_snapshots_warn_threshold" => Float64(MIN_KEPT_SNAPSHOTS_WARN),
        "kept_snapshots_warn_flag" => kept_diag.kept_per_ensemble < MIN_KEPT_SNAPSHOTS_WARN ? 1.0 : 0.0,
    )
    write_metrics(METRICS_PATH, metrics)
    if !PIPELINE_MODE
        manifest_path = L96RunLayout.update_run_manifest!(
            RUN_DIR;
            stage="sample_and_compare",
            parameters=Dict(
                "langevin_profile" => LANGEVIN_PROFILE,
                "langevin_dt" => LANGEVIN_DT,
                "langevin_sample_dt" => LANGEVIN_SAMPLE_DT,
                "langevin_nsteps" => LANGEVIN_STEPS,
                "langevin_resolution" => LANGEVIN_RESOLUTION,
                "langevin_burn_in" => LANGEVIN_BURN_IN,
                "langevin_ensembles" => LANGEVIN_ENSEMBLES,
                "langevin_kept_snapshots_per_ensemble" => kept_diag.kept_per_ensemble,
                "langevin_seed" => LANGEVIN_SEED,
                "langevin_device_requested" => LANGEVIN_DEVICE_NAME,
                "langevin_device_effective" => effective_device_name,
                "pdf_bins" => PDF_BINS,
                "target_avg_mode_kl" => TARGET_AVG_KL,
            ),
            paths=Dict(
                "data_path" => abspath(DATA_PATH),
                "model_path" => abspath(MODEL_PATH),
                "eval_root" => isempty(EVAL_ROOT) ? "" : abspath(EVAL_ROOT),
                "eval_figures_dir" => abspath(FIG_DIR),
                "eval_metrics_path" => abspath(METRICS_PATH),
            ),
            artifacts=Dict(
                "eval_config" => isempty(eval_config_path) ? "" : abspath(eval_config_path),
                "figB_stats_3x3" => abspath(figB_path),
                "figC_dynamics_3x2" => abspath(figC_path),
            ),
            metrics=metrics,
        )
        summary_path = L96RunLayout.write_run_summary!(RUN_DIR)
        index_paths = L96RunLayout.refresh_runs_index!(L96RunLayout.default_runs_root(@__DIR__))
        compat_links = L96RunLayout.update_compat_links!(@__DIR__; figures_dir=FIG_DIR)
        L96RunLayout.write_latest_run!(@__DIR__, RUN_DIR)

        @info "Saved L96 comparison figures" dir = FIG_DIR run_dir = RUN_DIR eval_config = eval_config_path manifest = manifest_path summary = summary_path run_index = index_paths.index compat_links = compat_links
    else
        @info "Saved L96 comparison figures" dir = FIG_DIR run_dir = RUN_DIR
    end
    @info "KL divergence estimate" kl = result.kl_divergence
    @info "Mode-wise univariate metrics (clipped)" avg_mode_kl = avg_mode_kl_clip max_mode_kl = max_mode_kl_clip avg_x_kl = avg_x_kl_clip avg_y_kl = avg_y_kl_clip avg_mode_js = avg_mode_js_clip avg_mode_l1 = avg_mode_l1_clip
    @info "Mode-wise univariate metrics (raw)" avg_mode_kl = avg_mode_kl_raw max_mode_kl = max_mode_kl_raw avg_x_kl = avg_x_kl_raw avg_y_kl = avg_y_kl_raw avg_mode_js = avg_mode_js_raw avg_mode_l1 = avg_mode_l1_raw
    @info "Score prediction diagnostics" mse = score_diag.mse corr = score_diag.corr
    avg_mode_kl_clip <= TARGET_AVG_KL || @warn "Average clipped mode-wise KL is above target" avg_mode_kl = avg_mode_kl_clip target = TARGET_AVG_KL
    @info "Loaded cfg channels" in_channels = cfg.in_channels out_channels = cfg.out_channels
end

main()
