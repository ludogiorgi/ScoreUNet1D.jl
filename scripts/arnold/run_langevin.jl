# Standard command (from repository root):
# julia --project=. scripts/arnold/run_langevin.jl --params scripts/arnold/parameters_langevin.toml
# Nohup command:
# nohup julia --project=. scripts/arnold/run_langevin.jl --params scripts/arnold/parameters_langevin.toml > scripts/arnold/nohup_run_langevin.log 2>&1 &

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using BSON
using Flux
using HDF5
using KernelDensity
using LinearAlgebra
using Plots
using Printf
using Random
using ScoreUNet1D
using Statistics
using StatsBase
using TOML

include(joinpath(@__DIR__, "lib", "ArnoldCommon.jl"))
using .ArnoldCommon

const FIG_DPI_DEFAULT = 180

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_langevin.toml")
    checkpoint_override = ""
    output_dir_override = ""
    metrics_override = ""

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = args[i + 1]
            i += 2
        elseif a == "--checkpoint"
            i == length(args) && error("--checkpoint expects a value")
            checkpoint_override = args[i + 1]
            i += 2
        elseif a == "--output-dir"
            i == length(args) && error("--output-dir expects a value")
            output_dir_override = args[i + 1]
            i += 2
        elseif a == "--metrics-path"
            i == length(args) && error("--metrics-path expects a value")
            metrics_override = args[i + 1]
            i += 2
        else
            error("Unknown argument: $a")
        end
    end

    return (
        params_path=abspath(params_path),
        checkpoint_override=checkpoint_override,
        output_dir_override=output_dir_override,
        metrics_override=metrics_override,
    )
end

function as_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_bool(tbl::Dict{String,Any}, key::String, default) = parse_bool(get(tbl, key, default))

function load_config(path::AbstractString)
    isfile(path) || error("Langevin parameter file not found: $path")
    doc = TOML.parsefile(path)

    paths = as_table(doc, "paths")
    data = haskey(doc, "data") ? Dict{String,Any}(doc["data"]) : Dict{String,Any}()
    langevin = as_table(doc, "langevin")
    metrics = haskey(doc, "metrics") ? Dict{String,Any}(doc["metrics"]) : Dict{String,Any}()
    figures = haskey(doc, "figures") ? Dict{String,Any}(doc["figures"]) : Dict{String,Any}()

    cfg = Dict{String,Any}(
        "paths.data_params" => abspath(as_str(paths, "data_params", "scripts/arnold/parameters_data.toml")),
        "paths.model_path" => as_str(paths, "model_path", ""),
        "paths.output_dir" => as_str(paths, "output_dir", "scripts/arnold/output/langevin"),
        "paths.metrics_path" => as_str(paths, "metrics_path", "scripts/arnold/output/langevin/metrics.txt"),

        "data.observations_role" => as_str(data, "observations_role", "train_stochastic"),
        # Populated from central data config in main.
        "data.observations_path" => "",
        "data.dataset_key" => "",

        "langevin.device" => as_str(langevin, "device", "GPU:1"),
        "langevin.dt" => as_float(langevin, "dt", 0.004),
        "langevin.resolution" => as_int(langevin, "resolution", 25),
        "langevin.nsteps" => as_int(langevin, "nsteps", 80_000),
        "langevin.burn_in" => as_int(langevin, "burn_in", 10_000),
        "langevin.ensembles" => as_int(langevin, "ensembles", 512),
        "langevin.seed" => as_int(langevin, "seed", 21),
        "langevin.progress" => as_bool(langevin, "progress", false),
        "langevin.use_boundary" => as_bool(langevin, "use_boundary", true),
        "langevin.boundary_min" => as_float(langevin, "boundary_min", -12.0),
        "langevin.boundary_max" => as_float(langevin, "boundary_max", 12.0),
        "langevin.pdf_bins" => as_int(langevin, "pdf_bins", 80),

        "metrics.kl_low_q" => as_float(metrics, "kl_low_q", 0.001),
        "metrics.kl_high_q" => as_float(metrics, "kl_high_q", 0.999),
        "metrics.target_avg_mode_kl" => as_float(metrics, "target_avg_mode_kl", 0.01),
        "metrics.max_acf_lag" => as_int(metrics, "max_acf_lag", 200),

        "figures.dpi" => as_int(figures, "dpi", FIG_DPI_DEFAULT),
    )

    cfg["langevin.nsteps"] > cfg["langevin.burn_in"] || error("langevin.nsteps must be > burn_in")
    cfg["langevin.ensembles"] >= 1 || error("langevin.ensembles must be >= 1")
    cfg["langevin.resolution"] >= 1 || error("langevin.resolution must be >= 1")
    cfg["data.observations_role"] in ArnoldCommon.ARNOLD_DATASET_ROLES || error("data.observations_role must be one of $(join(ArnoldCommon.ARNOLD_DATASET_ROLES, ", "))")

    return cfg, doc
end

function load_x_tensor(path::AbstractString, key::AbstractString)
    isfile(path) || error("Observations file not found: $path")
    raw = h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset '$key' not found in $path")
        Float32.(read(h5[key]))
    end
    ndims(raw) == 2 || error("Expected 2D X dataset (N,K), found ndims=$(ndims(raw))")
    N, K = size(raw)
    return permutedims(reshape(raw, N, 1, K), (3, 2, 1))
end

function normalize_with_stats(tensor::Array{Float32,3}, stats::DataStats)
    K, C, _ = size(tensor)
    mean_lc = permutedims(stats.mean, (2, 1))
    std_lc = permutedims(stats.std, (2, 1))
    out = (tensor .- reshape(mean_lc, K, C, 1)) ./ reshape(std_lc, K, C, 1)
    return Array{Float32,3}(out)
end

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

function modewise_metrics(obs::Array{Float32,3}, gen::Array{Float32,3};
    nbins::Int,
    low_q::Float64,
    high_q::Float64)
    K, C, _ = size(obs)
    C == 1 || error("Expected one channel")
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

function collect_bivariate_pairs(tensor::Array{Float32,3}, lag::Int)
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
    m = min(max(max_lag, 1), n - 1)
    centered = series .- mean(series)
    variance = sum(abs2, centered) / max(n, 1)
    variance <= eps(Float64) && return ones(Float64, m + 1)
    acf = Array{Float64}(undef, m + 1)
    acf[1] = 1.0
    @inbounds for lag in 1:m
        t = n - lag
        acf[lag + 1] = dot(view(centered, 1:t), view(centered, lag + 1:lag + t)) / (t * variance)
    end
    return acf
end

function average_mode_acf(tensor::Array{Float32,3}, max_lag::Int)
    K, C, _ = size(tensor)
    C == 1 || error("Expected one channel")
    acc = zeros(Float64, max_lag + 1)
    for k in 1:K
        acc .+= autocorrelation(Float64.(vec(@view tensor[k, 1, :])), max_lag)
    end
    return acc ./ K
end

function save_stats_figure(path::AbstractString,
    obs::Array{Float32,3},
    gen::Array{Float32,3},
    kl_mode::Vector{Float64},
    js_mode::Vector{Float64},
    bins::Int,
    stein_matrix::Matrix{Float64})
    K = size(obs, 1)

    obs_vals = Float64.(vec(obs))
    gen_vals = Float64.(vec(gen))

    p_pdf = histogram(obs_vals; bins=bins, normalize=:pdf, alpha=0.4, color=:dodgerblue3, label="Observed", xlabel="X", ylabel="PDF", title="Univariate PDF")
    histogram!(p_pdf, gen_vals; bins=bins, normalize=:pdf, alpha=0.4, color=:tomato3, label="Generated")

    probs = collect(range(0.001, 0.999; length=min(length(obs_vals), length(gen_vals), 6000)))
    q_obs = quantile(obs_vals, probs)
    q_gen = quantile(gen_vals, probs)
    lo = min(minimum(q_obs), minimum(q_gen))
    hi = max(maximum(q_obs), maximum(q_gen))
    lo, hi = with_padding(lo, hi)
    p_qq = scatter(q_obs, q_gen; markersize=2, alpha=0.55, color=:darkslateblue, label="quantiles", xlabel="Observed quantiles", ylabel="Generated quantiles", title="QQ")
    plot!(p_qq, [lo, hi], [lo, hi]; color=:black, linestyle=:dash, label="y=x")

    obs_mean = [mean(@view obs[k, 1, :]) for k in 1:K]
    gen_mean = [mean(@view gen[k, 1, :]) for k in 1:K]
    obs_var = [var(@view obs[k, 1, :]) for k in 1:K]
    gen_var = [var(@view gen[k, 1, :]) for k in 1:K]
    p_mom = plot(1:K, obs_mean; marker=:circle, color=:dodgerblue3, label="Obs mean", xlabel="Mode k", ylabel="Moment", title="Per-mode moments")
    plot!(p_mom, 1:K, gen_mean; marker=:diamond, linestyle=:dash, color=:dodgerblue3, label="Gen mean")
    plot!(p_mom, 1:K, obs_var; marker=:circle, color=:darkorange3, label="Obs var")
    plot!(p_mom, 1:K, gen_var; marker=:diamond, linestyle=:dash, color=:darkorange3, label="Gen var")

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
            label="Observed",
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
            label="Generated",
        )
        p_biv[idx_lag] = p_pair
    end

    size(stein_matrix, 1) == size(stein_matrix, 2) || error("Stein matrix must be square")
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

    for p in (p_pdf, p_qq, p_mom, p_kl, p_biv[1], p_biv[2], p_biv[3], p_stein)
        plot!(p; left_margin=12Plots.mm, right_margin=6Plots.mm, top_margin=8Plots.mm, bottom_margin=10Plots.mm)
    end

    fig = plot(
        p_pdf,
        p_qq,
        p_mom,
        p_kl,
        p_biv[1],
        p_biv[2],
        p_biv[3],
        p_stein;
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

function save_dynamics_figure(path::AbstractString,
    obs::Array{Float32,3},
    gen::Array{Float32,3},
    max_lag::Int)
    K, C, To = size(obs)
    _, _, Tg = size(gen)
    C == 1 || error("Expected one channel")

    Tplot = min(To, Tg, 400)
    obs_hm = Float64.(obs[:, 1, 1:Tplot])
    gen_hm = Float64.(gen[:, 1, 1:Tplot])

    p_hm_obs = heatmap(1:Tplot, 1:K, obs_hm; xlabel="Time index", ylabel="Mode k", title="Observed X", color=:viridis)
    p_hm_gen = heatmap(1:Tplot, 1:K, gen_hm; xlabel="Time index", ylabel="Mode k", title="Generated X", color=:plasma)

    acf_obs = average_mode_acf(obs[:, :, 1:Tplot], max_lag)
    acf_gen = average_mode_acf(gen[:, :, 1:Tplot], max_lag)
    lag = 0:(length(acf_obs) - 1)
    p_acf = plot(lag, acf_obs; color=:dodgerblue3, linewidth=2, label="Observed", xlabel="Lag", ylabel="ACF", title="Average ACF over modes")
    plot!(p_acf, lag, acf_gen; color=:tomato3, linewidth=2, linestyle=:dash, label="Generated")
    hline!(p_acf, [0.0]; color=:gray40, linestyle=:dot, label="")

    t = 1:Tplot
    p_ts = plot(t, Float64.(vec(obs[1, 1, 1:Tplot])); color=:dodgerblue3, linewidth=2, label="Observed x1", xlabel="Time index", ylabel="X", title="Sample timeseries at k=1")
    plot!(p_ts, t, Float64.(vec(gen[1, 1, 1:Tplot])); color=:tomato3, linestyle=:dash, linewidth=2, label="Generated x1")

    for p in (p_hm_obs, p_hm_gen, p_acf, p_ts)
        plot!(p; left_margin=12Plots.mm, right_margin=6Plots.mm, top_margin=8Plots.mm, bottom_margin=10Plots.mm)
    end

    fig = plot(p_hm_obs, p_hm_gen, p_acf, p_ts; layout=(2, 2), size=(1900, 1300), left_margin=6Plots.mm, right_margin=6Plots.mm, top_margin=6Plots.mm, bottom_margin=6Plots.mm)
    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function write_metrics(path::AbstractString, metrics::Dict{String,Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        for k in sort!(collect(keys(metrics)))
            println(io, "$k=$(metrics[k])")
        end
    end
    return path
end

function resolve_device(name::AbstractString)
    try
        device = select_device(name)
        activate_device!(device)
        return device, name
    catch err
        @warn "Requested Langevin device unavailable; using CPU" requested = name error = sprint(showerror, err)
        device = ScoreUNet1D.CPUDevice()
        activate_device!(device)
        return device, "CPU"
    end
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg, raw_doc = load_config(parsed.params_path)
    data_cfg, _ = load_data_config(cfg["paths.data_params"])

    obs_role = cfg["data.observations_role"]
    obs_info = ensure_arnold_dataset_role!(data_cfg, obs_role)
    cfg["data.observations_path"] = obs_info["path"]
    cfg["data.dataset_key"] = obs_info["key"]

    checkpoint_path = if isempty(strip(parsed.checkpoint_override))
        as_str(Dict("m" => cfg["paths.model_path"]), "m", "")
    else
        parsed.checkpoint_override
    end
    isempty(strip(checkpoint_path)) && error("Model checkpoint path missing. Set [paths].model_path or --checkpoint")
    checkpoint_path = abspath(checkpoint_path)
    isfile(checkpoint_path) || error("Checkpoint not found: $checkpoint_path")

    output_dir = isempty(strip(parsed.output_dir_override)) ? cfg["paths.output_dir"] : parsed.output_dir_override
    output_dir = abspath(output_dir)
    metrics_path = isempty(strip(parsed.metrics_override)) ? cfg["paths.metrics_path"] : parsed.metrics_override
    metrics_path = abspath(metrics_path)

    contents = BSON.load(checkpoint_path)
    haskey(contents, :model) || error("Checkpoint missing :model")
    haskey(contents, :trainer_cfg) || error("Checkpoint missing :trainer_cfg")
    haskey(contents, :stats) || error("Checkpoint missing :stats")

    model = contents[:model]
    trainer_cfg = contents[:trainer_cfg]
    stats = contents[:stats]

    tensor_obs_raw = load_x_tensor(cfg["data.observations_path"], cfg["data.dataset_key"])
    tensor_obs = normalize_with_stats(tensor_obs_raw, stats)
    dataset = NormalizedDataset(tensor_obs, stats)

    device, device_name = resolve_device(cfg["langevin.device"])
    model = move_model(model, is_gpu(device) ? device : ScoreUNet1D.CPUDevice())
    sigma_train = Float32(getproperty(trainer_cfg, :sigma))

    lg_cfg = LangevinConfig(
        dt=cfg["langevin.dt"],
        sample_dt=cfg["langevin.dt"] * cfg["langevin.resolution"],
        nsteps=cfg["langevin.nsteps"],
        burn_in=cfg["langevin.burn_in"],
        resolution=cfg["langevin.resolution"],
        n_ensembles=cfg["langevin.ensembles"],
        nbins=cfg["langevin.pdf_bins"],
        sigma=sigma_train,
        seed=cfg["langevin.seed"],
        mode=:all,
        boundary=cfg["langevin.use_boundary"] ? (cfg["langevin.boundary_min"], cfg["langevin.boundary_max"]) : nothing,
        progress=cfg["langevin.progress"],
    )

    @info "Running Arnold Langevin integration" checkpoint = checkpoint_path device = device_name ensembles = cfg["langevin.ensembles"] nsteps = cfg["langevin.nsteps"]
    result = run_langevin(model, dataset, lg_cfg; device=device)

    K, C, _ = size(tensor_obs)
    traj4 = reshape(result.trajectory, K, C, :, size(result.trajectory, 3))
    tensor_gen = reshape(permutedims(traj4, (1, 2, 4, 3)), K, C, :)

    kl_mode, js_mode = modewise_metrics(
        tensor_obs,
        tensor_gen;
        nbins=cfg["langevin.pdf_bins"],
        low_q=cfg["metrics.kl_low_q"],
        high_q=cfg["metrics.kl_high_q"],
    )

    avg_mode_kl = mean(kl_mode)
    max_mode_kl = maximum(kl_mode)
    avg_mode_js = mean(js_mode)

    stein_sample_target = 6_000
    stein_stride = max(cld(length(dataset), stein_sample_target), 1)
    stein_matrix = compute_stein_matrix(
        model,
        dataset,
        sigma_train;
        batch_size=256,
        device=device,
        sample_stride=stein_stride,
    )

    figB = save_stats_figure(
        joinpath(output_dir, "figB_stats_4x2.png"),
        tensor_obs,
        tensor_gen,
        kl_mode,
        js_mode,
        cfg["langevin.pdf_bins"],
        stein_matrix,
    )
    figC = save_dynamics_figure(joinpath(output_dir, "figC_dynamics_2x2.png"), tensor_obs, tensor_gen, cfg["metrics.max_acf_lag"])

    metrics = Dict(
        "avg_mode_kl_clipped" => avg_mode_kl,
        "max_mode_kl_clipped" => max_mode_kl,
        "avg_mode_js_clipped" => avg_mode_js,
        "global_kl_from_run_langevin" => Float64(result.kl_divergence),
        "target_avg_mode_kl" => cfg["metrics.target_avg_mode_kl"],
        "langevin_ensembles" => Float64(cfg["langevin.ensembles"]),
        "langevin_nsteps" => Float64(cfg["langevin.nsteps"]),
        "langevin_resolution" => Float64(cfg["langevin.resolution"]),
        "stein_sample_stride" => Float64(stein_stride),
    )
    write_metrics(metrics_path, Dict{String,Float64}(k => Float64(v) for (k, v) in metrics))

    summary = Dict(
        "checkpoint_path" => checkpoint_path,
        "observations_path" => cfg["data.observations_path"],
        "device" => device_name,
        "figB" => abspath(figB),
        "figC" => abspath(figC),
        "metrics_path" => abspath(metrics_path),
        "metrics" => Dict{String,Any}(k => v for (k, v) in metrics),
    )
    summary_path = joinpath(output_dir, "langevin_summary.toml")
    mkpath(dirname(summary_path))
    open(summary_path, "w") do io
        TOML.print(io, summary)
    end

    if avg_mode_kl > cfg["metrics.target_avg_mode_kl"]
        @warn "Average mode KL above target" avg_mode_kl = avg_mode_kl target = cfg["metrics.target_avg_mode_kl"]
    end

    open(joinpath(output_dir, "parameters_langevin_used.toml"), "w") do io
        TOML.print(io, raw_doc)
    end

    println("figB=$(abspath(figB))")
    println("figC=$(abspath(figC))")
    println("metrics=$(abspath(metrics_path))")
    println("summary=$(abspath(summary_path))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
