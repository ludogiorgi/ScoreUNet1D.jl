#!/usr/bin/env julia
# Rebuild convergence figure from an existing calibration run folder.
#
# Usage:
#   julia --project=. scripts/arnold/replot_convergence.jl \
#       --params scripts/arnold/parameters_replot_convergence.toml

using TOML
using Plots
using Printf

const OBS_NAMES = [
    "phi1_mean_x",
    "phi2_mean_x2",
    "phi3_mean_x_xm1",
    "phi4_mean_x_xm2",
    "phi5_mean_x_xm3",
]

const PARAM_NAMES = ["alpha0", "alpha1", "alpha2", "alpha3", "sigma"]

const PARAM_TITLES_LATEX = [
    raw"$\alpha_0$",
    raw"$\alpha_1$",
    raw"$\alpha_2$",
    raw"$\alpha_3$",
    raw"$\sigma$",
]

const OBS_TITLES_LATEX = [
    raw"$\phi_1=\langle X_k \rangle$",
    raw"$\phi_2=\langle X_k^2 \rangle$",
    raw"$\phi_3=\langle X_k X_{k-1} \rangle$",
    raw"$\phi_4=\langle X_k X_{k-2} \rangle$",
    raw"$\phi_5=\langle X_k X_{k-3} \rangle$",
]

as_bool(tbl::Dict{String,Any}, key::String, default) = Bool(get(tbl, key, default))
as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_replot_convergence.toml")

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

    return abspath(params_path)
end

function normalize_indices(raw, nmax::Int)
    vals = Int.(collect(raw))
    isempty(vals) && return collect(1:nmax)
    all(1 .<= vals .<= nmax) || error("Index values must be in 1:$nmax")
    return sort(unique(vals))
end

function trim_series(xs::Vector{Int}, ys::Vector{Float64}, drop_last_n::Int)
    drop_last_n < 0 && error("plot.drop_last_n must be >= 0")
    n = length(ys)
    keep = n - drop_last_n
    if keep <= 0
        return Int[], Float64[]
    end
    return xs[1:keep], ys[1:keep]
end

function style_for_method(method::String)
    if method == "unet"
        return (label="UNet", color=:orangered3, linestyle=:solid, marker=:circle, markersize=5)
    elseif method == "gaussian"
        return (label="Gaussian", color=:black, linestyle=:dash, marker=:utriangle, markersize=5)
    elseif method == "finite_difference"
        return (label="Finite-difference", color=:dodgerblue3, linestyle=:solid, marker=:diamond, markersize=5)
    end
    return (label=method, color=:gray30, linestyle=:solid, marker=:auto, markersize=4)
end

function method_order()
    return ["unet", "gaussian", "finite_difference"]
end

function enabled_methods(run_cfg::Dict{String,Any}, params_cfg::Dict{String,Any})
    override_tbl = haskey(params_cfg, "methods") ? Dict{String,Any}(params_cfg["methods"]) : Dict{String,Any}()
    methods = String[]

    for method in method_order()
        key = haskey(override_tbl, method) ? Bool(override_tbl[method]) : Bool(run_cfg["methods"][method])
        key && push!(methods, method)
    end

    isempty(methods) && error("No method enabled for plotting")
    return methods
end

function list_iteration_dirs(run_dir::String)
    dirs = String[]
    for name in readdir(run_dir)
        path = joinpath(run_dir, name)
        if isdir(path) && startswith(name, "iter_")
            push!(dirs, name)
        end
    end

    sort!(dirs; by = s -> begin
        v = tryparse(Int, split(s, "_")[end])
        v === nothing ? typemax(Int) : v
    end)
    return [joinpath(run_dir, d) for d in dirs]
end

function parse_observables_csv(path::String)
    isfile(path) || error("Missing observables file: $path")

    out = Dict{String,Vector{Float64}}()
    open(path, "r") do io
        first_line = true
        for ln in eachline(io)
            first_line && (first_line = false; continue)
            row = split(strip(ln), ",")
            length(row) == 3 || continue
            method, obs_name, val_s = row
            idx = findfirst(==(obs_name), OBS_NAMES)
            idx === nothing && continue

            if !haskey(out, method)
                out[method] = fill(NaN, length(OBS_NAMES))
            end
            out[method][idx] = parse(Float64, val_s)
        end
    end

    return out
end

function load_iter0_observables(run_dir::String)
    path = joinpath(run_dir, "truth", "observables_iter0.csv")
    if !isfile(path)
        return Dict{String,Vector{Float64}}()
    end
    return parse_observables_csv(path)
end

function load_target_from_truth(run_dir::String)
    path = joinpath(run_dir, "truth", "target_observables.csv")
    if !isfile(path)
        return nothing
    end

    vals = fill(NaN, length(OBS_NAMES))
    open(path, "r") do io
        first_line = true
        for ln in eachline(io)
            first_line && (first_line = false; continue)
            row = split(strip(ln), ",")
            length(row) == 2 || continue
            obs_name, val_s = row
            idx = findfirst(==(obs_name), OBS_NAMES)
            idx === nothing && continue
            vals[idx] = parse(Float64, val_s)
        end
    end

    return vals
end

function load_histories(run_dir::String, run_cfg::Dict{String,Any}, methods::Vector{String})
    theta0 = Float64[
        run_cfg["initial_theta"]["alpha0"],
        run_cfg["initial_theta"]["alpha1"],
        run_cfg["initial_theta"]["alpha2"],
        run_cfg["initial_theta"]["alpha3"],
        run_cfg["initial_theta"]["sigma"],
    ]

    theta_per_method = Dict{String,Vector{Vector{Float64}}}(m => [copy(theta0)] for m in methods)
    obs_history = Dict{String,Vector{Vector{Float64}}}(m => Vector{Vector{Float64}}() for m in methods)

    obs0_tbl = load_iter0_observables(run_dir)
    for method in methods
        if haskey(obs0_tbl, method)
            push!(obs_history[method], copy(obs0_tbl[method]))
        elseif haskey(obs0_tbl, "shared")
            push!(obs_history[method], copy(obs0_tbl["shared"]))
        end
    end

    iter_dirs = list_iteration_dirs(run_dir)
    for iter_dir in iter_dirs
        diag_path = joinpath(iter_dir, "results", "diagnostics.toml")
        obs_path = joinpath(iter_dir, "results", "observables.csv")
        isfile(diag_path) || continue
        isfile(obs_path) || continue

        diag = TOML.parsefile(diag_path)
        per_method = haskey(diag, "per_method") ? Dict{String,Any}(diag["per_method"]) : Dict{String,Any}()

        for method in methods
            if haskey(per_method, method)
                md = Dict{String,Any}(per_method[method])
                if haskey(md, "theta_proposed")
                    push!(theta_per_method[method], Float64.(collect(md["theta_proposed"])))
                end
            end
        end

        obs_tbl = parse_observables_csv(obs_path)
        for method in methods
            if haskey(obs_tbl, method)
                push!(obs_history[method], copy(obs_tbl[method]))
            end
        end
    end

    return theta_per_method, obs_history
end

function save_replotted_convergence(run_dir::String, output_path::String, methods, free_idx, active_idx, theta_per_method, obs_history, target_obs, dpi::Int, drop_last_n::Int, drop_iteration0_first_row::Bool)
    ncols = max(length(free_idx), length(active_idx))
    ncols >= 1 || error("Need at least one free parameter or active observable")

    default(fontfamily="Computer Modern", dpi=dpi, legendfontsize=10, guidefontsize=11, tickfontsize=10, titlefontsize=12)

    panels = Vector{Any}(undef, 2 * ncols)

    for col in 1:ncols
        if col <= length(free_idx)
            pidx = free_idx[col]
            pn = plot(; title=PARAM_TITLES_LATEX[pidx], xlabel="iteration", ylabel="value", legend=(col == 1 ? :best : false))
            for method in methods
                style = style_for_method(method)
                hist = get(theta_per_method, method, Vector{Vector{Float64}}())
                isempty(hist) && continue
                ys = Float64[v[pidx] for v in hist]
                xs = collect(0:(length(ys) - 1))
                if drop_iteration0_first_row && !isempty(ys)
                    xs = xs[2:end]
                    ys = ys[2:end]
                end
                xs, ys = trim_series(xs, ys, drop_last_n)
                isempty(ys) && continue
                plot!(pn, xs, ys; color=style.color, linestyle=style.linestyle, linewidth=2.0,
                      marker=style.marker, markersize=style.markersize, markerstrokewidth=0.5,
                      label=(col == 1 ? style.label : ""))
            end
            # Intentionally no horizontal reference line in the first row.
            panels[col] = pn
        else
            panels[col] = plot(; axis=false, grid=false)
        end
    end

    for col in 1:ncols
        idx = ncols + col
        if col <= length(active_idx)
            oidx = active_idx[col]
            pn = plot(; title=OBS_TITLES_LATEX[oidx], xlabel="iteration", ylabel="observable", legend=(col == 1 ? :best : false))
            for method in methods
                style = style_for_method(method)
                hist = get(obs_history, method, Vector{Vector{Float64}}())
                isempty(hist) && continue
                ys = Float64[v[oidx] for v in hist]
                has_obs0 = length(ys) > 0
                xs = has_obs0 ? collect(0:(length(ys) - 1)) : Int[]
                xs, ys = trim_series(xs, ys, drop_last_n)
                isempty(ys) && continue
                plot!(pn, xs, ys; color=style.color, linestyle=style.linestyle, linewidth=2.0,
                      marker=style.marker, markersize=style.markersize, markerstrokewidth=0.5,
                      label=(col == 1 ? style.label : ""))
            end
            if target_obs !== nothing
                hline!(pn, [target_obs[oidx]]; color=:gray40, linestyle=:dash, linewidth=1.5, label=(col == 1 ? "target" : ""))
            end
            panels[idx] = pn
        else
            panels[idx] = plot(; axis=false, grid=false)
        end
    end

    fig = plot(
        panels...;
        layout=(2, ncols),
        size=(max(1320, 430 * ncols), 930),
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
        top_margin=11Plots.mm,
        bottom_margin=10Plots.mm,
        plot_title="Calibration convergence",
        plot_titlefontsize=15,
    )

    mkpath(dirname(output_path))
    savefig(fig, output_path)
    return output_path
end

function main(args=ARGS)
    params_path = parse_args(args)
    params_cfg = TOML.parsefile(params_path)

    haskey(params_cfg, "paths") || error("Missing [paths] table in $params_path")
    p = Dict{String,Any}(params_cfg["paths"])
    run_dir = abspath(String(p["run_dir"]))
    isdir(run_dir) || error("run_dir does not exist: $run_dir")

    run_cfg_path = joinpath(run_dir, "config", "parameters_calibration.toml")
    isfile(run_cfg_path) || error("Missing run config: $run_cfg_path")
    run_cfg = TOML.parsefile(run_cfg_path)

    methods = enabled_methods(run_cfg, params_cfg)

    selection = haskey(params_cfg, "selection") ? Dict{String,Any}(params_cfg["selection"]) : Dict{String,Any}()
    free_idx = haskey(selection, "free_parameters") ? normalize_indices(selection["free_parameters"], 5) : normalize_indices(run_cfg["calibration"]["free_parameters"], 5)
    active_idx = haskey(selection, "active_observables") ? normalize_indices(selection["active_observables"], 5) : normalize_indices(run_cfg["calibration"]["active_observables"], 5)

    plot_cfg = haskey(params_cfg, "plot") ? Dict{String,Any}(params_cfg["plot"]) : Dict{String,Any}()
    dpi = as_int(plot_cfg, "dpi", haskey(run_cfg, "figures") ? Int(run_cfg["figures"]["dpi"]) : 180)
    drop_iteration0_first_row = as_bool(plot_cfg, "drop_iteration0_first_row", false)
    drop_last_n = if haskey(plot_cfg, "drop_last_n")
        as_int(plot_cfg, "drop_last_n", 0)
    elseif as_bool(plot_cfg, "drop_last_iteration", false)
        1
    else
        0
    end

    output_name = as_str(p, "output_file", "convergence_replot.png")
    output_path = isabspath(output_name) ? output_name : joinpath(run_dir, output_name)

    theta_per_method, obs_history = load_histories(run_dir, run_cfg, methods)
    target_obs = load_target_from_truth(run_dir)

    out = save_replotted_convergence(
        run_dir,
        output_path,
        methods,
        free_idx,
        active_idx,
        theta_per_method,
        obs_history,
        target_obs,
        dpi,
        drop_last_n,
        drop_iteration0_first_row,
    )

    @info "Saved replotted convergence figure" output=out run_dir
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
