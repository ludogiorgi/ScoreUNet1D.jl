#!/usr/bin/env julia

if isempty(get(ENV, "GKSwstype", ""))
    ENV["GKSwstype"] = "100"
end

using HDF5
using LaTeXStrings
using Plots
using Printf
using TOML

const DEFAULT_PARAMS = joinpath(@__DIR__, "parameters_publication_calibration_figures.toml")
const METHOD_ORDER = ("unet", "gaussian", "finite_difference")
const METHOD_STYLE = Dict(
    "unet" => (label="UNet", color=:orangered3, linestyle=:solid, marker=:circle, markersize=6),
    "gaussian" => (label="Gaussian", color=:black, linestyle=:dash, marker=:utriangle, markersize=6),
    "finite_difference" => (label="Finite difference", color=:dodgerblue3, linestyle=:solid, marker=:diamond, markersize=6),
)

function parse_args(args::Vector{String})
    params_path = DEFAULT_PARAMS
    i = 1
    while i <= length(args)
        arg = strip(args[i])
        if arg == "--params"
            i == length(args) && error("--params expects a value")
            params_path = strip(args[i + 1])
            i += 2
        elseif startswith(arg, "--params=")
            params_path = strip(split(arg, "="; limit=2)[2])
            i += 1
        else
            error("Unknown argument: $arg")
        end
    end
    return abspath(params_path)
end

function require_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table in plotting parameters")
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

function maybe_table(doc::Dict{String,Any}, key::String)
    if !haskey(doc, key)
        return Dict{String,Any}()
    end
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))

function load_plot_config(path::String)
    isfile(path) || error("Plot parameter file not found: $path")
    doc = TOML.parsefile(path)
    input_tbl = require_table(doc, "input")
    output_tbl = maybe_table(doc, "output")
    style_tbl = maybe_table(doc, "style")

    run_dir = abspath(as_str(input_tbl, "run_dir", ""))
    isempty(run_dir) && error("[input].run_dir must be provided")

    default_output = joinpath(run_dir, "publication_figures")
    output_dir = abspath(as_str(output_tbl, "output_dir", default_output))

    return Dict(
        "params_path" => path,
        "run_dir" => run_dir,
        "output_dir" => output_dir,
        "convergence_basename" => as_str(output_tbl, "convergence_basename", "calibration_convergence_publication"),
        "gap_basename" => as_str(output_tbl, "gap_basename", "observable_gap_norm_publication"),
        "dpi" => as_int(style_tbl, "dpi", 300),
        "fontfamily" => as_str(style_tbl, "fontfamily", "Computer Modern"),
        "convergence_width" => as_int(style_tbl, "convergence_width", 3200),
        "convergence_height" => as_int(style_tbl, "convergence_height", 2100),
        "gap_width" => as_int(style_tbl, "gap_width", 1800),
        "gap_height" => as_int(style_tbl, "gap_height", 980),
        "line_width" => as_float(style_tbl, "line_width", 2.8),
        "reference_line_width" => as_float(style_tbl, "reference_line_width", 1.8),
        "marker_size" => as_float(style_tbl, "marker_size", 6.5),
        "title_font_size" => as_int(style_tbl, "title_font_size", 13),
        "guide_font_size" => as_int(style_tbl, "guide_font_size", 12),
        "tick_font_size" => as_int(style_tbl, "tick_font_size", 10),
        "legend_font_size" => as_int(style_tbl, "legend_font_size", 11),
    )
end

function method_order(methods::Vector{String})
    ordered = String[]
    for method in METHOD_ORDER
        method in methods && push!(ordered, method)
    end
    for method in sort(methods)
        method in ordered || push!(ordered, method)
    end
    return ordered
end

function load_run_history(run_dir::String)
    history_path = joinpath(run_dir, "history", "calibration_history.hdf5")
    isfile(history_path) || error("Calibration history not found: $history_path")

    cfg_path = joinpath(run_dir, "config", "parameters_calibration.toml")
    isfile(cfg_path) || error("Copied calibration config not found: $cfg_path")
    run_cfg = TOML.parsefile(cfg_path)

    out = Dict{String,Any}()
    out["history_path"] = history_path
    out["run_cfg"] = run_cfg

    h5open(history_path, "r") do h5
        methods = String.(vec(read(h5["meta/enabled_methods"])))
        out["methods"] = method_order(methods)
        out["param_names"] = String.(vec(read(h5["meta/param_names"])))
        out["observable_names"] = String.(vec(read(h5["meta/observable_names"])))
        out["target_observables"] = vec(Float64.(read(h5["target_observables"])))
        out["free_parameters"] = Int.(vec(read(h5["selection/base_free_parameters"])))
        out["active_observables"] = Int.(vec(read(h5["selection/base_active_observables"])))

        theta_history = Dict{String,Matrix{Float64}}()
        observable_history = Dict{String,Matrix{Float64}}()
        gap_iterations = Dict{String,Vector{Int}}()
        gap_values = Dict{String,Vector{Float64}}()
        for method in out["methods"]
            theta_history[method] = Array{Float64}(read(h5["theta_history/$method"]))
            observable_history[method] = Array{Float64}(read(h5["observable_history/$method"]))
            gap_iterations[method] = Int.(vec(read(h5["observable_gap_history/$method/iterations"])))
            gap_values[method] = Float64.(vec(read(h5["observable_gap_history/$method/values"])))
        end
        out["theta_history"] = theta_history
        out["observable_history"] = observable_history
        out["gap_iterations"] = gap_iterations
        out["gap_values"] = gap_values
    end

    return out
end

function closure_reference(run_cfg::Dict{String,Any})
    closure_tbl = require_table(run_cfg, "closure")
    return Float64[
        get(closure_tbl, "alpha0_initial", get(closure_tbl, "alpha0", 0.0)),
        get(closure_tbl, "alpha1_initial", get(closure_tbl, "alpha1", 0.0)),
        get(closure_tbl, "alpha2_initial", get(closure_tbl, "alpha2", 0.0)),
        get(closure_tbl, "alpha3_initial", get(closure_tbl, "alpha3", 0.0)),
        get(closure_tbl, "sigma_initial", get(closure_tbl, "sigma", 0.0)),
    ]
end

function padded_limits(values::Vector{Float64}; frac::Float64=0.08)
    finite_vals = [v for v in values if isfinite(v)]
    isempty(finite_vals) && return (-1.0, 1.0)
    lo = minimum(finite_vals)
    hi = maximum(finite_vals)
    if isapprox(lo, hi; atol=1e-12, rtol=1e-9)
        span = max(abs(lo), 1.0)
        return (lo - frac * span, hi + frac * span)
    end
    pad = frac * (hi - lo)
    return (lo - pad, hi + pad)
end

function padded_xlimits(xs::Vector{Int}; frac::Float64=0.035)
    isempty(xs) && return (-0.5, 0.5)
    lo = minimum(xs)
    hi = maximum(xs)
    span = max(hi - lo, 1)
    pad = frac * span
    return (lo - pad, hi + pad)
end

function xtick_values(nmax::Int)
    nmax <= 6 && return collect(0:nmax)
    step = nmax <= 10 ? 2 : max(2, ceil(Int, nmax / 6))
    ticks = collect(0:step:nmax)
    last(ticks) == nmax || push!(ticks, nmax)
    return ticks
end

param_title(idx::Int) = idx == 5 ? latexstring("\\sigma") : latexstring("\\alpha_{$(idx - 1)}")

function observable_title(obs_idx::Int)
    if obs_idx == 1
        return latexstring("\\phi_{1} = \\langle X_k \\rangle")
    elseif obs_idx == 2
        return latexstring("\\phi_{2} = \\langle X_k^2 \\rangle")
    end
    lag = obs_idx - 2
    return latexstring("\\phi_{$obs_idx} = \\langle X_k X_{k+$lag} \\rangle")
end

function make_base_plot(style_cfg::Dict{String,Any}; size::Tuple{Int,Int})
    default(
        fontfamily=style_cfg["fontfamily"],
        dpi=style_cfg["dpi"],
        legendfontsize=style_cfg["legend_font_size"],
        guidefontsize=style_cfg["guide_font_size"],
        tickfontsize=style_cfg["tick_font_size"],
        titlefontsize=style_cfg["title_font_size"],
        linewidth=style_cfg["line_width"],
        markerstrokewidth=0.8,
        framestyle=:box,
        grid=:y,
        gridalpha=0.18,
        foreground_color_grid=:gray75,
        foreground_color_subplot=:black,
        background_color=:white,
        background_color_inside=:white,
        background_color_outside=:white,
        background_color_legend=:white,
        size=size,
    )
end

function build_parameter_panel(history::Dict{String,Matrix{Float64}},
    methods::Vector{String},
    pidx::Int,
    reference_value::Float64,
    style_cfg::Dict{String,Any};
    show_legend::Bool=false,
    show_xlabel::Bool=false,
    show_ylabel::Bool=false)
    xs = collect(0:(size(first(values(history)), 1) - 1))
    plt = plot(
        xlabel=show_xlabel ? "Iteration" : "",
        ylabel=show_ylabel ? "Parameter value" : "",
        title=param_title(pidx),
        legend=show_legend ? :topright : false,
        titleloc=:center,
        xticks=xtick_values(last(xs)),
        tickdirection=:out,
        top_margin=3Plots.mm,
    )

    all_vals = Float64[]
    for method in methods
        ys = history[method][:, pidx]
        append!(all_vals, ys)
        style = METHOD_STYLE[method]
        plot!(
            plt,
            xs,
            ys;
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=style_cfg["marker_size"],
            label=style.label,
        )
    end
    ylims!(plt, padded_limits(all_vals))
    xlims!(plt, padded_xlimits(xs))
    return plt
end

function build_observable_panel(history::Dict{String,Matrix{Float64}},
    methods::Vector{String},
    obs_idx::Int,
    target_value::Float64,
    style_cfg::Dict{String,Any};
    show_legend::Bool=false,
    show_xlabel::Bool=false,
    show_ylabel::Bool=false)
    xs = collect(0:(size(first(values(history)), 1) - 1))
    plt = plot(
        xlabel=show_xlabel ? "Iteration" : "",
        ylabel=show_ylabel ? "Observable value" : "",
        title=observable_title(obs_idx),
        legend=show_legend ? :topright : false,
        titleloc=:center,
        xticks=xtick_values(last(xs)),
        tickdirection=:out,
        top_margin=3Plots.mm,
    )

    all_vals = Float64[target_value]
    for method in methods
        ys = history[method][:, obs_idx]
        append!(all_vals, ys)
        style = METHOD_STYLE[method]
        plot!(
            plt,
            xs,
            ys;
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=style_cfg["marker_size"],
            label=style.label,
        )
    end
    hline!(
        plt,
        [target_value];
        color=:gray35,
        linestyle=:dash,
        linewidth=style_cfg["reference_line_width"],
        label=show_legend ? "Target" : "",
    )
    ylims!(plt, padded_limits(all_vals))
    xlims!(plt, padded_xlimits(xs))
    return plt
end

function save_plot_pair(fig, basepath::String)
    png_path = basepath * ".png"
    pdf_path = basepath * ".pdf"
    mkpath(dirname(basepath))
    savefig(fig, png_path)
    savefig(fig, pdf_path)
    return (png=png_path, pdf=pdf_path)
end

function build_convergence_figure(history_data::Dict{String,Any}, plot_cfg::Dict{String,Any})
    methods = history_data["methods"]
    theta_history = history_data["theta_history"]
    obs_history = history_data["observable_history"]
    free_parameters = history_data["free_parameters"]
    active_observables = history_data["active_observables"]
    target = history_data["target_observables"]
    reference = closure_reference(history_data["run_cfg"])

    length(free_parameters) == 5 || error("Publication convergence figure expects exactly 5 free parameters, got $(length(free_parameters))")
    length(active_observables) <= 10 || error("Publication convergence figure expects at most 10 active observables, got $(length(active_observables))")

    make_base_plot(plot_cfg; size=(plot_cfg["convergence_width"], plot_cfg["convergence_height"]))

    obs_slots = vcat(active_observables, fill(0, max(0, 10 - length(active_observables))))
    panels = Any[]
    for col in 1:5
        push!(
            panels,
            build_parameter_panel(
                theta_history,
                methods,
                free_parameters[col],
                reference[free_parameters[col]],
                plot_cfg;
                show_legend=(col == 1),
                show_xlabel=false,
                show_ylabel=(col == 1),
            ),
        )
    end

    for row in 1:2
        for col in 1:5
            obs_idx = obs_slots[(row - 1) * 5 + col]
            if obs_idx == 0
                push!(panels, plot(axis=false, grid=false, ticks=false, border=false))
                continue
            end
            push!(
                panels,
                build_observable_panel(
                    obs_history,
                    methods,
                    obs_idx,
                    target[obs_idx],
                    plot_cfg;
                    show_legend=false,
                    show_xlabel=(row == 2),
                    show_ylabel=(col == 1),
                ),
            )
        end
    end

    fig = plot(
        panels...;
        layout=grid(3, 5, heights=[0.34, 0.33, 0.33]),
        size=(plot_cfg["convergence_width"], plot_cfg["convergence_height"]),
        left_margin=15Plots.mm,
        right_margin=8Plots.mm,
        top_margin=11Plots.mm,
        bottom_margin=10Plots.mm,
    )
    return save_plot_pair(fig, joinpath(plot_cfg["output_dir"], plot_cfg["convergence_basename"]))
end

function build_gap_figure(history_data::Dict{String,Any}, plot_cfg::Dict{String,Any})
    methods = history_data["methods"]
    gap_iterations = history_data["gap_iterations"]
    gap_values = history_data["gap_values"]

    make_base_plot(plot_cfg; size=(plot_cfg["gap_width"], plot_cfg["gap_height"]))

    plt = plot(
        xlabel="Iteration",
        ylabel=L"\mathcal{E}^{(n)} = \| G(\theta^{(n)}) - A \|",
        title=L"\mathcal{E}^{(n)}",
        legend=:topright,
        framestyle=:box,
        grid=:both,
        gridalpha=0.18,
        tickdirection=:out,
        top_margin=4Plots.mm,
    )

    all_vals = Float64[]
    xmax = 0
    for method in methods
        xs = gap_iterations[method]
        ys = gap_values[method]
        append!(all_vals, ys)
        xmax = max(xmax, isempty(xs) ? 0 : maximum(xs))
        style = METHOD_STYLE[method]
        plot!(
            plt,
            xs,
            ys;
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=plot_cfg["marker_size"],
            label=style.label,
        )
    end

    xlims!(plt, padded_xlimits(vcat([0], [xmax]); frac=0.05))
    ylims!(plt, padded_limits(all_vals; frac=0.09))
    xticks!(plt, xtick_values(xmax))
    plot!(
        plt;
        left_margin=22Plots.mm,
        right_margin=9Plots.mm,
        top_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
    )
    return save_plot_pair(plt, joinpath(plot_cfg["output_dir"], plot_cfg["gap_basename"]))
end

function main(args=ARGS)
    params_path = parse_args(args)
    plot_cfg = load_plot_config(params_path)
    history_data = load_run_history(plot_cfg["run_dir"])
    mkpath(plot_cfg["output_dir"])

    convergence_paths = build_convergence_figure(history_data, plot_cfg)
    gap_paths = build_gap_figure(history_data, plot_cfg)

    println("Saved convergence figure PNG: ", convergence_paths.png)
    println("Saved convergence figure PDF: ", convergence_paths.pdf)
    println("Saved observable-gap figure PNG: ", gap_paths.png)
    println("Saved observable-gap figure PDF: ", gap_paths.pdf)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
