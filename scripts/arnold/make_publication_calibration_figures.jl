#!/usr/bin/env julia

using GLMakie
using HDF5
using LaTeXStrings
using TOML

try
    using CairoMakie
catch
end

GLMakie.activate!()

const DEFAULT_PARAMS = joinpath(@__DIR__, "parameters_publication_calibration_figures.toml")
const PANEL_BG = RGBAf(1, 1, 1, 1)
const FIG_BG = RGBAf(1, 1, 1, 1)

const METHOD_COLORS = Dict(
    :finite_difference => :darkorange,
    :gaussian => :black,
    :unet => :steelblue,
)

const METHOD_LABELS = Dict(
    :finite_difference => "Finite diff",
    :gaussian => "Gaussian",
    :unet => "UNet",
)

const METHOD_LINESTYLES = Dict(
    :finite_difference => nothing,
    :gaussian => :dash,
    :unet => nothing,
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
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))

function load_plot_config(path::String)
    isfile(path) || error("Plot parameter file not found: $path")
    doc = TOML.parsefile(path)
    input_tbl = require_table(doc, "input")
    output_tbl = maybe_table(doc, "output")
    style_tbl = maybe_table(doc, "style")

    run_dir = abspath(as_str(input_tbl, "run_dir", ""))
    isempty(run_dir) && error("[input].run_dir must be provided")
    default_output = joinpath(run_dir, "publication_figures")

    return Dict(
        "run_dir" => run_dir,
        "output_dir" => abspath(as_str(output_tbl, "output_dir", default_output)),
        "convergence_basename" => as_str(output_tbl, "convergence_basename", "calibration_convergence_publication"),
        "gap_basename" => as_str(output_tbl, "gap_basename", "observable_gap_norm_publication"),
        "fontfamily" => as_str(style_tbl, "fontfamily", "Helvetica"),
        "basefontsize" => as_int(style_tbl, "basefontsize", 34),
        "ticksize" => as_int(style_tbl, "ticksize", 26),
        "linewidth" => as_float(style_tbl, "linewidth", 4.0),
        "target_linewidth" => as_float(style_tbl, "target_linewidth", 1.8),
        "convergence_panel_width" => as_int(style_tbl, "convergence_panel_width", 520),
        "convergence_panel_height" => as_int(style_tbl, "convergence_panel_height", 360),
        "gap_width" => as_int(style_tbl, "gap_width", 2200),
        "gap_height" => as_int(style_tbl, "gap_height", 1250),
        "px_per_unit" => as_float(style_tbl, "px_per_unit", 2.0),
        "gap_max_iteration" => as_int(style_tbl, "gap_max_iteration", -1),
    )
end

function publication_theme(; basefontsize::Int=40, fontname::String="Helvetica", ticksize::Int=30)
    Theme(
        fontsize=basefontsize,
        font=fontname,
        backgroundcolor=FIG_BG,
        Axis=(
            backgroundcolor=PANEL_BG,
            xlabelsize=basefontsize,
            ylabelsize=basefontsize,
            titlesize=basefontsize + 6,
            titlealign=:center,
            xgridvisible=false,
            ygridvisible=false,
            xticklabelsize=ticksize,
            yticklabelsize=ticksize,
            spinewidth=1.1,
            topspinevisible=false,
            rightspinevisible=false,
            minorticksvisible=false,
            xautolimitmargin=(0.06f0, 0.06f0),
            yautolimitmargin=(0.10f0, 0.10f0),
        ),
        Legend=(
            framevisible=true,
            patchsize=(28, 18),
            labelsize=basefontsize - 8,
            titlegap=6,
            padding=(12, 12, 12, 12),
        ),
    )
end

function method_order(methods::Vector{Symbol})
    preferred = [:finite_difference, :gaussian, :unet]
    ordered = Symbol[]
    for method in preferred
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

    out = Dict{String,Any}()
    h5open(history_path, "r") do h5
        methods = Symbol.(String.(vec(read(h5["meta/enabled_methods"]))))
        out["methods"] = method_order(methods)
        out["target_observables"] = Float64.(vec(read(h5["target_observables"])))
        out["free_parameters"] = Int.(vec(read(h5["selection/base_free_parameters"])))
        out["active_observables"] = Int.(vec(read(h5["selection/base_active_observables"])))

        theta_history = Dict{Symbol,Matrix{Float64}}()
        observable_history = Dict{Symbol,Matrix{Float64}}()
        gap_iterations = Dict{Symbol,Vector{Int}}()
        gap_values = Dict{Symbol,Vector{Float64}}()
        for method in out["methods"]
            theta_history[method] = Array{Float64}(read(h5["theta_history/$(String(method))"]))
            observable_history[method] = Array{Float64}(read(h5["observable_history/$(String(method))"]))
            gap_iterations[method] = Int.(vec(read(h5["observable_gap_history/$(String(method))/iterations"])))
            gap_values[method] = Float64.(vec(read(h5["observable_gap_history/$(String(method))/values"])))
        end
        out["theta_history"] = theta_history
        out["observable_history"] = observable_history
        out["gap_iterations"] = gap_iterations
        out["gap_values"] = gap_values
    end
    return out
end

function param_title(idx::Int)
    if idx == 5
        return L"\sigma"
    end
    return latexstring("\\alpha_{$(idx - 1)}")
end

observable_title(idx::Int) = latexstring("\\phi_{$idx}")

function add_method_lines!(ax::Axis, methods::Vector{Symbol}, series_lookup::Function;
    linewidth::Real,
    collect_legend::Bool=false,
    legend_lines=Makie.AbstractPlot[],
    legend_labels=String[])
    seen = Set{Symbol}()
    for method in methods
        ys = series_lookup(method)
        xs = collect(0:(length(ys) - 1))
        plt = lines!(
            ax,
            xs,
            ys;
            color=METHOD_COLORS[method],
            linewidth=linewidth,
            linestyle=METHOD_LINESTYLES[method],
        )
        if collect_legend && !(method in seen)
            push!(legend_lines, plt)
            push!(legend_labels, METHOD_LABELS[method])
            push!(seen, method)
        end
    end
    return nothing
end

function save_pub(basepath::String, fig; px::Real=2)
    png_path = basepath * ".png"
    mkpath(dirname(basepath))
    save(png_path, fig; px_per_unit=px)
    pdf_path = ""
    return (png=png_path, pdf=pdf_path)
end

function build_convergence_figure(history::Dict{String,Any}, cfg::Dict{String,Any})
    methods = history["methods"]
    theta_history = history["theta_history"]
    observable_history = history["observable_history"]
    free_parameters = history["free_parameters"]
    active_observables = history["active_observables"]
    target = history["target_observables"]

    length(free_parameters) == 5 || error("Expected 5 free parameters, got $(length(free_parameters))")
    length(active_observables) == 10 || error("Expected 10 active observables, got $(length(active_observables))")

    local_theme = publication_theme(basefontsize=cfg["basefontsize"], fontname=cfg["fontfamily"], ticksize=cfg["ticksize"])
    return with_theme(local_theme) do
        fig = Figure(size=(5 * cfg["convergence_panel_width"], 3 * cfg["convergence_panel_height"] + 180))
        top = fig[1, 1] = GridLayout(tellwidth=false)
        mid = fig[2, 1] = GridLayout(tellwidth=false)
        bot = fig[3, 1] = GridLayout(tellwidth=false)

        legend_lines = Makie.AbstractPlot[]
        legend_labels = String[]
        seen = Set{Symbol}()

        for col in 1:5
            pidx = free_parameters[col]
            ax = Axis(top[1, col], xlabel="", ylabel=col == 1 ? "θᵢ" : "", title=param_title(pidx))
            for method in methods
                ys = theta_history[method][:, pidx]
                xs = collect(0:(length(ys) - 1))
                plt = lines!(
                    ax,
                    xs,
                    ys;
                    color=METHOD_COLORS[method],
                    linewidth=cfg["linewidth"],
                    linestyle=METHOD_LINESTYLES[method],
                )
                if !(method in seen)
                    push!(legend_lines, plt)
                    push!(legend_labels, METHOD_LABELS[method])
                    push!(seen, method)
                end
            end
            tightlimits!(ax)
        end

        for col in 1:5
            obs_idx = active_observables[col]
            ax = Axis(mid[1, col], xlabel="", ylabel=col == 1 ? "Aᵢ" : "", title=observable_title(obs_idx))
            hlines!(ax, [target[obs_idx]]; color=:gray55, linestyle=:dot, linewidth=cfg["target_linewidth"])
            for method in methods
                ys = observable_history[method][:, obs_idx]
                xs = collect(0:(length(ys) - 1))
                lines!(
                    ax,
                    xs,
                    ys;
                    color=METHOD_COLORS[method],
                    linewidth=cfg["linewidth"],
                    linestyle=METHOD_LINESTYLES[method],
                )
            end
            tightlimits!(ax)
        end

        for col in 1:5
            obs_idx = active_observables[col + 5]
            ax = Axis(bot[1, col], xlabel="iteration", ylabel=col == 1 ? "Aᵢ" : "", title=observable_title(obs_idx))
            hlines!(ax, [target[obs_idx]]; color=:gray55, linestyle=:dot, linewidth=cfg["target_linewidth"])
            for method in methods
                ys = observable_history[method][:, obs_idx]
                xs = collect(0:(length(ys) - 1))
                lines!(
                    ax,
                    xs,
                    ys;
                    color=METHOD_COLORS[method],
                    linewidth=cfg["linewidth"],
                    linestyle=METHOD_LINESTYLES[method],
                )
            end
            tightlimits!(ax)
        end

        leg = Legend(fig, legend_lines, legend_labels; orientation=:horizontal, framevisible=true, tellwidth=false)
        leg.halign = :center
        fig[4, 1] = leg

        save_pub(joinpath(cfg["output_dir"], cfg["convergence_basename"]), fig; px=cfg["px_per_unit"])
    end
end

function build_gap_figure(history::Dict{String,Any}, cfg::Dict{String,Any})
    methods = history["methods"]
    gap_iterations = history["gap_iterations"]
    gap_values = history["gap_values"]
    max_iter = cfg["gap_max_iteration"]

    local_theme = publication_theme(basefontsize=cfg["basefontsize"], fontname=cfg["fontfamily"], ticksize=cfg["ticksize"])
    return with_theme(local_theme) do
        fig = Figure(size=(cfg["gap_width"], cfg["gap_height"]))
        ax = Axis(
            fig[1, 1],
            xlabel="iteration",
            ylabel=L"\mathcal{E}^{(n)} = \Vert G(\theta^{(n)}) - A \Vert_2",
            title=L"\mathcal{E}^{(n)}",
        )

        legend_lines = Makie.AbstractPlot[]
        legend_labels = String[]
        for method in methods
            xs_full = gap_iterations[method]
            ys_full = gap_values[method]
            keep = max_iter < 0 ? eachindex(xs_full) : findall(x -> x <= max_iter, xs_full)
            xs = xs_full[keep]
            ys = ys_full[keep]
            plt = lines!(
                ax,
                xs,
                ys;
                color=METHOD_COLORS[method],
                linewidth=cfg["linewidth"],
                linestyle=METHOD_LINESTYLES[method],
            )
            push!(legend_lines, plt)
            push!(legend_labels, METHOD_LABELS[method])
        end
        tightlimits!(ax)

        leg = Legend(fig, legend_lines, legend_labels; orientation=:horizontal, framevisible=true, tellwidth=false)
        leg.halign = :center
        fig[2, 1] = leg

        save_pub(joinpath(cfg["output_dir"], cfg["gap_basename"]), fig; px=cfg["px_per_unit"])
    end
end

function main(args=ARGS)
    params_path = parse_args(args)
    cfg = load_plot_config(params_path)
    history = load_run_history(cfg["run_dir"])
    mkpath(cfg["output_dir"])

    convergence_paths = build_convergence_figure(history, cfg)
    gap_paths = build_gap_figure(history, cfg)

    println("Saved convergence figure PNG: ", convergence_paths.png)
    println("Saved convergence figure PDF: ", convergence_paths.pdf)
    println("Saved observable-gap figure PNG: ", gap_paths.png)
    println("Saved observable-gap figure PDF: ", gap_paths.pdf)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
