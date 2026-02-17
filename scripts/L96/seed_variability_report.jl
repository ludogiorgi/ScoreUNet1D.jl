if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Dates
using Plots
using Statistics
using TOML

function parse_args(args::Vector{String})
    cfg = Dict{String,Any}(
        "runs_root" => abspath("scripts/L96"),
        "J" => 10,
        "params" => "",
        "outdir" => "",
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--runs-root"
            i == length(args) && error("--runs-root expects a value")
            cfg["runs_root"] = abspath(args[i + 1])
            i += 2
        elseif a == "--J"
            i == length(args) && error("--J expects a value")
            cfg["J"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--params"
            i == length(args) && error("--params expects a value")
            cfg["params"] = abspath(args[i + 1])
            i += 2
        elseif a == "--outdir"
            i == length(args) && error("--outdir expects a value")
            cfg["outdir"] = abspath(args[i + 1])
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return cfg
end

function strip_seed_fields(x)
    if x isa Dict
        out = Dict{String,Any}()
        for (k, v) in x
            ks = String(k)
            if ks == "seed" || ks == "rng_seed"
                continue
            end
            out[ks] = strip_seed_fields(v)
        end
        return out
    elseif x isa Vector
        return [strip_seed_fields(v) for v in x]
    else
        return x
    end
end

function canonical_toml_string(tbl::Dict{String,Any})
    io = IOBuffer()
    TOML.print(io, tbl; sorted=true)
    return String(take!(io))
end

function same_hyperparams(a::Dict{String,Any}, b::Dict{String,Any})
    return canonical_toml_string(strip_seed_fields(a)) == canonical_toml_string(strip_seed_fields(b))
end

function collect_run_dirs(group_dir::AbstractString)
    if !isdir(group_dir)
        return String[]
    end
    dirs = String[]
    for name in sort(readdir(group_dir))
        startswith(name, "run_") || continue
        path = joinpath(group_dir, name)
        isdir(path) || continue
        push!(dirs, path)
    end
    return dirs
end

function as_int_vec(x)
    if x isa Vector
        return [Int(v) for v in x]
    end
    return Int[]
end

function as_float_vec(x)
    if x isa Vector
        return [Float64(v) for v in x]
    end
    return Float64[]
end

function load_run_record(run_dir::AbstractString)
    params_path = joinpath(run_dir, "parameters_used.toml")
    summary_path = joinpath(run_dir, "metrics", "run_summary.toml")
    isfile(params_path) || return nothing
    isfile(summary_path) || return nothing

    params = TOML.parsefile(params_path)
    summary = TOML.parsefile(summary_path)
    eval = get(summary, "evaluation", Dict{String,Any}())

    epochs = as_int_vec(get(eval, "epochs", Int[]))
    kl = as_float_vec(get(eval, "avg_mode_kl_clipped", Float64[]))
    length(epochs) == length(kl) || return nothing

    return Dict{String,Any}(
        "run_dir" => abspath(run_dir),
        "run_id" => basename(run_dir),
        "seed" => Int(get(get(params, "run", Dict{String,Any}()), "seed", -1)),
        "params" => params,
        "epochs" => epochs,
        "kl" => kl,
        "best_epoch" => Int(get(eval, "best_epoch", -1)),
        "best_kl" => Float64(get(eval, "best_avg_mode_kl_clipped", NaN)),
    )
end

function epoch_stats(records::Vector{Dict{String,Any}})
    epoch_map = Dict{Int,Vector{Float64}}()
    for rec in records
        epochs = rec["epochs"]
        kls = rec["kl"]
        for i in eachindex(epochs)
            ep = epochs[i]
            if !haskey(epoch_map, ep)
                epoch_map[ep] = Float64[]
            end
            push!(epoch_map[ep], kls[i])
        end
    end

    eps = sort(collect(keys(epoch_map)))
    means = Float64[]
    stds = Float64[]
    mins = Float64[]
    maxs = Float64[]
    counts = Int[]
    for ep in eps
        vals = epoch_map[ep]
        push!(means, mean(vals))
        push!(stds, length(vals) > 1 ? std(vals) : 0.0)
        push!(mins, minimum(vals))
        push!(maxs, maximum(vals))
        push!(counts, length(vals))
    end
    return (epochs=eps, mean=means, std=stds, min=mins, max=maxs, count=counts)
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{Any}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
    return path
end

function save_plots(outdir::AbstractString, stats, records)
    default(
        dpi=170,
        size=(1080, 720),
        linewidth=2,
        grid=true,
        gridalpha=0.25,
        framestyle=:box,
    )

    p1 = plot(stats.epochs, stats.mean;
              label="mean KL",
              color=:dodgerblue3,
              marker=:circle,
              markersize=4,
              xlabel="Epoch",
              ylabel="Avg mode KL (clipped)",
              title="KL Variability Across Seeds")
    plot!(p1, stats.epochs, stats.min; label="min KL", color=:seagreen4, linestyle=:dash)
    plot!(p1, stats.epochs, stats.max; label="max KL", color=:firebrick3, linestyle=:dash)
    savefig(p1, joinpath(outdir, "seed_variability_kl_by_epoch.png"))

    seeds = [Int(r["seed"]) for r in records]
    bests = [Float64(r["best_kl"]) for r in records]
    order = sortperm(seeds)
    seeds = seeds[order]
    bests = bests[order]

    p2 = scatter(seeds, bests;
                 color=:darkorange3,
                 markersize=6,
                 xlabel="Run seed",
                 ylabel="Best avg mode KL (clipped)",
                 title="Best Checkpoint KL by Seed",
                 label="seed runs")
    hline!(p2, [mean(bests)]; label="mean=$(round(mean(bests); digits=5))", color=:black, linestyle=:dash)
    savefig(p2, joinpath(outdir, "seed_variability_best_kl.png"))
end

function write_markdown_summary(path::AbstractString, cfg::Dict{String,Any}, records, stats)
    best_idx = argmin([Float64(r["best_kl"]) for r in records])
    worst_idx = argmax([Float64(r["best_kl"]) for r in records])
    best_run = records[best_idx]
    worst_run = records[worst_idx]

    open(path, "w") do io
        println(io, "# L96 Seed Variability Report")
        println(io)
        println(io, "- generated_at: `", Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"), "`")
        println(io, "- runs_root: `", cfg["runs_root"], "`")
        println(io, "- J: `", cfg["J"], "`")
        println(io, "- selected_runs: `", length(records), "`")
        println(io)
        println(io, "## Summary")
        println(io)
        println(io, "- best_run: `", best_run["run_id"], "` (seed=", best_run["seed"], ", best_kl=", round(best_run["best_kl"]; digits=6), ")")
        println(io, "- worst_run: `", worst_run["run_id"], "` (seed=", worst_run["seed"], ", best_kl=", round(worst_run["best_kl"]; digits=6), ")")
        println(io, "- mean_best_kl: `", round(mean([Float64(r["best_kl"]) for r in records]); digits=6), "`")
        println(io, "- std_best_kl: `", round(std([Float64(r["best_kl"]) for r in records]); digits=6), "`")
        println(io)
        println(io, "## Epoch Stats")
        println(io)
        println(io, "| Epoch | Mean KL | Std KL | Min KL | Max KL | n |")
        println(io, "|---:|---:|---:|---:|---:|---:|")
        for i in eachindex(stats.epochs)
            println(io, "| ", stats.epochs[i], " | ",
                    round(stats.mean[i]; digits=6), " | ",
                    round(stats.std[i]; digits=6), " | ",
                    round(stats.min[i]; digits=6), " | ",
                    round(stats.max[i]; digits=6), " | ",
                    stats.count[i], " |")
        end
    end
    return path
end

function main(args=ARGS)
    cfg = parse_args(args)
    runs_root = String(cfg["runs_root"])
    j = Int(cfg["J"])
    params_ref_path = String(cfg["params"])
    group_dir = joinpath(runs_root, "runs_J$(j)")
    outdir = isempty(String(cfg["outdir"])) ? joinpath(group_dir, "seed_variability_report") : String(cfg["outdir"])
    mkpath(outdir)

    ref_params = if isempty(params_ref_path)
        nothing
    else
        isfile(params_ref_path) || error("Reference params file not found: $params_ref_path")
        TOML.parsefile(params_ref_path)
    end

    records = Dict{String,Any}[]
    for run_dir in collect_run_dirs(group_dir)
        rec = load_run_record(run_dir)
        rec === nothing && continue
        if ref_params !== nothing && !same_hyperparams(rec["params"], ref_params)
            continue
        end
        push!(records, rec)
    end
    isempty(records) && error("No matching runs found in $group_dir")

    stats = epoch_stats(records)

    run_rows = Vector{Vector{Any}}()
    for rec in sort(records; by=r -> Int(r["seed"]))
        push!(run_rows, Any[
            rec["run_id"],
            rec["seed"],
            rec["best_epoch"],
            rec["best_kl"],
            rec["run_dir"],
        ])
    end
    runs_csv = write_csv(joinpath(outdir, "runs_best_kl.csv"),
                         ["run_id", "seed", "best_epoch", "best_kl", "run_dir"],
                         run_rows)

    epoch_rows = Vector{Vector{Any}}()
    for i in eachindex(stats.epochs)
        push!(epoch_rows, Any[
            stats.epochs[i],
            stats.mean[i],
            stats.std[i],
            stats.min[i],
            stats.max[i],
            stats.count[i],
        ])
    end
    epochs_csv = write_csv(joinpath(outdir, "epoch_kl_stats.csv"),
                           ["epoch", "mean_kl", "std_kl", "min_kl", "max_kl", "n"],
                           epoch_rows)

    save_plots(outdir, stats, records)
    md = write_markdown_summary(joinpath(outdir, "SEED_VARIABILITY_REPORT.md"), cfg, records, stats)

    println("Seed variability report completed")
    println("outdir=$(outdir)")
    println("runs_csv=$(runs_csv)")
    println("epochs_csv=$(epochs_csv)")
    println("summary=$(md)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

