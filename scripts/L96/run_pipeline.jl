# NOHUP example (run from repository root):
# nohup julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters.toml > scripts/L96/nohup_l96_J10.log 2>&1 &
# julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters.toml

using Dates

include(joinpath(@__DIR__, "lib", "Config.jl"))
include(joinpath(@__DIR__, "lib", "RunManager.jl"))
include(joinpath(@__DIR__, "lib", "TrainStage.jl"))
include(joinpath(@__DIR__, "lib", "LangevinStage.jl"))
include(joinpath(@__DIR__, "lib", "FigureFactory.jl"))
include(joinpath(@__DIR__, "lib", "Reporting.jl"))

using .L96Config
using .L96RunManager
using .L96TrainStage
using .L96LangevinStage
using .L96FigureFactory
using .L96Reporting

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters.toml")
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = args[i + 1]
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return abspath(params_path)
end

function append_log(log_path::AbstractString, msg::AbstractString)
    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    open(log_path, "a") do io
        println(io, "[$timestamp] $msg")
    end
end

function prune_run_layout!(run_dir::AbstractString, eval_rows::Vector{Dict{String,Any}})
    run_dir_abs = abspath(run_dir)
    keep = Set{String}()

    push!(keep, abspath(joinpath(run_dir, "parameters_used.toml")))
    push!(keep, abspath(joinpath(run_dir, "model", "score_model.bson")))
    push!(keep, abspath(joinpath(run_dir, "metrics", "run_summary.toml")))
    push!(keep, abspath(joinpath(run_dir, "logs", "pipeline.log")))
    push!(keep, abspath(joinpath(run_dir, "figures", "training", "figA_training_2x1.png")))

    for row in eval_rows
        epoch = Int(row["epoch"])
        epoch_tag = lpad(string(epoch), 4, '0')
        push!(keep, abspath(joinpath(run_dir, "metrics", "epoch_$(epoch_tag)_metrics.toml")))
        eval_dir = joinpath(run_dir, "figures", "eval_epoch_$(epoch_tag)")
        push!(keep, abspath(joinpath(eval_dir, "figB_stats_3x3.png")))
        push!(keep, abspath(joinpath(eval_dir, "figC_dynamics_3x2.png")))
    end

    for (root, _, files) in walkdir(run_dir; topdown=false)
        for f in files
            p = abspath(joinpath(root, f))
            if !(p in keep)
                rm(p; force=true)
            end
        end
        root_abs = abspath(root)
        if root_abs != run_dir_abs
            isempty(readdir(root)) && rm(root; force=true, recursive=true)
        end
    end

    mkpath(joinpath(run_dir, "model"))
    mkpath(joinpath(run_dir, "metrics"))
    mkpath(joinpath(run_dir, "figures", "training"))
    mkpath(joinpath(run_dir, "logs"))
    return nothing
end

function prune_training_intermediate!(figures_training_dir::AbstractString, training_metrics_csv::AbstractString)
    keep = Set([abspath(training_metrics_csv)])
    for name in readdir(figures_training_dir)
        p = abspath(joinpath(figures_training_dir, name))
        if isfile(p) && !(p in keep)
            rm(p; force=true)
        end
    end
    return nothing
end

function ensure_data_available!(params::Dict{String,Any}; base_dir::AbstractString=pwd())
    data_path = abspath(joinpath(base_dir, params["data.path"]))
    if isfile(data_path)
        return data_path
    end

    # Recover gracefully from legacy/broken symlinks at the configured data path.
    if islink(data_path) && !isfile(data_path)
        rm(data_path; force=true)
    end

    if !params["data.generate_if_missing"]
        error("Data file not found: $data_path. Set [data].generate_if_missing=true or provide existing data.path.")
    end

    env = copy(ENV)
    env["L96_J"] = string(params["run.J"])
    env["L96_DATA_PATH"] = data_path
    env["L96_RNG_SEED"] = string(params["run.seed"])
    env["L96_PIPELINE_MODE"] = "true"
    cmd = setenv(`julia --project=. scripts/L96/lib/generate_data.jl`, env)
    run(cmd)
    isfile(data_path) || error("Failed generating data at $data_path")
    return data_path
end

function main(args=ARGS)
    params_path = parse_args(args)
    params = L96Config.load_parameters(params_path)

    run_info = L96RunManager.next_run_dir(params; base_dir=pwd())
    dirs = L96RunManager.create_run_scaffold(run_info.run_dir)
    params_used = L96RunManager.copy_parameters_file(params_path, run_info.run_dir)

    log_path = joinpath(dirs["logs"], "pipeline.log")
    append_log(log_path, "pipeline start")
    append_log(log_path, "params_path=$params_path")
    append_log(log_path, "run_dir=$(run_info.run_dir)")

    data_path = ensure_data_available!(params; base_dir=pwd())
    append_log(log_path, "data_path=$data_path")

    training = L96TrainStage.run_training!(params, Dict{String,String}(k => String(v) for (k, v) in dirs); base_dir=pwd())
    prune_training_intermediate!(dirs["figures_training"], training["training_metrics_csv"])

    eval_rows = Dict{String,Any}[]
    eval_enabled = params["train.kl_eval.enabled"]
    ckpts = L96TrainStage.list_checkpoints(training["checkpoint_dir"])

    if eval_enabled
        isempty(ckpts) && error("No checkpoints found for KL evaluation. Check training checkpoint interval.")
        for (epoch, model_path) in ckpts
            row = L96LangevinStage.run_eval!(params, Dict{String,String}(k => String(v) for (k, v) in dirs), epoch, model_path; base_dir=pwd())
            push!(eval_rows, row)
        end
    end

    final_epoch = params["train.num_training_epochs"]
    if !eval_enabled || all(r -> Int(r["epoch"]) != final_epoch, eval_rows)
        row = L96LangevinStage.run_eval!(params, Dict{String,String}(k => String(v) for (k, v) in dirs), final_epoch, training["model_path"]; base_dir=pwd())
        push!(eval_rows, row)
    end

    sort!(eval_rows; by=r -> Int(r["epoch"]))

    figA_path = joinpath(dirs["figures_training"], "figA_training_2x1.png")
    L96FigureFactory.save_training_figure(training["training_metrics_csv"], eval_rows, figA_path; dpi=params["figures.dpi"])

    summary_path = L96Reporting.write_run_summary(run_info.run_dir, params, params_used, training, eval_rows, figA_path)

    # Keep final run layout minimal and plan-compliant.
    if haskey(training, "tmp_root")
        tmp_root = String(training["tmp_root"])
        isdir(tmp_root) && rm(tmp_root; recursive=true, force=true)
    end
    prune_run_layout!(run_info.run_dir, eval_rows)

    append_log(log_path, "summary_path=$summary_path")
    append_log(log_path, "pipeline done")

    println("L96 pipeline completed")
    println("run_dir=$(run_info.run_dir)")
    println("summary=$summary_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
