# NOHUP example (run from repository root):
# nohup julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters.toml > scripts/L96/nohup_l96_J10.log 2>&1 &
# julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters.toml

using Dates
using HDF5
using TOML

include(joinpath(@__DIR__, "lib", "Config.jl"))
include(joinpath(@__DIR__, "lib", "RunManager.jl"))
include(joinpath(@__DIR__, "lib", "TrainStage.jl"))
include(joinpath(@__DIR__, "lib", "LangevinStage.jl"))
include(joinpath(@__DIR__, "lib", "ResponseStage.jl"))
include(joinpath(@__DIR__, "lib", "FigureFactory.jl"))
include(joinpath(@__DIR__, "lib", "Reporting.jl"))

using .L96Config
using .L96RunManager
using .L96TrainStage
using .L96LangevinStage
using .L96ResponseStage
using .L96FigureFactory
using .L96Reporting

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters.toml")
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = args[i+1]
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
    model_dir = joinpath(run_dir, "model")
    if isdir(model_dir)
        for name in readdir(model_dir)
            p = abspath(joinpath(model_dir, name))
            if isfile(p) && endswith(lowercase(name), ".bson")
                push!(keep, p)
            end
        end
    else
        push!(keep, abspath(joinpath(run_dir, "model", "score_model.bson")))
    end
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
        push!(keep, abspath(joinpath(eval_dir, "figD_responses_5x4.png")))
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

function archive_checkpoint_models!(checkpoint_pairs::Vector{Tuple{Int,String}}, model_dir::AbstractString)
    mkpath(model_dir)
    saved = Tuple{Int,String}[]
    for (epoch, src_path) in checkpoint_pairs
        isfile(src_path) || continue
        epoch_tag = lpad(string(epoch), 4, '0')
        dst = joinpath(model_dir, "score_model_epoch_$(epoch_tag).bson")
        cp(src_path, dst; force=true)
        push!(saved, (epoch, dst))
    end
    sort!(saved; by=x -> x[1])
    return saved
end

function observations_paths(params::Dict{String,Any}; base_dir::AbstractString=pwd())
    root = abspath(joinpath(base_dir, params["data.observations_root"]))
    j = Int(params["run.J"])
    jdir = joinpath(root, "J$(j)")
    data_path = joinpath(jdir, params["data.dataset_filename"])
    params_toml = joinpath(jdir, params["data.integration_params_filename"])
    return Dict(
        "root" => root,
        "jdir" => jdir,
        "data_path" => data_path,
        "params_toml" => params_toml,
    )
end

function read_dataset_metadata(path::AbstractString, dataset_key::AbstractString)
    return h5open(path, "r") do h5
        haskey(h5, dataset_key) || error("Dataset key '$dataset_key' not found in $path")
        dset = h5[dataset_key]
        dsize = size(dset)
        length(dsize) == 3 || error("Expected 3D dataset '$dataset_key' in $path")
        attrs = attributes(dset)

        out = Dict{String,Any}()
        out["shape"] = (Int(dsize[1]), Int(dsize[2]), Int(dsize[3]))

        for k in ("K", "J", "spinup_steps", "save_every")
            if haskey(attrs, k)
                out[k] = Int(read(attrs[k]))
            end
        end
        for k in ("F", "h", "c", "b", "dt", "process_noise_sigma")
            if haskey(attrs, k)
                out[k] = Float64(read(attrs[k]))
            end
        end
        for k in ("dynamics_reference", "coupling_scale", "fast_topology")
            if haskey(attrs, k)
                out[k] = String(read(attrs[k]))
            end
        end
        if haskey(attrs, "stochastic_process_noise")
            out["stochastic_process_noise"] = Bool(read(attrs["stochastic_process_noise"]))
        end
        if haskey(attrs, "stochastic_x_noise")
            out["stochastic_x_noise"] = Bool(read(attrs["stochastic_x_noise"]))
        end
        return out
    end
end

function expected_dataset_metadata(params::Dict{String,Any})
    return Dict{String,Any}(
        "J" => Int(params["run.J"]),
        "K" => Int(params["data.generation.K"]),
        "F" => Float64(params["data.generation.F"]),
        "h" => Float64(params["data.generation.h"]),
        "c" => Float64(params["data.generation.c"]),
        "b" => Float64(params["data.generation.b"]),
        "dt" => Float64(params["data.generation.dt"]),
        "spinup_steps" => Int(params["data.generation.spinup_steps"]),
        "save_every" => Int(params["data.generation.save_every"]),
        "process_noise_sigma" => Float64(params["data.generation.process_noise_sigma"]),
        "dynamics_reference" => "Schneider et al. (2017) Eqs. (11-13)",
        "coupling_scale" => "h*c/J",
        "fast_topology" => "twisted_ring_KJ",
        "stochastic_process_noise" => Float64(params["data.generation.process_noise_sigma"]) > 0.0,
        "stochastic_x_noise" => Bool(get(params, "data.generation.stochastic_x_noise", false)),
        "shape" => (
            Int(params["data.generation.nsamples"]),
            Int(params["run.J"]) + 1,
            Int(params["data.generation.K"]),
        ),
    )
end

function metadata_mismatches(found::Dict{String,Any}, expected::Dict{String,Any})
    mismatches = String[]
    for (k, ev) in expected
        haskey(found, k) || begin
            push!(mismatches, "$k missing (expected $ev)")
            continue
        end

        fv = found[k]
        if ev isa Tuple
            fv == ev || push!(mismatches, "$k expected $ev but found $fv")
        elseif ev isa Integer
            Int(fv) == Int(ev) || push!(mismatches, "$k expected $(Int(ev)) but found $(Int(fv))")
        elseif ev isa AbstractFloat
            isapprox(Float64(fv), Float64(ev); rtol=1e-10, atol=1e-12) ||
                push!(mismatches, "$k expected $(Float64(ev)) but found $(Float64(fv))")
        else
            fv == ev || push!(mismatches, "$k expected $ev but found $fv")
        end
    end
    return mismatches
end

function write_dataset_params_toml(path::AbstractString, data_path::AbstractString, dataset_key::AbstractString, attrs::Dict{String,Any})
    mkpath(dirname(path))
    shape = h5open(data_path, "r") do h5
        size(h5[dataset_key])
    end
    integ = Dict{String,Any}(k => v for (k, v) in attrs if k != "shape")
    cfg = Dict{String,Any}(
        "dataset" => Dict(
            "key" => String(dataset_key),
            "path" => abspath(data_path),
            "shape" => [Int(shape[1]), Int(shape[2]), Int(shape[3])],
        ),
        "integration" => integ,
    )
    open(path, "w") do io
        TOML.print(io, cfg)
    end
    return path
end

function generate_observations_dataset!(params::Dict{String,Any}, data_path::AbstractString, params_toml::AbstractString)
    env = copy(ENV)
    env["L96_J"] = string(params["run.J"])
    env["L96_K"] = string(params["data.generation.K"])
    env["L96_F"] = string(params["data.generation.F"])
    env["L96_H"] = string(params["data.generation.h"])
    env["L96_C"] = string(params["data.generation.c"])
    env["L96_B"] = string(params["data.generation.b"])
    env["L96_DT"] = string(params["data.generation.dt"])
    env["L96_SPINUP_STEPS"] = string(params["data.generation.spinup_steps"])
    env["L96_SAVE_EVERY"] = string(params["data.generation.save_every"])
    env["L96_NSAMPLES"] = string(params["data.generation.nsamples"])
    env["L96_PROCESS_NOISE_SIGMA"] = string(params["data.generation.process_noise_sigma"])
    if haskey(params, "data.generation.stochastic_x_noise")
        env["L96_STOCHASTIC_X_NOISE"] = string(params["data.generation.stochastic_x_noise"])
    end
    env["L96_RNG_SEED"] = string(params["data.generation.rng_seed"])
    env["L96_DATA_PATH"] = data_path
    env["L96_DATASET_PARAMS_TOML_PATH"] = params_toml
    env["L96_PIPELINE_MODE"] = "true"
    cmd = setenv(`julia --project=. scripts/L96/lib/generate_data.jl`, env)
    run(cmd)
    isfile(data_path) || error("Failed generating data at $data_path")
    return data_path
end

function ensure_data_available!(params::Dict{String,Any}; base_dir::AbstractString=pwd())
    paths = observations_paths(params; base_dir=base_dir)
    data_path = paths["data_path"]
    params_toml = paths["params_toml"]
    mkpath(paths["jdir"])

    dataset_key = params["data.dataset_key"]
    expected = expected_dataset_metadata(params)

    if isfile(data_path)
        found = read_dataset_metadata(data_path, dataset_key)
        mismatches = metadata_mismatches(found, expected)
        if !isempty(mismatches)
            msg = "Existing dataset metadata mismatch at $data_path:\n  - " * join(mismatches, "\n  - ")
            if !params["data.generate_if_missing"]
                error(msg * "\nSet data.generate_if_missing=true to regenerate.")
            end
            @warn msg * "\nRegenerating dataset because data.generate_if_missing=true"
            rm(data_path; force=true)
        end
    elseif islink(data_path) && !isfile(data_path)
        rm(data_path; force=true)
    end

    if !isfile(data_path)
        if !params["data.generate_if_missing"]
            error("Dataset not found for J=$(params["run.J"]) at $data_path and generation is disabled.")
        end
        generate_observations_dataset!(params, data_path, params_toml)
    end

    # Keep integration metadata TOML synchronized with the currently used dataset.
    attrs = read_dataset_metadata(data_path, dataset_key)
    write_dataset_params_toml(params_toml, data_path, dataset_key, attrs)

    # Force downstream stages to use the J-indexed observation dataset.
    params["data.path"] = data_path
    return data_path
end

function run_eval_and_response!(params::Dict{String,Any},
                                dirs::Dict{String,String},
                                epoch::Int,
                                model_path::AbstractString;
                                base_dir::AbstractString=pwd())
    row = L96LangevinStage.run_eval!(params, dirs, epoch, model_path; base_dir=base_dir)
    tag = String(row["tag"])
    resp = L96ResponseStage.run_response_figure!(params, dirs, epoch, model_path, tag; base_dir=base_dir)
    if Bool(get(resp, "enabled", false))
        row["figure_D"] = String(get(resp, "figD_path", ""))
        row["response_run_dir"] = String(get(resp, "response_run_dir", ""))
    else
        row["figure_D"] = ""
        row["response_run_dir"] = ""
    end
    return row
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
    obs_paths = observations_paths(params; base_dir=pwd())
    append_log(log_path, "observations_dir=$(obs_paths["jdir"])")
    append_log(log_path, "integration_params_toml=$(obs_paths["params_toml"])")

    training = L96TrainStage.run_training!(params, Dict{String,String}(k => String(v) for (k, v) in dirs); base_dir=pwd())
    prune_training_intermediate!(dirs["figures_training"], training["training_metrics_csv"])

    eval_rows = Dict{String,Any}[]
    eval_enabled = params["train.kl_eval.enabled"]
    ckpts = L96TrainStage.list_checkpoints(training["checkpoint_dir"])
    archived_ckpts = archive_checkpoint_models!(ckpts, dirs["model"])
    for (epoch, path) in archived_ckpts
        append_log(log_path, "[train] archived_checkpoint epoch=$epoch path=$(abspath(path))")
    end

    if eval_enabled
        isempty(ckpts) && error("No checkpoints found for KL evaluation. Check training checkpoint interval.")
        for (epoch, model_path) in ckpts
            row = run_eval_and_response!(params, Dict{String,String}(k => String(v) for (k, v) in dirs), epoch, model_path; base_dir=pwd())
            push!(eval_rows, row)
        end
    end

    final_epoch = params["train.num_training_epochs"]
    if !eval_enabled || all(r -> Int(r["epoch"]) != final_epoch, eval_rows)
        row = run_eval_and_response!(params, Dict{String,String}(k => String(v) for (k, v) in dirs), final_epoch, training["model_path"]; base_dir=pwd())
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
