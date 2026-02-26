# Standard command (from repository root):
# julia --project=. scripts/arnold/run_pipeline.jl --params scripts/arnold/parameters_pipeline.toml
# Nohup command:
# nohup julia --project=. scripts/arnold/run_pipeline.jl --params scripts/arnold/parameters_pipeline.toml > scripts/arnold/nohup_run_pipeline.log 2>&1 &

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Dates
using Distributed
using Plots
using Printf
using Statistics
using TOML

include(joinpath(@__DIR__, "lib", "ArnoldCommon.jl"))
using .ArnoldCommon

const DEFAULT_DPI = 180

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_pipeline.toml")

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

function require_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_bool(tbl::Dict{String,Any}, key::String, default) = parse_bool(get(tbl, key, default))

function load_pipeline_config(path::AbstractString)
    isfile(path) || error("Pipeline parameter file not found: $path")
    doc = TOML.parsefile(path)

    run = require_table(doc, "run")
    paths = require_table(doc, "paths")
    resources = haskey(doc, "resources") ? Dict{String,Any}(doc["resources"]) : Dict{String,Any}()
    evaluation = haskey(doc, "evaluation") ? Dict{String,Any}(doc["evaluation"]) : Dict{String,Any}()
    targets = haskey(doc, "targets") ? Dict{String,Any}(doc["targets"]) : Dict{String,Any}()

    cfg = Dict{String,Any}(
        "run.runs_root" => abspath(as_str(run, "runs_root", "scripts/arnold/runs")),
        "run.run_id_padding" => as_int(run, "run_id_padding", 3),

        "paths.train_params" => abspath(as_str(paths, "train_params", "scripts/arnold/parameters_train.toml")),
        "paths.langevin_params" => abspath(as_str(paths, "langevin_params", "scripts/arnold/parameters_langevin.toml")),
        "paths.responses_params" => abspath(as_str(paths, "responses_params", "scripts/arnold/parameters_responses.toml")),

        "resources.train_device" => as_str(resources, "train_device", "GPU:0"),
        "resources.langevin_device" => as_str(resources, "langevin_device", "GPU:1"),
        "resources.responses_score_device" => as_str(resources, "responses_score_device", "GPU:1"),
        "resources.responses_threads" => as_str(resources, "responses_threads", "auto"),
        "resources.checkpoint_poll_seconds" => as_float(resources, "checkpoint_poll_seconds", 2.0),
        "resources.queue_poll_seconds" => as_float(resources, "queue_poll_seconds", 1.0),

        "evaluation.run_langevin" => as_bool(evaluation, "run_langevin", true),
        "evaluation.run_responses" => as_bool(evaluation, "run_responses", true),
        "evaluation.evaluate_every_checkpoint" => as_bool(evaluation, "evaluate_every_checkpoint", true),
        "evaluation.evaluate_final_model" => as_bool(evaluation, "evaluate_final_model", true),
        "evaluation.response_apply_correction" => as_bool(evaluation, "response_apply_correction", true),
        "evaluation.response_kind" => lowercase(as_str(evaluation, "response_kind", "heaviside")),

        "targets.target_avg_mode_kl" => as_float(targets, "target_avg_mode_kl", 0.01),
    )

    cfg["evaluation.response_kind"] in ("heaviside", "impulse") || error("evaluation.response_kind must be heaviside or impulse")
    isfile(cfg["paths.train_params"]) || error("Missing train params file: $(cfg["paths.train_params"])")
    isfile(cfg["paths.langevin_params"]) || error("Missing Langevin params file: $(cfg["paths.langevin_params"])")
    isfile(cfg["paths.responses_params"]) || error("Missing responses params file: $(cfg["paths.responses_params"])")

    return cfg, doc
end

function table!(doc::Dict{String,Any}, key::String)
    if !haskey(doc, key) || !(doc[key] isa AbstractDict)
        doc[key] = Dict{String,Any}()
    elseif !(doc[key] isa Dict{String,Any})
        doc[key] = Dict{String,Any}(doc[key])
    end
    return doc[key]
end

function write_toml(path::AbstractString, doc::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return path
end

function read_train_meta(path::AbstractString)
    doc = TOML.parsefile(path)
    train = haskey(doc, "train") ? Dict{String,Any}(doc["train"]) : Dict{String,Any}()
    return (
        epochs=Int(get(train, "epochs", 1)),
        checkpoint_every=Int(get(train, "checkpoint_every", 0)),
        save_state_every=Int(get(train, "save_state_every", 0)),
    )
end

function read_dataset_roles(train_params_path::AbstractString, responses_params_path::AbstractString)
    train_doc = TOML.parsefile(train_params_path)
    train_data = haskey(train_doc, "data") ? Dict{String,Any}(train_doc["data"]) : Dict{String,Any}()
    train_role = String(get(train_data, "dataset_role", "train_stochastic"))

    responses_doc = TOML.parsefile(responses_params_path)
    responses_paths = haskey(responses_doc, "paths") ? Dict{String,Any}(responses_doc["paths"]) : Dict{String,Any}()
    gfdt_role = String(get(responses_paths, "gfdt_dataset_role", "gfdt_stochastic"))

    return train_role, gfdt_role
end

function build_effective_params(cfg::Dict{String,Any}, dirs::Dict{String,String}, pipeline_params_path::AbstractString)
    params_dir = dirs["params"]

    train_doc = TOML.parsefile(cfg["paths.train_params"])
    train_paths = table!(train_doc, "paths")
    train_data = table!(train_doc, "data")
    train_tbl = table!(train_doc, "train")
    train_tbl["device"] = cfg["resources.train_device"]
    data_params_src = abspath(String(get(train_paths, "data_params", "scripts/arnold/parameters_data.toml")))
    isfile(data_params_src) || error("Missing central data params file: $data_params_src")
    data_params_used = joinpath(params_dir, "parameters_data_used.toml")
    cp(data_params_src, data_params_used; force=true)
    train_paths["data_params"] = data_params_used
    train_used = write_toml(joinpath(params_dir, "parameters_train_used.toml"), train_doc)

    train_role = String(get(train_data, "dataset_role", "train_stochastic"))

    langevin_doc = TOML.parsefile(cfg["paths.langevin_params"])
    langevin_paths = table!(langevin_doc, "paths")
    langevin_data = table!(langevin_doc, "data")
    langevin_cfg = table!(langevin_doc, "langevin")
    langevin_metrics = table!(langevin_doc, "metrics")

    langevin_paths["model_path"] = ""
    langevin_paths["data_params"] = data_params_used
    langevin_data["observations_role"] = train_role
    langevin_cfg["device"] = cfg["resources.langevin_device"]
    langevin_metrics["target_avg_mode_kl"] = cfg["targets.target_avg_mode_kl"]
    langevin_used = write_toml(joinpath(params_dir, "parameters_langevin_used.toml"), langevin_doc)

    responses_doc = TOML.parsefile(cfg["paths.responses_params"])
    responses_paths = table!(responses_doc, "paths")
    responses_methods = table!(responses_doc, "methods")
    responses_gfdt = table!(responses_doc, "gfdt")

    responses_paths["checkpoint_path"] = ""
    responses_paths["data_params"] = data_params_used
    responses_paths["cache_root"] = joinpath(dirs["cache"], "responses")
    responses_paths["output_root"] = joinpath(dirs["figures"], "responses")
    responses_methods["apply_score_correction"] = cfg["evaluation.response_apply_correction"]
    responses_methods["response_kind"] = cfg["evaluation.response_kind"]
    responses_gfdt["score_device"] = cfg["resources.responses_score_device"]
    responses_used = write_toml(joinpath(params_dir, "parameters_responses_used.toml"), responses_doc)

    cp(pipeline_params_path, joinpath(params_dir, "parameters_pipeline_used.toml"); force=true)

    return Dict{String,String}(
        "data" => data_params_used,
        "train" => train_used,
        "langevin" => langevin_used,
        "responses" => responses_used,
    )
end

function launch_logged(cmd::Cmd, log_path::AbstractString)
    mkpath(dirname(log_path))
    io = open(log_path, "a")
    proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    return proc, io
end

function latest_run_dir(root::AbstractString)
    isdir(root) || return ""
    best_id = -1
    best_name = ""
    for name in readdir(root)
        m = match(r"^run_(\d+)$", name)
        m === nothing && continue
        id = parse(Int, m.captures[1])
        if id > best_id
            best_id = id
            best_name = name
        end
    end
    best_id < 0 && return ""
    return joinpath(root, best_name)
end

function get_nested_float(doc::Dict{String,Any}, path::Vector{String}; default::Float64=NaN)
    node = doc
    for (i, key) in enumerate(path)
        if i == length(path)
            haskey(node, key) || return default
            return Float64(node[key])
        end
        haskey(node, key) || return default
        nxt = node[key]
        nxt isa Dict{String,Any} || return default
        node = nxt
    end
    return default
end

function pipeline_project_root()
    project_file = Base.active_project()
    project_file === nothing && return abspath(joinpath(@__DIR__, "..", ".."))
    return dirname(project_file)
end

function eval_layout(dirs::Dict{String,String}, epoch::Int)
    epoch_tag = lpad(string(epoch), 4, '0')
    eval_tag = "eval_epoch_" * epoch_tag
    eval_dir = joinpath(dirs["figures"], eval_tag)
    return (
        epoch_tag=epoch_tag,
        eval_tag=eval_tag,
        eval_dir=eval_dir,
        responses_root=joinpath(eval_dir, "responses"),
        metrics_txt=joinpath(dirs["metrics"], "epoch_" * epoch_tag * "_langevin_metrics.txt"),
        metrics_toml=joinpath(dirs["metrics"], "epoch_" * epoch_tag * "_metrics.toml"),
    )
end

function start_stage_worker(stage::Symbol, script_path::AbstractString;
    threads::Union{Nothing,String}=nothing)
    project_root = pipeline_project_root()
    exeflags = threads === nothing ?
        "--project=$(project_root)" :
        "--project=$(project_root) --threads $(threads)"

    pid = only(addprocs(1; exeflags=exeflags))
    script_abs = abspath(script_path)
    remotecall_fetch(
        Base.eval,
        pid,
        Main,
        quote
            if get(ENV, "GKSwstype", "") == ""
                ENV["GKSwstype"] = "100"
            end
            import Pkg
            Pkg.activate($project_root; io=devnull)
            cd($project_root)
            include($script_abs)
            isdefined(Main, :main) || error("Worker stage script did not define main(args): $script_abs")
            global __arnold_stage_main = getfield(Main, :main)
            global __arnold_stage_name = $(QuoteNode(stage))
            nothing
        end,
    )
    return pid
end

function stop_stage_workers!(stage_workers::Dict{Symbol,Int}, pipeline_log::AbstractString)
    isempty(stage_workers) && return nothing
    pids = collect(values(stage_workers))
    try
        rmprocs(pids)
        append_log(pipeline_log, "stopped persistent workers pids=$(join(pids, ','))")
    catch err
        append_log(pipeline_log, "warning failed to stop persistent workers: $(sprint(showerror, err))")
    end
    return nothing
end

function run_stage_on_worker!(pid::Int, stage::Symbol, args::Vector{String}, log_path::AbstractString)
    mkpath(dirname(log_path))
    worker_log = abspath(log_path)
    argv = collect(args)
    arg_preview = join(argv, ' ')
    log_header = "[persistent-worker] stage=$(stage) args=$(arg_preview)"
    remotecall_fetch(
        Base.eval,
        pid,
        Main,
        quote
            open($worker_log, "a") do io
                println(io, $log_header)
                flush(io)
                redirect_stdout(io) do
                    redirect_stderr(io) do
                        try
                            Main.__arnold_stage_main($argv)
                        catch err
                            showerror(io, err, catch_backtrace())
                            println(io)
                            flush(io)
                            rethrow(err)
                        end
                    end
                end
            end
            nothing
        end,
    )
    return nothing
end

function required_eval_stages(cfg::Dict{String,Any})
    stages = Symbol[]
    cfg["evaluation.run_langevin"] && push!(stages, :langevin)
    cfg["evaluation.run_responses"] && push!(stages, :responses)
    return stages
end

function run_langevin_stage_for_checkpoint!(cfg::Dict{String,Any},
    effective_params::Dict{String,String},
    dirs::Dict{String,String},
    stage_workers::Dict{Symbol,Int},
    epoch::Int,
    checkpoint_path::AbstractString)
    layout = eval_layout(dirs, epoch)
    mkpath(layout.eval_dir)

    pipeline_log = joinpath(dirs["logs"], "pipeline.log")
    checkpoint_abs = abspath(checkpoint_path)
    lg_log = joinpath(dirs["logs"], "eval_epoch_" * layout.epoch_tag * "_langevin.log")
    append_log(pipeline_log, "[langevin] start epoch=$epoch checkpoint=$checkpoint_abs")

    args = [
        "--params", effective_params["langevin"],
        "--checkpoint", checkpoint_abs,
        "--output-dir", layout.eval_dir,
        "--metrics-path", layout.metrics_txt,
    ]
    try
        run_stage_on_worker!(stage_workers[:langevin], :langevin, args, lg_log)
    catch err
        error("epoch $epoch langevin stage failed. See log: $lg_log\n$(sprint(showerror, err))")
    end

    metrics = read_keyval_metrics(layout.metrics_txt)
    avg_mode_kl = get(metrics, "avg_mode_kl_clipped", NaN)
    global_kl = get(metrics, "global_kl_from_run_langevin", NaN)
    append_log(pipeline_log, "[langevin] done epoch=$epoch avg_mode_kl=$(avg_mode_kl)")

    figB = joinpath(layout.eval_dir, "figB_stats_4x2.png")
    figC = joinpath(layout.eval_dir, "figC_dynamics_2x2.png")
    return Dict{String,Any}(
        "stage" => "langevin",
        "epoch" => epoch,
        "checkpoint_path" => checkpoint_abs,
        "metrics" => metrics,
        "avg_mode_kl_clipped" => avg_mode_kl,
        "global_kl" => global_kl,
        "figure_B" => isfile(figB) ? abspath(figB) : "",
        "figure_C" => isfile(figC) ? abspath(figC) : "",
    )
end

function run_responses_stage_for_checkpoint!(cfg::Dict{String,Any},
    effective_params::Dict{String,String},
    dirs::Dict{String,String},
    stage_workers::Dict{Symbol,Int},
    epoch::Int,
    checkpoint_path::AbstractString)
    layout = eval_layout(dirs, epoch)
    mkpath(layout.eval_dir)
    mkpath(layout.responses_root)

    pipeline_log = joinpath(dirs["logs"], "pipeline.log")
    checkpoint_abs = abspath(checkpoint_path)
    rp_log = joinpath(dirs["logs"], "eval_epoch_" * layout.epoch_tag * "_responses.log")
    append_log(pipeline_log, "[responses] start epoch=$epoch checkpoint=$checkpoint_abs")

    args = [
        "--params", effective_params["responses"],
        "--checkpoint", checkpoint_abs,
        "--output-dir", layout.responses_root,
        "--apply-correction", string(cfg["evaluation.response_apply_correction"]),
        "--response-kind", cfg["evaluation.response_kind"],
    ]
    try
        run_stage_on_worker!(stage_workers[:responses], :responses, args, rp_log)
    catch err
        error("epoch $epoch responses stage failed. See log: $rp_log\n$(sprint(showerror, err))")
    end

    response_run_dir = latest_run_dir(layout.responses_root)
    isempty(response_run_dir) && error("Response stage completed but no run_### folder was created in $(layout.responses_root)")

    response_summary = joinpath(response_run_dir, "responses_5x5_summary.toml")
    isfile(response_summary) || error("Missing response summary: $response_summary")

    summary_doc = TOML.parsefile(response_summary)
    smape_unet = get_nested_float(summary_doc, ["jacobian_distance_smape", "unet_vs_numerical"])
    smape_gaussian = get_nested_float(summary_doc, ["jacobian_distance_smape", "gaussian_vs_numerical"])
    if !isfinite(smape_unet)
        # Backward-compatible fallback for older summaries that only exported RMSE.
        smape_unet = get_nested_float(summary_doc, ["rmse", "numerics_vs_unet_corrected", "overall"])
    end
    if !isfinite(smape_unet)
        smape_unet = get_nested_float(summary_doc, ["rmse", "numerics_vs_unet_raw", "overall"])
    end
    if !isfinite(smape_gaussian)
        # Backward-compatible fallback for older summaries that only exported RMSE.
        smape_gaussian = get_nested_float(summary_doc, ["rmse", "numerics_vs_gaussian", "overall"])
    end

    figD = ""
    figD_src = joinpath(response_run_dir, "responses_5x5_selected_methods.png")
    if isfile(figD_src)
        figD = joinpath(layout.eval_dir, "figD_responses_5x5.png")
        cp(figD_src, figD; force=true)
    end

    append_log(pipeline_log, "[responses] done epoch=$epoch smape_unet=$(smape_unet) smape_gaussian=$(smape_gaussian)")
    return Dict{String,Any}(
        "stage" => "responses",
        "epoch" => epoch,
        "checkpoint_path" => checkpoint_abs,
        "smape_unet_vs_numerical" => smape_unet,
        "smape_gaussian_vs_numerical" => smape_gaussian,
        "figure_D" => isempty(figD) ? "" : abspath(figD),
        "response_run_dir" => response_run_dir,
        "response_summary" => response_summary,
    )
end

function compose_eval_row!(cfg::Dict{String,Any},
    dirs::Dict{String,String},
    epoch::Int,
    checkpoint_path::AbstractString,
    stage_payloads::Dict{Symbol,Dict{String,Any}})
    layout = eval_layout(dirs, epoch)
    metrics = haskey(stage_payloads, :langevin) ? stage_payloads[:langevin]["metrics"] : Dict{String,Float64}()
    avg_mode_kl = haskey(stage_payloads, :langevin) ? stage_payloads[:langevin]["avg_mode_kl_clipped"] : NaN
    global_kl = haskey(stage_payloads, :langevin) ? stage_payloads[:langevin]["global_kl"] : NaN
    figB = haskey(stage_payloads, :langevin) ? String(stage_payloads[:langevin]["figure_B"]) : ""
    figC = haskey(stage_payloads, :langevin) ? String(stage_payloads[:langevin]["figure_C"]) : ""
    figD = haskey(stage_payloads, :responses) ? String(stage_payloads[:responses]["figure_D"]) : ""
    response_run_dir = haskey(stage_payloads, :responses) ? String(stage_payloads[:responses]["response_run_dir"]) : ""
    response_summary = haskey(stage_payloads, :responses) ? String(stage_payloads[:responses]["response_summary"]) : ""
    smape_unet = haskey(stage_payloads, :responses) ? stage_payloads[:responses]["smape_unet_vs_numerical"] : NaN
    smape_gaussian = haskey(stage_payloads, :responses) ? stage_payloads[:responses]["smape_gaussian_vs_numerical"] : NaN

    metrics_toml = layout.metrics_toml
    metrics_doc = Dict{String,Any}(
        "epoch" => epoch,
        "checkpoint_path" => abspath(checkpoint_path),
        "langevin_metrics" => Dict{String,Any}(k => v for (k, v) in metrics),
        "avg_mode_kl_clipped" => avg_mode_kl,
        "global_kl" => global_kl,
        "smape_unet_vs_numerical" => smape_unet,
        "smape_gaussian_vs_numerical" => smape_gaussian,
        "figures" => Dict(
            "figB" => isfile(figB) ? abspath(figB) : "",
            "figC" => isfile(figC) ? abspath(figC) : "",
            "figD" => isempty(figD) ? "" : abspath(figD),
        ),
        "responses" => Dict(
            "run_dir" => response_run_dir,
            "summary" => response_summary,
        ),
    )
    write_toml(metrics_toml, metrics_doc)

    pipeline_log = joinpath(dirs["logs"], "pipeline.log")
    append_log(pipeline_log, "[eval] done epoch=$epoch avg_mode_kl=$(avg_mode_kl) smape_unet=$(smape_unet) smape_gaussian=$(smape_gaussian)")

    return Dict{String,Any}(
        "epoch" => epoch,
        "tag" => layout.eval_tag,
        "checkpoint_path" => abspath(checkpoint_path),
        "fig_dir" => layout.eval_dir,
        "figure_B" => figB,
        "figure_C" => figC,
        "figure_D" => figD,
        "metrics_toml" => abspath(metrics_toml),
        "avg_mode_kl_clipped" => avg_mode_kl,
        "global_kl" => global_kl,
        "smape_unet_vs_numerical" => smape_unet,
        "smape_gaussian_vs_numerical" => smape_gaussian,
        "response_run_dir" => response_run_dir,
        "response_summary" => response_summary,
    )
end

function maybe_finalize_epoch_row!(cfg::Dict{String,Any},
    dirs::Dict{String,String},
    required_stages::Vector{Symbol},
    partial_by_epoch::Dict{Int,Dict{Symbol,Dict{String,Any}}},
    rows_by_epoch::Dict{Int,Dict{String,Any}},
    lock_state::ReentrantLock,
    epoch::Int,
    checkpoint_path::AbstractString,
    stage::Symbol,
    payload::Dict{String,Any})
    ready_payloads = nothing
    lock(lock_state) do
        slot = get!(partial_by_epoch, epoch, Dict{Symbol,Dict{String,Any}}())
        slot[stage] = payload
        if all(s -> haskey(slot, s), required_stages)
            ready_payloads = Dict{Symbol,Dict{String,Any}}(k => v for (k, v) in slot)
            delete!(partial_by_epoch, epoch)
        end
    end

    if ready_payloads !== nothing
        row = compose_eval_row!(cfg, dirs, epoch, checkpoint_path, ready_payloads)
        lock(lock_state) do
            rows_by_epoch[epoch] = row
        end
    end
    return nothing
end

function stage_worker_loop(stage::Symbol,
    pending::Channel{Tuple{Int,String}},
    cfg::Dict{String,Any},
    effective_params::Dict{String,String},
    dirs::Dict{String,String},
    stage_workers::Dict{Symbol,Int},
    required_stages::Vector{Symbol},
    partial_by_epoch::Dict{Int,Dict{Symbol,Dict{String,Any}}},
    rows_by_epoch::Dict{Int,Dict{String,Any}},
    lock_state::ReentrantLock)
    for (epoch, checkpoint_path) in pending
        payload = if stage == :langevin
            run_langevin_stage_for_checkpoint!(cfg, effective_params, dirs, stage_workers, epoch, checkpoint_path)
        elseif stage == :responses
            run_responses_stage_for_checkpoint!(cfg, effective_params, dirs, stage_workers, epoch, checkpoint_path)
        else
            error("Unknown stage: $stage")
        end

        maybe_finalize_epoch_row!(
            cfg,
            dirs,
            required_stages,
            partial_by_epoch,
            rows_by_epoch,
            lock_state,
            epoch,
            checkpoint_path,
            stage,
            payload,
        )
    end
    return nothing
end

function save_figA(path::AbstractString,
    training_metrics_csv::AbstractString,
    eval_rows::Vector{Dict{String,Any}};
    target_kl::Float64,
    dpi::Int=DEFAULT_DPI)
    default(fontfamily="Computer Modern", dpi=dpi, legendfontsize=9, guidefontsize=10, tickfontsize=9, titlefontsize=11)

    epochs, losses = read_epoch_losses(training_metrics_csv)
    eval_epochs = [Int(r["epoch"]) for r in eval_rows]
    eval_kls = [Float64(r["avg_mode_kl_clipped"]) for r in eval_rows]
    eval_smape = [Float64(get(r, "smape_unet_vs_numerical", NaN)) for r in eval_rows]
    eval_smape_gaussian = [Float64(get(r, "smape_gaussian_vs_numerical", NaN)) for r in eval_rows]
    gaussian_ref_vals = [v for v in eval_smape_gaussian if isfinite(v)]
    gaussian_ref = isempty(gaussian_ref_vals) ? NaN : mean(gaussian_ref_vals)

    p1 = plot(epochs, losses;
        marker=:circle,
        markersize=3,
        color=:dodgerblue3,
        linewidth=2,
        label="train loss",
        xlabel="Epoch",
        ylabel="Loss",
        title="Loss vs epoch",
        left_margin=12Plots.mm,
        right_margin=5Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=8Plots.mm)

    p2 = plot(eval_epochs, eval_kls;
        marker=:diamond,
        markersize=4,
        color=:firebrick3,
        linewidth=2,
        label="avg mode KL",
        xlabel="Checkpoint epoch",
        ylabel="KL",
        title="KL(obs vs Langevin) vs checkpoint",
        left_margin=12Plots.mm,
        right_margin=5Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=8Plots.mm)
    hline!(p2, [target_kl]; color=:black, linestyle=:dash, linewidth=1.5, label=@sprintf("target %.3f", target_kl))

    p3 = plot(eval_epochs, eval_smape;
        marker=:utriangle,
        markersize=4,
        color=:darkorange3,
        linewidth=2,
        label="sMAPE (UNet vs numerical)",
        xlabel="Checkpoint epoch",
        ylabel="sMAPE",
        title="Asymptotic Jacobian sMAPE vs checkpoint",
        left_margin=12Plots.mm,
        right_margin=5Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=8Plots.mm)
    if isfinite(gaussian_ref)
        hline!(p3, [gaussian_ref]; color=:gray25, linestyle=:dash, linewidth=1.7, label="sMAPE (Gaussian vs numerical)")
    end

    fig = plot(p1, p2, p3;
        layout=(3, 1),
        size=(1320, 1500),
        left_margin=6Plots.mm,
        right_margin=6Plots.mm,
        top_margin=6Plots.mm,
        bottom_margin=6Plots.mm)

    mkpath(dirname(path))
    savefig(fig, path)
    return path
end

function write_run_summary(run_dir::AbstractString,
    params_used::Dict{String,String},
    training::Dict{String,Any},
    eval_rows::Vector{Dict{String,Any}},
    figA_path::AbstractString,
    target_kl::Float64)
    best_idx = 0
    best_kl = Inf
    for (i, row) in enumerate(eval_rows)
        kl = Float64(row["avg_mode_kl_clipped"])
        if isfinite(kl) && kl < best_kl
            best_kl = kl
            best_idx = i
        end
    end

    best = best_idx > 0 ? eval_rows[best_idx] : Dict{String,Any}()
    summary = Dict{String,Any}(
        "updated_at" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "run_dir" => abspath(run_dir),
        "parameters_used" => Dict{String,Any}(k => abspath(v) for (k, v) in params_used),
        "training" => Dict(
            "model_path" => abspath(training["model_path"]),
            "state_path" => abspath(training["state_path"]),
            "checkpoint_dir" => abspath(training["checkpoint_dir"]),
            "training_metrics_csv" => abspath(training["training_metrics_csv"]),
            "figure_A" => abspath(figA_path),
            "epochs" => Int(training["epochs"]),
            "checkpoint_every" => Int(training["checkpoint_every"]),
        ),
        "evaluation" => Dict(
            "target_avg_mode_kl" => target_kl,
            "num_evaluations" => length(eval_rows),
            "epochs" => [Int(r["epoch"]) for r in eval_rows],
            "avg_mode_kl_clipped" => [Float64(r["avg_mode_kl_clipped"]) for r in eval_rows],
            "smape_unet_vs_numerical" => [Float64(get(r, "smape_unet_vs_numerical", NaN)) for r in eval_rows],
            "smape_gaussian_vs_numerical" => [Float64(get(r, "smape_gaussian_vs_numerical", NaN)) for r in eval_rows],
            "details" => [Dict{String,Any}(k => v for (k, v) in r) for r in eval_rows],
            "best_epoch" => best_idx > 0 ? Int(best["epoch"]) : -1,
            "best_avg_mode_kl_clipped" => best_idx > 0 ? Float64(best["avg_mode_kl_clipped"]) : NaN,
            "best_metrics_toml" => best_idx > 0 ? abspath(String(best["metrics_toml"])) : "",
        ),
    )

    out_path = joinpath(run_dir, "metrics", "run_summary.toml")
    write_toml(out_path, summary)
    return out_path
end

function main(args=ARGS)
    pipeline_params_path = parse_args(args)
    cfg, _ = load_pipeline_config(pipeline_params_path)

    run_info = next_run_dir(cfg["run.runs_root"]; pad=cfg["run.run_id_padding"])
    dirs = create_run_scaffold(run_info.run_dir)
    pipeline_log = joinpath(dirs["logs"], "pipeline.log")

    append_log(pipeline_log, "pipeline start")
    append_log(pipeline_log, "run_dir=$(abspath(run_info.run_dir))")
    append_log(pipeline_log, "pipeline_params=$(pipeline_params_path)")

    effective_params = build_effective_params(cfg, dirs, pipeline_params_path)
    append_log(pipeline_log, "effective_train_params=$(effective_params["train"])")
    append_log(pipeline_log, "effective_langevin_params=$(effective_params["langevin"])")
    append_log(pipeline_log, "effective_responses_params=$(effective_params["responses"])")
    append_log(pipeline_log, "effective_data_params=$(effective_params["data"])")

    data_cfg, _ = load_data_config(effective_params["data"])
    train_role, gfdt_role = read_dataset_roles(effective_params["train"], effective_params["responses"])
    required_roles = String[train_role, gfdt_role]
    if data_cfg["closure.auto_fit"] && !("two_scale_observed" in required_roles)
        push!(required_roles, "two_scale_observed")
    end
    unique!(required_roles)
    append_log(pipeline_log, "dataset prewarm roles=$(join(required_roles, ","))")
    ds_info = ensure_arnold_datasets!(data_cfg; roles=required_roles)
    for role in sort!(collect(keys(ds_info)))
        append_log(pipeline_log, "dataset role=$(role) generated=$(ds_info[role]["generated"]) path=$(ds_info[role]["path"]) key=$(ds_info[role]["key"])")
    end

    train_meta = read_train_meta(effective_params["train"])
    epochs = train_meta.epochs
    checkpoint_every = train_meta.checkpoint_every

    checkpoint_dir = joinpath(dirs["model"], "checkpoints")
    final_model_path = joinpath(dirs["model"], "score_model.bson")
    training_metrics_csv = joinpath(dirs["figures_training"], "training_metrics.csv")

    required_stages = required_eval_stages(cfg)
    stage_workers = Dict{Symbol,Int}()

    pending_langevin = Channel{Tuple{Int,String}}(1024)
    pending_responses = Channel{Tuple{Int,String}}(1024)

    rows_by_epoch = Dict{Int,Dict{String,Any}}()
    partial_by_epoch = Dict{Int,Dict{Symbol,Dict{String,Any}}}()
    lock_state = ReentrantLock()
    monitor_task = nothing
    worker_tasks = Task[]
    train_proc = nothing
    train_io = nothing
    train_log = joinpath(dirs["logs"], "train.log")

    try
        if cfg["evaluation.run_langevin"]
            stage_workers[:langevin] = start_stage_worker(:langevin, joinpath(@__DIR__, "run_langevin.jl"))
            append_log(pipeline_log, "persistent worker started stage=langevin pid=$(stage_workers[:langevin])")
        end
        if cfg["evaluation.run_responses"]
            stage_workers[:responses] = start_stage_worker(
                :responses,
                joinpath(@__DIR__, "compute_responses.jl");
                threads=cfg["resources.responses_threads"],
            )
            append_log(pipeline_log, "persistent worker started stage=responses pid=$(stage_workers[:responses]) threads=$(cfg["resources.responses_threads"])")
        end

        train_cmd = `julia --project=. scripts/arnold/train_unet.jl --params $(effective_params["train"]) --run-dir $(abspath(run_info.run_dir)) --pipeline-mode true`
        train_proc, train_io = launch_logged(train_cmd, train_log)
        append_log(pipeline_log, "training launched")

        seen = Set{Int}()
        monitor_task = @async begin
            poll_s = cfg["resources.checkpoint_poll_seconds"]
            eval_each = cfg["evaluation.evaluate_every_checkpoint"]

            while true
                ckpts = list_checkpoints(checkpoint_dir)
                for (ep, path) in ckpts
                    if !(ep in seen)
                        push!(seen, ep)
                        if eval_each
                            haskey(stage_workers, :langevin) && put!(pending_langevin, (ep, path))
                            haskey(stage_workers, :responses) && put!(pending_responses, (ep, path))
                            append_log(pipeline_log, "queued checkpoint epoch=$ep path=$(abspath(path))")
                        end
                    end
                end

                Base.process_exited(train_proc) && break
                sleep(poll_s)
            end

            ckpts = list_checkpoints(checkpoint_dir)
            for (ep, path) in ckpts
                if !(ep in seen)
                    push!(seen, ep)
                    if cfg["evaluation.evaluate_every_checkpoint"]
                        haskey(stage_workers, :langevin) && put!(pending_langevin, (ep, path))
                        haskey(stage_workers, :responses) && put!(pending_responses, (ep, path))
                        append_log(pipeline_log, "queued checkpoint epoch=$ep path=$(abspath(path))")
                    end
                end
            end

            isopen(pending_langevin) && close(pending_langevin)
            isopen(pending_responses) && close(pending_responses)
            append_log(pipeline_log, "checkpoint monitor stopped")
        end

        if haskey(stage_workers, :langevin)
            push!(
                worker_tasks,
                @async stage_worker_loop(
                    :langevin,
                    pending_langevin,
                    cfg,
                    effective_params,
                    dirs,
                    stage_workers,
                    required_stages,
                    partial_by_epoch,
                    rows_by_epoch,
                    lock_state,
                )
            )
        end

        if haskey(stage_workers, :responses)
            push!(
                worker_tasks,
                @async stage_worker_loop(
                    :responses,
                    pending_responses,
                    cfg,
                    effective_params,
                    dirs,
                    stage_workers,
                    required_stages,
                    partial_by_epoch,
                    rows_by_epoch,
                    lock_state,
                )
            )
        end

        wait(monitor_task)
        wait(train_proc)
        close(train_io)
        train_io = nothing
        success(train_proc) || error("Training stage failed. See log: $train_log")
        append_log(pipeline_log, "training completed")

        for task in worker_tasks
            wait(task)
        end

        eval_final = cfg["evaluation.evaluate_final_model"]
        if eval_final
            haskey(rows_by_epoch, epochs) || begin
                isfile(final_model_path) || error("Final model not found at $final_model_path")
                append_log(pipeline_log, "running final evaluation from final model epoch=$epochs")

                stage_payloads = Dict{Symbol,Dict{String,Any}}()
                if cfg["evaluation.run_langevin"]
                    stage_payloads[:langevin] = run_langevin_stage_for_checkpoint!(cfg, effective_params, dirs, stage_workers, epochs, final_model_path)
                end
                if cfg["evaluation.run_responses"]
                    stage_payloads[:responses] = run_responses_stage_for_checkpoint!(cfg, effective_params, dirs, stage_workers, epochs, final_model_path)
                end
                if !isempty(stage_payloads)
                    rows_by_epoch[epochs] = compose_eval_row!(cfg, dirs, epochs, final_model_path, stage_payloads)
                end
            end
        end

        eval_rows = collect(values(rows_by_epoch))
        sort!(eval_rows; by=r -> Int(r["epoch"]))

        figA_path = save_figA(
            joinpath(dirs["figures_training"], "figA_training_3panels.png"),
            training_metrics_csv,
            eval_rows;
            target_kl=cfg["targets.target_avg_mode_kl"],
            dpi=DEFAULT_DPI,
        )

        summary_path = write_run_summary(
            run_info.run_dir,
            Dict(
                "pipeline" => joinpath(dirs["params"], "parameters_pipeline_used.toml"),
                "train" => effective_params["train"],
                "langevin" => effective_params["langevin"],
                "responses" => effective_params["responses"],
            ),
            Dict{String,Any}(
                "model_path" => final_model_path,
                "state_path" => joinpath(dirs["model"], "training_state_latest.bson"),
                "checkpoint_dir" => checkpoint_dir,
                "training_metrics_csv" => training_metrics_csv,
                "epochs" => epochs,
                "checkpoint_every" => checkpoint_every,
            ),
            eval_rows,
            figA_path,
            cfg["targets.target_avg_mode_kl"],
        )

        if !isempty(eval_rows)
            finite_kls = [Float64(r["avg_mode_kl_clipped"]) for r in eval_rows if isfinite(Float64(r["avg_mode_kl_clipped"]))]
            if !isempty(finite_kls)
                best_kl = minimum(finite_kls)
                if best_kl > cfg["targets.target_avg_mode_kl"]
                    append_log(pipeline_log, "warning best_avg_mode_kl=$best_kl above target=$(cfg["targets.target_avg_mode_kl"])")
                end
            end
        end

        append_log(pipeline_log, "summary=$(abspath(summary_path))")
        append_log(pipeline_log, "pipeline done")

        println("run_dir=$(abspath(run_info.run_dir))")
        println("summary=$(abspath(summary_path))")
    finally
        isopen(pending_langevin) && close(pending_langevin)
        isopen(pending_responses) && close(pending_responses)

        for task in worker_tasks
            if !istaskdone(task)
                try
                    wait(task)
                catch
                end
            end
        end

        if train_proc !== nothing && Base.process_running(train_proc)
            try
                kill(train_proc)
            catch
            end
        end
        if train_io !== nothing
            try
                close(train_io)
            catch
            end
        end

        stop_stage_workers!(stage_workers, pipeline_log)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
