module L96LangevinStage

using TOML

function _append_log(log_path::AbstractString, msg::AbstractString)
    open(log_path, "a") do io
        println(io, msg)
    end
end

function _run_logged(cmd::Cmd, log_path::AbstractString)
    run(pipeline(pipeline(cmd; stderr=stdout), `tee -a $log_path`))
end

function read_keyval_metrics(path::AbstractString)
    out = Dict{String,Float64}()
    isfile(path) || return out
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        occursin("=", s) || continue
        k, v = split(s, "="; limit=2)
        try
            out[strip(k)] = parse(Float64, strip(v))
        catch
        end
    end
    return out
end

function write_metrics_toml(path::AbstractString, epoch::Int, metrics_txt_path::AbstractString, eval_config_path::AbstractString, metrics::Dict{String,Float64})
    doc = Dict{String,Any}(
        "epoch" => epoch,
        "metrics_txt_path" => metrics_txt_path,
        "eval_config_path" => eval_config_path,
        "metrics" => Dict{String,Any}(k => v for (k, v) in metrics),
    )
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return path
end

function run_eval!(params::Dict{String,Any}, dirs::Dict{String,String}, epoch::Int, model_path::AbstractString; base_dir::AbstractString=pwd())
    log_path = joinpath(dirs["logs"], "pipeline.log")
    eval_tag = "eval_epoch_" * lpad(string(epoch), 4, '0')

    data_path = abspath(joinpath(base_dir, params["data.path"]))
    fig_dir = joinpath(dirs["figures"], eval_tag)
    metrics_txt = joinpath(dirs["metrics"], "epoch_" * lpad(string(epoch), 4, '0') * "_metrics.txt")
    metrics_toml = joinpath(dirs["metrics"], "epoch_" * lpad(string(epoch), 4, '0') * "_metrics.toml")
    eval_cfg = begin
        p, io = mktemp()
        close(io)
        p
    end

    mkpath(fig_dir)

    _append_log(log_path, "[eval] start epoch=$epoch tag=$eval_tag model=$(abspath(model_path))")

    env = copy(ENV)
    env["L96_RUN_DIR"] = dirs["run"]
    env["L96_DATA_PATH"] = data_path
    env["L96_DATASET_KEY"] = params["data.dataset_key"]
    env["L96_MODEL_PATH"] = abspath(model_path)

    env["L96_EVAL_TAG"] = eval_tag
    env["L96_FIG_DIR"] = fig_dir
    env["L96_METRICS_PATH"] = metrics_txt
    env["L96_EVAL_CONFIG_PATH"] = eval_cfg
    env["L96_PIPELINE_MODE"] = "true"

    env["L96_LANGEVIN_DEVICE"] = params["langevin.device"]
    env["L96_LANGEVIN_DT"] = string(params["langevin.dt"])
    env["L96_LANGEVIN_RESOLUTION"] = string(params["langevin.resolution"])
    env["L96_LANGEVIN_STEPS"] = string(params["langevin.nsteps"])
    env["L96_LANGEVIN_BURN_IN"] = string(params["langevin.burn_in"])
    env["L96_LANGEVIN_ENSEMBLES"] = string(params["langevin.ensembles"])
    env["L96_LANGEVIN_PROGRESS"] = params["langevin.progress"] ? "true" : "false"
    env["L96_PDF_BINS"] = string(params["langevin.pdf_bins"])
    env["L96_LANGEVIN_SEED"] = string(params["run.seed"] + epoch)
    env["L96_USE_BOUNDARY"] = params["langevin.use_boundary"] ? "true" : "false"
    env["L96_BOUNDARY_MIN"] = string(params["langevin.boundary_min"])
    env["L96_BOUNDARY_MAX"] = string(params["langevin.boundary_max"])
    env["L96_FIG_DPI"] = string(params["figures.dpi"])

    cmd = setenv(`julia --project=. scripts/L96/lib/sample_and_compare.jl`, env)
    _run_logged(cmd, log_path)

    # Keep only canonical eval artifacts in pipeline mode.
    keep_eval = Set([
        abspath(joinpath(fig_dir, "figB_stats_3x3.png")),
        abspath(joinpath(fig_dir, "figC_dynamics_3x2.png")),
    ])
    for name in readdir(fig_dir)
        p = abspath(joinpath(fig_dir, name))
        if isfile(p) && !(p in keep_eval)
            rm(p; force=true)
        end
    end

    metrics = read_keyval_metrics(metrics_txt)
    write_metrics_toml(metrics_toml, epoch, "", "", metrics)
    isfile(eval_cfg) && rm(eval_cfg; force=true)
    isfile(metrics_txt) && rm(metrics_txt; force=true)

    _append_log(log_path, "[eval] done epoch=$epoch avg_mode_kl=$(get(metrics, "avg_mode_kl_clipped", NaN))")

    return Dict{String,Any}(
        "epoch" => epoch,
        "tag" => eval_tag,
        "fig_dir" => fig_dir,
        "metrics_txt" => metrics_txt,
        "metrics_toml" => metrics_toml,
        "eval_config" => "",
        "avg_mode_kl_clipped" => get(metrics, "avg_mode_kl_clipped", NaN),
        "global_kl" => get(metrics, "global_kl_from_run_langevin", NaN),
    )
end

end # module
