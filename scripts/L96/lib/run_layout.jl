module L96RunLayout

using Dates
using TOML

const LATEST_RUN_FILE = "LATEST_RUN.txt"
const RUN_MANIFEST_FILE = "run_manifest.toml"
const RUN_SUMMARY_FILE = "RUN_SUMMARY.md"
const RUN_INDEX_FILE = "RUN_INDEX.md"
const RUN_REGISTRY_FILE = "RUN_REGISTRY.csv"

const STAGE_PRIORITY = Dict(
    "generate_data" => 1,
    "train_unet" => 2,
    "sample_and_compare" => 3,
)

timestamp_id() = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

normalize_path(path::AbstractString) = abspath(expanduser(path))

default_runs_root(script_dir::AbstractString) = normalize_path(joinpath(script_dir, "runs"))

latest_run_file(script_dir::AbstractString) = joinpath(script_dir, LATEST_RUN_FILE)

function read_latest_run(script_dir::AbstractString)
    file = latest_run_file(script_dir)
    if isfile(file)
        value = strip(read(file, String))
        !isempty(value) && return normalize_path(value)
    end
    return nothing
end

function write_latest_run!(script_dir::AbstractString, run_dir::AbstractString)
    file = latest_run_file(script_dir)
    mkpath(dirname(file))
    open(file, "w") do io
        println(io, normalize_path(run_dir))
    end
    return file
end

run_manifest_path(run_dir::AbstractString) = joinpath(run_dir, RUN_MANIFEST_FILE)
run_summary_path(run_dir::AbstractString) = joinpath(run_dir, RUN_SUMMARY_FILE)

function _safe_set_symlink!(link_path::AbstractString, target_path::AbstractString)
    if ispath(link_path) && !islink(link_path)
        return false
    end
    if ispath(link_path) || islink(link_path)
        rm(link_path; force=true, recursive=true)
    end
    symlink(normalize_path(target_path), link_path)
    return true
end

function update_compat_links!(script_dir::AbstractString;
                              data_path::Union{Nothing,AbstractString}=nothing,
                              model_path::Union{Nothing,AbstractString}=nothing,
                              checkpoints_dir::Union{Nothing,AbstractString}=nothing,
                              figures_dir::Union{Nothing,AbstractString}=nothing)
    status = Dict{String,Any}()
    if data_path !== nothing
        status["l96_timeseries.hdf5"] = _safe_set_symlink!(joinpath(script_dir, "l96_timeseries.hdf5"), data_path)
    end
    if model_path !== nothing
        status["trained_model.bson"] = _safe_set_symlink!(joinpath(script_dir, "trained_model.bson"), model_path)
    end
    if checkpoints_dir !== nothing
        status["checkpoints"] = _safe_set_symlink!(joinpath(script_dir, "checkpoints"), checkpoints_dir)
    end
    if figures_dir !== nothing
        status["figures"] = _safe_set_symlink!(joinpath(script_dir, "figures"), figures_dir)
    end
    return status
end

function infer_run_dir_from_data_path(path::AbstractString)
    data_path = normalize_path(path)
    parent = dirname(data_path)
    if basename(parent) == "data"
        return dirname(parent)
    end
    return nothing
end

function pick_generation_run_dir(script_dir::AbstractString, j::Integer)
    if haskey(ENV, "L96_RUN_DIR") && !isempty(strip(ENV["L96_RUN_DIR"]))
        return normalize_path(ENV["L96_RUN_DIR"])
    end
    runs_root = normalize_path(get(ENV, "L96_RUNS_ROOT", default_runs_root(script_dir)))
    run_group = get(ENV, "L96_RUN_GROUP", "J$(j)")
    run_id = get(ENV, "L96_RUN_ID", "run_" * timestamp_id())
    return normalize_path(joinpath(runs_root, run_group, run_id))
end

function pick_stage_run_dir(script_dir::AbstractString)
    if haskey(ENV, "L96_RUN_DIR") && !isempty(strip(ENV["L96_RUN_DIR"]))
        return normalize_path(ENV["L96_RUN_DIR"])
    end

    if haskey(ENV, "L96_RUN_GROUP") || haskey(ENV, "L96_RUN_ID")
        runs_root = normalize_path(get(ENV, "L96_RUNS_ROOT", default_runs_root(script_dir)))
        run_group = get(ENV, "L96_RUN_GROUP", "Junknown")
        run_id = get(ENV, "L96_RUN_ID", "run_" * timestamp_id())
        return normalize_path(joinpath(runs_root, run_group, run_id))
    end

    latest = read_latest_run(script_dir)
    latest !== nothing && return latest

    runs_root = normalize_path(get(ENV, "L96_RUNS_ROOT", default_runs_root(script_dir)))
    run_group = get(ENV, "L96_RUN_GROUP", "Junknown")
    run_id = get(ENV, "L96_RUN_ID", "run_" * timestamp_id())
    return normalize_path(joinpath(runs_root, run_group, run_id))
end

default_data_path(run_dir::AbstractString) = joinpath(run_dir, "data", "l96_timeseries.hdf5")

function default_train_paths(run_dir::AbstractString)
    train_root = joinpath(run_dir, "train")
    return (
        model_path=joinpath(train_root, "model.bson"),
        diagnostics_dir=joinpath(train_root, "figures"),
        checkpoints_dir=joinpath(train_root, "checkpoints"),
    )
end

function default_eval_root(run_dir::AbstractString)
    tag = get(ENV, "L96_EVAL_TAG", "eval_" * timestamp_id())
    return joinpath(run_dir, "eval", tag)
end

function write_toml_file(path::AbstractString, config::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, config)
    end
    return path
end

function default_config_path(run_dir::AbstractString, stage::AbstractString)
    return joinpath(run_dir, "configs", "$(stage).toml")
end

function _to_string_dict(d::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in d
        key = string(k)
        if v isa AbstractDict
            out[key] = _to_string_dict(v)
        elseif v isa Tuple
            out[key] = collect(v)
        elseif v isa AbstractVector
            out[key] = [x isa Tuple ? collect(x) : x for x in v]
        else
            out[key] = v
        end
    end
    return out
end

function _merge_dict!(dst::Dict{String,Any}, src::Dict{String,Any})
    for (k, v) in src
        if v isa Dict{String,Any} && get(dst, k, nothing) isa Dict{String,Any}
            _merge_dict!(dst[k], v)
        else
            dst[k] = v
        end
    end
    return dst
end

function _sorted_stages(stages::Vector{String})
    return sort(unique(stages); by=s -> get(STAGE_PRIORITY, s, 10_000))
end

function update_run_manifest!(run_dir::AbstractString;
                              stage::Union{Nothing,AbstractString}=nothing,
                              parameters::AbstractDict=Dict{String,Any}(),
                              paths::AbstractDict=Dict{String,Any}(),
                              artifacts::AbstractDict=Dict{String,Any}(),
                              metrics::AbstractDict=Dict{String,Any}(),
                              notes::AbstractDict=Dict{String,Any}())
    manifest_file = run_manifest_path(run_dir)
    manifest = isfile(manifest_file) ? _to_string_dict(TOML.parsefile(manifest_file)) : Dict{String,Any}()
    run_dir_abs = normalize_path(run_dir)

    manifest["run_dir"] = run_dir_abs
    manifest["run_group"] = basename(dirname(run_dir_abs))
    manifest["run_id"] = basename(run_dir_abs)
    manifest["updated_at"] = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")

    stages = String[]
    raw_stages = get(manifest, "stages", Any[])
    if raw_stages isa AbstractVector
        for s in raw_stages
            push!(stages, string(s))
        end
    end
    if stage !== nothing
        push!(stages, String(stage))
    end
    manifest["stages"] = _sorted_stages(stages)

    for (section_name, section_data) in (
        "parameters" => parameters,
        "paths" => paths,
        "artifacts" => artifacts,
        "metrics" => metrics,
        "notes" => notes,
    )
        data = _to_string_dict(section_data)
        isempty(data) && continue
        section_existing = get(manifest, section_name, Dict{String,Any}())
        if section_existing isa AbstractDict
            section_existing = _to_string_dict(section_existing)
        else
            section_existing = Dict{String,Any}()
        end
        merged = _merge_dict!(section_existing, data)
        manifest[section_name] = merged
    end

    write_toml_file(manifest_file, manifest)
    return manifest_file
end

function _fmt_value(v)
    if v isa AbstractFloat
        return string(round(v; digits=6))
    elseif v isa AbstractVector
        return join(string.(v), ", ")
    else
        return string(v)
    end
end

function _as_section_dict(v)
    if v isa AbstractDict
        return _to_string_dict(v)
    elseif v isa AbstractVector
        return Dict{String,Any}("items" => [string(x) for x in v])
    elseif v === nothing
        return Dict{String,Any}()
    else
        return Dict{String,Any}("value" => v)
    end
end

function _write_kv_section(io, title::AbstractString, data::Dict{String,Any}; run_dir::AbstractString)
    isempty(data) && return
    println(io, "## ", title)
    for key in sort(collect(keys(data)))
        value = data[key]
        if value isa Dict{String,Any}
            println(io, "- `", key, "`:")
            for subkey in sort(collect(keys(value)))
                subval = value[subkey]
                if subval isa AbstractString && isabspath(subval)
                    rel = relpath(subval, run_dir)
                    println(io, "  - `", subkey, "`: `", rel, "`")
                else
                    println(io, "  - `", subkey, "`: `", _fmt_value(subval), "`")
                end
            end
        elseif value isa AbstractString && isabspath(value)
            rel = relpath(value, run_dir)
            println(io, "- `", key, "`: `", rel, "`")
        else
            println(io, "- `", key, "`: `", _fmt_value(value), "`")
        end
    end
    println(io)
    return
end

function write_run_summary!(run_dir::AbstractString)
    manifest_file = run_manifest_path(run_dir)
    isfile(manifest_file) || return nothing
    manifest = _to_string_dict(TOML.parsefile(manifest_file))
    summary_file = run_summary_path(run_dir)

    open(summary_file, "w") do io
        println(io, "# L96 Run Summary")
        println(io)
        println(io, "- `run_group`: `", get(manifest, "run_group", "unknown"), "`")
        println(io, "- `run_id`: `", get(manifest, "run_id", basename(run_dir)), "`")
        println(io, "- `updated_at`: `", get(manifest, "updated_at", "unknown"), "`")
        stages = get(manifest, "stages", String[])
        println(io, "- `stages`: `", join(string.(stages), " -> "), "`")
        println(io)

        _write_kv_section(io, "Paths", _as_section_dict(get(manifest, "paths", Dict{String,Any}())); run_dir=run_dir)
        _write_kv_section(io, "Parameters", _as_section_dict(get(manifest, "parameters", Dict{String,Any}())); run_dir=run_dir)
        _write_kv_section(io, "Metrics", _as_section_dict(get(manifest, "metrics", Dict{String,Any}())); run_dir=run_dir)
        _write_kv_section(io, "Artifacts", _as_section_dict(get(manifest, "artifacts", Dict{String,Any}())); run_dir=run_dir)
        _write_kv_section(io, "Notes", _as_section_dict(get(manifest, "notes", Dict{String,Any}())); run_dir=run_dir)
    end
    return summary_file
end

function list_run_dirs(runs_root::AbstractString)
    dirs = String[]
    for group in sort(readdir(runs_root))
        group_dir = joinpath(runs_root, group)
        isdir(group_dir) || continue
        startswith(group, "J") || continue
        for run_id in sort(readdir(group_dir))
            run_dir = joinpath(group_dir, run_id)
            islink(run_dir) && continue
            isdir(run_dir) || continue
            push!(dirs, run_dir)
        end
    end
    return dirs
end

function _bool_str(x::Bool)
    return x ? "1" : "0"
end

function _extract_j_from_group(group::AbstractString)
    m = match(r"^J(\d+)$", group)
    m === nothing && return -1
    return parse(Int, m.captures[1])
end

function refresh_runs_index!(runs_root::AbstractString)
    ensure_runs_readme!(runs_root)
    runs = list_run_dirs(runs_root)
    index_path = joinpath(runs_root, RUN_INDEX_FILE)
    registry_path = joinpath(runs_root, RUN_REGISTRY_FILE)

    rows = Dict{String,Any}[]
    for run_dir in runs
        manifest_file = run_manifest_path(run_dir)
        manifest = isfile(manifest_file) ? _to_string_dict(TOML.parsefile(manifest_file)) : Dict{String,Any}()
        paths = get(manifest, "paths", Dict{String,Any}())
        metrics = get(manifest, "metrics", Dict{String,Any}())
        parameters = get(manifest, "parameters", Dict{String,Any}())

        eval_root = get(paths, "eval_root", joinpath(run_dir, "eval"))
        eval_count = isdir(eval_root) ? length(filter(n -> isdir(joinpath(eval_root, n)), readdir(eval_root))) : 0

        push!(rows, Dict{String,Any}(
            "run_group" => get(manifest, "run_group", basename(dirname(run_dir))),
            "run_id" => get(manifest, "run_id", basename(run_dir)),
            "updated_at" => get(manifest, "updated_at", "unknown"),
            "stages" => join(string.(get(manifest, "stages", String[])), ";"),
            "run_dir" => run_dir,
            "has_data" => ispath(joinpath(run_dir, "data", "l96_timeseries.hdf5")) || isdir(joinpath(run_dir, "data")),
            "has_model" => ispath(joinpath(run_dir, "train", "model.bson")),
            "eval_count" => eval_count,
            "avg_mode_kl_clipped" => get(metrics, "avg_mode_kl_clipped", NaN),
            "global_kl" => get(metrics, "global_kl_from_run_langevin", NaN),
            "train_sigma" => get(parameters, "train_noise_sigma", get(parameters, "sigma", NaN)),
            "process_noise_sigma" => get(parameters, "process_noise_sigma", NaN),
            "normalization_mode" => get(parameters, "normalization_mode", ""),
            "epochs" => get(parameters, "epochs", NaN),
            "learning_rate" => get(parameters, "learning_rate", get(parameters, "lr", NaN)),
            "base_channels" => get(parameters, "base_channels", NaN),
            "channel_multipliers" => get(parameters, "channel_multipliers", ""),
            "J" => get(parameters, "J", _extract_j_from_group(get(manifest, "run_group", "J0"))),
        ))
    end

    open(index_path, "w") do io
        println(io, "# L96 Runs Index")
        println(io)
        println(io, "| Group | Run | Updated | Stages | Data | Model | Evals | Avg KL (clip) | Train sigma | Process noise |")
        println(io, "|---|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows
            kl = row["avg_mode_kl_clipped"]
            ts = row["train_sigma"]
            pn = row["process_noise_sigma"]
            kl_s = isnan(kl) ? "-" : string(round(kl; digits=5))
            ts_s = isnan(ts) ? "-" : string(round(ts; digits=5))
            pn_s = isnan(pn) ? "-" : string(round(pn; digits=5))
            println(io,
                    "| ", row["run_group"],
                    " | `", row["run_id"], "`",
                    " | ", row["updated_at"],
                    " | ", row["stages"],
                    " | ", _bool_str(row["has_data"]),
                    " | ", _bool_str(row["has_model"]),
                    " | ", row["eval_count"],
                    " | ", kl_s,
                    " | ", ts_s,
                    " | ", pn_s,
                    " |")
        end
    end

    open(registry_path, "w") do io
        println(io, "run_group,run_id,updated_at,stages,run_dir,has_data,has_model,eval_count,avg_mode_kl_clipped,global_kl,train_sigma,process_noise_sigma,normalization_mode,epochs,learning_rate,base_channels,channel_multipliers,J")
        for row in rows
            mult = row["channel_multipliers"]
            mult_s = mult isa AbstractVector ? join(string.(mult), "|") : string(mult)
            println(io,
                    row["run_group"], ",",
                    row["run_id"], ",",
                    row["updated_at"], ",",
                    row["stages"], ",",
                    row["run_dir"], ",",
                    _bool_str(row["has_data"]), ",",
                    _bool_str(row["has_model"]), ",",
                    row["eval_count"], ",",
                    row["avg_mode_kl_clipped"], ",",
                    row["global_kl"], ",",
                    row["train_sigma"], ",",
                    row["process_noise_sigma"], ",",
                    row["normalization_mode"], ",",
                    row["epochs"], ",",
                    row["learning_rate"], ",",
                    row["base_channels"], ",",
                    mult_s, ",",
                    row["J"])
        end
    end

    return (index=index_path, registry=registry_path)
end

function ensure_runs_readme!(runs_root::AbstractString)
    mkpath(runs_root)
    readme_path = joinpath(runs_root, "README.md")
    open(readme_path, "w") do io
        println(io, "# L96 Runs Layout")
        println(io)
        println(io, "- `J*/run_id/data`: datasets and data-generation metadata")
        println(io, "- `J*/run_id/train`: model, checkpoints, training figures, training metadata")
        println(io, "- `J*/run_id/eval`: evaluation figures, discrepancy metrics, eval metadata")
        println(io, "- `J*/run_id/configs`: full parameter snapshots used at each stage")
        println(io, "- `J*/run_id/run_manifest.toml`: merged machine-readable summary of run parameters and metrics")
        println(io, "- `J*/run_id/RUN_SUMMARY.md`: human-readable summary for quick inspection")
        println(io, "- `RUN_INDEX.md` and `RUN_REGISTRY.csv`: cross-run inventory and metrics overview")
    end
    return readme_path
end

end # module
