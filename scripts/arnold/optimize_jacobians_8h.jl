if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Dates
using Printf
using SHA
using Statistics
using TOML

const DEFAULT_BUDGET_HOURS = 8.0
const DEFAULT_PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_PIPELINE_PARAMS = joinpath(@__DIR__, "parameters_pipeline.toml")
const DEFAULT_TRAIN_PARAMS = joinpath(@__DIR__, "parameters_train.toml")
const DEFAULT_LANGEVIN_PARAMS = joinpath(@__DIR__, "parameters_langevin.toml")
const DEFAULT_RESPONSES_PARAMS = joinpath(@__DIR__, "parameters_responses.toml")
const DEFAULT_DATA_PARAMS = joinpath(@__DIR__, "parameters_data.toml")

Base.@kwdef struct CandidateSpec
    name::String
    notes::String
    data_mods::Dict{String,Any} = Dict{String,Any}()
    train_mods::Dict{String,Any} = Dict{String,Any}()
    responses_mods::Dict{String,Any} = Dict{String,Any}()
    pipeline_mods::Dict{String,Any} = Dict{String,Any}()
    epochs::Int = 120
    seed_delta::Int = 0
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))

function parse_args(args::Vector{String})
    budget_hours = DEFAULT_BUDGET_HOURS
    max_runs = 0
    campaign_root = joinpath(@__DIR__, "optimization_campaigns")
    run_tag = Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS")

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--budget-hours"
            i == length(args) && error("--budget-hours expects a value")
            budget_hours = parse(Float64, args[i + 1])
            i += 2
        elseif a == "--max-runs"
            i == length(args) && error("--max-runs expects a value")
            max_runs = parse(Int, args[i + 1])
            i += 2
        elseif a == "--campaign-root"
            i == length(args) && error("--campaign-root expects a value")
            campaign_root = abspath(args[i + 1])
            i += 2
        elseif a == "--tag"
            i == length(args) && error("--tag expects a value")
            run_tag = strip(args[i + 1])
            i += 2
        else
            error("Unknown argument: $a")
        end
    end

    budget_hours > 0 || error("--budget-hours must be > 0")
    max_runs >= 0 || error("--max-runs must be >= 0")

    return (
        budget_hours=budget_hours,
        max_runs=max_runs,
        campaign_root=campaign_root,
        run_tag=run_tag,
    )
end

function table!(doc::Dict{String,Any}, key::AbstractString)
    k = String(key)
    if !haskey(doc, k) || !(doc[k] isa AbstractDict)
        doc[k] = Dict{String,Any}()
    elseif !(doc[k] isa Dict{String,Any})
        doc[k] = Dict{String,Any}(doc[k])
    end
    return doc[k]
end

function set_dotted!(doc::Dict{String,Any}, dotted::String, value)
    parts = split(dotted, ".")
    isempty(parts) && return doc
    node = doc
    for p in parts[1:end-1]
        node = table!(node, p)
    end
    node[parts[end]] = value
    return doc
end

function apply_mods!(doc::Dict{String,Any}, mods::Dict{String,Any})
    for (k, v) in mods
        set_dotted!(doc, k, v)
    end
    return doc
end

function safe_name(s::AbstractString)
    return replace(lowercase(s), r"[^a-z0-9]+" => "_")
end

function write_toml(path::AbstractString, doc::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return path
end

function latest_run_dir(root::AbstractString)
    isdir(root) || return ""
    best_id = -1
    best_name = ""
    for name in readdir(root)
        m = match(r"^run_(\d+)$", name)
        m === nothing && continue
        rid = parse(Int, m.captures[1])
        if rid > best_id
            best_id = rid
            best_name = name
        end
    end
    best_id < 0 && return ""
    return joinpath(root, best_name)
end

function parse_run_dir_from_log(path::AbstractString)
    isfile(path) || return ""
    for line in eachline(path)
        s = strip(line)
        startswith(s, "run_dir=") || continue
        return abspath(strip(split(s, "="; limit=2)[2]))
    end
    return ""
end

function find_response_summaries(run_dir::AbstractString)
    figs_dir = joinpath(run_dir, "figures")
    isdir(figs_dir) || return String[]
    out = String[]
    for (dir, _, files) in walkdir(figs_dir)
        for f in files
            if endswith(f, "_summary.toml") && occursin("responses_", f)
                push!(out, joinpath(dir, f))
            end
        end
    end
    return out
end

function parse_epoch_from_path(path::AbstractString)
    m = match(r"eval_epoch_(\d+)", path)
    m === nothing && return -1
    return parse(Int, m.captures[1])
end

function get_nested_float(doc::Dict{String,Any}, path::Vector{String}; default::Float64=NaN)
    node = doc
    for (i, key) in enumerate(path)
        haskey(node, key) || return default
        val = node[key]
        if i == length(path)
            try
                return Float64(val)
            catch
                return default
            end
        end
        val isa AbstractDict || return default
        node = Dict{String,Any}(val)
    end
    return default
end

function read_epoch_metric(run_dir::AbstractString, epoch::Int, key::String)
    epoch < 0 && return NaN
    metric_path = joinpath(run_dir, "metrics", @sprintf("epoch_%04d_metrics.toml", epoch))
    isfile(metric_path) || return NaN
    doc = TOML.parsefile(metric_path)
    try
        return Float64(get(doc, key, NaN))
    catch
        return NaN
    end
end

function best_overall_kl(run_dir::AbstractString)
    metrics_dir = joinpath(run_dir, "metrics")
    isdir(metrics_dir) || return NaN
    vals = Float64[]
    for f in readdir(metrics_dir)
        occursin(r"^epoch_\d+_metrics\.toml$", f) || continue
        doc = TOML.parsefile(joinpath(metrics_dir, f))
        v = try
            Float64(get(doc, "avg_mode_kl_clipped", NaN))
        catch
            NaN
        end
        isfinite(v) && push!(vals, v)
    end
    isempty(vals) && return NaN
    return minimum(vals)
end

function extract_run_metrics(run_dir::AbstractString)
    summaries = find_response_summaries(run_dir)
    best_smape = Inf
    best_epoch = -1
    best_summary = ""
    for path in summaries
        doc = TOML.parsefile(path)
        v = get_nested_float(doc, ["jacobian_distance_smape", "unet_vs_numerical"])
        if !isfinite(v)
            v = get_nested_float(doc, ["rmse", "numerics_vs_unet_corrected", "overall"])
        end
        isfinite(v) || continue
        if v < best_smape
            best_smape = v
            best_epoch = parse_epoch_from_path(path)
            best_summary = path
        end
    end
    isfinite(best_smape) || (best_smape = NaN)
    return Dict{String,Any}(
        "best_smape" => best_smape,
        "best_epoch" => best_epoch,
        "best_summary" => best_summary,
        "kl_at_best_epoch" => read_epoch_metric(run_dir, best_epoch, "avg_mode_kl_clipped"),
        "best_kl_overall" => best_overall_kl(run_dir),
        "num_response_summaries" => length(summaries),
    )
end

function merge_mods(base::AbstractDict{String,<:Any}, delta::AbstractDict{String,<:Any})
    out = Dict{String,Any}(base)
    for (k, v) in delta
        out[k] = v
    end
    return out
end

function candidate_state_signature(data_doc::Dict{String,Any})
    io = IOBuffer()
    tmp = deepcopy(data_doc)
    paths = table!(tmp, "paths")
    # Remove concrete path to keep signature driven by data-generation settings.
    paths["datasets_hdf5"] = ""
    TOML.print(io, tmp)
    return bytes2hex(sha1(take!(io)))
end

function ensure_campaign_layout(campaign_dir::AbstractString)
    mkpath(campaign_dir)
    for sub in ("configs", "logs", "reports", "runs", "data")
        mkpath(joinpath(campaign_dir, sub))
    end
    return campaign_dir
end

function candidate_report_rows(results::Vector{Dict{String,Any}})
    rows = copy(results)
    sort!(rows; by=r -> begin
        v = get(r, "best_smape", NaN)
        isfinite(v) ? v : Inf
    end)
    return rows
end

function save_results_toml(path::AbstractString, results::Vector{Dict{String,Any}})
    doc = Dict{String,Any}(
        "updated_at" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "runs" => results,
    )
    write_toml(path, doc)
end

function write_report(path::AbstractString,
    campaign_meta::Dict{String,Any},
    results::Vector{Dict{String,Any}},
    best_state::Dict{Symbol,Dict{String,Any}})
    rows = candidate_report_rows(results)
    best = isempty(rows) ? Dict{String,Any}() : rows[1]
    baseline = isempty(results) ? Dict{String,Any}() : results[1]
    campaign_dir = get(campaign_meta, "campaign_dir", "")
    started_at = get(campaign_meta, "started_at", "")
    ended_at = get(campaign_meta, "ended_at", "")
    budget_hours = get(campaign_meta, "budget_hours", "")

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# 8-Hour Jacobian Optimization Report")
        println(io)
        println(io, "- campaign_dir: `", campaign_dir, "`")
        println(io, "- started_at: `", started_at, "`")
        println(io, "- ended_at: `", ended_at, "`")
        println(io, "- budget_hours: `", budget_hours, "`")
        println(io, "- runs_completed: `", length(results), "`")
        println(io)

        println(io, "## Best Configuration Found")
        if isempty(best)
            println(io, "No successful runs with usable metrics.")
        else
            println(io, "- run_name: `", get(best, "name", ""), "`")
            println(io, "- notes: `", get(best, "notes", ""), "`")
            println(io, "- best_smape_unet_vs_numerical: `", get(best, "best_smape", NaN), "`")
            println(io, "- best_epoch: `", get(best, "best_epoch", -1), "`")
            println(io, "- avg_mode_kl_at_best_epoch: `", get(best, "kl_at_best_epoch", NaN), "`")
            println(io, "- best_kl_overall: `", get(best, "best_kl_overall", NaN), "`")
            println(io, "- run_dir: `", get(best, "run_dir", ""), "`")
            println(io, "- best_summary_file: `", get(best, "best_summary", ""), "`")
        end
        println(io)

        println(io, "## Largest Improvements")
        if !isempty(best) && !isempty(baseline)
            b0 = get(baseline, "best_smape", NaN)
            b1 = get(best, "best_smape", NaN)
            if isfinite(b0) && isfinite(b1)
                rel = 100 * (b0 - b1) / max(abs(b0), 1e-12)
                println(io, "- best vs baseline sMAPE improvement: `$(round(rel; digits=2))%` (`$b0` -> `$b1`)")
            else
                println(io, "- Unable to compute relative improvement (missing finite baseline/best metric).")
            end
        end
        for r in rows[1:min(end, 5)]
            r_name = get(r, "name", "")
            r_smape = get(r, "best_smape", NaN)
            r_kl = get(r, "kl_at_best_epoch", NaN)
            r_dur = round(Float64(get(r, "duration_min", NaN)); digits=1)
            println(io, "- `", r_name, "`: sMAPE=`", r_smape, "`, KL=`", r_kl, "`, duration_min=`", r_dur, "`")
        end
        println(io)

        println(io, "## Top Runs (Ranked)")
        if isempty(rows)
            println(io, "No runs.")
        else
            for (idx, r) in enumerate(rows)
                r_name = get(r, "name", "")
                r_smape = get(r, "best_smape", NaN)
                r_kl = get(r, "kl_at_best_epoch", NaN)
                r_dur = round(Float64(get(r, "duration_min", NaN)); digits=1)
                r_run_dir = get(r, "run_dir", "")
                println(io, idx, ". `", r_name, "` | sMAPE=`", r_smape, "` | KL=`", r_kl, "` | duration_min=`", r_dur, "` | run_dir=`", r_run_dir, "`")
            end
        end
        println(io)

        println(io, "## Best State Overrides")
        for section in (:data, :train, :responses, :pipeline)
            println(io, "- ", section, ":")
            mods = get(best_state, section, Dict{String,Any}())
            if isempty(mods)
                println(io, "  - (none)")
            else
                for (k, v) in sort!(collect(mods); by=x -> x[1])
                    println(io, "  - ", k, " = ", repr(v))
                end
            end
        end
    end
end

function run_candidate!(;
    candidate::CandidateSpec,
    run_index::Int,
    campaign_dir::AbstractString,
    project_root::AbstractString,
    base_docs::Dict{Symbol,Dict{String,Any}},
    base_train_seed::Int)
    run_tag = @sprintf("%02d_%s", run_index, safe_name(candidate.name))
    cfg_dir = joinpath(campaign_dir, "configs", run_tag)
    log_path = joinpath(campaign_dir, "logs", run_tag * ".log")
    mkpath(cfg_dir)

    data_doc = deepcopy(base_docs[:data])
    train_doc = deepcopy(base_docs[:train])
    responses_doc = deepcopy(base_docs[:responses])
    pipeline_doc = deepcopy(base_docs[:pipeline])

    apply_mods!(data_doc, candidate.data_mods)
    apply_mods!(train_doc, candidate.train_mods)
    apply_mods!(responses_doc, candidate.responses_mods)
    apply_mods!(pipeline_doc, candidate.pipeline_mods)

    # Compute a deterministic dataset path so repeated candidates with the same
    # data settings reuse generated datasets.
    data_sig = candidate_state_signature(data_doc)
    data_h5 = joinpath(campaign_dir, "data", "dataset_" * data_sig * ".hdf5")
    table!(data_doc, "paths")["datasets_hdf5"] = data_h5

    train_tbl = table!(train_doc, "train")
    ckpt_every = max(min(candidate.epochs, 60), 1)
    train_tbl["epochs"] = candidate.epochs
    train_tbl["checkpoint_every"] = ckpt_every
    train_tbl["save_state_every"] = ckpt_every
    train_tbl["seed"] = base_train_seed + candidate.seed_delta

    # Always enforce correction and GPU score-device in the campaign baseline.
    table!(responses_doc, "methods")["apply_score_correction"] = true
    table!(responses_doc, "cache")["force_regenerate"] = false
    table!(responses_doc, "gfdt")["score_device"] = "GPU:1"

    table!(pipeline_doc, "run")["runs_root"] = joinpath(campaign_dir, "runs")
    table!(pipeline_doc, "resources")["responses_score_device"] = "GPU:1"
    table!(pipeline_doc, "evaluation")["evaluate_every_checkpoint"] = false
    table!(pipeline_doc, "evaluation")["evaluate_final_model"] = true
    table!(pipeline_doc, "evaluation")["run_langevin"] = true
    table!(pipeline_doc, "evaluation")["run_responses"] = true
    table!(pipeline_doc, "evaluation")["response_apply_correction"] = true

    data_path = joinpath(cfg_dir, "parameters_data.toml")
    train_path = joinpath(cfg_dir, "parameters_train.toml")
    responses_path = joinpath(cfg_dir, "parameters_responses.toml")
    pipeline_path = joinpath(cfg_dir, "parameters_pipeline.toml")

    table!(train_doc, "paths")["data_params"] = data_path
    table!(responses_doc, "paths")["data_params"] = data_path

    ppaths = table!(pipeline_doc, "paths")
    default_langevin_params = abspath(String(get(table!(base_docs[:pipeline], "paths"), "langevin_params", DEFAULT_LANGEVIN_PARAMS)))
    isfile(default_langevin_params) || error("Missing Langevin params file for campaign: $default_langevin_params")
    ppaths["train_params"] = train_path
    ppaths["langevin_params"] = default_langevin_params
    ppaths["responses_params"] = responses_path

    write_toml(data_path, data_doc)
    write_toml(train_path, train_doc)
    write_toml(responses_path, responses_doc)
    write_toml(pipeline_path, pipeline_doc)

    pipeline_script = abspath(joinpath(project_root, "scripts", "arnold", "run_pipeline.jl"))
    isfile(pipeline_script) || error("Missing run_pipeline.jl: $pipeline_script")
    cmd = `julia --project=$(project_root) $(pipeline_script) --params $(pipeline_path)`
    t0 = time()
    status = "ok"
    err_msg = ""
    started_at_str = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    open(log_path, "w") do io
        println(io, "[campaign] candidate=", candidate.name)
        println(io, "[campaign] notes=", candidate.notes)
        println(io, "[campaign] started_at=", started_at_str)
        println(io, "[campaign] command=", cmd)
        flush(io)
        try
            run(pipeline(cmd; stdout=io, stderr=io))
        catch err
            status = "failed"
            err_msg = sprint(showerror, err)
            println(io, "[campaign] error=", err_msg)
        end
    end
    duration_min = (time() - t0) / 60.0

    run_dir = parse_run_dir_from_log(log_path)
    if isempty(run_dir)
        run_dir = latest_run_dir(joinpath(campaign_dir, "runs"))
    end
    run_dir = isempty(run_dir) ? "" : abspath(run_dir)

    metrics = isempty(run_dir) ? Dict{String,Any}() : extract_run_metrics(run_dir)
    return Dict{String,Any}(
        "name" => candidate.name,
        "notes" => candidate.notes,
        "status" => status,
        "error" => err_msg,
        "duration_min" => duration_min,
        "run_dir" => run_dir,
        "log_path" => log_path,
        "pipeline_params" => pipeline_path,
        "data_params" => data_path,
        "train_params" => train_path,
        "responses_params" => responses_path,
        "best_smape" => get(metrics, "best_smape", NaN),
        "best_epoch" => get(metrics, "best_epoch", -1),
        "best_summary" => get(metrics, "best_summary", ""),
        "kl_at_best_epoch" => get(metrics, "kl_at_best_epoch", NaN),
        "best_kl_overall" => get(metrics, "best_kl_overall", NaN),
        "num_response_summaries" => get(metrics, "num_response_summaries", 0),
    )
end

function metric_value(row::Dict{String,Any})
    v = get(row, "best_smape", NaN)
    try
        vv = Float64(v)
        return isfinite(vv) ? vv : Inf
    catch
        return Inf
    end
end

function best_finite_metric(rows::Vector{Dict{String,Any}})
    vals = Float64[]
    for row in rows
        v = metric_value(row)
        isfinite(v) && push!(vals, v)
    end
    return isempty(vals) ? Inf : minimum(vals)
end

function should_start_new_run(now_s::Float64, deadline_s::Float64, durations_min::Vector{Float64})
    remaining = deadline_s - now_s
    avg_run_s = isempty(durations_min) ? 90.0 * 60.0 : mean(durations_min) * 60.0
    min_needed = max(35.0 * 60.0, 0.6 * avg_run_s)
    return remaining >= min_needed
end

function main(args=ARGS)
    parsed = parse_args(args)
    campaign_dir = joinpath(parsed.campaign_root, "campaign_" * parsed.run_tag)
    ensure_campaign_layout(campaign_dir)

    started_at = Dates.now()
    deadline_s = time() + parsed.budget_hours * 3600.0

    base_docs = Dict{Symbol,Dict{String,Any}}(
        :pipeline => TOML.parsefile(DEFAULT_PIPELINE_PARAMS),
        :train => TOML.parsefile(DEFAULT_TRAIN_PARAMS),
        :responses => TOML.parsefile(DEFAULT_RESPONSES_PARAMS),
        :data => TOML.parsefile(DEFAULT_DATA_PARAMS),
    )
    base_train_seed = as_int(table!(base_docs[:train], "train"), "seed", 42)

    # Baseline state is intentionally aligned with the historically strongest
    # regime (sigma=0.05, correction enabled), then sweeps are targeted around it.
    baseline_state = Dict{Symbol,Dict{String,Any}}(
        :data => Dict{String,Any}(
            "closure.auto_fit" => false,
            "datasets.train_stochastic.nsamples" => 200_000,
            "datasets.gfdt_stochastic.nsamples" => 100_000,
        ),
        :train => Dict{String,Any}(
            "train.sigma" => 0.05,
            "train.lr" => 8e-4,
            "train.batch_size" => 128,
            "train.base_channels" => 32,
            "train.channel_multipliers" => [1, 2],
        ),
        :responses => Dict{String,Any}(
            "methods.apply_score_correction" => true,
            "gfdt.divergence_mode" => "hutchinson",
            "gfdt.divergence_probes" => 10,
            "gfdt.divergence_eps" => 0.03,
            "cache.force_regenerate" => false,
        ),
        :pipeline => Dict{String,Any}(
            "resources.responses_score_device" => "GPU:1",
            "evaluation.evaluate_every_checkpoint" => false,
            "evaluation.evaluate_final_model" => true,
            "evaluation.run_langevin" => true,
            "evaluation.run_responses" => true,
            "evaluation.response_apply_correction" => true,
        ),
    )

    best_state = Dict{Symbol,Dict{String,Any}}(
        :data => Dict{String,Any}(baseline_state[:data]),
        :train => Dict{String,Any}(baseline_state[:train]),
        :responses => Dict{String,Any}(baseline_state[:responses]),
        :pipeline => Dict{String,Any}(baseline_state[:pipeline]),
    )

    campaign_log = joinpath(campaign_dir, "reports", "campaign.log")
    campaign_started_at = Dates.format(started_at, dateformat"yyyy-mm-ddTHH:MM:SS")
    open(campaign_log, "w") do io
        println(io, "[campaign] dir=", campaign_dir)
        println(io, "[campaign] started_at=", campaign_started_at)
        println(io, "[campaign] budget_hours=", parsed.budget_hours)
    end

    results = Dict{String,Any}[]
    durations = Float64[]
    run_count = 0

    function run_and_record!(cand::CandidateSpec)
        run_count += 1
        row = run_candidate!(
            candidate=cand,
            run_index=run_count,
            campaign_dir=campaign_dir,
            project_root=DEFAULT_PROJECT_ROOT,
            base_docs=base_docs,
            base_train_seed=base_train_seed,
        )
        push!(results, row)
        push!(durations, row["duration_min"])
        save_results_toml(joinpath(campaign_dir, "reports", "results.toml"), results)
        open(campaign_log, "a") do io
            row_name = get(row, "name", "")
            row_status = get(row, "status", "")
            row_smape = get(row, "best_smape", NaN)
            row_kl = get(row, "kl_at_best_epoch", NaN)
            row_duration = round(Float64(get(row, "duration_min", NaN)); digits=2)
            row_run_dir = get(row, "run_dir", "")
            println(io, "[run] name=", row_name, " status=", row_status, " smape=", row_smape, " kl=", row_kl, " duration_min=", row_duration, " run_dir=", row_run_dir)
        end
        return row
    end

    function budget_ok()
        if parsed.max_runs > 0 && run_count >= parsed.max_runs
            return false
        end
        return should_start_new_run(time(), deadline_s, durations)
    end

    function exploration_epochs()
        remaining_h = (deadline_s - time()) / 3600.0
        if remaining_h >= 4.0
            return 120
        elseif remaining_h >= 2.0
            return 90
        else
            return 60
        end
    end

    # 1) Baseline
    if budget_ok()
        baseline = CandidateSpec(
            name="baseline",
            notes="Baseline from strongest prior regime (sigma=0.05, correction on, hutchinson)",
            data_mods=best_state[:data],
            train_mods=best_state[:train],
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        )
        run_and_record!(baseline)
    end

    # 2) Training sigma sweep
    sigma_candidates = [
        CandidateSpec(
            name="sigma_0p035",
            notes="Training sigma sweep down",
            data_mods=best_state[:data],
            train_mods=merge_mods(best_state[:train], Dict("train.sigma" => 0.035)),
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
        CandidateSpec(
            name="sigma_0p070",
            notes="Training sigma sweep up",
            data_mods=best_state[:data],
            train_mods=merge_mods(best_state[:train], Dict("train.sigma" => 0.070)),
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
    ]

    sigma_rows = Dict{String,Any}[]
    sigma_prev_best = best_finite_metric(results)
    for cand in sigma_candidates
        budget_ok() || break
        row = run_and_record!(cand)
        push!(sigma_rows, row)
    end
    if !isempty(sigma_rows)
        best_sigma_row = sigma_rows[argmin(metric_value.(sigma_rows))]
        if metric_value(best_sigma_row) < sigma_prev_best
            if best_sigma_row["name"] == "sigma_0p035"
                best_state[:train]["train.sigma"] = 0.035
            elseif best_sigma_row["name"] == "sigma_0p070"
                best_state[:train]["train.sigma"] = 0.070
            end
        end
    end

    # 3) Response/Jacobian sweep
    response_candidates = [
        CandidateSpec(
            name="response_exact_div",
            notes="Response-stage exact divergence for UNet conjugates",
            data_mods=best_state[:data],
            train_mods=best_state[:train],
            responses_mods=merge_mods(best_state[:responses], Dict(
                "gfdt.divergence_mode" => "exact",
                "gfdt.divergence_probes" => 1,
            )),
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
        CandidateSpec(
            name="response_hutch_probe20",
            notes="Response-stage higher-probe Hutchinson divergence",
            data_mods=best_state[:data],
            train_mods=best_state[:train],
            responses_mods=merge_mods(best_state[:responses], Dict(
                "gfdt.divergence_mode" => "hutchinson",
                "gfdt.divergence_probes" => 20,
                "gfdt.divergence_eps" => 0.02,
            )),
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
    ]

    response_rows = Dict{String,Any}[]
    response_prev_best = best_finite_metric(results)
    for cand in response_candidates
        budget_ok() || break
        row = run_and_record!(cand)
        push!(response_rows, row)
    end
    if !isempty(response_rows)
        best_resp_row = response_rows[argmin(metric_value.(response_rows))]
        if metric_value(best_resp_row) < response_prev_best
            if best_resp_row["name"] == "response_exact_div"
                best_state[:responses]["gfdt.divergence_mode"] = "exact"
                best_state[:responses]["gfdt.divergence_probes"] = 1
            elseif best_resp_row["name"] == "response_hutch_probe20"
                best_state[:responses]["gfdt.divergence_mode"] = "hutchinson"
                best_state[:responses]["gfdt.divergence_probes"] = 20
                best_state[:responses]["gfdt.divergence_eps"] = 0.02
            end
        end
    end

    # 4) Data-generation sweep
    data_candidates = [
        CandidateSpec(
            name="data_autofit_on",
            notes="Enable closure auto-fit against two-scale data",
            data_mods=merge_mods(best_state[:data], Dict(
                "closure.auto_fit" => true,
            )),
            train_mods=best_state[:train],
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
        CandidateSpec(
            name="data_more_samples",
            notes="Increase train/gfdt dataset sample counts",
            data_mods=merge_mods(best_state[:data], Dict(
                "datasets.train_stochastic.nsamples" => 300_000,
                "datasets.gfdt_stochastic.nsamples" => 150_000,
            )),
            train_mods=best_state[:train],
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
    ]

    data_rows = Dict{String,Any}[]
    data_prev_best = best_finite_metric(results)
    for cand in data_candidates
        budget_ok() || break
        row = run_and_record!(cand)
        push!(data_rows, row)
    end
    if !isempty(data_rows)
        best_data_row = data_rows[argmin(metric_value.(data_rows))]
        if metric_value(best_data_row) < data_prev_best
            if best_data_row["name"] == "data_autofit_on"
                best_state[:data]["closure.auto_fit"] = true
            elseif best_data_row["name"] == "data_more_samples"
                best_state[:data]["datasets.train_stochastic.nsamples"] = 300_000
                best_state[:data]["datasets.gfdt_stochastic.nsamples"] = 150_000
            end
        end
    end

    # 5) Learning-rate local sweep around the best-so-far
    lr_candidates = [
        CandidateSpec(
            name="lr_6e4",
            notes="Learning-rate local sweep down",
            data_mods=best_state[:data],
            train_mods=merge_mods(best_state[:train], Dict("train.lr" => 6e-4)),
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
        CandidateSpec(
            name="lr_1e3",
            notes="Learning-rate local sweep up",
            data_mods=best_state[:data],
            train_mods=merge_mods(best_state[:train], Dict("train.lr" => 1e-3)),
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=exploration_epochs(),
            seed_delta=0,
        ),
    ]

    lr_rows = Dict{String,Any}[]
    lr_prev_best = best_finite_metric(results)
    for cand in lr_candidates
        budget_ok() || break
        row = run_and_record!(cand)
        push!(lr_rows, row)
    end
    if !isempty(lr_rows)
        best_lr_row = lr_rows[argmin(metric_value.(lr_rows))]
        if metric_value(best_lr_row) < lr_prev_best
            if best_lr_row["name"] == "lr_6e4"
                best_state[:train]["train.lr"] = 6e-4
            elseif best_lr_row["name"] == "lr_1e3"
                best_state[:train]["train.lr"] = 1e-3
            end
        end
    end

    # 6) Confirmatory runs near the best region until time budget is consumed.
    confirm_idx = 1
    while budget_ok()
        rem_h = (deadline_s - time()) / 3600.0
        epochs = rem_h > 2.5 ? 180 : (rem_h > 1.4 ? 120 : 90)
        cand = CandidateSpec(
            name=@sprintf("confirm_%02d", confirm_idx),
            notes="Confirmatory run in best region",
            data_mods=best_state[:data],
            train_mods=best_state[:train],
            responses_mods=best_state[:responses],
            pipeline_mods=best_state[:pipeline],
            epochs=epochs,
            seed_delta=100 + confirm_idx,
        )
        run_and_record!(cand)
        confirm_idx += 1
    end

    ended_at = Dates.now()
    campaign_meta = Dict{String,Any}(
        "campaign_dir" => campaign_dir,
        "started_at" => Dates.format(started_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
        "ended_at" => Dates.format(ended_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
        "budget_hours" => parsed.budget_hours,
    )
    write_report(joinpath(campaign_dir, "reports", "report.md"), campaign_meta, results, best_state)
    save_results_toml(joinpath(campaign_dir, "reports", "results.toml"), results)

    println("campaign_dir=", campaign_dir)
    println("report=", joinpath(campaign_dir, "reports", "report.md"))
    println("results=", joinpath(campaign_dir, "reports", "results.toml"))
    println("runs_completed=", length(results))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
