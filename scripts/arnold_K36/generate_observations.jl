# Standard command (from repository root):
# julia --project=. scripts/arnold_K36/generate_observations.jl --params scripts/arnold_K36/parameters_observations.toml
# Nohup command:
# nohup julia --project=. scripts/arnold_K36/generate_observations.jl --params scripts/arnold_K36/parameters_observations.toml > scripts/arnold_K36/nohup_generate_observations.log 2>&1 &

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Dates
using TOML

include(joinpath(@__DIR__, "lib", "ArnoldK36Common.jl"))
using .ArnoldK36Common

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_observations.toml")

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

    return (params_path=abspath(params_path),)
end

function require_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
as_bool(tbl::Dict{String,Any}, key::String, default) = ArnoldK36Common.parse_bool(get(tbl, key, default))

function load_config(path::AbstractString)
    isfile(path) || error("Observation parameter file not found: $path")
    doc = TOML.parsefile(path)

    paths = require_table(doc, "paths")
    twoscale = require_table(doc, "twoscale")
    data = require_table(doc, "data")

    cfg = Dict{String,Any}(
        "paths.params_file" => abspath(path),
        "paths.output_hdf5" => abspath(as_str(paths, "output_hdf5", "scripts/arnold_K36/data/l96_k36_observations.hdf5")),
        "paths.dataset_key" => as_str(paths, "dataset_key", "x_two_scale_observed"),
        "paths.summary_toml" => abspath(as_str(paths, "summary_toml", "scripts/arnold_K36/data/observations_summary.toml")),

        "twoscale.K" => as_int(twoscale, "K", 36),
        "twoscale.J" => as_int(twoscale, "J", 10),
        "twoscale.F" => as_float(twoscale, "F", 10.0),
        "twoscale.h" => as_float(twoscale, "h", 1.0),
        "twoscale.c" => as_float(twoscale, "c", 10.0),
        "twoscale.b" => as_float(twoscale, "b", 10.0),
        "twoscale.dt" => as_float(twoscale, "dt", 0.005),
        "twoscale.process_noise_sigma" => as_float(twoscale, "process_noise_sigma", 0.0),
        "twoscale.stochastic_x_noise" => as_bool(twoscale, "stochastic_x_noise", false),

        "data.spinup_steps" => as_int(data, "spinup_steps", 5_000),
        "data.save_every" => as_int(data, "save_every", 10),
        "data.nsamples" => as_int(data, "nsamples", 120_000),
        "data.rng_seed" => as_int(data, "rng_seed", 101),
        "data.force_regenerate" => as_bool(data, "force_regenerate", false),
    )

    cfg["twoscale.K"] >= 2 || error("twoscale.K must be >= 2")
    cfg["twoscale.J"] >= 1 || error("twoscale.J must be >= 1")
    cfg["twoscale.dt"] > 0 || error("twoscale.dt must be > 0")
    cfg["data.spinup_steps"] >= 0 || error("data.spinup_steps must be >= 0")
    cfg["data.save_every"] >= 1 || error("data.save_every must be >= 1")
    cfg["data.nsamples"] >= 2 || error("data.nsamples must be >= 2")
    cfg["twoscale.process_noise_sigma"] >= 0 || error("twoscale.process_noise_sigma must be >= 0")

    return cfg
end

function params_signature(cfg::Dict{String,Any})
    return dict_signature(Dict(
        "model" => "two_scale_deterministic",
        "twoscale" => Dict(
            "K" => cfg["twoscale.K"],
            "J" => cfg["twoscale.J"],
            "F" => cfg["twoscale.F"],
            "h" => cfg["twoscale.h"],
            "c" => cfg["twoscale.c"],
            "b" => cfg["twoscale.b"],
            "dt" => cfg["twoscale.dt"],
            "process_noise_sigma" => cfg["twoscale.process_noise_sigma"],
            "stochastic_x_noise" => cfg["twoscale.stochastic_x_noise"],
        ),
        "data" => Dict(
            "spinup_steps" => cfg["data.spinup_steps"],
            "save_every" => cfg["data.save_every"],
            "nsamples" => cfg["data.nsamples"],
            "rng_seed" => cfg["data.rng_seed"],
        ),
    ))
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg = load_config(parsed.params_path)

    out_path = cfg["paths.output_hdf5"]
    key = cfg["paths.dataset_key"]
    summary_path = cfg["paths.summary_toml"]
    sig = params_signature(cfg)

    generated = true
    if !cfg["data.force_regenerate"]
        old_sig = read_dataset_signature(out_path, key)
        if old_sig == sig
            generated = false
        end
    end

    if generated
        X = generate_two_scale_x_timeseries(
            K=cfg["twoscale.K"],
            J=cfg["twoscale.J"],
            F=cfg["twoscale.F"],
            h=cfg["twoscale.h"],
            c=cfg["twoscale.c"],
            b=cfg["twoscale.b"],
            dt=cfg["twoscale.dt"],
            spinup_steps=cfg["data.spinup_steps"],
            save_every=cfg["data.save_every"],
            nsamples=cfg["data.nsamples"],
            rng_seed=cfg["data.rng_seed"],
            process_noise_sigma=cfg["twoscale.process_noise_sigma"],
            stochastic_x_noise=cfg["twoscale.stochastic_x_noise"],
        )
        attrs = Dict{String,Any}(
            "role" => "two_scale_observed",
            "model" => "two_scale_deterministic",
            "generated_at" => string(Dates.now()),
            "params_signature" => sig,
            "source_parameters" => parsed.params_path,
            "K" => cfg["twoscale.K"],
            "J" => cfg["twoscale.J"],
            "F" => cfg["twoscale.F"],
            "h" => cfg["twoscale.h"],
            "c" => cfg["twoscale.c"],
            "b" => cfg["twoscale.b"],
            "dt" => cfg["twoscale.dt"],
            "process_noise_sigma" => cfg["twoscale.process_noise_sigma"],
            "stochastic_x_noise" => cfg["twoscale.stochastic_x_noise"],
            "spinup_steps" => cfg["data.spinup_steps"],
            "save_every" => cfg["data.save_every"],
            "nsamples" => cfg["data.nsamples"],
            "rng_seed" => cfg["data.rng_seed"],
        )
        save_x_dataset(out_path, key, X, attrs)
    end

    summary = Dict{String,Any}(
        "parameters_file" => parsed.params_path,
        "dataset_path" => out_path,
        "dataset_key" => key,
        "generated" => generated,
        "params_signature" => sig,
        "dt_obs" => dataset_time_spacing(cfg["twoscale.dt"], cfg["data.save_every"]),
        "twoscale" => Dict(
            "K" => cfg["twoscale.K"],
            "J" => cfg["twoscale.J"],
            "F" => cfg["twoscale.F"],
            "h" => cfg["twoscale.h"],
            "c" => cfg["twoscale.c"],
            "b" => cfg["twoscale.b"],
            "dt" => cfg["twoscale.dt"],
        ),
        "data" => Dict(
            "spinup_steps" => cfg["data.spinup_steps"],
            "save_every" => cfg["data.save_every"],
            "nsamples" => cfg["data.nsamples"],
            "rng_seed" => cfg["data.rng_seed"],
        ),
    )
    mkpath(dirname(summary_path))
    open(summary_path, "w") do io
        TOML.print(io, summary)
    end

    println("dataset_path=$out_path")
    println("dataset_key=$key")
    println("generated=$generated")
    println("summary=$summary_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
