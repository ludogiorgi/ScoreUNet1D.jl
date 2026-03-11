# Standard command (from repository root):
# julia --threads auto --project=. scripts/arnold_K36/fit_first_guess_closure.jl --params scripts/arnold_K36/parameters_first_guess.toml
# Nohup command:
# nohup julia --threads auto --project=. scripts/arnold_K36/fit_first_guess_closure.jl --params scripts/arnold_K36/parameters_first_guess.toml > scripts/arnold_K36/nohup_fit_first_guess_closure.log 2>&1 &

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Dates
using Printf
using Statistics
using TOML

include(joinpath(@__DIR__, "lib", "ArnoldK36Common.jl"))
using .ArnoldK36Common
include(joinpath(@__DIR__, "..", "arnold", "lib", "ArnoldStatsPlots.jl"))
using .ArnoldStatsPlots

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_first_guess.toml")

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

function maybe_attr_float(attrs::Dict{String,Any}, key::String, default::Float64)
    if haskey(attrs, key)
        v = attrs[key]
        if v isa Number
            return Float64(v)
        end
    end
    return default
end

function maybe_attr_int(attrs::Dict{String,Any}, key::String, default::Int)
    if haskey(attrs, key)
        v = attrs[key]
        if v isa Number
            return Int(round(Float64(v)))
        end
    end
    return default
end

function load_config(path::AbstractString)
    isfile(path) || error("First-guess parameter file not found: $path")
    doc = TOML.parsefile(path)

    paths = require_table(doc, "paths")
    fit = require_table(doc, "fit")
    stability = require_table(doc, "stability")
    reduced = require_table(doc, "reduced_long")
    observables = require_table(doc, "observables")
    figures = require_table(doc, "figures")

    cfg = Dict{String,Any}(
        "paths.params_file" => abspath(path),
        "paths.observations_hdf5" => abspath(as_str(paths, "observations_hdf5", "scripts/arnold_K36/data/l96_k36_observations.hdf5")),
        "paths.observations_key" => as_str(paths, "observations_key", "x_two_scale_observed"),
        "paths.output_dir" => abspath(as_str(paths, "output_dir", "scripts/arnold_K36/output/first_guess")),
        "paths.reduced_hdf5" => abspath(as_str(paths, "reduced_hdf5", "scripts/arnold_K36/output/first_guess/reduced_first_guess.hdf5")),
        "paths.reduced_key" => as_str(paths, "reduced_key", "x_reduced_first_guess"),
        "paths.first_guess_toml" => abspath(as_str(paths, "first_guess_toml", "scripts/arnold_K36/output/first_guess/first_guess_parameters.toml")),
        "paths.summary_toml" => abspath(as_str(paths, "summary_toml", "scripts/arnold_K36/output/first_guess/first_guess_summary.toml")),

        "fit.F" => as_float(fit, "F", 10.0),
        "fit.fit_start_index" => as_int(fit, "fit_start_index", 2),
        "fit.fit_samples" => as_int(fit, "fit_samples", 100_000),
        "fit.fit_min_samples" => as_int(fit, "fit_min_samples", 10_000),
        "fit.sigma_floor" => as_float(fit, "sigma_floor", 1e-8),
        "fit.sigma_scale" => as_float(fit, "sigma_scale", 1.0),
        "fit.sigma_cap" => as_float(fit, "sigma_cap", 4.0),

        "stability.screen_samples" => as_int(stability, "screen_samples", 40_000),
        "stability.screen_spinup_steps" => as_int(stability, "screen_spinup_steps", 3_000),
        "stability.screen_save_every" => as_int(stability, "screen_save_every", 10),
        "stability.screen_rng_seed" => as_int(stability, "screen_rng_seed", 444),
        "stability.screen_trajectories" => as_int(stability, "screen_trajectories", 8),
        "stability.parallel_trajectories" => as_bool(stability, "parallel_trajectories", true),
        "stability.allow_sigma_backoff" => as_bool(stability, "allow_sigma_backoff", true),
        "stability.sigma_backoff_factor" => as_float(stability, "sigma_backoff_factor", 0.8),
        "stability.max_sigma_backoffs" => as_int(stability, "max_sigma_backoffs", 5),
        "stability.max_abs_state" => as_float(stability, "max_abs_state", 1e4),
        "stability.state_min" => as_float(stability, "state_min", -200.0),
        "stability.state_max" => as_float(stability, "state_max", 200.0),
        "stability.max_boundary_hits" => as_int(stability, "max_boundary_hits", 80),

        "reduced_long.nsamples" => as_int(reduced, "nsamples", 240_000),
        "reduced_long.spinup_steps" => as_int(reduced, "spinup_steps", 5_000),
        "reduced_long.save_every" => as_int(reduced, "save_every", 10),
        "reduced_long.rng_seed" => as_int(reduced, "rng_seed", 777),
        "reduced_long.trajectories" => as_int(reduced, "trajectories", 12),
        "reduced_long.parallel_trajectories" => as_bool(reduced, "parallel_trajectories", true),

        "observables.m" => as_int(observables, "m", 3),
        "observables.compare_samples" => as_int(observables, "compare_samples", 120_000),

        "figures.pdf_bins" => as_int(figures, "pdf_bins", 80),
        "figures.max_acf_lag" => as_int(figures, "max_acf_lag", 200),
        "figures.figB_path" => abspath(as_str(figures, "figB_path", "scripts/arnold_K36/output/first_guess/figB_stats_deterministic_vs_stochastic_4x2.png")),
    )

    cfg["fit.fit_samples"] >= 10 || error("fit.fit_samples must be >= 10")
    cfg["fit.fit_min_samples"] >= 10 || error("fit.fit_min_samples must be >= 10")
    cfg["fit.sigma_floor"] >= 0 || error("fit.sigma_floor must be >= 0")
    cfg["fit.sigma_scale"] > 0 || error("fit.sigma_scale must be > 0")
    cfg["fit.sigma_cap"] >= cfg["fit.sigma_floor"] || error("fit.sigma_cap must be >= fit.sigma_floor")

    cfg["stability.screen_samples"] >= 2 || error("stability.screen_samples must be >= 2")
    cfg["stability.screen_spinup_steps"] >= 0 || error("stability.screen_spinup_steps must be >= 0")
    cfg["stability.screen_save_every"] >= 1 || error("stability.screen_save_every must be >= 1")
    cfg["stability.screen_trajectories"] >= 1 || error("stability.screen_trajectories must be >= 1")
    cfg["stability.sigma_backoff_factor"] > 0 || error("stability.sigma_backoff_factor must be > 0")
    cfg["stability.sigma_backoff_factor"] < 1 || error("stability.sigma_backoff_factor must be < 1")
    cfg["stability.max_sigma_backoffs"] >= 0 || error("stability.max_sigma_backoffs must be >= 0")
    cfg["stability.max_abs_state"] > 0 || error("stability.max_abs_state must be > 0")
    cfg["stability.state_min"] <= cfg["stability.state_max"] || error("stability.state_min must be <= stability.state_max")
    cfg["stability.max_boundary_hits"] >= 1 || error("stability.max_boundary_hits must be >= 1")

    cfg["reduced_long.nsamples"] >= 2 || error("reduced_long.nsamples must be >= 2")
    cfg["reduced_long.spinup_steps"] >= 0 || error("reduced_long.spinup_steps must be >= 0")
    cfg["reduced_long.save_every"] >= 1 || error("reduced_long.save_every must be >= 1")
    cfg["reduced_long.trajectories"] >= 1 || error("reduced_long.trajectories must be >= 1")

    cfg["observables.m"] >= 0 || error("observables.m must be >= 0")
    cfg["observables.compare_samples"] >= 2 || error("observables.compare_samples must be >= 2")
    cfg["figures.pdf_bins"] >= 10 || error("figures.pdf_bins must be >= 10")
    cfg["figures.max_acf_lag"] >= 1 || error("figures.max_acf_lag must be >= 1")

    return cfg
end

function observable_names(m::Int)
    m >= 0 || error("m must be >= 0")
    out = String["phi1_mean_x", "phi2_mean_x2"]
    for lag in 1:m
        push!(out, "phi$(lag + 2)_mean_x_xp$(lag)")
    end
    return out
end

function matrix_to_tensor(x::Matrix{Float64})
    K, N = size(x)
    out = Array{Float32}(undef, K, 1, N)
    @inbounds for n in 1:N, k in 1:K
        out[k, 1, n] = Float32(x[k, n])
    end
    return out
end

function stable_screen(theta::NTuple{5,Float64},
    cfg::Dict{String,Any},
    K::Int,
    F::Float64,
    dt::Float64,
    save_every::Int)
    generate_reduced_ensemble_x_timeseries(
        K=K,
        F=F,
        alpha0=theta[1],
        alpha1=theta[2],
        alpha2=theta[3],
        alpha3=theta[4],
        sigma=theta[5],
        dt=dt,
        spinup_steps=cfg["stability.screen_spinup_steps"],
        save_every=save_every,
        nsamples=cfg["stability.screen_samples"],
        rng_seed=cfg["stability.screen_rng_seed"],
        trajectories=cfg["stability.screen_trajectories"],
        parallel_trajectories=cfg["stability.parallel_trajectories"],
        max_abs_state=cfg["stability.max_abs_state"],
        max_restarts=20,
        state_min=cfg["stability.state_min"],
        state_max=cfg["stability.state_max"],
        max_boundary_hits=cfg["stability.max_boundary_hits"],
    )
    return true
end

function find_stable_theta(theta_raw::NTuple{5,Float64},
    cfg::Dict{String,Any},
    K::Int,
    F::Float64,
    dt::Float64,
    save_every::Int)
    sigma_backoff_factor = cfg["stability.sigma_backoff_factor"]
    max_backoffs = cfg["stability.max_sigma_backoffs"]
    allow_backoff = cfg["stability.allow_sigma_backoff"]

    theta_cur = theta_raw
    sigma_attempts = Float64[theta_cur[5]]
    attempts = 0
    last_error = ""

    while true
        try
            stable_screen(theta_cur, cfg, K, F, dt, save_every)
            return theta_cur, Dict{String,Any}(
                "stable" => true,
                "sigma_attempts" => sigma_attempts,
                "sigma_backoffs_used" => attempts,
                "last_error" => "",
            )
        catch err
            last_error = sprint(showerror, err)
            if !allow_backoff || attempts >= max_backoffs
                return theta_cur, Dict{String,Any}(
                    "stable" => false,
                    "sigma_attempts" => sigma_attempts,
                    "sigma_backoffs_used" => attempts,
                    "last_error" => last_error,
                )
            end
            attempts += 1
            sigma_new = max(cfg["fit.sigma_floor"], theta_cur[5] * sigma_backoff_factor)
            theta_cur = (theta_cur[1], theta_cur[2], theta_cur[3], theta_cur[4], sigma_new)
            push!(sigma_attempts, sigma_new)
        end
    end
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg = load_config(parsed.params_path)

    obs_path = cfg["paths.observations_hdf5"]
    obs_key = cfg["paths.observations_key"]
    out_dir = ensure_dir(cfg["paths.output_dir"])

    Xdet = load_x_matrix(obs_path, obs_key)
    attrs = load_dataset_attributes(obs_path, obs_key)

    K, Ndet = size(Xdet)
    m_obs = cfg["observables.m"]
    m_obs <= K - 1 || error("observables.m=$(m_obs) must be <= K-1=$(K-1)")

    dt = maybe_attr_float(attrs, "dt", 0.005)
    save_every_obs = maybe_attr_int(attrs, "save_every", cfg["reduced_long.save_every"])
    dt_obs = dataset_time_spacing(dt, save_every_obs)
    F_fit = maybe_attr_float(attrs, "F", cfg["fit.F"])

    theta_raw, fit_meta = fit_polynomial_closure(
        Xdet,
        dt_obs,
        F_fit;
        fit_start_index=cfg["fit.fit_start_index"],
        fit_samples=cfg["fit.fit_samples"],
        fit_min_samples=cfg["fit.fit_min_samples"],
        sigma_floor=cfg["fit.sigma_floor"],
        sigma_scale=cfg["fit.sigma_scale"],
        sigma_cap=cfg["fit.sigma_cap"],
    )

    theta_stable, stability_meta = find_stable_theta(theta_raw, cfg, K, F_fit, dt, save_every_obs)
    stability_meta["stable"] || error("Could not find a stable reduced model. Last error: $(stability_meta["last_error"])")

    Xred = generate_reduced_ensemble_x_timeseries(
        K=K,
        F=F_fit,
        alpha0=theta_stable[1],
        alpha1=theta_stable[2],
        alpha2=theta_stable[3],
        alpha3=theta_stable[4],
        sigma=theta_stable[5],
        dt=dt,
        spinup_steps=cfg["reduced_long.spinup_steps"],
        save_every=cfg["reduced_long.save_every"],
        nsamples=cfg["reduced_long.nsamples"],
        rng_seed=cfg["reduced_long.rng_seed"],
        trajectories=cfg["reduced_long.trajectories"],
        parallel_trajectories=cfg["reduced_long.parallel_trajectories"],
        max_abs_state=cfg["stability.max_abs_state"],
        max_restarts=40,
        state_min=cfg["stability.state_min"],
        state_max=cfg["stability.state_max"],
        max_boundary_hits=cfg["stability.max_boundary_hits"],
    )

    reduced_attrs = Dict{String,Any}(
        "role" => "reduced_first_guess",
        "model" => "one_scale_stochastic",
        "generated_at" => string(Dates.now()),
        "source_observations_path" => obs_path,
        "source_observations_key" => obs_key,
        "K" => K,
        "dt" => dt,
        "save_every" => cfg["reduced_long.save_every"],
        "nsamples" => cfg["reduced_long.nsamples"],
        "spinup_steps" => cfg["reduced_long.spinup_steps"],
        "rng_seed" => cfg["reduced_long.rng_seed"],
        "F" => F_fit,
        "alpha0" => theta_stable[1],
        "alpha1" => theta_stable[2],
        "alpha2" => theta_stable[3],
        "alpha3" => theta_stable[4],
        "sigma" => theta_stable[5],
    )
    save_x_dataset(cfg["paths.reduced_hdf5"], cfg["paths.reduced_key"], Xred, reduced_attrs)

    ncmp = min(cfg["observables.compare_samples"], size(Xdet, 2), size(Xred, 1))
    Xdet_cmp = Xdet[:, 1:ncmp]
    Xred_cmp = permutedims(Float64.(Xred[1:ncmp, :]), (2, 1))

    obs_det = vec(mean(compute_observables_series(Xdet_cmp, m_obs); dims=2))
    obs_red = vec(mean(compute_observables_series(Xred_cmp, m_obs); dims=2))
    obs_rmse = sqrt(mean((obs_red .- obs_det).^2))

    obs_tensor = matrix_to_tensor(Xdet_cmp)
    red_tensor = matrix_to_tensor(Xred_cmp)
    kl_mode, js_mode = modewise_metrics(obs_tensor, red_tensor; nbins=cfg["figures.pdf_bins"], low_q=0.001, high_q=0.999)
    avg_kl = mean(kl_mode)
    avg_js = mean(js_mode)

    fig_path = cfg["figures.figB_path"]
    mkpath(dirname(fig_path))
    save_stats_figure_acf(
        fig_path,
        obs_tensor,
        red_tensor,
        kl_mode,
        js_mode,
        cfg["figures.pdf_bins"];
        max_lag=cfg["figures.max_acf_lag"],
        obs_label="deterministic K36",
        gen_label="stochastic first guess",
    )

    first_guess_doc = Dict{String,Any}(
        "closure" => Dict(
            "F" => F_fit,
            "alpha0" => theta_stable[1],
            "alpha1" => theta_stable[2],
            "alpha2" => theta_stable[3],
            "alpha3" => theta_stable[4],
            "sigma" => theta_stable[5],
        ),
        "closure_initial_from_fit" => Dict(
            "alpha0" => theta_raw[1],
            "alpha1" => theta_raw[2],
            "alpha2" => theta_raw[3],
            "alpha3" => theta_raw[4],
            "sigma" => theta_raw[5],
        ),
        "fit_meta" => fit_meta,
        "stability_meta" => stability_meta,
    )
    mkpath(dirname(cfg["paths.first_guess_toml"]))
    open(cfg["paths.first_guess_toml"], "w") do io
        TOML.print(io, first_guess_doc)
    end

    obs_names = observable_names(m_obs)
    obs_table = Dict{String,Any}()
    for i in 1:length(obs_det)
        key = i <= length(obs_names) ? obs_names[i] : "obs_$i"
        obs_table[key] = Dict("deterministic" => obs_det[i], "stochastic" => obs_red[i], "abs_diff" => abs(obs_red[i] - obs_det[i]))
    end

    summary = Dict{String,Any}(
        "parameters_file" => parsed.params_path,
        "observations" => Dict(
            "path" => obs_path,
            "key" => obs_key,
            "K" => K,
            "N" => Ndet,
            "dt" => dt,
            "save_every" => save_every_obs,
            "dt_obs" => dt_obs,
        ),
        "reduced_output" => Dict(
            "path" => cfg["paths.reduced_hdf5"],
            "key" => cfg["paths.reduced_key"],
            "nsamples" => cfg["reduced_long.nsamples"],
        ),
        "first_guess" => Dict(
            "F" => F_fit,
            "alpha0" => theta_stable[1],
            "alpha1" => theta_stable[2],
            "alpha2" => theta_stable[3],
            "alpha3" => theta_stable[4],
            "sigma" => theta_stable[5],
        ),
        "metrics" => Dict(
            "compare_samples" => ncmp,
            "observable_rmse" => obs_rmse,
            "avg_mode_kl" => avg_kl,
            "avg_mode_js" => avg_js,
        ),
        "observables" => obs_table,
        "figure_figB_4x2" => fig_path,
        "fit_meta" => fit_meta,
        "stability_meta" => stability_meta,
    )

    mkpath(dirname(cfg["paths.summary_toml"]))
    open(cfg["paths.summary_toml"], "w") do io
        TOML.print(io, summary)
    end

    println(@sprintf("first_guess_alpha=[%.8f, %.8f, %.8f, %.8f]", theta_stable[1], theta_stable[2], theta_stable[3], theta_stable[4]))
    println(@sprintf("first_guess_sigma=%.8f", theta_stable[5]))
    println(@sprintf("observable_rmse=%.8e", obs_rmse))
    println(@sprintf("avg_mode_kl=%.8e", avg_kl))
    println("reduced_hdf5=$(cfg["paths.reduced_hdf5"])")
    println("first_guess_toml=$(cfg["paths.first_guess_toml"])")
    println("summary=$(cfg["paths.summary_toml"])")
    println("figB=$(fig_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
