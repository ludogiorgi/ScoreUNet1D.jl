#!/usr/bin/env julia
# Recompute UNet GFDT response figures from saved UNet checkpoints.
#
# Example:
#   julia --threads auto --project=. scripts/arnold/replot_unet_responses_run.jl --params scripts/arnold/parameters_replot_unet_responses.toml

using HDF5
using Printf
using TOML

include(joinpath(@__DIR__, "lib", "ArnoldStatsPlots.jl"))
include(joinpath(@__DIR__, "compute_responses.jl"))
include(joinpath(@__DIR__, "lib", "CalibrationCommon.jl"))
using .ArnoldCommon
using .CalibrationCommon

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_replot_unet_responses.toml")

    i = 1
    while i <= length(args)
        a = strip(args[i])
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = strip(args[i + 1])
            i += 2
        elseif startswith(a, "--params=")
            params_path = strip(split(a, "=", limit=2)[2])
            i += 1
        else
            error("Unknown argument: $a")
        end
    end

    return (params_path=abspath(params_path),)
end

function maybe_table(doc::Dict{String,Any}, key::String)
    if !haskey(doc, key)
        return Dict{String,Any}()
    end
    doc[key] isa Dict{String,Any} || error("[$key] must be a TOML table")
    return Dict{String,Any}(doc[key])
end

function as_str(tbl::Dict{String,Any}, key::String, default)
    return String(get(tbl, key, default))
end

function as_float(tbl::Dict{String,Any}, key::String, default)
    return Float64(get(tbl, key, default))
end

function as_int(tbl::Dict{String,Any}, key::String, default)
    return Int(get(tbl, key, default))
end

function maybe_as_float(tbl::Dict{String,Any}, key::String)
    return haskey(tbl, key) ? Float64(tbl[key]) : NaN
end

function load_params(path::String)
    isfile(path) || error("Parameters file not found: $path")
    doc = TOML.parsefile(path)

    paths = maybe_table(doc, "paths")
    plot = maybe_table(doc, "plot")

    target_dir = abspath(as_str(paths, "target_dir", ""))
    isempty(strip(target_dir)) && error("[paths].target_dir is required")

    calibration_config = strip(as_str(paths, "calibration_config", ""))
    calibration_config_abs = isempty(calibration_config) ? "" : abspath(calibration_config)

    out_subdir = as_str(paths, "output_subdir", "unet_responses")
    response_tmax = as_float(plot, "response_tmax", 6.0)
    t_start_override = maybe_as_float(plot, "t_start")
    t_end_override = maybe_as_float(plot, "t_end")
    response_kind = lowercase(strip(as_str(plot, "response_kind", "heaviside")))
    dpi_override = as_int(plot, "dpi", -1)

    response_tmax > 0 || error("plot.response_tmax must be > 0")
    if isfinite(t_start_override) || isfinite(t_end_override)
        t_start_use = isfinite(t_start_override) ? t_start_override : 0.0
        t_end_use = isfinite(t_end_override) ? t_end_override : response_tmax
        t_start_use >= 0 || error("plot.t_start must be >= 0")
        t_end_use > t_start_use || error("plot.t_end must be > plot.t_start")
        t_end_use <= response_tmax + 1e-12 || error("plot.t_end must be <= plot.response_tmax")
    end
    response_kind in ("heaviside", "impulse") || error("plot.response_kind must be 'heaviside' or 'impulse'")

    return (
        target_dir=target_dir,
        calibration_config=calibration_config_abs,
        out_subdir=out_subdir,
        response_tmax=response_tmax,
        t_start_override=t_start_override,
        t_end_override=t_end_override,
        response_kind=response_kind,
        dpi_override=dpi_override,
    )
end

function find_run_dir_from_path(path::String)
    cur = abspath(path)
    for _ in 1:10
        cfg_path = joinpath(cur, "config", "parameters_calibration.toml")
        if isfile(cfg_path)
            return cur
        end
        parent = dirname(cur)
        parent == cur && break
        cur = parent
    end
    error("Could not locate run_### root (missing config/parameters_calibration.toml) from: $path")
end

function iter_number(name::AbstractString)
    startswith(name, "iter_") || return typemax(Int)
    n = tryparse(Int, split(String(name), "_")[end])
    n === nothing && return typemax(Int)
    return n
end

function resolve_iteration_dirs(target_dir::String)
    target = abspath(target_dir)
    isdir(target) || error("Target directory not found: $target")

    base = basename(target)
    if startswith(base, "run_")
        method_root = joinpath(target, "method_unet")
        isdir(method_root) || error("Missing method_unet directory under run folder: $method_root")
        iter_names = [name for name in readdir(method_root) if startswith(name, "iter_") && isdir(joinpath(method_root, name))]
        sort!(iter_names; by=iter_number)
        isempty(iter_names) && error("No iter_* directories found under $method_root")
        return find_run_dir_from_path(target), [joinpath(method_root, name) for name in iter_names]
    elseif startswith(base, "iter_")
        data_dir = joinpath(target, "data")
        model_dir = joinpath(target, "model")
        if isdir(data_dir) && isdir(model_dir)
            return find_run_dir_from_path(target), [target]
        end

        # Handle top-level run/iter_### folders whose UNet artifacts live under method_unet/iter_###.
        run_dir = find_run_dir_from_path(target)
        iter_name = basename(target)
        redirected = joinpath(run_dir, "method_unet", iter_name)
        data_dir_r = joinpath(redirected, "data")
        model_dir_r = joinpath(redirected, "model")
        if isdir(data_dir_r) && isdir(model_dir_r)
            @info "Redirecting iter folder to method_unet path" from = target to = redirected
            return run_dir, [redirected]
        end
        error("iter folder missing data/model directories: $target (also checked $redirected)")
    elseif base == "method_unet"
        iter_names = [name for name in readdir(target) if startswith(name, "iter_") && isdir(joinpath(target, name))]
        sort!(iter_names; by=iter_number)
        isempty(iter_names) && error("No iter_* directories found under $target")
        return find_run_dir_from_path(target), [joinpath(target, name) for name in iter_names]
    else
        error("target_dir must point to run_###, method_unet, or iter_### folder; got: $target")
    end
end

function detect_dataset_key(path::AbstractString)
    return h5open(path, "r") do h5
        for key in keys(h5)
            obj = h5[key]
            if obj isa HDF5.Dataset
                return String(key)
            end
        end
        error("No datasets found in $path")
    end
end

function load_x_matrix(path::AbstractString, key::AbstractString, K_expected::Int)
    isfile(path) || error("Dataset file not found: $path")
    key_use = isempty(strip(String(key))) ? detect_dataset_key(path) : String(key)
    raw = h5open(path, "r") do h5
        haskey(h5, key_use) || error("Dataset key '$key_use' not found in $path")
        Float64.(read(h5[key_use]))
    end

    ndims(raw) == 2 || error("Expected 2D dataset at $path/$key_use")
    if size(raw, 2) == K_expected
        return permutedims(raw, (2, 1)), key_use
    elseif size(raw, 1) == K_expected
        return raw, key_use
    end
    error("Cannot infer orientation for $path/$key_use with shape $(size(raw)) and K=$K_expected")
end

function read_sigma_attr(path::AbstractString, key::AbstractString, fallback::Float64)
    return h5open(path, "r") do h5
        haskey(h5, key) || return fallback
        dset = h5[key]
        ad = HDF5.attributes(dset)
        if haskey(ad, "sigma")
            try
                return Float64(read(ad["sigma"]))
            catch
                return fallback
            end
        end
        return fallback
    end
end

function make_tmax_tag(x::Float64)
    s = @sprintf("%.2f", x)
    return replace(s, "." => "p")
end

function main(args=ARGS)
    cli = parse_args(args)
    parsed = load_params(cli.params_path)

    run_dir, iter_dirs = resolve_iteration_dirs(parsed.target_dir)
    cfg_path = isempty(parsed.calibration_config) ? joinpath(run_dir, "config", "parameters_calibration.toml") : parsed.calibration_config
    isfile(cfg_path) || error("Missing config file in run dir: $cfg_path")
    cfg = load_calibration_config(cfg_path)

    cfg["responses.response_tmax"] = parsed.response_tmax
    if isfinite(parsed.t_start_override)
        cfg["responses.t_start"] = parsed.t_start_override
    end
    if isfinite(parsed.t_end_override)
        cfg["responses.t_end"] = parsed.t_end_override
    end
    cfg["responses.t_start"] >= 0 || error("responses.t_start must be >= 0")
    cfg["responses.t_end"] > cfg["responses.t_start"] || error("responses.t_end must be > responses.t_start")
    cfg["responses.t_end"] <= cfg["responses.response_tmax"] + 1e-12 || error("responses.t_end must be <= responses.response_tmax")
    cfg["methods.unet"] = true
    cfg["methods.gaussian"] = false
    cfg["methods.finite_difference"] = false
    if parsed.dpi_override > 0
        cfg["figures.dpi"] = parsed.dpi_override
    end

    out_dir = joinpath(run_dir, parsed.out_subdir)
    mkpath(out_dir)

    K = cfg["integration.K"]
    gfdt_key_cfg = cfg["datasets.gfdt_key"]

    obs_ref = (
        F_ref = cfg["observables.F_ref"],
        alpha0_ref = cfg["observables.alpha0_ref"],
        alpha1_ref = cfg["observables.alpha1_ref"],
        alpha2_ref = cfg["observables.alpha2_ref"],
        alpha3_ref = cfg["observables.alpha3_ref"],
    )

    created = String[]
    tmax_tag = make_tmax_tag(parsed.response_tmax)

    for iter_dir in iter_dirs
        iter_name = basename(iter_dir)
        it = iter_number(iter_name)

        gfdt_path = joinpath(iter_dir, "data", "gfdt_stochastic.hdf5")
        ckpt_path = joinpath(iter_dir, "model", "final_checkpoint.bson")
        isfile(gfdt_path) || error("Missing GFDT data file: $gfdt_path")
        isfile(ckpt_path) || error("Missing checkpoint file: $ckpt_path")

        X, gfdt_key = load_x_matrix(gfdt_path, gfdt_key_cfg, K)
        sigma_param = read_sigma_attr(gfdt_path, gfdt_key, Float64(cfg["closure.sigma"]))

        A = ArnoldCommon.compute_observables_series(
            X,
            obs_ref.F_ref,
            obs_ref.alpha0_ref,
            obs_ref.alpha1_ref,
            obs_ref.alpha2_ref,
            obs_ref.alpha3_ref,
            Int(cfg["observables.m"]),
        )

        n_lags, dt_obs = CalibrationCommon.response_n_lags(cfg, size(X, 2))

        unet = unet_conjugates(
            X,
            sigma_param,
            ckpt_path;
            batch_size=cfg["responses.batch_size"],
            score_device=cfg["responses.score_device"],
            score_forward_mode=cfg["responses.score_forward_mode"],
            apply_correction=cfg["responses.apply_score_correction"],
            divergence_mode=cfg["responses.divergence_mode"],
            divergence_eps=cfg["responses.divergence_eps"],
            divergence_probes=cfg["responses.divergence_probes"],
            divergence_seed=cfg["datasets.gfdt_rng_seed_base"] + 10_000 + it,
        )

        G_conj = cfg["responses.apply_score_correction"] ? unet.G_corr : unet.G_raw
        C, R_step, times = ArnoldCommon.build_gfdt_response(
            A,
            G_conj,
            dt_obs,
            n_lags;
            mean_center=cfg["responses.mean_center"],
            impulse_tail_debias=cfg["responses.impulse_tail_debias"],
            t_start=cfg["responses.t_start"],
            t_end=cfg["responses.t_end"],
            tail_taper=cfg["responses.impulse_tail_taper"],
        )
        S = extract_asymptotic_jacobians(times, R_step; t_start=cfg["responses.t_start"], t_end=cfg["responses.t_end"])
        response_data = parsed.response_kind == "heaviside" ? R_step : ArnoldCommon.step_to_impulse(R_step, times)

        curves = NamedTuple[]
        push!(curves, (
            method_key = "unet",
            label = "UNet",
            color = :orangered3,
            linestyle = :solid,
            data = response_data,
        ))

        asym = NamedTuple[]
        if parsed.response_kind == "heaviside"
            push!(asym, (
                label = "UNet",
                color = :orangered3,
                linestyle = :solid,
                jacobians = S,
            ))
        end

        out_name = @sprintf("responses_unet_%03d_%s_tmax%s.png", it, parsed.response_kind, tmax_tag)
        out_path = joinpath(out_dir, out_name)
        save_response_figure(
            out_path,
            times,
            curves;
            asymptotic_curves=asym,
            title_text=@sprintf("UNet %s response functions — %s (tmax=%.2f)", parsed.response_kind, iter_name, parsed.response_tmax),
            dpi=cfg["figures.dpi"],
        )

        push!(created, out_path)
        @info "Saved UNet response figure" iteration=it out_path
    end

    @info "Completed UNet response replot" run_dir count=length(created) out_dir response_kind=parsed.response_kind
    return created
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
