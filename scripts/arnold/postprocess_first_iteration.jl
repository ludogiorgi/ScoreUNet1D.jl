#!/usr/bin/env julia

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using HDF5
using Printf

include(joinpath(@__DIR__, "run_calibration.jl"))

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_calibration.toml")
    run_dir = ""
    iteration = 1

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
        elseif a == "--run-dir"
            i == length(args) && error("--run-dir expects a value")
            run_dir = strip(args[i + 1])
            i += 2
        elseif startswith(a, "--run-dir=")
            run_dir = strip(split(a, "=", limit=2)[2])
            i += 1
        elseif a == "--iteration"
            i == length(args) && error("--iteration expects a value")
            iteration = parse(Int, strip(args[i + 1]))
            i += 2
        elseif startswith(a, "--iteration=")
            iteration = parse(Int, strip(split(a, "=", limit=2)[2]))
            i += 1
        else
            error("Unknown argument: $a")
        end
    end

    return (
        params_path=abspath(params_path),
        run_dir=run_dir,
        iteration=iteration,
    )
end

function latest_run_dir(root::AbstractString)
    isdir(root) || error("Calibration runs root does not exist: $root")
    runs = filter(name -> startswith(name, "run_") && isdir(joinpath(root, name)), readdir(root))
    isempty(runs) && error("No calibration run directories found under $root")
    sort!(runs)
    return joinpath(root, runs[end])
end

function read_value_csv(path::AbstractString)
    isfile(path) || error("CSV file not found: $path")
    lines = readlines(path)
    length(lines) >= 2 || error("CSV file has no data rows: $path")
    values = Float64[]
    for line in lines[2:end]
        s = strip(line)
        isempty(s) && continue
        parts = split(s, ',')
        push!(values, parse(Float64, parts[end]))
    end
    return values
end

function read_iter0_observables(path::AbstractString)
    isfile(path) || error("Iteration-0 observables CSV not found: $path")
    lines = readlines(path)
    length(lines) >= 2 || error("Iteration-0 observables CSV has no data rows: $path")

    rows = lines[2:end]
    first_row = split(strip(rows[1]), ',')
    first_method = first_row[1]

    values = Float64[]
    for line in rows
        s = strip(line)
        isempty(s) && continue
        parts = split(s, ',')
        parts[1] == first_method || continue
        push!(values, parse(Float64, parts[end]))
    end
    return values
end

function read_theta_from_dataset(path::AbstractString, key::AbstractString)
    isfile(path) || error("Dataset file not found: $path")
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        attrs = HDF5.attributes(h5[key])
        return Float64[
            read(attrs["alpha0"]),
            read(attrs["alpha1"]),
            read(attrs["alpha2"]),
            read(attrs["alpha3"]),
            read(attrs["sigma"]),
        ]
    end
end

function init_postprocess_state(theta::Vector{Float64}, obs0::Vector{Float64}, observables::Dict{String,Vector{Float64}}, jacobians)
    methods = sort(collect(keys(observables)))
    return CalibrationCommon.CalibrationState(
        iteration=1,
        theta=copy(theta),
        theta_history=[copy(theta), copy(theta)],
        theta_per_method=Dict(method => [copy(theta), copy(theta)] for method in methods),
        obs_history=Dict(method => [copy(obs0), copy(observables[method])] for method in methods),
        jacobian_history=Dict(method => [copy(jacobians[method].S)] for method in methods),
        converged=false,
        convergence_metric=0.0,
    )
end

function main(args=ARGS)
    parsed = parse_args(args)
    cfg = load_calibration_config(parsed.params_path)

    run_dir = isempty(parsed.run_dir) ? latest_run_dir(cfg["paths.runs_root"]) : abspath(parsed.run_dir)
    iteration = parsed.iteration
    iter_dir = joinpath(run_dir, @sprintf("iter_%03d", iteration))
    method_unet_dir = joinpath(run_dir, "method_unet", @sprintf("iter_%03d", iteration))
    gfdt_path = joinpath(method_unet_dir, "data", "gfdt_stochastic.hdf5")
    train_path = joinpath(method_unet_dir, "data", "train_stochastic.hdf5")
    checkpoint_path = joinpath(method_unet_dir, "model", "final_checkpoint.bson")
    truth_csv = joinpath(run_dir, "truth", "target_observables.csv")
    iter0_csv = joinpath(run_dir, "truth", "observables_iter0.csv")
    truth_hdf5 = joinpath(run_dir, "truth", "truth_trajectory.hdf5")

    isfile(gfdt_path) || error("Missing saved GFDT dataset: $gfdt_path")
    isfile(train_path) || error("Missing saved train dataset: $train_path")
    isfile(checkpoint_path) || error("Missing saved UNet checkpoint: $checkpoint_path")

    theta0 = read_theta_from_dataset(gfdt_path, cfg["datasets.gfdt_key"])
    A_target = read_value_csv(truth_csv)
    obs0 = read_iter0_observables(iter0_csv)

    cfg["runtime.iteration"] = iteration
    cfg["runtime.truth_dir"] = joinpath(run_dir, "truth")
    cfg["runtime.A_target"] = copy(A_target)
    cfg["runtime.truth_matrix"] = CalibrationCommon.load_x_matrix(truth_hdf5, "x_truth", cfg["integration.K"])
    cfg["runtime.current_train_path"] = train_path
    cfg["runtime.current_checkpoint_path"] = checkpoint_path
    cfg["runtime.current_langevin_method"] = "unet"
    cfg["runtime.current_method"] = "postprocess"
    cfg["runtime.method_parallel_active"] = false
    cfg["runtime.iteration_diagnostics"] = Dict{String,Any}(
        "iteration" => iteration,
        "mode" => "postprocess_saved_iteration",
        "run_dir" => run_dir,
        "theta_reference" => copy(theta0),
        "train_data_path" => train_path,
        "gfdt_data_path" => gfdt_path,
        "checkpoint_path" => checkpoint_path,
    )

    theta_tuple = (theta0[1], theta0[2], theta0[3], theta0[4], theta0[5])
    jacobians = compute_iteration_jacobians(cfg, theta_tuple, gfdt_path, checkpoint_path, iteration, run_dir)
    observables = Dict(method => copy(jac.G) for (method, jac) in jacobians)
    cfg["runtime.primary_observables"] = copy(observables[primary_method(cfg)])
    cfg["runtime.current_observable_series"] = copy(jacobians[primary_method(cfg)].A)

    state = init_postprocess_state(theta0, obs0, observables, jacobians)
    save_iteration_outputs(state, cfg, run_dir, iteration, jacobians, observables)
    save_convergence_figure(state, cfg, run_dir)

    gap_iters = Dict{String,Vector{Int}}()
    gap_vals = Dict{String,Vector{Float64}}()
    for method in sort(collect(keys(observables)))
        gap_iters[method] = [0, iteration]
        gap_vals[method] = [observable_gap_norm(A_target, obs0), observable_gap_norm(A_target, observables[method])]
    end
    save_observable_gap_figure(joinpath(run_dir, "observable_gap_norm.png"), gap_iters, gap_vals; dpi=cfg["figures.dpi"])

    println("Postprocessed iteration $(iteration) in $(run_dir)")
    return run_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end