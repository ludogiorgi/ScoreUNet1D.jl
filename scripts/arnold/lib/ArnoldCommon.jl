module ArnoldCommon

using Dates
using FFTW
using HDF5
using LinearAlgebra
using Printf
using Random
using SHA
using Sockets
using Statistics
using TOML
using Base.Threads

export mod1idx,
    make_full_workspace,
    generate_two_scale_x_timeseries,
    generate_reduced_x_timeseries,
    save_x_dataset,
    read_dataset_signature,
    dict_signature,
    ensure_dir,
    load_data_config,
    ensure_arnold_dataset_role!,
    ensure_arnold_datasets!,
    dataset_role_spacing,
    resolve_closure_theta,
    load_role_x_matrix,
    next_run_dir,
    claim_next_run_dir,
    create_run_scaffold,
    list_checkpoints,
    wait_for_checkpoint,
    l96_reduced_step!,
    add_reduced_noise!,
    make_reduced_workspace,
    compute_observables_x,
    compute_observables_series,
    xcorr_one_sided_unbiased_fft,
    build_gfdt_response,
    step_to_impulse,
    response_output_times,
    linear_interpolate_3d_time,
    finite_rmse,
    dataset_time_spacing,
    count_effective_samples,
    read_epoch_losses,
    parse_bool,
    append_log,
    read_keyval_metrics

mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

function ensure_dir(path::AbstractString)
    mkpath(path)
    return path
end

function dict_signature(value)
    io = IOBuffer()
    TOML.print(io, value)
    return bytes2hex(sha1(take!(io)))
end

function append_log(path::AbstractString, msg::AbstractString)
    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    open(path, "a") do io
        println(io, "[$timestamp] $msg")
    end
    return nothing
end

function _path_lock_dir(path::AbstractString)
    return String(path) * ".lockdir"
end

function _path_lock_owner_file(lock_dir::AbstractString)
    return joinpath(String(lock_dir), "owner.toml")
end

function _write_lock_owner(lock_dir::AbstractString)
    owner = Dict(
        "pid" => getpid(),
        "hostname" => gethostname(),
        "created_at" => string(Dates.now()),
        "path" => abspath(dirname(String(lock_dir))),
    )
    open(_path_lock_owner_file(lock_dir), "w") do io
        TOML.print(io, owner)
    end
    return owner
end

function _pid_alive_local(pid::Integer)
    pid > 0 || return false
    return ispath(joinpath("/proc", string(pid)))
end

function _lock_is_stale(lock_dir::AbstractString; fallback_seconds::Float64=300.0)
    isdir(lock_dir) || return false

    owner_file = _path_lock_owner_file(lock_dir)
    if isfile(owner_file)
        try
            owner = TOML.parsefile(owner_file)
            pid = Int(get(owner, "pid", -1))
            host = String(get(owner, "hostname", ""))
            if host == gethostname() && !_pid_alive_local(pid)
                return true
            end
        catch
        end
    end

    # Recovery for legacy empty lock directories left behind by killed runs.
    try
        isempty(readdir(lock_dir)) || return false
    catch
        return false
    end

    age_seconds = time() - mtime(lock_dir)
    return age_seconds > fallback_seconds
end

function _clear_stale_lock!(lock_dir::AbstractString)
    _lock_is_stale(lock_dir) || return false
    try
        rm(lock_dir; recursive=true, force=true)
        return true
    catch
        return false
    end
end

function with_path_lock(path::AbstractString, f::Function; poll_seconds::Float64=0.25, timeout_seconds::Float64=14_400.0)
    lock_dir = _path_lock_dir(path)
    mkpath(dirname(lock_dir))
    t0 = time()

    while true
        try
            mkdir(lock_dir)
            _write_lock_owner(lock_dir)
            break
        catch err
            if isdir(lock_dir)
                _clear_stale_lock!(lock_dir) && continue
                if time() - t0 > timeout_seconds
                    error("Timed out waiting for filesystem lock '$lock_dir'")
                end
                sleep(poll_seconds)
                continue
            end
            rethrow(err)
        end
    end

    try
        return f()
    finally
        if isdir(lock_dir)
            rm(lock_dir; recursive=true, force=true)
        end
    end
end

function with_path_lock(f::Function, path::AbstractString; poll_seconds::Float64=0.25, timeout_seconds::Float64=14_400.0)
    return with_path_lock(path, f; poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
end

function parse_bool(x)
    if x isa Bool
        return x
    end
    s = lowercase(strip(String(x)))
    if s in ("1", "true", "yes", "y", "on")
        return true
    elseif s in ("0", "false", "no", "n", "off")
        return false
    end
    error("Cannot parse boolean value: $x")
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

const REQUIRED_ARNOLD_DATASET_ROLES = ("two_scale_observed", "train_stochastic", "gfdt_stochastic")
const OPTIONAL_ARNOLD_DATASET_ROLES = ("two_scale_fit",)
const ARNOLD_DATASET_ROLES = (REQUIRED_ARNOLD_DATASET_ROLES..., OPTIONAL_ARNOLD_DATASET_ROLES...)
const TWO_SCALE_DATASET_ROLES = ("two_scale_observed", "two_scale_fit")
const _AUTO_FIT_CACHE = Dict{String,NamedTuple{(:theta, :meta),Tuple{NTuple{5,Float64},Dict{String,Any}}}}()

_as_int(tbl::Dict{String,Any}, key::String, default) = Int(get(tbl, key, default))
_as_float(tbl::Dict{String,Any}, key::String, default) = Float64(get(tbl, key, default))
_as_str(tbl::Dict{String,Any}, key::String, default) = String(get(tbl, key, default))
_as_bool(tbl::Dict{String,Any}, key::String, default) = parse_bool(get(tbl, key, default))

function _as_table(doc::Dict{String,Any}, key::String)
    haskey(doc, key) || error("Missing [$key] table")
    doc[key] isa AbstractDict || error("[$key] must be TOML table")
    return Dict{String,Any}(doc[key])
end

function _as_subtable(tbl::Dict{String,Any}, key::String)
    haskey(tbl, key) || error("Missing nested table [$key]")
    tbl[key] isa AbstractDict || error("[$key] must be TOML table")
    return Dict{String,Any}(tbl[key])
end

function _maybe_subtable(tbl::Dict{String,Any}, key::String)
    if !haskey(tbl, key)
        return Dict{String,Any}()
    end
    tbl[key] isa AbstractDict || error("[$key] must be TOML table")
    return Dict{String,Any}(tbl[key])
end

function _check_role(role::AbstractString)
    role_s = String(role)
    role_s in ARNOLD_DATASET_ROLES || error("Unknown dataset role '$role_s'. Valid roles: $(join(ARNOLD_DATASET_ROLES, ", "))")
    return role_s
end

function load_data_config(path::AbstractString)
    isfile(path) || error("Data parameter file not found: $path")
    doc = TOML.parsefile(path)

    paths = _as_table(doc, "paths")
    twoscale = _as_table(doc, "twoscale")
    closure = _as_table(doc, "closure")
    datasets = _as_table(doc, "datasets")

    legacy_process_noise_sigma = _as_float(twoscale, "process_noise_sigma", 0.0)
    legacy_stochastic_x_noise = _as_bool(twoscale, "stochastic_x_noise", false)
    process_noise_sigma_y = _as_float(twoscale, "process_noise_sigma_y", legacy_process_noise_sigma)
    process_noise_sigma_x = _as_float(twoscale, "process_noise_sigma_x", legacy_stochastic_x_noise ? legacy_process_noise_sigma : 0.0)

    cfg = Dict{String,Any}(
        "paths.parameters_data_path" => abspath(path),
        "paths.datasets_hdf5" => abspath(_as_str(paths, "datasets_hdf5", "scripts/arnold/data/l96_arnold_datasets.hdf5")),

        "twoscale.K" => _as_int(twoscale, "K", 8),
        "twoscale.J" => _as_int(twoscale, "J", 32),
        "twoscale.F" => _as_float(twoscale, "F", 20.0),
        "twoscale.h" => _as_float(twoscale, "h", 1.0),
        "twoscale.c" => _as_float(twoscale, "c", 10.0),
        "twoscale.b" => _as_float(twoscale, "b", 10.0),
        "twoscale.dt" => _as_float(twoscale, "dt", 0.005),
        "twoscale.process_noise_sigma" => process_noise_sigma_y,
        "twoscale.process_noise_sigma_y" => process_noise_sigma_y,
        "twoscale.process_noise_sigma_x" => process_noise_sigma_x,
        "twoscale.stochastic_x_noise" => process_noise_sigma_x > 0.0,

        "closure.F" => _as_float(closure, "F", _as_float(twoscale, "F", 20.0)),
        "closure.alpha0_initial" => _as_float(closure, "alpha0_initial", _as_float(closure, "alpha0", 0.0)),
        "closure.alpha1_initial" => _as_float(closure, "alpha1_initial", _as_float(closure, "alpha1", 0.0)),
        "closure.alpha2_initial" => _as_float(closure, "alpha2_initial", _as_float(closure, "alpha2", 0.0)),
        "closure.alpha3_initial" => _as_float(closure, "alpha3_initial", _as_float(closure, "alpha3", 0.0)),
        "closure.sigma_initial" => _as_float(closure, "sigma_initial", _as_float(closure, "sigma", 1.0)),
        "closure.auto_fit" => _as_bool(closure, "auto_fit", true),
        "closure.fit_dataset_role" => _as_str(closure, "fit_dataset_role", "two_scale_fit"),
        "closure.fit_start_index" => _as_int(closure, "fit_start_index", 2),
        "closure.fit_samples" => _as_int(closure, "fit_samples", 50_000),
        "closure.fit_min_samples" => _as_int(closure, "fit_min_samples", 500),
    )

    for role in REQUIRED_ARNOLD_DATASET_ROLES
        tbl = _as_subtable(datasets, role)
        cfg["datasets.$role.key"] = _as_str(tbl, "key", role)
        cfg["datasets.$role.spinup_steps"] = _as_int(tbl, "spinup_steps", 50_000)
        cfg["datasets.$role.save_every"] = _as_int(tbl, "save_every", 1)
        cfg["datasets.$role.nsamples"] = _as_int(tbl, "nsamples", 100_000)
        cfg["datasets.$role.rng_seed"] = _as_int(tbl, "rng_seed", 11)
        cfg["datasets.$role.target_spacing"] = _as_float(tbl, "target_spacing", dataset_time_spacing(cfg["twoscale.dt"], cfg["datasets.$role.save_every"]))
        cfg["datasets.$role.ensemble_trajectories"] = _as_int(tbl, "ensemble_trajectories", 1)
        cfg["datasets.$role.parallel_trajectories"] = _as_bool(tbl, "parallel_trajectories", false)
    end

    fit_nsamples_default = max(cfg["closure.fit_start_index"] + cfg["closure.fit_samples"], 50_000)
    for role in OPTIONAL_ARNOLD_DATASET_ROLES
        tbl = _maybe_subtable(datasets, role)
        cfg["datasets.$role.key"] = _as_str(tbl, "key", role)
        cfg["datasets.$role.spinup_steps"] = _as_int(tbl, "spinup_steps", cfg["datasets.two_scale_observed.spinup_steps"])
        cfg["datasets.$role.save_every"] = _as_int(tbl, "save_every", 2)
        cfg["datasets.$role.nsamples"] = _as_int(tbl, "nsamples", fit_nsamples_default)
        cfg["datasets.$role.rng_seed"] = _as_int(tbl, "rng_seed", cfg["datasets.two_scale_observed.rng_seed"])
        cfg["datasets.$role.target_spacing"] = _as_float(tbl, "target_spacing", dataset_time_spacing(cfg["twoscale.dt"], cfg["datasets.$role.save_every"]))
        cfg["datasets.$role.ensemble_trajectories"] = _as_int(tbl, "ensemble_trajectories", 1)
        cfg["datasets.$role.parallel_trajectories"] = _as_bool(tbl, "parallel_trajectories", false)
    end

    cfg["twoscale.K"] >= 2 || error("twoscale.K must be >= 2")
    cfg["twoscale.J"] >= 1 || error("twoscale.J must be >= 1")
    cfg["twoscale.dt"] > 0 || error("twoscale.dt must be > 0")
    cfg["closure.fit_dataset_role"] = _check_role(cfg["closure.fit_dataset_role"])
    cfg["closure.fit_dataset_role"] in TWO_SCALE_DATASET_ROLES || error("closure.fit_dataset_role must reference a deterministic two-scale dataset role")
    cfg["closure.fit_samples"] >= 10 || error("closure.fit_samples must be >= 10")
    cfg["closure.fit_min_samples"] >= 10 || error("closure.fit_min_samples must be >= 10")

    for role in ARNOLD_DATASET_ROLES
        cfg["datasets.$role.spinup_steps"] >= 0 || error("datasets.$role.spinup_steps must be >= 0")
        cfg["datasets.$role.save_every"] >= 1 || error("datasets.$role.save_every must be >= 1")
        cfg["datasets.$role.nsamples"] >= 2 || error("datasets.$role.nsamples must be >= 2")
        cfg["datasets.$role.ensemble_trajectories"] >= 1 || error("datasets.$role.ensemble_trajectories must be >= 1")
    end

    return cfg, doc
end

function dataset_role_spacing(cfg::Dict{String,Any}, role::AbstractString)
    role_s = _check_role(role)
    return dataset_time_spacing(cfg["twoscale.dt"], cfg["datasets.$role_s.save_every"])
end

function _dataset_role_signature(cfg::Dict{String,Any}, role::String; closure_theta::Union{Nothing,NTuple{5,Float64}}=nothing)
    role_s = _check_role(role)
    base = Dict{String,Any}(
        "equations_version" => "two_scale_l96_hc_over_J",
        "role" => role_s,
        "dataset_key" => cfg["datasets.$role_s.key"],
        "spinup_steps" => cfg["datasets.$role_s.spinup_steps"],
        "save_every" => cfg["datasets.$role_s.save_every"],
        "nsamples" => cfg["datasets.$role_s.nsamples"],
        "rng_seed" => cfg["datasets.$role_s.rng_seed"],
        "ensemble_trajectories" => cfg["datasets.$role_s.ensemble_trajectories"],
        "parallel_trajectories" => cfg["datasets.$role_s.parallel_trajectories"],
        "twoscale_dt" => cfg["twoscale.dt"],
        "twoscale_K" => cfg["twoscale.K"],
    )

    if role_s in TWO_SCALE_DATASET_ROLES
        base["model"] = "two_scale_deterministic"
        base["twoscale"] = Dict(
            "J" => cfg["twoscale.J"],
            "F" => cfg["twoscale.F"],
            "h" => cfg["twoscale.h"],
            "c" => cfg["twoscale.c"],
            "b" => cfg["twoscale.b"],
            "process_noise_sigma_y" => cfg["twoscale.process_noise_sigma_y"],
            "process_noise_sigma_x" => cfg["twoscale.process_noise_sigma_x"],
        )
    else
        base["model"] = "reduced_stochastic"
        base["reduced"] = Dict(
            "F" => cfg["closure.F"],
            "alpha0" => closure_theta === nothing ? cfg["closure.alpha0_initial"] : closure_theta[1],
            "alpha1" => closure_theta === nothing ? cfg["closure.alpha1_initial"] : closure_theta[2],
            "alpha2" => closure_theta === nothing ? cfg["closure.alpha2_initial"] : closure_theta[3],
            "alpha3" => closure_theta === nothing ? cfg["closure.alpha3_initial"] : closure_theta[4],
            "sigma" => closure_theta === nothing ? cfg["closure.sigma_initial"] : closure_theta[5],
            "auto_fit" => cfg["closure.auto_fit"],
        )
    end

    return dict_signature(base)
end

function _load_role_bounds(ds, nsamples::Int, start_index::Int, label::AbstractString)
    n_total = size(ds, 1)
    n_total >= 2 || error("Dataset '$label' has too few samples: $n_total")
    n_use = min(max(nsamples, 2), n_total)
    max_start = max(1, n_total - n_use + 1)
    s_use = clamp(start_index, 1, max_start)
    e_use = s_use + n_use - 1
    if n_use != nsamples || s_use != start_index
        @warn "Adjusted subset bounds" subset = label requested_nsamples = nsamples used_nsamples = n_use requested_start = start_index used_start = s_use total = n_total
    end
    return s_use, e_use
end

function load_role_x_matrix(cfg::Dict{String,Any}, role::AbstractString;
    nsamples::Int=0,
    start_index::Int=1,
    label::AbstractString=String(role))
    role_s = _check_role(role)
    ensure_arnold_dataset_role!(cfg, role_s)
    path = cfg["paths.datasets_hdf5"]
    key = cfg["datasets.$role_s.key"]
    return with_path_lock(path) do
        h5open(path, "r") do h5
            haskey(h5, key) || error("Dataset key '$key' not found in $path")
            ds = h5[key]
            n_total = size(ds, 1)
            n_req = nsamples <= 0 ? n_total : nsamples
            s_use, e_use = _load_role_bounds(ds, n_req, start_index, label)
            raw = Float64.(ds[s_use:e_use, :])
            permutedims(raw, (2, 1))
        end
    end
end

function _fit_polynomial_closure(X::Matrix{Float64},
    dt_obs::Float64,
    F::Float64;
    fit_samples::Int,
    fit_min_samples::Int)
    K, N = size(X)
    N >= 3 || error("Need at least 3 samples to fit closure; got $N")
    ns = min(fit_samples, N - 2)
    ns >= fit_min_samples || error("Not enough samples for closure fit: need >= $fit_min_samples, got $ns")

    start = 2
    stop = start + ns - 1

    A = Array{Float64}(undef, ns * K, 4)
    bvec = Array{Float64}(undef, ns * K)
    idx = 1
    @inbounds for t in start:stop
        for k in 1:K
            km2 = mod1idx(k - 2, K)
            km1 = mod1idx(k - 1, K)
            kp1 = mod1idx(k + 1, K)

            xm = X[k, t - 1]
            xp = X[k, t + 1]
            xk = X[k, t]
            xkm2 = X[km2, t]
            xkm1 = X[km1, t]
            xkp1 = X[kp1, t]

            xdot = (xp - xm) / (2 * dt_obs)
            adv = -xkm1 * (xkm2 - xkp1)
            y = F + adv - xk - xdot

            A[idx, 1] = 1.0
            A[idx, 2] = xk
            A[idx, 3] = xk * xk
            A[idx, 4] = xk * xk * xk
            bvec[idx] = y
            idx += 1
        end
    end

    coeff = A \ bvec
    resid = bvec .- A * coeff
    resid_mat = reshape(resid, K, ns)
    sigma_fit, sigma_meta = _estimate_effective_sigma_green_kubo(resid_mat, dt_obs)
    theta_fit = (coeff[1], coeff[2], coeff[3], coeff[4], max(sigma_fit, 1e-8))
    fit_meta = Dict{String,Any}(
        "fit_window_start" => start,
        "fit_window_stop" => stop,
        "fit_window_samples" => ns,
        "fit_dt_obs" => dt_obs,
        "sigma_estimator" => sigma_meta,
    )
    return theta_fit, fit_meta
end

function _estimate_effective_sigma_green_kubo(resid_mat::Matrix{Float64}, dt_obs::Float64)
    K, N = size(resid_mat)
    N >= 2 || error("Need at least two residual samples to estimate sigma")
    dt_obs > 0 || error("dt_obs must be > 0")

    max_lag = min(N - 1, max(1, Int(floor(5.0 / dt_obs))))
    acov = zeros(Float64, max_lag + 1)

    for k in 1:K
        rk = Float64.(view(resid_mat, k, :))
        rk_center = rk .- mean(rk)
        acov .+= xcorr_one_sided_unbiased_fft(rk_center, rk_center, max_lag)
    end
    acov ./= K

    cutoff = max_lag
    for lag in 1:max_lag
        if acov[lag + 1] <= 0.0
            cutoff = lag
            break
        end
    end

    integral = 0.5 * acov[1]
    if cutoff >= 1
        if cutoff > 1
            integral += sum(acov[2:cutoff])
        end
        integral += 0.5 * acov[cutoff + 1]
    end
    integral *= dt_obs

    sigma2 = 2.0 * max(integral, 0.0)
    sigma = sqrt(sigma2)
    sigma_fallback = std(vec(resid_mat)) * sqrt(dt_obs)
    mode = "green_kubo"
    if !(isfinite(sigma) && sigma > 1e-10)
        sigma = max(sigma_fallback, 1e-8)
        sigma2 = sigma^2
        mode = "std_fallback"
    end

    meta = Dict{String,Any}(
        "mode" => mode,
        "dt_obs" => dt_obs,
        "max_lag" => max_lag,
        "cutoff_lag" => cutoff,
        "cutoff_time" => cutoff * dt_obs,
        "integrated_autocovariance" => integral,
        "sigma_fallback" => sigma_fallback,
        "lag0_covariance" => acov[1],
    )
    return sigma, meta
end

function resolve_closure_theta(cfg::Dict{String,Any}; force_refit::Bool=false)
    theta_initial = (
        cfg["closure.alpha0_initial"],
        cfg["closure.alpha1_initial"],
        cfg["closure.alpha2_initial"],
        cfg["closure.alpha3_initial"],
        cfg["closure.sigma_initial"],
    )

    if !cfg["closure.auto_fit"]
        meta = Dict{String,Any}(
            "mode" => "initial_only",
            "theta_initial" => collect(theta_initial),
            "theta_used" => collect(theta_initial),
            "fit_dataset_role" => "",
            "fit_start_index" => 0,
            "fit_samples" => 0,
        )
        return theta_initial, meta
    end

    fit_role = cfg["closure.fit_dataset_role"]
    fit_required_total = cfg["closure.fit_start_index"] + cfg["closure.fit_samples"]
    fit_dataset_nsamples_key = "datasets.$fit_role.nsamples"
    cfg[fit_dataset_nsamples_key] = max(Int(cfg[fit_dataset_nsamples_key]), fit_required_total)
    ensure_arnold_dataset_role!(cfg, fit_role)
    role_sig = _dataset_role_signature(cfg, fit_role)
    cache_key = dict_signature(Dict(
        "fit_role" => fit_role,
        "fit_role_signature" => role_sig,
        "fit_start_index" => cfg["closure.fit_start_index"],
        "fit_samples" => cfg["closure.fit_samples"],
        "fit_min_samples" => cfg["closure.fit_min_samples"],
        "F" => cfg["closure.F"],
        "theta_initial" => collect(theta_initial),
    ))

    if !force_refit && haskey(_AUTO_FIT_CACHE, cache_key)
        cached = _AUTO_FIT_CACHE[cache_key]
        return cached.theta, copy(cached.meta)
    end

    fit_load_start = max(1, cfg["closure.fit_start_index"] - 1)
    fit_load_nsamples = cfg["closure.fit_samples"] + 2
    Xfit = load_role_x_matrix(
        cfg,
        fit_role;
        nsamples=fit_load_nsamples,
        start_index=fit_load_start,
        label="closure_auto_fit",
    )
    dt_obs = dataset_role_spacing(cfg, fit_role)
    theta_fit, fit_meta = _fit_polynomial_closure(
        Xfit,
        dt_obs,
        cfg["closure.F"];
        fit_samples=cfg["closure.fit_samples"],
        fit_min_samples=cfg["closure.fit_min_samples"],
    )

    meta = Dict{String,Any}(
        "mode" => "auto_fit",
        "theta_initial" => collect(theta_initial),
        "theta_used" => collect(theta_fit),
        "fit_dataset_role" => fit_role,
        "fit_start_index" => cfg["closure.fit_start_index"],
        "fit_samples" => cfg["closure.fit_samples"],
        "fit_dataset_load_start_index" => fit_load_start,
        "fit_dataset_load_samples" => fit_load_nsamples,
        "fit_dt_obs" => dt_obs,
        "fit_details" => fit_meta,
    )
    _AUTO_FIT_CACHE[cache_key] = (theta=theta_fit, meta=copy(meta))
    return theta_fit, meta
end

function ensure_arnold_dataset_role!(cfg::Dict{String,Any}, role::AbstractString)
    role_s = _check_role(role)
    path = cfg["paths.datasets_hdf5"]
    key = cfg["datasets.$role_s.key"]

    closure_theta = nothing
    closure_meta = Dict{String,Any}()
    if !(role_s in TWO_SCALE_DATASET_ROLES)
        closure_theta, closure_meta = resolve_closure_theta(cfg)
    end

    sig = _dataset_role_signature(cfg, role_s; closure_theta=closure_theta)

    return with_path_lock(path) do
        found_sig = read_dataset_signature(path, key)
        if isfile(path) && found_sig == sig
            return Dict{String,Any}(
                "role" => role_s,
                "path" => path,
                "key" => key,
                "signature" => sig,
                "generated" => false,
            )
        end

        data = if role_s in TWO_SCALE_DATASET_ROLES
            generate_two_scale_x_timeseries_ensemble(
                K=cfg["twoscale.K"],
                J=cfg["twoscale.J"],
                F=cfg["twoscale.F"],
                h=cfg["twoscale.h"],
                c=cfg["twoscale.c"],
                b=cfg["twoscale.b"],
                dt=cfg["twoscale.dt"],
                spinup_steps=cfg["datasets.$role_s.spinup_steps"],
                save_every=cfg["datasets.$role_s.save_every"],
                nsamples=cfg["datasets.$role_s.nsamples"],
                rng_seed=cfg["datasets.$role_s.rng_seed"],
                process_noise_sigma_y=cfg["twoscale.process_noise_sigma_y"],
                process_noise_sigma_x=cfg["twoscale.process_noise_sigma_x"],
                ensemble_trajectories=cfg["datasets.$role_s.ensemble_trajectories"],
                parallel_trajectories=cfg["datasets.$role_s.parallel_trajectories"],
            )
        else
            theta = closure_theta::NTuple{5,Float64}
            generate_reduced_x_timeseries_ensemble(
                K=cfg["twoscale.K"],
                F=cfg["closure.F"],
                alpha0=theta[1],
                alpha1=theta[2],
                alpha2=theta[3],
                alpha3=theta[4],
                sigma=theta[5],
                dt=cfg["twoscale.dt"],
                spinup_steps=cfg["datasets.$role_s.spinup_steps"],
                save_every=cfg["datasets.$role_s.save_every"],
                nsamples=cfg["datasets.$role_s.nsamples"],
                rng_seed=cfg["datasets.$role_s.rng_seed"],
                max_abs_state=1e4,
                state_min=get(cfg, "numerical.state_min", -Inf),
                state_max=get(cfg, "numerical.state_max", Inf),
                max_boundary_hits=Int(get(cfg, "numerical.max_boundary_hits", 40)),
                ensemble_trajectories=cfg["datasets.$role_s.ensemble_trajectories"],
                parallel_trajectories=cfg["datasets.$role_s.parallel_trajectories"],
            )
        end

        all(isfinite, data) || error("Generated dataset role '$role_s' contains non-finite values")

        attrs = Dict{String,Any}(
            "role" => role_s,
            "model" => role_s in TWO_SCALE_DATASET_ROLES ? "two_scale_deterministic" : "reduced_stochastic",
            "dt" => cfg["twoscale.dt"],
            "save_every" => cfg["datasets.$role_s.save_every"],
            "nsamples" => cfg["datasets.$role_s.nsamples"],
            "spinup_steps" => cfg["datasets.$role_s.spinup_steps"],
            "rng_seed" => cfg["datasets.$role_s.rng_seed"],
            "params_signature" => sig,
            "generated_at" => string(now()),
            "source_parameters_data" => cfg["paths.parameters_data_path"],
        )

        if role_s in TWO_SCALE_DATASET_ROLES
            attrs["K"] = cfg["twoscale.K"]
            attrs["J"] = cfg["twoscale.J"]
            attrs["F"] = cfg["twoscale.F"]
            attrs["h"] = cfg["twoscale.h"]
            attrs["c"] = cfg["twoscale.c"]
            attrs["b"] = cfg["twoscale.b"]
            attrs["process_noise_sigma"] = cfg["twoscale.process_noise_sigma_y"]
            attrs["stochastic_x_noise"] = cfg["twoscale.process_noise_sigma_x"] > 0.0
            attrs["process_noise_sigma_y"] = cfg["twoscale.process_noise_sigma_y"]
            attrs["process_noise_sigma_x"] = cfg["twoscale.process_noise_sigma_x"]
        else
            theta = closure_theta::NTuple{5,Float64}
            attrs["K"] = cfg["twoscale.K"]
            attrs["F"] = cfg["closure.F"]
            attrs["alpha0"] = theta[1]
            attrs["alpha1"] = theta[2]
            attrs["alpha2"] = theta[3]
            attrs["alpha3"] = theta[4]
            attrs["sigma"] = theta[5]
            attrs["closure_auto_fit"] = cfg["closure.auto_fit"]
            attrs["closure_fit_mode"] = get(closure_meta, "mode", "")
            attrs["closure_fit_dataset_role"] = get(closure_meta, "fit_dataset_role", "")
        end
        attrs["ensemble_trajectories"] = cfg["datasets.$role_s.ensemble_trajectories"]
        attrs["parallel_trajectories"] = cfg["datasets.$role_s.parallel_trajectories"]

        save_x_dataset(path, key, data, attrs)
        return Dict{String,Any}(
            "role" => role_s,
            "path" => path,
            "key" => key,
            "signature" => sig,
            "generated" => true,
        )
    end
end

function ensure_arnold_datasets!(cfg::Dict{String,Any}; roles::Vector{String}=String[ARNOLD_DATASET_ROLES...])
    out = Dict{String,Dict{String,Any}}()
    for role in roles
        out[role] = ensure_arnold_dataset_role!(cfg, role)
    end
    return out
end

function ensure_arnold_datasets!(data_params_path::AbstractString; roles::Vector{String}=String[ARNOLD_DATASET_ROLES...])
    cfg, _ = load_data_config(data_params_path)
    return ensure_arnold_datasets!(cfg; roles=roles)
end

# -----------------------------------------------------------------------------
# Run management
# -----------------------------------------------------------------------------

function _parse_run_num(name::AbstractString)
    m = match(r"^run_(\d+)$", String(name))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function next_run_dir(root::AbstractString; pad::Int=3)
    mkpath(root)
    max_id = 0
    for name in readdir(root)
        n = _parse_run_num(name)
        n === nothing && continue
        max_id = max(max_id, n)
    end
    run_id = max_id + 1
    run_name = "run_" * lpad(string(run_id), pad, '0')
    run_dir = joinpath(root, run_name)
    return (run_dir=run_dir, run_id=run_id, run_name=run_name)
end

function claim_next_run_dir(root::AbstractString; pad::Int=3, max_attempts::Int=10_000)
    run_info = next_run_dir(root; pad=pad)
    start_id = run_info.run_id - 1

    for offset in 1:max_attempts
        run_id = start_id + offset
        run_name = "run_" * lpad(string(run_id), pad, '0')
        run_dir = joinpath(root, run_name)
        try
            mkdir(run_dir)
            return (run_dir=run_dir, run_id=run_id, run_name=run_name)
        catch err
            if isdir(run_dir)
                continue
            end
            rethrow(err)
        end
    end

    error("Could not reserve a unique run directory in '$root' after $max_attempts attempts")
end

function create_run_scaffold(run_dir::AbstractString)
    dirs = Dict(
        "run" => run_dir,
        "model" => joinpath(run_dir, "model"),
        "metrics" => joinpath(run_dir, "metrics"),
        "figures" => joinpath(run_dir, "figures"),
        "figures_training" => joinpath(run_dir, "figures", "training"),
        "logs" => joinpath(run_dir, "logs"),
        "params" => joinpath(run_dir, "params"),
        "cache" => joinpath(run_dir, "cache"),
    )
    for d in values(dirs)
        mkpath(d)
    end
    return dirs
end

function list_checkpoints(checkpoint_dir::AbstractString)
    isdir(checkpoint_dir) || return Tuple{Int,String}[]
    out = Tuple{Int,String}[]
    for name in readdir(checkpoint_dir)
        m = match(r"^epoch_(\d+)\.bson$", name)
        m === nothing && continue
        push!(out, (parse(Int, m.captures[1]), joinpath(checkpoint_dir, name)))
    end
    sort!(out; by=x -> x[1])
    return out
end

function _wait_file_stable(path::AbstractString; checks::Int=3, sleep_seconds::Float64=0.5)
    prev_size = -1
    stable = 0
    while stable < checks
        if isfile(path)
            sz = filesize(path)
            if sz > 0 && sz == prev_size
                stable += 1
            else
                stable = 0
            end
            prev_size = sz
        else
            stable = 0
            prev_size = -1
        end
        sleep(sleep_seconds)
    end
    return path
end

function wait_for_checkpoint(checkpoint_dir::AbstractString, epoch::Int;
    train_process=nothing,
    poll_seconds::Float64=2.0)
    while true
        for (ep, path) in list_checkpoints(checkpoint_dir)
            ep == epoch || continue
            _wait_file_stable(path)
            return path
        end

        if train_process !== nothing && Base.process_exited(train_process)
            wait(train_process)
            for (ep, path) in list_checkpoints(checkpoint_dir)
                ep == epoch || continue
                _wait_file_stable(path)
                return path
            end
            error("Training process exited before checkpoint epoch=$epoch was written")
        end

        sleep(poll_seconds)
    end
end

# -----------------------------------------------------------------------------
# Full two-scale L96 dynamics (observations generator)
# -----------------------------------------------------------------------------

function make_full_workspace(K::Int, J::Int)
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
    )
end

function l96_two_scale_drift!(dx::AbstractVector{Float64},
    dy::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    y::AbstractMatrix{Float64},
    K::Int,
    J::Int,
    F::Float64,
    h::Float64,
    c::Float64,
    b::Float64)
    coupling = h * c / J

    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)
        dx[k] = -x[km1] * (x[km2] - x[kp1]) - x[k] + F - coupling * sum(@view y[:, k])
    end

    @inbounds for k in 1:K
        xk_term = coupling * x[k]
        for j in 1:J
            jm1 = mod1idx(j - 1, J)
            jp1 = mod1idx(j + 1, J)
            jp2 = mod1idx(j + 2, J)
            dy[j, k] = -c * b * y[jp1, k] * (y[jp2, k] - y[jm1, k]) - c * y[j, k] + xk_term
        end
    end

    return nothing
end

function rk4_full_step!(x::Vector{Float64},
    y::Matrix{Float64},
    dt::Float64,
    ws,
    K::Int,
    J::Int,
    F::Float64,
    h::Float64,
    c::Float64,
    b::Float64)
    l96_two_scale_drift!(ws.dx1, ws.dy1, x, y, K, J, F, h, c, b)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    l96_two_scale_drift!(ws.dx2, ws.dy2, ws.xtmp, ws.ytmp, K, J, F, h, c, b)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    l96_two_scale_drift!(ws.dx3, ws.dy3, ws.xtmp, ws.ytmp, K, J, F, h, c, b)

    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    l96_two_scale_drift!(ws.dx4, ws.dy4, ws.xtmp, ws.ytmp, K, J, F, h, c, b)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    return nothing
end

function add_full_process_noise!(x::Vector{Float64},
    y::Matrix{Float64},
    rng::AbstractRNG,
    sigma_y::Float64,
    sigma_x::Float64,
    dt::Float64)
    sigma_y <= 0 && sigma_x <= 0 && return nothing
    step_sigma_y = sigma_y * sqrt(dt)
    step_sigma_x = sigma_x * sqrt(dt)
    @inbounds begin
        if step_sigma_x > 0.0
            for k in eachindex(x)
                x[k] += step_sigma_x * randn(rng)
            end
        end
        if step_sigma_y > 0.0
            for idx in eachindex(y)
                y[idx] += step_sigma_y * randn(rng)
            end
        end
    end
    return nothing
end

function generate_two_scale_x_timeseries(;K::Int,
    J::Int,
    F::Float64,
    h::Float64,
    c::Float64,
    b::Float64,
    dt::Float64,
    spinup_steps::Int,
    save_every::Int,
    nsamples::Int,
    rng_seed::Int,
    process_noise_sigma::Union{Nothing,Float64}=nothing,
    process_noise_sigma_y::Union{Nothing,Float64}=nothing,
    process_noise_sigma_x::Union{Nothing,Float64}=nothing,
    stochastic_x_noise::Bool=false)
    sigma_y = process_noise_sigma_y === nothing ? (process_noise_sigma === nothing ? 0.0 : process_noise_sigma) : process_noise_sigma_y
    sigma_x = process_noise_sigma_x === nothing ? ((process_noise_sigma === nothing || !stochastic_x_noise) ? 0.0 : process_noise_sigma) : process_noise_sigma_x
    rng = MersenneTwister(rng_seed)
    x = F .+ 0.01 .* randn(rng, Float64, K)
    y = 0.01 .* randn(rng, Float64, J, K)
    ws = make_full_workspace(K, J)

    for _ in 1:spinup_steps
        rk4_full_step!(x, y, dt, ws, K, J, F, h, c, b)
        add_full_process_noise!(x, y, rng, sigma_y, sigma_x, dt)
    end

    out = Array{Float32}(undef, nsamples, K)
    for n in 1:nsamples
        for _ in 1:save_every
            rk4_full_step!(x, y, dt, ws, K, J, F, h, c, b)
            add_full_process_noise!(x, y, rng, sigma_y, sigma_x, dt)
        end
        @inbounds out[n, :] .= Float32.(x)
    end
    return out
end

function generate_two_scale_x_timeseries_ensemble(;K::Int,
    J::Int,
    F::Float64,
    h::Float64,
    c::Float64,
    b::Float64,
    dt::Float64,
    spinup_steps::Int,
    save_every::Int,
    nsamples::Int,
    rng_seed::Int,
    process_noise_sigma::Union{Nothing,Float64}=nothing,
    process_noise_sigma_y::Union{Nothing,Float64}=nothing,
    process_noise_sigma_x::Union{Nothing,Float64}=nothing,
    stochastic_x_noise::Bool=false,
    ensemble_trajectories::Int=1,
    parallel_trajectories::Bool=false)
    ensemble_trajectories >= 1 || error("ensemble_trajectories must be >= 1")
    if ensemble_trajectories == 1
        return generate_two_scale_x_timeseries(
            K=K,
            J=J,
            F=F,
            h=h,
            c=c,
            b=b,
            dt=dt,
            spinup_steps=spinup_steps,
            save_every=save_every,
            nsamples=nsamples,
            rng_seed=rng_seed,
            process_noise_sigma=process_noise_sigma,
            process_noise_sigma_y=process_noise_sigma_y,
            process_noise_sigma_x=process_noise_sigma_x,
            stochastic_x_noise=stochastic_x_noise,
        )
    end

    counts = fill(nsamples ÷ ensemble_trajectories, ensemble_trajectories)
    for i in 1:(nsamples % ensemble_trajectories)
        counts[i] += 1
    end
    parts = Vector{Matrix{Float32}}(undef, ensemble_trajectories)
    errors = fill("", ensemble_trajectories)
    seed_stride = 1_000_000

    run_one! = function (itr::Int)
        try
            parts[itr] = generate_two_scale_x_timeseries(
                K=K,
                J=J,
                F=F,
                h=h,
                c=c,
                b=b,
                dt=dt,
                spinup_steps=spinup_steps,
                save_every=save_every,
                nsamples=counts[itr],
                rng_seed=rng_seed + seed_stride * (itr - 1),
                process_noise_sigma=process_noise_sigma,
                process_noise_sigma_y=process_noise_sigma_y,
                process_noise_sigma_x=process_noise_sigma_x,
                stochastic_x_noise=stochastic_x_noise,
            )
        catch err
            errors[itr] = sprint(showerror, err)
        end
        return nothing
    end

    if parallel_trajectories && nthreads() > 1
        Threads.@threads for itr in 1:ensemble_trajectories
            run_one!(itr)
        end
    else
        for itr in 1:ensemble_trajectories
            run_one!(itr)
        end
    end

    for itr in 1:ensemble_trajectories
        isempty(errors[itr]) || error("Two-scale ensemble trajectory $(itr) failed: $(errors[itr])")
    end

    out = Array{Float32}(undef, nsamples, K)
    pos = 1
    for part in parts
        nrows = size(part, 1)
        @views out[pos:(pos + nrows - 1), :] .= part
        pos += nrows
    end
    return out
end

function generate_reduced_x_timeseries(;K::Int,
    F::Float64,
    alpha0::Float64,
    alpha1::Float64,
    alpha2::Float64,
    alpha3::Float64,
    sigma::Float64,
    dt::Float64,
    spinup_steps::Int,
    save_every::Int,
    nsamples::Int,
    rng_seed::Int,
    max_abs_state::Float64=1e4,
    max_restarts::Int=8,
    state_min::Float64=-Inf,
    state_max::Float64=Inf,
    max_boundary_hits::Int=40)
    max_abs_state > 0 || error("max_abs_state must be > 0")
    max_restarts >= 0 || error("max_restarts must be >= 0")
    state_min <= state_max || error("state_min must be <= state_max")
    max_boundary_hits >= 1 || error("max_boundary_hits must be >= 1")

    use_bounds = isfinite(state_min) || isfinite(state_max)
    boundary_hits = 0

    out = Array{Float32}(undef, nsamples, K)
    seed_stride = 10_000

    for attempt in 0:max_restarts
        seed = rng_seed + attempt * seed_stride
        rng = MersenneTwister(seed)
        x = F .+ 0.01 .* randn(rng, Float64, K)
        ws = make_reduced_workspace(K)
        stable = true
        hit_boundary = false

        for _ in 1:spinup_steps
            l96_reduced_step!(x, dt, F, alpha0, alpha1, alpha2, alpha3, ws)
            add_reduced_noise!(x, rng, sigma, dt)
            if use_bounds
                if any(v -> (v <= state_min || v >= state_max), x)
                    boundary_hits += 1
                    hit_boundary = true
                    stable = false
                    break
                end
            end
            if !(all(isfinite, x) && maximum(abs, x) <= max_abs_state)
                stable = false
                break
            end
        end

        if stable
            for n in 1:nsamples
                for _ in 1:save_every
                    l96_reduced_step!(x, dt, F, alpha0, alpha1, alpha2, alpha3, ws)
                    add_reduced_noise!(x, rng, sigma, dt)
                    if use_bounds
                        if any(v -> (v <= state_min || v >= state_max), x)
                            boundary_hits += 1
                            hit_boundary = true
                            stable = false
                            break
                        end
                    end
                    if !(all(isfinite, x) && maximum(abs, x) <= max_abs_state)
                        stable = false
                        break
                    end
                end
                stable || break
                @inbounds out[n, :] .= Float32.(x)
            end
        end

        stable && return out
        if hit_boundary && boundary_hits >= max_boundary_hits
            error("Reduced trajectory hit bounds [$(state_min), $(state_max)] $(boundary_hits) times (limit=$(max_boundary_hits), base rng_seed=$(rng_seed))")
        end
        attempt < max_restarts && @warn "Reduced trajectory became unstable; retrying with shifted seed" base_seed = rng_seed retry_seed = seed + seed_stride attempt = attempt + 1
    end

    error("Failed to generate stable reduced trajectory after $(max_restarts + 1) attempts (base rng_seed=$rng_seed)")
end

function generate_reduced_x_timeseries_ensemble(;K::Int,
    F::Float64,
    alpha0::Float64,
    alpha1::Float64,
    alpha2::Float64,
    alpha3::Float64,
    sigma::Float64,
    dt::Float64,
    spinup_steps::Int,
    save_every::Int,
    nsamples::Int,
    rng_seed::Int,
    max_abs_state::Float64=1e4,
    max_restarts::Int=8,
    state_min::Float64=-Inf,
    state_max::Float64=Inf,
    max_boundary_hits::Int=40,
    ensemble_trajectories::Int=1,
    parallel_trajectories::Bool=false)
    ensemble_trajectories >= 1 || error("ensemble_trajectories must be >= 1")
    if ensemble_trajectories == 1
        return generate_reduced_x_timeseries(
            K=K,
            F=F,
            alpha0=alpha0,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3,
            sigma=sigma,
            dt=dt,
            spinup_steps=spinup_steps,
            save_every=save_every,
            nsamples=nsamples,
            rng_seed=rng_seed,
            max_abs_state=max_abs_state,
            max_restarts=max_restarts,
            state_min=state_min,
            state_max=state_max,
            max_boundary_hits=max_boundary_hits,
        )
    end

    counts = fill(nsamples ÷ ensemble_trajectories, ensemble_trajectories)
    for i in 1:(nsamples % ensemble_trajectories)
        counts[i] += 1
    end
    parts = Vector{Matrix{Float32}}(undef, ensemble_trajectories)
    errors = fill("", ensemble_trajectories)
    seed_stride = 1_000_000

    run_one! = function (itr::Int)
        try
            parts[itr] = generate_reduced_x_timeseries(
                K=K,
                F=F,
                alpha0=alpha0,
                alpha1=alpha1,
                alpha2=alpha2,
                alpha3=alpha3,
                sigma=sigma,
                dt=dt,
                spinup_steps=spinup_steps,
                save_every=save_every,
                nsamples=counts[itr],
                rng_seed=rng_seed + seed_stride * (itr - 1),
                max_abs_state=max_abs_state,
                max_restarts=max_restarts,
                state_min=state_min,
                state_max=state_max,
                max_boundary_hits=max_boundary_hits,
            )
        catch err
            errors[itr] = sprint(showerror, err)
        end
        return nothing
    end

    if parallel_trajectories && nthreads() > 1
        Threads.@threads for itr in 1:ensemble_trajectories
            run_one!(itr)
        end
    else
        for itr in 1:ensemble_trajectories
            run_one!(itr)
        end
    end

    for itr in 1:ensemble_trajectories
        isempty(errors[itr]) || error("Reduced ensemble trajectory $(itr) failed: $(errors[itr])")
    end

    out = Array{Float32}(undef, nsamples, K)
    pos = 1
    for part in parts
        nrows = size(part, 1)
        @views out[pos:(pos + nrows - 1), :] .= part
        pos += nrows
    end
    return out
end

function save_x_dataset(path::AbstractString,
    key::AbstractString,
    data::Array{Float32,2},
    attrs::Dict{String,Any})
    mkpath(dirname(path))
    mode = isfile(path) ? "r+" : "w"
    h5open(path, mode) do h5
        if haskey(h5, key)
            delete_object(h5, key)
        end
        write(h5, key, data)
        dset = h5[key]
        ad = attributes(dset)
        for (k, v) in attrs
            ad[k] = v
        end
    end
    return path
end

function read_dataset_signature(path::AbstractString,
    key::AbstractString,
    signature_key::AbstractString="params_signature")
    isfile(path) || return ""
    return h5open(path, "r") do h5
        haskey(h5, key) || return ""
        ad = attributes(h5[key])
        haskey(ad, signature_key) || return ""
        return String(read(ad[signature_key]))
    end
end

function dataset_time_spacing(dt::Float64, save_every::Int)
    return dt * max(save_every, 1)
end

function count_effective_samples(path::AbstractString, key::AbstractString)
    isfile(path) || return 0
    return h5open(path, "r") do h5
        haskey(h5, key) || return 0
        return Int(size(h5[key], 1))
    end
end

# -----------------------------------------------------------------------------
# Reduced stochastic L96 model (parameterized closure)
# -----------------------------------------------------------------------------

function make_reduced_workspace(K::Int)
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        xtmp=zeros(Float64, K),
    )
end

function l96_reduced_drift!(dx::Vector{Float64},
    x::Vector{Float64},
    F::Float64,
    alpha0::Float64,
    alpha1::Float64,
    alpha2::Float64,
    alpha3::Float64)
    K = length(x)
    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)
        poly = alpha0 + alpha1 * x[k] + alpha2 * x[k]^2 + alpha3 * x[k]^3
        dx[k] = -x[km1] * (x[km2] - x[kp1]) - x[k] + F - poly
    end
    return nothing
end

function l96_reduced_step!(x::Vector{Float64},
    dt::Float64,
    F::Float64,
    alpha0::Float64,
    alpha1::Float64,
    alpha2::Float64,
    alpha3::Float64,
    ws)
    l96_reduced_drift!(ws.dx1, x, F, alpha0, alpha1, alpha2, alpha3)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    l96_reduced_drift!(ws.dx2, ws.xtmp, F, alpha0, alpha1, alpha2, alpha3)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    l96_reduced_drift!(ws.dx3, ws.xtmp, F, alpha0, alpha1, alpha2, alpha3)

    @. ws.xtmp = x + dt * ws.dx3
    l96_reduced_drift!(ws.dx4, ws.xtmp, F, alpha0, alpha1, alpha2, alpha3)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    return nothing
end

function add_reduced_noise!(x::Vector{Float64}, rng::AbstractRNG, sigma::Float64, dt::Float64)
    sigma <= 0 && return nothing
    step_sigma = sigma * sqrt(dt)
    @inbounds for k in eachindex(x)
        x[k] += step_sigma * randn(rng)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Observables and response helpers
# -----------------------------------------------------------------------------

function legacy_observable_names(m::Int)
    m >= 0 || error("Observable lag parameter m must be >= 0")
    names = String["mean_x", "mean_x2"]
    for lag in 1:m
        push!(names, "raw_xxp_lag$(lag)")
    end
    return names
end

function legacy_observable_labels(m::Int)
    m >= 0 || error("Observable lag parameter m must be >= 0")
    labels = String["<X_k>", "<X_k^2>"]
    for lag in 1:m
        push!(labels, "<X_k X_{k+$(lag)}>")
    end
    return labels
end

function observable_spec(name::AbstractString, K::Int)
    key = lowercase(strip(String(name)))
    if key == "mean_x"
        return (name="mean_x", label="<X_k>", kind=:mean, lag=0)
    elseif key == "mean_x2"
        return (name="mean_x2", label="<X_k^2>", kind=:mean_x2, lag=0)
    elseif key == "var_x"
        return (name="var_x", label="Var(X_k)", kind=:var, lag=0)
    elseif key in ("cm3_x", "centered_moment3_x")
        return (name="centered_moment3_x", label="< (X_k-<X>)^3 >", kind=:cm3, lag=0)
    elseif key in ("cm4_x", "centered_moment4_x")
        return (name="centered_moment4_x", label="< (X_k-<X>)^4 >", kind=:cm4, lag=0)
    elseif key == "skew_x"
        return (name="skew_x", label="Skew(X_k)", kind=:skew, lag=0)
    elseif key == "kurt_x"
        return (name="kurt_x", label="ExcessKurt(X_k)", kind=:kurt, lag=0)
    end

    for (prefix, kind, label_fmt) in (
        ("raw_xxp_lag", :raw_lag, lag -> "<X_k X_{k+$(lag)}>" ),
        ("cov_x_lag", :cov_lag, lag -> "Cov(X_k,X_{k+$(lag)})" ),
        ("corr_x_lag", :corr_lag, lag -> "Corr(X_k,X_{k+$(lag)})" ),
    )
        if startswith(key, prefix)
            lag_text = key[(lastindex(prefix) + 1):end]
            lag = try
                parse(Int, lag_text)
            catch
                error("Invalid observable lag in '$name'")
            end
            1 <= lag <= K - 1 || error("Observable lag $(lag) in '$name' exceeds allowed range 1:$(K - 1)")
            return (name="$(prefix)$(lag)", label=label_fmt(lag), kind=kind, lag=lag)
        end
    end

    error("Unsupported observable name '$name'")
end

function observable_specs_from_names(names::Vector{String}, K::Int)
    isempty(names) && error("Observable library must contain at least one observable")
    specs = [observable_spec(name, K) for name in names]
    spec_names = [spec.name for spec in specs]
    length(unique(spec_names)) == length(spec_names) || error("Observable library contains duplicate names: $(join(spec_names, ", "))")
    return specs
end

observable_names_from_specs(specs) = [spec.name for spec in specs]
observable_labels_from_specs(specs) = [spec.label for spec in specs]

function compute_observables_x(x::Vector{Float64}, observable_specs)
    K = length(x)
    n_obs = length(observable_specs)
    invK = 1.0 / K
    obs = zeros(Float64, n_obs)
    mu = sum(x) * invK
    raw2 = sum(abs2, x) * invK

    need_centered = any(spec.kind in (:var, :cm3, :cm4, :skew, :kurt, :cov_lag, :corr_lag) for spec in observable_specs)
    xc = need_centered ? x .- mu : Float64[]
    var = need_centered ? sum(abs2, xc) * invK : max(raw2 - mu^2, 0.0)
    cm3 = any(spec.kind in (:cm3, :skew) for spec in observable_specs) ? sum(v^3 for v in xc) * invK : 0.0
    cm4 = any(spec.kind in (:cm4, :kurt) for spec in observable_specs) ? sum(v^4 for v in xc) * invK : 0.0
    skew = var > eps(Float64) ? cm3 / (var^(1.5)) : 0.0
    kurt = var > eps(Float64) ? cm4 / (var^2) - 3.0 : 0.0

    for (i, spec) in enumerate(observable_specs)
        kind = spec.kind
        if kind == :mean
            obs[i] = mu
        elseif kind == :mean_x2
            obs[i] = raw2
        elseif kind == :var
            obs[i] = var
        elseif kind == :cm3
            obs[i] = cm3
        elseif kind == :cm4
            obs[i] = cm4
        elseif kind == :skew
            obs[i] = skew
        elseif kind == :kurt
            obs[i] = kurt
        elseif kind == :raw_lag
            lag = spec.lag
            acc = 0.0
            @inbounds for k in 1:K
                kp = mod1idx(k + lag, K)
                acc += x[k] * x[kp]
            end
            obs[i] = acc * invK
        elseif kind == :cov_lag
            lag = spec.lag
            acc = 0.0
            @inbounds for k in 1:K
                kp = mod1idx(k + lag, K)
                acc += xc[k] * xc[kp]
            end
            obs[i] = acc * invK
        elseif kind == :corr_lag
            lag = spec.lag
            acc = 0.0
            @inbounds for k in 1:K
                kp = mod1idx(k + lag, K)
                acc += xc[k] * xc[kp]
            end
            cov = acc * invK
            obs[i] = var > eps(Float64) ? cov / var : 0.0
        else
            error("Unsupported observable kind $(kind)")
        end
    end
    return obs
end

function compute_observables_x(x::Vector{Float64},
    F::Float64,
    alpha0_ref::Float64,
    alpha1_ref::Float64,
    alpha2_ref::Float64,
    alpha3_ref::Float64,
    m::Int=3)
    specs = observable_specs_from_names(legacy_observable_names(m), length(x))
    return compute_observables_x(x, specs)
end

function compute_observables_x(x::Vector{Float64},
    F::Float64,
    alpha0_ref::Float64,
    alpha1_ref::Float64,
    alpha2_ref::Float64,
    alpha3_ref::Float64,
    observable_specs::Vector)
    return compute_observables_x(x, observable_specs)
end

function compute_observables_series(xseries::Array{Float64,2},
    F::Float64,
    alpha0_ref::Float64,
    alpha1_ref::Float64,
    alpha2_ref::Float64,
    alpha3_ref::Float64,
    observable_specs::Vector)
    _, N = size(xseries)
    n_obs = length(observable_specs)
    out = Array{Float64}(undef, n_obs, N)
    x = zeros(Float64, size(xseries, 1))
    for n in 1:N
        @inbounds x .= view(xseries, :, n)
        out[:, n] .= compute_observables_x(x, observable_specs)
    end
    return out
end

function compute_observables_series(xseries::Array{Float64,2},
    F::Float64,
    alpha0_ref::Float64,
    alpha1_ref::Float64,
    alpha2_ref::Float64,
    alpha3_ref::Float64,
    m::Int=3)
    specs = observable_specs_from_names(legacy_observable_names(m), size(xseries, 1))
    return compute_observables_series(xseries, F, alpha0_ref, alpha1_ref, alpha2_ref, alpha3_ref, specs)
end

function xcorr_one_sided_unbiased_fft(x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    kmax::Int)
    n = length(x)
    n == length(y) || error("xcorr inputs must have same length")
    kmax = min(kmax, n - 1)

    L = 1
    target = 2 * n - 1
    while L < target
        L <<= 1
    end

    xp = zeros(Float64, L)
    yp = zeros(Float64, L)
    @inbounds xp[1:n] .= x
    @inbounds yp[1:n] .= y
    c = real(ifft(fft(xp) .* conj.(fft(yp))))

    out = Array{Float64}(undef, kmax + 1)
    @inbounds for k in 0:kmax
        out[k + 1] = c[k + 1] / (n - k)
    end
    return out
end

function _asymptotic_interval_indices(times::Vector{Float64}, t_start::Float64, t_end::Float64)
    t_start >= 0.0 || error("t_start must be >= 0")
    t_end > t_start || error("t_end must be > t_start")
    tmin = times[1]
    tmax = times[end]
    ts = clamp(t_start, tmin, tmax)
    te = clamp(t_end, tmin, tmax)
    te >= ts || return [length(times)], ts, te
    idx = findall(t -> t >= ts && t <= te, times)
    isempty(idx) && (idx = [length(times)])
    return idx, ts, te
end

function _tail_taper_weights(times::Vector{Float64},
    t_start::Float64,
    t_end::Float64,
    taper::String)
    mode = lowercase(strip(taper))
    n = length(times)
    w = zeros(Float64, n)

    if mode == "none"
        fill!(w, 1.0)
        return w
    elseif mode == "hard"
        @inbounds for k in eachindex(times)
            w[k] = times[k] >= t_start ? 1.0 : 0.0
        end
        return w
    elseif mode == "linear"
        span = t_end - t_start
        if span <= eps(Float64)
            w[end] = 1.0
            return w
        end
        inv_span = 1.0 / span
        @inbounds for k in eachindex(times)
            tk = times[k]
            if tk <= t_start
                w[k] = 0.0
            elseif tk >= t_end
                w[k] = 1.0
            else
                w[k] = (tk - t_start) * inv_span
            end
        end
        return w
    end

    error("tail_taper must be one of none|hard|linear (got '$taper')")
end

function build_gfdt_response(A::Matrix{Float64},
    G::Matrix{Float64},
    delta_t::Float64,
    n_lags::Int;
    mean_center::Bool=true,
    impulse_tail_debias::Bool=false,
    t_start::Float64=0.0,
    t_end::Float64=1.0,
    tail_taper::String="linear")
    m, N = size(A)
    p, N2 = size(G)
    N == N2 || error("A/G time length mismatch")
    n_lags = min(n_lags, N - 1)
    n_lags >= 1 || error("Need at least one lag")

    Ause = mean_center ? (A .- mean(A; dims=2)) : A
    Guse = mean_center ? (G .- mean(G; dims=2)) : G

    times = collect(0:n_lags) .* delta_t
    tail_idx = Int[]
    taper_weights = Float64[]
    if impulse_tail_debias
        tail_idx, ts, te = _asymptotic_interval_indices(times, t_start, t_end)
        taper_weights = _tail_taper_weights(times, ts, te, tail_taper)
    end

    C = zeros(Float64, m, p, n_lags + 1)
    R = zeros(Float64, m, p, n_lags + 1)

    Threads.@threads for pair in 1:(m * p)
        i = (pair - 1) ÷ p + 1
        j = (pair - 1) % p + 1
        ai = vec(@view Ause[i, :])
        gj = vec(@view Guse[j, :])
        cpos = xcorr_one_sided_unbiased_fft(ai, gj, n_lags)
        if impulse_tail_debias
            c_bias = mean(@view cpos[tail_idx])
            @inbounds for k in eachindex(cpos)
                cpos[k] -= taper_weights[k] * c_bias
            end
        end
        @views C[i, j, :] .= cpos
        R[i, j, 1] = 0.0
        acc = 0.0
        @inbounds for lag in 1:n_lags
            acc += cpos[lag + 1]
            R[i, j, lag + 1] = delta_t * acc
        end
    end

    return C, R, times
end

function step_to_impulse(R::Array{Float64,3}, times::Vector{Float64})
    m, p, nt = size(R)
    nt == length(times) || error("step_to_impulse time mismatch")
    nt >= 2 || error("Need at least two points")
    out = zeros(Float64, m, p, nt)
    @inbounds for i in 1:m, j in 1:p
        out[i, j, 1] = (R[i, j, 2] - R[i, j, 1]) / (times[2] - times[1])
        for t in 2:(nt - 1)
            out[i, j, t] = (R[i, j, t + 1] - R[i, j, t - 1]) / (times[t + 1] - times[t - 1])
        end
        out[i, j, nt] = (R[i, j, nt] - R[i, j, nt - 1]) / (times[nt] - times[nt - 1])
    end
    return out
end

function response_output_times(tmax::Float64; npoints::Int=301)
    tmax_eff = max(tmax, 1e-8)
    return collect(range(0.0, stop=tmax_eff, length=max(npoints, 2)))
end

function linear_interpolate_3d_time(R::Array{Float64,3},
    t_in::Vector{Float64},
    t_out::Vector{Float64})
    m, p, n_in = size(R)
    n_in == length(t_in) || error("Time/data mismatch")
    out = zeros(Float64, m, p, length(t_out))
    @inbounds for i in 1:m, j in 1:p
        for (k, t) in enumerate(t_out)
            if t <= t_in[1]
                out[i, j, k] = R[i, j, 1]
            elseif t >= t_in[end]
                out[i, j, k] = R[i, j, end]
            else
                idx = searchsortedlast(t_in, t)
                t0, t1 = t_in[idx], t_in[idx + 1]
                w = (t - t0) / (t1 - t0)
                out[i, j, k] = R[i, j, idx] + w * (R[i, j, idx + 1] - R[i, j, idx])
            end
        end
    end
    return out
end

function finite_rmse(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    size(a) == size(b) || error("RMSE arrays must have same shape")
    acc = 0.0
    n = 0
    @inbounds for i in eachindex(a)
        ai = Float64(a[i])
        bi = Float64(b[i])
        if isfinite(ai) && isfinite(bi)
            acc += (ai - bi)^2
            n += 1
        end
    end
    n > 0 || return NaN
    return sqrt(acc / n)
end

function read_epoch_losses(training_metrics_csv::AbstractString)
    losses = Float64[]
    epochs = Int[]
    isfile(training_metrics_csv) || return epochs, losses

    in_epoch_block = false
    for line in eachline(training_metrics_csv)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            in_epoch_block = startswith(s, "# epoch,")
            continue
        end
        in_epoch_block || continue
        parts = split(s, ",")
        length(parts) >= 2 || continue
        push!(epochs, parse(Int, strip(parts[1])))
        push!(losses, parse(Float64, strip(parts[2])))
    end
    return epochs, losses
end

end # module
