module ArnoldCommon

using Dates
using FFTW
using HDF5
using LinearAlgebra
using Printf
using Random
using SHA
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

const ARNOLD_DATASET_ROLES = ("two_scale_observed", "train_stochastic", "gfdt_stochastic")
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
        "twoscale.process_noise_sigma" => _as_float(twoscale, "process_noise_sigma", 0.0),
        "twoscale.stochastic_x_noise" => _as_bool(twoscale, "stochastic_x_noise", false),

        "closure.F" => _as_float(closure, "F", _as_float(twoscale, "F", 20.0)),
        "closure.alpha0_initial" => _as_float(closure, "alpha0_initial", _as_float(closure, "alpha0", 0.0)),
        "closure.alpha1_initial" => _as_float(closure, "alpha1_initial", _as_float(closure, "alpha1", 0.0)),
        "closure.alpha2_initial" => _as_float(closure, "alpha2_initial", _as_float(closure, "alpha2", 0.0)),
        "closure.alpha3_initial" => _as_float(closure, "alpha3_initial", _as_float(closure, "alpha3", 0.0)),
        "closure.sigma_initial" => _as_float(closure, "sigma_initial", _as_float(closure, "sigma", 1.0)),
        "closure.auto_fit" => _as_bool(closure, "auto_fit", true),
        "closure.fit_dataset_role" => _as_str(closure, "fit_dataset_role", "two_scale_observed"),
        "closure.fit_start_index" => _as_int(closure, "fit_start_index", 2),
        "closure.fit_samples" => _as_int(closure, "fit_samples", 50_000),
        "closure.fit_min_samples" => _as_int(closure, "fit_min_samples", 500),
    )

    for role in ARNOLD_DATASET_ROLES
        tbl = _as_subtable(datasets, role)
        cfg["datasets.$role.key"] = _as_str(tbl, "key", role)
        cfg["datasets.$role.spinup_steps"] = _as_int(tbl, "spinup_steps", 50_000)
        cfg["datasets.$role.save_every"] = _as_int(tbl, "save_every", 1)
        cfg["datasets.$role.nsamples"] = _as_int(tbl, "nsamples", 100_000)
        cfg["datasets.$role.rng_seed"] = _as_int(tbl, "rng_seed", 11)
        cfg["datasets.$role.target_spacing"] = _as_float(tbl, "target_spacing", dataset_time_spacing(cfg["twoscale.dt"], cfg["datasets.$role.save_every"]))
    end

    cfg["twoscale.K"] >= 2 || error("twoscale.K must be >= 2")
    cfg["twoscale.J"] >= 1 || error("twoscale.J must be >= 1")
    cfg["twoscale.dt"] > 0 || error("twoscale.dt must be > 0")
    cfg["closure.fit_dataset_role"] = _check_role(cfg["closure.fit_dataset_role"])
    cfg["closure.fit_dataset_role"] == "two_scale_observed" || error("closure.fit_dataset_role must be 'two_scale_observed' to satisfy auto_fit against bi-scale data")
    cfg["closure.fit_samples"] >= 10 || error("closure.fit_samples must be >= 10")
    cfg["closure.fit_min_samples"] >= 10 || error("closure.fit_min_samples must be >= 10")

    for role in ARNOLD_DATASET_ROLES
        cfg["datasets.$role.spinup_steps"] >= 0 || error("datasets.$role.spinup_steps must be >= 0")
        cfg["datasets.$role.save_every"] >= 1 || error("datasets.$role.save_every must be >= 1")
        cfg["datasets.$role.nsamples"] >= 2 || error("datasets.$role.nsamples must be >= 2")
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
        "role" => role_s,
        "dataset_key" => cfg["datasets.$role_s.key"],
        "spinup_steps" => cfg["datasets.$role_s.spinup_steps"],
        "save_every" => cfg["datasets.$role_s.save_every"],
        "nsamples" => cfg["datasets.$role_s.nsamples"],
        "rng_seed" => cfg["datasets.$role_s.rng_seed"],
        "twoscale_dt" => cfg["twoscale.dt"],
        "twoscale_K" => cfg["twoscale.K"],
    )

    if role_s == "two_scale_observed"
        base["model"] = "two_scale_deterministic"
        base["twoscale"] = Dict(
            "J" => cfg["twoscale.J"],
            "F" => cfg["twoscale.F"],
            "h" => cfg["twoscale.h"],
            "c" => cfg["twoscale.c"],
            "b" => cfg["twoscale.b"],
            "process_noise_sigma" => cfg["twoscale.process_noise_sigma"],
            "stochastic_x_noise" => cfg["twoscale.stochastic_x_noise"],
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
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        ds = h5[key]
        n_total = size(ds, 1)
        n_req = nsamples <= 0 ? n_total : nsamples
        s_use, e_use = _load_role_bounds(ds, n_req, start_index, label)
        raw = Float64.(ds[s_use:e_use, :])
        permutedims(raw, (2, 1))
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

    start = max(2, Int(floor((N - ns) / 2)))
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
    sigma_fit = std(resid) * sqrt(dt_obs)
    return (coeff[1], coeff[2], coeff[3], coeff[4], max(sigma_fit, 1e-8))
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

    Xfit = load_role_x_matrix(
        cfg,
        fit_role;
        nsamples=cfg["closure.fit_samples"],
        start_index=cfg["closure.fit_start_index"],
        label="closure_auto_fit",
    )
    dt_obs = dataset_role_spacing(cfg, fit_role)
    theta_fit = _fit_polynomial_closure(
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
        "fit_dt_obs" => dt_obs,
    )
    _AUTO_FIT_CACHE[cache_key] = (theta=theta_fit, meta=copy(meta))
    return theta_fit, meta
end

function ensure_arnold_dataset_role!(cfg::Dict{String,Any}, role::AbstractString)
    role_s = _check_role(role)
    path = cfg["paths.datasets_hdf5"]
    key = cfg["datasets.$role_s.key"]

    closure_theta = if role_s == "two_scale_observed"
        nothing
    else
        theta, _ = resolve_closure_theta(cfg)
        theta
    end

    sig = _dataset_role_signature(cfg, role_s; closure_theta=closure_theta)
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

    data = if role_s == "two_scale_observed"
        generate_two_scale_x_timeseries(
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
            process_noise_sigma=cfg["twoscale.process_noise_sigma"],
            stochastic_x_noise=cfg["twoscale.stochastic_x_noise"],
        )
    else
        theta, _ = resolve_closure_theta(cfg)
        generate_reduced_x_timeseries(
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
        )
    end

    all(isfinite, data) || error("Generated dataset role '$role_s' contains non-finite values")

    attrs = Dict{String,Any}(
        "role" => role_s,
        "model" => role_s == "two_scale_observed" ? "two_scale_deterministic" : "reduced_stochastic",
        "dt" => cfg["twoscale.dt"],
        "save_every" => cfg["datasets.$role_s.save_every"],
        "nsamples" => cfg["datasets.$role_s.nsamples"],
        "spinup_steps" => cfg["datasets.$role_s.spinup_steps"],
        "rng_seed" => cfg["datasets.$role_s.rng_seed"],
        "params_signature" => sig,
        "generated_at" => string(now()),
        "source_parameters_data" => cfg["paths.parameters_data_path"],
    )

    if role_s == "two_scale_observed"
        attrs["K"] = cfg["twoscale.K"]
        attrs["J"] = cfg["twoscale.J"]
        attrs["F"] = cfg["twoscale.F"]
        attrs["h"] = cfg["twoscale.h"]
        attrs["c"] = cfg["twoscale.c"]
        attrs["b"] = cfg["twoscale.b"]
        attrs["process_noise_sigma"] = cfg["twoscale.process_noise_sigma"]
        attrs["stochastic_x_noise"] = cfg["twoscale.stochastic_x_noise"]
    else
        theta, meta = resolve_closure_theta(cfg)
        attrs["K"] = cfg["twoscale.K"]
        attrs["F"] = cfg["closure.F"]
        attrs["alpha0"] = theta[1]
        attrs["alpha1"] = theta[2]
        attrs["alpha2"] = theta[3]
        attrs["alpha3"] = theta[4]
        attrs["sigma"] = theta[5]
        attrs["closure_auto_fit"] = cfg["closure.auto_fit"]
        attrs["closure_fit_mode"] = get(meta, "mode", "")
        attrs["closure_fit_dataset_role"] = get(meta, "fit_dataset_role", "")
    end

    save_x_dataset(path, key, data, attrs)
    return Dict{String,Any}(
        "role" => role_s,
        "path" => path,
        "key" => key,
        "signature" => sig,
        "generated" => true,
    )
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
    coupling = h * c / b

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
    sigma::Float64,
    dt::Float64;
    stochastic_x_noise::Bool=false)
    sigma <= 0 && return nothing
    step_sigma = sigma * sqrt(dt)
    @inbounds begin
        if stochastic_x_noise
            for k in eachindex(x)
                x[k] += step_sigma * randn(rng)
            end
        end
        for idx in eachindex(y)
            y[idx] += step_sigma * randn(rng)
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
    process_noise_sigma::Float64=0.0,
    stochastic_x_noise::Bool=false)
    rng = MersenneTwister(rng_seed)
    x = F .+ 0.01 .* randn(rng, Float64, K)
    y = 0.01 .* randn(rng, Float64, J, K)
    ws = make_full_workspace(K, J)

    for _ in 1:spinup_steps
        rk4_full_step!(x, y, dt, ws, K, J, F, h, c, b)
        add_full_process_noise!(x, y, rng, process_noise_sigma, dt; stochastic_x_noise=stochastic_x_noise)
    end

    out = Array{Float32}(undef, nsamples, K)
    for n in 1:nsamples
        for _ in 1:save_every
            rk4_full_step!(x, y, dt, ws, K, J, F, h, c, b)
            add_full_process_noise!(x, y, rng, process_noise_sigma, dt; stochastic_x_noise=stochastic_x_noise)
        end
        @inbounds out[n, :] .= Float32.(x)
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
    max_restarts::Int=8)
    max_abs_state > 0 || error("max_abs_state must be > 0")
    max_restarts >= 0 || error("max_restarts must be >= 0")

    out = Array{Float32}(undef, nsamples, K)
    seed_stride = 10_000

    for attempt in 0:max_restarts
        seed = rng_seed + attempt * seed_stride
        rng = MersenneTwister(seed)
        x = F .+ 0.01 .* randn(rng, Float64, K)
        ws = make_reduced_workspace(K)
        stable = true

        for _ in 1:spinup_steps
            l96_reduced_step!(x, dt, F, alpha0, alpha1, alpha2, alpha3, ws)
            add_reduced_noise!(x, rng, sigma, dt)
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
        attempt < max_restarts && @warn "Reduced trajectory became unstable; retrying with shifted seed" base_seed = rng_seed retry_seed = seed + seed_stride attempt = attempt + 1
    end

    error("Failed to generate stable reduced trajectory after $(max_restarts + 1) attempts (base rng_seed=$rng_seed)")
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

function compute_observables_x(x::Vector{Float64},
    F::Float64,
    alpha0_ref::Float64,
    alpha1_ref::Float64,
    alpha2_ref::Float64,
    alpha3_ref::Float64)
    K = length(x)
    invK = 1.0 / K
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    s5 = 0.0
    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        km3 = mod1idx(k - 3, K)

        xk = x[k]
        xkm1 = x[km1]
        xkm2 = x[km2]
        xkm3 = x[km3]

        x2 = xk * xk
        s1 += xk
        s2 += x2
        s3 += xk * xkm1
        s4 += xk * xkm2
        s5 += xk * xkm3
    end
    return [s1 * invK, s2 * invK, s3 * invK, s4 * invK, s5 * invK]
end

function compute_observables_series(xseries::Array{Float64,2},
    F::Float64,
    alpha0_ref::Float64,
    alpha1_ref::Float64,
    alpha2_ref::Float64,
    alpha3_ref::Float64)
    K, N = size(xseries)
    out = Array{Float64}(undef, 5, N)
    x = zeros(Float64, K)
    for n in 1:N
        @inbounds x .= view(xseries, :, n)
        out[:, n] .= compute_observables_x(x, F, alpha0_ref, alpha1_ref, alpha2_ref, alpha3_ref)
    end
    return out
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

function build_gfdt_response(A::Matrix{Float64},
    G::Matrix{Float64},
    delta_t::Float64,
    n_lags::Int;
    mean_center::Bool=true)
    m, N = size(A)
    p, N2 = size(G)
    N == N2 || error("A/G time length mismatch")
    n_lags = min(n_lags, N - 1)
    n_lags >= 1 || error("Need at least one lag")

    Ause = mean_center ? (A .- mean(A; dims=2)) : A
    Guse = mean_center ? (G .- mean(G; dims=2)) : G

    C = zeros(Float64, m, p, n_lags + 1)
    R = zeros(Float64, m, p, n_lags + 1)

    Threads.@threads for pair in 1:(m * p)
        i = (pair - 1) ÷ p + 1
        j = (pair - 1) % p + 1
        ai = vec(@view Ause[i, :])
        gj = vec(@view Guse[j, :])
        cpos = xcorr_one_sided_unbiased_fft(ai, gj, n_lags)
        @views C[i, j, :] .= cpos
        R[i, j, 1] = 0.0
        acc = 0.0
        @inbounds for lag in 1:n_lags
            acc += cpos[lag]
            R[i, j, lag + 1] = delta_t * acc
        end
    end

    times = collect(0:n_lags) .* delta_t
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
