module ArnoldK36Common

using Dates
using HDF5
using LinearAlgebra
using Random
using SHA
using Statistics
using TOML
using Base.Threads

export mod1idx,
    parse_bool,
    ensure_dir,
    dict_signature,
    dataset_time_spacing,
    make_full_workspace,
    generate_two_scale_x_timeseries,
    make_reduced_workspace,
    l96_reduced_step!,
    add_reduced_noise!,
    generate_reduced_x_timeseries,
    generate_reduced_ensemble_x_timeseries,
    fit_polynomial_closure,
    compute_observables_x,
    compute_observables_series,
    save_x_dataset,
    load_x_matrix,
    load_dataset_attributes,
    read_dataset_signature

mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

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

function ensure_dir(path::AbstractString)
    mkpath(path)
    return path
end

function dict_signature(value)
    io = IOBuffer()
    TOML.print(io, value)
    return bytes2hex(sha1(take!(io)))
end

dataset_time_spacing(dt::Float64, save_every::Int) = dt * max(save_every, 1)

# -----------------------------------------------------------------------------
# Two-scale deterministic model
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
    K >= 2 || error("K must be >= 2")
    J >= 1 || error("J must be >= 1")
    dt > 0 || error("dt must be > 0")
    spinup_steps >= 0 || error("spinup_steps must be >= 0")
    save_every >= 1 || error("save_every must be >= 1")
    nsamples >= 2 || error("nsamples must be >= 2")
    process_noise_sigma >= 0 || error("process_noise_sigma must be >= 0")

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

# -----------------------------------------------------------------------------
# Reduced stochastic model
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
    save_every >= 1 || error("save_every must be >= 1")
    nsamples >= 2 || error("nsamples must be >= 2")
    dt > 0 || error("dt must be > 0")
    state_min <= state_max || error("state_min must be <= state_max")
    max_boundary_hits >= 1 || error("max_boundary_hits must be >= 1")

    use_bounds = isfinite(state_min) || isfinite(state_max)
    boundary_hits = 0
    seed_stride = 10_000
    out = Array{Float32}(undef, nsamples, K)

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
    end

    error("Failed to generate stable reduced trajectory after $(max_restarts + 1) attempts (base rng_seed=$rng_seed)")
end

function generate_reduced_ensemble_x_timeseries(;K::Int,
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
    trajectories::Int=1,
    parallel_trajectories::Bool=true,
    max_abs_state::Float64=1e4,
    max_restarts::Int=8,
    state_min::Float64=-Inf,
    state_max::Float64=Inf,
    max_boundary_hits::Int=40)
    trajectories >= 1 || error("trajectories must be >= 1")
    nsamples >= 2 || error("nsamples must be >= 2")
    dt > 0 || error("dt must be > 0")

    base = div(nsamples, trajectories)
    remn = mod(nsamples, trajectories)
    counts = [base + (i <= remn ? 1 : 0) for i in 1:trajectories]
    chunks = Vector{Matrix{Float32}}(undef, trajectories)
    errs = fill("", trajectories)

    make_chunk! = function (i::Int)
        nloc = counts[i]
        if nloc <= 0
            chunks[i] = zeros(Float32, 0, K)
            errs[i] = ""
            return nothing
        end
        try
            seed = rng_seed + 100_000 * i
            chunks[i] = generate_reduced_x_timeseries(
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
                nsamples=nloc,
                rng_seed=seed,
                max_abs_state=max_abs_state,
                max_restarts=max_restarts,
                state_min=state_min,
                state_max=state_max,
                max_boundary_hits=max_boundary_hits,
            )
            errs[i] = ""
        catch err
            chunks[i] = zeros(Float32, 0, K)
            errs[i] = sprint(showerror, err)
        end
        return nothing
    end

    do_parallel = parallel_trajectories && trajectories > 1 && Threads.nthreads() > 1
    if do_parallel
        Threads.@threads for i in 1:trajectories
            make_chunk!(i)
        end
    else
        for i in 1:trajectories
            make_chunk!(i)
        end
    end

    for i in 1:trajectories
        isempty(errs[i]) || error("Reduced trajectory chunk $i failed: $(errs[i])")
    end

    out = Array{Float32}(undef, nsamples, K)
    pos = 1
    for i in 1:trajectories
        chunk = chunks[i]
        nloc = size(chunk, 1)
        if nloc > 0
            @views out[pos:(pos + nloc - 1), :] .= chunk
            pos += nloc
        end
    end
    pos == nsamples + 1 || error("Ensemble assembly mismatch: expected $nsamples rows, got $(pos - 1)")
    return out
end

# -----------------------------------------------------------------------------
# First-guess closure fit
# -----------------------------------------------------------------------------

function fit_polynomial_closure(X::Matrix{Float64},
    dt_obs::Float64,
    F::Float64;
    fit_start_index::Int=2,
    fit_samples::Int=50_000,
    fit_min_samples::Int=500,
    sigma_floor::Float64=1e-8,
    sigma_scale::Float64=1.0,
    sigma_cap::Float64=Inf)
    K, N = size(X)
    N >= 3 || error("Need at least 3 samples to fit closure; got $N")
    dt_obs > 0 || error("dt_obs must be > 0")
    fit_samples >= 10 || error("fit_samples must be >= 10")
    fit_min_samples >= 10 || error("fit_min_samples must be >= 10")
    sigma_floor >= 0 || error("sigma_floor must be >= 0")
    sigma_scale > 0 || error("sigma_scale must be > 0")
    sigma_cap >= sigma_floor || error("sigma_cap must be >= sigma_floor")

    ns = min(fit_samples, N - 2)
    ns >= fit_min_samples || error("Not enough samples for closure fit: need >= $fit_min_samples, got $ns")

    max_start = max(2, N - ns)
    start = clamp(fit_start_index, 2, max_start)
    stop = start + ns - 1
    nt = stop - start + 1

    A = Array{Float64}(undef, nt * K, 4)
    bvec = Array{Float64}(undef, nt * K)

    Threads.@threads for ti in 1:nt
        t = start + ti - 1
        row0 = (ti - 1) * K
        @inbounds for k in 1:K
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

            row = row0 + k
            A[row, 1] = 1.0
            A[row, 2] = xk
            A[row, 3] = xk * xk
            A[row, 4] = xk * xk * xk
            bvec[row] = y
        end
    end

    coeff = A \ bvec
    resid = bvec .- A * coeff
    sigma_fit = std(resid) * sqrt(dt_obs) * sigma_scale
    sigma_fit = clamp(max(sigma_fit, sigma_floor), sigma_floor, sigma_cap)

    theta = (coeff[1], coeff[2], coeff[3], coeff[4], sigma_fit)
    meta = Dict{String,Any}(
        "fit_start_index_requested" => fit_start_index,
        "fit_start_index_used" => start,
        "fit_stop_index_used" => stop,
        "fit_samples_requested" => fit_samples,
        "fit_samples_used" => ns,
        "fit_min_samples" => fit_min_samples,
        "dt_obs" => dt_obs,
        "residual_std" => std(resid),
        "sigma_scale" => sigma_scale,
        "sigma_floor" => sigma_floor,
        "sigma_cap" => sigma_cap,
        "threads" => Threads.nthreads(),
    )
    return theta, meta
end

# -----------------------------------------------------------------------------
# Observables and HDF5 I/O
# -----------------------------------------------------------------------------

function compute_observables_x(x::Vector{Float64}, m::Int=3)
    K = length(x)
    m >= 0 || error("Observable lag parameter m must be >= 0")
    m <= K - 1 || error("Observable lag parameter m=$(m) exceeds K-1=$(K-1)")

    n_obs = m + 2
    invK = 1.0 / K
    obs = zeros(Float64, n_obs)

    @inbounds for k in 1:K
        xk = x[k]
        obs[1] += xk
        obs[2] += xk * xk
        for lag in 1:m
            kp = mod1idx(k + lag, K)
            obs[lag + 2] += xk * x[kp]
        end
    end

    obs .*= invK
    return obs
end

function compute_observables_series(xseries::Matrix{Float64}, m::Int=3)
    K, N = size(xseries)
    m >= 0 || error("Observable lag parameter m must be >= 0")
    m <= K - 1 || error("Observable lag parameter m=$(m) exceeds K-1=$(K-1)")

    n_obs = m + 2
    out = Array{Float64}(undef, n_obs, N)
    x = zeros(Float64, K)
    for n in 1:N
        @inbounds x .= view(xseries, :, n)
        out[:, n] .= compute_observables_x(x, m)
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

function load_x_matrix(path::AbstractString, key::AbstractString)
    isfile(path) || error("Dataset file not found: $path")
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        raw = Float64.(read(h5[key]))
        ndims(raw) == 2 || error("Dataset at $path/$key must be 2D")
        return permutedims(raw, (2, 1))
    end
end

function _normalize_attr_value(v)
    if v isa AbstractString
        return String(v)
    elseif v isa AbstractVector{UInt8}
        return String(v)
    elseif v isa Bool
        return v
    elseif v isa Integer
        return Int(v)
    elseif v isa AbstractFloat
        return Float64(v)
    elseif v isa AbstractArray
        if eltype(v) <: Integer
            return Int.(collect(v))
        elseif eltype(v) <: AbstractFloat
            return Float64.(collect(v))
        end
        return collect(v)
    end
    return v
end

function load_dataset_attributes(path::AbstractString, key::AbstractString)
    isfile(path) || error("Dataset file not found: $path")
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Dataset key '$key' not found in $path")
        out = Dict{String,Any}()
        ad = attributes(h5[key])
        for k in keys(ad)
            out[String(k)] = _normalize_attr_value(read(ad[k]))
        end
        return out
    end
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

end # module
