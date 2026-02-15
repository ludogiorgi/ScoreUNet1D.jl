using HDF5
using Random

include(joinpath(@__DIR__, "run_layout.jl"))
using .L96RunLayout

const PIPELINE_MODE = lowercase(get(ENV, "L96_PIPELINE_MODE", "false")) == "true"
const K = parse(Int, get(ENV, "L96_K", "36"))
const J = parse(Int, get(ENV, "L96_J", "10"))
const F = parse(Float64, get(ENV, "L96_F", "10.0"))
const H = parse(Float64, get(ENV, "L96_H", "1.0"))
const C = parse(Float64, get(ENV, "L96_C", "10.0"))
const B_PARAM = parse(Float64, get(ENV, "L96_B", "10.0"))

const DT = parse(Float64, get(ENV, "L96_DT", "0.005"))
const SPINUP_STEPS = parse(Int, get(ENV, "L96_SPINUP_STEPS", "20000"))
const SAVE_EVERY = parse(Int, get(ENV, "L96_SAVE_EVERY", "10"))
const NSAMPLES = parse(Int, get(ENV, "L96_NSAMPLES", "12000"))
const RNG_SEED = parse(Int, get(ENV, "L96_RNG_SEED", "1234"))
const PROCESS_NOISE_SIGMA = parse(Float64, get(ENV, "L96_PROCESS_NOISE_SIGMA", "0.03"))

const RUN_DIR = let
    default_dir = L96RunLayout.pick_generation_run_dir(@__DIR__, J)
    data_path_env = get(ENV, "L96_DATA_PATH", "")
    if isempty(strip(data_path_env))
        default_dir
    else
        inferred = L96RunLayout.infer_run_dir_from_data_path(data_path_env)
        inferred === nothing ? default_dir : inferred
    end
end

const OUTPUT_PATH = get(ENV, "L96_DATA_PATH", L96RunLayout.default_data_path(RUN_DIR))

mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

function l96_two_scale_drift!(dx::AbstractVector{Float64},
                              dy::AbstractMatrix{Float64},
                              x::AbstractVector{Float64},
                              y::AbstractMatrix{Float64})
    coupling_scale = H * C / B_PARAM

    @inbounds for k in 1:K
        km2 = mod1idx(k - 2, K)
        km1 = mod1idx(k - 1, K)
        kp1 = mod1idx(k + 1, K)

        coupling = coupling_scale * sum(@view y[:, k])
        dx[k] = x[km1] * (x[kp1] - x[km2]) - x[k] + F - coupling
    end

    @inbounds for k in 1:K
        xk_term = coupling_scale * x[k]
        for j in 1:J
            jm1 = mod1idx(j - 1, J)
            jp1 = mod1idx(j + 1, J)
            jp2 = mod1idx(j + 2, J)
            dy[j, k] = C * B_PARAM * y[jp1, k] * (y[jm1, k] - y[jp2, k]) - C * y[j, k] + xk_term
        end
    end

    return nothing
end

function rk4_step!(x::Vector{Float64},
                   y::Matrix{Float64},
                   dt::Float64,
                   ws::NamedTuple)
    l96_two_scale_drift!(ws.dx1, ws.dy1, x, y)

    @. ws.xtmp = x + 0.5 * dt * ws.dx1
    @. ws.ytmp = y + 0.5 * dt * ws.dy1
    l96_two_scale_drift!(ws.dx2, ws.dy2, ws.xtmp, ws.ytmp)

    @. ws.xtmp = x + 0.5 * dt * ws.dx2
    @. ws.ytmp = y + 0.5 * dt * ws.dy2
    l96_two_scale_drift!(ws.dx3, ws.dy3, ws.xtmp, ws.ytmp)

    @. ws.xtmp = x + dt * ws.dx3
    @. ws.ytmp = y + dt * ws.dy3
    l96_two_scale_drift!(ws.dx4, ws.dy4, ws.xtmp, ws.ytmp)

    @. x = x + (dt / 6.0) * (ws.dx1 + 2.0 * ws.dx2 + 2.0 * ws.dx3 + ws.dx4)
    @. y = y + (dt / 6.0) * (ws.dy1 + 2.0 * ws.dy2 + 2.0 * ws.dy3 + ws.dy4)
    return nothing
end

function add_process_noise!(x::Vector{Float64},
                            y::Matrix{Float64},
                            rng::AbstractRNG,
                            sigma::Float64,
                            dt::Float64)
    sigma_step = sigma * sqrt(dt)
    sigma_step == 0.0 && return nothing
    @inbounds begin
        for k in eachindex(x)
            x[k] += sigma_step * randn(rng)
        end
        for idx in eachindex(y)
            y[idx] += sigma_step * randn(rng)
        end
    end
    return nothing
end

function make_workspace()
    return (
        dx1=zeros(Float64, K), dx2=zeros(Float64, K), dx3=zeros(Float64, K), dx4=zeros(Float64, K),
        dy1=zeros(Float64, J, K), dy2=zeros(Float64, J, K), dy3=zeros(Float64, J, K), dy4=zeros(Float64, J, K),
        xtmp=zeros(Float64, K), ytmp=zeros(Float64, J, K),
    )
end

function write_snapshot!(dest::AbstractMatrix{Float32},
                         x::AbstractVector{Float64},
                         y::AbstractMatrix{Float64})
    @inbounds for k in 1:K
        dest[1, k] = Float32(x[k])
        for j in 1:J
            dest[j + 1, k] = Float32(y[j, k])
        end
    end
    return nothing
end

function generate_l96_dataset()
    rng = MersenneTwister(RNG_SEED)
    x = F .+ 0.1 .* randn(rng, Float64, K)
    y = 0.1 .* randn(rng, Float64, J, K)
    ws = make_workspace()

    @info "Spin-up started" steps = SPINUP_STEPS dt = DT process_noise_sigma = PROCESS_NOISE_SIGMA
    for _ in 1:SPINUP_STEPS
        rk4_step!(x, y, DT, ws)
        add_process_noise!(x, y, rng, PROCESS_NOISE_SIGMA, DT)
    end

    raw = Array{Float32}(undef, NSAMPLES, J + 1, K)
    snap = Array{Float32}(undef, J + 1, K)

    @info "Sampling trajectory" nsamples = NSAMPLES save_every = SAVE_EVERY
    for n in 1:NSAMPLES
        for _ in 1:SAVE_EVERY
            rk4_step!(x, y, DT, ws)
            add_process_noise!(x, y, rng, PROCESS_NOISE_SIGMA, DT)
        end
        write_snapshot!(snap, x, y)
        @views raw[n, :, :] .= snap
    end

    return raw
end

function save_dataset(raw::Array{Float32,3})
    mkpath(dirname(OUTPUT_PATH))
    h5open(OUTPUT_PATH, "w") do h5
        write(h5, "timeseries", raw)
        dset = h5["timeseries"]
        attrs = attributes(dset)
        attrs["F"] = Float64(F)
        attrs["h"] = Float64(H)
        attrs["c"] = Float64(C)
        attrs["b"] = Float64(B_PARAM)
        attrs["dt"] = Float64(DT)
        attrs["K"] = Int(K)
        attrs["J"] = Int(J)
        attrs["spinup_steps"] = Int(SPINUP_STEPS)
        attrs["save_every"] = Int(SAVE_EVERY)
        attrs["process_noise_sigma"] = Float64(PROCESS_NOISE_SIGMA)
    end
    return OUTPUT_PATH
end

function save_generation_config(raw::Array{Float32,3}, out_path::AbstractString)
    PIPELINE_MODE && return ""

    cfg = Dict{String,Any}(
        "stage" => "generate_data",
        "run_dir" => RUN_DIR,
        "output_path" => abspath(out_path),
        "dataset_key" => "timeseries",
        "shape" => [size(raw, 1), size(raw, 2), size(raw, 3)],
        "parameters" => Dict(
            "K" => K,
            "J" => J,
            "F" => F,
            "h" => H,
            "c" => C,
            "b" => B_PARAM,
            "dt" => DT,
            "spinup_steps" => SPINUP_STEPS,
            "save_every" => SAVE_EVERY,
            "nsamples" => NSAMPLES,
            "rng_seed" => RNG_SEED,
            "process_noise_sigma" => PROCESS_NOISE_SIGMA,
        ),
    )
    config_path = L96RunLayout.default_config_path(RUN_DIR, "generate_data")
    L96RunLayout.write_toml_file(config_path, cfg)
    return config_path
end

raw = generate_l96_dataset()
out_path = save_dataset(raw)
config_path = save_generation_config(raw, out_path)
if !PIPELINE_MODE
    L96RunLayout.ensure_runs_readme!(L96RunLayout.default_runs_root(@__DIR__))
    manifest_path = L96RunLayout.update_run_manifest!(
        RUN_DIR;
        stage="generate_data",
        parameters=Dict(
            "K" => K,
            "J" => J,
            "F" => F,
            "h" => H,
            "c" => C,
            "b" => B_PARAM,
            "dt" => DT,
            "spinup_steps" => SPINUP_STEPS,
            "save_every" => SAVE_EVERY,
            "nsamples" => NSAMPLES,
            "rng_seed" => RNG_SEED,
            "process_noise_sigma" => PROCESS_NOISE_SIGMA,
        ),
        paths=Dict(
            "data_path" => abspath(out_path),
            "configs_dir" => abspath(dirname(config_path)),
        ),
        artifacts=Dict(
            "generate_data_config" => abspath(config_path),
            "dataset_key" => "timeseries",
        ),
        notes=Dict(
            "raw_shape" => [size(raw, 1), size(raw, 2), size(raw, 3)],
        ),
    )
    summary_path = L96RunLayout.write_run_summary!(RUN_DIR)
    index_paths = L96RunLayout.refresh_runs_index!(L96RunLayout.default_runs_root(@__DIR__))
    compat_links = L96RunLayout.update_compat_links!(@__DIR__; data_path=out_path)
    L96RunLayout.write_latest_run!(@__DIR__, RUN_DIR)

    @info "L96 dataset saved" path = out_path size = size(raw) run_dir = RUN_DIR config = config_path manifest = manifest_path summary = summary_path run_index = index_paths.index compat_links = compat_links
else
    @info "L96 dataset saved" path = out_path size = size(raw)
end
