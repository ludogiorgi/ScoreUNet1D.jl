"""
    RunnerUtils

Generic utilities for running training, integration, and analysis scripts.
These utilities are system-agnostic and can be reused across different dynamical systems.
"""
module RunnerUtils

using TOML
using BSON
using HDF5
using Dates
using Random
using LinearAlgebra
using SparseArrays
using Flux

export load_config, resolve_path, save_model, load_model,
       activation_from_string, symbol_from_string, ensure_dir,
       load_phi_sigma, save_phi_sigma, timed, verbose_log,
       create_run_directory

#─────────────────────────────────────────────────────────────────────────────
# Path and Config Utilities
#─────────────────────────────────────────────────────────────────────────────

"""
    load_config(path::AbstractString) -> Dict{String,Any}

Load and parse a TOML configuration file.
"""
function load_config(path::AbstractString)
    isfile(path) || error("Config file not found: $path")
    return TOML.parsefile(path)
end

"""
    resolve_path(path::AbstractString, project_root::AbstractString) -> String

Convert relative paths to absolute using project_root as base.
"""
function resolve_path(path::AbstractString, project_root::AbstractString)
    isabspath(path) ? path : normpath(joinpath(project_root, path))
end

"""
    ensure_dir(path::AbstractString) -> String

Create directory if it doesn't exist. Returns the path.
"""
function ensure_dir(path::AbstractString)
    mkpath(path)
    return path
end

"""
    create_run_directory(run_root::AbstractString; prefix::String="run") -> String

Create a timestamped run directory with a random slug.
"""
function create_run_directory(run_root::AbstractString; prefix::String="run")
    mkpath(run_root)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    slug = Random.randstring(4)
    run_dir = joinpath(run_root, "$(prefix)_$(timestamp)_$(slug)")
    mkpath(run_dir)
    return run_dir
end

#─────────────────────────────────────────────────────────────────────────────
# Model I/O
#─────────────────────────────────────────────────────────────────────────────

"""
    save_model(path, model; cfg=nothing, trainer_cfg=nothing)

Save a trained model to BSON format with optional config metadata.
"""
function save_model(path::AbstractString, model; 
                    cfg=nothing, trainer_cfg=nothing)
    mkpath(dirname(path))
    payload = Dict(:model => Flux.cpu(model))
    cfg !== nothing && (payload[:cfg] = cfg)
    trainer_cfg !== nothing && (payload[:trainer_cfg] = trainer_cfg)
    BSON.@save path payload
    return path
end

"""
    load_model(path::AbstractString) -> (model, cfg, trainer_cfg)

Load a model from BSON. Returns (model, cfg, trainer_cfg) where cfg and 
trainer_cfg may be nothing if not saved.
"""
function load_model(path::AbstractString)
    isfile(path) || error("Model file not found: $path")
    contents = BSON.load(path)
    model = haskey(contents, :payload) ? contents[:payload][:model] : contents[:model]
    cfg = get(contents, :cfg, nothing)
    trainer_cfg = get(contents, :trainer_cfg, nothing)
    Flux.testmode!(model)
    return model, cfg, trainer_cfg
end

#─────────────────────────────────────────────────────────────────────────────
# Phi/Sigma I/O
#─────────────────────────────────────────────────────────────────────────────

"""
    load_phi_sigma(path::AbstractString) -> (alpha, Phi, Sigma, aux)

Load Φ and Σ matrices from an HDF5 file. Returns (alpha, Phi, Sigma, aux_dict).

The returned `aux` dictionary may include additional datasets when present
(e.g. `:Q`, `:pi_vec`, `:dt`, `:V_data`).
"""
function load_phi_sigma(path::AbstractString)
    isfile(path) || error("Phi/Sigma file not found: $path")
    h5open(path, "r") do h5
        @assert haskey(h5, "Phi") "Dataset 'Phi' not found in $path"
        @assert haskey(h5, "Sigma") "Dataset 'Sigma' not found in $path"
        alpha = haskey(h5, "Alpha") ? read(h5, "Alpha") : 1.0
        Phi = read(h5, "Phi")
        Sigma = read(h5, "Sigma")
        aux = Dict{Symbol,Any}()
        if haskey(h5, "Q")
            aux[:Q] = read(h5, "Q")
        elseif haskey(h5, "Q_csc")
            g = h5["Q_csc"]
            m = Int(read(g, "m"))
            n = Int(read(g, "n"))
            colptr = Int.(read(g, "colptr"))
            rowval = Int.(read(g, "rowval"))
            nzval = Float64.(read(g, "values"))
            aux[:Q] = SparseMatrixCSC(m, n, colptr, rowval, nzval)
        end
        haskey(h5, "pi_vec") && (aux[:pi_vec] = read(h5, "pi_vec"))
        haskey(h5, "dt") && (aux[:dt] = read(h5, "dt"))
        haskey(h5, "V_data") && (aux[:V_data] = read(h5, "V_data"))
        return alpha, Phi, Sigma, aux
    end
end

"""
    save_phi_sigma(path, Phi, Sigma; alpha=1.0, Q=nothing, pi_vec=nothing, dt=nothing, V_data=nothing)

Save Φ and Σ matrices to an HDF5 file.

When provided, `V_data` stores the Stein matrix estimate `E[s(y) yᵀ]` used
internally by the estimator so downstream plotting can be consistent.
"""
function save_phi_sigma(path::AbstractString, Phi, Sigma;
                        alpha::Real=1.0, Q=nothing, pi_vec=nothing, dt=nothing, V_data=nothing)
    mkpath(dirname(path))
    h5open(path, "w") do h5
        write(h5, "Alpha", Float64(alpha))
        write(h5, "Phi", Float64.(Phi))
        write(h5, "Sigma", Float64.(Sigma))
        if Q !== nothing
            if Q isa SparseMatrixCSC
                g = create_group(h5, "Q_csc")
                write(g, "m", size(Q, 1))
                write(g, "n", size(Q, 2))
                write(g, "colptr", Int.(Q.colptr))
                write(g, "rowval", Int.(Q.rowval))
                write(g, "values", Float64.(Q.nzval))
            else
                write(h5, "Q", Float64.(Q))
            end
        end
        pi_vec !== nothing && write(h5, "pi_vec", Float64.(pi_vec))
        dt !== nothing && write(h5, "dt", Float64(dt))
        V_data !== nothing && write(h5, "V_data", Float64.(V_data))
    end
    return path
end

#─────────────────────────────────────────────────────────────────────────────
# Config Parsing Utilities
#─────────────────────────────────────────────────────────────────────────────

"""
    symbol_from_string(value::AbstractString) -> Symbol

Convert a string to a lowercase Symbol.
"""
function symbol_from_string(value::AbstractString)
    return Symbol(lowercase(value))
end

"""
    activation_from_string(name::AbstractString) -> Function

Parse activation function name to Flux function.
"""
function activation_from_string(name::AbstractString)
    lname = lowercase(name)
    lname == "swish" && return Flux.swish
    lname == "gelu" && return Flux.gelu
    lname == "relu" && return Flux.relu
    lname == "tanh" && return tanh
    lname == "identity" && return identity
    lname == "softplus" && return Flux.softplus
    error("Unsupported activation: $name")
end

#─────────────────────────────────────────────────────────────────────────────
# Logging Utilities
#─────────────────────────────────────────────────────────────────────────────

"""
    verbose_log(verbose::Bool, message::AbstractString; kwargs...)

Log a message if verbose mode is enabled.
"""
function verbose_log(verbose::Bool, message::AbstractString; kwargs...)
    verbose || return
    @info message kwargs...
end

"""
    timed(label::AbstractString, verbose::Bool, f::Function)

Execute function f and log timing if verbose.
"""
function timed(label::AbstractString, verbose::Bool, f::Function)
    verbose_log(verbose, "$label started")
    t0 = time_ns()
    result = f()
    elapsed = (time_ns() - t0) / 1e9
    verbose_log(verbose, "$label finished"; seconds=elapsed)
    return result
end

timed(f::Function, label::AbstractString, verbose::Bool) = timed(label, verbose, f)

end # module
