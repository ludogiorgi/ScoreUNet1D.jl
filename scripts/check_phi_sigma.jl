#!/usr/bin/env julia

using LinearAlgebra
using HDF5

const PROJECT_ROOT = dirname(@__DIR__)

function collect_phi_sigma_paths()
    paths = String[]

    # Global phi_sigma (used by some scripts)
    global_path = joinpath(PROJECT_ROOT, "data", "phi_sigma.hdf5")
    push!(paths, global_path)

    # Model-specific phi_sigma files under data/models/**
    models_dir = joinpath(PROJECT_ROOT, "data", "models")
    if isdir(models_dir)
        for entry in readdir(models_dir; join = true)
            path = joinpath(entry, "phi_sigma.hdf5")
            push!(paths, path)
        end
    end

    # Deduplicate while preserving order
    seen = Set{String}()
    out = String[]
    for p in paths
        if !in(p, seen)
            push!(seen, p)
            push!(out, p)
        end
    end
    return out
end

function check_phi_sigma(path::AbstractString)
    if !isfile(path)
        println("Skipping $(path): file does not exist.")
        return
    end

    println("Checking Φ, Σ in: $path")
    h5open(path, "r") do h5
        @assert haskey(h5, "Phi") "Dataset 'Phi' not found in $path"
        @assert haskey(h5, "Sigma") "Dataset 'Sigma' not found in $path"

        Phi = read(h5, "Phi")
        Sigma = read(h5, "Sigma")

        # Symmetric part of Phi
        Phi_sym = 0.5 .* (Phi .+ Phi')

        # Cholesky factor of the symmetric part (lower-triangular)
        chol = cholesky(Symmetric(Phi_sym))
        L = Matrix(chol.L)

        diff = Sigma .- L
        norm_L = norm(L)
        rel_norm = norm(diff) / max(norm_L, eps())
        max_abs_diff = maximum(abs.(diff))

        println("  ‖Σ - chol(0.5*(Φ+Φ'))‖ / ‖chol‖ = $rel_norm")
        println("  max |Σ - chol(0.5*(Φ+Φ'))|      = $max_abs_diff")
        println()
    end
end

function main()
    paths = collect_phi_sigma_paths()
    isempty(paths) && println("No candidate phi_sigma.hdf5 files found.")
    for path in paths
        check_phi_sigma(path)
    end
end

main()

