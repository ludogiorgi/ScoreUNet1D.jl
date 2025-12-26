# ==============================================================================
# Generate L96: Two-Scale Lorenz-96 Data Generation (Slow Variables Only)
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/L96/generate_data_l96.jl
#
# This script simulates the two-scale Lorenz-96 system using RK4 and saves
# only the slow variables X to an HDF5 file.
#
# ==============================================================================

using Random
using HDF5

# ----------------------------
# System Parameters (Fixed)
# ----------------------------
const K = 36
const J = 10
const F_X = 10.0  # Forcing for slow variables (X)
const F_Y = 10.0   # Forcing for fast variables (Y)
const c = 10.0
const b = 10.0
const h = 3.0
const hcb = h * c / b
const cb = c * b

# ----------------------------
# Integration Parameters
# ----------------------------
const dt = 0.001
const transient_steps = 100_000
const save_stride = 10
const n_save = 200_000
const seed = 42

# ----------------------------
# Output
# ----------------------------
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "data", "L96")
const OUTPUT_PATH = joinpath(OUTPUT_DIR, "new_l96.hdf5")
const DATASET_KEY = "timeseries"

@inline idxp1(i, n) = i == n ? 1 : i + 1
@inline idxm1(i, n) = i == 1 ? n : i - 1
@inline idxm2(i, n) = i <= 2 ? i + n - 2 : i - 2

function l96_rhs!(dX::AbstractVector, dY::AbstractMatrix,
    X::AbstractVector, Y::AbstractMatrix)
    @inbounds for k in 1:K
        kp1 = idxp1(k, K)
        km1 = idxm1(k, K)
        km2 = idxm2(k, K)
        sumY = 0.0
        for j in 1:J
            sumY += Y[k, j]
        end
        dX[k] = (X[kp1] - X[km2]) * X[km1] - X[k] + F_X - hcb * sumY
    end

    @inbounds for k in 1:K
        xk = X[k]
        for j in 1:J
            jp1 = idxp1(j, J)
            jm1 = idxm1(j, J)
            jm2 = idxm2(j, J)
            dY[k, j] = cb * (Y[k, jp1] - Y[k, jm2]) * Y[k, jm1] - c * Y[k, j] + F_Y + hcb * xk
        end
    end
    return nothing
end

function rk4_step!(X::AbstractVector, Y::AbstractMatrix, dt::Float64, work)
    dX1, dY1, dX2, dY2, dX3, dY3, dX4, dY4, Xtmp, Ytmp = work
    l96_rhs!(dX1, dY1, X, Y)
    @. Xtmp = X + 0.5 * dt * dX1
    @. Ytmp = Y + 0.5 * dt * dY1
    l96_rhs!(dX2, dY2, Xtmp, Ytmp)
    @. Xtmp = X + 0.5 * dt * dX2
    @. Ytmp = Y + 0.5 * dt * dY2
    l96_rhs!(dX3, dY3, Xtmp, Ytmp)
    @. Xtmp = X + dt * dX3
    @. Ytmp = Y + dt * dY3
    l96_rhs!(dX4, dY4, Xtmp, Ytmp)

    @. X += (dt / 6) * (dX1 + 2 * dX2 + 2 * dX3 + dX4)
    @. Y += (dt / 6) * (dY1 + 2 * dY2 + 2 * dY3 + dY4)
    return nothing
end

function allocate_work()
    dX1 = zeros(K)
    dX2 = similar(dX1)
    dX3 = similar(dX1)
    dX4 = similar(dX1)
    Xtmp = similar(dX1)
    dY1 = zeros(K, J)
    dY2 = similar(dY1)
    dY3 = similar(dY1)
    dY4 = similar(dY1)
    Ytmp = similar(dY1)
    return (dX1, dY1, dX2, dY2, dX3, dY3, dX4, dY4, Xtmp, Ytmp)
end

function main()
    Random.seed!(seed)
    X = F_X .+ 0.01 .* randn(K)
    Y = 0.01 .* randn(K, J)

    work = allocate_work()
    total_steps = transient_steps + n_save * save_stride

    @info "Starting L96 simulation" K J F_X F_Y dt transient_steps save_stride n_save total_steps output = OUTPUT_PATH

    X_save = Array{Float32}(undef, K, n_save)
    sample_idx = 0

    for step in 1:total_steps
        rk4_step!(X, Y, dt, work)
        if step > transient_steps && (step - transient_steps) % save_stride == 0
            sample_idx += 1
            @inbounds for k in 1:K
                X_save[k, sample_idx] = Float32(X[k])
            end
        end
    end

    @assert sample_idx == n_save "Sample count mismatch: expected $n_save, got $sample_idx"

    mkpath(OUTPUT_DIR)
    h5open(OUTPUT_PATH, "w") do file
        write(file, DATASET_KEY, X_save)
    end

    @info "Saved L96 dataset" path = OUTPUT_PATH key = DATASET_KEY size = size(X_save) dt_effective = dt * save_stride
end

main()
