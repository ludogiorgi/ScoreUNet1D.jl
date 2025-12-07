module PhiSigmaEstimator

using LinearAlgebra
using Statistics
using ProgressMeter
using MarkovChainHammer
using MarkovChainHammer.TransitionMatrix: generator
using StateSpacePartitions
using StateSpacePartitions.Trees
if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"  # headless GR
end
using Plots
using ..ScoreUNet1D: score_from_model
const Threads = Base.Threads

# -----------------------
# Core utilities
# -----------------------

@inline function empirical_acf(x::AbstractVector, max_lag::Int; normalize::Bool = false)
    x̄ = mean(x)
    xc = x .- x̄
    var0 = dot(xc, xc) / max(length(xc) - 1, 1)

    acf = zeros(Float64, max_lag + 1)
    for lag in 0:max_lag
        n = length(xc) - lag
        n <= 1 && (acf[lag + 1] = NaN; continue)
        acf[lag + 1] = dot(@view(xc[1:n]), @view(xc[1+lag:lag+n])) / max(n - 1, 1)
    end
    return normalize && var0 > 0 ? acf ./ var0 : acf
end

function generator_acf(Q::AbstractMatrix, π::AbstractVector, g_vals::AbstractVector;
                       dt::Real, max_lag::Int, normalize::Bool = true)
    n = size(Q, 1)
    π_norm = π ./ sum(π)

    if n <= 1000
        eig = eigen(Matrix(Q))
        V, Vinv, λ = eig.vectors, inv(eig.vectors), eig.values

        μ = dot(g_vals, π_norm)
        g̃ = g_vals .- μ
        v0 = Diagonal(π_norm) * g̃
        v0̂ = Vinv * v0

        acf = zeros(ComplexF64, max_lag + 1)
        for (k, lag) in enumerate(0:max_lag)
            t = lag * dt
            ŵ = exp.(λ .* t) .* v0̂
            w = V * ŵ
            acf[k] = dot(g̃, w)
        end
        acf_real = real.(acf)
    else
        μ = dot(g_vals, π_norm)
        g_centered = g_vals .- μ
        v0 = Diagonal(π_norm) * g_centered

        expQdt = exp(dt * Matrix(Q))

        acf_real = zeros(Float64, max_lag + 1)
        acf_real[1] = dot(g_centered, v0)

        w = Vector{Float64}(v0)
        for lag in 1:max_lag
            w = expQdt * w
            acf_real[lag + 1] = real(dot(g_centered, w))
        end
    end

    return normalize ? acf_real ./ acf_real[1] : acf_real
end

# -----------------------
# Helper computations
# -----------------------

function compute_clusters(X_noisy::AbstractMatrix; q_min_prob::Real, dt::Real)
    partition = StateSpacePartition(X_noisy; method = Tree(false, q_min_prob))
    labels = partition.partitions
    n_states = maximum(labels)
    mc = reshape(labels, 1, :)
    Q = generator(mc; dt = dt)
    return labels, n_states, Q
end

function compute_centers_and_pi(X_noisy::AbstractMatrix,
                                labels::AbstractVector{<:Integer},
                                n_states::Int)
    D, T = size(X_noisy)
    nthreads = Threads.nthreads()
    centers_locals = [zeros(Float64, D, n_states) for _ in 1:nthreads]
    counts_locals = [zeros(Int, n_states) for _ in 1:nthreads]

    Threads.@threads for idx in 1:T
        tid = Threads.threadid()
        s = labels[idx]
        @views centers_locals[tid][:, s] .+= X_noisy[:, idx]
        counts_locals[tid][s] += 1
    end

    centers = zeros(Float64, D, n_states)
    counts = zeros(Int, n_states)
    for tid in 1:nthreads
        centers .+= centers_locals[tid]
        counts .+= counts_locals[tid]
    end

    for j in 1:n_states
        counts[j] > 0 && (@views centers[:, j] ./= counts[j])
    end

    pi_vec = counts ./ sum(counts)
    return centers, pi_vec
end

function compute_M(centers::AbstractMatrix, Q::AbstractMatrix, π::AbstractVector)
    π_norm = π ./ sum(π)
    return centers * Q * Diagonal(π_norm) * centers'
end

function compute_V_data(data_noisy::Array{<:Real,3},
                        model,
                        sigma::Real;
                        v_data_resolution::Int = 10,
                        batch_size::Int = 512)
    L, C, B = size(data_noisy)
    D = L * C
    data_indices = collect(1:v_data_resolution:B)
    n_batches = cld(length(data_indices), batch_size)
    nthreads = Threads.nthreads()
    V_locals = [zeros(Float64, D, D) for _ in 1:nthreads]
    counts_local = zeros(Int, nthreads)

    Threads.@threads for batch_idx in 1:n_batches
        tid = Threads.threadid()
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, length(data_indices))
        batch_selection = data_indices[start_idx:end_idx]
        batch_noisy = data_noisy[:, :, batch_selection]

        scores = score_from_model(model, batch_noisy, sigma)
        S_flat = Float64.(reshape(scores, D, :))
        Y_flat = Float64.(reshape(batch_noisy, D, :))

        V_locals[tid] .+= S_flat * Y_flat'
        counts_local[tid] += size(Y_flat, 2)
    end

    V_data = zeros(Float64, D, D)
    total = 0
    for tid in 1:nthreads
        V_data .+= V_locals[tid]
        total += counts_local[tid]
    end
    V_data ./= total
    return V_data
end

function average_component_acf_data(x::AbstractArray{<:Real,2},
                                    max_lag::Int;
                                    normalize::Bool = true)
    D, T = size(x)
    nthreads = Threads.nthreads()
    sums = [zeros(Float64, max_lag + 1) for _ in 1:nthreads]
    counts = [zeros(Int, max_lag + 1) for _ in 1:nthreads]

    Threads.@threads for i in 1:D
        tid = Threads.threadid()
        acf_i = empirical_acf(@view(x[i, :]), max_lag; normalize = normalize)
        @inbounds for k in eachindex(acf_i)
            if isfinite(acf_i[k])
                sums[tid][k] += acf_i[k]
                counts[tid][k] += 1
            end
        end
    end

    total_sum = zeros(Float64, max_lag + 1)
    total_count = zeros(Int, max_lag + 1)
    for tid in 1:nthreads
        total_sum .+= sums[tid]
        total_count .+= counts[tid]
    end
    return total_sum ./ max.(total_count, 1)
end

function average_component_acf_generator(Q::AbstractMatrix,
                                         π::AbstractVector,
                                         centers::AbstractMatrix;
                                         dt::Real,
                                         max_lag::Int)
    D, n_states = size(centers)
    nthreads = Threads.nthreads()
    sums = [zeros(Float64, max_lag + 1) for _ in 1:nthreads]
    counts = [zeros(Int, max_lag + 1) for _ in 1:nthreads]

    Threads.@threads for comp in 1:D
        tid = Threads.threadid()
        g_vals = @view centers[comp, :]
        acf_comp = generator_acf(Q, π, g_vals; dt = dt, max_lag = max_lag, normalize = true)
        @inbounds for k in eachindex(acf_comp)
            if isfinite(acf_comp[k])
                sums[tid][k] += acf_comp[k]
                counts[tid][k] += 1
            end
        end
    end

    total_sum = zeros(Float64, max_lag + 1)
    total_count = zeros(Int, max_lag + 1)
    for tid in 1:nthreads
        total_sum .+= sums[tid]
        total_count .+= counts[tid]
    end
    return total_sum ./ max.(total_count, 1)
end

# -----------------------
# Public API
# -----------------------

"""
    estimate_phi_sigma(data_clean, model, sigma; kwargs...) -> (Φ, Σ, info)

Estimate the drift matrix Φ and diffusion factor Σ from an observed time series
`data_clean` (shape `(L, C, T)`) and a score model. A Gaussian perturbation of
width `sigma` is added to the data, the state space is clustered, a generator
`Q` is built, and `M = centers * Q * Diagonal(π) * centers'` together with the
Stein matrix estimate `V_data` are used to solve `M = Φ V_data`. `Σ` is obtained
from the Cholesky factor of the symmetric part of `Φ`.

Keyword defaults mirror `scripts/test_noise_lang.jl`:
  - `resolution::Int = 1`
  - `q_min_prob::Real = 1e-4`
  - `dt_original::Real = 1.0`
  - `max_lag::Int = 50`
  - `lag_res::Int = Int(1 / dt_original)`
  - `regularization::Real = 5e-4`
  - `v_data_resolution::Int = 10`
  - `batch_size_scores::Int = 256`
  - `plot_mean_acf::Bool = false`
  - `plot_path::Union{Nothing,String} = nothing`

When `plot_mean_acf=true`, the function plots the average ACF across all state
components for the perturbed data and for the generator `Q`.
"""
function estimate_phi_sigma(data_clean::Array{<:Real,3},
                            model,
                            sigma::Real;
                            resolution::Int = 1,
                            q_min_prob::Real = 1e-4,
                            dt_original::Real = 1.0,
                            max_lag::Int = 50,
                            lag_res::Int = Int(1 / dt_original),
                            regularization::Real = 5e-4,
                            v_data_resolution::Int = 10,
                            batch_size_scores::Int = 256,
                            plot_mean_acf::Bool = false,
                            plot_path::Union{Nothing,String} = nothing)
    @assert max_lag > 0 "max_lag must be positive"
    @assert lag_res > 0 "lag_res must be at least 1"

    data_clean = data_clean[:, :, 1:resolution:end]
    L, C, B = size(data_clean)
    D = L * C
    dt = dt_original * resolution

    # Perturb data
    noise = randn(Float32, size(data_clean)...)
    data_noisy = data_clean .+ Float32(sigma) .* noise
    X_noisy = reshape(data_noisy, D, size(data_noisy, 3))

    # Cluster and build Q
    labels, n_states, Q = compute_clusters(X_noisy; q_min_prob = q_min_prob, dt = dt)
    centers, pi_vec = compute_centers_and_pi(X_noisy, labels, n_states)

    # M matrix
    M = compute_M(centers, Q, pi_vec)

    # Score at centers (batched)
    score_at_centers = zeros(Float32, D, n_states)
    batch_size = min(batch_size_scores, n_states)
    for start_idx in 1:batch_size:n_states
        end_idx = min(start_idx + batch_size - 1, n_states)
        batch_centers = centers[:, start_idx:end_idx]
        batch_input = reshape(Float32.(batch_centers), L, C, :)
        preds = model(batch_input)
        scores = -preds ./ Float32(sigma)
        score_at_centers[:, start_idx:end_idx] = reshape(scores, D, :)
    end

    # Stein matrix V_data
    V_data = compute_V_data(data_noisy, model, sigma;
                            v_data_resolution = v_data_resolution,
                            batch_size = 512)

    # Solve for Φ
    Φ = M / V_data

    # Diffusion from symmetric part
    Φ_S = 0.5 * (Φ + Φ')
    Φ_S_reg = Φ_S
    eigvals_S = eigvals(Symmetric(Φ_S_reg))
    min_eig = minimum(eigvals_S)
    if min_eig <= 0
        shift = abs(min_eig) + max(regularization, 1e-8)
        Φ_S_reg = Φ_S_reg + shift * I
    end
    chol = cholesky(Symmetric(Φ_S_reg); check = true)
    Σ = LowerTriangular(chol.L)

    # Optional mean ACF comparison
    mean_acf_data = nothing
    mean_acf_Q = nothing
    if plot_mean_acf
        lag_points = collect(0:lag_res:((max_lag - 1) * lag_res))
        τs = lag_points .* dt
        max_lag_idx = lag_points[end]

        mean_acf_data = average_component_acf_data(reshape(data_clean, D, :), max_lag_idx; normalize = true)
        mean_acf_data = mean_acf_data[lag_points .+ 1]

        mean_acf_Q = average_component_acf_generator(Q, pi_vec, centers; dt = dt, max_lag = max_lag_idx)
        mean_acf_Q = mean_acf_Q[lag_points .+ 1]

        plt = Plots.plot(τs, mean_acf_data; label = "Data (avg ACF)", color = :black, lw = 2.5,
                         xlabel = "Time lag τ", ylabel = "Average ACF",
                         title = "Average component ACF: data vs generator Q")
        Plots.plot!(plt, τs, mean_acf_Q; label = "Generator Q", color = :red, lw = 2.0, ls = :dash)
        if plot_path !== nothing
            Plots.savefig(plt, plot_path)
        else
            display(plt)
        end
    end

    info = Dict(
        :Q => Q,
        :pi_vec => pi_vec,
        :centers => centers,
        :mean_acf_data => mean_acf_data,
        :mean_acf_Q => mean_acf_Q,
        :dt => dt,
        :V_data => V_data
    )
    return Φ, Matrix(Σ), info
end

end # module
