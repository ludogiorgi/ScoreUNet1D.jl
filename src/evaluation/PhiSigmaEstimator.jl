module PhiSigmaEstimator

using LinearAlgebra
using SparseArrays
using Statistics
using ProgressMeter
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

@inline function empirical_acf(x::AbstractVector, max_lag::Int; normalize::Bool=false)
    x̄ = mean(x)
    xc = x .- x̄
    var0 = dot(xc, xc) / max(length(xc) - 1, 1)

    acf = zeros(Float64, max_lag + 1)
    for lag in 0:max_lag
        n = length(xc) - lag
        n <= 1 && (acf[lag+1] = NaN; continue)
        acf[lag+1] = dot(@view(xc[1:n]), @view(xc[1+lag:lag+n])) / max(n - 1, 1)
    end
    return normalize && var0 > 0 ? acf ./ var0 : acf
end

function generator_acf(Q::AbstractMatrix, π::AbstractVector, g_vals::AbstractVector;
    dt::Real, max_lag::Int, normalize::Bool=true)
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
            acf_real[lag+1] = real(dot(g_centered, w))
        end
    end

    return normalize ? acf_real ./ acf_real[1] : acf_real
end

# -----------------------
# Helper computations
# -----------------------

"""
    sparse_generator(labels::AbstractVector{<:Integer}, n_states::Int; dt::Real,
        finite_dt_correction::Bool=false, finite_dt_correction_mode::Symbol=:per_state)

Build the infinitesimal generator Q as a sparse matrix directly from transition counts.
This avoids allocating a dense n_states × n_states matrix.

Convention: we return a *column-generator* compatible with `ρ̇ = Qρ`, i.e.
`Q[j,k]` is the transition rate from state `k → j`, and each column sums to 0.
"""
function sparse_generator(labels::AbstractVector{<:Integer}, n_states::Int; dt::Real,
    finite_dt_correction::Bool=false,
    finite_dt_correction_mode::Symbol=:per_state)
    T = length(labels)

    # Count transitions: use a dictionary to accumulate (i,j) -> count
    # This is memory efficient for sparse transition patterns
    transition_counts = Dict{Tuple{Int,Int},Int}()
    # Outgoing opportunities per state (t = 1..T-1)
    state_out = zeros(Int, n_states)
    # Total number of observed departures per state (i->j with i != j)
    state_leaves = zeros(Int, n_states)

    @inbounds for t in 1:(T-1)
        i = labels[t]
        j = labels[t+1]
        state_out[i] += 1
        if i != j
            state_leaves[i] += 1
            key = (i, j)
            transition_counts[key] = get(transition_counts, key, 0) + 1
        end
    end

    # Diagnostics: if the probability of changing cluster per dt is large,
    # then using one-step transition counts to estimate a continuous-time
    # generator will systematically underestimate rates (multiple jumps can
    # occur between observations).
    total_steps = max(T - 1, 1)
    observed_jumps = 0
    for v in values(transition_counts)
        observed_jumps += v
    end
    leave_fraction = observed_jumps / total_steps
    @info "Transition statistics" dt leave_fraction observed_jumps total_steps
    if leave_fraction > 0.3
        lf = clamp(leave_fraction, 0.0, 1.0 - eps(Float64))
        correction = -log1p(-lf) / max(lf, eps(Float64))
        msg = if !finite_dt_correction
            "High cluster-switch probability per dt; generator exit rates are likely underestimated (missing multiple jumps). Consider enabling `finite_dt_correction`, using smaller dt (higher-res data), or coarser clustering."
        elseif finite_dt_correction_mode === :per_state
            "High cluster-switch probability per dt; applying finite-Δt correction to exit rates using p_stay = exp(q_ii dt)."
        elseif finite_dt_correction_mode === :mean_diag_scale
            "High cluster-switch probability per dt; applying finite-Δt global scaling of Q to match the mean diagonal exit rate of the per-state corrected generator."
        else
            "High cluster-switch probability per dt; finite-Δt correction enabled."
        end
        @warn msg dt leave_fraction correction
    end

    # Optional finite-Δt corrections:
    #
    # (1) `finite_dt_correction_mode = :per_state` rescales each column i by
    #     κ_i = -log(p_stay_i) / (1 - p_stay_i),
    #     where p_stay_i is the observed one-step probability to remain in i.
    #
    # (2) `finite_dt_correction_mode = :mean_diag_scale` computes
    #     alpha_naive = mean(diag(Q_naive)) and alpha_corr = mean(diag(Q_per_state)),
    #     then rescales the naive generator globally by alpha_corr/alpha_naive.
    #     This matches the *mean* exit-rate scale while preserving naive destinations.
    col_scale = ones(Float64, n_states)
    if finite_dt_correction
        finite_dt_correction_mode in (:per_state, :mean_diag_scale) ||
            error("finite_dt_correction_mode must be :per_state or :mean_diag_scale (got $finite_dt_correction_mode)")

        if finite_dt_correction_mode === :per_state
            # Compute per-state P_ii from outgoing counts
            @inbounds for i in 1:n_states
                n_out = state_out[i]
                n_out == 0 && continue
                n_leave = state_leaves[i]
                n_stay = n_out - n_leave
                # Clamp to avoid log(0) and division by 0
                p_stay = clamp(n_stay / n_out, eps(Float64), 1.0)
                p_leave = clamp(1.0 - p_stay, eps(Float64), 1.0)
                col_scale[i] = -log(p_stay) / p_leave
            end
            # Report the average multiplicative correction (weighted by visits)
            w = Float64.(state_out)
            wsum = sum(w)
            avg_scale = wsum > 0 ? dot(w, col_scale) / wsum : mean(col_scale)
            @info "Finite-Δt correction enabled" mode=finite_dt_correction_mode avg_exit_rate_scale=avg_scale
        elseif finite_dt_correction_mode === :mean_diag_scale
            alpha_naive = 0.0
            alpha_corr = 0.0
            @inbounds for i in 1:n_states
                n_out = state_out[i]
                if n_out == 0
                    continue
                end
                n_leave = state_leaves[i]
                # Naive diagonal: q_ii = -p_leave/dt
                p_leave = clamp(n_leave / n_out, 0.0, 1.0)
                alpha_naive += -(p_leave / dt)

                # Per-state corrected diagonal: q_ii = log(p_stay)/dt
                p_stay = clamp(1.0 - p_leave, eps(Float64), 1.0)
                alpha_corr += log(p_stay) / dt
            end
            alpha_naive /= n_states
            alpha_corr /= n_states

            abs(alpha_naive) > 0 || error("Cannot apply :mean_diag_scale: mean(diag(Q_naive)) is zero.")
            global_scale = alpha_corr / alpha_naive
            global_scale > 0 || @warn "Computed non-positive Q scaling factor; check transition statistics" global_scale alpha_naive alpha_corr
            fill!(col_scale, global_scale)
            @info "Finite-Δt correction enabled" mode=finite_dt_correction_mode alpha_naive alpha_corr q_scale=global_scale
        end
    end

    # Build sparse matrix in COO format
    n_transitions = length(transition_counts)
    # We need n_transitions off-diagonal + n_states diagonal entries
    I = Vector{Int}(undef, n_transitions + n_states)
    J = Vector{Int}(undef, n_transitions + n_states)
    V = Vector{Float64}(undef, n_transitions + n_states)

    # Off-diagonal (column-generator): Q[j,i] = count[i→j] / (time_in_state_i * dt)
    # so each column `i` contains the rates *leaving* state `i`.
    col_sums = zeros(Float64, n_states)
    idx = 1
    for ((i, j), count) in transition_counts
        denom = state_out[i] * dt
        rate = denom > 0 ? (col_scale[i] * count) / denom : 0.0
        I[idx] = j
        J[idx] = i
        V[idx] = rate
        col_sums[i] += rate
        idx += 1
    end

    # Diagonal: Q[i,i] = -sum of column i (probability conservation)
    for i in 1:n_states
        I[idx] = i
        J[idx] = i
        V[idx] = -col_sums[i]
        idx += 1
    end

    return sparse(I, J, V, n_states, n_states)
end

function compute_clusters(X_noisy::AbstractMatrix; q_min_prob::Real, dt::Real,
    finite_dt_correction::Bool=false,
    finite_dt_correction_mode::Symbol=:per_state)
    partition = StateSpacePartition(X_noisy; method=Tree(false, q_min_prob))
    labels = partition.partitions
    n_states = maximum(labels)
    # Build Q directly as sparse - avoids dense n_states² allocation
    Q = sparse_generator(labels, n_states;
        dt=dt,
        finite_dt_correction=finite_dt_correction,
        finite_dt_correction_mode=finite_dt_correction_mode)
    return labels, n_states, Q
end

function compute_centers_and_pi(X_noisy::AbstractMatrix,
    labels::AbstractVector{<:Integer},
    n_states::Int)
    D, T = size(X_noisy)

    # Memory-efficient: single accumulator instead of nthreads copies
    # For n_states=50k, D=64, nthreads=32: saves ~80 GB RAM
    centers = zeros(Float64, D, n_states)
    counts = zeros(Int, n_states)

    # Sequential accumulation - memory-bandwidth bound anyway
    @inbounds for idx in 1:T
        s = labels[idx]
        @views centers[:, s] .+= X_noisy[:, idx]
        counts[s] += 1
    end

    @inbounds for j in 1:n_states
        counts[j] > 0 && (@views centers[:, j] ./= counts[j])
    end

    pi_vec = counts ./ sum(counts)
    return centers, pi_vec
end

function compute_M(centers::AbstractMatrix, Q::AbstractMatrix, π::AbstractVector)
    π_norm = π ./ sum(π)
    # For sparse Q: centers * Q is (D, n_states), then scale columns, then multiply by centers'
    # This avoids creating a dense (n_states, n_states) intermediate
    CQ = centers * Q  # (D, n_states) - sparse mult keeps result reasonable
    CQ_scaled = CQ * Diagonal(π_norm)  # (D, n_states)
    return CQ_scaled * centers'  # (D, D)
end

function compute_V_data(data_noisy::Array{<:Real,3},
    model,
    sigma::Real;
    v_data_resolution::Int=10,
    batch_size::Int=512)
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
    normalize::Bool=true)
    D, T = size(x)
    nthreads = Threads.nthreads()
    sums = [zeros(Float64, max_lag + 1) for _ in 1:nthreads]
    counts = [zeros(Int, max_lag + 1) for _ in 1:nthreads]

    Threads.@threads for i in 1:D
        tid = Threads.threadid()
        acf_i = empirical_acf(@view(x[i, :]), max_lag; normalize=normalize)
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

"""
    average_acf_ensemble(traj::Array{<:Real,4}, max_lag::Int) -> Vector{Float64}

Compute averaged ACF over ensembles and dimensions for a 4D trajectory tensor.

Input shape: `(L, C, T, E)` where:
- `L` = spatial dimension
- `C` = channels
- `T` = time snapshots per ensemble
- `E` = number of ensembles

For each ensemble `e` and each dimension `d`, we compute a normalized ACF of length
`max_lag + 1`. We then average over all `D × E` individual ACFs, where `D = L × C`.

This ensures that ACF at lag 0 equals 1.0, and that we do not incorrectly treat
concatenated ensembles as a single continuous trajectory.
"""
function average_acf_ensemble(traj::Array{<:Real,4}, max_lag::Int)
    L, C, T, E = size(traj)
    D = L * C
    nthreads = Threads.nthreads()
    sums = [zeros(Float64, max_lag + 1) for _ in 1:nthreads]
    counts = [zeros(Int, max_lag + 1) for _ in 1:nthreads]

    # Reshape to (D, T, E) for easier iteration
    traj_flat = reshape(traj, D, T, E)

    # Create list of all (dim, ensemble) pairs
    pairs = [(d, e) for d in 1:D, e in 1:E]

    Threads.@threads for idx in eachindex(pairs)
        d, e = pairs[idx]
        tid = Threads.threadid()
        series = @view traj_flat[d, :, e]
        acf_de = empirical_acf(series, max_lag; normalize=true)
        @inbounds for k in eachindex(acf_de)
            if isfinite(acf_de[k])
                sums[tid][k] += acf_de[k]
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

"""
    average_acf_3d(data::Array{<:Real,3}, max_lag::Int) -> Vector{Float64}

Compute averaged ACF over dimensions for a 3D data tensor (single trajectory).

Input shape: `(L, C, T)` where:
- `L` = spatial dimension
- `C` = channels
- `T` = time samples

This is used for reference data that is a single long trajectory.
"""
function average_acf_3d(data::Array{<:Real,3}, max_lag::Int)
    L, C, T = size(data)
    D = L * C
    # Reshape to (D, T) and use the 2D function
    data_flat = reshape(data, D, T)
    return average_component_acf_data(data_flat, max_lag; normalize=true)
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
        acf_comp = generator_acf(Q, π, g_vals; dt=dt, max_lag=max_lag, normalize=true)
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
  - `finite_dt_correction::Bool = false`
  - `finite_dt_correction_mode::Symbol = :per_state`
  - `max_lag::Int = 50`
  - `lag_res::Int = Int(1 / dt_original)`
  - `regularization::Real = 5e-4`
  - `v_data_resolution::Int = 10`
  - `plot_mean_acf::Bool = false`
  - `plot_path::Union{Nothing,String} = nothing`

When `plot_mean_acf=true`, the function plots the average ACF across all state
components for the perturbed data and for the generator `Q`.

Finite-Δt correction modes (enabled when `finite_dt_correction=true`):
  - `:per_state` rescales each state's exit rate using the observed `p_stay`.
  - `:mean_diag_scale` rescales the naive generator globally so that
    `mean(diag(Q))` matches the per-state corrected generator.
"""
function estimate_phi_sigma(data_clean::Array{<:Real,3},
    model,
    sigma::Real;
    resolution::Int=1,
    q_min_prob::Real=1e-4,
    dt_original::Real=1.0,
    finite_dt_correction::Bool=false,
    finite_dt_correction_mode::Symbol=:per_state,
    max_lag::Int=50,
    lag_res::Int=Int(1 / dt_original),
    regularization::Real=5e-4,
    v_data_resolution::Int=10,
    plot_mean_acf::Bool=false,
    plot_path::Union{Nothing,String}=nothing)
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
    labels, n_states, Q = compute_clusters(X_noisy;
        q_min_prob=q_min_prob,
        dt=dt,
        finite_dt_correction=finite_dt_correction,
        finite_dt_correction_mode=finite_dt_correction_mode)

    # Log n_states and memory estimate
    Q_nnz = nnz(Q)
    Q_mem_mb = (Q_nnz * 8 + (n_states + 1) * 8 + Q_nnz * 4) / 1e6  # CSC format: values + colptr + rowval
    centers_mem_mb = (D * n_states * 8) / 1e6
    @info "Clustering complete" n_states Q_nnz Q_mem_MB = round(Q_mem_mb, digits=1) centers_mem_MB = round(centers_mem_mb, digits=1)

    centers, pi_vec = compute_centers_and_pi(X_noisy, labels, n_states)

    # M matrix
    M = compute_M(centers, Q, pi_vec)

    # Check if M is negative semi-definite
    M_sym = 0.5 * (M + M')
    max_eig_M = maximum(eigvals(Symmetric(M_sym)))
    if max_eig_M > 0
        @warn "M matrix is not negative semi-definite" max_eigenvalue = max_eig_M
    end

    # Stein matrix V_data
    V_data = compute_V_data(data_noisy, model, sigma;
        v_data_resolution=v_data_resolution,
        batch_size=512)

    # Diagnostic: Stein identity check. For correctly scaled data and score,
    # E[s(y) yᵀ] should be approximately -I when y ~ p_σ.
    diag_V = diag(V_data)
    diag_mean = mean(diag_V)
    diag_std = std(diag_V)
    @info "Stein matrix diag stats" mean=diag_mean std=diag_std min=minimum(diag_V) max=maximum(diag_V)
    # Keep the warning threshold loose: imperfect score models and small sample
    # sizes can bias this estimate. We mainly want to catch gross scaling issues.
    if !(isfinite(diag_mean)) || abs(diag_mean + 1) > 0.5 || maximum(abs.(diag_V)) > 10
        @warn "Stein matrix deviates from -I (check data normalization and sigma consistency)" sigma dt
    end

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
    chol = cholesky(Symmetric(Φ_S_reg); check=true)
    Σ = LowerTriangular(chol.L)

    # Optional mean ACF comparison
    mean_acf_data = nothing
    mean_acf_Q = nothing
    if plot_mean_acf
        lag_points = collect(0:lag_res:((max_lag-1)*lag_res))
        τs = lag_points .* dt
        max_lag_idx = lag_points[end]

        mean_acf_data = average_component_acf_data(reshape(data_clean, D, :), max_lag_idx; normalize=true)
        mean_acf_data = mean_acf_data[lag_points.+1]

        mean_acf_Q = average_component_acf_generator(Q, pi_vec, centers; dt=dt, max_lag=max_lag_idx)
        mean_acf_Q = mean_acf_Q[lag_points.+1]

        plt = Plots.plot(τs, mean_acf_data; label="Data (avg ACF)", color=:black, lw=2.5,
            xlabel="Time lag τ", ylabel="Average ACF",
            title="Average component ACF: data vs generator Q")
        Plots.plot!(plt, τs, mean_acf_Q; label="Generator Q", color=:red, lw=2.0, ls=:dash)
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
        :V_data => V_data,
        :finite_dt_correction => finite_dt_correction,
        :finite_dt_correction_mode => finite_dt_correction_mode
    )
    return Φ, Matrix(Σ), info
end

end # module
