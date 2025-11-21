using LinearAlgebra
using Random
using Statistics

Base.@kwdef mutable struct MomentMatchingConfig
    dt::Float64 = 1.0
    max_samples::Int = 20_000
    stride::Int = 1
    batch_size::Int = 256
    min_eig::Float64 = 1e-6
    seed::Int = 0
end

Base.@kwdef mutable struct CorrelationConfig
    dt::Float64 = 1.0
    max_lag::Int = 512
    stride::Int = 1
    threshold::Float64 = exp(-1)
    multiple::Float64 = 5.0
end

struct DriftDiffusionEstimate
    phi::Matrix{Float64}
    sigma::Matrix{Float64}
    samples::Int
end

struct CorrelationInfo
    observed::Vector{Float64}
    time::Vector{Float64}
    decorrelation_time::Float64
    plot_lag::Int
    dt::Float64
end

"""
    correct_discrete_time_bias(Φ, acf_lag1, dt; verbose=false)

Corrects the drift matrix Φ for discrete-time bias in finite-difference estimation.

When estimating Φ from discrete-time data using finite differences, the resulting
estimate is systematically biased, especially when dynamics are fast relative to
the timestep. This function applies an empirical correction based on the observed
autocorrelation at lag 1.

The correction factor is: -log(ρ) / (ρ - 1), where ρ = ACF(dt).
This accounts for the difference between discrete-time transitions and
continuous-time derivatives.
"""
function correct_discrete_time_bias(Φ::Matrix{Float64},
                                    acf_lag1::Float64,
                                    dt::Float64;
                                    verbose::Bool=false)
    # Ensure ACF is in valid range
    ρ = clamp(acf_lag1, 1e-6, 1.0 - 1e-6)

    # Correction factor: log(ρ) / (ρ - 1)
    # For ρ → 1 (slow dynamics): factor → 1 (no correction needed)
    # For ρ → 0 (fast dynamics): factor → +∞ (large correction needed)
    # For ρ = 0.5: factor = log(0.5) / (0.5 - 1) = -0.693 / -0.5 ≈ 1.386
    correction_factor = log(ρ) / (ρ - 1.0)

    if verbose
        @info "Discrete-time bias correction" acf_lag1=ρ correction_factor=correction_factor
    end

    # Decompose into symmetric and antisymmetric parts
    S = (Φ + transpose(Φ)) / 2
    A = (Φ - transpose(Φ)) / 2

    # Apply correction to symmetric part (conservative dynamics)
    # The symmetric part relates to diffusion and relaxation timescales
    S_corrected = S * correction_factor

    # Antisymmetric part (circulatory dynamics) represents divergence-free flow
    # For now, apply the same correction, as it also affects timescales
    A_corrected = A * correction_factor

    Φ_corrected = S_corrected + A_corrected

    return Φ_corrected
end

function compute_drift_diffusion(model,
                                 dataset::NormalizedDataset,
                                 trainer_cfg::ScoreTrainerConfig,
                                 cfg::MomentMatchingConfig,
                                 corr_info::Union{Nothing,CorrelationInfo}=nothing;
                                 apply_correction::Bool=true,
                                 verbose::Bool=false)
    L = sample_length(dataset)
    C = num_channels(dataset)
    total_steps = size(dataset.data, 3)
    total_steps > 2 || error("Dataset must have at least 3 samples to estimate derivatives")
    stride = max(cfg.stride, 1)
    idxs = collect(2:stride:(total_steps - 1))
    isempty(idxs) && error("No samples available after applying stride=$(cfg.stride)")
    rng = MersenneTwister(cfg.seed)
    Random.shuffle!(rng, idxs)
    if cfg.max_samples > 0
        nsel = min(cfg.max_samples, length(idxs))
        idxs = idxs[1:nsel]
    end
    batch_size = max(1, min(cfg.batch_size, length(idxs)))
    D = L * C
    M = zeros(Float64, D, D)
    V = zeros(Float64, D, D)
    processed = 0
    tmp_batch = Array{Float32}(undef, L, C, batch_size)
    for chunk in Iterators.partition(idxs, batch_size)
        b = length(chunk)
        x_mat = Array{Float64}(undef, D, b)
        dxdt_mat = Array{Float64}(undef, D, b)
        for (j, idx) in enumerate(chunk)
            view_current = view(dataset.data, :, :, idx)
            view_prev = view(dataset.data, :, :, idx - 1)
            view_next = view(dataset.data, :, :, idx + 1)
            tmp_batch[:, :, j] .= view_current
            x_mat[:, j] .= reshape(view_current, D)
            prev_vec = reshape(view_prev, D)
            next_vec = reshape(view_next, D)
            dxdt_mat[:, j] .= (next_vec .- prev_vec) ./ (2 * cfg.dt)
        end
        scores = score_from_model(model, view(tmp_batch, :, :, 1:b), trainer_cfg.sigma)
        score_mat = Float64.(reshape(scores, D, b))
        M .+= dxdt_mat * transpose(x_mat)
        V .+= score_mat * transpose(x_mat)
        processed += b
    end
    processed > 0 || error("No samples processed while estimating drift and diffusion")
    M ./= processed
    V ./= processed
    Φ = M * pinv(transpose(V); rtol=1e-8)

    # Apply discrete-time bias correction if ACF data is available
    if apply_correction && corr_info !== nothing && length(corr_info.observed) >= 2
        acf_lag1 = corr_info.observed[2]  # Index 1 is lag 0, index 2 is lag 1
        Φ = correct_discrete_time_bias(Φ, acf_lag1, cfg.dt; verbose=verbose)
    elseif apply_correction && corr_info !== nothing
        @warn "Insufficient ACF data for bias correction; using uncorrected Φ"
    end

    S = (Φ + transpose(Φ)) ./ 2
    evals, evecs = eigen(Symmetric(S))
    min_eig = max(cfg.min_eig, 0.0)
    clamped = map(x -> x < min_eig ? min_eig : x, evals)
    S_pos = evecs * Diagonal(clamped) * transpose(evecs)
    S_pos = Symmetric((S_pos + transpose(S_pos)) ./ 2)
    shift = min_eig
    ch = nothing
    for attempt in 1:6
        try
            ch = cholesky(S_pos + shift * I, check=true)
            break
        catch
            shift = shift == 0 ? min_eig : shift * 10
        end
    end
    ch === nothing && error("Failed to compute SPD diffusion factor for moment matching")
    Sigma = Matrix(ch.L)
    return DriftDiffusionEstimate(Matrix(Φ), Sigma, processed)
end

function average_mode_acf(values::AbstractMatrix{<:Real}, max_lag::Int)
    D, T = size(values)
    max_lag = min(max_lag, T - 1)
    max_lag >= 0 || error("Insufficient samples to compute autocorrelation")
    acf = zeros(Float64, max_lag + 1)
    for d in 1:D
        series = values[d, :]
        μ = mean(series)
        centered = series .- μ
        variance = sum(abs2, centered) / max(T, 1)
        variance <= eps(Float64) && continue
        for lag in 0:max_lag
            total = T - lag
            total <= 0 && break
            acf[lag + 1] += dot(view(centered, 1:total),
                                view(centered, 1 + lag:lag + total)) / (total * variance)
        end
    end
    acf ./= D
    return acf
end

function compute_correlation_info(dataset::NormalizedDataset,
                                  cfg::CorrelationConfig)
    L = sample_length(dataset)
    C = num_channels(dataset)
    stride = max(cfg.stride, 1)
    full_matrix = reshape(dataset.data, L * C, :)
    subsampled = full_matrix[:, 1:stride:end]
    samples = size(subsampled, 2)
    samples > 1 || error("Not enough samples to compute autocorrelation")
    max_lag = min(cfg.max_lag, samples - 1)
    acf_full = average_mode_acf(subsampled, max_lag)
    idx = findfirst(i -> acf_full[i] <= cfg.threshold, 1:length(acf_full))
    decor_idx = idx === nothing ? length(acf_full) : idx
    decor_time = (decor_idx - 1) * cfg.dt
    target_time = min(cfg.multiple * decor_time, max_lag * cfg.dt)
    plot_lag = clamp(ceil(Int, target_time / cfg.dt), 1, max_lag)
    time_axis = (0:plot_lag) .* cfg.dt
    observed = acf_full[1:plot_lag + 1]
    return CorrelationInfo(observed, time_axis, decor_time, plot_lag, cfg.dt)
end
