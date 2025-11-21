#!/usr/bin/env julia
using Plots
using KernelDensity
using BSON
using CairoMakie
using Printf
using ScoreUNet1D
using ScoreUNet1D: sample_length, num_channels
using Statistics, LinearAlgebra
using Random
using Optim
using FastSDE
using Base.Threads
using Base.Iterators: partition

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const MODEL_PATH = joinpath(PROJECT_ROOT, "scripts", "trained_model.bson")
const DATA_DT = 1.0
const MM_MAX_SAMPLES = 4_096
const MM_STRIDE = 10
const MM_BATCH_SIZE = 256
const MM_MIN_EIG = 1e-6
const LOW_DATA_PATH = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
const HR_DATA_PATH = joinpath(PROJECT_ROOT, "data", "new_ks_hr.hdf5")
const DATASET_KEY = "timeseries"
const HEATMAP_BINS = 192
const BIVARIATE_OFFSETS = (1, 2, 3)

function load_datasets()
    base = load_hdf5_dataset(LOW_DATA_PATH;
                             dataset_key=DATASET_KEY,
                             samples_orientation=:columns)
    hr_raw = load_hdf5_dataset(HR_DATA_PATH;
                               dataset_key=DATASET_KEY,
                               samples_orientation=:columns,
                               normalize=false)
    hr_norm = ScoreUNet1D.apply_stats(hr_raw.data, base.stats)
    ratio = max(size(hr_norm, 3) ÷ max(size(base.data, 3), 1), 1)
    if ratio > 1
        @info "Downsampling high-resolution dataset" stride=ratio
        hr_norm = hr_norm[:, :, 1:ratio:end]
    end
    hr_dataset = NormalizedDataset(hr_norm, base.stats)
    return base, hr_dataset
end

function load_model()
    contents = BSON.load(MODEL_PATH)
    model = contents[:model]
    cfg = get(contents, :cfg, nothing)
    sigma = haskey(contents, :trainer_cfg) ? contents[:trainer_cfg].sigma :
            (cfg === nothing ? 0.05 : 0.05)
    trainer_cfg = ScoreTrainerConfig(; sigma=sigma)
    return model, trainer_cfg
end

function score_from_model_local(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    return inv_sigma .* preds
end

function lagged_correlation(X::AbstractMatrix, m::Integer; time_dim::Int=2)
    m >= 0 || throw(ArgumentError("lag m must be nonnegative"))
    if time_dim == 1
        T, D = size(X)
        Tm = T - m
        Tm > 0 || throw(ArgumentError("lag m=$(m) too large for T=$(T)"))
        A = @view X[1:Tm, :]
        B = @view X[1+m:T, :]
        return (transpose(A) * B) / Tm
    elseif time_dim == 2
        D, T = size(X)
        Tm = T - m
        Tm > 0 || throw(ArgumentError("lag m=$(m) too large for T=$(T)"))
        A = @view X[:, 1:Tm]
        B = @view X[:, 1+m:T]
        return (A * transpose(B)) / Tm
    else
        throw(ArgumentError("time_dim must be 1 (rows are time) or 2 (columns are time)"))
    end
end

"""
    real_schur_logm(A)

Robust matrix logarithm that handles matrices with negative eigenvalues.
First performs eigendecomposition, shifts any eigenvalues with negative real parts
to small positive values (epsilon), then reassembles the matrix before computing log.
This avoids complex logarithm branches.
"""
function real_schur_logm(A::AbstractMatrix{<:Real}; epsilon::Float64=1e-10)
    size(A,1) == size(A,2) || throw(ArgumentError("A must be square"))
    
    # Eigendecomposition
    evals, evecs = eigen(A)
    
    # Check if any eigenvalues have negative real parts
    has_negative = any(x -> real(x) <= 0, evals)
    
    if has_negative
        # Shift negative/zero eigenvalues to small positive value
        evals_shifted = map(evals) do λ
            if real(λ) <= epsilon
                # If eigenvalue is negative or too close to zero, set to epsilon
                # Preserve imaginary part if it exists
                complex(epsilon, imag(λ))
            else
                λ
            end
        end
        
        # Reassemble matrix with shifted eigenvalues
        A_shifted = evecs * Diagonal(evals_shifted) * inv(evecs)
        
        # Take real part (imaginary should be negligible for real input)
        A_shifted = real.(A_shifted)
        
        # Now compute log of the shifted matrix
        L = log(A_shifted)
    else
        # No negative eigenvalues, compute log directly
        L = log(A)
    end
    
    # Ensure result is real
    if eltype(L) <: Complex
        max_im = maximum(abs, imag.(L))
        if max_im > 1e-7
            @warn "Matrix logarithm still has significant imaginary part after shifting" max_im
        end
        return real.(L)
    else
        return Matrix(L)
    end
end

"""
    fit_generator(Cs; weights=:exp)

Given a vector of correlation matrices Cs where Cs[1] = C(0) and Cs[m+1] = C(m),
estimate Q such that C(m) ≈ C(0) * exp(Q*m).

Uses multi-lag logarithmic aggregation:
    Q_m = (1/m) * log( C(0)^{-1} * C(m) )
and returns weighted average over m=1..M.

weights can be:
  :uniform  -> w_m = 1
  :exp      -> w_m ∝ exp(-m / (M/2)) emphasize smaller lags
  :mid      -> w_m ∝ exp(-((m - M/2)^2)/(2*(M/4)^2)) emphasize mid-range
"""
function fit_generator(Cs::Vector{<:AbstractMatrix}; weights::Symbol=:exp, M::Int=0)
    L = length(Cs) - 1
    if M == 0
        M = L
    end
    M >= 0 || throw(ArgumentError("Need at least one positive lag"))
    C0 = Cs[1]
    D = size(C0,1)

    # Use pseudo-inverse for better numerical stability
    invC0 = pinv(C0)

    # Pre-allocate thread-local storage for LS aggregation:
    # numerator = Σ w_m m log(G_m), denominator = Σ w_m m^2
    nthreads = Threads.nthreads()
    N_locals = [zeros(Float64, D, D) for _ in 1:nthreads]
    d_locals = zeros(Float64, nthreads)

    Threads.@threads for m in 1:M
        tid = Threads.threadid()
        Cm = Cs[m+1]
        Gm = invC0 * Cm

        # Skip ill-conditioned lags
        condG = cond(Gm)
        if !isfinite(condG) || condG > 1e8
            continue
        end

        # Principal matrix log of G_m
        logGm = real_schur_logm(Gm)

        # Weighting scheme
        w = if weights == :uniform
            1.0
        elseif weights == :exp
            exp(-m / (M/2))
        elseif weights == :mid
            σ = M/4
            exp(-((m - M/2)^2) / (2σ^2))
        else
            throw(ArgumentError("Unknown weights symbol $weights"))
        end

        N_locals[tid] .+= (w * m) .* logGm
        d_locals[tid] += w * (m^2)
    end

    # Reduce thread-local results
    N = sum(N_locals)
    d = sum(d_locals)
    d == 0 && error("No valid lags contributed to Q estimate")
    Q = N / d

    # Diagnostics: reconstruction error per lag (parallelized)
    errors = Vector{Float64}(undef, L + 1)
    Threads.@threads for m in 0:L
        pred = C0 * exp(Q * m)
        errors[m + 1] = norm(Cs[m+1] - pred) / max(norm(Cs[m+1]), eps())
    end

    evals = eigvals(Q)
    return Q, errors, evals
end

##

model, trainer_cfg = load_model()
base_dataset, hr_dataset = load_datasets()
mm_dataset = hr_dataset
obs = mm_dataset.data[:,1,:]

std(obs, dims=2)

# 1) Determine the number of timesteps corresponding to the decorrelation length τ
# obs is assumed (D, T) with time along columns
D, T = size(obs)
max_lag = max(1000, 0)
acf = ScoreUNet1D.average_mode_acf(obs, max_lag)
idx = findfirst(x -> x <= 0.1, acf)
tau_idx = idx === nothing ? length(acf) : idx
# Number of steps is tau_idx - 1 because acf[1] corresponds to lag 0
tau_steps = max(tau_idx - 1, 0)
@info "Estimated decorrelation length (steps)" tau_steps tau_idx

Plots.plot(acf, xlabel="Lag", ylabel="ACF", title="Average Mode Autocorrelation Function")
Plots.hline!([0.1], linestyle=:dash, color=:red, label="0.1 Threshold")

##
# 2) Parallel computation of C(m) for m in 0:tau_steps
Cs = Vector{Matrix{Float64}}(undef, tau_steps + 1)
Threads.@threads for m in 0:tau_steps
    Cs[m + 1] = lagged_correlation(obs, m)  # time_dim=2 by default for (D,T)
end

# Cs now contains [C(0), C(1), ..., C(tau_steps)]
@info "Computed correlation matrices up to decorrelation length" count=length(Cs)

Plots.heatmap(Cs[10], color=:viridis)

##
# 3) Fit generator Q from Cs
@info "Fitting generator Q from correlation matrices" tau_steps
Q, rel_errors, evalsQ = fit_generator(Cs; weights=:uniform, M=2)
@info "Fit complete" mean_rel_error=mean(rel_errors[2:end]) max_rel_error=maximum(rel_errors[2:end]) evals_sample=evalsQ[1:min(end,4)]

Plots.plot(rel_errors, xlabel="m", ylabel="Rel Frobenius Error", title="Correlation Fit Error")


##


function estimate_V_from_scores(model,
                                dataset::NormalizedDataset,
                                trainer_cfg::ScoreTrainerConfig;
                                stride::Int=MM_STRIDE,
                                max_samples::Int=0,
                                batch_size::Int=256)
    data = dataset.data                 # (L, C, T)
    L, C, T = size(data)
    T > 0 || error("Dataset has no time samples")

    stride = max(stride, 1)
    idxs = collect(1:stride:T)
    isempty(idxs) && error("No valid time indices with stride=$(stride)")

    if max_samples > 0
        idxs = idxs[1:min(max_samples, length(idxs))]
    end
    batch_size = max(1, min(batch_size, length(idxs)))

    D = L * C
    nthreads = Threads.nthreads()
    Vs = [zeros(Float64, D, D) for _ in 1:nthreads]
    processed_loc = zeros(Int, nthreads)
    tmp_batches = [Array{Float32}(undef, L, C, batch_size) for _ in 1:nthreads]

    chunk_groups = collect(partition(idxs, batch_size))
    Threads.@threads for chunk in chunk_groups
        tid = Threads.threadid()
        tmp_batch = tmp_batches[tid]
        b = length(chunk)
        x_mat = Array{Float64}(undef, D, b)

        @inbounds for (j, t) in enumerate(chunk)
            view_t = view(data, :, :, t)
            tmp_batch[:, :, j] .= view_t
            x_mat[:, j] = reshape(view_t, D)
        end

        # s(x): same layout as x
        scores = score_from_model_local(model, view(tmp_batch, :, :, 1:b), trainer_cfg.sigma)
        s_mat = Float64.(reshape(scores, D, b))

        Vs[tid] .+= s_mat * transpose(x_mat)  # E[s(x) xᵀ]
        processed_loc[tid] += b
    end

    total_processed = sum(processed_loc)
    total_processed > 0 || error("No samples processed when estimating V")

    V = sum(Vs)
    V ./= total_processed
    return V
end


V = estimate_V_from_scores(model, mm_dataset, trainer_cfg;
                           max_samples=0, batch_size=256)

eigvals(V)

Plots.heatmap(V, color=:viridis)                           
##
Phi = Cs[1] * Q * pinv(V)
Phi .+= 1e-3 * Matrix(I, size(Phi, 1), size(Phi, 1))
Phi_S = 0.5 * (Phi + Phi')
Sigma = Matrix(cholesky(Phi_S, check=true).L)


display(eigvals(Phi_S))
plt1 = Plots.plot((eigvals(Phi_S)))
plt2 = Plots.heatmap(Phi, color=:viridis)
plt3 = Plots.heatmap(Sigma, color=:viridis)
Plots.plot(plt1, plt2, plt3, layout=(3,1))


##
λ = 1.0
Phi = λ * Matrix(I, size(Phi, 1), size(Phi, 1))
Sigma = √λ * Matrix(I, size(Sigma, 1), size(Sigma, 1))

Phi = λ * copy(Phi_corrected)
Sigma = √λ * copy(Sigma_corrected)


fig = Figure(size=(900, 400))

# Phi panel
phi_range = maximum(abs, Phi)
phi_range = max(phi_range, eps(Float64))
ax_phi = Axis(fig[1, 1]; title="Phi (Drift Matrix)")
hm_phi = CairoMakie.heatmap!(ax_phi, Phi; colormap=:viridis, colorrange=(-phi_range, phi_range))
ax_phi.xticksvisible = false
ax_phi.yticksvisible = false
Colorbar(fig[1, 2], hm_phi; width=20)

# Sigma panel
sigma_range = maximum(abs, Sigma)
sigma_range = max(sigma_range, eps(Float64))
ax_sigma = Axis(fig[1, 3]; title="Sigma (Diffusion Matrix)")
hm_sigma = CairoMakie.heatmap!(ax_sigma, Sigma; colormap=:viridis)
ax_sigma.xticksvisible = false
ax_sigma.yticksvisible = false
Colorbar(fig[1, 4], hm_sigma; width=20)

save_path = joinpath(PROJECT_ROOT, "runs", "phi_sigma_matrices.png")
mkpath(dirname(save_path))
save(save_path, fig)
@info "Saved Phi and Sigma matrices figure" path=save_path
display(fig)

##

"""
    build_grouped_diagonal(counts, rates, D)

Construct a diagonal matrix of size D×D where the first `counts[1]` diagonal elements
are set to `rates[1]`, the next `counts[2]` elements to `rates[2]`, and so on.

# Arguments
- `counts::AbstractVector`: Vector of group sizes [c_1, c_2, ...]
- `rates::AbstractVector`: Vector of decay rates [r_1, r_2, ...]
- `D::Int`: Total dimension of the output matrix

# Returns
- `Matrix{Float64}`: D×D diagonal matrix with grouped rates
"""
function build_grouped_diagonal(counts::AbstractVector, rates::AbstractVector, D::Int)
    length(counts) == length(rates) || throw(ArgumentError("counts and rates must have the same length"))
    sum(counts) == D || throw(ArgumentError("sum of counts must equal D"))
    all(c -> c > 0, counts) || throw(ArgumentError("all counts must be positive"))
    
    diag_vals = Vector{Float64}(undef, D)
    idx = 1
    for (count, rate) in zip(counts, rates)
        for _ in 1:count
            diag_vals[idx] = rate
            idx += 1
        end
    end
    
    return Diagonal(diag_vals)
end

counts = [16, 16]
rates = [1.0, 0.01]
D = 32
Phi_L = build_grouped_diagonal(counts, rates, D)
Phi = Matrix(Phi_L) #* Cs[1] .+ 0.0882 * Matrix(I, size(Phi_L, 1), size(Phi_L, 1))
(eigvals(Phi+Phi'))
Sigma = Matrix(cholesky(0.5 * (Phi + Phi'), check=true).L)

##
# Phi = λ * copy(Phi_local)
# Sigma = √λ * copy(Sigma_local)

Phi = - Cs[1] * Q
Phi .+= 1e-3 * Matrix(I, size(Phi, 1), size(Phi, 1))
# Phi .*= 2.0
Phi_S = 0.5 * (Phi + Phi')
Sigma = Matrix(cholesky(Phi_S, check=true).L)


display(eigvals(Phi_S))
plt1 = Plots.plot((eigvals(Phi_S)))
plt2 = Plots.heatmap(Phi, color=:viridis)
plt3 = Plots.heatmap(Sigma, color=:viridis)
Plots.plot(plt1, plt2, plt3, layout=(3,1))

##
LANGEVIN_DT = 0.01
LANGEVIN_STEPS = 500_000
LANGEVIN_BURN_IN = 5_000
LANGEVIN_RESOLUTION = 100
LANGEVIN_ENSEMBLES = 1 #max(nthreads(), 1)


struct DriftWorkspace
    tensor::Array{Float32,3}
    flat::Vector{Float32}
    score::Vector{Float64}
end

function create_fast_drift(model, trainer_cfg::ScoreTrainerConfig,
                           Phi::Matrix{Float64}, L::Int, C::Int)
    D = L * C
    workspaces = [DriftWorkspace(Array{Float32}(undef, L, C, 1),
                                 Vector{Float32}(undef, D),
                                 Vector{Float64}(undef, D)) for _ in 1:nthreads()]
    function drift!(du, u, params, t)
        ws = workspaces[threadid()]
        @inbounds @simd for i in 1:D
            ws.flat[i] = Float32(u[i])
        end
        ws.tensor[:, :, 1] .= reshape(ws.flat, L, C)
        scores = score_from_model_local(model, ws.tensor, trainer_cfg.sigma)
        score_vec = reshape(scores, D)
        @inbounds @simd for i in 1:D
            ws.score[i] = Float64(score_vec[i])
        end
        mul!(du, Phi, ws.score)
        return nothing
    end
    function drift!(du, u, t)
        drift!(du, u, nothing, t)
    end
    return drift!
end

function integrate_langevin_fast(model, trainer_cfg::ScoreTrainerConfig,
                                 Phi::Matrix{Float64}, Sigma::Matrix{Float64},
                                 dataset::NormalizedDataset;
                                 resolution::Int=LANGEVIN_RESOLUTION)
    L = sample_length(dataset)
    C = num_channels(dataset)
    D = L * C
    drift! = create_fast_drift(model, trainer_cfg, Phi, L, C)
    diffusion = sqrt(2.0) .* Matrix(Sigma)
    idx = rand(1:size(dataset.data, 3))
    sample = view(dataset.data, :, :, idx)
    u0 = Float64.(reshape(sample, D))
    raw = FastSDE.evolve_ens(u0, LANGEVIN_DT, LANGEVIN_STEPS, drift!, diffusion;
                             resolution=resolution,
                             n_ens=LANGEVIN_ENSEMBLES,
                             timestepper=:euler,
                             seed=0x5eed42,
                             manage_blas_threads=true,
                             sigma_inplace=false,
                             batched_drift=false)
    nsaves = size(raw, 2)
    burn_cols = min(nsaves, fld(LANGEVIN_BURN_IN, resolution) + 1)
    keep_cols = nsaves - burn_cols
    keep_cols > 0 || error("Burn-in removes all Langevin samples. Increase steps or reduce burn-in.")
    kept = raw[:, burn_cols+1:end, :]
    reshaped = reshape(kept, D, keep_cols * size(raw, 3))
    tensor = reshape(Float32.(reshaped), L, C, :)
    return tensor
end

function ensure_bounds(bounds::Tuple{Float64,Float64})
    a, b = bounds
    if !isfinite(a) || !isfinite(b)
        return (-1.0, 1.0)
    elseif a == b
        δ = max(abs(a), 1.0) * 1e-3
        return (a - δ, b + δ)
    elseif a > b
        return (b, a)
    else
        return (a, b)
    end
end

function pair_ranges(tensor::Array{Float32,3}, j::Int)
    L, C, B = size(tensor)
    L > j || error("Lag j=$j exceeds spatial length L=$L")
    xmin, xmax = Inf, -Inf
    ymin, ymax = Inf, -Inf
    @inbounds for b in 1:B, c in 1:C, i in 1:(L - j)
        x = tensor[i, c, b]
        y = tensor[i + j, c, b]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    end
    return (Float64(xmin), Float64(xmax)), (Float64(ymin), Float64(ymax))
end

function pair_samples_tensor(tensor::Array{Float32,3}, j::Int)
    L, C, B = size(tensor)
    L > j || error("Lag j=$j exceeds spatial length L=$L")
    n = (L - j) * C * B
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    idx = 1
    @inbounds for b in 1:B, c in 1:C, i in 1:(L - j)
        xs[idx] = Float64(tensor[i, c, b])
        ys[idx] = Float64(tensor[i + j, c, b])
        idx += 1
    end
    return xs, ys
end

function kde_heatmap_tensor(tensor::Array{Float32,3}, j::Int,
                            bounds::Tuple{Float64,Float64}, npoints::Int)
    xs, ys = pair_samples_tensor(tensor, j)
    xgrid = range(bounds[1], bounds[2]; length=npoints)
    ygrid = range(bounds[1], bounds[2]; length=npoints)
    kd = kde((xs, ys), (xgrid, ygrid))
    return kd.x, kd.y, kd.density
end

function compute_heatmap_specs(sim_tensor::Array{Float32,3},
                               obs_tensor::Array{Float32,3},
                               offsets::Tuple{Vararg{Int}};
                               nbins::Int=64,
                               bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    specs = NamedTuple[]
    for j in offsets
        local_bounds = bounds
        if local_bounds === nothing
            sim_x, sim_y = pair_ranges(sim_tensor, j)
            obs_x, obs_y = pair_ranges(obs_tensor, j)
            low = min(sim_x[1], obs_x[1])
            high = max(sim_x[2], obs_x[2])
            local_bounds = ensure_bounds((low, high))
        end
        sx, sy, sim_counts = kde_heatmap_tensor(sim_tensor, j, local_bounds, nbins)
        _, _, obs_counts = kde_heatmap_tensor(obs_tensor, j, local_bounds, nbins)
        push!(specs, (
            j = j,
            simulated = (sx, sy, sim_counts),
            observed = (sx, sy, obs_counts),
        ))
    end
    return specs
end

# Euler-Maruyama integration via FastSDE (multi-threaded)
@info "Starting FastSDE Langevin integration" dt=LANGEVIN_DT steps=LANGEVIN_STEPS burn_in=LANGEVIN_BURN_IN ensembles=LANGEVIN_ENSEMBLES
trajectory = integrate_langevin_fast(model, trainer_cfg, Phi, Sigma, mm_dataset)
@info "Langevin integration complete" trajectory_size=size(trajectory)

# Two-panel figure: top = trajectory (first variable), bottom = observed vs simulated ACFs
traj_series = vec(view(trajectory, 1, 1, :))
n_steps = length(traj_series)
obs_series = vec(mm_dataset.data[1, 1, 1:min(n_steps, size(mm_dataset.data, 3))])

mean(obs_series)

traj_plot = Plots.plot(obs_series;
    xlabel="Step",
    ylabel="x[1]",
    title="Time Series Comparison (Variable 1)",
    label="Observed",
    color=:navy,
    linewidth=2)
Plots.plot!(traj_plot, traj_series; label="Simulated", color=:firebrick, linewidth=2, linestyle=:dash)

# Compute ACF for simulated trajectory (matching lag length of observed ACF)
sim_slice = trajectory[:, 1, :]  # (L, T_sim)
max_lag_sim = min(length(acf) - 1, size(sim_slice, 2) - 1)
acf_sim = ScoreUNet1D.average_mode_acf(sim_slice, max_lag_sim)

acf_plot = Plots.plot(acf[1:max_lag_sim+1];
    xlabel="Lag",
    ylabel="ACF",
    title="ACF Comparison (Observed vs Simulated)",
    label="Observed",
    color=:navy,
    linewidth=2)
Plots.plot!(acf_plot, acf_sim; label="Simulated", color=:firebrick, linewidth=2, linestyle=:dash)

two_panel_fig = Plots.plot(traj_plot, acf_plot; layout=(2,1), size=(800,600))
display(two_panel_fig)

##
# Compute PDFs for each variable using KernelDensity
n_vars = size(trajectory, 1)
n_samples_langevin = size(trajectory, 3)
n_samples_data = size(mm_dataset.data, 3)

# Storage for PDFs and ACFs
langevin_pdfs = []
data_pdfs = []
langevin_acfs = []
data_acfs = []

# Common grid for PDF evaluation
grid_points = range(-5.0, 5.0, length=256)

@info "Computing PDFs and ACFs for each variable" n_vars=n_vars

# Helper function to compute ACF for a single time series
function compute_single_acf(series::AbstractVector, max_lag::Int)
    T = length(series)
    max_lag = min(max_lag, T - 1)
    acf = zeros(Float64, max_lag + 1)
    μ = mean(series)
    centered = series .- μ
    variance = sum(abs2, centered) / T
    if variance <= eps(Float64)
        return acf
    end
    for lag in 0:max_lag
        total = T - lag
        if total > 0
            acf[lag + 1] = dot(view(centered, 1:total), 
                              view(centered, 1 + lag:lag + total)) / (total * variance)
        end
    end
    return acf
end

for i in 1:n_vars
    # Extract time series for variable i
    langevin_series = vec(view(trajectory, i, 1, :))
    data_series = vec(mm_dataset.data[i, 1, :])
    
    # Compute PDFs using KernelDensity
    kde_langevin = kde(langevin_series)
    kde_data = kde(data_series)
    
    pdf_langevin = pdf(kde_langevin, grid_points)
    pdf_data = pdf(kde_data, grid_points)
    
    push!(langevin_pdfs, pdf_langevin)
    push!(data_pdfs, pdf_data)
    
    # Compute ACFs using local function
    max_lag = min(500, length(langevin_series) - 1, length(data_series) - 1)
    acf_langevin = compute_single_acf(langevin_series, max_lag)
    acf_data = compute_single_acf(data_series, max_lag)
    
    push!(langevin_acfs, acf_langevin)
    push!(data_acfs, acf_data)
end

# Average PDFs and ACFs across all variables
avg_pdf_langevin = mean(hcat(langevin_pdfs...), dims=2) |> vec
avg_pdf_data = mean(hcat(data_pdfs...), dims=2) |> vec

# Find minimum length for ACFs and truncate
min_acf_length = minimum(length.(vcat(langevin_acfs, data_acfs)))
langevin_acfs_trunc = [acf[1:min_acf_length] for acf in langevin_acfs]
data_acfs_trunc = [acf[1:min_acf_length] for acf in data_acfs]

avg_acf_langevin = mean(hcat(langevin_acfs_trunc...), dims=2) |> vec
avg_acf_data = mean(hcat(data_acfs_trunc...), dims=2) |> vec

# Create time vector for ACF
acf_time = collect(0:min_acf_length-1) .* DATA_DT

function fit_double_exponential(acf_time::AbstractVector, acf_vals::AbstractVector)
    t = Float64.(acf_time)
    y = Float64.(acf_vals)
    max_t = max(maximum(t), eps())
    init_c1 = first(y)
    init_c3 = first(y) / 4
    init_c2 = max(1 / max_t, 1e-3)
    init_c4 = 5 * init_c2
    function objective(p)
        c1, log_c2, c3, log_c4 = p
        c2 = exp(log_c2)
        c4 = exp(log_c4)
        preds = @. c1 * exp(-c2 * t) + c3 * exp(-c4 * t)
        return sum((preds .- y).^2)
    end
    init = [init_c1, log(init_c2), init_c3, log(init_c4)]
    result = optimize(objective, init; autodiff=:forward)
    p_hat = Optim.minimizer(result)
    c1, log_c2, c3, log_c4 = p_hat
    c2 = exp(log_c2)
    c4 = exp(log_c4)
    fit_vals = @. c1 * exp(-c2 * t) + c3 * exp(-c4 * t)
    return fit_vals, (c1=c1, c2=c2, c3=c3, c4=c4, loss=Optim.minimum(result))
end

# Prepare tensors for the bivariate heatmaps
obs_tensor = Float32.(mm_dataset.data)
sim_tensor = Float32.(trajectory)
min_val = min(Float64(minimum(obs_tensor)), Float64(minimum(sim_tensor)))
max_val = max(Float64(maximum(obs_tensor)), Float64(maximum(sim_tensor)))
value_bounds = ensure_bounds((min_val, max_val))
available_offsets = filter(j -> j < n_vars, BIVARIATE_OFFSETS)
offsets = Tuple(available_offsets)
specs = isempty(offsets) ? NamedTuple[] :
        compute_heatmap_specs(sim_tensor, obs_tensor, offsets;
                              bounds=value_bounds,
                              nbins=HEATMAP_BINS)
fig_height = 360 + 320 * length(specs)

# Plot comparison figure
fig = Figure(size=(1400, fig_height))

# Left panel: PDF comparison
ax_pdf = Axis(fig[1, 1]; xlabel="Value", ylabel="PDF",
              title="Averaged Univariate PDFs")
lines!(ax_pdf, grid_points, avg_pdf_data; color=:navy, linewidth=3, label="Data")
lines!(ax_pdf, grid_points, avg_pdf_langevin; color=:firebrick, linewidth=3, label="Langevin")
CairoMakie.xlims!(ax_pdf, value_bounds...)
axislegend(ax_pdf, position=:rt)

# Right panel: ACF comparison
ax_acf = Axis(fig[1, 2]; xlabel="Time Lag", ylabel="Average ACF",
              title="Langevin vs Data ACF")
lines!(ax_acf, acf_time, avg_acf_data; color=:navy, linewidth=3, label="Data")
lines!(ax_acf, acf_time, avg_acf_langevin; color=:firebrick, linewidth=3, label="Langevin")
axislegend(ax_acf, position=:rt)

for (row, spec) in enumerate(specs)
    lx = Axis(fig[row + 1, 1];
              xlabel="x[i]",
              ylabel="x[i+$(spec.j)]",
              title=@sprintf("Langevin: j = %d", spec.j))
    rx = Axis(fig[row + 1, 2];
              xlabel="x[i]",
              ylabel="x[i+$(spec.j)]",
              title=@sprintf("Data: j = %d", spec.j))
    sx, sy, sim_counts = spec.simulated
    _, _, obs_counts = spec.observed
    CairoMakie.heatmap!(lx, sx, sy, sim_counts; colormap=:plasma)
    CairoMakie.heatmap!(rx, sx, sy, obs_counts; colormap=:plasma)
    CairoMakie.xlims!(lx, value_bounds...)
    CairoMakie.ylims!(lx, value_bounds...)
    CairoMakie.xlims!(rx, value_bounds...)
    CairoMakie.ylims!(rx, value_bounds...)
end

save_path = joinpath(PROJECT_ROOT, "scripts", "pdf_acf_bivariate_comparison.png")
save(save_path, fig; px_per_unit=1)
@info "Saved PDF, ACF and bivariate comparison figure" path=save_path

display(fig)

##

Plots.plot(acf_time, avg_acf_data, xlabel="Lag", ylabel="Average ACF", label="Data", linewidth=2)
fit_curve, fit_params = fit_double_exponential(acf_time, avg_acf_data)
@info "Double-exponential ACF fit" c1=fit_params.c1 c2=fit_params.c2 c3=fit_params.c3 c4=fit_params.c4 loss=fit_params.loss
Plots.plot!(acf_time, fit_curve;
            label="Fit: c1*exp(-c2*tau)+c3*exp(-c4*tau)",
            linewidth=2,
            linestyle=:dash,
            color=:darkgreen)
