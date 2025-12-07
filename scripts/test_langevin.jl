#!/usr/bin/env julia

# Lightweight Langevin validation script.
# - Reads KS data (new_ks.hdf5)
# - Loads Φ, Σ, α, and model.bson from a chosen models/<folder> directory
# - Rescales Φ, Σ by α
# - Runs Langevin on GPU
# - Builds a publication-style figure (heatmaps, PDFs, ACF, bivariate KDEs)
# Every block is top-level so you can execute line by line in the REPL.

# nohup julia --project=. scripts/test_langevin.jl > test_langevin.log 2>&1 &

using CairoMakie
using LinearAlgebra
using Random
using Statistics
using KernelDensity
using HDF5
using BSON
using Flux
using ScoreUNet1D
import ScoreUNet1D.PhiSigmaEstimator: average_component_acf_data
import ScoreUNet1D.EnsembleIntegrator: CPU_STATE_CACHE, GPU_STATE_CACHE

CairoMakie.activate!()

###############################################################################
# Paths & user input
###############################################################################
const PROJECT_ROOT = dirname(@__DIR__)
const DATA_PATH = joinpath(PROJECT_ROOT, "data", "new_ks.hdf5")
const DATASET_KEY = "timeseries"

# Name of the folder inside data/models containing model.bson and phi_sigma.hdf5
const MODEL_FOLDER = "model2"  # change as needed
const MODEL_DIR = joinpath(PROJECT_ROOT, "data", "models", MODEL_FOLDER)
const MODEL_PATH = joinpath(MODEL_DIR, "model.bson")
const PHI_SIGMA_PATH = joinpath(MODEL_DIR, "phi_sigma.hdf5")
const FIG_PATH = joinpath(@__DIR__, "test_langevin_fig.png")

###############################################################################
# Parameters
###############################################################################
const RNG_SEED = 2025
const HEATMAP_SNAPSHOTS = 1_000
const PDF_NBINS = 256
const PDF_STRIDE = 25
const BIV_OFFSETS = (1, 2, 3)
const BIV_NPOINTS = 160
const BIV_STRIDE = 10
const ACF_MAX_LAG = 100   # in data time units

# Langevin schedule (GPU)
const LANGEVIN_DT = 1.0e-3
const LANGEVIN_RESOLUTION = 1000              # snapshots every 1.0 units (dt*resolution)
const LANGEVIN_EFFECTIVE_DT = LANGEVIN_DT * LANGEVIN_RESOLUTION
const LANGEVIN_N_STEPS = 200_000
const LANGEVIN_BURN_IN = 100_000
const LANGEVIN_N_ENSEMBLES = 1024

###############################################################################
# Helper utilities
###############################################################################
function quantile_bounds(tensors...; stride::Int=20, probs=(0.001, 0.999))
    rng = MersenneTwister(RNG_SEED + 7)
    buf = Float64[]
    for tensor in tensors
        flat = vec(tensor)
        step = max(1, stride)
        if length(flat) <= 2_000_000
            append!(buf, flat[1:step:end])
        else
            ns = min(length(flat) ÷ step, 2_000_000)
            idxs = rand(rng, 1:length(flat), ns)
            append!(buf, flat[idxs])
        end
    end
    q = quantile(buf, [probs[1], probs[2]])
    low, high = Float64(q[1]), Float64(q[2])
    low == high && ((low -= 1e-3); (high += 1e-3))
    return (low, high)
end

function averaged_univariate_pdf(tensor::AbstractArray{<:Real,3};
                                 nbins::Int,
                                 stride::Int,
                                 bounds::Tuple{Float64,Float64})
    slice = @view tensor[:, :, 1:stride:end]
    grid = range(bounds[1], bounds[2]; length=nbins)
    kd = kde(vec(slice), grid)
    return kd.x, kd.density
end

function pair_kde(tensor::AbstractArray{<:Real,3},
                  offset::Int;
                  bounds::Tuple{Float64,Float64},
                  npoints::Int,
                  stride::Int)
    L, C, B = size(tensor)
    @assert offset < L "Offset $offset exceeds length $L"
    n = (L - offset) * C * cld(B, stride)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    idx = 1
    @inbounds for b in 1:stride:B, c in 1:C, i in 1:(L - offset)
        xs[idx] = tensor[i, c, b]
        ys[idx] = tensor[i + offset, c, b]
        idx += 1
    end
    xs = xs[1:idx-1]; ys = ys[1:idx-1]
    grid = range(bounds[1], bounds[2]; length=npoints)
    kd = kde((xs, ys), (grid, grid))
    return kd.x, kd.y, kd.density
end

function smooth_vec(v; w::Int=3)
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w); hi = min(n, i + w)
        out[i] = mean(@view v[lo:hi])
    end
    return out
end

###############################################################################
# Load model, Φ, Σ, α
###############################################################################
@info "Loading Φ, Σ, α" PHI_SIGMA_PATH
h5open(PHI_SIGMA_PATH, "r") do h5
    global alpha = read(h5, "Alpha")
    global Phi = read(h5, "Phi")
    global Sigma = read(h5, "Sigma")
end
alpha = 1.0
Phi_scaled = alpha .* Phi
Sigma_scaled = sqrt(alpha) .* Sigma

@info "Loading model" MODEL_PATH
model_artifacts = BSON.load(MODEL_PATH)
model = model_artifacts[:model]
sigma_model = haskey(model_artifacts, :trainer_cfg) ? Float32(model_artifacts[:trainer_cfg].sigma) : 0.1f0
Flux.testmode!(model)

###############################################################################
# Load dataset
###############################################################################
@info "Loading dataset" DATA_PATH
dataset = ScoreUNet1D.load_hdf5_dataset(DATA_PATH;
                                        dataset_key=DATASET_KEY,
                                        samples_orientation=:columns)
data_clean = dataset.data
data_stats = dataset.stats
L, C, B = size(data_clean)
D = L * C

###############################################################################
# Langevin integration on GPU
###############################################################################
rng = MersenneTwister(RNG_SEED)
score_wrapper = ScoreUNet1D.ScoreWrapper(model, sigma_model, L, C, D)
integrator = ScoreUNet1D.build_snapshot_integrator(score_wrapper; device="gpu")
empty!(CPU_STATE_CACHE); empty!(GPU_STATE_CACHE)

# Initial conditions from data
x0 = Matrix{Float32}(undef, D, LANGEVIN_N_ENSEMBLES)
@inbounds for i in 1:LANGEVIN_N_ENSEMBLES
    idx = rand(rng, 1:B)
    x0[:, i] = reshape(@view(data_clean[:, :, idx]), D)
end

Identity = Matrix{Float32}(I, D, D)

@assert isapprox(LANGEVIN_EFFECTIVE_DT, 1.0; atol=1e-8) "dt*resolution must be 1.0"
traj_state = integrator(x0, Float32.(Phi_scaled), Float32.(Sigma_scaled);
                        dt=LANGEVIN_DT,
                        n_steps=LANGEVIN_N_STEPS,
                        burn_in=LANGEVIN_BURN_IN,
                        resolution=LANGEVIN_RESOLUTION,
                        boundary=nothing,
                        progress=false,
                        progress_desc="Langevin GPU")
traj = Array(traj_state)  # (D, snapshots, ensembles)
T_snap = size(traj, 2)
traj_tensor = reshape(traj, L, C, T_snap, :)
langevin_tensor = reshape(traj_tensor, L, C, :)

###############################################################################
# Data products
###############################################################################
n_hm = min(HEATMAP_SNAPSHOTS, size(traj_tensor, 3))
heatmap_langevin = Array(traj_tensor[:, 1, 1:n_hm, 1])
heatmap_data = Array(@view data_clean[:, 1, 1:n_hm])

value_bounds = quantile_bounds(data_clean, langevin_tensor; stride=50)
pdf_x, pdf_langevin = averaged_univariate_pdf(langevin_tensor;
                                              nbins=PDF_NBINS,
                                              stride=PDF_STRIDE,
                                              bounds=value_bounds)
_, pdf_data = averaged_univariate_pdf(data_clean;
                                      nbins=PDF_NBINS,
                                      stride=PDF_STRIDE,
                                      bounds=value_bounds)

# ACFs:
# - Langevin: compute per-(mode, ensemble) ACFs of length ACF_MAX_LAG, then average
# - Data: average over modes and time (unchanged)
lags = collect(0:ACF_MAX_LAG)

function mean_acf_langevin(traj::Array{Float32,3}, L::Int, C::Int, n_ens::Int, max_lag::Int)
    # traj is (D, T_snap, n_ens)
    D = L * C
    acf_accum = zeros(Float64, max_lag + 1)
    count = 0
    for e in 1:n_ens
        slice = traj[:, :, e]
        acf_e = average_component_acf_data(slice, max_lag)
        acf_accum .+= acf_e
        count += 1
    end
    return acf_accum ./ max(count, 1)
end

acf_langevin = mean_acf_langevin(traj, L, C, LANGEVIN_N_ENSEMBLES, ACF_MAX_LAG)
acf_data = average_component_acf_data(reshape(data_clean, D, :), ACF_MAX_LAG)

acf_langevin_sm = smooth_vec(acf_langevin; w=2); acf_langevin_sm[1] = acf_langevin[1]
acf_data_sm = smooth_vec(acf_data; w=2); acf_data_sm[1] = acf_data[1]

# Bivariate KDEs
bivariate_specs = Dict{Int,NamedTuple}()
for offset in BIV_OFFSETS
    xg, yg, dens_lang = pair_kde(langevin_tensor, offset;
                                 bounds=value_bounds,
                                 npoints=BIV_NPOINTS,
                                 stride=BIV_STRIDE)
    _, _, dens_data = pair_kde(data_clean, offset;
                               bounds=value_bounds,
                               npoints=BIV_NPOINTS,
                               stride=BIV_STRIDE)
    bivariate_specs[offset] = (; x = xg, y = yg,
                                langevin = dens_lang,
                                data = dens_data)
end

###############################################################################
# Figure (mirrors publication_figures.jl Figure 1)
###############################################################################
fig1 = Figure(size=(1500, 2100), font="TeX Gyre Heros")

ax_hm_lang = Axis(fig1[1, 1]; title="Langevin snapshots", xlabel="t (first $(HEATMAP_SNAPSHOTS))", ylabel="mode")
hm_lang = heatmap!(ax_hm_lang, 1:HEATMAP_SNAPSHOTS, 1:L, heatmap_langevin';
                   colormap=:batlow, colorrange=value_bounds)
ax_hm_data = Axis(fig1[1, 2]; title="Data (new_ks)", xlabel="t (first $(HEATMAP_SNAPSHOTS))", ylabel="mode")
hm_data = heatmap!(ax_hm_data, 1:HEATMAP_SNAPSHOTS, 1:L, heatmap_data';
                   colormap=:batlow, colorrange=value_bounds)
Colorbar(fig1[1, 3], hm_data; label="value", width=10)

ax_pdf = Axis(fig1[2, 1]; title="Averaged univariate PDF", xlabel="value", ylabel="density")
lines!(ax_pdf, pdf_x, pdf_data; color=:black, linewidth=3.0, label="data")
lines!(ax_pdf, pdf_x, pdf_langevin; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
ylims!(ax_pdf, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
xlims!(ax_pdf, minimum(pdf_x), maximum(pdf_x))
axislegend(ax_pdf, position=:rt, framevisible=false)

ax_acf = Axis(fig1[2, 2]; title="Average ACF", xlabel="lag (time units)", ylabel="ACF")
lines!(ax_acf, lags, acf_data_sm; color=:black, linewidth=3.0, label="data")
lines!(ax_acf, lags, acf_langevin_sm; color=:dodgerblue4, linewidth=3.0, linestyle=:dash, label="Langevin")
hlines!(ax_acf, [0.0]; color=:gray, linestyle=:dot, linewidth=1.5)
xlims!(ax_acf, 0, ACF_MAX_LAG)
ylims!(ax_acf, -0.1, 1.05)
axislegend(ax_acf, position=:rt, framevisible=false)

for (row, offset) in enumerate(BIV_OFFSETS)
    spec = bivariate_specs[offset]
    density_max = max(maximum(spec.langevin), maximum(spec.data))
    density_max = density_max == 0 ? 1e-9 : density_max
    ax_l = Axis(fig1[row + 2, 1];
                title = "Langevin, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]")
    hm_l = heatmap!(ax_l, spec.x, spec.y, spec.langevin;
                    colormap=:vik, colorrange=(0, density_max))

    ax_r = Axis(fig1[row + 2, 2];
                title = "Data, offset = $offset",
                xlabel = "x[i]", ylabel = "x[i+$offset]")
    hm_r = heatmap!(ax_r, spec.x, spec.y, spec.data;
                    colormap=:vik, colorrange=(0, density_max))
    Colorbar(fig1[row + 2, 3], hm_r; label="density", width=10)
end

save(FIG_PATH, fig1; px_per_unit=1)
@info "Figure saved" path=FIG_PATH

display(fig1)
