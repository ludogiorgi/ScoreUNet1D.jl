"""
    PlotRunner

Generic publication figure generation. Can be used for any dynamical system.
"""
module PlotRunner

using CairoMakie
using Flux
using HDF5
using KernelDensity
using LinearAlgebra
using Random
using Statistics
using TOML

using ..ScoreUNet1D: load_hdf5_dataset
using ..PhiSigmaEstimator: average_acf_ensemble, average_acf_3d, compute_V_data
using ..RunnerUtils: load_config, resolve_path, ensure_dir, symbol_from_string, load_model

export FigurePaths, plot_publication_figure, plot_comparison_figure

CairoMakie.activate!()

#─────────────────────────────────────────────────────────────────────────────
# Result Struct
#─────────────────────────────────────────────────────────────────────────────

"""
    FigurePaths

Paths to generated publication figures.
"""
struct FigurePaths
    main_figure::String
    grid_figure::Union{Nothing,String}
    horizontal_figure::Union{Nothing,String}
    matrices_figure::Union{Nothing,String}
end

#─────────────────────────────────────────────────────────────────────────────
# Helper Functions
#─────────────────────────────────────────────────────────────────────────────

function quantile_bounds(tensors::AbstractArray...; stride::Int=20, probs=(0.001, 0.999))
    buffer = Float64[]
    for tensor in tensors
        flat = vec(tensor)
        step = max(1, stride)
        append!(buffer, flat[1:step:end])
    end
    qvals = quantile(buffer, collect(probs))
    low, high = Float64(qvals[1]), Float64(qvals[2])
    low == high && ((low -= 1e-3); (high += 1e-3))
    return (low, high)
end

function averaged_univariate_pdf(tensor::AbstractArray{<:Real,3}; nbins::Int, stride::Int,
    bounds::Tuple{Float64,Float64})
    slice = @view tensor[:, :, 1:stride:end]
    grid = range(bounds[1], bounds[2]; length=nbins)
    kd = kde(vec(slice), grid)
    return kd.x, kd.density
end

function pair_kde(tensor::AbstractArray{<:Real,3}, offset::Int;
    bounds::Tuple{Float64,Float64}, npoints::Int, stride::Int)
    L, C, B = size(tensor)
    @assert offset < L "Offset $offset exceeds length $L"
    n = (L - offset) * C * cld(B, stride)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    idx = 1
    @inbounds for b in 1:stride:B, c in 1:C, i in 1:(L-offset)
        xs[idx] = tensor[i, c, b]
        ys[idx] = tensor[i+offset, c, b]
        idx += 1
    end
    xs = xs[1:idx-1]
    ys = ys[1:idx-1]
    grid = range(bounds[1], bounds[2]; length=npoints)
    kd = kde((xs, ys), (grid, grid))
    return kd.x, kd.y, kd.density
end

function smooth_vec(v; w::Int=2)
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w)
        hi = min(n, i + w)
        out[i] = mean(@view v[lo:hi])
    end
    return out
end

function load_trajectory_h5(path::AbstractString)
    h5open(path, "r") do h5
        traj = read(h5, "trajectory")
        Phi = haskey(h5, "Phi") ? read(h5, "Phi") : nothing
        Sigma = haskey(h5, "Sigma") ? read(h5, "Sigma") : nothing
        model_path = haskey(h5, "model_path") ? read(h5, "model_path") : nothing
        params = h5["langevin_params"]
        dt = read(params, "dt")
        res = read(params, "resolution")
        data_path = read(h5, "data_path")
        data_path_hr = haskey(h5, "data_path_hr") ? read(h5, "data_path_hr") : data_path
        dataset_key = read(h5, "dataset_key")
        dataset_orient = Symbol(read(h5, "dataset_orientation"))

        pdf_stride = haskey(params, "pdf_stride") ? Int(read(params, "pdf_stride")) : 25
        pdf_nbins = haskey(params, "pdf_nbins") ? Int(read(params, "pdf_nbins")) : 256
        acf_max_lag = haskey(params, "acf_max_lag") ? Int(read(params, "acf_max_lag")) : 100
        biv_offsets = haskey(params, "biv_offsets") ? Int.(read(params, "biv_offsets")) : [1, 2, 3]
        biv_npoints = haskey(params, "biv_npoints") ? Int(read(params, "biv_npoints")) : 160
        biv_stride = haskey(params, "biv_stride") ? Int(read(params, "biv_stride")) : 10
        heatmap_snapshots = haskey(params, "heatmap_snapshots") ? Int(read(params, "heatmap_snapshots")) : 1000

        return (
            trajectory=traj,
            Phi=Phi,
            Sigma=Sigma,
            model_path=model_path,
            dt=dt,
            resolution=res,
            dt_effective=dt * res,
            data_path=data_path,
            data_path_hr=data_path_hr,
            dataset_key=dataset_key,
            dataset_orientation=dataset_orient,
            pdf_stride=pdf_stride,
            pdf_nbins=pdf_nbins,
            acf_max_lag=acf_max_lag,
            biv_offsets=biv_offsets,
            biv_npoints=biv_npoints,
            biv_stride=biv_stride,
            heatmap_snapshots=heatmap_snapshots
        )
    end
end

#─────────────────────────────────────────────────────────────────────────────
# Main Functions
#─────────────────────────────────────────────────────────────────────────────

"""
    plot_publication_figure(config_path; project_root=nothing) -> FigurePaths

Generate publication-quality figures from Langevin trajectory files.

# Arguments
- `config_path`: Path to TOML configuration file
- `project_root=nothing`: Project root for resolving paths

# Config Sections
- `[paths]`: identity_h5, phi_sigma_h5, output_dir
- `[figure]`: font_family, title_size, colormap, etc.

# Returns
- `FigurePaths` with paths to generated figures
"""
function plot_publication_figure(config_path::AbstractString;
    project_root::Union{Nothing,AbstractString}=nothing)

    if project_root === nothing
        project_root = dirname(dirname(abspath(config_path)))
    end

    @info "Loading plot configuration" config = config_path
    config = load_config(config_path)

    paths_cfg = get(config, "paths", Dict{String,Any}())
    figure_cfg = get(config, "figure", Dict{String,Any}())

    identity_h5 = resolve_path(get(paths_cfg, "identity_h5", "plot_data/trajectory_identity.hdf5"), project_root)
    phi_sigma_h5 = resolve_path(get(paths_cfg, "phi_sigma_h5", "plot_data/trajectory.hdf5"), project_root)
    output_dir = resolve_path(get(paths_cfg, "output_dir", "plot_data"), project_root)

    @assert isfile(identity_h5) "Identity trajectory not found: $identity_h5"
    @assert isfile(phi_sigma_h5) "Phi/Sigma trajectory not found: $phi_sigma_h5"

    # Figure settings
    font_family = get(figure_cfg, "font_family", "TeX Gyre Heros")
    title_size = Int(get(figure_cfg, "title_size", 28))
    label_size = Int(get(figure_cfg, "label_size", 20))
    tick_size = Int(get(figure_cfg, "tick_size", 16))
    legend_size = Int(get(figure_cfg, "legend_size", 16))
    colormap_heatmap = Symbol(get(figure_cfg, "colormap_heatmap", "balance"))
    colormap_bivariate = Symbol(get(figure_cfg, "colormap_bivariate", "vik"))
    acf_max_time = Float64(get(figure_cfg, "acf_max_time", 100.0))
    smooth_window = Int(get(figure_cfg, "smooth_window", 2))

    @info "Figure font settings" font_family=font_family title_size=title_size label_size=label_size tick_size=tick_size legend_size=legend_size

    # Load trajectories
    @info "Loading trajectories"
    h5_id = load_trajectory_h5(identity_h5)
    h5_phi = load_trajectory_h5(phi_sigma_h5)

    traj_id = Array(h5_id.trajectory)
    traj_phi = Array(h5_phi.trajectory)

    L, C, T_id, E_id = size(traj_id)
    L_phi, C_phi, T_phi, E_phi = size(traj_phi)
    @assert L == L_phi && C == C_phi "Trajectory dimension mismatch"
    D = L * C

    langevin_tensor_id = reshape(traj_id, L, C, :)
    langevin_tensor_phi = reshape(traj_phi, L, C, :)

    # Load reference data
    dataset = load_hdf5_dataset(h5_phi.data_path;
        dataset_key=h5_phi.dataset_key,
        samples_orientation=h5_phi.dataset_orientation)
    data_clean = dataset.data

    dt_data = isfile(h5_phi.data_path_hr) ?
              h5open(h5 -> haskey(h5, "dt") ? read(h5, "dt") : 1.0, h5_phi.data_path_hr, "r") : 1.0
    dt_langevin = h5_phi.dt_effective

    @info "Data loaded" size = size(data_clean)

    # Use params from files
    pdf_stride = h5_id.pdf_stride
    pdf_nbins = h5_id.pdf_nbins
    biv_offsets = h5_id.biv_offsets
    biv_npoints = h5_id.biv_npoints
    biv_stride = h5_id.biv_stride
    heatmap_snapshots = h5_phi.heatmap_snapshots
    acf_max_lag = h5_phi.acf_max_lag

    # Compute statistics
    @info "Computing statistics"

    n_hm = min(heatmap_snapshots, size(traj_phi, 3))
    heatmap_langevin = Array(traj_phi[:, 1, 1:n_hm, 1])
    heatmap_data = Array(@view data_clean[:, 1, 1:n_hm])

    heat_range = maximum(abs, vcat(abs.(heatmap_langevin[:]), abs.(heatmap_data[:])))
    heat_range = heat_range == 0 ? 1.0 : heat_range

    value_bounds = quantile_bounds(data_clean, langevin_tensor_id; stride=20, probs=(0.001, 0.999))

    pdf_x, pdf_langevin = averaged_univariate_pdf(langevin_tensor_id; nbins=pdf_nbins, stride=pdf_stride, bounds=value_bounds)
    _, pdf_data = averaged_univariate_pdf(data_clean; nbins=pdf_nbins, stride=pdf_stride, bounds=value_bounds)

    acf_langevin = average_acf_ensemble(traj_phi, acf_max_lag)
    acf_data = average_acf_3d(data_clean, acf_max_lag)

    lags_data = collect(0:acf_max_lag) .* dt_data
    lags_langevin = collect(0:acf_max_lag) .* dt_langevin

    mask_data = lags_data .<= acf_max_time
    mask_lang = lags_langevin .<= acf_max_time

    bivariate_specs = Dict{Int,NamedTuple}()
    for offset in biv_offsets
        xg, yg, dens_lang = pair_kde(langevin_tensor_id, offset; bounds=value_bounds, npoints=biv_npoints, stride=biv_stride)
        _, _, dens_data = pair_kde(data_clean, offset; bounds=value_bounds, npoints=biv_npoints, stride=biv_stride)
        bivariate_specs[offset] = (; x=xg, y=yg, langevin=dens_lang, data=dens_data)
    end

    biv_offsets_sorted = sort(collect(keys(bivariate_specs)))

    # Generate main figure
    @info "Generating figures"

    fig = Figure(size=(1500, 2100), font=font_family)

    ax_hm_lang = Axis(fig[1, 1]; title=L"\textrm{Langevin snapshots}", xlabel="t (first $n_hm)", ylabel="mode",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    heatmap!(ax_hm_lang, 1:n_hm, 1:L, heatmap_langevin'; colormap=colormap_heatmap, colorrange=(-heat_range, heat_range))

    ax_hm_data = Axis(fig[1, 2]; title=L"\textrm{Data snapshots}", xlabel="t (first $n_hm)", ylabel="mode",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    hm_data = heatmap!(ax_hm_data, 1:n_hm, 1:L, heatmap_data'; colormap=colormap_heatmap, colorrange=(-heat_range, heat_range))
    Colorbar(fig[1, 3], hm_data; label="value", width=24, labelsize=label_size, ticklabelsize=tick_size)

    ax_pdf = Axis(fig[2, 1]; title=L"\textrm{Univariate PDF}", xlabel="value", ylabel="density",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    lines!(ax_pdf, pdf_x, pdf_data; color=:black, linewidth=3.0, label="Data")
    lines!(ax_pdf, pdf_x, pdf_langevin; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
    ylims!(ax_pdf, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
    axislegend(ax_pdf, position=:rt, framevisible=false, labelsize=legend_size)

    ax_acf = Axis(fig[2, 2]; title=L"\textrm{Average ACF}", xlabel="lag (time units)", ylabel="ACF",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    lines!(ax_acf, lags_data[mask_data], acf_data[mask_data]; color=:black, linewidth=3.0, label="Data")
    lines!(ax_acf, lags_langevin[mask_lang], acf_langevin[mask_lang]; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
    hlines!(ax_acf, [0.0]; color=:gray, linestyle=:dot, linewidth=1.5)
    xlims!(ax_acf, 0, acf_max_time)
    ylims!(ax_acf, -0.1, 1.05)
    axislegend(ax_acf, position=:rt, framevisible=false, labelsize=legend_size)

    # Use a single colorbar for all bivariate PDFs (shared colorrange).
    biv_density_max = 0.0
    for spec in values(bivariate_specs)
        biv_density_max = max(biv_density_max, maximum(spec.langevin), maximum(spec.data))
    end
    biv_density_max = biv_density_max == 0 ? 1e-9 : biv_density_max

    hm_biv_ref = nothing
    for (row, offset) in enumerate(biv_offsets_sorted)
        spec = bivariate_specs[offset]

        ax_l = Axis(fig[row+2, 1]; title=L"\textrm{Bivariate PDF (Langevin)}", xlabel="x[i]", ylabel="x[i+$offset]",
            titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
            xticklabelsize=tick_size, yticklabelsize=tick_size)
        heatmap!(ax_l, spec.x, spec.y, spec.langevin; colormap=colormap_bivariate, colorrange=(0, biv_density_max))

        ax_r = Axis(fig[row+2, 2]; title=L"\textrm{Bivariate PDF (Data)}", xlabel="x[i]", ylabel="x[i+$offset]",
            titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
            xticklabelsize=tick_size, yticklabelsize=tick_size)
        hm_r = heatmap!(ax_r, spec.x, spec.y, spec.data; colormap=colormap_bivariate, colorrange=(0, biv_density_max))
        hm_biv_ref === nothing && (hm_biv_ref = hm_r)
    end

    if hm_biv_ref !== nothing
        Colorbar(fig[3:(2+length(biv_offsets_sorted)), 3], hm_biv_ref;
            label="density", width=24, labelsize=label_size, ticklabelsize=tick_size)
    end

    ensure_dir(output_dir)
    main_path = joinpath(output_dir, "publication_figure.png")
    save(main_path, fig; px_per_unit=1)

    # ---------------------------------------------------------------------
    # Additional figure 1: Horizontal layout (4 columns x 2 rows)
    #   Col 1: snapshots (Langevin top, Data bottom)
    #   Col 2: univariate PDF (top) and ACF (bottom)
    #   Col 3-4: bivariate PDFs for first two offsets (Langevin top, Data bottom)
    # ---------------------------------------------------------------------
    # Horizontal figure should include offsets 1,2,3 (if available).
    offsets_h = [o for o in (1, 2, 3) if o in biv_offsets_sorted]
    if offsets_h != [1, 2, 3]
        @warn "Horizontal figure requested offsets (1,2,3) but some are missing; plotting available offsets" available=biv_offsets_sorted used=offsets_h
    end

    fig_h = Figure(size=(3300, 1100), font=font_family)

    ax_hm_lang_h = Axis(fig_h[1, 1]; title=L"\textrm{Langevin snapshots}", xlabel="t (first $n_hm)", ylabel="mode",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    hm_snap = heatmap!(ax_hm_lang_h, 1:n_hm, 1:L, heatmap_langevin'; colormap=colormap_heatmap, colorrange=(-heat_range, heat_range))

    ax_hm_data_h = Axis(fig_h[2, 1]; title=L"\textrm{Data snapshots}", xlabel="t (first $n_hm)", ylabel="mode",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    heatmap!(ax_hm_data_h, 1:n_hm, 1:L, heatmap_data'; colormap=colormap_heatmap, colorrange=(-heat_range, heat_range))

    # Single shared colorbar for the snapshots column.
    Colorbar(fig_h[1:2, 2], hm_snap; label="value", width=24, labelsize=label_size, ticklabelsize=tick_size)

    ax_pdf_h = Axis(fig_h[1, 3]; title=L"\textrm{PDFs}", xlabel="value", ylabel="density",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    lines!(ax_pdf_h, pdf_x, pdf_data; color=:black, linewidth=3.0, label="Data")
    lines!(ax_pdf_h, pdf_x, pdf_langevin; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
    ylims!(ax_pdf_h, 0, maximum(vcat(pdf_data, pdf_langevin)) * 1.05)
    axislegend(ax_pdf_h, position=:rt, framevisible=false, labelsize=legend_size)

    ax_acf_h = Axis(fig_h[2, 3]; title=L"\textrm{ACFs}", xlabel="lag (time units)", ylabel="ACF",
        titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
        xticklabelsize=tick_size, yticklabelsize=tick_size)
    lines!(ax_acf_h, lags_data[mask_data], acf_data[mask_data]; color=:black, linewidth=3.0, label="Data")
    lines!(ax_acf_h, lags_langevin[mask_lang], acf_langevin[mask_lang]; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
    hlines!(ax_acf_h, [0.0]; color=:gray, linestyle=:dot, linewidth=1.5)
    xlims!(ax_acf_h, 0, acf_max_time)
    ylims!(ax_acf_h, -0.1, 1.05)
    axislegend(ax_acf_h, position=:rt, framevisible=false, labelsize=legend_size)

    hm_biv_ref_h = nothing
    for (col_offset, offset) in enumerate(offsets_h)
        spec = bivariate_specs[offset]

        ax_biv_lang = Axis(fig_h[1, 3 + col_offset]; title=L"\textrm{Langevin PDF}", xlabel="x[i]", ylabel="x[i+$offset]",
            titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
            xticklabelsize=tick_size, yticklabelsize=tick_size)
        hm_lang = heatmap!(ax_biv_lang, spec.x, spec.y, spec.langevin; colormap=colormap_bivariate, colorrange=(0, biv_density_max))
        hm_biv_ref_h === nothing && (hm_biv_ref_h = hm_lang)

        ax_biv_data = Axis(fig_h[2, 3 + col_offset]; title=L"\textrm{Data PDF}", xlabel="x[i]", ylabel="x[i+$offset]",
            titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
            xticklabelsize=tick_size, yticklabelsize=tick_size)
        heatmap!(ax_biv_data, spec.x, spec.y, spec.data; colormap=colormap_bivariate, colorrange=(0, biv_density_max))
    end

    if hm_biv_ref_h !== nothing
        # Place the shared bivariate colorbar after the last bivariate column.
        Colorbar(fig_h[1:2, 4 + length(offsets_h)], hm_biv_ref_h;
            label="density", width=24, labelsize=label_size, ticklabelsize=tick_size)
    end

    horizontal_path = joinpath(output_dir, "publication_figure_horizontal.png")
    save(horizontal_path, fig_h; px_per_unit=1)

    # ---------------------------------------------------------------------
    # Additional figure 2: Single-row heatmaps of Phi/Phi_S/Phi_A/Sigma/V
    #   V is shown as diffusion covariance: V = Sigma * Sigma'
    # ---------------------------------------------------------------------
    matrices_path = nothing
    if h5_phi.Phi !== nothing && h5_phi.Sigma !== nothing
        Phi_mat = Array(h5_phi.Phi)
        Sigma_mat = Array(h5_phi.Sigma)
        Phi_S = 0.5 .* (Phi_mat .+ Phi_mat')
        Phi_A = 0.5 .* (Phi_mat .- Phi_mat')

        # Compute <s(x) x^T> (Stein matrix) as in `estimate_phi_sigma_ks.jl`.
        V_data = nothing
        if h5_phi.model_path === nothing
            @warn "Skipping <s(x)x^T> (model_path not found in trajectory HDF5)" phi_sigma_h5
        else
            model, _, trainer_cfg = load_model(String(h5_phi.model_path))
            model = Flux.cpu(model)
            sigma = trainer_cfg === nothing ? 0.1f0 : Float32(trainer_cfg.sigma)

            noise = randn(Float32, size(data_clean)...)
            data_noisy = data_clean .+ sigma .* noise
            V_data = compute_V_data(data_noisy, model, sigma; v_data_resolution=10, batch_size=512)
        end

        if V_data !== nothing
            # Shared colormap and shared (symmetric) colorrange across all panels.
            clim_all = 0.0
            for M in (Phi_mat, Phi_S, Phi_A, Sigma_mat, V_data)
                clim_all = max(clim_all, maximum(abs, M))
            end
            clim_all = clim_all == 0 ? 1e-12 : clim_all

            fig_m = Figure(size=(3000, 520), font=font_family)

            ax1 = Axis(fig_m[1, 1]; title=L"\Phi", xlabel="col", ylabel="row",
                titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
                xticklabelsize=tick_size, yticklabelsize=tick_size)
            heatmap!(ax1, Phi_mat; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

            ax2 = Axis(fig_m[1, 2]; title=L"\Phi_S", xlabel="col", ylabel="row",
                titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
                xticklabelsize=tick_size, yticklabelsize=tick_size)
            heatmap!(ax2, Phi_S; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

            ax3 = Axis(fig_m[1, 3]; title=L"\Phi_A", xlabel="col", ylabel="row",
                titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
                xticklabelsize=tick_size, yticklabelsize=tick_size)
            heatmap!(ax3, Phi_A; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

            ax4 = Axis(fig_m[1, 4]; title=L"\Sigma", xlabel="col", ylabel="row",
                titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
                xticklabelsize=tick_size, yticklabelsize=tick_size)
            heatmap!(ax4, Sigma_mat; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

            ax5 = Axis(fig_m[1, 5]; title=L"\langle s(x)x^T \rangle", xlabel="col", ylabel="row",
                titlesize=title_size, xlabelsize=label_size, ylabelsize=label_size,
                xticklabelsize=tick_size, yticklabelsize=tick_size)
            hm_v = heatmap!(ax5, V_data; colormap=colormap_heatmap, colorrange=(-clim_all, clim_all))

            Colorbar(fig_m[1, 6], hm_v; label="value", width=24, labelsize=label_size, ticklabelsize=tick_size)

            matrices_path = joinpath(output_dir, "publication_matrices_row.png")
            save(matrices_path, fig_m; px_per_unit=1)
        else
            @warn "Skipping matrices row figure (<s(x)x^T> not available)" phi_sigma_h5
        end
    else
        @warn "Skipping matrices row figure (Phi/Sigma not found in trajectory HDF5)" phi_sigma_h5
    end

    @info "Publication figure complete" path = main_path

    return FigurePaths(main_path, nothing, horizontal_path, matrices_path)
end

"""
    plot_comparison_figure(result::IntegrationResult, output_path; kwargs...) -> String

Generate comparison figure directly from IntegrationResult (without needing config file).
"""
function plot_comparison_figure(stats::Dict{Symbol,Any}, output_path::AbstractString;
    font_family::String="TeX Gyre Heros",
    title_size::Int=28)

    # Extract statistics
    pdf_x = stats[:pdf_x]
    pdf_langevin = stats[:pdf_langevin]
    pdf_data = stats[:pdf_data]
    acf_langevin = stats[:acf_langevin]
    acf_data = stats[:acf_data]
    lags_data = stats[:lags_data]
    lags_langevin = stats[:lags_langevin]
    bivariate_specs = stats[:bivariate_specs]
    heatmap_langevin = stats[:heatmap_langevin]
    heatmap_data = stats[:heatmap_data]
    L = stats[:L]

    n_hm = size(heatmap_langevin, 2)
    heat_range = maximum(abs, vcat(abs.(heatmap_langevin[:]), abs.(heatmap_data[:])))
    heat_range = heat_range == 0 ? 1.0 : heat_range

    mask_data = lags_data .<= 100.0
    mask_lang = lags_langevin .<= 100.0

    biv_offsets_sorted = sort(collect(keys(bivariate_specs)))

    fig = Figure(size=(1500, 2100), font=font_family)

    ax_hm_lang = Axis(fig[1, 1]; title="Langevin", xlabel="t", ylabel="mode", titlesize=title_size)
    heatmap!(ax_hm_lang, 1:n_hm, 1:L, heatmap_langevin'; colormap=:balance, colorrange=(-heat_range, heat_range))

    ax_hm_data = Axis(fig[1, 2]; title="Data", xlabel="t", ylabel="mode", titlesize=title_size)
    hm = heatmap!(ax_hm_data, 1:n_hm, 1:L, heatmap_data'; colormap=:balance, colorrange=(-heat_range, heat_range))
    Colorbar(fig[1, 3], hm; label="value", width=12)

    ax_pdf = Axis(fig[2, 1]; title="PDF", xlabel="value", ylabel="density", titlesize=title_size)
    lines!(ax_pdf, pdf_x, pdf_data; color=:black, linewidth=3.0, label="Data")
    lines!(ax_pdf, pdf_x, pdf_langevin; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
    axislegend(ax_pdf, position=:rt)

    ax_acf = Axis(fig[2, 2]; title="ACF", xlabel="lag", ylabel="ACF", titlesize=title_size)
    lines!(ax_acf, lags_data[mask_data], acf_data[mask_data]; color=:black, linewidth=3.0, label="Data")
    lines!(ax_acf, lags_langevin[mask_lang], acf_langevin[mask_lang]; color=:firebrick, linewidth=3.0, linestyle=:dash, label="Langevin")
    xlims!(ax_acf, 0, 100)
    ylims!(ax_acf, -0.1, 1.05)
    axislegend(ax_acf, position=:rt)

    for (row, offset) in enumerate(biv_offsets_sorted)
        spec = bivariate_specs[offset]
        dmax = max(maximum(spec.langevin), maximum(spec.data))

        ax_l = Axis(fig[row+2, 1]; title="Langevin j=$offset", titlesize=title_size)
        heatmap!(ax_l, spec.x, spec.y, spec.langevin; colormap=:vik, colorrange=(0, dmax))

        ax_r = Axis(fig[row+2, 2]; title="Data j=$offset", titlesize=title_size)
        hm_r = heatmap!(ax_r, spec.x, spec.y, spec.data; colormap=:vik, colorrange=(0, dmax))
        Colorbar(fig[row+2, 3], hm_r; label="density", width=12)
    end

    ensure_dir(dirname(output_path))
    save(output_path, fig; px_per_unit=1)

    return output_path
end

end # module
