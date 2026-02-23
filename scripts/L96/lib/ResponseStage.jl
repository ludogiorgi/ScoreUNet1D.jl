module L96ResponseStage

using TOML
using Printf
using HDF5
using Plots

const RESP_TARGET_TMAX = 2.0
const RESP_MIN_NATIVE_POINTS = 301
const RESP_MIN_OUTPUT_POINTS = 250

function _append_log(log_path::AbstractString, msg::AbstractString)
    open(log_path, "a") do io
        println(io, msg)
    end
end

function _run_logged(cmd::Cmd, log_path::AbstractString)
    open(log_path, "a") do io
        run(pipeline(cmd; stdout=io, stderr=io))
    end
end

function _table!(cfg::Dict{String,Any}, key::String)
    if !haskey(cfg, key) || !(cfg[key] isa AbstractDict)
        cfg[key] = Dict{String,Any}()
    elseif !(cfg[key] isa Dict{String,Any})
        cfg[key] = Dict{String,Any}(cfg[key])
    end
    return cfg[key]
end

function _latest_subdir(root::AbstractString)
    isdir(root) || return ""
    dirs = String[]
    for name in readdir(root)
        p = joinpath(root, name)
        isdir(p) && push!(dirs, p)
    end
    isempty(dirs) && return ""
    sort!(dirs; by=p -> stat(p).mtime)
    return dirs[end]
end

function _dataset_sample_count(path::AbstractString, key::AbstractString)
    isfile(path) || error("Response dataset file not found: $path")
    return h5open(path, "r") do h5
        haskey(h5, key) || error("Response dataset key '$key' not found in $path")
        dset = h5[key]
        n_total = size(dset, 1)
        n_total >= 1 || error("Response dataset '$key' in $path has no samples")
        return Int(n_total)
    end
end

function _clamp_subset!(cfg::Dict{String,Any},
    n_key::String,
    start_key::String,
    n_total::Int,
    label::String,
    log_path::AbstractString)
    n_raw = Int(get(cfg, n_key, 1))
    s_raw = Int(get(cfg, start_key, 1))

    n_req = max(1, n_raw)
    s_req = max(1, s_raw)

    n_use = min(n_req, n_total)
    max_start = max(1, n_total - n_use + 1)
    s_use = clamp(s_req, 1, max_start)

    cfg[n_key] = n_use
    cfg[start_key] = s_use

    if n_use != n_raw || s_use != s_raw
        _append_log(
            log_path,
            @sprintf(
                "[response] clamped %s subset: %s %d->%d, %s %d->%d, dataset_samples=%d",
                label,
                n_key,
                n_raw,
                n_use,
                start_key,
                s_raw,
                s_use,
                n_total,
            ),
        )
    end
end

function _response_params_path(params::Dict{String,Any}; base_dir::AbstractString=pwd())
    raw = String(get(params, "responses.params_path", "scripts/L96/parameters_responses.toml"))
    return isabspath(raw) ? raw : abspath(joinpath(base_dir, raw))
end

function _parse_methods_override(raw::AbstractString)
    s = lowercase(strip(raw))
    isempty(s) && return String[]
    vals = String[]
    for tok in split(s, ",")
        t = strip(tok)
        isempty(t) && continue
        t in ("gaussian", "unet", "numerical") || error("Unsupported response method override '$t'. Expected gaussian,unet,numerical")
        push!(vals, t)
    end
    return unique(vals)
end

function _highres_save_every(dt::Float64, requested::Int)
    dt > 0 || return max(requested, 1)
    max_save_every = max(1, Int(floor(RESP_TARGET_TMAX / ((RESP_MIN_NATIVE_POINTS - 1) * dt))))
    if requested <= 0
        return max_save_every
    end
    return min(max(requested, 1), max_save_every)
end

function _assert_response_output_grid(run_dir::AbstractString)
    h5_path = joinpath(run_dir, "responses_5x4_selected_methods.hdf5")
    isfile(h5_path) || error("Response HDF5 output not found: $h5_path")
    n_times = h5open(h5_path, "r") do h5
        haskey(h5, "times") || error("Response HDF5 missing 'times' dataset: $h5_path")
        return length(read(h5["times"]))
    end
    n_times >= RESP_MIN_OUTPUT_POINTS || error("Response output too coarse: got $n_times time points, expected at least $(RESP_MIN_OUTPUT_POINTS)")
    return n_times
end

function _eval_epoch_from_name(name::AbstractString)
    m = match(r"^eval_epoch_(\d+)$", String(name))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function _read_response_h5(path::AbstractString)
    return h5open(path, "r") do h5
        haskey(h5, "times") || return nothing
        times = Float64.(read(h5["times"]))
        responses = Dict{String,Array{Float64,3}}()
        for key in (
            "responses/gfdt_unet_corrected",
            "responses/gfdt_unet_raw",
            "responses/gfdt_gaussian_corrected",
            "responses/gfdt_gaussian_raw",
            "responses/numerical_integration",
        )
            haskey(h5, key) || continue
            responses[split(key, "/")[end]] = Float64.(read(h5[key]))
        end
        return (times=times, responses=responses)
    end
end

function _interp_response_time(data::Array{Float64,3}, t_in::Vector{Float64}, t_out::Vector{Float64})
    m, p, n_in = size(data)
    n_out = length(t_out)
    n_in == length(t_in) || error("Time/data mismatch in response interpolation")
    out = zeros(Float64, m, p, n_out)
    for i in 1:m, j in 1:p
        for (k, t) in enumerate(t_out)
            if t <= t_in[1]
                out[i, j, k] = data[i, j, 1]
            elseif t >= t_in[end]
                out[i, j, k] = data[i, j, end]
            else
                idx = searchsortedlast(t_in, t)
                t0 = t_in[idx]
                t1 = t_in[idx+1]
                w = (t - t0) / (t1 - t0)
                out[i, j, k] = data[i, j, idx] + w * (data[i, j, idx+1] - data[i, j, idx])
            end
        end
    end
    return out
end

function _collect_unet_history(figures_root::AbstractString, current_epoch::Int)
    curves = NamedTuple[]
    isdir(figures_root) || return curves

    for name in readdir(figures_root)
        epoch = _eval_epoch_from_name(name)
        epoch === nothing && continue
        epoch > current_epoch && continue

        eval_dir = joinpath(figures_root, name)
        responses_root = joinpath(eval_dir, "responses")
        run_dir = _latest_subdir(responses_root)
        isempty(run_dir) && continue

        h5_path = joinpath(run_dir, "responses_5x4_selected_methods.hdf5")
        isfile(h5_path) || continue
        payload = _read_response_h5(h5_path)
        payload === nothing && continue
        responses = payload.responses
        data = haskey(responses, "gfdt_unet_corrected") ? responses["gfdt_unet_corrected"] :
            (haskey(responses, "gfdt_unet_raw") ? responses["gfdt_unet_raw"] : nothing)
        data === nothing && continue
        length(payload.times) == size(data, 3) || continue

        push!(curves, (epoch=epoch, times=payload.times, data=data))
    end

    sort!(curves; by=c -> c.epoch)
    return curves
end

function _align_curve_to_times(data::Array{Float64,3}, times::Vector{Float64}, ref_times::Vector{Float64})
    if length(times) == length(ref_times) && times == ref_times && size(data, 3) == length(ref_times)
        return data
    end
    return _interp_response_time(data, times, ref_times)
end

function _save_unet_history_figure(path::AbstractString,
    curves::Vector{<:NamedTuple};
    min_alpha::Float64=0.2,
    gaussian_curve::Union{Nothing,NamedTuple}=nothing,
    numerical_curve::Union{Nothing,NamedTuple}=nothing)
    isempty(curves) && return false
    alpha_floor = clamp(min_alpha, 0.0, 1.0)

    ref = curves[end]
    ref_times = ref.times
    m, p, nt = size(ref.data)
    length(ref_times) == nt || return false

    aligned = NamedTuple[]
    for c in curves
        data = _align_curve_to_times(c.data, c.times, ref_times)
        push!(aligned, (epoch=c.epoch, data=data))
    end

    gauss_data = gaussian_curve === nothing ? nothing : _align_curve_to_times(gaussian_curve.data, gaussian_curve.times, ref_times)
    num_data = numerical_curve === nothing ? nothing : _align_curve_to_times(numerical_curve.data, numerical_curve.times, ref_times)

    default(fontfamily="Computer Modern", dpi=180, legendfontsize=8, guidefontsize=9, tickfontsize=8, titlefontsize=10)
    param_names = ["F", "h", "c", "b"]
    resp_obs_labels = [
        "phi1 = <Xk>",
        "phi2 = <Yk,j>",
        "phi3 = <Xk^2>",
        "phi4 = <Yk,j^2>",
        "phi5 = <Xk X_(k-1)>",
    ]

    n_curves = length(aligned)
    panels = Vector{Plots.Plot}(undef, m * p)
    for i in 1:m, j in 1:p
        idx = (i - 1) * p + j
        legend_mode = (idx == 1 ? :topright : false)
        title_txt = i == 1 ? "d/d" * param_names[j] : ""
        ylabel_txt = j == 1 ? resp_obs_labels[i] : ""
        xlabel_txt = i == m ? "time" : ""

        pn = plot(; legend=legend_mode, title=title_txt, ylabel=ylabel_txt, xlabel=xlabel_txt)
        if num_data !== nothing
            plot!(pn, ref_times, vec(@view num_data[i, j, :]); color=:dodgerblue3, linestyle=:solid, linewidth=2.0, label=(idx == 1 ? "Numerical" : ""))
        end
        if gauss_data !== nothing
            plot!(pn, ref_times, vec(@view gauss_data[i, j, :]); color=:black, linestyle=:dash, linewidth=2.0, label=(idx == 1 ? "GFDT+Gaussian" : ""))
        end
        for (k, c) in enumerate(aligned)
            alpha = n_curves == 1 ? 1.0 : alpha_floor + (1.0 - alpha_floor) * (k - 1) / (n_curves - 1)
            line_w = k == n_curves ? 2.2 : 1.6
            label = idx == 1 ? "UNet epoch " * lpad(string(c.epoch), 4, '0') : ""
            plot!(pn, ref_times, vec(@view c.data[i, j, :]); color=:orangered3, linealpha=alpha, linewidth=line_w, label=label)
        end
        hline!(pn, [0.0]; color=:gray55, linestyle=:dot, linewidth=1.0, label="")
        panels[idx] = pn
    end

    fig = plot(panels...; layout=(m, p), size=(2200, 1700))
    mkpath(dirname(path))
    savefig(fig, path)
    return true
end

function run_response_figure!(params::Dict{String,Any},
    dirs::Dict{String,String},
    epoch::Int,
    model_path::AbstractString,
    eval_tag::AbstractString;
    base_dir::AbstractString=pwd())
    enabled = Bool(get(params, "responses.enabled", true))
    !enabled && return Dict{String,Any}("enabled" => false, "figD_path" => "")

    log_path = joinpath(dirs["logs"], "pipeline.log")
    template = _response_params_path(params; base_dir=base_dir)
    isfile(template) || error("Response parameters TOML not found: $template")

    eval_dir = joinpath(dirs["figures"], eval_tag)
    output_root = joinpath(eval_dir, "responses")
    mkpath(output_root)

    cfg = TOML.parsefile(template)
    paths_cfg = _table!(cfg, "paths")
    integ_cfg = _table!(cfg, "integration")
    dset_cfg = _table!(cfg, "dataset")
    meth_cfg = _table!(cfg, "methods")
    gfdt_cfg = _table!(cfg, "gfdt")
    num_cfg = _table!(cfg, "numerical")
    ref_cfg = _table!(cfg, "reference")

    paths_cfg["run_dir"] = abspath(dirs["run"])
    paths_cfg["checkpoint_path"] = abspath(model_path)
    paths_cfg["checkpoint_epoch"] = epoch
    paths_cfg["output_root"] = abspath(output_root)
    paths_cfg["run_prefix"] = String(get(params, "responses.run_prefix", "run_"))

    integ_cfg["K"] = Int(params["data.generation.K"])
    integ_cfg["J"] = Int(params["run.J"])
    integ_cfg["F"] = Float64(params["data.generation.F"])
    integ_cfg["h"] = Float64(params["data.generation.h"])
    integ_cfg["c"] = Float64(params["data.generation.c"])
    integ_cfg["b"] = Float64(params["data.generation.b"])
    integ_cfg["dt"] = Float64(params["data.generation.dt"])
    save_every_override = Int(get(params, "responses.save_every_override", 0))
    requested_save_every = save_every_override > 0 ? save_every_override : Int(params["data.generation.save_every"])
    highres_save_every = _highres_save_every(Float64(params["data.generation.dt"]), requested_save_every)
    integ_cfg["save_every"] = highres_save_every
    integ_cfg["process_noise_sigma"] = Float64(params["data.generation.process_noise_sigma"])
    integ_cfg["stochastic_x_noise"] = Bool(get(params, "data.generation.stochastic_x_noise", false))
    if highres_save_every != requested_save_every
        _append_log(log_path, "[response] adjusted integration.save_every for high-resolution responses: $(requested_save_every)->$(highres_save_every)")
    end

    dset_path = abspath(joinpath(base_dir, String(params["data.path"])))
    dset_key = String(params["data.dataset_key"])
    dset_cfg["path"] = dset_path
    dset_cfg["key"] = dset_key
    dset_cfg["attr_sync_mode"] = lowercase(String(get(params, "responses.dataset_attr_sync_mode", "override")))

    response_score_device = String(get(params, "responses.score_device", params["langevin.device"]))
    !isempty(strip(response_score_device)) && (gfdt_cfg["score_device"] = response_score_device)
    gfdt_cfg["response_tmax"] = RESP_TARGET_TMAX

    gfdt_nsamples_override = Int(get(params, "responses.gfdt_nsamples_override", 0))
    gfdt_nsamples_override > 0 && (gfdt_cfg["nsamples"] = gfdt_nsamples_override)

    num_ens_override = Int(get(params, "responses.numerical_ensembles_override", 0))
    num_ens_override > 0 && (num_cfg["ensembles"] = num_ens_override)

    ref_cfg["cache_root"] = String(get(params, "responses.reference.cache_root", "scripts/L96/reference_responses_cache"))
    ref_cfg["force_regenerate"] = Bool(get(params, "responses.reference.force_regenerate", false))
    ref_cfg["gfdt_nsamples"] = Int(get(params, "responses.reference.gfdt_nsamples", get(gfdt_cfg, "nsamples", 200_000)))
    ref_cfg["gfdt_start_index"] = Int(get(params, "responses.reference.gfdt_start_index", get(gfdt_cfg, "start_index", 50_001)))
    ref_cfg["numerical_ensembles"] = Int(get(params, "responses.reference.numerical_ensembles", get(num_cfg, "ensembles", 16_384)))
    ref_cfg["numerical_start_index"] = Int(get(params, "responses.reference.numerical_start_index", get(num_cfg, "start_index", 80_001)))
    ref_cfg["numerical_method"] = String(get(params, "responses.reference.numerical_method", "tangent"))
    ref_cfg["h_rel"] = Float64(get(params, "responses.reference.h_rel", get(num_cfg, "h_rel", 5e-3)))
    ref_cfg["h_abs"] = Float64.(collect(get(params, "responses.reference.h_abs", get(num_cfg, "h_abs", [1e-2, 1e-3, 1e-2, 1e-2]))))
    ref_cfg["numerical_seed_base"] = Int(get(params, "responses.reference.numerical_seed_base", get(num_cfg, "seed_base", 1_920_000)))
    ref_cfg["tmax"] = Float64(get(params, "responses.reference.tmax", RESP_TARGET_TMAX))
    ref_cfg["mean_center"] = Bool(get(params, "responses.reference.mean_center", get(gfdt_cfg, "mean_center", true)))

    plot_gaussian = Bool(get(params, "responses.plot_gaussian", false))
    plot_numerical = Bool(get(params, "responses.plot_numerical", false))
    history_min_alpha = Float64(get(params, "responses.history_min_alpha", 0.2))

    methods_override = _parse_methods_override(String(get(params, "responses.methods_override", "")))
    use_gaussian = Bool(get(meth_cfg, "gaussian", false))
    use_unet = Bool(get(meth_cfg, "unet", true))
    use_numerical = Bool(get(meth_cfg, "numerical", false))
    if !isempty(methods_override)
        use_gaussian = "gaussian" in methods_override
        use_unet = "unet" in methods_override
        use_numerical = "numerical" in methods_override
    end
    # Pipeline response stage always computes current UNet responses.
    use_unet = true
    # Overlay toggles require baseline availability in the current response file.
    plot_gaussian && (use_gaussian = true)
    plot_numerical && (use_numerical = true)
    meth_cfg["gaussian"] = use_gaussian
    meth_cfg["unet"] = use_unet
    meth_cfg["numerical"] = use_numerical

    n_total = _dataset_sample_count(dset_path, dset_key)
    _clamp_subset!(gfdt_cfg, "nsamples", "start_index", n_total, "gfdt", log_path)
    _clamp_subset!(num_cfg, "ensembles", "start_index", n_total, "numerical", log_path)

    tmp_cfg_path = joinpath(eval_dir, @sprintf("responses_epoch_%04d.toml", epoch))
    open(tmp_cfg_path, "w") do io
        TOML.print(io, cfg)
    end

    argv = String[
        "julia",
        "--threads",
        "auto",
        "--project=.",
        "scripts/L96/compute_responses.jl",
        "--params",
        tmp_cfg_path,
    ]
    _append_log(log_path, "[response] method flags gaussian=$(use_gaussian) unet=$(use_unet) numerical=$(use_numerical)")

    _append_log(log_path, "[response] start epoch=$epoch checkpoint=$(abspath(model_path))")
    _run_logged(Cmd(argv), log_path)

    run_dir = _latest_subdir(output_root)
    isempty(run_dir) && error("Response stage did not create an output run folder under $output_root")
    n_times = _assert_response_output_grid(run_dir)
    _append_log(log_path, "[response] output_time_points=$(n_times)")
    h5_path = joinpath(run_dir, "responses_5x4_selected_methods.hdf5")
    payload = _read_response_h5(h5_path)

    fig_corr = joinpath(run_dir, "responses_5x4_selected_methods_corrected.png")
    fig_raw = joinpath(run_dir, "responses_5x4_selected_methods_raw.png")
    fig_src = isfile(fig_corr) ? fig_corr : fig_raw
    isfile(fig_src) || error("Response stage did not produce expected response figure in $run_dir")

    figD = joinpath(eval_dir, "figD_responses_5x4.png")
    figures_root = dirs["figures"]
    unet_history = _collect_unet_history(figures_root, epoch)
    gaussian_curve = nothing
    numerical_curve = nothing
    if payload !== nothing
        if plot_gaussian
            gauss_key = haskey(payload.responses, "gfdt_gaussian_corrected") ? "gfdt_gaussian_corrected" :
                (haskey(payload.responses, "gfdt_gaussian_raw") ? "gfdt_gaussian_raw" : "")
            !isempty(gauss_key) && (gaussian_curve = (times=payload.times, data=payload.responses[gauss_key]))
        end
        if plot_numerical && haskey(payload.responses, "numerical_integration")
            numerical_curve = (times=payload.times, data=payload.responses["numerical_integration"])
        end
    end

    made_overlay = _save_unet_history_figure(
        figD,
        unet_history;
        min_alpha=history_min_alpha,
        gaussian_curve=gaussian_curve,
        numerical_curve=numerical_curve,
    )
    if !made_overlay
        cp(fig_src, figD; force=true)
    else
        _append_log(log_path, "[response] figD history overlay curves=$(length(unet_history)) min_alpha=$(history_min_alpha) plot_gaussian=$(plot_gaussian) plot_numerical=$(plot_numerical)")
    end

    isfile(tmp_cfg_path) && rm(tmp_cfg_path; force=true)
    _append_log(log_path, "[response] done epoch=$epoch figD=$(abspath(figD))")

    return Dict{String,Any}(
        "enabled" => true,
        "figD_path" => figD,
        "response_run_dir" => run_dir,
    )
end

end # module
