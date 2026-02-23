module L96Config

using TOML

function _require(cfg::Dict{String,Any}, key::String)
    haskey(cfg, key) || error("Missing required section/key: $key")
    return cfg[key]
end

function _as_dict(x, name::String)
    x isa Dict{String,Any} || error("Expected [$name] to be a TOML table")
    return x
end

function _get(d::Dict{String,Any}, k::String, default)
    return haskey(d, k) ? d[k] : default
end

function _as_int(x, label::String)
    x isa Integer || error("$label must be integer, got $(typeof(x))")
    return Int(x)
end

function _as_float(x, label::String)
    x isa Real || error("$label must be numeric, got $(typeof(x))")
    return Float64(x)
end

function _as_bool(x, label::String)
    x isa Bool || error("$label must be bool, got $(typeof(x))")
    return Bool(x)
end

function _as_string(x, label::String)
    x isa AbstractString || error("$label must be string, got $(typeof(x))")
    s = strip(String(x))
    isempty(s) && error("$label cannot be empty")
    return s
end

function _as_string_allow_empty(x, label::String)
    x isa AbstractString || error("$label must be string, got $(typeof(x))")
    return strip(String(x))
end

function _as_int_vec(x, label::String)
    x isa Vector || error("$label must be array")
    vals = Int[]
    for v in x
        v isa Integer || error("$label entries must be integer")
        push!(vals, Int(v))
    end
    isempty(vals) && error("$label cannot be empty")
    return vals
end

function _as_float_vec(x, label::String)
    x isa Vector || error("$label must be array")
    vals = Float64[]
    for v in x
        v isa Real || error("$label entries must be numeric")
        push!(vals, Float64(v))
    end
    isempty(vals) && error("$label cannot be empty")
    return vals
end

function load_parameters(path::AbstractString)
    isfile(path) || error("Parameters file not found: $path")
    cfg = TOML.parsefile(path)

    run = _as_dict(_require(cfg, "run"), "run")
    data = _as_dict(_require(cfg, "data"), "data")
    data_gen = haskey(data, "generation") ? _as_dict(data["generation"], "data.generation") : Dict{String,Any}()
    train = _as_dict(_require(cfg, "train"), "train")
    train_ema = haskey(train, "ema") ? _as_dict(train["ema"], "train.ema") : Dict{String,Any}()
    train_loss = haskey(train, "loss") ? _as_dict(train["loss"], "train.loss") : Dict{String,Any}()
    train_resume = haskey(train, "resume") ? _as_dict(train["resume"], "train.resume") : Dict{String,Any}()
    kl_eval = _as_dict(_require(train, "kl_eval"), "train.kl_eval")
    langevin = _as_dict(_require(cfg, "langevin"), "langevin")
    langevin_quick = haskey(langevin, "quick") ? _as_dict(langevin["quick"], "langevin.quick") : Dict{String,Any}()
    responses = haskey(cfg, "responses") ? _as_dict(cfg["responses"], "responses") : Dict{String,Any}()
    responses_reference = haskey(responses, "reference") ? _as_dict(responses["reference"], "responses.reference") : Dict{String,Any}()
    figures = _as_dict(_require(cfg, "figures"), "figures")
    output = _as_dict(_require(cfg, "output"), "output")

    params = Dict{String,Any}(
        "_raw" => cfg,
        "_path" => abspath(path),

        "run.J" => _as_int(_get(run, "J", 10), "run.J"),
        "run.runs_root" => _as_string(_get(run, "runs_root", "scripts/L96"), "run.runs_root"),
        "run.run_folder_prefix" => _as_string(_get(run, "run_folder_prefix", "runs_J"), "run.run_folder_prefix"),
        "run.run_id_padding" => _as_int(_get(run, "run_id_padding", 3), "run.run_id_padding"),
        "run.seed" => _as_int(_get(run, "seed", 42), "run.seed"),

        "data.path" => _as_string(_get(data, "path", "scripts/L96/l96_timeseries.hdf5"), "data.path"),
        "data.observations_root" => _as_string(_get(data, "observations_root", "scripts/L96/observations"), "data.observations_root"),
        "data.dataset_filename" => _as_string(_get(data, "dataset_filename", "l96_timeseries.hdf5"), "data.dataset_filename"),
        "data.integration_params_filename" => _as_string(_get(data, "integration_params_filename", "integration_params.toml"), "data.integration_params_filename"),
        "data.dataset_key" => _as_string(_get(data, "dataset_key", "timeseries"), "data.dataset_key"),
        "data.normalization_mode" => _as_string(_get(data, "normalization_mode", "split_xy"), "data.normalization_mode"),
        "data.generate_if_missing" => _as_bool(_get(data, "generate_if_missing", false), "data.generate_if_missing"),
        "data.generation.K" => _as_int(_get(data_gen, "K", 36), "data.generation.K"),
        "data.generation.F" => _as_float(_get(data_gen, "F", 10.0), "data.generation.F"),
        "data.generation.h" => _as_float(_get(data_gen, "h", 1.0), "data.generation.h"),
        "data.generation.c" => _as_float(_get(data_gen, "c", 10.0), "data.generation.c"),
        "data.generation.b" => _as_float(_get(data_gen, "b", 10.0), "data.generation.b"),
        "data.generation.dt" => _as_float(_get(data_gen, "dt", 0.005), "data.generation.dt"),
        "data.generation.spinup_steps" => _as_int(_get(data_gen, "spinup_steps", 20_000), "data.generation.spinup_steps"),
        "data.generation.save_every" => _as_int(_get(data_gen, "save_every", 10), "data.generation.save_every"),
        "data.generation.nsamples" => _as_int(_get(data_gen, "nsamples", 12_000), "data.generation.nsamples"),
        "data.generation.process_noise_sigma" => _as_float(_get(data_gen, "process_noise_sigma", 0.03), "data.generation.process_noise_sigma"),
        "data.generation.rng_seed" => _as_int(_get(data_gen, "rng_seed", _get(run, "seed", 42)), "data.generation.rng_seed"),

        "train.device" => _as_string(_get(train, "device", "GPU:0"), "train.device"),
        "train.num_training_epochs" => _as_int(_get(train, "num_training_epochs", 80), "train.num_training_epochs"),
        "train.batch_size" => _as_int(_get(train, "batch_size", 256), "train.batch_size"),
        "train.lr" => _as_float(_get(train, "lr", 8e-4), "train.lr"),
        "train.sigma" => _as_float(_get(train, "sigma", 0.05), "train.sigma"),
        "train.base_channels" => _as_int(_get(train, "base_channels", 32), "train.base_channels"),
        "train.channel_multipliers" => _as_int_vec(_get(train, "channel_multipliers", [1, 2, 4]), "train.channel_multipliers"),
        "train.model_arch" => lowercase(_as_string(_get(train, "model_arch", "schneider_dualstream"), "train.model_arch")),
        "train.progress" => _as_bool(_get(train, "progress", false), "train.progress"),
        "train.use_lr_schedule" => _as_bool(_get(train, "use_lr_schedule", true), "train.use_lr_schedule"),
        "train.warmup_steps" => _as_int(_get(train, "warmup_steps", 500), "train.warmup_steps"),
        "train.min_lr_factor" => _as_float(_get(train, "min_lr_factor", 0.1), "train.min_lr_factor"),
        "train.norm_type" => _as_string(_get(train, "norm_type", "group"), "train.norm_type"),
        "train.norm_groups" => _as_int(_get(train, "norm_groups", 0), "train.norm_groups"),
        "train.loss.x_weight" => _as_float(_get(train_loss, "x_weight", 1.0), "train.loss.x_weight"),
        "train.loss.y_weight" => _as_float(_get(train_loss, "y_weight", 1.0), "train.loss.y_weight"),
        "train.loss.mean_weight" => _as_float(_get(train_loss, "mean_weight", 0.0), "train.loss.mean_weight"),
        "train.loss.cov_weight" => _as_float(_get(train_loss, "cov_weight", 0.0), "train.loss.cov_weight"),
        "train.ema.enabled" => _as_bool(_get(train_ema, "enabled", true), "train.ema.enabled"),
        "train.ema.decay" => _as_float(_get(train_ema, "decay", 0.999), "train.ema.decay"),
        "train.ema.use_for_eval" => _as_bool(_get(train_ema, "use_for_eval", true), "train.ema.use_for_eval"),
        "train.resume.enabled" => _as_bool(_get(train_resume, "enabled", false), "train.resume.enabled"),
        "train.resume.source_run_dir" => _as_string_allow_empty(_get(train_resume, "source_run_dir", ""), "train.resume.source_run_dir"),
        "train.resume.state_path" => _as_string_allow_empty(_get(train_resume, "state_path", ""), "train.resume.state_path"),

        "train.kl_eval.enabled" => _as_bool(_get(kl_eval, "enabled", true), "train.kl_eval.enabled"),
        "train.kl_eval.kl_eval_interval_epochs" => _as_int(_get(kl_eval, "kl_eval_interval_epochs", 10), "train.kl_eval.kl_eval_interval_epochs"),

        "langevin.device" => _as_string(_get(langevin, "device", "GPU:0"), "langevin.device"),
        "langevin.dt" => _as_float(_get(langevin, "dt", 0.004), "langevin.dt"),
        "langevin.resolution" => _as_int(_get(langevin, "resolution", 25), "langevin.resolution"),
        "langevin.nsteps" => _as_int(_get(langevin, "nsteps", 25000), "langevin.nsteps"),
        "langevin.burn_in" => _as_int(_get(langevin, "burn_in", 5000), "langevin.burn_in"),
        "langevin.ensembles" => _as_int(_get(langevin, "ensembles", 256), "langevin.ensembles"),
        "langevin.progress" => _as_bool(_get(langevin, "progress", true), "langevin.progress"),
        "langevin.pdf_bins" => _as_int(_get(langevin, "pdf_bins", 100), "langevin.pdf_bins"),
        "langevin.use_boundary" => _as_bool(_get(langevin, "use_boundary", true), "langevin.use_boundary"),
        "langevin.boundary_min" => _as_float(_get(langevin, "boundary_min", -10.0), "langevin.boundary_min"),
        "langevin.boundary_max" => _as_float(_get(langevin, "boundary_max", 10.0), "langevin.boundary_max"),
        "langevin.min_kept_snapshots_warn" => _as_int(_get(langevin, "min_kept_snapshots_warn", 800), "langevin.min_kept_snapshots_warn"),
        "langevin.eval_profile" => lowercase(_as_string(_get(langevin, "eval_profile", "full"), "langevin.eval_profile")),
        "langevin.quick.dt" => _as_float(_get(langevin_quick, "dt", _get(langevin, "dt", 0.004)), "langevin.quick.dt"),
        "langevin.quick.resolution" => _as_int(_get(langevin_quick, "resolution", _get(langevin, "resolution", 25)), "langevin.quick.resolution"),
        "langevin.quick.nsteps" => _as_int(_get(langevin_quick, "nsteps", _get(langevin, "nsteps", 25000)), "langevin.quick.nsteps"),
        "langevin.quick.burn_in" => _as_int(_get(langevin_quick, "burn_in", _get(langevin, "burn_in", 5000)), "langevin.quick.burn_in"),
        "langevin.quick.ensembles" => _as_int(_get(langevin_quick, "ensembles", _get(langevin, "ensembles", 256)), "langevin.quick.ensembles"),

        "responses.enabled" => _as_bool(_get(responses, "enabled", true), "responses.enabled"),
        "responses.params_path" => _as_string(_get(responses, "params_path", "scripts/L96/parameters_responses.toml"), "responses.params_path"),
        "responses.methods_override" => _as_string_allow_empty(_get(responses, "methods_override", ""), "responses.methods_override"),
        "responses.score_device" => _as_string_allow_empty(_get(responses, "score_device", ""), "responses.score_device"),
        "responses.run_prefix" => _as_string(_get(responses, "run_prefix", "run_"), "responses.run_prefix"),
        "responses.dataset_attr_sync_mode" => _as_string(_get(responses, "dataset_attr_sync_mode", "override"), "responses.dataset_attr_sync_mode"),
        "responses.save_every_override" => _as_int(_get(responses, "save_every_override", 0), "responses.save_every_override"),
        "responses.history_min_alpha" => _as_float(_get(responses, "history_min_alpha", 0.2), "responses.history_min_alpha"),
        "responses.gfdt_nsamples_override" => _as_int(_get(responses, "gfdt_nsamples_override", 0), "responses.gfdt_nsamples_override"),
        "responses.numerical_ensembles_override" => _as_int(_get(responses, "numerical_ensembles_override", 0), "responses.numerical_ensembles_override"),
        "responses.plot_gaussian" => _as_bool(_get(responses, "plot_gaussian", false), "responses.plot_gaussian"),
        "responses.plot_numerical" => _as_bool(_get(responses, "plot_numerical", false), "responses.plot_numerical"),
        "responses.reference.cache_root" => _as_string(_get(responses_reference, "cache_root", "scripts/L96/reference_responses_cache"), "responses.reference.cache_root"),
        "responses.reference.force_regenerate" => _as_bool(_get(responses_reference, "force_regenerate", false), "responses.reference.force_regenerate"),
        "responses.reference.gfdt_nsamples" => _as_int(_get(responses_reference, "gfdt_nsamples", 200_000), "responses.reference.gfdt_nsamples"),
        "responses.reference.gfdt_start_index" => _as_int(_get(responses_reference, "gfdt_start_index", 50_001), "responses.reference.gfdt_start_index"),
        "responses.reference.numerical_ensembles" => _as_int(_get(responses_reference, "numerical_ensembles", 16_384), "responses.reference.numerical_ensembles"),
        "responses.reference.numerical_start_index" => _as_int(_get(responses_reference, "numerical_start_index", 80_001), "responses.reference.numerical_start_index"),
        "responses.reference.numerical_method" => _as_string(_get(responses_reference, "numerical_method", "tangent"), "responses.reference.numerical_method"),
        "responses.reference.h_rel" => _as_float(_get(responses_reference, "h_rel", 5e-3), "responses.reference.h_rel"),
        "responses.reference.h_abs" => _as_float_vec(_get(responses_reference, "h_abs", [1e-2, 1e-3, 1e-2, 1e-2]), "responses.reference.h_abs"),
        "responses.reference.numerical_seed_base" => _as_int(_get(responses_reference, "numerical_seed_base", 1_920_000), "responses.reference.numerical_seed_base"),
        "responses.reference.tmax" => _as_float(_get(responses_reference, "tmax", 2.0), "responses.reference.tmax"),
        "responses.reference.mean_center" => _as_bool(_get(responses_reference, "mean_center", true), "responses.reference.mean_center"),

        "figures.dpi" => _as_int(_get(figures, "dpi", 180), "figures.dpi"),
        "figures.style" => _as_string(_get(figures, "style", "publication"), "figures.style"),

        "output.save_checkpoints" => _as_bool(_get(output, "save_checkpoints", true), "output.save_checkpoints"),
        "output.checkpoint_every" => _as_int(_get(output, "checkpoint_every", 10), "output.checkpoint_every"),
        "output.save_per_epoch_eval" => _as_bool(_get(output, "save_per_epoch_eval", true), "output.save_per_epoch_eval"),
    )

    validate!(params)
    return params
end

function validate!(params::Dict{String,Any})
    params["run.J"] >= 1 || error("run.J must be >= 1")
    params["train.num_training_epochs"] >= 1 || error("train.num_training_epochs must be >= 1")
    params["train.batch_size"] >= 1 || error("train.batch_size must be >= 1")
    params["train.warmup_steps"] >= 0 || error("train.warmup_steps must be >= 0")
    params["train.min_lr_factor"] > 0 || error("train.min_lr_factor must be > 0")
    params["train.min_lr_factor"] <= 1 || error("train.min_lr_factor must be <= 1")
    params["train.norm_groups"] >= 0 || error("train.norm_groups must be >= 0")
    params["train.loss.x_weight"] > 0 || error("train.loss.x_weight must be > 0")
    params["train.loss.y_weight"] > 0 || error("train.loss.y_weight must be > 0")
    params["train.loss.mean_weight"] >= 0 || error("train.loss.mean_weight must be >= 0")
    params["train.loss.cov_weight"] >= 0 || error("train.loss.cov_weight must be >= 0")
    params["train.ema.decay"] > 0 || error("train.ema.decay must be > 0")
    params["train.ema.decay"] < 1 || error("train.ema.decay must be < 1")
    if params["train.resume.enabled"]
        isempty(params["train.resume.source_run_dir"]) && isempty(params["train.resume.state_path"]) &&
            error("train.resume.enabled=true requires train.resume.source_run_dir or train.resume.state_path")
    end
    params["langevin.ensembles"] >= 1 || error("langevin.ensembles must be >= 1")
    params["langevin.nsteps"] > params["langevin.burn_in"] || error("langevin.nsteps must be > langevin.burn_in")
    params["langevin.min_kept_snapshots_warn"] >= 1 || error("langevin.min_kept_snapshots_warn must be >= 1")
    params["langevin.quick.ensembles"] >= 1 || error("langevin.quick.ensembles must be >= 1")
    params["langevin.quick.nsteps"] > params["langevin.quick.burn_in"] || error("langevin.quick.nsteps must be > langevin.quick.burn_in")
    params["langevin.quick.resolution"] >= 1 || error("langevin.quick.resolution must be >= 1")
    params["langevin.quick.dt"] > 0 || error("langevin.quick.dt must be > 0")
    ep = params["langevin.eval_profile"]
    (ep == "full" || ep == "quick") || error("langevin.eval_profile must be 'full' or 'quick'")
    params["data.generation.K"] >= 2 || error("data.generation.K must be >= 2")
    params["data.generation.dt"] > 0 || error("data.generation.dt must be > 0")
    params["data.generation.spinup_steps"] >= 0 || error("data.generation.spinup_steps must be >= 0")
    params["data.generation.save_every"] >= 1 || error("data.generation.save_every must be >= 1")
    params["data.generation.nsamples"] >= 1 || error("data.generation.nsamples must be >= 1")
    params["data.generation.process_noise_sigma"] >= 0 || error("data.generation.process_noise_sigma must be >= 0")
    params["responses.save_every_override"] >= 0 || error("responses.save_every_override must be >= 0")
    params["responses.history_min_alpha"] >= 0 || error("responses.history_min_alpha must be >= 0")
    params["responses.history_min_alpha"] <= 1 || error("responses.history_min_alpha must be <= 1")
    params["responses.gfdt_nsamples_override"] >= 0 || error("responses.gfdt_nsamples_override must be >= 0")
    params["responses.numerical_ensembles_override"] >= 0 || error("responses.numerical_ensembles_override must be >= 0")
    params["responses.reference.gfdt_nsamples"] >= 1 || error("responses.reference.gfdt_nsamples must be >= 1")
    params["responses.reference.gfdt_start_index"] >= 1 || error("responses.reference.gfdt_start_index must be >= 1")
    params["responses.reference.numerical_ensembles"] >= 1 || error("responses.reference.numerical_ensembles must be >= 1")
    params["responses.reference.numerical_start_index"] >= 1 || error("responses.reference.numerical_start_index must be >= 1")
    params["responses.reference.h_rel"] > 0 || error("responses.reference.h_rel must be > 0")
    length(params["responses.reference.h_abs"]) == 4 || error("responses.reference.h_abs must contain exactly 4 entries")
    params["responses.reference.numerical_seed_base"] >= 0 || error("responses.reference.numerical_seed_base must be >= 0")
    params["responses.reference.tmax"] > 0 || error("responses.reference.tmax must be > 0")

    if params["train.kl_eval.enabled"]
        params["train.kl_eval.kl_eval_interval_epochs"] >= 1 || error("train.kl_eval.kl_eval_interval_epochs must be >= 1")
    end

    nt = lowercase(params["train.norm_type"])
    (nt == "batch" || nt == "group") || error("train.norm_type must be 'batch' or 'group'")
    rsm = lowercase(params["responses.dataset_attr_sync_mode"])
    (rsm == "off" || rsm == "warn" || rsm == "error" || rsm == "override") ||
        error("responses.dataset_attr_sync_mode must be one of: off, warn, error, override")
    rnm = lowercase(params["responses.reference.numerical_method"])
    (rnm == "tangent" || rnm == "finite_difference") ||
        error("responses.reference.numerical_method must be 'tangent' or 'finite_difference'")
    ma = params["train.model_arch"]
    (ma == "schneider_dualstream" || ma == "multichannel_unet") ||
        error("train.model_arch must be 'schneider_dualstream' or 'multichannel_unet'")

    return params
end

end # module
