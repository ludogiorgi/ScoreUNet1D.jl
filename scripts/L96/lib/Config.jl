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

function load_parameters(path::AbstractString)
    isfile(path) || error("Parameters file not found: $path")
    cfg = TOML.parsefile(path)

    run = _as_dict(_require(cfg, "run"), "run")
    data = _as_dict(_require(cfg, "data"), "data")
    train = _as_dict(_require(cfg, "train"), "train")
    kl_eval = _as_dict(_require(train, "kl_eval"), "train.kl_eval")
    langevin = _as_dict(_require(cfg, "langevin"), "langevin")
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
        "data.dataset_key" => _as_string(_get(data, "dataset_key", "timeseries"), "data.dataset_key"),
        "data.normalization_mode" => _as_string(_get(data, "normalization_mode", "split_xy"), "data.normalization_mode"),
        "data.generate_if_missing" => _as_bool(_get(data, "generate_if_missing", false), "data.generate_if_missing"),

        "train.device" => _as_string(_get(train, "device", "GPU:0"), "train.device"),
        "train.num_training_epochs" => _as_int(_get(train, "num_training_epochs", 80), "train.num_training_epochs"),
        "train.batch_size" => _as_int(_get(train, "batch_size", 256), "train.batch_size"),
        "train.lr" => _as_float(_get(train, "lr", 8e-4), "train.lr"),
        "train.sigma" => _as_float(_get(train, "sigma", 0.05), "train.sigma"),
        "train.base_channels" => _as_int(_get(train, "base_channels", 32), "train.base_channels"),
        "train.channel_multipliers" => _as_int_vec(_get(train, "channel_multipliers", [1, 2, 4]), "train.channel_multipliers"),
        "train.progress" => _as_bool(_get(train, "progress", false), "train.progress"),

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
    params["langevin.ensembles"] >= 1 || error("langevin.ensembles must be >= 1")
    params["langevin.nsteps"] > params["langevin.burn_in"] || error("langevin.nsteps must be > langevin.burn_in")

    if params["train.kl_eval.enabled"]
        params["train.kl_eval.kl_eval_interval_epochs"] >= 1 || error("train.kl_eval.kl_eval_interval_epochs must be >= 1")
    end

    return params
end

end # module
