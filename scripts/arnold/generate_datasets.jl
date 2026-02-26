# Standard command (from repository root):
# julia --project=. scripts/arnold/generate_datasets.jl --params scripts/arnold/parameters_data.toml
# Nohup command:
# nohup julia --project=. scripts/arnold/generate_datasets.jl --params scripts/arnold/parameters_data.toml > scripts/arnold/nohup_generate_datasets.log 2>&1 &

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using TOML

include(joinpath(@__DIR__, "lib", "ArnoldCommon.jl"))
using .ArnoldCommon
include(joinpath(@__DIR__, "lib", "ArnoldStatsPlots.jl"))
using .ArnoldStatsPlots

function parse_args(args::Vector{String})
    params_path = joinpath(@__DIR__, "parameters_data.toml")
    roles_raw = "all"

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--params"
            i == length(args) && error("--params expects a value")
            params_path = args[i + 1]
            i += 2
        elseif a == "--roles"
            i == length(args) && error("--roles expects comma-separated values or 'all'")
            roles_raw = args[i + 1]
            i += 2
        else
            error("Unknown argument: $a")
        end
    end

    return (params_path=abspath(params_path), roles_raw=roles_raw)
end

function parse_roles(raw::AbstractString)
    s = lowercase(strip(String(raw)))
    if s == "all"
        return String[ArnoldCommon.ARNOLD_DATASET_ROLES...]
    end
    roles = String[]
    for token in split(s, ",")
        r = strip(token)
        isempty(r) && continue
        push!(roles, r)
    end
    isempty(roles) && error("No valid dataset roles parsed from --roles=$raw")
    return roles
end

function _as_table(doc::Dict{String,Any}, key::String)
    if haskey(doc, key) && (doc[key] isa AbstractDict)
        return Dict{String,Any}(doc[key])
    end
    return Dict{String,Any}()
end

function _role_available(cfg::Dict{String,Any}, result::Dict{String,Dict{String,Any}}, role::String)
    if haskey(result, role)
        return true
    end
    key = cfg["datasets.$role.key"]
    sig = read_dataset_signature(cfg["paths.datasets_hdf5"], key)
    return !isempty(sig)
end

function _pick_stochastic_role(cfg::Dict{String,Any}, result::Dict{String,Dict{String,Any}})
    if _role_available(cfg, result, "train_stochastic")
        return "train_stochastic"
    end
    if _role_available(cfg, result, "gfdt_stochastic")
        return "gfdt_stochastic"
    end
    return ""
end

function _matrix_to_tensor(x::AbstractMatrix{<:Real})
    return reshape(Float32.(x), size(x, 1), 1, size(x, 2))
end

function build_dataset_compare_figure(cfg::Dict{String,Any},
    result::Dict{String,Dict{String,Any}},
    summary_path::AbstractString,
    doc::Dict{String,Any})
    deterministic_role = "two_scale_observed"
    stochastic_role = _pick_stochastic_role(cfg, result)
    if isempty(stochastic_role) || !_role_available(cfg, result, deterministic_role)
        return Dict{String,Any}(
            "path" => "",
            "deterministic_role" => deterministic_role,
            "stochastic_role" => stochastic_role,
            "generated" => false,
        )
    end

    figures = _as_table(doc, "figures")
    bins = Int(get(figures, "pdf_bins", 80))
    max_acf_lag = Int(get(figures, "max_acf_lag", 200))
    bins >= 10 || error("figures.pdf_bins must be >= 10")
    max_acf_lag >= 1 || error("figures.max_acf_lag must be >= 1")

    x_det = load_role_x_matrix(cfg, deterministic_role; label="dataset_compare_$deterministic_role")
    x_sto = load_role_x_matrix(cfg, stochastic_role; label="dataset_compare_$stochastic_role")
    obs = _matrix_to_tensor(x_det)
    gen = _matrix_to_tensor(x_sto)

    kl_mode, js_mode = modewise_metrics(
        obs,
        gen;
        nbins=bins,
        low_q=0.001,
        high_q=0.999,
    )

    fig_path = joinpath(dirname(summary_path), "figB_stats_dataset_compare_4x2.png")
    save_stats_figure_acf(
        fig_path,
        obs,
        gen,
        kl_mode,
        js_mode,
        bins;
        max_lag=max_acf_lag,
        obs_label=deterministic_role,
        gen_label=stochastic_role,
    )

    return Dict{String,Any}(
        "path" => abspath(fig_path),
        "deterministic_role" => deterministic_role,
        "stochastic_role" => stochastic_role,
        "generated" => true,
        "pdf_bins" => bins,
        "max_acf_lag" => max_acf_lag,
    )
end

function main(args=ARGS)
    parsed = parse_args(args)
    roles = parse_roles(parsed.roles_raw)

    cfg, doc = load_data_config(parsed.params_path)
    result = ensure_arnold_datasets!(cfg; roles=roles)
    theta, meta = resolve_closure_theta(cfg)

    summary_path = joinpath(dirname(cfg["paths.datasets_hdf5"]), "datasets_generation_summary.toml")
    fig_info = build_dataset_compare_figure(cfg, result, summary_path, doc)

    out = Dict{String,Any}(
        "parameters_data_path" => parsed.params_path,
        "datasets_hdf5" => cfg["paths.datasets_hdf5"],
        "roles_requested" => roles,
        "closure_used" => Dict(
            "alpha0" => theta[1],
            "alpha1" => theta[2],
            "alpha2" => theta[3],
            "alpha3" => theta[4],
            "sigma" => theta[5],
        ),
        "closure_meta" => meta,
        "datasets" => result,
        "figures" => Dict(
            "figB_stats_dataset_compare_4x2" => get(fig_info, "path", ""),
            "deterministic_role" => get(fig_info, "deterministic_role", ""),
            "stochastic_role" => get(fig_info, "stochastic_role", ""),
            "max_acf_lag" => get(fig_info, "max_acf_lag", 200),
            "pdf_bins" => get(fig_info, "pdf_bins", 80),
        ),
    )

    open(summary_path, "w") do io
        TOML.print(io, out)
    end

    println("datasets_hdf5=$(cfg["paths.datasets_hdf5"])")
    println("summary=$(abspath(summary_path))")
    println("figB_dataset_compare=$(get(fig_info, "path", ""))")
    for role in roles
        info = result[role]
        println("role=$(role) generated=$(info["generated"]) key=$(info["key"])")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
