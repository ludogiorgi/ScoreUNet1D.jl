# Standard command (from repository root):
# julia --project=. scripts/arnold/generate_datasets.jl --params scripts/arnold/parameters_data.toml
# Nohup command:
# nohup julia --project=. scripts/arnold/generate_datasets.jl --params scripts/arnold/parameters_data.toml > scripts/arnold/nohup_generate_datasets.log 2>&1 &

using TOML

include(joinpath(@__DIR__, "lib", "ArnoldCommon.jl"))
using .ArnoldCommon

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

function main(args=ARGS)
    parsed = parse_args(args)
    roles = parse_roles(parsed.roles_raw)

    cfg, doc = load_data_config(parsed.params_path)
    result = ensure_arnold_datasets!(cfg; roles=roles)
    theta, meta = resolve_closure_theta(cfg)

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
    )

    summary_path = joinpath(dirname(cfg["paths.datasets_hdf5"]), "datasets_generation_summary.toml")
    open(summary_path, "w") do io
        TOML.print(io, out)
    end

    println("datasets_hdf5=$(cfg["paths.datasets_hdf5"])")
    println("summary=$(abspath(summary_path))")
    for role in roles
        info = result[role]
        println("role=$(role) generated=$(info["generated"]) key=$(info["key"])")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
