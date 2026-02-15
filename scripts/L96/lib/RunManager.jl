module L96RunManager

using Dates
using TOML

function _ensure(path::AbstractString)
    mkpath(path)
    return path
end

function _parse_run_num(name::AbstractString)
    m = match(r"^run_(\d+)$", name)
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function next_run_dir(params::Dict{String,Any}; base_dir::AbstractString=pwd())
    root = abspath(joinpath(base_dir, params["run.runs_root"]))
    group_name = "$(params["run.run_folder_prefix"])$(params["run.J"])"
    group_dir = _ensure(joinpath(root, group_name))

    max_id = 0
    for name in readdir(group_dir)
        n = _parse_run_num(name)
        n === nothing && continue
        max_id = max(max_id, n)
    end

    run_n = max_id + 1
    pad = params["run.run_id_padding"]
    run_id = "run_" * lpad(string(run_n), pad, '0')
    run_dir = joinpath(group_dir, run_id)

    return (root=root, group_dir=group_dir, run_dir=run_dir, run_id=run_id, run_n=run_n)
end

function create_run_scaffold(run_dir::AbstractString)
    dirs = Dict(
        "run" => run_dir,
        "model" => joinpath(run_dir, "model"),
        "metrics" => joinpath(run_dir, "metrics"),
        "figures" => joinpath(run_dir, "figures"),
        "figures_training" => joinpath(run_dir, "figures", "training"),
        "logs" => joinpath(run_dir, "logs"),
    )
    for d in values(dirs)
        mkpath(d)
    end
    return dirs
end

function copy_parameters_file(src::AbstractString, dst_run_dir::AbstractString)
    dst = joinpath(dst_run_dir, "parameters_used.toml")
    cp(src, dst; force=true)
    return dst
end

function write_summary_toml(path::AbstractString, cfg::Dict{String,Any}, data::Dict{String,Any})
    out = Dict{String,Any}(
        "updated_at" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "parameters" => cfg,
        "summary" => data,
    )
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, out)
    end
    return path
end

end # module
