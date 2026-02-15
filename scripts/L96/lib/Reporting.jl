module L96Reporting

using Dates
using TOML

function write_run_summary(run_dir::AbstractString,
                           params::Dict{String,Any},
                           params_used_path::AbstractString,
                           training_info::Dict{String,Any},
                           eval_rows::Vector{Dict{String,Any}},
                           figA_path::AbstractString)
    best_idx = 0
    best_kl = Inf
    for (i, row) in enumerate(eval_rows)
        kl = Float64(row["avg_mode_kl_clipped"])
        if isfinite(kl) && kl < best_kl
            best_kl = kl
            best_idx = i
        end
    end

    best_eval = best_idx > 0 ? eval_rows[best_idx] : Dict{String,Any}()

    summary = Dict{String,Any}(
        "updated_at" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "run_dir" => abspath(run_dir),
        "parameters_used_path" => abspath(params_used_path),
        "training" => Dict{String,Any}(
            "model_path" => abspath(training_info["model_path"]),
            "figure_A" => abspath(figA_path),
        ),
        "evaluation" => Dict{String,Any}(
            "num_langevin_evaluations" => length(eval_rows),
            "epochs" => [Int(r["epoch"]) for r in eval_rows],
            "avg_mode_kl_clipped" => [Float64(r["avg_mode_kl_clipped"]) for r in eval_rows],
            "global_kl" => [Float64(r["global_kl"]) for r in eval_rows],
            "best_epoch" => best_idx > 0 ? Int(best_eval["epoch"]) : -1,
            "best_avg_mode_kl_clipped" => best_idx > 0 ? Float64(best_eval["avg_mode_kl_clipped"]) : NaN,
            "best_metrics_toml" => best_idx > 0 ? abspath(String(best_eval["metrics_toml"])) : "",
        ),
        "raw_parameters" => params["_raw"],
    )

    out_path = joinpath(run_dir, "metrics", "run_summary.toml")
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        TOML.print(io, summary)
    end
    return out_path
end

end # module
