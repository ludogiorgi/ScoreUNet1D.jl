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
    checkpoint_models = String[]
    model_dir = joinpath(run_dir, "model")
    if isdir(model_dir)
        for name in sort(readdir(model_dir))
            lname = lowercase(name)
            endswith(lname, ".bson") || continue
            startswith(lname, "training_state") && continue
            push!(checkpoint_models, abspath(joinpath(model_dir, name)))
        end
    end

    eval_details = Dict{String,Any}[]
    for row in eval_rows
        epoch = Int(row["epoch"])
        fig_dir = String(row["fig_dir"])
        figD_raw = haskey(row, "figure_D") ? String(row["figure_D"]) : ""
        figD = isempty(strip(figD_raw)) ? "" : abspath(figD_raw)
        push!(eval_details, Dict{String,Any}(
            "epoch" => epoch,
            "langevin_profile" => get(row, "langevin_profile", "full"),
            "avg_mode_kl_clipped" => Float64(row["avg_mode_kl_clipped"]),
            "global_kl" => Float64(row["global_kl"]),
            "metrics_toml" => abspath(String(row["metrics_toml"])),
            "figure_B" => abspath(joinpath(fig_dir, "figB_stats_3x3.png")),
            "figure_C" => abspath(joinpath(fig_dir, "figC_dynamics_3x2.png")),
            "figure_D" => figD,
            "langevin_seed" => Int(params["run.seed"]) + epoch,
        ))
    end

    summary = Dict{String,Any}(
        "updated_at" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "run_dir" => abspath(run_dir),
        "parameters_used_path" => abspath(params_used_path),
        "reproducibility" => Dict{String,Any}(
            "seed_chain" => Dict{String,Any}(
                "run_seed" => Int(params["run.seed"]),
                "data_generation_seed" => Int(params["data.generation.rng_seed"]),
                "training_seed" => Int(params["run.seed"]),
                "training_gpu_noise_seed_rule" => "train_seed + global_step",
                "langevin_eval_seed_rule" => "run_seed + epoch",
                "langevin_eval_seeds" => [Int(params["run.seed"]) + Int(r["epoch"]) for r in eval_rows],
            ),
            "optimizer" => Dict{String,Any}(
                "batch_size" => Int(params["train.batch_size"]),
                "epochs" => Int(params["train.num_training_epochs"]),
                "learning_rate" => Float64(params["train.lr"]),
                "use_lr_schedule" => Bool(params["train.use_lr_schedule"]),
                "warmup_steps" => Int(params["train.warmup_steps"]),
                "min_lr_factor" => Float64(params["train.min_lr_factor"]),
                "sigma" => Float64(params["train.sigma"]),
                "loss_x_weight" => Float64(params["train.loss.x_weight"]),
                "loss_y_weight" => Float64(params["train.loss.y_weight"]),
                "loss_mean_weight" => Float64(params["train.loss.mean_weight"]),
                "loss_cov_weight" => Float64(params["train.loss.cov_weight"]),
            ),
            "model" => Dict{String,Any}(
                "architecture" => String(params["train.model_arch"]),
                "base_channels" => Int(params["train.base_channels"]),
                "channel_multipliers" => params["train.channel_multipliers"],
                "norm_type" => String(params["train.norm_type"]),
                "norm_groups" => Int(params["train.norm_groups"]),
                "normalization_mode" => String(params["data.normalization_mode"]),
            ),
            "ema" => Dict{String,Any}(
                "enabled" => Bool(params["train.ema.enabled"]),
                "decay" => Float64(params["train.ema.decay"]),
                "use_for_eval" => Bool(params["train.ema.use_for_eval"]),
            ),
            "langevin" => Dict{String,Any}(
                "device" => String(params["langevin.device"]),
                "eval_profile" => String(params["langevin.eval_profile"]),
                "dt_full" => Float64(params["langevin.dt"]),
                "resolution_full" => Int(params["langevin.resolution"]),
                "nsteps_full" => Int(params["langevin.nsteps"]),
                "burn_in_full" => Int(params["langevin.burn_in"]),
                "ensembles_full" => Int(params["langevin.ensembles"]),
                "dt_quick" => Float64(params["langevin.quick.dt"]),
                "resolution_quick" => Int(params["langevin.quick.resolution"]),
                "nsteps_quick" => Int(params["langevin.quick.nsteps"]),
                "burn_in_quick" => Int(params["langevin.quick.burn_in"]),
                "ensembles_quick" => Int(params["langevin.quick.ensembles"]),
                "min_kept_snapshots_warn" => Int(params["langevin.min_kept_snapshots_warn"]),
            ),
            "data_generation" => Dict{String,Any}(
                "K" => Int(params["data.generation.K"]),
                "J" => Int(params["run.J"]),
                "F" => Float64(params["data.generation.F"]),
                "h" => Float64(params["data.generation.h"]),
                "c" => Float64(params["data.generation.c"]),
                "b" => Float64(params["data.generation.b"]),
                "dt" => Float64(params["data.generation.dt"]),
                "spinup_steps" => Int(params["data.generation.spinup_steps"]),
                "save_every" => Int(params["data.generation.save_every"]),
                "nsamples" => Int(params["data.generation.nsamples"]),
                "process_noise_sigma" => Float64(params["data.generation.process_noise_sigma"]),
            ),
        ),
        "training" => Dict{String,Any}(
            "model_path" => abspath(training_info["model_path"]),
            "training_state_path" => haskey(training_info, "train_state_path") ? abspath(String(training_info["train_state_path"])) : "",
            "figure_A" => abspath(figA_path),
            "checkpoint_models" => checkpoint_models,
        ),
        "evaluation" => Dict{String,Any}(
            "num_langevin_evaluations" => length(eval_rows),
            "epochs" => [Int(r["epoch"]) for r in eval_rows],
            "avg_mode_kl_clipped" => [Float64(r["avg_mode_kl_clipped"]) for r in eval_rows],
            "global_kl" => [Float64(r["global_kl"]) for r in eval_rows],
            "details" => eval_details,
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
