# ==============================================================================
# Run Pipeline KS: Iterative Train-Integrate-Plot Pipeline
# ==============================================================================
#
# This script:
#   1. Trains score network (continuing from previous if exists)
#   2. Creates timestamped run folder
#   3. Saves NN and loss plot to run folder and scripts/KS/
#   4. Integrates Langevin dynamics
#   5. Computes and prints relative entropy
#   6. Generates comparison figure
#   7. Saves all TOML configs to run folder
#   8. Repeats for n_iterations
#   9. Plots all relative entropy values
#
# Usage:
#   nohup julia --project=. scripts/KS/run_pipeline_ks.jl > scripts/KS/pipeline.log 2>&1 &
#
# ==============================================================================

using BSON
using CairoMakie
using Dates
using Flux
using HDF5
using Statistics
using TOML

using ScoreUNet1D
using ScoreUNet1D: NormalizedDataset, load_hdf5_dataset

CairoMakie.activate!()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, "..", ".."))
const PIPELINE_CONFIG = joinpath(SCRIPT_DIR, "pipeline_params.toml")

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

function compute_relative_entropy(pdf_data::Vector{Float64}, pdf_pred::Vector{Float64};
    eps::Float64=1e-10)
    # Normalize PDFs
    p = pdf_data ./ sum(pdf_data)
    q = pdf_pred ./ sum(pdf_pred)
    # Clip for numerical stability
    p = max.(p, eps)
    q = max.(q, eps)
    # KL divergence
    return sum(p .* log.(p ./ q))
end

function create_run_folder(run_root::String)
    mkpath(run_root)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    slug = Random.randstring(4)
    run_dir = joinpath(run_root, "run_$(timestamp)_$(slug)")
    mkpath(run_dir)
    return run_dir
end

function plot_losses(epoch_losses::Vector{<:Real}, output_path::String)
    fig = Figure(size=(800, 500), font="TeX Gyre Heros")
    ax = Axis(fig[1, 1];
        title="Training Loss",
        xlabel="Epoch",
        ylabel="Loss",
        yscale=log10)
    lines!(ax, 1:length(epoch_losses), epoch_losses;
        color=:firebrick, linewidth=2.5)
    scatter!(ax, 1:length(epoch_losses), epoch_losses;
        color=:firebrick, markersize=8)
    save(output_path, fig; px_per_unit=1)
    return output_path
end

function plot_entropy_history(entropies::Vector{Float64}, output_path::String)
    fig = Figure(size=(800, 500), font="TeX Gyre Heros")
    ax = Axis(fig[1, 1];
        title="Relative Entropy Over Iterations",
        xlabel="Iteration",
        ylabel="KL Divergence (Data || Langevin)")
    lines!(ax, 1:length(entropies), entropies;
        color=:navy, linewidth=2.5)
    scatter!(ax, 1:length(entropies), entropies;
        color=:navy, markersize=12)
    save(output_path, fig; px_per_unit=1)
    return output_path
end

function copy_toml_files(run_dir::String, toml_paths::Vector{String})
    for path in toml_paths
        if isfile(path)
            cp(path, joinpath(run_dir, basename(path)); force=true)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

using Random

@info "Loading pipeline configuration" config = PIPELINE_CONFIG
pipeline_cfg = TOML.parsefile(PIPELINE_CONFIG)
pipeline = pipeline_cfg["pipeline"]

n_iterations = Int(get(pipeline, "n_iterations", 3))
run_root = joinpath(PROJECT_ROOT, get(pipeline, "run_root", "runs/KS"))
model_path_scripts = joinpath(PROJECT_ROOT, get(pipeline, "model_path", "scripts/KS/trained_model.bson"))

train_config = joinpath(SCRIPT_DIR, get(pipeline, "train_config", "train_params.toml"))
integrate_config = joinpath(SCRIPT_DIR, get(pipeline, "integrate_config", "integrate_params.toml"))
plot_config = joinpath(SCRIPT_DIR, get(pipeline, "plot_config", "plot_params.toml"))

@info "Pipeline settings" n_iterations = n_iterations run_root = run_root

# Create main run folder for this pipeline execution
run_dir = create_run_folder(run_root)
@info "Created run folder" path = run_dir

# Track entropy across iterations
entropy_history = Float64[]
all_losses = Vector{Float64}[]

# Load existing model if available
current_model = nothing
if isfile(model_path_scripts)
    @info "Loading existing model for continuation" path = model_path_scripts
    contents = BSON.load(model_path_scripts)
    current_model = Flux.cpu(contents[:model])
    Flux.testmode!(current_model, false)  # Enable training mode
end

for iter in 1:n_iterations
    @info "="^60
    @info "ITERATION $iter / $n_iterations"
    @info "="^60

    iter_dir = joinpath(run_dir, "iter_$(lpad(iter, 2, '0'))")
    mkpath(iter_dir)

    # ─────────────────────────────────────────────────────────────────────
    # 1. TRAIN
    # ─────────────────────────────────────────────────────────────────────
    @info "Training score network..."

    train_result = train_score_network(train_config;
        model=current_model,
        project_root=PROJECT_ROOT)

    global current_model = train_result.model
    push!(all_losses, train_result.history.epoch_losses)

    @info "Training complete" final_loss = train_result.history.epoch_losses[end]

    # Save model to iteration folder
    model_iter_path = joinpath(iter_dir, "model.bson")
    BSON.@save model_iter_path model = current_model cfg = train_result.model_config trainer_cfg = train_result.trainer_config

    # Save/update model in scripts/KS (for next iteration)
    BSON.@save model_path_scripts model = current_model cfg = train_result.model_config trainer_cfg = train_result.trainer_config
    @info "Model saved" iter_path = model_iter_path scripts_path = model_path_scripts

    # Plot losses
    loss_fig_path = joinpath(iter_dir, "training_loss.png")
    plot_losses(train_result.history.epoch_losses, loss_fig_path)
    @info "Loss plot saved" path = loss_fig_path

    # ─────────────────────────────────────────────────────────────────────
    # 2. INTEGRATE
    # ─────────────────────────────────────────────────────────────────────
    @info "Integrating Langevin dynamics..."

    integrate_result = integrate_langevin(integrate_config; project_root=PROJECT_ROOT)

    @info "Integration complete" mode = integrate_result.phi_sigma_mode samples = size(integrate_result.trajectory)

    # ─────────────────────────────────────────────────────────────────────
    # 3. COMPUTE RELATIVE ENTROPY
    # ─────────────────────────────────────────────────────────────────────
    stats = integrate_result.statistics
    pdf_data = Float64.(stats[:pdf_data])
    pdf_langevin = Float64.(stats[:pdf_langevin])

    rel_entropy = compute_relative_entropy(pdf_data, pdf_langevin)
    push!(entropy_history, rel_entropy)

    @info "Relative entropy (KL divergence)" iteration = iter kl = rel_entropy

    # ─────────────────────────────────────────────────────────────────────
    # 4. GENERATE FIGURE
    # ─────────────────────────────────────────────────────────────────────
    @info "Generating comparison figure..."

    fig_path = joinpath(iter_dir, "comparison.png")
    plot_comparison_figure(stats, fig_path)
    @info "Comparison figure saved" path = fig_path

    # ─────────────────────────────────────────────────────────────────────
    # 5. COPY TOML FILES
    # ─────────────────────────────────────────────────────────────────────
    toml_files = [train_config, integrate_config, plot_config, PIPELINE_CONFIG]
    copy_toml_files(iter_dir, toml_files)
    @info "TOML files copied to iteration folder"

    # Save iteration summary
    summary_path = joinpath(iter_dir, "summary.toml")
    open(summary_path, "w") do io
        println(io, "# Iteration $iter Summary")
        println(io, "iteration = $iter")
        println(io, "final_loss = $(train_result.history.epoch_losses[end])")
        println(io, "relative_entropy = $rel_entropy")
        println(io, "phi_sigma_mode = \"$(integrate_result.phi_sigma_mode)\"")
        println(io, "trajectory_samples = $(prod(size(integrate_result.trajectory)[3:4]))")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Final: Plot Entropy History
# ─────────────────────────────────────────────────────────────────────────────

@info "="^60
@info "PIPELINE COMPLETE"
@info "="^60

entropy_fig_path = joinpath(run_dir, "entropy_history.png")
plot_entropy_history(entropy_history, entropy_fig_path)
@info "Entropy history plot saved" path = entropy_fig_path

# Save final summary
final_summary_path = joinpath(run_dir, "pipeline_summary.toml")
open(final_summary_path, "w") do io
    println(io, "# Pipeline Summary")
    println(io, "n_iterations = $n_iterations")
    println(io, "entropy_values = $(entropy_history)")
    println(io, "final_entropy = $(entropy_history[end])")
    println(io, "entropy_improvement = $(entropy_history[1] - entropy_history[end])")
end

# Copy all TOML files to main run folder
copy_toml_files(run_dir, [train_config, integrate_config, plot_config, PIPELINE_CONFIG])

@info "Results summary:"
for (i, ent) in enumerate(entropy_history)
    @info "  Iteration $i: KL = $(round(ent, digits=6))"
end
@info "Run folder: $run_dir"
