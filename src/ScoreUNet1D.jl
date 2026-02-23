module ScoreUNet1D

using Flux
using Functors
using NNlib
using Statistics
using KernelDensity
using Random
using LinearAlgebra
using HDF5
using StatsBase
using ProgressMeter

include("architecture/PeriodicConv.jl")
include("architecture/Blocks.jl")
include("architecture/UNet1D.jl")
include("models/L96SchneiderUNet.jl")
include("data/DataPipeline.jl")
include("Device.jl")
include("training/Trainer.jl")
include("EnsembleIntegrator.jl")
using .EnsembleIntegrator
include("evaluation/LangevinEngine.jl")
include("evaluation/LangevinPlots.jl")
include("evaluation/PhiSigmaEstimator.jl")
using .PhiSigmaEstimator: estimate_phi_sigma
include("runners/RunnerUtils.jl")
using .RunnerUtils
include("runners/TrainRunner.jl")
using .TrainRunner
include("runners/IntegrateRunner.jl")
using .IntegrateRunner
include("runners/PlotRunner.jl")
using .PlotRunner

export ScoreUNetConfig, ScoreUNet, build_unet, PeriodicConv1D,
    L96SchneiderScoreConfig, L96SchneiderScoreModel, L96SchneiderLegacyAdapter,
    build_l96_schneider_model, build_l96_schneider_legacy_model,
    broadcast_X_to_fast, blockmean_fast_to_slow,
    NormalizedDataset, DataStats, load_hdf5_dataset, get_batch,
    ScoreTrainerConfig, TrainingHistory, TrainingState, train!, score_from_model,
    CorrelationConfig, CorrelationInfo, compute_correlation_info,
    average_mode_acf,
    ExecutionDevice, CPUDevice, GPUDevice, select_device, move_model, move_array, is_gpu,
    gpu_count, activate_device!,
    LangevinConfig, LangevinResult, run_langevin, compute_stein_matrix, compare_pdfs, relative_entropy,
    evolve_sde, evolve_sde_snapshots, SnapshotIntegrator, build_snapshot_integrator, ScoreWrapper,
    plot_langevin_vs_observed, estimate_phi_sigma,
    load_config, resolve_path, save_model, load_model, load_phi_sigma, save_phi_sigma,
    activation_from_string, symbol_from_string, ensure_dir, create_run_directory,
    timed, verbose_log,
    # Runner wrappers
    TrainingResult, train_score_network,
    IntegrationResult, integrate_langevin,
    FigurePaths, plot_publication_figure, plot_comparison_figure

end # module
