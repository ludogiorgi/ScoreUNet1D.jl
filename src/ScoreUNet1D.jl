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
include("data/DataPipeline.jl")
include("Device.jl")
include("training/Trainer.jl")
include("EnsembleIntegrator.jl")
using .EnsembleIntegrator
include("evaluation/LangevinEngine.jl")
include("evaluation/LangevinPlots.jl")
include("evaluation/PhiSigmaEstimator.jl")
using .PhiSigmaEstimator: estimate_phi_sigma

export ScoreUNetConfig, ScoreUNet, build_unet, PeriodicConv1D,
       NormalizedDataset, DataStats, load_hdf5_dataset, get_batch,
       ScoreTrainerConfig, TrainingHistory, train!, score_from_model,
       CorrelationConfig, CorrelationInfo, compute_correlation_info,
       average_mode_acf,
       ExecutionDevice, CPUDevice, GPUDevice, select_device, move_model, move_array, is_gpu,
       gpu_count, activate_device!,
       LangevinConfig, LangevinResult, run_langevin, compute_stein_matrix, compare_pdfs, relative_entropy,
       evolve_sde, evolve_sde_snapshots, SnapshotIntegrator, build_snapshot_integrator, ScoreWrapper,
       plot_langevin_vs_observed, estimate_phi_sigma

end # module
