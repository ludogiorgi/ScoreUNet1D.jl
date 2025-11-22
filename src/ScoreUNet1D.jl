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
using FastSDE

include("architecture/PeriodicConv.jl")
include("architecture/Blocks.jl")
include("architecture/UNet1D.jl")
include("data/DataPipeline.jl")
include("Device.jl")
include("training/Trainer.jl")
include("evaluation/Langevin.jl")

export ScoreUNetConfig, ScoreUNet, build_unet, PeriodicConv1D,
       NormalizedDataset, DataStats, load_hdf5_dataset, get_batch,
       ScoreTrainerConfig, TrainingHistory, train!, score_from_model,
       CorrelationConfig, CorrelationInfo, compute_correlation_info,
       average_mode_acf,
       ExecutionDevice, CPUDevice, GPUDevice, select_device, move_model, move_array, is_gpu,
       gpu_count, activate_device!,
        LangevinConfig, LangevinResult, run_langevin, compute_stein_matrix, compare_pdfs, relative_entropy

end # module
