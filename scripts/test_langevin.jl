#!/usr/bin/env julia

using ScoreUNet1D
using ScoreUNet1D: NormalizedDataset, sample_length, num_channels,
    ExecutionDevice, CPUDevice, GPUDevice, select_device

# Load a small dataset
dataset_path = joinpath(@__DIR__, "..", "data", "ks.h5")
dataset = load_hdf5_dataset(dataset_path)

# Tiny subset for quick testing
small_dataset = NormalizedDataset(dataset.data[:, :, 1:100], dataset.stats)

# Create a minimal model
L = sample_length(small_dataset)
C = num_channels(small_dataset)

config = ScoreUNetConfig(
    in_channels=C,
    time_emb_dim=32,
    base_channels=16,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
    attention_levels=(),
    dropout=0.0f0,
    activation=gelu,
    final_activation=identity
)

model = build_unet(config; periodic=true)

# Select device
device = select_device("GPU")

# Move model to device
model = move_model(model, device)

# Langevin config with minimal steps for testing
langevin_cfg = LangevinConfig(
    dt=0.01,
    sample_dt=0.1,
    nsteps=100,  # Very small for testing
    resolution=10,
    n_ensembles=8,  # Small number
    burn_in=0,
    nbins=32,
    sigma=0.05f0,
    seed=42,
    mode=:all,
    boundary=(-10.0, 10.0)
)

@info "Starting Langevin integration test" device=device n_ensembles=langevin_cfg.n_ensembles nsteps=langevin_cfg.nsteps

try
    result = run_langevin(model, small_dataset, langevin_cfg, nothing; device=device)
    @info "Langevin integration succeeded!" kl_divergence=result.kl_divergence
catch err
    @error "Langevin integration failed" exception=(err, catch_backtrace())
    rethrow(err)
end
