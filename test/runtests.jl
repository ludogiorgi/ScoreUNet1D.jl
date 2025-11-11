using Test
using Random
using HDF5
using ScoreUNet1D

function toy_dataset(nsamples::Int, length::Int)
    data = Array{Float32}(undef, nsamples, length)
    for i in 1:nsamples
        phase = 2f0 * π * rand()
        for j in 1:length
            data[i, j] = sin(2f0 * π * j / length + phase) + 0.1f0 * randn(Float32)
        end
    end
    return data
end

@testset "ScoreUNet1D pipeline" begin
    tmpdir = mktempdir()
    file = joinpath(tmpdir, "toy.h5")
    data = toy_dataset(32, 64)
    h5open(file, "w") do io
        write(io, "train", data)
    end

    dataset = load_hdf5_dataset(file; dataset_key="train")
    @test size(dataset.data) == (64, 1, 32)

    cfg = ScoreUNetConfig(in_channels=1, base_channels=8, channel_multipliers=[1, 2])
    model = build_unet(cfg)
    batch = dataset.data[:, :, 1:2]
    preds = model(batch)
    @test size(preds) == size(batch)

    trainer_cfg = ScoreTrainerConfig(epochs=1, batch_size=4, lr=1e-3,
                                     sigma=0.05f0, progress=false,
                                     max_steps_per_epoch=2)
    history = train!(model, dataset, trainer_cfg)
    @test length(history.epoch_losses) == 1
    @test !isempty(history.batch_losses)

    langevin_cfg = LangevinConfig(dt=1e-2, nsteps=400, resolution=20,
                                  n_ensembles=2, burn_in=40, nbins=16,
                                  sigma=trainer_cfg.sigma, mode=1)
    result = run_langevin(model, dataset, langevin_cfg)
    @test result.kl_divergence ≥ 0
    @test length(result.bin_centers) == 16
end
