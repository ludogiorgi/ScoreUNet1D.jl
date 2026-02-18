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

    cfg_multi = ScoreUNetConfig(in_channels=3, out_channels=5, periodic=false,
                                base_channels=8, channel_multipliers=[1, 2])
    model_multi = build_unet(cfg_multi)
    x_multi = rand(Float32, 64, 3, 4)
    y_multi = model_multi(x_multi)
    @test size(y_multi) == (64, 5, 4)

    trainer_cfg = ScoreTrainerConfig(epochs=1, batch_size=4, lr=1e-3,
                                     sigma=0.05f0, progress=false,
                                     max_steps_per_epoch=2)
    history = train!(model, dataset, trainer_cfg)
    @test length(history.epoch_losses) == 1
    @test !isempty(history.batch_losses)

    langevin_cfg = LangevinConfig(dt=1e-3, nsteps=4000, resolution=200,
                                  n_ensembles=2, burn_in=400, nbins=16,
                                  sigma=trainer_cfg.sigma, mode=1)
    result = run_langevin(model, dataset, langevin_cfg)
    @test result.kl_divergence ≥ 0
    @test length(result.bin_centers) == 16
end

include("l96_schneider_tests.jl")
