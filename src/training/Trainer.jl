using Flux
using Flux.Losses: mse
using Flux.Optimisers
using Random
using ProgressMeter
using Base.Threads
import CUDA

Base.@kwdef mutable struct ScoreTrainerConfig
    batch_size::Int = 32
    epochs::Int = 50
    lr::Float64 = 1e-3
    sigma::Float32 = 0.05f0
    shuffle::Bool = true
    seed::Int = 42
    progress::Bool = true
    max_steps_per_epoch::Union{Nothing,Int} = nothing
end

struct TrainingHistory
    epoch_losses::Vector{Float32}
    batch_losses::Vector{Float32}
end

"""
    train!(model, dataset, cfg; callback=nothing)

Runs denoising score matching with σ = cfg.sigma.
"""
function train!(model, dataset::NormalizedDataset, cfg::ScoreTrainerConfig;
                callback::Function = (_, _) -> nothing,
                epoch_callback::Function = (_, _, _) -> nothing,
                device::ExecutionDevice=CPUDevice())
    n = length(dataset)
    n == 0 && error("Dataset is empty")
    rng = MersenneTwister(cfg.seed)
    thread_rngs = seed_thread_rngs(cfg.seed)
    model_on_device = model
    opt_state = Flux.setup(Flux.Optimisers.Adam(cfg.lr), model_on_device)
    epoch_losses = Float32[]
    batch_losses = Float32[]
    steps_per_epoch = ceil(Int, n / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    progress = cfg.progress ? Progress(total_steps; desc="Training score network") : nothing

    for epoch in 1:cfg.epochs
        epoch_t0 = time_ns()
        idxs = collect(1:n)
        cfg.shuffle && Random.shuffle!(rng, idxs)
        batches = Iterators.partition(idxs, cfg.batch_size)
        step = 0
        accum = 0.0
        for batch_idxs in batches
            batch_cpu = Array(dataset.data[:, :, batch_idxs])
            batch = move_array(batch_cpu, device)
            noise = noise_like(batch, device, thread_rngs)
            noisy = batch .+ cfg.sigma .* noise
            loss, grads = Flux.withgradient(model_on_device) do m
                pred = m(noisy)
                mse(pred, noise)
            end
            opt_state, model_on_device = Flux.update!(opt_state, model_on_device, grads[1])
            loss32 = Float32(loss)
            push!(batch_losses, loss32)
            accum += loss32
            step += 1
            progress !== nothing && ProgressMeter.next!(progress; showvalues=[(:epoch, epoch), (:loss, loss32)])
            callback(loss32, epoch)
            if cfg.max_steps_per_epoch !== nothing && step >= cfg.max_steps_per_epoch
                break
            end
        end
        push!(epoch_losses, accum / max(step, 1))
        epoch_time = (time_ns() - epoch_t0) / 1e9
        epoch_callback(epoch, model_on_device, epoch_time)
    end
    progress !== nothing && ProgressMeter.finish!(progress)
    return TrainingHistory(epoch_losses, batch_losses)
end

"""
    score_from_model(model, batch, sigma)

Returns the estimated score ∇ log p(x) by rescaling the predicted noise.
"""
function score_from_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    return inv_sigma .* preds
end

function seed_thread_rngs(seed::Int)
    return [MersenneTwister(seed + tid) for tid in 1:nthreads()]
end

function fill_noise!(buffer, rngs::Vector{<:AbstractRNG})
    idx = clamp(threadid(), 1, length(rngs))
    randn!(rngs[idx], buffer)
    return buffer
end

function noise_like(batch, device::ExecutionDevice, rngs::Vector{<:AbstractRNG})
    noise = similar(batch)
    if device isa GPUDevice
        CUDA.randn!(noise)
    else
        fill_noise!(noise, rngs)
    end
    return noise
end
