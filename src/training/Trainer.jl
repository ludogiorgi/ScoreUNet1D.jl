using Flux
using Flux.Losses: mse
using Flux.Optimisers
using Random
using ProgressMeter
using Base.Threads

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
                callback::Function = (_, _) -> nothing)
    n = length(dataset)
    n == 0 && error("Dataset is empty")
    rng = MersenneTwister(cfg.seed)
    thread_rngs = seed_thread_rngs(cfg.seed)
    opt_state = Flux.setup(Flux.Optimisers.Adam(cfg.lr), model)
    epoch_losses = Float32[]
    batch_losses = Float32[]
    steps_per_epoch = ceil(Int, n / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    progress = cfg.progress ? Progress(total_steps; desc="Training score network") : nothing

    for epoch in 1:cfg.epochs
        idxs = collect(1:n)
        cfg.shuffle && Random.shuffle!(rng, idxs)
        batches = Iterators.partition(idxs, cfg.batch_size)
        step = 0
        accum = 0.0
        for batch_idxs in batches
            batch = Array(dataset.data[:, :, batch_idxs])
            noise = similar(batch)
            fill_noise!(noise, thread_rngs)
            noisy = batch .+ cfg.sigma .* noise
            loss, grads = Flux.withgradient(model) do m
                pred = m(noisy)
                mse(pred, noise)
            end
            opt_state, model = Flux.update!(opt_state, model, grads[1])
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
    return (-1 / sigma) .* preds
end

function seed_thread_rngs(seed::Int)
    return [MersenneTwister(seed)]
end

function fill_noise!(buffer, rngs::Vector{<:AbstractRNG})
    randn!(rngs[1], buffer)
    return buffer
end
