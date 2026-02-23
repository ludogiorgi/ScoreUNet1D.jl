using Flux
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
    accumulation_steps::Int = 1  # Number of batches to accumulate before update
    use_lr_schedule::Bool = false  # Enable learning rate scheduling
    warmup_steps::Int = 500  # Linear warmup steps
    min_lr_factor::Float64 = 0.1  # Minimum LR as fraction of max_lr
    epoch_subset_size::Int = 0  # If > 0, randomly sample this many samples each epoch
    x_loss_weight::Float32 = 1.0f0
    y_loss_weight::Float32 = 1.0f0
    mean_match_weight::Float32 = 0.0f0
    cov_match_weight::Float32 = 0.0f0
end

struct TrainingHistory
    epoch_losses::Vector{Float32}
    batch_losses::Vector{Float32}
end

Base.@kwdef mutable struct TrainingState
    epoch::Int = 0
    global_step::Int = 0
    opt_state::Any = nothing
    epoch_losses::Vector{Float32} = Float32[]
    batch_losses::Vector{Float32} = Float32[]
    rng::MersenneTwister = MersenneTwister(0)
    thread_rngs::Vector{MersenneTwister} = MersenneTwister[]
end

"""
    train!(model, dataset, cfg; callback=nothing)

Runs denoising score matching with σ = cfg.sigma.
"""
function train!(model, dataset::NormalizedDataset, cfg::ScoreTrainerConfig;
    callback::Function=(_, _) -> nothing,
    epoch_callback::Function=(_, _, _) -> nothing,
    state_callback::Function=(_) -> nothing,
    initial_state::Union{Nothing,TrainingState}=nothing,
    device::ExecutionDevice=CPUDevice())

    if device isa GPUDevice && gpu_count(device) > 1
        initial_state === nothing || error("Resuming from a saved training state is not supported in multi-GPU mode.")
        return train_multi_gpu!(model, dataset, cfg;
            callback=callback,
            epoch_callback=epoch_callback,
            state_callback=state_callback,
            initial_state=initial_state,
            device=device)
    end

    return train_single_device!(model, dataset, cfg;
        callback=callback,
        epoch_callback=epoch_callback,
        state_callback=state_callback,
        initial_state=initial_state,
        device=device)
end

function train_single_device!(model, dataset::NormalizedDataset, cfg::ScoreTrainerConfig;
    callback::Function,
    epoch_callback::Function,
    state_callback::Function,
    initial_state::Union{Nothing,TrainingState},
    device::ExecutionDevice)
    n = length(dataset)
    n == 0 && error("Dataset is empty")
    start_epoch = initial_state === nothing ? 0 : Int(initial_state.epoch)
    start_epoch <= cfg.epochs || error("initial_state.epoch=$(start_epoch) exceeds configured epochs=$(cfg.epochs)")
    rng = initial_state === nothing ? MersenneTwister(cfg.seed) : copy(initial_state.rng)
    thread_rngs = initial_state === nothing ? seed_thread_rngs(cfg.seed) : [copy(r) for r in initial_state.thread_rngs]
    isempty(thread_rngs) && (thread_rngs = seed_thread_rngs(cfg.seed))

    # Ensure we are on the correct device
    if device isa GPUDevice
        CUDA.device!(device.ids[1])
    end

    model_on_device = model
    Flux.trainmode!(model_on_device) # Ensure BatchNorm stats are updated
    opt_state = initial_state === nothing ? Flux.setup(Flux.Optimisers.Adam(cfg.lr), model_on_device) : initial_state.opt_state
    epoch_losses = initial_state === nothing ? Float32[] : copy(initial_state.epoch_losses)
    batch_losses = initial_state === nothing ? Float32[] : copy(initial_state.batch_losses)
    steps_per_epoch = ceil(Int, n / cfg.batch_size)
    total_steps = max(cfg.epochs - start_epoch, 0) * steps_per_epoch
    progress = cfg.progress ? Progress(total_steps; desc="Training score network") : nothing

    # Setup learning rate scheduler if enabled
    lr_scheduler = cfg.use_lr_schedule ? create_lr_schedule(cfg, steps_per_epoch) : nothing
    global_step = initial_state === nothing ? 0 : Int(initial_state.global_step)

    # Gradient accumulation state
    accumulated_grads = nothing
    accum_count = 0

    # Pre-allocate CPU buffer for batch data to avoid allocations
    # Dimensions: (Length, Channels, BatchSize)
    L, C, _ = size(dataset.data)
    batch_cpu_buffer = Array{Float32}(undef, L, C, cfg.batch_size)

    # Pre-allocate buffers to avoid allocation in loop
    batch_gpu_buffer = nothing
    noise_gpu_buffer = nothing
    noisy_gpu_buffer = nothing
    noise_cpu_buffer = Array{Float32}(undef, L, C, cfg.batch_size)

    if device isa GPUDevice
        CUDA.seed!(UInt64(cfg.seed))
        batch_gpu_buffer = CUDA.CuArray{Float32}(undef, L, C, cfg.batch_size)
        noise_gpu_buffer = CUDA.CuArray{Float32}(undef, L, C, cfg.batch_size)
        noisy_gpu_buffer = CUDA.CuArray{Float32}(undef, L, C, cfg.batch_size)
    end

    # Channel-wise denoising loss weighting:
    # channel 1 is x, channels 2:C are y.
    channel_weights = fill(Float32(cfg.y_loss_weight), C)
    channel_weights[1] = Float32(cfg.x_loss_weight)
    channel_weights ./= mean(channel_weights)
    channel_weight_cpu = reshape(channel_weights, 1, C, 1)
    channel_weight_gpu = device isa GPUDevice ? CUDA.cu(channel_weight_cpu) : nothing

    for epoch in (start_epoch + 1):cfg.epochs
        epoch_t0 = time_ns()

        # Per-epoch random subset selection if epoch_subset_size > 0
        if cfg.epoch_subset_size > 0 && cfg.epoch_subset_size < n
            idxs = collect(1:n)
            Random.shuffle!(rng, idxs)
            idxs = idxs[1:cfg.epoch_subset_size]
            # Optionally shuffle the subset order too
            cfg.shuffle && Random.shuffle!(rng, idxs)
        else
            idxs = collect(1:n)
            cfg.shuffle && Random.shuffle!(rng, idxs)
        end
        batches = Iterators.partition(idxs, cfg.batch_size)
        step = 0
        accum = 0.0

        for batch_idxs in batches
            global_step += 1
            current_batch_size = length(batch_idxs)

            # Adjust learning rate if scheduling is enabled
            if lr_scheduler !== nothing
                new_lr = lr_scheduler(global_step)
                Flux.adjust!(opt_state, new_lr)
            end

            # Efficient data loading: copy into pre-allocated buffer
            # We use a view of the buffer if the batch is smaller than batch_size (last batch)
            if current_batch_size == cfg.batch_size
                # Direct copy into buffer
                Threads.@threads for b in 1:current_batch_size
                    idx = batch_idxs[b]
                    @inbounds @simd for i in 1:(L*C)
                        batch_cpu_buffer[i+(b-1)*L*C] = dataset.data[i+(idx-1)*L*C]
                    end
                end
                batch_cpu = batch_cpu_buffer
            else
                # Handle last batch (smaller)
                batch_view = view(batch_cpu_buffer, :, :, 1:current_batch_size)
                Threads.@threads for b in 1:current_batch_size
                    idx = batch_idxs[b]
                    @inbounds @simd for i in 1:(L*C)
                        batch_view[i+(b-1)*L*C] = dataset.data[i+(idx-1)*L*C]
                    end
                end
                batch_cpu = batch_view
            end

            # Prepare batch and noise on device
            local batch, noise

            if device isa GPUDevice
                # GPU Path: Use pre-allocated buffers
                batch_view_gpu = view(batch_gpu_buffer, :, :, 1:current_batch_size)
                CUDA.@allowscalar copyto!(batch_view_gpu, batch_cpu)
                batch = batch_view_gpu

                noise_view_gpu = view(noise_gpu_buffer, :, :, 1:current_batch_size)
                CUDA.randn!(noise_view_gpu)
                noise = noise_view_gpu
            else
                # CPU Path: Use pre-allocated buffers and threaded noise
                batch = batch_cpu
                noise = view(noise_cpu_buffer, :, :, 1:current_batch_size)

                # Threaded noise generation
                Threads.@threads for b in 1:current_batch_size
                    tid = Threads.threadid()
                    rng_local = thread_rngs[tid]
                    @inbounds @simd for i in 1:(L*C)
                        noise[i+(b-1)*L*C] = randn(rng_local, Float32)
                    end
                end
            end

            if device isa GPUDevice
                noisy_view = view(noisy_gpu_buffer, :, :, 1:current_batch_size)
                @. noisy_view = batch + cfg.sigma * noise
                noisy = noisy_view
            else
                noisy = batch .+ cfg.sigma .* noise
            end

            # Scale loss by accumulation steps for proper gradient magnitude
            loss, grads = Flux.withgradient(model_on_device) do m
                pred = m(noisy)
                if device isa GPUDevice
                    denoising_loss_with_moments(pred, noise, noisy, batch, channel_weight_gpu, cfg) / cfg.accumulation_steps
                else
                    denoising_loss_with_moments(pred, noise, noisy, batch, channel_weight_cpu, cfg) / cfg.accumulation_steps
                end
            end

            # Accumulate gradients
            if accumulated_grads === nothing
                accumulated_grads = grads[1]
            else
                # Add current gradients to accumulated
                accumulated_grads = Functors.fmap(accumulated_grads, grads[1]) do acc, grad
                    if acc === nothing || grad === nothing
                        return acc === nothing ? grad : acc
                    elseif acc isa AbstractArray
                        return acc .+ grad
                    else
                        return acc
                    end
                end
            end
            accum_count += 1

            loss32 = Float32(loss) * cfg.accumulation_steps  # Unscale for logging
            push!(batch_losses, loss32)
            accum += loss32
            step += 1

            # Update parameters every accumulation_steps or at end of epoch
            should_update = (accum_count >= cfg.accumulation_steps) ||
                            (cfg.max_steps_per_epoch !== nothing && step >= cfg.max_steps_per_epoch)

            if should_update && accumulated_grads !== nothing
                opt_state, model_on_device = Flux.update!(opt_state, model_on_device, accumulated_grads)
                accumulated_grads = nothing
                accum_count = 0
            end

            progress !== nothing && ProgressMeter.next!(progress; showvalues=[(:epoch, epoch), (:loss, loss32)])
            callback(loss32, epoch)

            if cfg.max_steps_per_epoch !== nothing && step >= cfg.max_steps_per_epoch
                break
            end
        end

        # Update with any remaining accumulated gradients at end of epoch
        if accumulated_grads !== nothing && accum_count > 0
            opt_state, model_on_device = Flux.update!(opt_state, model_on_device, accumulated_grads)
            accumulated_grads = nothing
            accum_count = 0
        end

        push!(epoch_losses, accum / max(step, 1))
        epoch_time = (time_ns() - epoch_t0) / 1e9
        epoch_callback(epoch, model_on_device, epoch_time)
        state_callback(TrainingState(
            epoch=epoch,
            global_step=global_step,
            opt_state=opt_state,
            epoch_losses=copy(epoch_losses),
            batch_losses=copy(batch_losses),
            rng=copy(rng),
            thread_rngs=[copy(r) for r in thread_rngs],
        ))
    end
    progress !== nothing && ProgressMeter.finish!(progress)
    return TrainingHistory(epoch_losses, batch_losses)
end

"""
    score_from_model(model, batch, sigma)

Returns the estimated score ∇ log p(x) by rescaling the predicted noise.

The denoising score matching identity: for y = x + σz where z ~ N(0,I),
the model learns to predict E[z|y]. The score of the smoothed density is:
    ∇ log p_σ(y) = -E[σz|y] / σ² = -E[z|y] / σ = -model(y) / σ
"""
function score_from_model(model, batch, sigma::Real)
    preds = model(batch)
    inv_sigma = -one(eltype(preds)) / sigma
    @. preds *= inv_sigma
    return preds
end

function seed_thread_rngs(seed::Int)
    return [MersenneTwister(seed + tid) for tid in 1:nthreads()]
end

function fill_noise!(buffer, rngs::Vector{<:AbstractRNG})
    idx = clamp(threadid(), 1, length(rngs))
    randn!(rngs[idx], buffer)
    return buffer
end

@inline function xy_group_moments(batch)
    C = size(batch, 2)
    xvals = @view batch[:, 1, :]
    μx = mean(xvals)
    dx = xvals .- μx
    varx = mean(dx .* dx)

    if C >= 2
        yvals = @view batch[:, 2:C, :]
        μy = mean(yvals)
        dy = yvals .- μy
        vary = mean(dy .* dy)
    else
        μy = zero(μx)
        vary = zero(varx)
    end

    return μx, μy, varx, vary
end

function denoising_loss_with_moments(pred, noise, noisy, clean_batch, channel_weight, cfg::ScoreTrainerConfig)
    base = mean(((pred .- noise) .^ 2) .* channel_weight)

    if cfg.mean_match_weight == 0f0 && cfg.cov_match_weight == 0f0
        return base
    end

    denoised = noisy .- cfg.sigma .* pred
    μx_pred, μy_pred, varx_pred, vary_pred = xy_group_moments(denoised)
    μx_true, μy_true, varx_true, vary_true = xy_group_moments(clean_batch)

    mean_term = (μx_pred - μx_true)^2 + (μy_pred - μy_true)^2
    cov_term = (varx_pred - varx_true)^2 + (vary_pred - vary_true)^2

    return base + cfg.mean_match_weight * mean_term + cfg.cov_match_weight * cov_term
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

function split_batch_for_gpus(batch::Array{T,3}, ngpu::Int) where {T}
    L, C, B = size(batch)
    splits = fill(div(B, ngpu), ngpu)
    for i in 1:rem(B, ngpu)
        splits[i] += 1
    end
    chunks = Vector{Array{T,3}}(undef, ngpu)
    idx = 1
    for i in 1:ngpu
        s = splits[i]
        if s > 0
            chunks[i] = Array(batch[:, :, idx:idx+s-1])
            idx += s
        else
            chunks[i] = Array{T}(undef, L, C, 0)
        end
    end
    return chunks, splits
end

"""
    split_batch_for_gpus_views(batch, ngpu)

Zero-copy batch splitting using views instead of array copies.
This eliminates unnecessary memory allocation and copying before GPU transfer.
"""
function split_batch_for_gpus_views(batch::AbstractArray{T,3}, ngpu::Int) where {T}
    L, C, B = size(batch)
    base = div(B, ngpu)
    extra = rem(B, ngpu)
    splits = fill(base, ngpu)
    for i in 1:extra
        splits[i] += 1
    end

    chunks = Vector{SubArray{T,3}}(undef, ngpu)
    offset = 0
    for i in 1:ngpu
        count = splits[i]
        chunks[i] = view(batch, :, :, (offset+1):(offset+count))
        offset += count
    end
    return chunks, splits
end

"""
    create_lr_schedule(cfg, total_batches)

Creates a learning rate schedule with linear warmup and cosine decay.
Returns a function that takes step number and returns the learning rate.
"""
function create_lr_schedule(cfg::ScoreTrainerConfig, total_batches::Int)
    warmup_steps = cfg.warmup_steps
    total_steps = cfg.epochs * total_batches
    max_lr = cfg.lr
    min_lr = max_lr * cfg.min_lr_factor

    function lr_scheduler(step::Int)
        if step < warmup_steps
            # Linear warmup
            return max_lr * (step / warmup_steps)
        else
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = clamp(progress, 0.0, 1.0)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
        end
    end

    return lr_scheduler
end

scale_tree(tree, w) = Functors.fmap(x -> x isa AbstractArray ? x .* w : x, tree)
to_cpu_tree(tree) = Functors.fmap(x -> x isa AbstractArray ? Array(x) : x, tree)
to_device_tree(tree, dev_id::Int) = begin
    CUDA.device!(dev_id)
    Functors.fmap(x -> x isa AbstractArray ? CUDA.cu(x) : x, tree)
end

function accumulate_trees(dest, src)
    dest === nothing && return src
    return Functors.fmap(dest, src) do a, b
        if a === nothing
            return b
        elseif b === nothing
            return a
        elseif a isa AbstractArray
            return a .+ b
        else
            return a
        end
    end
end

"""
    accumulate_gpu_tree_inplace!(dest, src, weight)

Accumulate src into dest with weighting, keeping everything on GPU.
Modifies dest in place. Returns dest.
"""
# Helper for safe tree walking with potential Nothing values
function safe_walk(recurse, x, ys...)
    if any(isnothing, ys) || x isa Tuple
        return x
    end
    return Functors.DefaultWalk()(recurse, x, ys...)
end

function accumulate_gpu_tree_inplace!(dest, src, weight::Float32)
    Functors.fmap(dest, src; walk=safe_walk) do d, s
        if d === nothing
            # If dest is nothing, we only take src if it's numeric/array
            if s isa AbstractArray
                return s .* weight
            else
                return d
            end
        elseif s === nothing
            return d
        elseif d isa CUDA.CuArray && s isa CUDA.CuArray
            # In-place accumulation on GPU
            CUDA.@. d = d + s * weight
            return d
        elseif d isa AbstractArray && s isa AbstractArray
            return d .+ s .* weight
        else
            return d
        end
    end
    return dest
end

"""
    copy_tree_to_gpu!(dest_tree, src_tree, dest_device_id, src_device_id)

Copy src_tree from src GPU to dest GPU directly, without going through CPU.
Both trees must already be on their respective GPUs.
"""
function copy_tree_to_gpu!(dest_tree, src_tree, dest_device_id::Int, src_device_id::Int)
    CUDA.device!(dest_device_id)
    Functors.fmap(dest_tree, src_tree; walk=safe_walk) do d, s
        if s === nothing
            # Source is missing. Zero out destination to represent zero gradient/value.
            if d isa AbstractArray
                fill!(d, 0)
            elseif !isnothing(d) && !isa(d, Tuple)
                # Recurse to zero out children of struct
                Functors.fmap(x -> (x isa AbstractArray ? fill!(x, 0) : x), d)
            end
            return d
        elseif d === nothing
            return d
        elseif d isa CUDA.CuArray && s isa CUDA.CuArray
            # Direct GPU-to-GPU copy using copyto!
            copyto!(d, s)
            return d
        elseif d isa AbstractArray && s isa AbstractArray
            copyto!(d, s)
            return d
        else
            return d
        end
    end
    return dest_tree
end

"""
    allocate_like_tree(tree, device_id)

Allocate a new tree with the same structure as the input, on the specified GPU.
"""
function allocate_like_tree(tree, device_id::Int)
    CUDA.device!(device_id)
    return Functors.fmap(tree) do x
        if x isa AbstractArray
            similar_array = CUDA.CuArray{eltype(x)}(undef, size(x)...)
            fill!(similar_array, zero(eltype(x)))
            return similar_array
        else
            return x
        end
    end
end

# Multi-GPU training functions removed for optimization
# Single-GPU training is now enforced for better performance

function train_multi_gpu!(model, dataset::NormalizedDataset, cfg::ScoreTrainerConfig;
    callback::Function,
    epoch_callback::Function,
    state_callback::Function,
    initial_state::Union{Nothing,TrainingState},
    device::GPUDevice)
    initial_state === nothing || error("Resuming from a saved training state is not supported in multi-GPU mode.")
    ngpus = length(device.ids)
    n = length(dataset)
    n == 0 && error("Dataset is empty")

    # Master device is always the first one
    master_dev = device.ids[1]
    CUDA.device!(master_dev)

    # 1. Setup models on all GPUs
    # We maintain a separate model replica on each GPU to avoid constant transfer of weights
    models = Vector{Any}(undef, ngpus)
    models[1] = model # Already on master (or will be ensured)

    # Ensure master model is on master device
    models[1] = to_device_tree(models[1], master_dev)

    # Replicate to other GPUs
    for i in 2:ngpus
        # We copy from master to device i
        # We use deepcopy to get the structure, then copy weights
        # But deepcopying CUDA arrays across devices is tricky.
        # Easiest: Move to CPU, then to GPU i. But slow.
        # Better: Create structure on GPU i, then copyto!

        # Fast path: allocate like tree on device i, then copy
        models[i] = allocate_like_tree(models[1], device.ids[i])
        copy_tree_to_gpu!(models[i], models[1], device.ids[i], master_dev)
    end

    # 2. Setup Optimizer (only on master)
    CUDA.device!(master_dev)
    opt_state = Flux.setup(Flux.Optimisers.Adam(cfg.lr), models[1])

    # 3. Training State
    epoch_losses = Float32[]
    batch_losses = Float32[]
    steps_per_epoch = ceil(Int, n / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    progress = cfg.progress ? Progress(total_steps; desc="Training score network (Multi-GPU)") : nothing

    lr_scheduler = cfg.use_lr_schedule ? create_lr_schedule(cfg, steps_per_epoch) : nothing
    global_step = 0

    # RNGs for each GPU thread
    # We need a matrix of RNGs: [ngpus]
    # Actually, we spawn threads, so we can just use thread-local RNGs or pass them.
    # Let's create a vector of RNG vectors, one per GPU? 
    # No, just one RNG per GPU is enough for noise generation.
    gpu_rngs = [MersenneTwister(cfg.seed + i) for i in 1:ngpus]

    # Pre-allocate CPU buffer
    L, C, _ = size(dataset.data)
    batch_cpu_buffer = Array{Float32}(undef, L, C, cfg.batch_size)

    # Pre-allocate transfer buffers on Master to receive gradients from workers
    # This avoids allocating a new tree every step for aggregation
    # We need one buffer per worker (indices 2:ngpus)
    transfer_buffers = Vector{Any}(undef, ngpus)
    for i in 2:ngpus
        transfer_buffers[i] = allocate_like_tree(models[1], master_dev)
    end

    channel_weights = fill(Float32(cfg.y_loss_weight), C)
    channel_weights[1] = Float32(cfg.x_loss_weight)
    channel_weights ./= mean(channel_weights)
    channel_weight_cpu = reshape(channel_weights, 1, C, 1)
    channel_weight_gpu = Vector{Any}(undef, ngpus)
    for i in 1:ngpus
        CUDA.device!(device.ids[i])
        channel_weight_gpu[i] = CUDA.cu(channel_weight_cpu)
    end

    for epoch in 1:cfg.epochs
        epoch_t0 = time_ns()
        epoch_rng = MersenneTwister(cfg.seed + epoch)

        # Per-epoch random subset selection if epoch_subset_size > 0
        if cfg.epoch_subset_size > 0 && cfg.epoch_subset_size < n
            idxs = collect(1:n)
            Random.shuffle!(epoch_rng, idxs)
            idxs = idxs[1:cfg.epoch_subset_size]
            cfg.shuffle && Random.shuffle!(epoch_rng, idxs)
        else
            idxs = collect(1:n)
            cfg.shuffle && Random.shuffle!(epoch_rng, idxs)
        end
        batches = Iterators.partition(idxs, cfg.batch_size)
        step = 0
        accum = 0.0

        for batch_idxs in batches
            global_step += 1
            current_batch_size = length(batch_idxs)

            # LR Schedule
            if lr_scheduler !== nothing
                new_lr = lr_scheduler(global_step)
                Flux.adjust!(opt_state, new_lr)
            end

            # Load data to CPU buffer
            # (Same optimized loading as single-GPU)
            if current_batch_size == cfg.batch_size
                Threads.@threads for b in 1:current_batch_size
                    idx = batch_idxs[b]
                    @inbounds @simd for i in 1:(L*C)
                        batch_cpu_buffer[i+(b-1)*L*C] = dataset.data[i+(idx-1)*L*C]
                    end
                end
                batch_cpu = batch_cpu_buffer
            else
                batch_view = view(batch_cpu_buffer, :, :, 1:current_batch_size)
                Threads.@threads for b in 1:current_batch_size
                    idx = batch_idxs[b]
                    @inbounds @simd for i in 1:(L*C)
                        batch_view[i+(b-1)*L*C] = dataset.data[i+(idx-1)*L*C]
                    end
                end
                batch_cpu = batch_view
            end

            # Split batch for GPUs
            chunks, counts = split_batch_for_gpus_views(batch_cpu, ngpus)

            # Parallel Gradient Computation
            grads_vector = Vector{Any}(undef, ngpus)
            losses_vector = Vector{Float32}(undef, ngpus)

            # We use @threads to launch tasks for each GPU
            # Note: Julia threads != GPU streams, but with CUDA.device! it works.
            Threads.@threads for i in 1:ngpus
                dev_id = device.ids[i]
                CUDA.device!(dev_id)

                # Move data to GPU
                # chunks[i] is a CPU view. CUDA.cu copies it.
                batch_gpu = CUDA.cu(chunks[i])

                # Generate noise
                # We use a simple randn! on GPU
                noise = CUDA.randn(size(batch_gpu)...)
                noisy = batch_gpu .+ cfg.sigma .* noise

                # Compute Gradients
                # We scale loss by 1/ngpus to average across devices
                loss, grads = Flux.withgradient(models[i]) do m
                    pred = m(noisy)
                    denoising_loss_with_moments(pred, noise, noisy, batch_gpu, channel_weight_gpu[i], cfg) / ngpus
                end

                grads_vector[i] = grads[1]
                losses_vector[i] = loss
            end

            # Aggregation on Master
            CUDA.device!(master_dev)
            total_grads = grads_vector[1] # Grads from master

            for i in 2:ngpus
                # Copy grads from worker i to master transfer buffer
                # This uses P2P if available
                copy_tree_to_gpu!(transfer_buffers[i], grads_vector[i], master_dev, device.ids[i])

                # Accumulate
                accumulate_gpu_tree_inplace!(total_grads, transfer_buffers[i], 1.0f0)
            end

            # Update Master Model
            Flux.update!(opt_state, models[1], total_grads)

            # Broadcast updated weights to workers
            # We can do this in parallel threads too
            Threads.@threads for i in 2:ngpus
                copy_tree_to_gpu!(models[i], models[1], device.ids[i], master_dev)
            end

            # Logging
            avg_loss = sum(losses_vector) # Already scaled by 1/ngpus inside withgradient? 
            # Wait, mse returns mean over batch.
            # If we split batch B into B/2 and B/2.
            # Loss1 = sum(sq_err1) / (B/2)
            # Loss2 = sum(sq_err2) / (B/2)
            # Total Loss = sum(sq_err_total) / B = (Loss1 * B/2 + Loss2 * B/2) / B = (Loss1 + Loss2) / 2
            # So yes, we should average the losses.
            # In withgradient I divided by ngpus? 
            # mse(pred, noise) is mean.
            # So I should return mse(pred, noise) and then average them outside.
            # But for gradients:
            # L = (L1 + L2) / 2
            # dL/dw = 0.5 * dL1/dw + 0.5 * dL2/dw
            # So I should scale gradients by 1/ngpus.
            # In the code above: `mse(pred, noise) / ngpus`. This scales loss and grads by 1/ngpus.
            # So `total_grads` is sum(grads_i / ngpus) = mean(grads). Correct.
            # And `sum(losses_vector)` will be sum(L_i / ngpus) = mean(L). Correct.

            loss32 = Float32(avg_loss)
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
        epoch_callback(epoch, models[1], epoch_time)
    end

    progress !== nothing && ProgressMeter.finish!(progress)
    return TrainingHistory(epoch_losses, batch_losses)
end
