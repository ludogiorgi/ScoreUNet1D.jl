"""
Configuration for the Schneider-topology dual-stream score model.

The slow stream lives on a ring of length `K`, while the fast stream lives on a
single twisted ring of length `K*J`.
"""
Base.@kwdef mutable struct L96SchneiderScoreConfig
    K::Int = 36
    J::Int = 10
    slow_base_channels::Int = 16
    fast_base_channels::Int = 32
    slow_channel_multipliers::Vector{Int} = [1, 2]
    fast_channel_multipliers::Vector{Int} = [1, 2, 4]
    kernel_size::Int = 5
    norm_type::Symbol = :group
    norm_groups::Int = 0
    activation::Function = Flux.gelu
end

"""
Dual-stream score network aligned with Schneider et al. fast-ring topology.

Input/output convention:
- slow stream: `(K, 1, B)` for `X`
- fast stream: `(K*J, 1, B)` for flattened fast ring `Y_flat`
- return value: `(score_X, score_Y_flat)` with matching shapes
"""
struct L96SchneiderScoreModel{S,F}
    K::Int
    J::Int
    slow_net::S
    fast_net::F
end

Functors.@functor L96SchneiderScoreModel

"""
Adapter for legacy `(K, J+1, B)` tensors where channel 1 is slow `X_k` and
channels `2:J+1` are fast `Y_{j,k}`.

Internally this adapter flattens fast variables to `(K*J, 1, B)`, runs the
dual-stream Schneider model, and maps scores back to `(K, J+1, B)`.
"""
struct L96SchneiderLegacyAdapter{M}
    model::M
    J::Int
end

Functors.@functor L96SchneiderLegacyAdapter

function build_l96_schneider_model(cfg::L96SchneiderScoreConfig)
    cfg.K > 0 || throw(ArgumentError("K must be positive, got $(cfg.K)"))
    cfg.J > 0 || throw(ArgumentError("J must be positive, got $(cfg.J)"))

    slow_cfg = ScoreUNetConfig(
        in_channels=2,
        out_channels=1,
        base_channels=cfg.slow_base_channels,
        channel_multipliers=cfg.slow_channel_multipliers,
        kernel_size=cfg.kernel_size,
        periodic=true,
        norm_type=cfg.norm_type,
        norm_groups=cfg.norm_groups,
        activation=cfg.activation,
    )
    fast_cfg = ScoreUNetConfig(
        in_channels=2,
        out_channels=1,
        base_channels=cfg.fast_base_channels,
        channel_multipliers=cfg.fast_channel_multipliers,
        kernel_size=cfg.kernel_size,
        periodic=true,
        norm_type=cfg.norm_type,
        norm_groups=cfg.norm_groups,
        activation=cfg.activation,
    )

    return L96SchneiderScoreModel(
        cfg.K,
        cfg.J,
        build_unet(slow_cfg),
        build_unet(fast_cfg),
    )
end

function broadcast_X_to_fast(x::AbstractVector, J::Int)::Vector{Float32}
    J > 0 || throw(ArgumentError("J must be positive, got $J"))
    K = length(x)
    out = Vector{Float32}(undef, K * J)
    @inbounds for k in 1:K
        xk = Float32(x[k])
        base = (k - 1) * J
        for j in 1:J
            out[base + j] = xk
        end
    end
    return out
end

function broadcast_X_to_fast(x::AbstractArray{<:Real,3}, J::Int)
    size(x, 2) == 1 || throw(ArgumentError("Expected slow tensor with one channel, got size $(size(x))"))
    J > 0 || throw(ArgumentError("J must be positive, got $J"))
    x32 = Float32.(x)
    return NNlib.upsample_nearest(x32, (J,))
end

function channelized_fast_to_flat(y::AbstractArray{<:Real,3})
    K = size(y, 1)
    J = size(y, 2)
    B = size(y, 3)
    y32 = Float32.(y)
    # Convert (K, J, B) to (J, K, B), then flatten J within each K block.
    yjk = permutedims(y32, (2, 1, 3))
    return reshape(yjk, K * J, 1, B)
end

function flat_fast_to_channelized(y_flat::AbstractArray{<:Real,3}, J::Int)
    size(y_flat, 2) == 1 || throw(ArgumentError("Expected flattened fast tensor with one channel"))
    L = size(y_flat, 1)
    B = size(y_flat, 3)
    L % J == 0 || throw(ArgumentError("Fast ring length $L must be divisible by J=$J"))
    K = div(L, J)
    y32 = Float32.(y_flat)
    yjk = reshape(y32, J, K, B)
    return permutedims(yjk, (2, 1, 3))
end

function blockmean_fast_to_slow(y_flat::AbstractVector, J::Int)::Vector{Float32}
    J > 0 || throw(ArgumentError("J must be positive, got $J"))
    length(y_flat) % J == 0 ||
        throw(ArgumentError("Fast ring length $(length(y_flat)) must be divisible by J=$J"))
    K = div(length(y_flat), J)
    out = Vector{Float32}(undef, K)
    invJ = Float32(1 / J)
    @inbounds for k in 1:K
        acc = 0.0f0
        base = (k - 1) * J
        for j in 1:J
            acc += Float32(y_flat[base + j])
        end
        out[k] = acc * invJ
    end
    return out
end

function blockmean_fast_to_slow(y_flat::AbstractArray{<:Real,3}, J::Int)
    size(y_flat, 2) == 1 || throw(ArgumentError("Expected fast tensor with one channel, got size $(size(y_flat))"))
    L = size(y_flat, 1)
    L % J == 0 || throw(ArgumentError("Fast ring length $L must be divisible by J=$J"))
    K = div(L, J)
    B = size(y_flat, 3)
    y32 = Float32.(y_flat)
    yjk = reshape(y32, J, K, B)
    ymean = mean(yjk; dims=1)
    return permutedims(ymean, (2, 1, 3))
end

function (model::L96SchneiderScoreModel)(x_slow::AbstractArray{<:Real,3},
                                         y_fast::AbstractArray{<:Real,3})
    size(x_slow, 2) == 1 || throw(ArgumentError("Expected X with one channel, got size $(size(x_slow))"))
    size(y_fast, 2) == 1 || throw(ArgumentError("Expected Y_flat with one channel, got size $(size(y_fast))"))
    size(x_slow, 1) == model.K || throw(ArgumentError("Expected slow length K=$(model.K), got $(size(x_slow, 1))"))
    size(y_fast, 1) == model.K * model.J ||
        throw(ArgumentError("Expected fast length K*J=$(model.K * model.J), got $(size(y_fast, 1))"))
    size(x_slow, 3) == size(y_fast, 3) || throw(ArgumentError("Batch size mismatch"))

    x_up = broadcast_X_to_fast(x_slow, model.J)
    y_bar = blockmean_fast_to_slow(y_fast, model.J)

    slow_in = cat(Float32.(x_slow), y_bar; dims=2)
    fast_in = cat(Float32.(y_fast), x_up; dims=2)

    score_x = model.slow_net(slow_in)
    score_y = model.fast_net(fast_in)
    return score_x, score_y
end

function build_l96_schneider_legacy_model(cfg::L96SchneiderScoreConfig)
    core = build_l96_schneider_model(cfg)
    return L96SchneiderLegacyAdapter(core, cfg.J)
end

function (adapter::L96SchneiderLegacyAdapter)(state::AbstractArray{<:Real,3})
    size(state, 2) == adapter.J + 1 ||
        throw(ArgumentError("Expected legacy tensor with channels J+1=$(adapter.J + 1), got size $(size(state))"))

    x_slow = Float32.(@view state[:, 1:1, :])
    y_channelized = @view state[:, 2:(adapter.J + 1), :]
    y_fast = channelized_fast_to_flat(y_channelized)

    score_x, score_y_flat = adapter.model(x_slow, y_fast)
    score_y_channelized = flat_fast_to_channelized(score_y_flat, adapter.J)
    return cat(score_x, score_y_channelized; dims=2)
end
