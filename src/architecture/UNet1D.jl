using Flux

"""
Configuration container for the 1D Score U-Net.
"""
Base.@kwdef mutable struct ScoreUNetConfig
    in_channels::Int = 1
    out_channels::Int = 1
    base_channels::Int = 32
    channel_multipliers::Vector{Int} = [1, 2, 4]
    kernel_size::Int = 5
    periodic::Bool = false
    activation::Function = Flux.gelu
    final_activation::Function = identity
end

"""
U-Net wrapper that keeps the down path, bottleneck, up path, and final projection.
"""
struct ScoreUNet{D,B,U,F}
    down_blocks::Vector{D}
    bottleneck::B
    up_blocks::Vector{U}
    final_layer::F
end

Functors.@functor ScoreUNet

"""
    build_unet(cfg::ScoreUNetConfig)

Constructs a `ScoreUNet` based on the provided configuration.
"""
function build_unet(cfg::ScoreUNetConfig)
    channels = cfg.base_channels .* cfg.channel_multipliers
    isempty(channels) && throw(ArgumentError("channel_multipliers must not be empty"))

    down_blocks = Vector{DownBlock}(undef, length(channels))
    in_ch = cfg.in_channels
    for (i, ch) in enumerate(channels)
        down_blocks[i] = DownBlock(in_ch, ch;
                                   kernel=cfg.kernel_size,
                                   periodic=cfg.periodic,
                                   activation=cfg.activation)
        in_ch = ch
    end

    bottleneck_channels = 2 * in_ch
    bottleneck = ConvBlock(in_ch, bottleneck_channels;
                           kernel=cfg.kernel_size,
                           periodic=cfg.periodic,
                           activation=cfg.activation)

    up_blocks = Vector{UpBlock}(undef, length(channels))
    current = bottleneck_channels
    for (i, ch) in enumerate(reverse(channels))
        up_blocks[i] = UpBlock(current, ch, ch;
                               kernel=cfg.kernel_size,
                               periodic=cfg.periodic,
                               activation=cfg.activation)
        current = ch
    end

    final_projection = make_conv1d(current, cfg.out_channels;
                                   kernel=1,
                                   periodic=cfg.periodic,
                                   activation=identity)
    final_layer = cfg.final_activation === identity ?
        final_projection :
        Chain(final_projection, cfg.final_activation)
    return ScoreUNet(down_blocks, bottleneck, up_blocks, final_layer)
end

function (model::ScoreUNet)(x)
    h = x
    skips = ()
    for block in model.down_blocks
        skip, h = block(h)
        skips = (skips..., skip)
    end
    h = model.bottleneck(h)
    for (block, skip) in zip(model.up_blocks, Base.Iterators.reverse(skips))
        h = block(h, skip)
    end
    return model.final_layer(h)
end
