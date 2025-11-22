using Flux
using NNlib

"""
    same_padding(kernel, dilation)

Computes the amount of padding needed to emulate `SamePad` for manual padding paths.
"""
same_padding(kernel::Integer, dilation::Integer) = begin
    total = dilation * (kernel - 1)
    left = fld(total, 2)
    right = total - left
    return left, right
end

"""
    make_conv1d(in_ch, out_ch; kwargs...)

Factory that returns either a standard convolution or a periodic variant.
"""
function make_conv1d(in_ch::Integer, out_ch::Integer;
                     kernel::Integer=3, stride::Integer=1, dilation::Integer=1,
                     periodic::Bool=false, pad::Union{Symbol,Tuple,Int}=:same,
                     activation::Function=identity, bias::Bool=true)
    layer = periodic ?
        PeriodicConv1D(in_ch, out_ch; kernel=kernel, stride=stride, dilation=dilation,
                       pad=pad, use_bias=bias) :
        Flux.Conv((kernel,), in_ch=>out_ch;
                  stride=(stride,), dilation=(dilation,),
                  pad=(pad === :same ? Flux.SamePad() : pad),
                  bias=bias)
    return activation === identity ? layer : Chain(layer, activation)
end

"""
    ConvBlock

Two-layer convolutional block with normalization and activation in between.
"""
struct ConvBlock{C}
    layers::C
end

Functors.@functor ConvBlock

function ConvBlock(in_ch::Integer, out_ch::Integer;
                   kernel::Integer=3, periodic::Bool=false,
                   activation::Function=Flux.gelu)
    conv1 = make_conv1d(in_ch, out_ch; kernel=kernel, periodic=periodic)
    bn1 = Flux.BatchNorm(out_ch)
    conv2 = make_conv1d(out_ch, out_ch; kernel=kernel, periodic=periodic)
    bn2 = Flux.BatchNorm(out_ch)
    return ConvBlock(Chain(conv1, bn1, activation, conv2, bn2, activation))
end

(block::ConvBlock)(x) = block.layers(x)

"""
    DownBlock

Encoder stage that records a skip tensor and downsamples for the next stage.
"""
struct DownBlock{B,D}
    conv::B
    downsample::D
end

Functors.@functor DownBlock

function DownBlock(in_ch::Integer, out_ch::Integer;
                   kernel::Integer=3, periodic::Bool=false,
                   activation::Function=Flux.gelu)
    conv = ConvBlock(in_ch, out_ch; kernel=kernel, periodic=periodic, activation=activation)
    down = make_conv1d(out_ch, out_ch; kernel=2, stride=2, periodic=periodic, activation=identity, pad=0)
    return DownBlock(conv, down)
end

function (block::DownBlock)(x)
    h = block.conv(x)
    return h, block.downsample(h)
end

"""
Nearest-neighbor upsampling on the spatial dimension.
"""
struct Upsample1D
    factor::Int
end

(layer::Upsample1D)(x) = NNlib.upsample_nearest(x, (layer.factor,))

"""
    UpBlock

Decoder stage that upsamples, merges the skip connection, and applies a conv block.
"""
struct UpBlock{U,C}
    upsample::U
    conv::C
end

Functors.@functor UpBlock

function UpBlock(in_ch::Integer, skip_ch::Integer, out_ch::Integer;
                 kernel::Integer=3, periodic::Bool=false,
                 activation::Function=Flux.gelu)
    up = Upsample1D(2)
    block = ConvBlock(in_ch + skip_ch, out_ch;
                      kernel=kernel, periodic=periodic, activation=activation)
    return UpBlock(up, block)
end

function (block::UpBlock)(x, skip)
    h = block.upsample(x)
    h = match_length(h, skip)
    h = cat(h, skip; dims=2)
    return block.conv(h)
end

"""
    match_length(x, ref)

Ensures that the spatial dimension of `x` matches `ref` via symmetric crop/pad.
"""
function match_length(x, ref)
    lx = size(x, 1)
    lr = size(ref, 1)
    if lx == lr
        return x
    elseif lx > lr
        start = fld(lx - lr, 2) + 1
        stop = start + lr - 1
        return @view(x[start:stop, :, :])
    else
        pad_total = lr - lx
        left = fld(pad_total, 2)
        right = pad_total - left
        return zero_pad(x, left, right)
    end
end

"""
    zero_pad(x, left, right)

Zero padding helper on the spatial dimension.
"""
function zero_pad(x, left::Integer, right::Integer)
    (left == 0 && right == 0) && return x
    w, c, b = size(x)
    out = similar(x, w + left + right, c, b)
    if left > 0
        out[1:left, :, :] .= 0
    end
    out[left+1:left+w, :, :] .= x
    if right > 0
        out[left+w+1:end, :, :] .= 0
    end
    return out
end
