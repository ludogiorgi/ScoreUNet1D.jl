using Flux
using NNlib

"""
    PeriodicConv1D(conv::Flux.Conv, pad_left::Integer, pad_right::Integer)

Wraps a standard 1D convolution and injects periodic padding (circular padding)
before applying the kernel. This layer keeps the semantics of `Flux.Conv` while
enabling periodic boundary conditions.
"""
struct PeriodicConv1D{C}
    conv::C
    pad_left::Int
    pad_right::Int
end

Functors.@functor PeriodicConv1D

"""
    PeriodicConv1D(in_ch, out_ch; kernel=3, stride=1, dilation=1, pad=:same, init=Flux.glorot_uniform, use_bias=true)

Convenience constructor that creates a `Flux.Conv` internally and augments it
with periodic padding.
"""
function PeriodicConv1D(in_ch::Integer, out_ch::Integer;
                        kernel::Integer=3, stride::Integer=1, dilation::Integer=1,
                        pad::Union{Symbol,Tuple,Int}=:same, init=Flux.glorot_uniform,
                        use_bias::Bool=true)
    conv = Flux.Conv((kernel,), in_ch=>out_ch;
                     stride=(stride,), dilation=(dilation,),
                     pad=(0,), bias=use_bias, init=init)
    left, right = resolve_padding(pad, kernel, dilation)
    return PeriodicConv1D(conv, left, right)
end

"""
    (layer::PeriodicConv1D)(x)

Applies periodic padding followed by the wrapped convolution.
"""
function (layer::PeriodicConv1D)(x)
    padded = periodic_pad(x, layer.pad_left, layer.pad_right)
    return layer.conv(padded)
end

"""
    periodic_pad(x, left, right)

Construct circular padding for the spatial (first) dimension of a tensor with
layout `(length, channels, batch)`.
"""
function periodic_pad(x, left::Integer, right::Integer)
    (left == 0 && right == 0) && return x
    L = size(x, 1)
    left = max(left, 0)
    right = max(right, 0)
    parts = Tuple{}
    if left > 0
        parts = (view(x, L - left + 1:L, :, :),)
    end
    parts = (parts..., x)
    if right > 0
        parts = (parts..., view(x, 1:right, :, :))
    end
    return cat(parts...; dims=1)
end

"""
    resolve_padding(pad, kernel, dilation)

Normalizes padding specifications. Returns `(left, right)` integers.
"""
function resolve_padding(pad::Symbol, kernel::Integer, dilation::Integer)
    pad === :same || error("Unsupported symbolic padding $pad")
    total = dilation * (kernel - 1)
    left = fld(total, 2)
    right = total - left
    return left, right
end

function resolve_padding(pad::Tuple{Vararg{Int}}, kernel::Integer, dilation::Integer)
    length(pad) == 2 || error("Padding tuple must have two entries for 1D data")
    return pad[1], pad[2]
end

function resolve_padding(pad::Integer, kernel::Integer, dilation::Integer)
    return pad, pad
end
