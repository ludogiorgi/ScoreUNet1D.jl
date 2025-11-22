using Base.Threads
using Flux
using LinearAlgebra
import CUDA

abstract type ExecutionDevice end

struct CPUDevice <: ExecutionDevice end

struct GPUDevice <: ExecutionDevice
    ids::Vector{Int}
end

function GPUDevice()
    CUDA.has_cuda() || error("device=GPU requested but CUDA is not available")
    count = length(CUDA.devices())
    count > 0 || error("No CUDA devices detected")
    return GPUDevice(collect(0:(count - 1)))
end

function select_device(name::AbstractString)
    up = uppercase(name)
    if up == "GPU"
        return GPUDevice()
    else
        return CPUDevice()
    end
end

is_gpu(dev::ExecutionDevice) = dev isa GPUDevice

function activate_device!(::CPUDevice)
    try
        BLAS.set_num_threads(Sys.CPU_THREADS)
    catch err
        @warn "Failed to set BLAS threads" error=err
    end
    return nothing
end

function activate_device!(dev::GPUDevice)
    CUDA.device!(first(dev.ids))
    CUDA.allowscalar(false)
    return nothing
end

function move_model(model, ::CPUDevice)
    return Flux.fmap(cpu, model)
end

function move_model(model, dev::GPUDevice; device_index::Int=1)
    idx = clamp(device_index, 1, length(dev.ids))
    CUDA.device!(dev.ids[idx])
    return Flux.fmap(CUDA.cu, model)
end

function move_array(arr, ::CPUDevice; device_index::Int=1)
    return arr
end

function move_array(arr, dev::GPUDevice; device_index::Int=1)
    idx = clamp(device_index, 1, length(dev.ids))
    CUDA.device!(dev.ids[idx])
    return CUDA.cu(arr)
end

gpu_count(dev::GPUDevice) = length(dev.ids)
