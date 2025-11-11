using HDF5
using Statistics
using Random
using Base.Threads

struct DataStats
    mean::Array{Float32,2}  # (channels, length)
    std::Array{Float32,2}
end

struct NormalizedDataset
    data::Array{Float32,3}  # (length, channels, batch)
    stats::DataStats
end

Base.length(dataset::NormalizedDataset) = size(dataset.data, 3)

sample_length(dataset::NormalizedDataset) = size(dataset.data, 1)
num_channels(dataset::NormalizedDataset) = size(dataset.data, 2)

"""
    load_hdf5_dataset(path; dataset_key=nothing, normalize=true, samples_orientation=:rows)

Loads the dataset stored in `data/` as an HDF5 tensor, reshapes it to
`(length, channels, batch)`, and optionally normalizes it. Set
`samples_orientation = :columns` when each column represents an independent
sample (as in the KS dataset).
"""
function load_hdf5_dataset(path::AbstractString;
                           dataset_key::Union{Nothing,String}=nothing,
                           normalize::Bool=true,
                           samples_orientation::Symbol=:rows)
    raw = read_hdf5_array(path, dataset_key)
    raw = orient_samples(raw, samples_orientation)
    tensor = ensure_tensor_layout(raw)
    tensor = Array{Float32,3}(tensor)
    return normalize ? normalize_dataset(tensor) :
        NormalizedDataset(tensor,
                          DataStats(zeros(Float32, size(tensor,2), size(tensor,1)),
                                    ones(Float32, size(tensor,2), size(tensor,1))))
end

function orient_samples(data, orientation::Symbol)
    orientation === :rows && return data
    if orientation === :columns
        nd = ndims(data)
        nd >= 2 || throw(ArgumentError("Column-oriented data must have at least 2 dims"))
        order = (2, 1, (3:nd)...)
        return permutedims(data, order)
    else
        throw(ArgumentError("Unsupported samples_orientation: $orientation"))
    end
end

"""
    read_hdf5_array(path, dataset_key)

Reads the requested dataset from an HDF5 file. When `dataset_key` is `nothing`
the first dataset found in the file hierarchy is used.
"""
function read_hdf5_array(path::AbstractString, dataset_key::Union{Nothing,String})
    h5open(path, "r") do file
        if dataset_key === nothing
            dset = locate_first_dataset(file)
            dset === nothing && error("No datasets found in $path")
            return read(dset)
        else
            return read(file[dataset_key])
        end
    end
end

function locate_first_dataset(group)
    for key in keys(group)
        child = group[key]
        if child isa HDF5.Dataset
            return child
        elseif child isa HDF5.Group
            result = locate_first_dataset(child)
            result !== nothing && return result
        end
    end
    return nothing
end

"""
    ensure_tensor_layout(data)

Converts loaded data to `(length, channels, batch)` layout expected by Flux.
"""
function ensure_tensor_layout(data::AbstractArray)
    nd = ndims(data)
    if nd == 1
        data = reshape(data, 1, length(data))
    end
    if nd == 2
        samples, length_ = size(data)
        data = reshape(data, samples, 1, length_)
    elseif nd == 3
        samples, channels, length_ = size(data)
        data = reshape(data, samples, channels, length_)
    else
        throw(ArgumentError("Unsupported data dimensions: $nd"))
    end
    return permutedims(data, (3, 2, 1))
end

"""
    normalize_dataset(tensor)

Normalizes a `(length, channels, batch)` tensor and returns a `NormalizedDataset`.
"""
function normalize_dataset(tensor::Array{Float32,3})
    stats = compute_stats(tensor)
    normalized = apply_stats(tensor, stats)
    return NormalizedDataset(normalized, stats)
end

"""
    compute_stats(tensor)

Computes per-feature mean and std using CPU threads.
"""
function compute_stats(tensor::Array{Float32,3})
    L, C, B = size(tensor)
    flat = reshape(permutedims(tensor, (3, 2, 1)), B, :)
    means, stds = feature_stats(flat)
    mean_tensor = reshape(means, C, L)
    std_tensor = reshape(stds, C, L)
    return DataStats(mean_tensor, std_tensor)
end

function feature_stats(flat::Array{Float32,2})
    n, f = size(flat)
    means = zeros(Float32, f)
    stds = zeros(Float32, f)
    Threads.@threads for j in 1:f
        μ = zero(Float32)
        @inbounds for i in 1:n
            μ += flat[i, j]
        end
        μ /= max(n, 1)
        σ2 = zero(Float32)
        @inbounds for i in 1:n
            diff = flat[i, j] - μ
            σ2 += diff * diff
        end
        means[j] = μ
        stds[j] = sqrt(σ2 / max(n, 1) + eps(Float32))
    end
    return means, stds
end

"""
    apply_stats(tensor, stats)

Normalize using the provided statistics.
"""
function apply_stats(tensor::Array{Float32,3}, stats::DataStats)
    L, C, B = size(tensor)
    flat = reshape(permutedims(tensor, (3, 2, 1)), B, :)
    mean_vec = reshape(stats.mean, 1, :)
    std_vec = reshape(stats.std, 1, :)
    normalized = (flat .- mean_vec) ./ std_vec
    normalized = permutedims(reshape(normalized, B, C, L), (3, 2, 1))
    return normalized
end

"""
    denormalize_sample(dataset::NormalizedDataset, sample)

Converts a normalized sample back to the original data space.
"""
function denormalize_sample(dataset::NormalizedDataset, sample::Array{Float32,2})
    stats = dataset.stats
    mean = permutedims(stats.mean, (2, 1))
    std = permutedims(stats.std, (2, 1))
    return sample .* std .+ mean
end

"""
    get_batch(dataset, idxs)

Creates a contiguous batch array for the provided indices.
"""
function get_batch(dataset::NormalizedDataset, idxs::AbstractVector{<:Integer})
    return Array(dataset.data[:, :, idxs])
end
