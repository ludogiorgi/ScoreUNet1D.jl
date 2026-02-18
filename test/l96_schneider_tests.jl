using Test

include(joinpath(@__DIR__, "..", "scripts", "L96", "lib", "SchneiderTopology.jl"))
using .L96SchneiderTopology

@testset "L96 Schneider twisted fast topology" begin
    K = 5
    J = 4

    @test validate_twisted_mapping(K, J)

    y = Array{Int}(undef, J, K)
    @inbounds for k in 1:K
        for j in 1:J
            y[j, k] = 1000 * k + j
        end
    end

    jn, kn = fast_neighbor_jk(J, 1, 1, K, J)
    @test (jn, kn) == (1, 2)
    @test y[jn, kn] == 2001
    @test y[jn, kn] != y[1, 1]

    jn2, kn2 = fast_neighbor_jk(J, K, 1, K, J)
    @test (jn2, kn2) == (1, 1)
    @test y[jn2, kn2] == 1001
end

@testset "L96 Schneider coupling helpers" begin
    x = Float32[1, 2, 3]
    x_up = broadcast_X_to_fast(x, 2)
    @test x_up == Float32[1, 1, 2, 2, 3, 3]

    y_flat = Float32[1, 3, 2, 4, 5, 7]
    y_bar = blockmean_fast_to_slow(y_flat, 2)
    @test y_bar == Float32[2, 3, 6]
end

@testset "L96 Schneider legacy adapter" begin
    cfg = L96SchneiderScoreConfig(
        K=4,
        J=3,
        slow_base_channels=8,
        fast_base_channels=8,
        slow_channel_multipliers=[1],
        fast_channel_multipliers=[1],
        norm_type=:group,
    )
    model = build_l96_schneider_legacy_model(cfg)
    state = rand(Float32, 4, 4, 2) # (K, J+1, B)
    score = model(state)
    @test size(score) == size(state)
end
