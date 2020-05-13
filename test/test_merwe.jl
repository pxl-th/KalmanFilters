@testset "Test number of weights" begin
    for n = 1:5
        σ_parameters = MerweScaled(n=n, α=1.0, β=0.0, κ=3.0 - n)
        target_num = 2n + 1
        @test num_σ_points(σ_parameters) == target_num
        @test size(σ_parameters.m_w, 1) == target_num
        @test size(σ_parameters.Σ_w, 1) == target_num
    end
end

@testset "Test that weights are normalized" begin
    for n = 1:5
        σ_parameters = MerweScaled(n=n, α=1.0, β=0.0, κ=3.0 - n)
        @test sum(σ_parameters.m_w) - 1.0 < 1e-6
        @test sum(σ_parameters.Σ_w) - 1.0 < 1e-6
    end
end
