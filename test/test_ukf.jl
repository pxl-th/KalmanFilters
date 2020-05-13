using LinearAlgebra: I, diagm
import KalmanFilters.UnscentedKalmanFilter: unscented_transform

@testset """
Test unscented transform gives original mean and covariance
if σ-points not changed.
""" begin
    for n = 1:5
        x = Float64[1.0 for i = 1:n]
        P = Matrix{Float64}(I, n, n) * 5.0
        σ_parameters = MerweScaled(n=n, α=1e-3, β=2.0, κ=3.0 - n)
        ukf = UKFState(dim_x=n, dim_z=1, σ_parameters=σ_parameters)

        Σ = calculate_σ_points(σ_parameters=σ_parameters, x=x, P=P)
        xn, Pn = unscented_transform(
            ukf=ukf, Σ=Σ, residual_f=ukf.residual_x, mean_f=ukf.mean_x,
        )
        @test all(abs.(x - xn) .< 1e-6)
        @test all(abs.(P - Pn) .< 1e-6)
    end
end

@testset "Test linear" begin
    function fx(
        ;x::AbstractArray{Float64, 1}, δt::Float64,
    )::AbstractArray{Float64, 1}
        F = Float64[[1 δt 0 0];[0 1 0 0];[0 0 1 δt];[0 0 0 1]]
        F * x
    end

    hx(;x::AbstractArray{Float64, 1})::AbstractArray{Float64, 1} = x[[1, 3]]

    δt = 1.0
    σ_parameters = MerweScaled(n=4, α=0.1, β=2.0, κ=-1.0)
    ukf = UKFState(dim_x=4, dim_z=2, σ_parameters=σ_parameters)
    ukf.x .= [0, 0, 0, 0]
    ukf.P .*= 0.01
    Q = nothing
    R = diagm([0.01 ^ 2, 0.01 ^ 2])

    n = 10
    target_state = Float64[n, 1., n, 1.]
    measurements = vcat([Float64[i i] for i = 1:n]...)
    for i = 1:n
        predict!(ukf=ukf, δt=δt, Q=Q, fx=fx)
        update!(ukf=ukf, z=measurements[i, :], R=R, hx=hx)
    end
    @test all(abs.(ukf.x - target_state) .< 1e-2)
end
