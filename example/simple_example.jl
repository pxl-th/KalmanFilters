using LinearAlgebra: I, diagm
using KalmanFilters.UnscentedKalmanFilter
using KalmanFilters.SigmaPoints

function fx(
    ;x::AbstractArray{Float64, 1}, δt::Float64,
)::AbstractArray{Float64, 1}
    F = Float64[[1 δt 0 0];[0 1 0 0];[0 0 1 δt];[0 0 0 1]]
    F * x
end

hx(;x::AbstractArray{Float64, 1})::AbstractArray{Float64, 1} = x[[1, 3]]

function main()
    δt = 1.0
    σ_parameters = MerweScaled(n=4, α=0.1, β=2.0, κ=-1.0)
    ukf = UKFState(dim_x=4, dim_z=2, σ_parameters=σ_parameters)
    ukf.x .= [0, 1, 0, 1]
    ukf.P .*= 0.01
    Q = nothing
    R = diagm([0.01 ^ 2, 0.01 ^ 2])

    measurements = vcat([Float64[i i] for i = 1:10]...)
    for i = 1:size(measurements, 1)
        predict!(ukf=ukf, δt=δt, Q=Q, fx=fx)
        update!(ukf=ukf, z=measurements[i, :], R=R, hx=hx)
    end
    println(ukf.x)
end

main()
