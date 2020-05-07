include("UnscentedKalmanFilter.jl")
using LinearAlgebra: dot, I
using .UnscentedKalmanFilter
using .UnscentedKalmanFilter.SigmaPoints

function normalize_angle(α::Float64)::Float64
    α = α % (2 * π)
    if α > π
        α -= 2 * π
    end
    α
end

function state_residual(
    ;y::Array{Float64, 1}, x::Array{Float64, 1}
)::Array{Float64, 1}
    r = y - x
    r[3] = normalize_angle(r[3])
    r
end

function state_add(
    ;y::Array{Float64, 1}, x::Array{Float64, 1}
)::Array{Float64, 1}
    r = y + x
    r[3] = normalize_angle(r[3])
    r
end

function state_mean(
    ;Σ::Array{Float64, 2}, Σ_w::Array{Float64, 1}
)::Array{Float64, 1}
    Float64[
        dot(Σ[:, 1], Σ_w);
        dot(Σ[:, 2], Σ_w);
        atan(dot(sin.(Σ[:, 3]), Σ_w), dot(cos.(Σ[:, 3]), Σ_w))
    ]
end

function hx(;x::Array{Float64, 1})::Array{Float64, 1}
    x
end

function fx(
    ;x::Array{Float64, 1}, δt::Float64,
    μ::Array{Float64, 1}, wheelbase::Float64
)::Array{Float64, 1}
    ν, α = μ
    if abs(ν) < 0.3
        return x
    end

    θ = x[3]
    s = ν * δt
    if abs(α) < 0.017
        return x + Float64[s * cos(θ); s * sin(θ); 0.0]
    end

    β = (s / wheelbase) * tan(α)
    r = s / β
    x + [-r * sin(θ) + r * sin(θ + β); r * cos(θ) - r * cos(θ + β); β]
end

const wheelbase = 2.9591

σ_parameters = MerweScaled(n=3, α=1e-3, β=2.0, κ=0.0, residual_x=state_residual)
ukf = UKFState(
    dim_x=3, dim_z=3,
    σ_parameters=σ_parameters,
    add_x=state_add, residual_x=state_residual, residual_z=state_residual
)
ukf.x[:] = [0;0;1.5588782]
ukf.P[:] *= 0.1

δt = 1.0 / 50.0
μ = [1.32280445e+01, 9.72143200e-04]
z = [-0.4884202, 1.03592605, 1.55887818]
Q = [[4e-12 4e-10 2e-8]; [4e-10 4e-8 2e-6]; [2e-8 2e-6 1e-4]]
R = Matrix{Float64}(I, 3, 3) * 0.1

println("Raw")
println(ukf.x)
println(ukf.P)

predict!(ukf_state=ukf, δt=δt, Q=Q, fx=fx, μ=μ, wheelbase=wheelbase)
println("Predict")
println(ukf.x)
println(ukf.P)

update!(ukf=ukf, z=z, R=R, hx=hx)
println("Update")
println(ukf.x)
println(ukf.P)
