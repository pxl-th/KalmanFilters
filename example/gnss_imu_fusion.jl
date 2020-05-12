using DelimitedFiles: readdlm
using LinearAlgebra: dot, I

#= using Plots: plot, plot!, gr, gui =#
#= gr() =#

using KalmanFilters.UnscentedKalmanFilter
using KalmanFilters.SigmaPoints

function normalize_angle(α::Float64)::Float64
    m = 2π
    α = α > 0 ? α % m : m + (α % m)
    if α > π
        α -= m
    end
    α
end

function state_residual(
    y::Array{Float64, 1}, x::Array{Float64, 1}
)::Array{Float64, 1}
    r = y - x
    r[3] = normalize_angle(r[3])
    r
end

function state_residual(
    y::Array{Float64, 2}, x::Array{Float64, 1},
)::Array{Float64, 2}
    R = y .- x'
    R[:, 3] = normalize_angle.(R[:, 3])
    R
end

function state_add(
    y::Array{Float64, 1}, x::Array{Float64, 1}
)::Array{Float64, 1}
    r = y + x
    r[3] = normalize_angle(r[3])
    r
end

function state_mean(
    Σ::Array{Float64, 2}, m_w::Array{Float64, 1}
)::Array{Float64, 1}
    Float64[
        dot(Σ[:, 1], m_w);
        dot(Σ[:, 2], m_w);
        atan(dot(sin.(Σ[:, 3]), m_w), dot(cos.(Σ[:, 3]), m_w))
    ]
end

hx(;x::Array{Float64, 1})::Array{Float64, 1} = x

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

load_from_txt(path::String) = readdlm(path, ' ', Float64, '\n')

function main()
    base_path = raw"C:\Users\tonys\projects\carla-dataset\high-quality\recording-2020-05-01-13-52-24-005689"
    positions_time = load_from_txt(joinpath(base_path, "gnss", "timestamp"))
    positions = load_from_txt(joinpath(base_path, "gnss", "enu"))[:, 1:2]

    imu_time = load_from_txt(joinpath(base_path, "imu", "timestamp"))
    compass = load_from_txt(joinpath(base_path, "imu", "compass"))
    speed = load_from_txt(joinpath(base_path, "imu", "speed"))
    steer = load_from_txt(joinpath(base_path, "imu", "steer"))
    # Convert data
    measurement_noise = 1.5
    compass *= π / 180
    steer *= -70.0 * π / 180
    wheelbase = 2.9591
    positions += randn(size(positions)...) * measurement_noise
    println("Loaded data")

    σ_parameters = MerweScaled(n=3, α=1e-3, β=2.0, κ=0.0, residual_x=state_residual)
    ukf = UKFState(
        dim_x=3, dim_z=3, σ_parameters=σ_parameters,
        add_x=state_add,
        mean_x=state_mean, mean_z=state_mean,
        residual_x=state_residual, residual_z=state_residual
    )
    ukf.x[:] = [positions[1, 1]; positions[1, 2]; compass[1]]
    ukf.P[:] *= 0.1

    δt = 1.0 / 50
    μ = zeros(Float64, 2)
    z = zeros(Float64, 3)
    R = Matrix{Float64}(I, 3, 3) * measurement_noise
    Q = Float64[[4e-12 4e-10 2e-8]; [4e-10 4e-8 2e-6]; [2e-8 2e-6 1e-4]]

    track = zeros(Float64, size(imu_time, 1), 2)
    cmp = IOContext(stdout, :compact => true, :limit => true)
    println(cmp, ukf)

    pos_i::Int64 = 1
    can_update::Bool = false
    for (i, t) = enumerate(imu_time)
        μ[1] = speed[i]
        μ[2] = steer[i]
        predict!(ukf=ukf, δt=δt, Q=Q, fx=fx, μ=μ, wheelbase=wheelbase)

        can_update = pos_i < size(positions_time, 1) && (
            abs(t - positions_time[pos_i]) < δt
            || t > positions_time[pos_i]
        )
        if can_update
            z[1:2] = positions[pos_i, :]
            z[3] = compass[i]
            update!(ukf=ukf, z=z, R=R, hx=hx)
            pos_i += 1
        end
        track[i, :] = ukf.x[1:2]
    end
    println(cmp, ukf)

    #= plot(positions[:, 1], positions[:, 2], label="Noised GNSS") =#
    #= plot!(track[:, 1], track[:, 2], label="Fused") =#
    #= gui() =#
end

main()
