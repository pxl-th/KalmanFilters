module UnscentedKalmanFilter
export UKFState, predict!, update!

using LinearAlgebra: I, dot, diagm

include("SigmaPoints.jl")
include("Helpers.jl")
using .SigmaPoints: MerweScaled, calculate_σ_points, num_σ_points
using .Helpers: default_mean_f, default_add_x, default_residual

"""
```julia
UKFState(
    ;dim_x::Int64, dim_z::Int64, σ_parameters::MerweScaled,
    add_x::Function = default_add_x,
    residual_x::Function = default_residual,
    residual_z::Function = default_residual,
    mean_x::Function = default_mean_f,
    mean_z::Function = default_mean_f
)
```

State of the Unscented Kalman Filter.

# Parameters
- `dim_x::Int64`: State dimension
- `dim_z::Int64`: Measurement dimension
- `x::Array{Float64, 1}`: State of the filter
- `P::Array{Float64, 2}`: Covariance matrix of the filter.
- `K::Array{Float64, 2}`: Kalman gain of the filter.
- `S::Array{Float64, 2}`: System uncertainty.
- `S_inv::Array{Float64, 2}`: Inverse of system uncertainty.
- `σ_parameters::MerweScaled`: Parameters for Van der Merwe sigma points.
- `σ_fx::Array{Float64, 2}`: Sigma points after `fx`.
- `σ_hx::Array{Float64, 2}`: Sigma points after `hx`.
- `add_x::Function`: Function to calculate addition between two states.
- `residual_x::Function`: Function to calculate residual between two states.
- `residual_z::Function`: Function to calculate residual between state and measurement.
- `mean_x::Function`: Function to calculate mean between Σ-points and weights
    after passing them through `fx`.
- `mean_z::Function`: Function to calculate mean between Σ-points and weights
    after passing them through `hx`.
"""
struct UKFState
    dim_x::Int64
    dim_z::Int64

    x::Array{Float64, 1}
    P::Array{Float64, 2}

    K::Array{Float64, 2}
    y::Array{Float64, 1}

    S::Array{Float64, 2}
    S_inv::Array{Float64, 2}

    σ_parameters::MerweScaled
    σ_fx::Array{Float64, 2}
    σ_hx::Array{Float64, 2}

    add_x::Function
    residual_x::Function
    residual_z::Function
    mean_x::Function
    mean_z::Function
end

function UKFState(
    ;dim_x::Int64, dim_z::Int64, σ_parameters::MerweScaled,
    add_x::Function = default_add_x,
    residual_x::Function = default_residual,
    residual_z::Function = default_residual,
    mean_x::Function = default_mean_f,
    mean_z::Function = default_mean_f
)
    x = zeros(Float64, dim_x)
    P = Matrix{Float64}(I, dim_x, dim_x)
    K = zeros(Float64, dim_x, dim_z)
    y = zeros(Float64, dim_z)
    σ_fx = zeros(Float64, num_σ_points(σ_parameters), dim_x)
    σ_hx = zeros(Float64, num_σ_points(σ_parameters), dim_z)
    S = zeros(Float64, dim_z, dim_z)
    S_inv = zeros(Float64, dim_z, dim_z)

    UKFState(
        dim_x, dim_z,
        x, P, K, y, S, S_inv,
        σ_parameters, σ_fx, σ_hx,
        add_x, residual_x, residual_z, mean_x, mean_z
    )
end

function Base.show(io::IO, ukf::UKFState)
    print(
        io, "UKF State:\n",
        "dim_x: ", repr(MIME("text/plain"), ukf.dim_x, context=io), "\n",
        "dim_z: ", repr(MIME("text/plain"), ukf.dim_z, context=io), "\n",
        "x: ", repr(MIME("text/plain"), ukf.x, context=io), "\n",
        "P: ", repr(MIME("text/plain"), ukf.P, context=io), "\n",
        "K: ", repr(MIME("text/plain"), ukf.K, context=io), "\n",
        "y: ", repr(MIME("text/plain"), ukf.y, context=io), "\n",
        "S: ", repr(MIME("text/plain"), ukf.S, context=io), "\n",
        "S_inv: ", repr(MIME("text/plain"), ukf.S_inv, context=io), "\n",
        "σ_parameters: ", repr(MIME("text/plain"), ukf.σ_parameters, context=io), "\n",
        "σ_fx: ", repr(MIME("text/plain"), ukf.σ_fx, context=io), "\n",
        "σ_hx: ", repr(MIME("text/plain"), ukf.σ_hx, context=io), "\n",
        "add_x: ", repr(MIME("text/plain"), ukf.add_x, context=io), "\n",
        "residual_x: ", repr(MIME("text/plain"), ukf.residual_x, context=io), "\n",
        "residual_z: ", repr(MIME("text/plain"), ukf.residual_z, context=io), "\n",
        "mean_x: ", repr(MIME("text/plain"), ukf.mean_x, context=io), "\n",
        "mean_z: ", repr(MIME("text/plain"), ukf.mean_z, context=io), "\n",
    )
end

"""
Pass σ-points for the current UKF state `(x, P)` through `fx` function.
"""
function calculate_process_σ!(
    ;ukf_state::UKFState, δt::Float64, fx::Function, fx_args...
)
    σ_points = calculate_σ_points(
        σ_parameters=ukf_state.σ_parameters, x=ukf_state.x, P=ukf_state.P
    )
    @inbounds for i = 1:size(σ_points, 1)
        ukf_state.σ_fx[i, :] = fx(x=σ_points[i, :], δt=δt; fx_args...)
    end
end

"""
Calculate cross-variance between state and measurement.
"""
function cross_variance(
    ;ukf::UKFState, x::Array{Float64, 1}, z::Array{Float64, 1}
)::Array{Float64, 2}
    Δx = ukf.residual_x(ukf.σ_fx, x)
    Δz = ukf.residual_z(ukf.σ_fx, z)
    Δx' * diagm(ukf.σ_parameters.Σ_w) * Δz
end

"""
```julia
unscented_transform(
    ;ukf=UKFState, Σ::Array{Float64, 2}, residual_f::Function,
    noise_cov::Union{Nothing, Array{Float64, 2}} = nothing,
)::Tuple{Array{Float64, 1}, Array{Float64, 2}}
```

Unscented transformation of Σ-points and their weights.
This is used to calculate new mean `x` and covariance `P`
for the state of Unscented Kalman Filter.

# Arguments
- `ukf::UKFState`: State of the UKF filter which contains weights for `Σ`.
- `Σ::Array{Float64, 2}`: `[2x+1, n]` matrix of Σ-points
- `residual_f::Function`: Function to calculate residual between `y` and `x`.
- `mean_f::Function`: Function to calculate mean between Σ points and weights.
- `noise_cov::Union{Nothing, Array{Float64, 2}} = nothing`:
    Optional noise, that will be added to final covariance matrix.
    Its size should match that of a covariance matrix.
"""
function unscented_transform(
    ;ukf=UKFState, Σ::Array{Float64, 2},
    residual_f::Function, mean_f::Function,
    noise_cov::Union{Nothing, Array{Float64, 2}} = nothing,
)::Tuple{Array{Float64, 1}, Array{Float64, 2}}
    x = mean_f(Σ, ukf.σ_parameters.m_w)
    Y = residual_f(Σ, x)
    P = Y' * diagm(ukf.σ_parameters.Σ_w) * Y
    if isa(noise_cov, Array{Float64, 2})
        P += noise_cov
    end
    x, P
end

"""
```julia
predict!(
    ;ukf::UKFState, δt::Float64, Q::Union{Nothing, Array{Float64, 2}},
    fx::Function, fx_args...
)
```

Perform prediction of the Unscented Kalman Filter (UKF).
This updates `x` and `P` of the `ukf`
to contain predicted state and covariance.

# Arguments
- `ukf::UKFState`: State of the UKF on which to perform prediction.
- `δt::Float64`: Time delta.
- `Q::Union{Nothing, Array{Float64, 2}}`: Process noise.
- `fx::Function(;x::Array{Float64, 1}, δt::Float64, ...)`:
    State transition function.
    Should accept keyword-only arguments, with mandatory `x` and `δt`.
    All other arguments may serve as control input or provide additional info.
- `fx_args...`: Arguments to pass to `fx` function.
    Must contain at least `x` and `δt` arguments.
"""
function predict!(
    ;ukf::UKFState, δt::Float64, Q::Union{Nothing, Array{Float64, 2}},
    fx::Function, fx_args...
)
    calculate_process_σ!(ukf_state=ukf, δt=δt, fx=fx; fx_args...)
    ukf.x[:], ukf.P[:] = unscented_transform(
        ukf=ukf, Σ=ukf.σ_fx, noise_cov=Q,
        residual_f=ukf.residual_x, mean_f=ukf.mean_x
    )
    ukf.σ_fx[:] = calculate_σ_points(
        σ_parameters=ukf.σ_parameters, x=ukf.x, P=ukf.P
    )
end

"""
```julia
update!(
    ;ukf::UKFState, z::Array{Float64, 1}, R::Union{Nothing, Array{Float64, 2}},
    hx::Function, hx_args...
)
```

Perform update step of the Unscented Kalman Filter (UKF)
given measurements.
This updates `x` and `P` of the `ukf_state`
to contain updated state and covariance.

# Arguments
- `ukf::UKFState`: State of the UKF on which to perform update step.
- `z::Array{Float64, 1}`: Measurements.
- `R::Union{Nothing, Array{Float64, 2}}`: Measurement noise.
- `hx::Function`:
    Measurement function.
    Should accept keyword-only arguments with mandatory `x` argument for state.
    Given state `x` of UKF it must transform it to the respective measurement.
- `hx_args...`: Arguments that will be passed to `hx` function.
    Should contain at least `x` argument.
"""
function update!(
    ;ukf::UKFState, z::Array{Float64, 1}, R::Union{Nothing, Array{Float64, 2}},
    hx::Function, hx_args...
)
    @inbounds for i = 1:size(ukf.σ_fx, 1)
        ukf.σ_hx[i, :] = hx(x=ukf.σ_fx[i, :]; hx_args...)
    end

    zp, ukf.S[:] = unscented_transform(
        ukf=ukf, Σ=ukf.σ_hx, noise_cov=R,
        residual_f=ukf.residual_z, mean_f=ukf.mean_z
    )
    ukf.S_inv[:] = inv(ukf.S)

    cv = cross_variance(ukf=ukf, x=zp, z=z)
    ukf.K[:] = cv * ukf.S_inv
    ukf.y[:] = ukf.residual_z(z, zp)

    ukf.x[:] = ukf.add_x(ukf.x, ukf.K * ukf.y)
    ukf.P[:] = ukf.P - ukf.K * ukf.S * ukf.K'
end

end
