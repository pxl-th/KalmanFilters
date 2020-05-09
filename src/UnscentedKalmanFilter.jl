module UnscentedKalmanFilter
include("SigmaPoints.jl")

using InteractiveUtils
using LinearAlgebra: I, dot, diagm
using .SigmaPoints: MerweScaled, calculate_σ_points, num_σ_points

export UKFState, unscented_transform, predict!, update!

"""
State of the Unscented Kalman Filter

# Parameters
- `dim_x::Int64`: State dimension
- `dim_z::Int64`: Measurement dimension
- `x::Array{Float64, 1}`: State of the filter
- `P::Array{Float64, 2}`: Covariance matrix of the filter.
- `K::Array{Float64, 2}`: Kalman gain of the filter.
- `S::Array{Float64, 2}`: System uncertainty.
- `S_inv::Array{Float64, 2}`: Inverse of system uncertainty.
"""
struct UKFState
    dim_x::Int64
    dim_z::Int64

    x::Array{Float64, 1}
    P::Array{Float64, 2}

    K::Array{Float64, 2}
    y::Array{Float64, 1}

    σ_parameters::MerweScaled
    σ_fx::Array{Float64, 2}
    σ_hx::Array{Float64, 2}

    S::Array{Float64, 2}
    S_inv::Array{Float64, 2}

    add_x::Function
    residual_x::Union{Nothing, Function}
    residual_z::Union{Nothing, Function}
    mean_f::Union{Nothing, Function}
end

function UKFState(
    ;dim_x::Int64, dim_z::Int64, σ_parameters::MerweScaled,
    add_x::Union{Nothing, Function} = nothing,
    residual_x::Union{Nothing, Function} = nothing,
    residual_z::Union{Nothing, Function} = nothing,
    mean_f::Union{Nothing, Function} = nothing
)
    x = zeros(Float64, dim_x)
    P = Matrix{Float64}(I, dim_x, dim_x)
    K = zeros(Float64, dim_x, dim_z)
    y = zeros(Float64, dim_z)
    σ_fx = zeros(Float64, num_σ_points(σ_parameters), dim_x)
    σ_hx = zeros(Float64, num_σ_points(σ_parameters), dim_z)
    S = zeros(Float64, dim_z, dim_z)
    S_inv = zeros(Float64, dim_z, dim_z)
    if isa(add_x, Nothing)
        add_x = +
    end
    UKFState(
        dim_x, dim_z,
        x, P, K, y,
        σ_parameters, σ_fx, σ_hx,
        S, S_inv,
        add_x, residual_x, residual_z, mean_f
    )
end

function Base.show(io::IO, ukf::UKFState)
    print(
        io,
        "UKF State:\n",
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
        "mean_f: ", repr(MIME("text/plain"), ukf.mean_f, context=io), "\n",
    )
end

function compute_process_σ!(
    ;ukf_state::UKFState, δt::Float64, fx::Function, fx_args...
)
    σ_points = calculate_σ_points(
        σ_parameters=ukf_state.σ_parameters, x=ukf_state.x, P=ukf_state.P
    )
    for i = 1:size(σ_points, 1)
        ukf_state.σ_fx[i, :] = fx(x=σ_points[i, :], δt=δt; fx_args...)
    end
end

function outer(x::Array{Float64, 1}, y::Array{Float64, 1})::Array{Float64, 2}
    rows = size(x, 1)
    cols = size(y, 1)
    result = Matrix{Float64}(undef, rows, cols)
    for (i, j) = Iterators.product(1:rows, 1:cols)
        result[i, j] = x[i] * y[j]
    end
    result
end

"""
Calculate cross-variance between filter's state and measurement.
"""
function cross_variance(
    ;ukf::UKFState, x::Array{Float64, 1}, z::Array{Float64, 1}
)::Array{Float64, 2}
    cv = zeros(Float64, size(ukf.σ_fx, 2), size(ukf.σ_hx, 2))
    for i = 1:size(ukf.σ_fx, 1)
        δx = ukf.residual_x(y=ukf.σ_fx[i, :], x=x)
        δz = ukf.residual_z(y=ukf.σ_hx[i, :], x=z)
        cv += ukf.σ_parameters.Σ_w[i] * outer(δx, δz)
    end
    cv
end

"""
```julia
unscented_transform(
    ;Σ::Array{Float64, 2}, Σ_w::Array{Float64, 1}, m_w::Array{Float64, 1},
    noise_cov::Union{Nothing, Array{Float64, 2}} = nothing,
    mean_f::Union{Nothing, Function} = nothing,
    residual_x::Union{Nothing, Function} = nothing
)::Tuple{Array{Float64, 1}, Array{Float64, 2}}
```

Unscented transformation of Σ-points and their weights.
This is used to calculate new mean `x` and covariance `P`
for the state of Unscented Kalman Filter.

# Arguments
- `Σ::Array{Float64, 2}`: `[2x+1, n]` matrix of Σ-points
- `Σ_w::Array{Float64, 1}`: Weights for each σ-point for the covariance.
- `m_w::Array{Float64, 1}`: Weights for each σ-point for the mean.
- `noise_cov::Union{Nothing, Array{Float64, 2}} = nothing`:
    Optional noise, that will be added to final covariance matrix.
    Its size should match that of a covariance matrix.
- `mean_f::Union{Nothing, Function} = nothing`:
    Function to compute mean of the Σ-points and its weights `Σ_w`.
    Should accept keyword-only `Σ` and `Σ_w` arguments.
    If `nothing`, then `dot` product will be used as a default function.
- `residual_x::Union{Nothing, Function} = nothing`:
    Function to compute residual between x and y.
    Should accept keyword-only `y` and `x` arguments.
    If `nothing` then operator `.-` will be used as a default residual function.
"""
function unscented_transform(
    ;Σ::Array{Float64, 2}, Σ_w::Array{Float64, 1}, m_w::Array{Float64, 1},
    noise_cov::Union{Nothing, Array{Float64, 2}} = nothing,
    mean_f::Union{Nothing, Function}             = nothing,
    residual_x::Union{Nothing, Function}         = nothing
)::Tuple{Array{Float64, 1}, Array{Float64, 2}}
    x = isa(mean_f, Nothing) ? Σ' * m_w : mean_f(Σ=Σ, Σ_w=m_w)
    if isa(residual_x, Nothing)
        y = Σ .- x'
        P = y' * (diagm(Σ_w) * y)
    else
        σ_num, n = size(Σ)
        P = zeros(Float64, n, n)
        for i = 1:σ_num
            y = residual_x(y=Σ[i, :], x=x)
            P += outer(y, y) * Σ_w[i]
        end
    end
    if isa(noise_cov, Array{Float64, 2})
        P += noise_cov
    end
    x, P
end

"""
```julia
predict!(
    ;ukf::UKFState, δt::Float64, fx::Function, fx_args...
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
    ;ukf::UKFState, δt::Float64,
    Q::Union{Nothing, Array{Float64, 2}},
    fx::Function, fx_args...
)
    compute_process_σ!(ukf_state=ukf, δt=δt, fx=fx; fx_args...)
    ukf.x[:], ukf.P[:] = unscented_transform(
        Σ=ukf.σ_fx,
        Σ_w=ukf.σ_parameters.Σ_w, m_w=ukf.σ_parameters.m_w,
        noise_cov=Q,
        mean_f=ukf.mean_f, residual_x=ukf.residual_x
    )
    ukf.σ_fx[:] = calculate_σ_points(
        σ_parameters=ukf.σ_parameters, x=ukf.x, P=ukf.P
    )
end

"""
```julia
update!(
    ;ukf::UKFState, z::Array{Float64, 1},
    R::Union{Nothing, Array{Float64, 2}},
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
    ;ukf::UKFState, z::Array{Float64, 1},
    R::Union{Nothing, Array{Float64, 2}},
    hx::Function, hx_args...
)
    for i = 1:size(ukf.σ_fx, 1)
        ukf.σ_hx[i, :] = hx(x=ukf.σ_fx[i, :]; hx_args...)
    end

    zp, ukf.S[:] = unscented_transform(
        Σ=ukf.σ_hx,
        Σ_w=ukf.σ_parameters.Σ_w, m_w=ukf.σ_parameters.m_w,
        noise_cov=R,
        mean_f=ukf.mean_f, residual_x=ukf.residual_z
    )
    ukf.S_inv[:] = inv(ukf.S)

    cv = cross_variance(ukf=ukf, x=zp, z=z)
    ukf.K[:] = cv * ukf.S_inv
    ukf.y[:] = ukf.residual_z(y=z, x=zp)

    ukf.x[:] = ukf.add_x(ukf.x, ukf.K * ukf.y)  # TODO fix doc for add_x
    ukf.P[:] = ukf.P - ukf.K * (ukf.S * ukf.K')
end

end
