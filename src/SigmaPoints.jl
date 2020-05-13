module SigmaPoints
export MerweScaled, calculate_σ_points, num_σ_points

using LinearAlgebra: cholesky, det, Symmetric

"""
```julia
MerweScaled(
    ;n::Int64, α::Float64, β::Float64, κ::Float64, residual_x::Function = -
)
```

# Parameters for the Van der Merwe's σ-points
- `n::Int64`: Dimensionality of the state. `2n+1` will be generated
- `α::Float64`: Spread of the σ-points around the mean.
- `β::Float64`: Prior knowledge of the distribution of the mean.
- `κ::Float64`: Second scaling parameter.
- `Σ_w::Array{Float64, 1}`: Weights for each σ-point for the covariance.
- `m_w::Array{Float64, 1}`: Weights for each σ-point for the mean.
- `residual_x::Function = -`: Function to compute residual between states.
"""
struct MerweScaled
    n::Int64
    α::Float64
    β::Float64
    κ::Float64

    Σ_w::Array{Float64, 1}
    m_w::Array{Float64, 1}

    residual_x::Function
end

function MerweScaled(
    ;n::Int64, α::Float64, β::Float64, κ::Float64, residual_x::Function = -
)
    Σ_w, m_w = σ_weights(n=n, α=α, β=β, κ=κ)
    MerweScaled(n, α, β, κ, Σ_w, m_w, residual_x)
end

function Base.show(io::IO, merwe::MerweScaled)
    print(
        io, "Merwe Scaled:\n",
        "n: ", repr(MIME("text/plain"), merwe.n, context=io), "\n",
        "α: ", repr(MIME("text/plain"), merwe.α, context=io), "\n",
        "β: ", repr(MIME("text/plain"), merwe.β, context=io), "\n",
        "κ: ", repr(MIME("text/plain"), merwe.κ, context=io), "\n",
        "Σ_w: ", repr(MIME("text/plain"), merwe.Σ_w, context=io), "\n",
        "m_w: ", repr(MIME("text/plain"), merwe.m_w, context=io), "\n",
        "residual_x: ", repr(MIME("text/plain"), merwe.residual_x, context=io), "\n",
    )
end

"""
```julia
num_σ_points(σ_parameters::MerweScaled)::Int64
```

Return number of σ-points for given σ-parameters.
"""
num_σ_points(σ_parameters::MerweScaled)::Int64 = 2σ_parameters.n + 1

"""
```julia
σ_weights(
    ;n::Int64, α::Float64, β::Float64, κ::Float64
)::Tuple{Array{Float64, 1}, Array{Float64, 1}}
```
Compute weights for each σ-point for the covariance `P` and mean `x`.

# Arguments
- `n::Int64`: Dimensionality of the state. `2n+1` σ-points will be generated
- `α::Float64`: Spread of the σ-points around the mean.
- `β::Float64`: Prior knowledge of the distribution of the mean.
- `κ::Float64`: Second scaling parameter.
"""
function σ_weights(
    ;n::Int64, α::Float64, β::Float64, κ::Float64
)::Tuple{Array{Float64, 1}, Array{Float64, 1}}
    λ = α ^ 2 * (n + κ) - n
    c = 0.5 / (n + λ)

    Σ_w = [c for i = 1:2n + 1]
    m_w = copy(Σ_w)

    w = λ / (n + λ)
    Σ_w[1] = w + (1 - α ^ 2 + β)
    m_w[1] = w
    Σ_w, m_w
end

"""
```julia
calculate_σ_points(
    ;σ_parameters::MerweScaled, x::Array{Float64, 1}, P::Array{Float64, 2}
)::Array{Float64, 2}
```

Compute `2n+1 x n` Van der Merwe's σ-points
for an Unscented Kalman Filter (UKF) given the mean `x` and covariance `P`.

# Arguments
- `σ_parameters::MerweScaled`: Parameters of the Van der Merwe's σ-points.
- `x::Array{Float64, 1}`: Mean of the UKF.
- `P::Array{Float64, 2}`: Covariance of the `x` of the UKF.
"""
function calculate_σ_points(
    ;σ_parameters::MerweScaled, x::Array{Float64, 1}, P::Array{Float64, 2}
)::Array{Float64, 2}
    λ = σ_parameters.α ^ 2 * (σ_parameters.n + σ_parameters.κ) - σ_parameters.n
    U = cholesky(Symmetric(P * (λ + σ_parameters.n))).U

    Σ = zeros(Float64, 2σ_parameters.n + 1, σ_parameters.n)
    @inbounds begin
    Σ[1, :] = x
    for i = 1:σ_parameters.n
        Σ[i + 1, :] = σ_parameters.residual_x(x, -@view(U[i, :]))
        Σ[σ_parameters.n + i + 1, :] = σ_parameters.residual_x(x, @view(U[i, :]))
    end
    end
    Σ
end

end
