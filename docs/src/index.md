# KalmanFilters.jl

This library provides implementation of different Kalman Filters.

## Example
Minimal example with constant velocity model.

Necessary imports:
```julia
using LinearAlgebra: I, diagm
using KalmanFilters.UnscentedKalmanFilter
using KalmanFilters.SigmaPoints
```
Define state-transition function:
```julia
function fx(;x::Array{Float64, 1}, δt::Float64)::Array{Float64, 1}
    F = Float64[[1 δt 0 0];[0 1 0 0];[0 0 1 δt];[0 0 0 1]]
    F * x
end
```
Define measurement function:
```julia
hx(;x::Array{Float64, 1})::Array{Float64, 1} = x[[1, 3]]
```
Create filter state, perform prediction and feed measurements to filter:
```julia
function main()
    δt = 0.1
    nstd = 0.1
    σ_parameters = MerweScaled(n=4, α=0.1, β=2.0, κ=-1.0)
    ukf = UKFState(dim_x=4, dim_z=2, σ_parameters=σ_parameters)
    ukf.x[:] = [-1, 1, -1, 1]
    ukf.P[:] *= 0.2
    Q = [
        [2.5e-09 5.0e-08 0.0e+00 0.0e+00];
        [5.0e-08 1.0e-06 0.0e+00 0.0e+00];
        [0.0e+00 0.0e+00 2.5e-09 5.0e-08];
        [0.0e+00 0.0e+00 5.0e-08 1.0e-06];
    ]
    R = diagm([nstd ^ 2, nstd ^ 2])

    measurements = vcat([[i + randn() * nstd i + randn() * nstd] for i = 1:10]...)
    for i = 1:size(measurements, 1)
        predict!(ukf=ukf, δt=δt, Q=Q, fx=fx)
        update!(ukf=ukf, z=measurements[i, :], R=R, hx=hx)
    end
    println(ukf.x) # position should be around [10, 10] coordinate
end

main()
```

!!! note
    This package is still in development,
    more filters will be implemented in future.
