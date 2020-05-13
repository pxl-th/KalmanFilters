using Test

using KalmanFilters.UnscentedKalmanFilter
using KalmanFilters.SigmaPoints

@testset "MerweScaled" begin
include("test_merwe.jl")
end

@testset "Unscented Kalman Filter" begin
include("test_ukf.jl")
end
