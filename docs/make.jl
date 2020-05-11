push!(LOAD_PATH, "./src/")

using Documenter
using SigmaPoints
using UnscentedKalmanFilter

makedocs(sitename="Kalman Filters")
deploydocs(repo="github.com/pxl-th/KalmanFilters.git")
