push!(LOAD_PATH, "./src/")

using Documenter
using KalmanFilters

makedocs(
    sitename="KalmanFilters",
    authors="Anton Smirnov",
    repo="https://github.com/pxl-th/KalmanFilters/blob/{commit}{path}#L{line}",
    modules=[UnscentedKalmanFilter, SigmaPoints],
    pages=[
        "index.md",
        "Filters" => "filters.md",
        "Sigma Points" => "sigma.md",
    ],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
)
deploydocs(repo="github.com/pxl-th/KalmanFilters.git")
