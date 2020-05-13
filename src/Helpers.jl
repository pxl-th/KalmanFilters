module Helpers
export default_mean_f, default_add_x, default_residual

"""
Default mean function.
"""
function default_mean_f(
    Σ::AbstractArray{Float64, 2}, m_w::AbstractArray{Float64, 1}
)::AbstractArray{Float64, 1}
    Σ' * m_w
end

"""
Default addition function.
"""
function default_add_x(
    y::AbstractArray{Float64, 1}, x::AbstractArray{Float64, 1}
)::AbstractArray{Float64, 1}
    y + x
end

"""
Default residual function.
"""
function default_residual(
    y::AbstractArray{Float64, 1}, x::AbstractArray{Float64, 1}
)::AbstractArray{Float64, 1}
    y - x
end

function default_residual(
    y::AbstractArray{Float64, 2}, x::AbstractArray{Float64, 1},
)::AbstractArray{Float64, 2}
    y .- x'
end

end
