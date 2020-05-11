module Helpers
export default_mean_f, default_add_x, default_residual

function default_mean_f(
    Σ::Array{Float64, 2}, m_w::Array{Float64, 1}
)::Array{Float64, 1}
    Σ' * m_w
end

function default_add_x(
    y::Array{Float64, 1}, x::Array{Float64, 1}
)::Array{Float64, 1}
    y + x
end

function default_residual(
    y::Array{Float64, 1}, x::Array{Float64, 1}
)::Array{Float64, 1}
    y - x
end

function default_residual(
    y::Array{Float64, 2}, x::Array{Float64, 1},
)::Array{Float64, 2}
    y .- x'
end

end
