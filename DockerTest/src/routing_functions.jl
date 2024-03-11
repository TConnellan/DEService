using StatsBase

"""
Chooses a destination from the vector nodes based on the probabilities in p_e_w
"""
function route_travel(nodes::Vector{Int}, p_e_w::Weights{Float64, Float64, Vector{Float64}})::Int64
    return sample(nodes, p_e_w)
end

"""
Determines if the destination of a job corresponds to it leaving the system. 
If so true, false otherwise.
"""
function is_leaving(dest::Int64)::Bool
    return dest == -1 ? true : false
end