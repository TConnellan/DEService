module DESParameters

using Parameters 
using LinearAlgebra 
using DataStructures
using Distributions
using StatsBase

export NetworkParameters, ext_arr_time, service_time, transit_time, create_scen1, create_scen2, create_scen3, create_scen4, create_scen5

"""
Stores the all the parameters needed to determine a system
"""
@with_kw mutable struct NetworkParameters
    #L::Int
    L::Int64
    scv::Float64 #This is constant for all scenarios at 3.0
    λ::Float64 #This is undefined for the scenarios since it is varied
    η::Float64 #This is assumed constant for all scenarios at 4.0
    μ_vector::Vector{Float64} #service rates
    P::Matrix{Float64} #routing matrix
    Q::Matrix{Float64} #overflow matrix
    p_e::Vector{Float64} #external arrival distribution
    K::Vector{Int64} #-1 means infinity 

    L_vec::Vector{Int64} # all possible destinations of travel
    P_w::Vector{Weights{Float64, Float64, Vector{Float64}}} # probability weights for routing from nodes after service
    Q_w::Vector{Weights{Float64, Float64, Vector{Float64}}} # probability weights for routing from nodes after overflow
    p_e_w::Weights{Float64, Float64, Vector{Float64}} # probability weights for routing when entering system
end

"""
Constructor for NetworkParameters which extracts dependent information from the variables
"""
function NetworkParameters(L::Int64,
                            scv::Float64, #This is constant for all scenarios at 3.0
                            λ::Float64, #This is undefined for the scenarios since it is varied
                            η::Float64, #This is assumed constant for all scenarios at 4.0
                            μ_vector::Vector{Float64}, #service rates
                            P::Matrix{Float64}, #routing matrix
                            Q::Matrix{Float64}, #overflow matrix
                            p_e::Vector{Float64}, #external arrival distribution
                            K::Vector{Int64}) #-1 means infinity )
    L_vec = [collect(1:L) ; -1]
    P_app = [P [1-sum(P[i,:]) for i in 1:L]]
    Q_app = [Q [1-sum(Q[i,:]) for i in 1:L]]
    P_w = [Weights(P_app[i,:]) for i in 1:L]
    Q_w = [Weights(Q_app[i,:]) for i in 1:L]
    p_e_w = Weights(p_e)
    return NetworkParameters(L=L, scv=scv, λ=λ, η=η, μ_vector=μ_vector, P=P, Q=Q, p_e=p_e, K=K, L_vec=L_vec, P_w=P_w, Q_w=Q_w, p_e_w=p_e_w)
end

"""
Draws a gamma distriibuted random variable with squared coefficient of variance scv and expectation 1/rate.
"""
function gamma_scv(scv::Float64, rate::Float64)::Float64
    return rand(Gamma(1/scv, scv/rate))
end

"""
Draws a random time period the system must wait before a new job joins
"""
function ext_arr_time(params::NetworkParameters)::Float64
    return gamma_scv(params.scv, params.λ)
end

"""
Draws a random time for a job to complete service at a station
"""
function service_time(params::NetworkParameters, node::Int64)::Float64
    return gamma_scv(params.scv, params.μ_vector[node])
end

"""
Draws a random time for a job to traverse between stations
"""
function transit_time(params::NetworkParameters)::Float64
    return gamma_scv(params.scv, params.η)
end 

"""
Creates parameters corresponding to the first scenario
"""
function create_scen1(λ::Float64)
    return NetworkParameters( 3, 
    3.0, 
    λ, 
    4.0, 
    ones(3),
    [0 1.0 0;
    0 0 1.0;
    0 0 0],
    zeros(3,3),
    [1.0, 0, 0],
    fill(5,3))
end

"""
Creates parameters corresponding to the second scenario
"""
function create_scen2(λ::Float64)
    return NetworkParameters(3, 
    3.0, 
    λ, 
    4.0, 
    ones(3),
    [0 1.0 0;
    0 0 1.0;
    0.5 0 0],
    zeros(3,3),
    [1.0, 0, 0],
    fill(5,3))
end

"""
Creates parameters corresponding to the third scenario
"""
function create_scen3(λ::Float64)
    return NetworkParameters(3, 
    3.0, 
    λ, 
    4.0, 
    ones(3),
    [0 1.0 0;
    0 0 1.0;
    0.5 0 0],
    [0 0.5 0;
    0 0 0.5;
    0.5 0 0],
    [1.0, 0, 0],
    fill(5,3))
end

"""
Creates parameters corresponding to the fourth scenario
"""
function create_scen4(λ::Float64)
    return NetworkParameters(5, 
    3.0, 
    λ, 
    4.0, 
    collect(5.0:-1.0:1.0),
    [0   0.5 0.5 0   0;
    0   0   0   1   0;
    0   0   0   0   1;
    0.5 0   0   0   0;
    0.2 0.2 0.2 0.2 0.2],
    [0 0 0 0 0;
    1. 0 0 0 0;
    1. 0 0 0 0;
    1. 0 0 0 0;
    1. 0 0 0 0],                             
    [0.2, 0.2, 0, 0, 0.6],
    [-1, -1, 10, 10, 10])
end

"""
Creates parameters corresponding to the fifth scenario
"""
function create_scen5(λ::Float64)
    return NetworkParameters(20, 
    3.0, 
    λ, 
    4.0, 
    ones(Float64, 20),
    zeros(20,20),
    diagm(3=>0.8*ones(17), -17=>ones(3)),                        
    vcat(1,zeros(19)),
    fill(5,20))
end

end