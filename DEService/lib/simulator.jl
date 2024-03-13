

# From https://github.com/TConnellan/Discrete-Event-Simulator-of-Queuing-System

module Simulator

using DataStructures
using Parameters #You need to install the Parameters.jl package: https://github.com/mauro3/Parameters.jl 
using LinearAlgebra 
using Distributions
using StatsBase
using Plots
using TimerOutputs
import Base: isless


# -------------------------------- Parameters --------------------------------

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


# -------------------------------- routing functions --------------------------------

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

# -------------------------------- state --------------------------------
abstract type State end

"""
Stores system state with the capability of tracking the trajectory of 
all jobs
"""
mutable struct TrackAllJobs <: State
    # maps each job that is currently in the system to its entry time
    entryTimes::Dict{Int64, Float64}

    # maps each job that is currently in the system to its location
    # value of 0 denotes in transit, abs(value) is the node, negative value denotes being served
    currentPosition::Dict{Int64, Int64}
    
    # stores the sojourn of the most recent job to have left the system
    # is set to -1 if no unrecorded times
    sojournTime::Float64
    
    # buffers for each node, front of queue denotes being served
    buffers::Vector{Queue{Int64}}
    # total number of jobs that have been (or attempted to have been in) the system
    jobCount::Int64
end

"""
Stores system state with the capability of tracking only the
number of jobs which are at a specific node, in transit or have
left the system.
"""
mutable struct TrackTotals <: State
    # number of jobs at each node
    atNodes::Vector{Int64}
    # number of jobs in transit
    transit::Int64

    # total number of jobs that have been (or attempted to have been in) the system
    jobCount::Int64
end

"""
Get a new unique identifier for a job
"""
function new_job(state::TrackAllJobs)::Int64
    return state.jobCount + 1
end

"""
Get a new unique identifier for a job
"""
function new_job(state::TrackTotals)::Int64
    return state.jobCount + 1
end

function job_join_system end

"""
Update the system to reflect a job joining it
"""
function job_join_sys(job::Int64, node::Int64, time::Float64, state::TrackAllJobs)::Int64
    # record job in transit 
    state.currentPosition[job] = 0
    # record its entry time
    state.entryTimes[job] = time
    # update number of jobs in system
    state.jobCount += 1
end

"""
Update the system to reflect a job joining it
"""
function job_join_sys(job::Int64, node::Int64, time::Float64, state::TrackTotals)::Int64
    state.transit += 1
    state.jobCount += 1
end

function job_leave_sys end

"""
Update the system to reflect a job leaving it
"""
function job_leave_sys(job::Int64, node::Int64, time::Float64, state::TrackAllJobs)::Nothing
    # record sojourn time
    state.sojournTime = time - arr_time(job, state)
    # remove tracking of job
    delete!(state.currentPosition, job)
    delete!(state.entryTimes, job)
    return nothing
end

"""
Update the system to reflect a job leaving it
"""
function job_leave_sys(job::Int64, node::Int64, time::Float64, state::TrackTotals)::Nothing
    # do nothing
end

# ----------------

function job_join_transit end

"""
Update the system to reflect a job entering transit
"""
function job_join_transit(job::Int64, node::Int64, state::TrackAllJobs)
    state.currentPosition[job] = 0
end   

"""
Update the system to reflect a job entering transit
"""
function job_join_transit(job::Int64, node::Int64, state::TrackTotals)::Int64
    return state.transit += 1
end

function job_leave_transit end

"""
Update the system to reflect a job leaving transit
"""
function job_leave_transit(job::Int64, state::TrackAllJobs)
    # do nothing
end

"""
Update the system to reflect a job leaving transit
"""
function job_leave_transit(job::Int64, state::TrackTotals)
    @assert state.transit >= 1
    state.transit -= 1
end

"""
Update the system to reflect a job joining a node
"""
function job_join_node(job::Int64, node::Int64, state::TrackAllJobs)
    # record jobs position
    state.currentPosition[job] = node
    # join the queue at this node
    enqueue!(state.buffers[node], job)
end

"""
Update the system to reflect a job joining a node
"""
function job_join_node(job::Int64, node::Int64, state::TrackTotals)::Int64
    return state.atNodes[node] +=1
end

function job_leave_node end

"""
Update the system to reflect a job leaving a node
"""
function job_leave_node(job::Int64, node::Int64, state::TrackAllJobs)::Int64
    #leave the buffer
    return dequeue!(state.buffers[node])
end

"""
Update the system to reflect a job leaving a node
"""
function job_leave_node(job::Int64, node::Int64, state::TrackTotals)::Int64
    @assert state.atNodes[node] >= 1
    state.atNodes[node] -= 1
    return 1
end   

"""
Update the system to reflect a job beginning service
"""
function job_begin_service(job::Int64, state::TrackAllJobs)
    state.currentPosition[job] = -abs(state.currentPosition[job])
    return
end

"""
Update the system to reflect a job beginning service
"""
function job_begin_service(job::Int64, state::TrackTotals)
    # do nothing
    return
end

"""
Update the system to reflect a job ending service
"""
function job_end_service(job::Int64, state::TrackAllJobs)
    state.currentPosition[job] = abs(state.currentPosition[job])
end

"""
Update the system to reflect a job ending service
"""
function job_end_service(job::Int64, state::TrackTotals)
    # do nothing
end

"""
Get the entry time of a job, if it is still in the system
"""
function arr_time(job::Int64, state::TrackAllJobs)::Float64
    if haskey(state.currentPosition, job)
        return state.entryTimes[job]
    else
        throw(error("Job $job is not currently in system"))
    end
end

"""
Find the job that is currently being served at a node
"""
function get_served(node::Int64, state::TrackAllJobs)::Int64
    return first(state.buffers[node])
end

"""
Find the job that is currently being served at a node however since
TrackTotals does not distinguish between jobs we return a meaningless value.
"""
function get_served(node::Int64, state::TrackTotals)::Int64
    @assert length(state.atNodes[node]) > 0
    return 1
end

"""
Check if there is room for a new job at a node. If so returns true
false otherwise.
"""
function check_capacity(node::Int64, params::NetworkParameters, state::TrackAllJobs)::Bool
    # check if buffer has infniite capacity or is below its capacity
    # first element in buffer is the one being served hence max elements in buffer is K + 1
    return params.K[node] == -1 || length(state.buffers[node]) < params.K[node] + 1
end

"""
Check if there is room for a new job at a node. If so returns true
false otherwise.
"""
function check_capacity(node::Int64, params::NetworkParameters, state::TrackTotals)::Bool
    # check if buffer has infniite capacity or is below its capacity
    # first element in buffer is the one being served hence max elements in buffer is K + 1
    return params.K[node] == -1 || state.atNodes[node] < params.K[node] + 1
end


"""
Returns the number of jobs at a node (being served and in buffer)
"""
function jobs_at_node(node::Int64, state::TrackAllJobs)::Int64
    return length(state.buffers[node])
end

"""
Returns the number of jobs at a node (being served and in buffer)
"""
function jobs_at_node(node::Int64, state::TrackTotals)::Int64
    return state.atNodes[node]
end

# -------------------------------- events --------------------------------

abstract type Event end

"""
Captures an event and the time it takes place
"""
struct TimedEvent
    event::Event
    time::Float64

    TimedEvent(event::Event, time::Float64) = new(event, time)
end

"""
Comparison of two timed events
"""
isless(te1::TimedEvent, te2::TimedEvent)::Bool = te1.time < te2.time

"""
 Abstract function for updating the system when an event occurs.
"""
function process_event end # This defines a function with zero methods (to be added later)

# Generic events that we can always use

"""
Return an event that ends the simulation.
"""
struct EndSimEvent <: Event end

"""
Processes an end of simulation event
"""
function process_event(time::Float64, state::State, params::NetworkParameters, 
                                es_event::EndSimEvent, new_ev::Vector{TimedEvent})::Nothing
    return nothing
end

"""
A job arriving at node from outside the system at time t
"""
struct ExternalArrivalEvent <: Event 

    # the destination of the new arrival
    node::Int64

    # the identifier of the new arrival
    job::Int64
end

"""
Processe an external arrival event, updates state and stores any new events in new_ev
"""
function process_event(time::Float64, state::State, params::NetworkParameters, 
                                ext_event::ExternalArrivalEvent, new_ev::Vector{TimedEvent})::Nothing
    # update state to have job join system
    job_join_sys(ext_event.job, ext_event.node, time, state)

    # jov attempts to join a node
    join_node(time, ext_event.job, ext_event.node, state, params, new_ev)

    # find route an time of next arrival
    t = time + ext_arr_time(params)
    dest = route_travel(params.L_vec, params.p_e_w)
    push!(new_ev, TimedEvent(ExternalArrivalEvent(dest, new_job(state)), t))
    return nothing
end

"""
A job attempting to join a node
"""
struct JoinNodeEvent <: Event 
    # the destination of the job in transit
    node::Int64
    # the identifier of the job in transit
    job::Int64
end

"""
Processes a join node event, updates state and stores any new events in new_ev
"""
function process_event(time::Float64, state::State, params::NetworkParameters, 
                        join_event::JoinNodeEvent, new_ev::Vector{TimedEvent})::Nothing
    join_node(time, join_event.job, join_event.node, state, params, new_ev)
    return nothing
end

"""
The completion of service of some job at a node
"""
struct ServiceCompleteEvent <: Event
    node::Int64
end

"""
Processes a service complete event, updates state and stores any new events in new_ev
"""
function process_event(time::Float64, state::State, params::NetworkParameters, 
                            sc_event::ServiceCompleteEvent, new_ev::Vector{TimedEvent})::Nothing
    # determined what job has completed service and update the state to reflect this
    done_service = get_served(sc_event.node, state)
    job_end_service(done_service, state)
    job_leave_node(done_service, sc_event.node, state)

    # route the next destination of the job
    dest = route_travel(params.L_vec, params.P_w[sc_event.node])
    # update state / create new event in necessary
    if is_leaving(dest)
        job_leave_sys(done_service, sc_event.node, time, state)
    else
        t = time + transit_time(params)
        push!(new_ev, TimedEvent(JoinNodeEvent(dest, done_service), t))
        job_join_transit(done_service, sc_event.node, state)
    end

    # if the buffer is not empty start serving a new job
    if jobs_at_node(sc_event.node, state) > 0
        t = time + service_time(params, sc_event.node)
        push!(new_ev, TimedEvent(ServiceCompleteEvent(sc_event.node), t))
        job_begin_service(get_served(sc_event.node, state), state)
    end
    return nothing
end

"""
Updates the state of the sysem as a job attempts to join a node. Stores any new events in new_ev
"""
function join_node(time::Float64, job::Int64, node::Int64, state::State, 
                                params::NetworkParameters, new_ev::Vector{TimedEvent})::Nothing
    
    # check if there is room in the buffer
    if (check_capacity(node, params, state))
        # join node and update state
        job_leave_transit(job, state)
        job_join_node(job, node, state)

        # job is first in buffer and thus being served
        if jobs_at_node(node, state) == 1
            t= time + service_time(params, node)
            push!(new_ev, TimedEvent(ServiceCompleteEvent(node), t))
        end
    else
        # buffer is full so overflow
        t = time + transit_time(params)
        dest = route_travel(params.L_vec, params.Q_w[node])
        if (is_leaving(dest)) 
            # leave system, update state
            job_leave_transit(job, state)
            job_leave_sys(job, node, time, state)
        else
            push!(new_ev, TimedEvent(JoinNodeEvent(dest, job), t))
        end
    end
    return nothing
end

# -------------------------------- system --------------------------------


"""
The main simulation function gets an initial state, parameters and event. 
Optional arguments are the maximal time for the simulation and a call-back function.
Returns any data generated by call-back.
"""
function simulate(params::NetworkParameters, init_state::State, init_timed_event::TimedEvent
                    ; 
                    max_time::Float64 = 10.0,
                    callback = (time, state, data, meta) -> nothing)

    # The event queue
    priority_queue = BinaryMinHeap{TimedEvent}()


    #containers for the simulation statistics that will be recorded
    data, meta = initialise_data(init_state)


    # Put the standard events in the queue
    push!(priority_queue, init_timed_event)
    push!(priority_queue, TimedEvent(EndSimEvent(), max_time))


    # initilize the state
    state = deepcopy(init_state)
    time = 0.0

    # Callback at simulation start
    callback(time, state, data, meta)

    # initialise vector to record all new events that must be added to the queue
    new_events = Vector{TimedEvent}()

    # The main discrete event simulation loop
    while true
        # Get the next event
        timed_event = pop!(priority_queue)

        # Advance the time
        time = timed_event.time

        # Act on the event
        process_event(time, state, params, timed_event.event, new_events) 

        # If the event was an end of simulation then stop
        if timed_event.event isa EndSimEvent
            break 
        end
        # The event may spawn 0 or more events which we put in the priority queue 
        while (!isempty(new_events))
            push!(priority_queue, pop!(new_events))
        end

        # Callback for each simulation event
        callback(time, state, data, meta)
    end
    #callback at simulation end
    callback(time, state, data, meta)

    # computing average of running statistics when in TrackTotals state
    if (typeof(state) <: TrackTotals)
        data[1] = data[1] / time
        data[2] = data[2] / time
    end
    return data
end;

"""
Initialises the data and meta containers for the TrackAllJobs state.
"""
function initialise_data(s::TrackAllJobs)::Tuple{Vector{Float64}, Vector{Float64}}
    out = Vector{Float64}(), Float64[]
    return out
end

"""
Initialises the data and meta containers for the TrackTotals state.
"""
function initialise_data(s::TrackTotals)::Tuple{Vector{Float64}, Vector{Float64}}
    return zeros(2), zeros(3)
end

"""
Callback function for the TrackAllJobs state, records a sojourn time, if there is one to be recorded.
"""
function record_data(time::Float64, state::TrackAllJobs, data::Vector{Float64}, meta::Vector{Float64})
    if state.sojournTime != -1
        push!(data, state.sojournTime)
        state.sojournTime = -1
    end
end


"""
Callback function for the TrackTotals state, accumulates a weighted tally of the number of 
items in the system and the proportion of items in orbit. (Note: Does NOT return an average)
Time is the current time of the sytem, state is the state of the system, data is the running statistics
and meta = [prev_time, prev_count, prev_prop] is metadata to compute the running statistics.
"""
function record_data(time::Float64, state::TrackTotals, data::Vector{Float64}, meta::Vector{Float64})
    node_sum = sum(state.atNodes) # the total number of items either being served or in a buffer at the nodes
    if time != 0
        # weight for the currently observed data
        time_step = time - meta[1]
        
        # accumulate total counts
        data[1] = data[1] + meta[2]*time_step

        # make sure the proportion is defined in the time-period ending at time
        # if undefined we add 0 so unchanged
        if (meta[2] != 0)
            # add to our running proportion count
            data[2] += meta[3]*time_step
        end
    end
    # update metadata
    meta[1] = time
    meta[2] = state.transit + node_sum
    meta[3] = state.transit / (node_sum + state.transit)
    return
end

# setting up and doing simulation ----------------------------

"""
Creates an initial, empty, state determined by the given parameters
"""
function create_init_state(s, p::NetworkParameters)
    if (s <: TrackAllJobs)
        return TrackAllJobs(Dict{Int64, Float64}(), Dict{Int64, Int64}(), -1, [Queue{Int64}() for _ in 1:p.L], 0)
    else 
        return TrackTotals(zeros(p.L), 0, 0)
    end
end    

"""
Creates an initial event determined by the parameters and state
"""
function create_init_event(p::NetworkParameters, s::State)
    # find initial destination
    dest = route_travel(p.L_vec, p.p_e_w)
    # find time of arrival
    t = transit_time(p)
    return TimedEvent(ExternalArrivalEvent(dest, new_job(s)), t)
end

"""
Creates and runs a simulation according to the state_type and parameters generated by 
param_func. Optional values are arrival rate λ and time-horizon max_time.
"""
function run_sim(state_type, param_func; λ::Float64 = 1.0, max_time::Float64=10.0)
    params = param_func(λ)
    state = create_init_state(state_type, params)
    init = create_init_event(params, state)
    return simulate(params, state, init, max_time = max_time, callback=record_data)
end

# ----------------------------- main ---------------------------

"""
Gets data simulated from a specific set of parameters, determined by the function scenario,
for a range of arrival rate values determined by Λ and soj_Λ for a time horizon determined by time.
"""
function collect_data(scenario, Λ, soj_Λ, time)

    means = Vector{Float64}(undef, length(Λ))
    mean_lock = ReentrantLock()
    props = Vector{Float64}(undef, length(Λ))
    prop_lock = ReentrantLock()
    sojourns = Vector{Vector{Float64}}(undef, length(soj_Λ))
    soj_lock = ReentrantLock()

    # collect mean and proportion data
    
        # scenario 4 uses different λ ranges for the first two plots
    #for (i, λ) in enumerate(Λ)
    Threads.@threads  for i in 1:size(Λ)[1]
        λ = Λ[i]
        if false && scenario == create_scen4
            x, y = run_sim(TrackTotals, scenario, λ=λ, max_time = time)
            if 0.75 <= λ <= 0.9
                lock(mean_lock) do 
                    # push!(means, x)
                    means[i] = x
                end
                lock(prop_lock) do 
                    # push!(props, y)
                    props[i] = y
                end
            else
                lock(prop_lock) do 
                    # push!(props, y)
                    props[i] = y
                end
            end
        else
            x, y = run_sim(TrackTotals, scenario, λ=λ, max_time = time)
            lock(mean_lock) do 
                # push!(means, x)
                means[i] = x
            end
            lock(prop_lock) do 
                # push!(props, y)
                props[i] = y
            end
        end
    end

    # collect sojourn data
    Threads.@threads for i in 1:size(soj_Λ)[1]
        λ = soj_Λ[i]
        lock(soj_lock) do 
            sojourns[i] = run_sim(TrackAllJobs, scenario, λ=λ, max_time=time)
        end
    end

    return Λ, means, props, soj_Λ, sojourns
end

"""
Specifies the range of λ values for each scenario. 
"""
function get_ranges(scen::Int64)
    scen == 1 && return ([collect(0.01:0.01:1.) ; collect(1.1:0.1:10) ;15 ;20], [0.1, 0.5, 1.5, 3, 10])
    scen == 2 && return ([collect(0.01:0.01:1.) ; collect(1.1:0.1:10) ;15 ;20], [0.1, 0.5, 1.5, 3, 10]) 
    scen == 3 && return ([collect(0.01:0.01:1.) ; collect(1.1:0.1:10) ;15 ;20], [0.1, 0.5, 1.5, 3, 10])
    scen == 4 && return ([collect(0.001:0.001:0.009) ; collect(0.01:0.01:0.75) ; collect(0.751:0.001:0.9) ; 
                                                                                collect(0.91:0.01:1.1)], 
                                                                                [0.01, 0.25, 0.5, .75, .85, .9])
    scen == 5 && return (collect(.01:.01:3), [0.1, 0.5, 2, 5, 10])
    throw("no such scenario specificied")
end

"""
Specifies the x-axis limits in the sojourn time distribution plots for each scenario.
"""
function get_lims(scen::Int64)
    scen == 1 && return [:auto, 40]
    scen == 2 && return [:auto, 80]
    scen == 3 && return [:auto, 100]
    scen == 4 && return [:auto, 300]
    scen == 5 && return [:auto, 25]
    throw("no such")
end

"""
Plots an/many empirical distribution function/s for each of the values in Λ and the corresponding sojourn data.
"""
function plot_emp(Λ::Vector{Float64}, data::Vector{Vector{Float64}}; title = "emp dist plot", xscale=:identity, xlims=[:auto, :auto], legend=:bottomright)
    # find the greatest sojourn time across all simulations
    m = maximum([maximum(d) for d in data])
    # construct empirical cumulative distribution function and range to compute it over
    f = ecdf(data[1])
    e = collect(0:0.01:(m+0.01))
    # create plot
    ecdfs_plot = plot(e, f(e), labels="$(Λ[1])", legend=legend, legendtitle="λ", title=title, xscale=xscale, xlims=xlims,
                    xlabel="Sojourn time", ylabel="Empirical Distribution")

    # add all other functions to the same plot
    for i in 2:length(data)
        f = ecdf(data[i])
        plot!(e, f(e), labels="$(Λ[i])")
    end
    return ecdfs_plot
end

"""
Creates and saves plots for a specific scenario and time horizon, requires a pre-existing file structure
The folder to be saved under can be determined by the save variable.
"""
function get_plots(scenario::Int64, time::Float64; save::String="test")
    t = floor(Int, log10(time))
    λ_vals, λ_soj_vals = get_ranges(scenario)

    scens = [create_scen1, create_scen2, create_scen3, create_scen4, create_scen5]
    Λ, means, props, soj_Λ, sojourns = collect_data(scens[scenario], λ_vals, λ_soj_vals, time)

    if true && scenario != 4
        Λ_props = Λ
        Λ_means = Λ
    else
        Λ_props = Λ
        Λ_means = [0.75:0.001:.9]
    end
    means_plot = plot(Λ_means, means, legend=false,
                        xlabel="Rate of arrival λ",
                        ylabel="Mean number of items",
                        title="The mean number of items in the system as\n λ varies with a time horizon of T=10^$t")
    
    savefig(means_plot, ".//$(save)plots//scen$(scenario)//scen$(scenario)_means_plot.png")
    
    props_plot = plot(Λ_props, props, legend=false,
    xlabel="Rate of arrival λ", 
    ylabel="Proportion in transit",
    title="The proportion of items in orbit\nas λ varies with a time horizon of T=10^$t")
    
    savefig(props_plot, ".//$(save)plots//scen$(scenario)//scen$(scenario)_props_plot")

    x_scale = get_lims(scenario)

    ecdf_plot = plot_emp(soj_Λ,sojourns, xlims = x_scale,
                            legend = :bottomright,
                            title="ECDF's of the sojourn time of an item\n for varied λ with a time horizon of T=10^$t\n")
    savefig(ecdf_plot, ".//$(save)plots//scen$(scenario)//scen$(scenario)_sojourn_plot")
end


"""
Creates and saves plots for each scenario.
"""
function create_all_plots(time::Float64; save::String="test")
    for i in 1:5
        get_plots(i, time, save=save)
    end
end



function test_timing(time::Float64; save::String="test")
    output = []
    for scenario in 1:5
        t = floor(Int, log10(time))
        λ_vals, λ_soj_vals = get_ranges(scenario)

        scens = [create_scen1, create_scen2, create_scen3, create_scen4, create_scen5]
        Λ, means, props, soj_Λ, sojourns = collect_data(scens[scenario], λ_vals, λ_soj_vals, time)
        push!(output, [Λ, means, props, soj_Λ, sojourns])
    end

    return output
end

end