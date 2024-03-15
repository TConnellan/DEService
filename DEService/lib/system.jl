# adapted from https://github.com/TConnellan/Discrete-Event-Simulator-of-Queuing-System

module DESSystem

using Parameters 
using LinearAlgebra 
using DataStructures
using Plots
using DEService.DESEvents, DEService.DESParameters, DEService.DESState, DEService.DESRoutingFunctions

export simulate, run_sim, record_data, create_init_state, create_init_event

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

end