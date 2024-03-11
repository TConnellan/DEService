using DataStructures
import Base: isless

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
