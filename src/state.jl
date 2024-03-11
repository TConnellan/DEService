
using DataStructures

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