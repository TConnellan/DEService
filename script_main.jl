using Distributions
using StatsBase
using TimerOutputs

include("./src/parameters.jl")
include("./src/state.jl")
include("./src/events.jl")
include("./src/system.jl")
include("./src/routing_functions.jl")

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
    for scenario in 1:5
        t = floor(Int, log10(time))
        λ_vals, λ_soj_vals = get_ranges(scenario)

        scens = [create_scen1, create_scen2, create_scen3, create_scen4, create_scen5]
        Λ, means, props, soj_Λ, sojourns = collect_data(scens[scenario], λ_vals, λ_soj_vals, time)
    end

end