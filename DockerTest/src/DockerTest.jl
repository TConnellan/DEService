module DockerTest

using Genie
using Parameters
using Distributions
using StatsBase
using TimerOutputs
using DataStructures
using Plots

const up = Genie.up
export up

function main()
  Genie.genie(; context = @__MODULE__)
end

end