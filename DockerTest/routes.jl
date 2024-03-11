using Genie.Router, Genie.Requests
include("./src/parameters.jl")
include("./src/state.jl")
include("./src/events.jl")
include("./src/system.jl")
include("./src/routing_functions.jl")

route("/") do
  "root"
end

function compute()::String
  return "compute"
end

route("/compute", compute, method = GET)