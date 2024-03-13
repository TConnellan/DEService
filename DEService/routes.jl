using Genie.Router, Genie.Requests, Genie
using DEService.Simulator

route("/") do
  serve_static_file("welcome.html")
end

route("/compute") do
  # test running simulations within container
  output = DEService.Simulator.test_timing(100.0)
end