using Genie.Router, Genie.Requests, Genie, Genie.Renderer.Json
using DEService.DESMain, DEService.DESSystem, DEService.DESState, DEService.DESParameters


route("/") do
  serve_static_file("welcome.html")
end

route("/compute") do
  # endpoint will accept a value of lambda (float), use this to create a NetworkParameters object
  λ = parse(Float64, getpayload(:lambda, 1))
  params = create_scen1(λ)
  # stateType = DESState.TrackTotals
  stateType = getpayload(:mode, "")
  if stateType == "tracktotals"
    stateType = DESState.TrackTotals
  elseif stateType == "trackalljobs"
    stateType = DESState.TrackAllJobs
  else
    return "mode does not exist"
  end
  initState = DESSystem.create_init_state(stateType, params) 
  initEvent = create_init_event(params, initState)
  maxTime = parse(Float64, getpayload(:timethreshold, 100))
  result = simulate(params, initState, initEvent, max_time = maxTime, callback = record_data)
  if stateType == DESState.TrackTotals
    return Genie.Renderer.Json.json(Dict("averageJobs" => result[1], "transitProportion" => result[2]))
  end
  if stateType == DESState.TrackAllJobs
    return Genie.Renderer.Json.json(Dict("sojournTimes" => result))
  end
end