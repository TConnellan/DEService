using Genie.Router, Genie.Requests, Genie.Responses, Genie, Genie.Renderer.Json
using DEService.DESMain, DEService.DESSystem, DEService.DESState, DEService.DESParameters


route("/") do
  serve_static_file("welcome.html")
end

route("/compute", method = POST) do
  # endpoint will accept a value of lambda (float), use this to create a NetworkParameters object
  λ = convert(Float64, jsonpayload("lambda"))
  params = create_scen1(λ)
  # stateType = DESState.TrackTotals
  stateType = jsonpayload("mode")
  if stateType == "tracktotals"
    stateType = DESState.TrackTotals
  elseif stateType == "trackalljobs"
    stateType = DESState.TrackAllJobs
  else
    Genie.Responses.setstatus(400)
    return Genie.Renderer.Json.json(Dict("error" => "mode does not exist"))
  end
  initState = DESSystem.create_init_state(stateType, params) 
  initEvent = create_init_event(params, initState)
  maxTime = convert(Float64, jsonpayload("timethreshold"))
  result = simulate(params, initState, initEvent, max_time = maxTime, callback = record_data)
  if stateType == DESState.TrackTotals
    Genie.Responses.setstatus(200)
    return Genie.Renderer.Json.json(Dict("averageJobs" => result[1], "transitProportion" => result[2]))
  end
  if stateType == DESState.TrackAllJobs
    Genie.Responses.setstatus(200)
    return Genie.Renderer.Json.json(Dict("sojournTimes" => result))
  end
end