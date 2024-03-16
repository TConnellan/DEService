using Genie.Router, Genie.Requests, Genie.Responses, Genie, Genie.Renderer.Json
using DEService.DESMain, DEService.DESSystem, DEService.DESState, DEService.DESParameters


route("/") do
  serve_static_file("welcome.html")
end

# takes in iterable whose elements are interpreted as rows, converts the elements
# in these rows to the given converType, then converts into a Matrix with rows corresponding
# to the input
function loadMatrix(convertType, input)
  return Matrix(transpose(hcat([convertType.(row) for row in input]...)))
end

requiredKeys = ["lambda", "scv", "eta", "servicerates", "routingmatrix", "overflowmatrix", "arrivaldistribution", "nodes", "nodelimit"]

route("/compute", method = POST) do
  
  # endpoint will accept a value of lambda (float), use this to create a NetworkParameters object
  
  # check everything is present
  payload = jsonpayload()
  errorReturn = Dict("error" => "A required value was missing in the payload", "missingKeys" => [])
  flag = false
  for key in requiredKeys
    if !haskey(payload, key)
      flag = true
      push!(errorReturn["missingKeys"], key)
    end
  end
  if flag
    return Genie.Renderer.Json.json(errorReturn)
  end

  # extract parameters
  λ = Float64(jsonpayload("lambda"))
  scv = Float64(jsonpayload("scv"))
  η = Float64(jsonpayload("eta"))

  serviceRates = Vector(Float64.(jsonpayload("servicerates")))

  routingMatrix = loadMatrix(Float64, jsonpayload("routingmatrix"))
  overflowMatrix = loadMatrix(Float64, jsonpayload("overflowmatrix"))
  arrivalDistribution = Float64.(Vector(jsonpayload("arrivaldistribution")))

  nodes = Int64(jsonpayload("nodes"))
  nodeLimit = Int64.(Vector(jsonpayload("nodelimit")))

  # create parameter and state structs
  params = NetworkParameters(nodes, scv, λ, η, serviceRates, routingMatrix, overflowMatrix, arrivalDistribution, nodeLimit)
  
  stateType = jsonpayload("mode")

  if stateType == "tracktotals"
    stateType = DESState.TrackTotals
  elseif stateType == "trackalljobs"
    stateType = DESState.TrackAllJobs
  else
    Genie.Responses.setstatus(400)
    return Genie.Renderer.Json.json(Dict("error" => "mode does not exist"))
  end
  
  # initialise and run simulation
  initState = DESSystem.create_init_state(stateType, params) 
  initEvent = create_init_event(params, initState)
  maxTime = Float64(jsonpayload("timethreshold"))
  result = simulate(params, initState, initEvent, max_time = maxTime, callback = record_data)
  
  # check what kind of data was collected to return correct statistics
  if stateType == DESState.TrackTotals
    Genie.Responses.setstatus(200)
    return Genie.Renderer.Json.json(Dict("averageJobs" => result[1], "transitProportion" => result[2]))
  end
  if stateType == DESState.TrackAllJobs
    Genie.Responses.setstatus(200)
    return Genie.Renderer.Json.json(Dict("sojournTimes" => result))
  end
end