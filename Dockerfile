# FROM julia:1.10.2-alpine3.19
FROM julia:latest

WORKDIR /

RUN julia -e 'using Pkg; Pkg.add("Genie");  Pkg.add("LinearAlgebra"); Pkg.add("Parameters"); Pkg.add("DataStructures"); Pkg.add("Distributions"); Pkg.add("StatsBase"); Pkg.add("TimerOutputs");'

COPY . . 

# COPY main.jl .

EXPOSE 8000

CMD ["julia", "main.jl"]