#!/usr/bin/env julia
# Standard command:
# julia --threads auto --project=. scripts/L96/compute_responses.jl --params scripts/L96/parameters_responses.toml
# Nohup command:
# nohup julia --threads auto --project=. scripts/L96/compute_responses.jl --params scripts/L96/parameters_responses.toml > scripts/L96/nohup_compute_responses.log 2>&1 &

include(joinpath(@__DIR__, "lib", "compute_responses_core.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
