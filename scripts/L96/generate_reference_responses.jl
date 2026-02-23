#!/usr/bin/env julia
# Standard command:
# julia --threads auto --project=. scripts/L96/generate_reference_responses.jl --params scripts/L96/parameters_responses.toml
# Nohup command:
# nohup julia --threads auto --project=. scripts/L96/generate_reference_responses.jl --params scripts/L96/parameters_responses.toml > scripts/L96/nohup_generate_reference_responses.log 2>&1 &

include(joinpath(@__DIR__, "lib", "compute_responses_core.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    main(vcat(["--mode", "reference"], ARGS))
end
