# L96 Pipeline

This folder contains a production-oriented Lorenz-96 workflow:

- `run_pipeline.jl`: end-to-end data generation, training, KL evaluation, and response figures.
- `compute_responses.jl`: response computation for the current checkpoint (UNet online + cached baselines).
- `generate_reference_responses.jl`: high-precision Gaussian and numerical baseline generation/cache refresh.

## Quick Commands

Standard run:
```bash
julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters.toml
```

Fast smoke run:
```bash
julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters_fast.toml
```

Nohup pipeline run:
```bash
nohup julia --project=. scripts/L96/run_pipeline.jl --params scripts/L96/parameters.toml > scripts/L96/nohup_l96_pipeline.log 2>&1 &
```

Generate/refresh reference cache:
```bash
julia --threads auto --project=. scripts/L96/generate_reference_responses.jl --params scripts/L96/parameters_responses.toml
```

