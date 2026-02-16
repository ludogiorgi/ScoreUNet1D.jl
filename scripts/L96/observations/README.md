# L96 Observations Repository

This folder stores generated L96 observation datasets used by `scripts/L96/run_pipeline.jl`.

Layout:
- `J2/l96_timeseries.hdf5`
- `J2/integration_params.toml`
- `J4/l96_timeseries.hdf5`
- `J4/integration_params.toml`
- `J10/l96_timeseries.hdf5`
- `J10/integration_params.toml`

Behavior:
- The pipeline selects the dataset folder from `run.J`.
- If the dataset exists and has the expected number of channels (`J + 1`), it is reused.
- If missing (or mismatched), it is generated using `[data.generation]` parameters from `scripts/L96/parameters.toml`.
