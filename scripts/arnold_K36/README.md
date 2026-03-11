# arnold_K36

This folder reproduces the initial Arnold-style calibration steps for a two-scale Lorenz-96 configuration:

- `K = 36`
- `J = 10`
- `h = 2`
- `F = 10`
- `c = 10`
- `b = 8`

It contains two executable stages:

1. `generate_observations.jl`:
   Integrates the deterministic two-scale system and saves observed slow variables `X` in HDF5.
2. `fit_first_guess_closure.jl`:
   Loads observed `X`, fits a first-guess closure of the form
   `alpha0 + alpha1*X + alpha2*X^2 + alpha3*X^3 + sigma*xi`,
   verifies stochastic-model stability on long rollouts, and writes a FigB-style statistics comparison.

## Run

From repository root:

```bash
julia --project=. scripts/arnold_K36/generate_observations.jl --params scripts/arnold_K36/parameters_observations.toml
julia --threads auto --project=. scripts/arnold_K36/fit_first_guess_closure.jl --params scripts/arnold_K36/parameters_first_guess.toml
```
