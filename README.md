# ScoreUNet1D.jl

A Julia package that trains a 1D score-based U-Net on generic time-series data and validates it by integrating the associated Langevin dynamics. The repository ships with a full training/inference pipeline, moment-matching utilities to build the linear drift/diffusion operators \( \Phi \) and \( \Sigma \), and rich diagnostics (loss curves, PDFs, ACFs, joint distributions) saved per run.

## Requirements

- Julia ≥ 1.10
- An HDF5 dataset containing the normalized 1D samples you want to model (placed under `data/`, e.g. `data/new_ks.hdf5`)
- CPU with sufficient RAM; the default settings integrate long Langevin trajectories with large ensembles.

All Julia dependencies are declared in `Project.toml`. After cloning:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Repository layout

| Path | Description |
| --- | --- |
| `src/` | Package code (architecture, data pipeline, trainer, moment matching, Langevin eval). |
| `scripts/run_ks.jl` | Main entry point used in the paper-style experiments. |
| `scripts/parameters.toml` | Central configuration file (data path, training hyperparameters, Langevin solver knobs, moment-matching options, output locations, etc.). |
| `data/` | Place your HDF5 dataset here (ignored by git). |
| `runs/` | Auto-generated experiment folders containing artifacts (ignored by git). |

## Data expectations

`load_hdf5_dataset` expects samples arranged as `(length, channels, batch)` after specifying `samples_orientation = :columns` for the KS data. The repository assumes *normalized* inputs (zero mean / unit variance) so that the score model operates in a numerically stable regime. Update `scripts/parameters.toml` if your layout differs.

## Running the KS pipeline

1. Ensure `scripts/parameters.toml` points to your dataset (defaults to `data/new_ks.hdf5`).
2. (Optional) delete `scripts/trained_model.bson` to force a fresh training run; otherwise the script reuses the cached network and only re-estimates \( \Phi, \Sigma \) + Langevin diagnostics.
3. Launch:

   ```bash
   julia --project=. scripts/run_ks.jl
   ```

Each run creates `runs/run_YYYYMMDD_HHMMSS_<slug>/` containing:

- `comparison.png`: averaged PDFs (with KernelDensity.jl smoothing), ACF overlay (0–5 decorrelation times), and six KDE-based joint distributions for lags `j = 1,2,3`.
- `training_metrics.png`: loss & KL vs. epoch when training occurs.
- `run_config.toml`: snapshot of every configuration block, KL divergence, decorrelation time, and artifact paths.
- `model.bson`: run-specific checkpoint; `scripts/trained_model.bson` stores the latest reusable model.
- `moment_matching.bson`: serialized \( \Phi \) and \( \Sigma \) factors used for Langevin integration.

## Key configuration knobs (in `scripts/parameters.toml`)

- `[data] dt, train_samples, subset_seed`: choose how many normalized samples to train on and the sampling cadence used for statistics.
- `[training] epochs, batch_size, sigma, langevin_eval_interval`: training behavior of the score U-Net.
- `[moment_matching] max_samples, stride, seed`: controls accuracy/performance of the moment-matching estimation for \( \Phi, \Sigma \).
- `[langevin] nsteps, n_ensembles, burn_in, boundary`: Langevin solver parameters. Increasing these improves PDF/ACF fidelity at higher cost.
- `[output] run_root, lag_offsets`: where artifacts land and which spatial offsets appear in the comparison figure.
- `[run] verbose`: toggle per-stage timing logs.

## Tests

The repository currently focuses on the end-to-end training/inference script; add unit tests under `test/` if you extend the package. At minimum, run `scripts/run_ks.jl` to ensure your changes keep the PDFs/ACFs aligned (KL divergence reported in the logs and `run_config.toml`).

## Citation / attribution

This implementation follows the approach described in `plasim.txt` (score-based ROM for KS / PlaSim). Please cite the corresponding work if you use this code in academic publications.
