# ScoreUNet1D.jl

Julia source for the Kuramoto--Sivashinsky experiments in:

Ludovico T. Giorgini, "Score-Based Modeling of Effective Langevin Dynamics,"
*Physical Review E* **114**, L012102 (2026),
[doi:10.1103/6qpv-lqmt](https://doi.org/10.1103/6qpv-lqmt).

The package trains a periodic one-dimensional U-Net by denoising score matching,
estimates the constant mobility and diffusion matrices from trajectory data,
and integrates the resulting reduced Langevin model. This repository contains
only source, configuration, tests, and publication metadata. Datasets, trained
models, generated figures, and run directories remain external artifacts.

## Requirements

- Julia 1.10 or 1.11 (the publication release was verified with Julia 1.10.5)
- HDF5 dataset with normalized 1D samples
- CPU/GPU with sufficient RAM for Langevin integration

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Repository structure

```
ScoreUNet1D.jl/
├── src/                    # Package source code
│   ├── ScoreUNet1D.jl      # Main module
│   ├── architecture/       # U-Net architecture
│   ├── training/           # Score matching trainer
│   ├── evaluation/         # Langevin engine, Phi/Sigma estimation
│   └── runners/            # Shared utilities (config, I/O)
├── scripts/
│   ├── KS/                 # Kuramoto-Sivashinsky system
│   │   ├── train_ks.jl           # Train score network
│   │   ├── train_params.toml
│   │   ├── integrate_ks.jl       # Langevin integration
│   │   ├── integrate_params.toml
│   │   ├── alpha_tuning_ks.jl    # α parameter tuning
│   │   ├── alpha_params.toml
│   │   └── plot_publication_ks.jl
│   └── check_phi_sigma.jl  # Utility script
├── data/KS/                # External KS datasets (gitignored)
├── runs/KS/                # Generated training runs (gitignored)
├── plot_data/KS/           # Generated figures and statistics (gitignored)
└── test/                   # Unit tests
```

## KS workflow

### 1. Train Score Network
```bash
julia --project=. scripts/KS/train_ks.jl
```
Edit `scripts/KS/train_params.toml` for hyperparameters.

### 2. Run Langevin Integration
```bash
# Edit integrate_params.toml: mode = "identity" or "file"
julia --project=. scripts/KS/integrate_ks.jl
```

### 3. (Optional) Tune α Parameter
```bash
julia --project=. scripts/KS/alpha_tuning_ks.jl
```

### 4. Generate Publication Figures
```bash
julia --project=. scripts/KS/plot_publication_ks.jl
```

## Configuration

All scripts use TOML configuration files in the same directory:

| Config | Key Settings |
|--------|-------------|
| `train_params.toml` | `epochs`, `batch_size`, `sigma`, `device` |
| `integrate_params.toml` | `phi_sigma.mode` ("identity"/"file"), `n_steps`, `n_ensembles` |
| `alpha_params.toml` | `alpha_lower`, `alpha_upper`, `max_evals` |

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The lightweight public-boundary checks used by continuous integration are:

```bash
julia test/source_checks.jl
```

## Citation and license

Citation metadata are provided in `CITATION.cff`. The source code is available
under the BSD 3-Clause License; see `LICENSE`.
