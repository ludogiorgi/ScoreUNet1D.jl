# ScoreUNet1D.jl

A Julia package for training 1D score-based U-Nets on time-series data and validating them via Langevin dynamics integration. Supports multiple dynamical systems with system-specific configurations.

## Requirements

- Julia ≥ 1.10
- HDF5 dataset with normalized 1D samples
- CPU/GPU with sufficient RAM for Langevin integration

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Repository Structure

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
├── data/KS/                # KS datasets (gitignored)
├── runs/KS/                # Training runs (gitignored)
├── plot_data/KS/           # Generated figures/data
└── test/                   # Unit tests
```

## Quick Start (KS System)

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

## Adding New Systems

1. Create `scripts/<SystemName>/` directory
2. Copy and adapt KS scripts
3. Create `data/<SystemName>/` for datasets
4. System-agnostic utilities remain in `src/runners/`

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Citation

If using this code, please cite the corresponding work on score-based reduced-order models.
