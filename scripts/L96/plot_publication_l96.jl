# ==============================================================================
# Plot Publication L96: Generate Publication-Ready Figures
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/L96/plot_publication_l96.jl
#
# Requires:
#   - plot_data/L96/trajectory_identity.hdf5 (from integrate_l96.jl with mode=identity)
#   - plot_data/L96/trajectory.hdf5 (from integrate_l96.jl with mode=file)
#
# ==============================================================================

using ScoreUNet1D

const CONFIG_PATH = joinpath(@__DIR__, "plot_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

paths = plot_publication_figure(CONFIG_PATH; project_root=PROJECT_ROOT)

@info "Publication figure generated"
@info "  Main figure: $(paths.main_figure)"
