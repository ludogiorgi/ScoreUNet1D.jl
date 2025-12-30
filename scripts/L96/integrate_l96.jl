# ==============================================================================
# Integrate L96: Langevin Dynamics Integration for Two-Scale Lorenz-96 System
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/L96/integrate_l96.jl
#   nohup julia --project=. scripts/L96/integrate_l96.jl > integrate_l96.log 2>&1 &
#
# Edit integrate_params.toml to set:
#   [phi_sigma] mode = "identity" or "file"
#
# ==============================================================================

using ScoreUNet1D

const CONFIG_PATH = joinpath(@__DIR__, "integrate_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIGURES_DIR = joinpath(PROJECT_ROOT, "figures", "L96")

result = integrate_langevin(CONFIG_PATH; project_root=PROJECT_ROOT)

# Generate comparison figure
figure_path = plot_comparison_figure(
    result.statistics,
    joinpath(FIGURES_DIR,
        result.phi_sigma_mode == :identity ? "comparison_identity.png" : "comparison.png")
)

@info "Integration complete"
@info "  Mode: $(result.phi_sigma_mode)"
@info "  Trajectory: $(result.output_path)"
@info "  Figure: $(figure_path)"
