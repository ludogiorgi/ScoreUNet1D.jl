# ==============================================================================
# Integrate KS: Langevin Dynamics Integration for Kuramoto-Sivashinsky System
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/KS/integrate_ks.jl
#   nohup julia --project=. scripts/KS/integrate_ks.jl > integrate_ks.log 2>&1 &
#
# Edit integrate_params.toml to set:
#   [phi_sigma] mode = "identity" or "file"
#
# ==============================================================================

using ScoreUNet1D

const CONFIG_PATH = joinpath(@__DIR__, "integrate_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

result = integrate_langevin(CONFIG_PATH; project_root=PROJECT_ROOT)

# Generate comparison figure
figure_path = plot_comparison_figure(
    result.statistics,
    joinpath(dirname(result.output_path),
        result.phi_sigma_mode == :identity ? "comparison_identity.png" : "comparison.png")
)

@info "Integration complete"
@info "  Mode: $(result.phi_sigma_mode)"
@info "  Trajectory: $(result.output_path)"
@info "  Figure: $(figure_path)"
