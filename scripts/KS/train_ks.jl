# ==============================================================================
# Train KS: Score Network Training for Kuramoto-Sivashinsky System
# ==============================================================================
#
# Usage:
#   julia --project=. scripts/KS/train_ks.jl
#   nohup julia --project=. scripts/KS/train_ks.jl > train_ks.log 2>&1 &
#
# ==============================================================================

using ScoreUNet1D

const CONFIG_PATH = joinpath(@__DIR__, "train_params.toml")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

result = train_score_network(CONFIG_PATH; project_root=PROJECT_ROOT)

@info "Training complete"
@info "  Final loss: $(result.history.epoch_losses[end])"
@info "  Model saved: $(result.model_path)"
@info "  Dataset: $(result.dataset_size) samples, trained on $(result.train_size)"
