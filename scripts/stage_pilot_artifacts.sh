#!/usr/bin/env bash
# Symlink the archived Colab/Drive pilot outputs into outputs/ so configs resolve. Read-only use; never overwrite.
# Usage: scripts/stage_pilot_artifacts.sh /path/to/metamer-sae-drive-root
set -euo pipefail
SRC="${1:?path to Drive root containing activations/, outputs/, representations/}"
mkdir -p outputs/representations/pca
ln -sfn "$SRC/activations"               outputs/activations
ln -sfn "$SRC/outputs/sae_preparation"   outputs/sae_preparation
ln -sfn "$SRC/outputs/topk_sae"          outputs/topk_sae
ln -sfn "$SRC/representations/pca/runs"  outputs/representations/pca/runs
echo "Staged. Next: metamer-sae split-audit --config configs/experiments/pilot_resnet50_convnext_stage3.yaml"
