#!/usr/bin/env bash
set -euo pipefail
CFG=configs/experiments/pilot_resnet50_convnext_stage3.yaml
metamer-sae validate-config --config $CFG
metamer-sae split-audit     --config $CFG
metamer-sae parity          --config $CFG --run-root outputs/runs
