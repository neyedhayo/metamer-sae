#!/usr/bin/env bash
# Phase A-D audit for the expanded study (network access needed). Pilot parity (run_pilot_parity.sh) must pass first.
set -euo pipefail
CFG=${1:-configs/experiments/alexnet_core_generator_native.yaml}
metamer-sae validate-matrix  --config "$CFG"
metamer-sae resolve-registry --config "$CFG"      # inspect candidates, then pin model_key/experiment_key in the registry
metamer-sae matrix-inventory --config "$CFG"      # model/layer/stimulus inventory + manifests (writes data/manifests/)
metamer-sae acceptance                            # evidence-based scope checklist
