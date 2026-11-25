#!/bin/bash
# From-scratch run of the pilot pipeline on MIT Engaging (no archived Colab/Drive artifacts needed):
# env -> tests -> Feather stimuli + Imagenette -> ConvNeXt extraction -> NEW PCA -> NEW TopK SAEs (seeds 123, 42)
# -> paired evaluation -> comparison against the pilot reference values -> expanded-study audit.
# Every fit is a NEW run under outputs/runs/; nothing from the pilot is overwritten.
# Partitions below are MIT Engaging ones; change them for another cluster.
#SBATCH -J metamer-sae
#SBATCH -p mit_normal_gpu,ou_bcs_normal,ou_bcs_high
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}   # submit from the repo root (mkdir -p logs first)
cd "$ROOT"
step() { echo; echo "=================== $* ($(date '+%H:%M:%S')) ==================="; }

step "environment"
module load miniforge/24.3.0-0
export PIP_CACHE_DIR=$ROOT/.cache/pip TORCH_HOME=$ROOT/data/external/torch_home MPLBACKEND=Agg
if [ ! -x .venv/bin/metamer-sae ]; then
  conda create -y -q -p "$ROOT/.conda312" python=3.12
  "$ROOT/.conda312/bin/python" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch==2.11.0 torchvision==0.26.0 || .venv/bin/pip install -q torch torchvision
  .venv/bin/pip install -q -e ".[dev,sae,baselines,feather]" h5py dill "torchaudio==2.11.0" \
    "git+https://github.com/jenellefeather/chcochleagram.git"   # Feather's all_model_info imports its own (audio) helpers
  .venv/bin/pip install -q "matplotlib==3.8.4"   # overcomplete 0.3.0 calls matplotlib.cm.get_cmap (removed in 3.9)
fi
source .venv/bin/activate
python - <<'EOF'
import torch, torchvision, overcomplete, sys
from importlib import metadata
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| overcomplete", metadata.version("overcomplete"), "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
nvidia-smi -L || true

step "pytest"
pytest -q -rs 2>&1 | tail -40 || echo "PYTEST REPORTED FAILURES (continuing)"

CFG=configs/experiments/pilot_resnet50_convnext_stage3.yaml
step "split ledger + Feather stimuli"
[ -f data/manifests/stimulus_allocation.csv ] || metamer-sae write-pilot-ledger --out data/manifests/stimulus_allocation.csv
metamer-sae validate-config --config $CFG
metamer-sae split-audit --config $CFG
metamer-sae download --config $CFG --throttle 0.05

step "Imagenette"
if [ ! -d data/external/imagenette2-320/train ]; then
  mkdir -p data/external
  curl -fsSL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -o data/external/imagenette2-320.tgz
  tar -xzf data/external/imagenette2-320.tgz -C data/external && rm data/external/imagenette2-320.tgz
fi

step "observer activations"
metamer-sae extract --config $CFG --target all --batch-size 128

step "training mean (natural train only)"
mkdir -p outputs/sae_preparation
python - <<'EOF'
import numpy as np, pathlib
out = pathlib.Path("outputs/sae_preparation/training_activation_mean.npy")
X = np.load("outputs/activations/natural_images/train/pooled_activations.npy", mmap_mode="r")
m = np.asarray(X, np.float64).mean(0).astype(np.float32)
if out.exists():
    assert np.array_equal(np.load(out), m), "existing training mean differs; refusing to overwrite"
else:
    np.save(out, m)
print("training mean", m.shape, float(m.mean()))
EOF

step "fit PCA (new run)"
PCA_RUN=$(metamer-sae fit-pca --config $CFG | tail -1); echo "$PCA_RUN"
step "train TopK SAE seed 123"
S123=$(metamer-sae train-sae --config $CFG --seed 123 | tail -1); echo "$S123"; cat "$S123/model/result.json"
step "train TopK SAE seed 42"
S42=$(metamer-sae train-sae --config $CFG --seed 42 | tail -1); echo "$S42"; cat "$S42/model/result.json"

step "derived config"
NEWCFG=outputs/configs/pilot_fromscratch_${SLURM_JOB_ID:-manual}.yaml
mkdir -p outputs/configs
python - "$CFG" "$NEWCFG" "$PCA_RUN" "$S123" "$S42" <<'EOF'
import sys, yaml, pathlib
src, dst, pca, s123, s42 = sys.argv[1:]
c = yaml.safe_load(open(src))
c["experiment"]["id"] = "pilot_resnet50_convnext_stage3_fromscratch"
c["experiment"]["description"] += " (from-scratch refit on Engaging; NOT the archived pilot artifacts)"
c["pca"]["run_id"] = pathlib.Path(pca).name
c["pca"]["artifact"] = f"{pca}/basis/pca_basis.npz"
c["sae"]["primary_checkpoint"] = f"{s123}/model/best_model.pt"
c["sae"]["replicate_checkpoint"] = f"{s42}/model/best_model.pt"
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print(open(dst).read())
EOF

step "evaluate (development, then primary)"
metamer-sae evaluate --config "$NEWCFG"
sleep 2
step "comparison with pilot reference values (refit, so small deviations are expected)"
metamer-sae parity --config "$NEWCFG" || echo "PARITY: at least one check outside tolerance (see table above)"

step "expanded-study audit (network; audit only)"
for M in alexnet_core_generator_native ipcl_core_generator_native common_observer_panel_core; do
  echo "--- $M"; sleep 2
  timeout 40m bash scripts/run_expanded_audit.sh configs/experiments/$M.yaml || echo "AUDIT $M: stopped/failed (see above)"
done

step "done"
ls -d outputs/runs/*
