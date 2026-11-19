# metamer-sae

**Which internal visual features differ between standard and adversarially trained networks when their
model metamers look different?**

Model metamers (Feather et al., 2023) are images optimised to match one internal activation of a *generator*
network. Standard-trained networks produce metamers that humans often cannot recognise; adversarially trained
ones produce more recognisable metamers. This project asks *what changed inside the representation*, by passing
Feather's published source images and metamers through a separate, frozen **observer** network and scoring them
with representations learned on natural images only: a **PCA** baseline and **TopK sparse autoencoders (SAE)**.

> Colab notebooks `01–09` are the validated pilot prototype (`notebooks/`). This repository is the tested,
> configurable implementation of that pipeline, built for **pilot parity first, then the expanded study** specified in
> [`docs/EXPERIMENTAL_SCOPE_AND_SCALE.md`](docs/EXPERIMENTAL_SCOPE_AND_SCALE.md): **ResNet-50, AlexNet and IPCL** are all
> first-class model families (not optional extras), with their training conditions kept distinct.

## Scope at a glance

| Family | Conditions (registry: `configs/models/registry.yaml`) | Status |
|---|---|---|
| ResNet-50 | standard (`experiment_1`), L2 ε=3 adversarial (`experiment_3`); random-perturbation control if official | pilot pair **verified**; rest audit-required |
| AlexNet | standard, L2 ε=3 adversarial, random-perturbation control (proposal: Experiment 4, stages relu0–relu4, fc0, fc1, final) | **expected, unverified** until `resolve-registry` |
| IPCL / AlexNetGN | self-supervised IPCL + architecture-matched supervised comparator (linear readout, no native head) | audit-required |
| Adversarial IPCL | only if an official checkpoint + metamers exist; otherwise a *new experiment* ([protocol](docs/protocols/ADVERSARIAL_IPCL_EXTENSION.md)) | feasibility gate; never relabelled |

Two analysis tracks are separate by construction: **Track A common-observer** (one natural-trained dictionary per observer × stage,
reused across all generators) and **Track B generator-native** (one dictionary per generator checkpoint × stage; source and its metamers
share it). A **transfer** track must be labelled and reports distribution mismatch.

## What the pilot established (and what it does not)

Pilot: ResNet-50 `layer2_fake_relu` generators (standard `experiment_1/resnet50` vs adversarial
`experiment_3/resnet50_l2_3_robust`), ConvNeXt-Tiny stage-3 observer, 48 Feather triplets fixed as
8 development / 32 primary / 8 reserve.

- Standard-metamer source distance exceeded adversarial in **32/32 triplets in all five spaces** (raw 384-D, PCA-16,
  PCA-64, SAE seed 123, SAE seed 42); all 17 paired contrasts survive Holm correction (adjusted p < 0.001).
- Adversarial metamers keep more of the source's active SAE features (support Jaccard ≈ 0.68 vs ≈ 0.53).

These are **model-representation** findings. They do not show human recognisability, semantic feature identity,
causality, or that SAEs beat PCA. Human-recognition analysis (condition-level Feather data, depth and smoothness
controls, noise ceiling) is a separate later stage; brain/fMRI is a future extension.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sae]"          # sae extra pins overcomplete==0.3.0 (validated version)
pip install -e ".[baselines]"        # optional: scikit-image + lpips
pip install -e ".[feather]"          # only if you need Feather's own helper imports (resampy, tensorboardX, cox)
pytest                               # unit + integration (36 pass in a torch-free sandbox; adapter/SAE tests need torch/overcomplete); parity skips without pilot artifacts
```

Pilot environment: Python 3.12, PyTorch 2.11, TorchVision 0.26, Overcomplete 0.3.0, Tesla T4. CPU works for evaluation.

## Repository map

| Path | Purpose |
|---|---|
| `configs/` | YAML configs (strict schema; unknown keys are an error). Start with `configs/experiments/pilot_*.yaml` |
| `src/metamer_sae/config` | Pydantic schema (pilot `RunConfig`, expanded `MatrixConfig` with declared layer subsets) |
| `src/metamer_sae/registry` | Model/training-condition registry (data, not branches), Feather-metadata resolution, adversarial-IPCL gate |
| `src/metamer_sae/feather` | Pinned-commit checkout, metadata, URL scheme, availability probing; `inventory.py`: model/layer/stimulus/behaviour-link inventories |
| `src/metamer_sae/data` | Fixed pilot split, source-level split generator, manifests, validated downloads |
| `src/metamer_sae/models` | ConvNeXt-Tiny pilot observer; `adapters.py`: common `ModelAdapter` (torchvision ResNet-50/AlexNet/ConvNeXt, Feather checkpoints incl. IPCL), per-model preprocessing, hook-based stage extraction |
| `src/metamer_sae/extraction` | Resumable batch extraction, cache keys, integrity checks |
| `src/metamer_sae/representations` | `Representation` interface; raw, PCA, Overcomplete TopK adapter; `tracks.py`: activation-space identity, track/dictionary-sharing guards, natural-only fit guard |
| `src/metamer_sae/training` | TopK training with the pilot checkpoint policy |
| `src/metamer_sae/features` | Density, mass/Gini, dead/rare features, decoder norms, Hungarian cross-seed matching |
| `src/metamer_sae/evaluation` | Metrics, pilot paired evaluation, `matrix.py` (many conditions, natural-scale normalisation, cross-model matrix), pixel/SSIM/LPIPS + smoothness, recognition-model baselines, split-guarded pools, parity checks |
| `src/metamer_sae/statistics` | Pilot triplet bootstrap/Wilcoxon/d_z/Holm/17-contrast family, partial Spearman, split-half; `matrix.py`: source-level contrasts per declared family, exploratory/confirmatory labels, clustered bootstrap |
| `src/metamer_sae/human` | Gated human-behaviour stage: inspect → link → validate → condition-level depth-controlled analysis |
| `src/metamer_sae/reporting` | Coverage heatmap and evidence-based acceptance audit (scope §15) |
| `src/metamer_sae/visualization` | Matplotlib figures (PNG + PDF) |
| `src/metamer_sae/provenance` | Hashing, immutable run directories, closure reports, `RunMetadata` lifecycle, `verify_run`/`mark_reviewed` |
| `src/metamer_sae/pipeline.py`, `pipeline_matrix.py`, `cli/` | Pilot stages; expanded-study stages (inventory, grouped split, per-space extraction, per-space PCA/SAE fit); the `metamer-sae` command |
| `tests/` | `unit/`, `integration/` (synthetic), `reference_parity/` (needs pilot artifacts), `fixtures/` |
| `notebooks/` | The nine pilot notebooks, outputs stripped (reference only) |
| `docs/` | Scope & scale directive (authoritative for model matrix), build guide, agent quick-start, data dictionary, `protocols/` |

Notebook → module: 02 `feather`,`data` · 03 `models`,`extraction` · 04 `representations/pca` · 05/06 `representations/sae`,`training`
· 07 `features` · 08 `evaluation`,`statistics` · 09 `visualization`.

## Reproducing the pilot (do this first)

The pilot PCA, SAE checkpoints, training mean and activations are **immutable reference artifacts**. Never refit,
retrain or regenerate them.

1. Stage the archived Drive outputs (symlinks, read-only use):
   `scripts/stage_pilot_artifacts.sh /path/to/metamer-sae-drive-root`
2. Edit `configs/experiments/pilot_resnet50_convnext_stage3.yaml`: set `pca.artifact` to the archived PCA file
   (`.joblib` is loaded as a *trusted* local scikit-learn pickle; add `artifact_sha256` to verify it), and point
   `stimuli.split_manifest` at the notebook's `fixed_triplet_split_ledger.csv` so `activation_index` matches the old
   activation pools. (Without it, `metamer-sae write-pilot-ledger` writes the same split with new row indices, which
   is only valid for pools you re-extract yourself.)
3. Run `scripts/run_pilot_parity.sh`. It validates the config, audits the split, evaluates development then primary,
   and compares against `tests/fixtures/reference_values.json` (PCA/SAE validation MSE, primary L2 means,
   32/32 direction, 17 contrasts, Holm range) with explicit tolerances. A non-zero exit means a parity failure:
   investigate before scaling.

## Commands

```bash
metamer-sae validate-config  --config CFG
metamer-sae split-audit      --config CFG                 # disjointness, counts, SHA-256
metamer-sae feather-inventory --config CFG [--all-conditions]   # network: availability per ID/condition
metamer-sae download         --config CFG                 # validated download + stimulus manifest
metamer-sae extract          --config CFG --target natural|feather|all
metamer-sae fit-pca          --config CFG                 # NEW run; natural train only; pilot PCA untouched
metamer-sae train-sae        --config CFG --seed 123      # NEW run directory; never overwrites checkpoints
metamer-sae evaluate         --config CFG                 # dev, then primary; tables, figures, closure report
metamer-sae parity           --config CFG                 # evaluate + compare to the pilot reference
```

Expanded study (audit before anything expensive; network where marked):

```bash
metamer-sae validate-matrix  --config configs/experiments/alexnet_core_generator_native.yaml
metamer-sae resolve-registry --config CFG     # network: which registry conditions exist in the pinned Feather metadata; adversarial-IPCL gate
metamer-sae matrix-inventory --config CFG     # network: model/layer/stimulus inventory, condition + pair manifests, availability
metamer-sae matrix-split     --config CFG     # grouped source-level split; every condition of a source inherits it
metamer-sae fit-space        --config CFG --space-dir outputs/activations/<space_id>   # natural-only PCA + SAE seeds per activation space
metamer-sae image-baselines  --pairs ... --stimuli ...    # pixel/SSIM/LPIPS + smoothness
metamer-sae human-inspect / human-link                    # human stage gate (no analysis until linkage validates)
metamer-sae coverage | acceptance                         # coverage heatmap; evidence-based completion audit
```

Each run writes `outputs/runs/<run_id>/` containing the resolved config, `provenance.json` (project and Feather
commits, package/GPU versions, config/PCA/checkpoint/split hashes), tables, figures and `closure_report.json`
with SHA-256 for every artifact. Runs are immutable; a changed config is a new run.

## Scientific rules enforced in code

1. PCA and SAE are fit on natural-image **train** activations only; validation picks representations; Feather
   stimuli never enter fitting. One shared frozen representation is applied to all conditions.
2. SAE preprocessing = subtraction of the saved train mean only (no scaling/whitening); PCA is mean-centred, unwhitened.
3. Pilot split is fixed (8/32/8). `FeatherPools` raises `PermissionError` on reserve rows. Full-scale splits are by
   **source identity** (`make_source_level_split`) and hashed.
4. Bootstrap resamples triplets (paired sources), not image rows. Holm correction spans the 17-contrast family
   (3 metrics × 5 spaces + support Jaccard × 2 SAEs). Exact p-values are saved; never report "0".
5. SAE feature IDs are only comparable within a seed; use `match_dictionaries` and report matching quality.
6. Checkpoints load with `weights_only=True` and strict state-dict matching. Overcomplete is pinned to 0.3.0.
7. Metrics keep their directions and units (see `docs/DATA_DICTIONARY.md`); spaces are not put on a shared axis.
8. No human behavioural data is loaded anywhere in the model-only pipeline.
9. Model identity is registry data. Unverified conditions are never trusted; unresolved/ambiguous ones are reported as gaps
   (`resolve_registry`). Adversarial, random-perturbation, self-supervised IPCL and matched supervised conditions are distinct
   objectives with explicit norm/ε; adversarial-IPCL is never inferred.
10. Layer names are not equivalences across architectures. Condition ids keep Feather stage identifiers verbatim
    (`<condition>|<experiment>|<stage>`); contrasts match stages by exact name; activation spaces are keyed by
    model + checkpoint hash + stage + module path + preprocessing + pooling (`ActivationSpace.space_id`).
10b. Each model has its own `PreprocSpec`; none is inherited from another model. Pooled and spatial activations are stored apart.
11. Dictionary sharing follows the track: never per image label, never on Feather images (`assert_natural_only` hashes the
    fitting corpus against Feather stimuli). Cross-seed feature indices require a matching report.
12. Missing source/metamer pairs stay in the pair manifest (`pair_complete=False`); complete-case subsets are explicit tables.
    A source keeps one split across all conditions (`assert_split_consistency`).
13. The replication unit is the **source**; contrasts report `n_sources`, Holm is per declared family, exploratory families are
    labelled. Raw distances are not ranked across representations; use within-representation or natural-scaled values.
14. The human stage is gated by `validate_linkage`; condition-mean data are never bootstrapped over participants or turned into
    per-image labels; recognition-model accuracy is never reported as human accuracy.
15. Runs are immutable; reviewed runs are hash-verified (`mark_reviewed`) and reruns get new ids.

## Staged execution of the expanded study

Follow scope §11; do not skip ahead. **A** pilot parity (`scripts/run_pilot_parity.sh`) and `scripts/run_expanded_audit.sh` →
**B** AlexNet (standard / L2 ε=3 / random control) → **C** IPCL + matched supervised → **D** adversarial-IPCL feasibility
(found → add to registry; absent → report the gap, protocol in `docs/protocols/`) → **E** all eligible stages (or a justified,
pre-declared subset) with Track A observer panel + Track B generator-native + baselines → **F** human recognition after linkage
validates → **G** brain extension (separate). For every step: probe availability per condition (the pilot found 397, not 400, complete
standard triplets), split by source, extract per activation space, fit on natural images only (validation MSE on natural
validation data selects sizes — never Feather results), evaluate, and report per generator/stage/observer, never as an unlabelled average.
Adding a model: `docs/protocols/NEW_MODEL_CHECKLIST.md`.

## Status and honest limitations

**Executed and tested here (synthetic data, no GPU/network):** configs and registry validation, resolution/ambiguity/gap
reporting, adversarial-IPCL gate, grouped splits and manifests with missing conditions, track/dictionary guards, run-metadata
lifecycle, immutable runs and `mark_reviewed`, matrix evaluation, source-level contrasts + Holm per family, cross-model matrix,
smoothness covariates, recognition-model bookkeeping, and the human-linkage gate/analysis on simulated tables
(plus the original pilot tests). `tests/unit/test_adapters.py` (ResNet-50/AlexNet/ConvNeXt hook extraction with random weights)
and the Overcomplete SAE tests are written but **skipped in the authoring sandbox** (no working torch/overcomplete) — run `pytest` on your machine.

**Written but not executed against real data:** observer/adapter extraction on real checkpoints, `FeatherAdapter` (the Feather
build-script/stage-map contract is configuration and **must be verified against the pinned commit**), per-space SAE fitting,
Feather network calls/downloads, `matrix-inventory`, and the pilot reference-parity run. Treat any deviation as a bug.

**Not yet done / needs a human decision:**
- Real Feather identifiers for AlexNet, IPCL and the matched supervised model are *unverified* (registry `verified: false`);
  `resolve-registry` + manual pinning is Phase B/C step 1. Whether an adversarial-IPCL condition exists is unresolved until that audit.
- Official 16-class mapping (`CategoryMap` csv), IPCL readout training data, and the actual behavioural file layout are not in this repo;
  the human stage cannot run before `human-inspect`/`human-link` on the real files.
- Mixed-effects models (participant × stimulus) are not implemented; condition-level partial Spearman with bootstrap is.
- Spatial observer/PCA diagnostics, feature galleries, a cross-dictionary *transfer* runner, and a full multi-observer evaluation
  driver (the building blocks exist) remain to be assembled once real activations exist. Phase G (brain/fMRI) is out of scope.
- Pilot artifact file names remain config inputs. Do not retrain/refit pilot PCA/SAE.

## References

Feather et al. (2023) *Model metamers reveal divergent invariances between biological and artificial neural networks*,
Nat Neurosci 26, 2017–2034. Conwell et al. (2024) *A large-scale examination of inductive biases shaping high-level
visual representation in brains and machines*, Nat Commun. Wu et al. (2025) *AxBench*, ICML. Feather code:
https://github.com/jenellefeather/model_metamers_pytorch (pinned `d9629b58ad43ce9a6bd74be951869cffbd8e8161`).
