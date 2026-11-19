# METAMER-SAE — Agent Quick-Start

**Read this together with:** `METAMER_SAE_FULL_SCALE_BUILD_GUIDE.md`  
**Project:** `metamer-sae`  
**Current milestone:** transition from completed Colab prototype to full-scale research codebase.

---

## 1. What is this project?

Metamer-SAE investigates which internal visual feature representations differ when standard-trained and adversarially trained networks produce model metamers with different perceptual properties.

The original proposal is titled *Why Model Metamers Appear Wrong from Their Feature Representation Maps*. Its central question is:

> Which features make a model blind to what we see?

The project is inspired by Feather et al. (2023), *Model metamers reveal divergent invariances between biological and artificial neural networks*.

A model metamer is an image optimised to match a selected internal activation of a fixed generator network, while its pixels may differ from the natural source.

The project uses:
- Natural-image-trained PCA as a linear baseline.
- Natural-image-trained sparse autoencoders to represent observer activations.
- Paired source/metamer comparisons.
- A later, separate comparison with Feather's existing human-recognition data.
- A possible future brain/fMRI extension.

## 2. The most important distinction

**The Colab notebooks are the prototype. The `metamer-sae` repository is the full research implementation.**

The notebooks demonstrate and validate the workflow on a deliberately limited pilot subset. The repository must convert that logic into reusable modules and scale it to the full eligible Feather stimulus/model/layer inventory.

Do not make the repository a copy of the notebooks. Build a tested package around the validated methods, configs, data contracts and provenance.

## 3. Completed notebooks

There are ten notebook files in the Notebook 01–09 sequence because Notebook 04 has a separate visualisation extension.

| Notebook | Purpose | State |
|---|---|---|
| 01 | Environment and Feather repository audit | Reviewed/completed |
| 02 | Stimulus collection, pairing and fixed split | Completed; downloads and galleries validated |
| 03 | ConvNeXt observer and activation extraction | Completed; integrity verification passed |
| 04 | Pooled PCA baseline | Fit and assessed |
| 04B | PCA visualisation extension | User-reported extension; verify archived outputs when packaging |
| 05 | Overcomplete environment and SAE preparation | Completed |
| 06 | TopK SAE training and checkpoint selection | Completed; eight-run grid, audits and extended checkpoints |
| 07 | TopK feature analysis | Completed; Feather development split only |
| 08 | Paired metamer evaluation | Completed; development and primary evaluation |
| 09 | Integrated model-level results | Executed; visual review still pending |

Do not rerun earlier notebooks or repeat downloads unless an integrity audit finds a real problem.

## 4. Pilot configuration — preserve exactly

### Feather generator conditions

| Role | Feather key | Model | Layer |
|---|---|---|---|
| Standard | `experiment_1` | `resnet50` | `layer2_fake_relu` |
| Adversarial | `experiment_3` | `resnet50_l2_3_robust` | `layer2_fake_relu` |

Feather repository commit:

`d9629b58ad43ce9a6bd74be951869cffbd8e8161`

### Common observer

- TorchVision ConvNeXt-Tiny, ImageNet-1K pretrained.
- Frozen weights.
- Stage 3.
- Spatial output `(384, 14, 14)`.
- Global-average-pooled output `(384,)`.
- Official pretrained-weight preprocessing.

The observer is separate from the ResNet-50 metamer generators.

### Pilot data

- Imagenette train: 9,469 images.
- Imagenette validation: 3,925 images.
- Feather pilot: 48 complete triplets / 144 images.
- Total activations extracted: 13,538 images.

Fixed Feather split (seed 42):
- Development: 8 triplets.
- Primary evaluation: 32 triplets.
- Reserve: 8 triplets.

Never modify the pilot allocation. The reserve activation rows remain unused.

### PCA

- Run ID: `pca_20261119T065920Z`.
- Fit on Imagenette train pooled activations only.
- Full 384-component basis saved.
- PCA-16 is working baseline.
- PCA-64 is sensitivity analysis.
- Mean-centering only; no channel standardisation or whitening.

### SAE

- Framework: Overcomplete 0.3.0.
- Validated environment: Python 3.12.13, PyTorch 2.11.0+cu128, TorchVision 0.26.0+cu128, Tesla T4.
- Actual import: `from overcomplete.sae import TopKSAE, train_sae`.
- TopK architecture: input 384, dictionary width 1,536 (4×), K=64.
- Seed 123 is primary; seed 42 is replicate.
- SAE preprocessing: saved Imagenette training-mean subtraction only.
- Selected seed-123 best validation MSE: `0.0016301164756807836`, best epoch 125.
- Selected seed-42 validation MSE: `0.00163357109`, best epoch 123.

Do not retrain these checkpoints for parity. Do not use unrestricted pickle loading.

## 5. Main model-level findings

Notebook 08/09 primary evaluation (32 triplets):

- Standard-metamer L2 distance is greater than adversarial-metamer L2 distance in **32/32 triplets in each of five spaces**: raw observer, PCA-16, PCA-64, SAE seed 123 and SAE seed 42.
- The paired L2 mean differences are about 1.16–1.63 depending on representation.
- All 17 saved paired contrasts survived Holm correction at adjusted \(p < 0.001\).
- Adversarial metamers have higher source-to-metamer active-support Jaccard in both SAE seeds: about 0.678 versus 0.533–0.536 for standard.
- Cosine difference distributions are skewed; inspect means and medians together.
- PCA and SAE reconstruction residuals show different patterns.
- Raw spatial observer diagnostics are supplementary, not spatial SAE maps.

These are model-representation findings. They do **not** establish human recognisability, semantic feature identity, causality, or SAE superiority to PCA.

## 6. Scientific/data rules that must not be broken

1. Fit PCA and SAE on natural-image train activations only.
2. Use natural-image validation for representation selection.
3. Keep all Feather source/metamer stimuli out of fitting.
4. Apply the same frozen representation to all paired conditions.
5. Keep generator, observer and recognition-model roles separate.
6. Keep pooled SAE and spatial observer/PCA analyses separate.
7. Never equate feature ID `i` across SAE seeds without explicit matching.
8. Keep development, primary and reserve split boundaries intact.
9. Bootstrap paired sources/triplets, not independent image-condition rows.
10. Preserve metric directions and units in every table/figure.
11. Do not report a rounded p-value of zero; use the saved exact values/scientific notation.
12. No human behaviour in the current model-only stage. Human-recognition analysis is later and must respect condition-level data aggregation, layer-depth confounding, image smoothness, baseline comparisons and human reliability.
13. Brain/fMRI is a separate future extension.

## 7. What to do next

**Do not start another SAE sweep. Do not refit PCA. Do not regenerate the pilot stimuli or activations.**

Begin the full repository build in this order:

1. Obtain/confirm the actual `metamer-sae` GitHub repository URL and inspect its branch, working tree, source, notebooks, configs and existing artifacts.
2. Produce a non-destructive repository audit. Do not overwrite or move files before understanding the current structure.
3. Create the package scaffolding, typed configs, data dictionary, manifests, run IDs and provenance system.
4. Implement the pilot reference configuration.
5. Build tests that reproduce the existing Notebook 04/06/08/09 numerical outputs within explicit tolerances.
6. Port the Feather adapter, pairing, observer loader, activation extractor, PCA, Overcomplete wrapper, feature analysis, paired metrics, statistics and visualisation modules.
7. Only after pilot parity passes, inventory and process the full eligible Feather stimulus/model/layer matrix.
8. Expand generator conditions, layers and observers incrementally, not all at once.

The complete architecture, output paths, proposal requirements, per-notebook history, methodological cautions, statistical details, config examples, risk register and acceptance criteria are in `METAMER_SAE_FULL_SCALE_BUILD_GUIDE.md`.

---

## 8. Full guide contents

The master guide is the authoritative detailed handover and includes:

- Original proposal motivation, research questions and hypotheses.
- The difference between proposal scope and pilot choices.
- Scientific pipeline and generator/observer distinction.
- Feather repository audit, data acquisition, fixed pilot IDs and full-scale split strategy.
- Detailed Notebook 01–09 history and results.
- Exact PCA and SAE preprocessing/checkpoint provenance.
- Notebook 07 feature-use and seed-matching evidence.
- Notebook 08 paired metrics, statistics and confidence intervals.
- Notebook 09 integrated results and figures.
- Proposed modular repository architecture.
- Phased full-scale implementation plan.
- Later human-recognition and future brain/fMRI requirements.
- Testing, reproducibility, data contracts, risks, acceptance criteria and agent instructions.

**When in doubt, preserve the existing artifacts and follow the master guide rather than inventing a new experimental design.**
