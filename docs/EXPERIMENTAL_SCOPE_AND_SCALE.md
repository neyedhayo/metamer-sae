# METAMER-SAE — Experimental Scope, Model Matrix, and Scale Specification

**Document type:** Research-scope README and implementation directive for the full-scale `metamer-sae` repository  
**Project:** Metamer-SAE  
**Updated:** 20 November 2026  
**Status:** Expanded experimental scope — ResNet-50, AlexNet, and IPCL are all in scope

> **Purpose of this document**  
> This README tells the repository-building agent what scientific and experimental scale the full `metamer-sae` codebase must support. It explains what the completed Colab prototype actually tested, what the full research implementation must add, and how the expanded model families and training conditions must be represented. It is a research-scope companion to the full-scale build guide and agent quick-start; it is not a replacement for their detailed software architecture, implementation phases, or acceptance tests.

---

## 0. Instructions to the repository-building agent

You are extending an existing research project. Do not treat the ResNet-50 prototype as the entire scientific scope.

The completed notebooks established a controlled pilot using one standard ResNet-50 generator, one adversarially trained ResNet-50 generator, one generator layer, one common observer, one natural-image activation dataset, and a fixed stimulus split. The full repository must first reproduce that pilot, then support the broader model-family study described here.

### Required model-family scope

The expanded study must explicitly support:

1. **ResNet-50:** standard/supervised and adversarially trained generator conditions, preserving the existing pilot pair and allowing additional verified Feather robustness variants and controls.
2. **AlexNet:** standard/supervised and adversarially trained generator conditions, including the L2, \(\epsilon=3\) condition described in the proposal, and the corresponding random-perturbation control when supplied by Feather.
3. **IPCL:** the published self-supervised IPCL model (an AlexNet-family architecture modified with Group Normalization) and its architecture-/augmentation-matched supervised comparison. IPCL is a required model family, not a disposable optional afterthought.

**Important qualification about IPCL:** IPCL is a self-supervised training method/model condition. The Feather paper documents IPCL and a matched supervised comparison, but the sources reviewed for this project do not establish an adversarially trained IPCL counterpart equivalent to the standard-versus-adversarial ResNet-50 and AlexNet pairs. The agent must search the official Feather metadata, checkpoints, and stimulus inventory for such a condition. If no matching adversarial-IPCL checkpoint and metamer stimuli exist, record that as an availability gap. Do not relabel self-supervised IPCL as adversarial. A newly adversarially trained IPCL model would be a separate, new experiment requiring a defined robust-training recipe, checkpoint, metamer generation, optimization-fidelity validation, and a separately versioned dataset.

The user explicitly wants the study to extend beyond ResNet-50 to AlexNet and IPCL. Therefore:

- AlexNet standard and adversarial conditions are **core required experimental conditions**, not optional future work.
- IPCL and its matched supervised comparator are **core required experimental conditions**.
- An adversarial-IPCL comparison is a specifically requested feasibility/extension target. It must be included if a source-supported condition exists; otherwise specify what new training and metamer-generation work would be required before implementing it.
- Do not silently drop AlexNet or IPCL because the initial prototype used ResNet-50.

### Non-negotiable rules

1. Preserve the existing pilot artifacts and exact 8-development / 32-primary / 8-reserve split. Never overwrite the original PCA or SAE checkpoints.
2. Keep pilot reproduction separate from expanded experiments. Never describe changed settings as an exact reproduction.
3. Never fit PCA or SAE dictionaries on Feather source images or model metamers. Fit representations on natural-image training activations only.
4. Preserve the distinction between generator models, common observer models, and recognition-baseline models.
5. Preserve the distinction between standard supervised, adversarially trained, random-perturbation control, and self-supervised/IPCL conditions.
6. Do not assume that identical layer names or layer indices mean equivalent representations across AlexNet, ResNet-50, and IPCL.
7. Do not assume that every model has the same available stages, input preprocessing, output head, stimulus IDs, or behavioral data.
8. Do not manufacture missing model variants, checkpoints, metamer stimuli, or behavioral observations. Missing conditions must be reported explicitly.
9. Preserve both spatial and pooled activations. The pilot's pooled 384-D SAE is not a spatial SAE.
10. Do not compare raw activation distances across different architectures/layers as though they share a common scale. Use within-representation comparisons and appropriately normalized or standardized summaries.
11. Do not compare SAE feature indices across separately trained dictionaries or seeds without explicit feature matching and reported matching quality.
12. Do not infer human recognizability from model activation metrics, image metrics, reconstruction error, or classifier accuracy.
13. Do not interpret SAE features as semantic concepts or causal units without suitable validation or intervention.
14. Keep human-behavior and brain/fMRI analyses as distinct stages with their own data and statistical units.
15. Do not declare a stage complete merely because code executes. Audit the outputs, data coverage, figures, manifests, and closure reports.

---

# 1. Project identity and scientific objective

Metamer-SAE investigates whether sparse autoencoder (SAE) features provide a useful description of how natural images and model-generated metamers differ in visual network representations, whether these differences vary by model architecture and training procedure, and whether feature-level measures relate to human recognition.

The project is inspired by Feather et al. (2023), *Model metamers reveal divergent invariances between biological and artificial neural networks*. A model metamer is an image optimized to match a reference natural image's activation at a selected layer or stage of a fixed model. The model weights remain fixed during metamer optimization; the image is optimized.

The central research motivation remains:

> **Which features make a model blind to what we see?**

The project moves beyond raw activation-vector comparisons by learning representations of natural-image activations using PCA and SAEs, then applying those frozen representations to paired natural sources and model metamers.

## 1.1 Research questions

The full repository should support the following connected questions:

- **RQ1 — Feature-space structure:** Do SAE features reveal source-image versus metamer differences that are less visible or differently structured in the original channel/neuron activation space?
- **RQ2 — Training-condition differences:** Do these differences vary between standard/supervised and adversarially trained models?
- **RQ3 — Model-family differences:** Do the patterns differ across ResNet-50, AlexNet, and IPCL/self-supervised models, including matched architecture/training comparisons where possible?
- **RQ4 — Layer dependence:** How do the patterns vary across the model stages at which the metamers were generated?
- **RQ5 — Representation comparison:** How do SAE representations compare with PCA and simpler image/recognition baselines in reconstruction, sparsity, stability, source–metamer discrimination, and relation to human recognition?
- **RQ6 — Observer agreement:** Do independent observer networks agree about which metamers remain recognizable or produce similar internal representations?
- **RQ7 — Human recognition:** Do feature-level metamer measures relate to existing human recognition performance across model families, generator stages, and stimulus conditions?
- **RQ8 — Future brain extension:** Do model-layer or SAE feature geometries align with human visual-cortex response geometries measured using suitable existing public fMRI data?

The hypotheses remain hypotheses. The software must allow null, mixed, and model-specific findings; it must not build in the expectation that SAE scores will outperform PCA or that all robust/self-supervised models will produce the same effect.

---

# 2. The two implementation milestones must remain distinct

## 2.1 Completed experimental prototype

The prototype was deliberately narrow. It validated a manageable end-to-end method before broadening the study.

| Component | Actual prototype configuration |
|---|---|
| Standard generator | Feather `experiment_1`, `resnet50` |
| Adversarial generator | Feather `experiment_3`, `resnet50_l2_3_robust` |
| Generator stage | `layer2_fake_relu`, Feather layer index 2 |
| Common observer | Frozen TorchVision ConvNeXt-Tiny, ImageNet-1K weights |
| Observer feature stage | Stage 3 |
| Observer spatial output | 384 × 14 × 14 |
| Observer pooled output | 384 dimensions |
| Natural-image activation dataset | Imagenette-320 |
| PCA | 384-component fitted basis; PCA-16 working baseline; PCA-64 sensitivity analysis |
| SAE | Overcomplete TopK, 384 inputs, width 1,536 (4×), K=64 |
| Final SAE seeds | 123 primary; 42 independent replicate |
| Feather pilot allocation | 8 development / 32 primary / 8 reserve triplets |

The pilot investigated one standard/adversarial ResNet-50 comparison. It did **not** complete the AlexNet, IPCL, full multi-layer, recognition-baseline, or human-recognition portions of the broader study.

## 2.2 Full-scale repository

The repository must convert the validated prototype into a configurable research pipeline and extend it to the required model families and eligible layers. It must support:

- ResNet-50 standard and adversarial generator conditions.
- AlexNet standard and adversarial generator conditions, plus matched random-perturbation controls where available.
- IPCL self-supervised and its matched supervised AlexNetGN comparison.
- An explicit availability audit for adversarial-IPCL; a separately designed adversarial-IPCL training/generation extension only if no official matching condition exists and the new experiment is approved.
- All eligible Feather-defined generator stages for the selected conditions, or a predeclared layer subset with clear scientific justification. Do not assume `layer2_fake_relu` is a universal layer name.
- A consistent observer panel, plus generator-native representation analysis where applicable.
- Reusable natural-image activation generation, PCA/SAE training, held-out metamer evaluation, image metrics, recognition baselines, and later behavioral analysis.

The full repository is not merely a conversion of the prototype notebooks into Python modules. It is the controlled experimental framework for the expanded research design.

---

# 3. Expanded generator model and training-condition matrix

The model matrix is the most important scope update in this README. The agent must implement it as data/configuration, not as hard-coded `if model == ...` branches scattered throughout the code.

## 3.1 Core model-family matrix

| Model family | Required condition(s) | Additional control/comparison | Status and implementation rule |
|---|---|---|---|
| ResNet-50 | Standard/supervised | Adversarially trained ResNet-50; random-perturbation control where an official matching condition exists | Pilot standard and L2 robust \(\epsilon=3\) pair must be reproduced exactly. Inventory other official variants separately. |
| AlexNet | Standard/supervised | Adversarially trained AlexNet, including L2 \(\epsilon=3\); random-perturbation-trained AlexNet control | Core expansion. The proposal identifies Feather Experiment 4 as a three-condition AlexNet comparison. Audit exact checkpoints, generated stages, and stimuli. |
| IPCL / AlexNetGN | Self-supervised IPCL | Matched supervised AlexNetGN comparison using the same architecture/augmentation setup described by Feather | Core expansion. Preserve IPCL as a distinct training objective, not as a synonym for adversarial robustness. |
| IPCL adversarial extension | Adversarially trained IPCL/AlexNetGN, if source-supported | Matched non-adversarial IPCL and architecture-matched supervised baselines | Search official metadata/checkpoints first. If absent, mark as a new experiment requiring a robust-training recipe, model checkpoint, generated metamers, and optimization validation. Do not fabricate this condition. |

### Why this distinction matters

The Feather paper reports that adversarial training was evaluated for both ResNet-50 and AlexNet architectures. It also reports a separate self-supervised-learning experiment that includes IPCL. IPCL uses an AlexNet architecture modified with Group Normalization and is compared with a supervised model using the same architecture. The paper's description of IPCL as self-supervised does not establish that IPCL itself is an adversarially trained model.

Therefore, the scientific comparison should be organized into two related axes:

1. **Robustness/training-type axis:** standard supervised vs adversarially trained vs random-perturbation control, especially for ResNet-50 and AlexNet.
2. **Learning-objective/architecture axis:** self-supervised IPCL vs its matched supervised AlexNetGN comparator.

An adversarial-IPCL experiment can add a third bridge between these axes, but it must be verified or newly created as a distinct experiment.

## 3.2 AlexNet — required conditions

AlexNet must not be treated only as an optional recognition baseline. It is a required **generator family** in the expanded study.

At minimum, audit and support:

- Standard/supervised AlexNet.
- Adversarially trained AlexNet with the proposal's L2, \(\epsilon=3\) condition.
- Random-perturbation-trained AlexNet control, if the corresponding Feather checkpoint and metamers are supplied.
- All corresponding official Feather generator stages and available source/metamer pairs.

The proposal identifies Experiment 4 as the initial AlexNet experiment, with stages described as `relu0`–`relu4`, `fc0`, `fc1`, and `final`. Treat these as expected identifiers to verify against the real Feather metadata and implementation; do not hard-code them without auditing the actual model objects and asset manifests.

Preserve perturbation type and magnitude in condition metadata. Do not collapse distinct adversarial training norms, epsilon values, or random-perturbation controls into one generic `robust` label.

## 3.3 ResNet-50 — preserve the pilot and extend it

Required baseline:

- Standard Feather ResNet-50: `experiment_1`, `resnet50`.
- Adversarial Feather ResNet-50: `experiment_3`, `resnet50_l2_3_robust`.
- Pilot generation layer: `layer2_fake_relu`.

Then inventory other official adversarial ResNet-50 variants, perturbation norms/magnitudes, control variants, and available generator stages. Each distinct training recipe is a separate condition with its own identifier and source metadata.

Do not overwrite the pilot configuration to make it look like a broader robustness sweep.

## 3.4 IPCL — required self-supervised comparison

IPCL must be included as a first-class experimental model family.

The source paper describes IPCL as a self-supervised model with an AlexNet-like backbone modified using Group Normalization. It also describes a supervised comparison model with the same architecture. The exact IPCL and comparison checkpoints, input preprocessing, augmentations, readout setup, and metamer stages must be verified from Feather's model-analysis folders and code.

The agent must:

1. Locate the official IPCL model configuration and checkpoint.
2. Locate the matched supervised AlexNetGN comparison and confirm what training data and augmentations are matched.
3. Identify the exact model modules/stages used to generate IPCL metamers.
4. Inventory the corresponding source images, metamer images, optimization summaries, and behavioral-condition records.
5. Determine how final/readout stages are represented. Do not assume a self-supervised model has the same native classification head as a supervised classifier. Feather's self-supervised comparisons may use a trained linear readout for task-level outputs; reproduce the documented setup where required.
6. Keep the IPCL objective and matched supervised comparison separate from standard-vs-adversarial training labels.
7. Search for an official adversarial-IPCL checkpoint and matched metamers. If absent, report the missing condition and write a separate proposed protocol for an adversarial-IPCL extension. Do not start expensive new training or synthesize a supposed published condition without an explicit experimental specification.

## 3.5 Random-perturbation controls

Where Feather provides models trained with random perturbations, retain them as their own experimental condition. They are important controls because they help distinguish the effect of adversarially directed training from merely exposing a model to perturbations during training.

Do not pool random-perturbation controls with standard models or adversarially trained models. Their training type, perturbation magnitude, and source identifiers must remain explicit.

## 3.6 Optional source-supported model conditions

The Feather inventory may include other visual model families, training variants, dataset diets, or model configurations. The repository architecture should be able to represent them, but the core planned study is ResNet-50 + AlexNet + IPCL. Any additional model should be added only after a scope decision and availability audit; do not let additional models displace the three required families.

---

# 4. Generator stages and layer coverage

The prototype used only one generator stage. The expanded research must investigate multiple Feather-defined stages across the required model families.

## 4.1 Layer inventory

For each model configuration, inventory:

- Official Feather stage/module identifier.
- Human-readable stage label.
- Module path in the actual loaded model.
- Tensor shape and data type for a test batch.
- Whether the stage is a valid metamer-generation stage under Feather's methodology.
- Whether downstream branches converge appropriately (especially for residual architectures).
- Whether metamer stimuli and optimization summaries exist for that stage.
- Whether a corresponding human-behavioral condition exists.

Use Feather's designated generation stages when available. A layer that can be hooked computationally is not automatically a valid or published metamer-generation condition.

## 4.2 Comparability across architectures

Do not equate AlexNet `relu2`, ResNet-50 `layer2`, and an IPCL module merely because they are early/intermediate or have similar ordinal positions. Preserve their original identifiers and compare within model/stage first. For cross-model summaries, use explicit stage grouping or depth metadata only when that mapping is scientifically justified.

## 4.3 Pilot versus expanded layers

- **Pilot parity:** retain only ResNet-50 `layer2_fake_relu` for exact reproduction.
- **Expanded study:** include all eligible stages available for the selected ResNet-50, AlexNet, and IPCL conditions, or explicitly define a layer subset before confirmatory evaluation.
- Record unavailable or unpaired stages rather than quietly skipping them.

---

# 5. Dataset scope and data governance

There are two different image datasets in the pipeline. They must never be conflated.

## 5.1 Feather source images and metamers — evaluation stimuli

The original proposal describes a Feather visual dataset built from approximately 400 base natural photographs across 16 object categories and multiple model/layer conditions. It identifies approximately 40 model/layer conditions as a starting inventory. Those are project-document estimates, not a guarantee that every condition contains 400 valid, accessible, paired stimuli.

The pilot checked candidate IDs `0–399` for its selected ResNet-50 standard/adversarial pair:

| Availability audit | Count |
|---|---:|
| Candidate IDs checked | 400 |
| Accessible source images | 400 |
| Accessible standard metamers | 397 |
| Accessible adversarial metamers | 400 |
| Complete source + standard + adversarial triplets | 397 |
| Triplets selected for pilot | 48 |

Three candidate IDs (`207`, `240`, `336`) lacked a standard metamer in the pilot's selected condition and were excluded from the complete triplet pool.

Pilot allocation:

| Split | Triplets | Images (source + standard + adversarial) |
|---|---:|---:|
| Development | 8 | 24 |
| Primary evaluation | 32 | 96 |
| Reserve | 8 | 24 |
| **Total** | **48** | **144** |

The expanded repository must perform a **separate availability scan for every model × training condition × generator stage**. It must not assume that the pilot's 397 complete triplets, 48 selected triplets, or 400 accessible source images apply to AlexNet, IPCL, or any other condition.

### Expanded stimulus inventory requirements

For every available condition:

- Identify the reference/source image and its stable source ID.
- Identify the exact generator checkpoint and model-training condition.
- Identify the metamer generation stage.
- Record perturbation/training variant (standard, adversarial norm and epsilon, random-perturbation control, self-supervised IPCL, supervised matched IPCL architecture, etc.).
- Record available standard/metamer/control image files and checksums.
- Record image dimensions, format, channel mode, decoding status, and preprocessing metadata.
- Record metamer optimization metadata and success/failure summaries where supplied.
- Record human behavioral data linkage and the true aggregation unit.
- Mark missing or corrupt files explicitly.
- Produce condition-level counts and completeness reports.

A source image can have multiple metamer versions across architectures, layers, and conditions. All versions must preserve the same source identity. Repeated measurements from one source are not independent source images.

### Dataset manifests

At minimum, create:

- `model_inventory.csv` / `.json`
- `layer_inventory.csv` / `.json`
- `stimulus_availability.csv`
- `stimulus_manifest.csv` (one row per image/file)
- `metamer_pair_manifest.csv` (one row per source–metamer pairing)
- `condition_manifest.csv`
- `human_behavior_manifest.csv` (only metadata/links until the behavioral stage is explicitly run)
- `download_validation_report.json`
- `split_manifest.json`

Each manifest must carry stable IDs, checksums, model configuration, stage, condition, source ID, split, and provenance.

## 5.2 Splitting and leakage prevention

Keep the pilot split immutable:

- Development: `[0, 260, 308, 84, 199, 119, 97, 27]`
- Primary: `[116, 101, 133, 125, 232, 362, 227, 326, 398, 358, 54, 78, 288, 45, 295, 249, 329, 197, 108, 393, 234, 59, 205, 245, 102, 343, 52, 241, 286, 81, 383, 361]`
- Reserve: `[46, 255, 186, 169, 395, 134, 194, 62]`

For the expanded study, define source-level grouped splits across the full model/layer inventory. All condition-specific metamers derived from a given source must inherit the same split. Never let the same source image appear in development for one model and primary evaluation for another if the intention is to treat the evaluation as source-held-out.

If the full inventory has missing conditions, preserve incomplete source groups in an availability table. Define complete-case subsets per planned comparison transparently rather than deleting rows without explanation. Use stratification by category only if reliable category metadata is available; record the rule and seed.

## 5.3 Natural-image dataset — representation training data

The prototype used Imagenette-320:

| Split | Images |
|---|---:|
| Training | 9,469 |
| Validation | 3,925 |

The full repository must retain this exact dataset and split for pilot parity. It may support larger or alternative natural-image datasets, but no replacement is automatically approved by this README. Every new natural-image corpus must have a distinct dataset version, license/provenance note, split manifest, checksums, preprocessing record, and separate PCA/SAE run IDs.

The Feather source/metamer evaluation set must never be silently merged into the natural-image training set.

---

# 6. Training and representation learning

## 6.1 Frozen visual generators

Do not train the official Feather generator networks as part of ordinary pipeline reproduction. Load the source-provided checkpoints and verify their architecture, weights, preprocessing, and stage definitions. Any newly trained model (for example, an adversarial-IPCL extension) is a separate research experiment with its own training protocol and provenance.

## 6.2 Frozen observer panel

The pilot uses ConvNeXt-Tiny as a separate common observer. The expanded pipeline should support a configurable observer panel, including compatible ResNet-50, AlexNet, and IPCL/AlexNetGN models where their checkpoints and preprocessing are available.

Each observer requires its own correct preprocessing, layer mapping, activation metadata, and (where classification is evaluated) readout/classifier setup. A self-supervised IPCL model must not be assumed to have the same native classifier head as a supervised network; use the source-documented linear readout or a separately documented feature-similarity measure.

The same stimulus set should be passed through the same fixed observer panel to enable cross-observer comparisons. The observer is not the generator and is not a human perceptual measure.

## 6.3 Preserve two complementary representation-analysis tracks

The project materials describe both a common-observer pipeline and model-/layer-specific dictionaries. These are complementary analyses with different scientific questions. Implement both as explicitly named tracks rather than mixing their assumptions.

### Track A — Common-observer representation (cross-generator comparison)

Purpose: compare stimuli generated by different generator families in a consistent external feature space.

1. Select a fixed observer model and layer (the pilot is ConvNeXt-Tiny Stage 3).
2. Extract natural-image training activations from that observer/layer.
3. Fit PCA and SAE only on the natural-image training activations.
4. Freeze the resulting representation.
5. Pass source images and every generator condition's metamer through the same observer and same frozen PCA/SAE.
6. Compare source-relative effects across generator families and training conditions within that common space.

A given common observer/layer uses the same fitted PCA/SAE for the whole comparison family. Do not fit separate transforms to standard and adversarial metamer images merely to improve their apparent separation.

### Track B — Generator-native representation (model-internal feature analysis)

Purpose: inspect the internal feature representation of each generator model at the stage used to produce its metamers.

1. Extract natural-image training activations from the selected generator checkpoint and stage.
2. Fit PCA/SAE using only those natural-image activations.
3. Freeze that representation.
4. Pass the paired source image and its corresponding metamer through that same generator checkpoint/stage and transform.
5. Compute within-model source-to-metamer feature metrics.
6. Repeat for each generator model × training condition × stage that is included in the study.

Different generator checkpoints/stages may produce different activation spaces and may require different dictionaries. A separate dictionary is permitted when it is fitted to a genuinely different activation space, not merely because an image is labelled source, standard metamer, or adversarial metamer.

Do not compare absolute L2 values or raw feature indices across independently trained model-specific dictionaries as though their coordinates are aligned. Compare within-condition effects, normalized measures, stability, and carefully designed condition-level associations. A cross-model dictionary-transfer experiment is possible, but it must be separately labelled and must quantify distribution mismatch.

### Shared versus model-specific dictionary decision

- **Common-observer track:** one shared natural-image-trained dictionary per observer × layer × preprocessing configuration, reused across all generator conditions being compared.
- **Generator-native track:** one natural-image-trained dictionary per generator checkpoint/activation-space × stage, reused for the source and its paired metamers.
- **Transfer track:** a dictionary trained in one activation space applied to another only as a separately specified experiment.

Never train a dictionary separately on source images, standard metamers, and adversarial metamers. Never fit on Feather images for the main analyses.

## 6.4 Pilot PCA baseline — preserve exactly

The pilot PCA was fitted on 9,469 Imagenette training pooled activations of shape `(9469, 384)` using:

- `sklearn.decomposition.PCA(n_components=384, svd_solver="full", whiten=False)`
- Mean-centering only.
- No per-channel standardization.
- No whitening.
- PCA-16 as the working baseline.
- PCA-64 as the dimensionality sensitivity analysis.

The pilot run ID is `pca_20261119T065920Z`. Preserve this fit and the saved basis. New observer/layer/dataset combinations require their own fitted PCA objects and run IDs. Do not refit the pilot PCA in place.

## 6.5 Pilot TopK SAE — preserve exactly

The validated prototype configuration is:

| Parameter | Pilot value |
|---|---:|
| Input dimension | 384 |
| Dictionary width | 1,536 |
| Expansion | 4× |
| TopK | 64 |
| Nonzero codes per input | Exactly 64 |
| Primary seed | 123 |
| Independent replicate seed | 42 |
| Optimizer | Adam |
| Loss | MSE |
| Learning rate | 0.0005 |
| Batch size | 256 |
| Extended budget | Up to 200 epochs, with early stopping |

Selected checkpoints:

- Seed 123: best epoch 125; validation MSE approximately `0.00163011648`; 1,239 active features on validation.
- Seed 42: best epoch 123; validation MSE approximately `0.00163357109`; 1,248 active features on validation.

These are immutable pilot reference artifacts. Do not silently replace their architecture, centering, seeds, training data, package version, checkpoint, or selection procedure.

The prototype also tested an eight-run grid spanning 2×/4× expansions and K=16/32/64. Preserve the run records and audit results.

## 6.6 Expanded training framework

The repository must make the following configurable and reproducible:

- Observer/generator model checkpoint and layer.
- Natural-image dataset and split.
- Preprocessing, activation pooling, and feature dimensionality.
- PCA solver, retained dimensions, centering, scaling, and whitening.
- SAE family (initially TopK; optional Vanilla/BatchTopK/JumpReLU only as separately declared experiments).
- Dictionary width/expansion, K, optimizer, learning rate, batch size, scheduler, regularization, gradient clipping, epochs, early stopping, and checkpoint policy.
- Multiple random seeds.
- Hyperparameter sweep and model-selection criteria based only on natural-image training/validation activations.
- Strict checkpoint loading, config validation, and run provenance.

Every new model × layer × natural dataset condition must have an identifiable activation corpus and fitted representation run. Do not assume the pilot's 384-dimensional shape or 14 × 14 spatial size applies to AlexNet or IPCL.

---

# 7. Activation extraction and storage

For every selected generator and observer, record both:

- Spatial activation tensor: preserve the real channel and spatial dimensions.
- Pooled activation vector: use global average pooling or another explicitly declared pooling operation.

The pilot's ConvNeXt-Tiny Stage 3 tensors are `(384, 14, 14)` spatial and `(384,)` pooled. These shapes are specific to that observer and must not be hard-coded for other architectures.

The extraction system must support:

- Batched inference and GPU/CPU execution.
- Resume after interruption.
- Stable image IDs and exact row-to-metadata alignment.
- Checkpoint/model-weight hashes.
- Transform/preprocessing hashes.
- Layer/module identifiers and tensor shapes.
- Spatial and pooled arrays stored separately.
- Appropriate dtypes and memory-mapped/chunked access for larger data.
- Finite-value, shape, count, and metadata integrity checks.
- Cache keys that include image checksum, model/checkpoint hash, layer, preprocessing, pooling method, and code version.

Never assume AlexNet, ResNet-50, and IPCL use the same resize, crop, normalization, input channel order, or input resolution. Recover the official transformations and store them with each model.

---

# 8. Evaluation metrics and experimental analyses

The expanded repository must preserve the prototype analyses and add the missing proposal analyses.

## 8.1 Representation-space metrics

For each source–metamer pair and representation:

- Euclidean/L2 distance between source and metamer activations/codes.
- Cosine similarity or distance.
- Source-normalized/relative distance, with the normalization definition recorded.
- SAE code L1/L2 changes.
- Active-feature support overlap (Jaccard).
- Feature-wise signed and absolute changes.
- SAE/PCA reconstruction MSE for each image.
- Metamer-minus-source reconstruction-error change.
- Feature density, dead-feature count, activation-mass concentration, decoder statistics, and feature galleries.
- Cross-seed dictionary stability and explicit feature matching.
- Spatial activation-map diagnostics, kept separate from pooled SAE analyses.

## 8.2 Image-space and perceptual-similarity baselines

Add the proposal's missing baselines:

- Pixel-space distance (with exact color-space and normalization definition).
- SSIM.
- LPIPS.

Do not equate any single metric with human recognizability. Record the exact implementation, backbone/weights, input preprocessing, and metric range/direction.

## 8.3 Recognition-model baselines

Evaluate a fixed recognition-observer panel on the same source/metamer stimuli. Potential observers include standard/adversarial ResNet-50, standard/adversarial AlexNet, and IPCL or its supervised matched comparator where the appropriate readout is available.

For supervised models, use their verified classifier outputs. For IPCL/self-supervised models, use the source-documented linear readout or a separately trained readout that does not alter the frozen feature backbone; record the training split and method. Do not compare classifier confidence values from different models without calibration/interpretation safeguards.

Keep recognition-model accuracy distinct from human recognition accuracy.

## 8.4 Generator versus common observer evaluation

For each metamer:

- Evaluate it in the generator's own representation (generator-native track).
- Evaluate it across the fixed observer panel (common-observer track).
- Record the generator's response and each observer's response separately.
- Ask whether a metamer is unusual only to its generator or also to other models.

This cross-model matrix is an important part of the broader study and should be supported in the data model from the start.

---

# 9. Human-recognition analysis

Human-recognition analysis was not performed in the completed prototype. It is a later but important research stage and should be supported by the full repository's data schemas and analysis modules.

The Feather visual task uses 16-way object categorization; nominal chance performance is `1/16 = 0.0625`. Preserve the original task definition, category mapping, trial exclusions, participant identifiers, stimulus IDs, and condition labels.

The agent must inspect the actual behavioral data structure before choosing statistical aggregation. Some proposal descriptions refer to condition-averaged accuracy; do not assume that every condition has independent per-image labels. If only condition-level means are released, the analysis must remain at the condition level. Do not invent per-image human labels or treat repeated participant trials as independent stimuli.

Planned analyses:

- Reproduce the published behavioral recognition summaries for the exact subset of models/stages used.
- Link behavioral conditions to the correct generator checkpoint, stage, metamer stimulus set, and source identity.
- Compare SAE reconstruction error and feature metrics against PCA, pixel distance, SSIM, LPIPS, and recognition-model baselines.
- Examine associations within matched layer depths and control for layer depth where appropriate.
- Examine image smoothness and other plausible stimulus confounds.
- Use partial Spearman correlations where justified, with uncertainty intervals.
- Use participant-aware and stimulus-aware bootstrap or appropriate mixed-effects/repeated-measures models when the raw data support them.
- Estimate split-half reliability/noise ceiling using the actual participant-level data and published protocol.
- Correct for multiple comparisons across model families, layers, metrics, and feature analyses.

Do not claim the SAE predicts human recognition until the analysis has been performed and validated. Do not treat the proposal's expected effect size or hypothesized correlation as an observed result.

---

# 10. Statistical design for a multi-model, multi-layer study

The broader design creates many related measurements. The repository must avoid pseudoreplication and uncontrolled multiple testing.

Requirements:

1. Preserve source-image identity across all generators, stages, and conditions.
2. Treat repeated conditions/stages from the same source as repeated measures, not independent stimuli.
3. Define primary outcomes and comparison families before confirmatory runs.
4. Use source-level paired comparisons for source versus metamer metrics.
5. Bootstrap at the correct unit: source/stimulus for stimulus-level measures and participant (and, where relevant, stimulus) for behavioral outcomes.
6. Where the behavioral sample supports it, use mixed-effects/repeated-measures models with participant and stimulus structure represented appropriately.
7. Correct for multiplicity across planned comparisons (for example, Holm correction or a justified alternative).
8. Report effect sizes, uncertainty intervals, exact sample sizes, exclusions, and missing-condition counts; not p-values alone.
9. Separate exploratory visualizations/feature searches from confirmatory analyses.
10. Report null results and failed or incomplete model/layer conditions.
11. Do not compare absolute L2 distances across unrelated representational coordinate systems without normalization and a clear interpretation.
12. Do not count five correlated metrics from the same 32 triplets as five independent replications.

---

# 11. Recommended experimental execution strategy

The repository should be designed to support the complete model matrix, but execution should be staged to protect scientific validity and compute resources.

## Phase A — Audit and pilot parity

1. Audit the live `metamer-sae` repository and current Feather source commit.
2. Inventory model checkpoints, training conditions, stages, stimulus files, behavioral files, and metadata.
3. Reproduce the exact ResNet-50 pilot using the immutable pilot configuration and split.
4. Compare stimulus counts, activations, PCA transforms, SAE checkpoints/results, paired metrics, statistics, and figures against the supplied prototype artifacts.
5. Produce a pilot-parity report showing exact matches, numerical tolerances, and any unexplained deviations.

## Phase B — AlexNet core expansion

1. Inventory standard AlexNet, adversarial AlexNet (including L2 \(\epsilon=3\)), and random-perturbation control checkpoints.
2. Inventory all matching generator stages and paired source/metamer stimuli.
3. Verify official human-behavioral data linkage for each condition.
4. Run the same extraction, natural-image representation training, and held-out evaluation framework.
5. Include the AlexNet results as a core model-family comparison, not a supplementary appendix by default.

## Phase C — IPCL core expansion

1. Load the official IPCL AlexNetGN self-supervised model.
2. Load its architecture-/augmentation-matched supervised comparator.
3. Verify layer/stage mapping, model heads/readout, data preprocessing, and available metamer stimuli.
4. Run natural-image activation extraction and model-/layer-specific representation learning as declared.
5. Apply the matched evaluation framework and, where available, link to the correct behavioral conditions.
6. Report IPCL separately from standard/adversarial supervised comparisons while enabling joint multi-family summaries.

## Phase D — Adversarial-IPCL feasibility/extension

1. Verify whether a published or source-provided adversarial-IPCL model/checkpoint and matching metamer stimuli exist.
2. If they exist, add them as a separate training condition with exact provenance.
3. If not, prepare a new experimental protocol: architecture, IPCL objective, robust objective, perturbation norm/epsilon, training data, schedule, checkpoint validation, metamer generation stages, optimizer settings, null-distribution optimization checks, and behavioral-data limitations.
4. Do not quietly substitute standard IPCL or supervised robust AlexNet for adversarial IPCL.
5. Do not claim an adversarial-IPCL result until the new model and its metamer stimuli have actually been generated and evaluated.

## Phase E — Multi-layer and common-observer expansion

1. Expand to all eligible Feather stages or the predeclared subset for ResNet-50, AlexNet, and IPCL.
2. Run the common-observer panel over the full selected generator stimulus set.
3. Run generator-native feature analysis for each selected generator/stage.
4. Keep model-specific and cross-model-transfer runs clearly separated.
5. Add pixel, SSIM, LPIPS, and recognition-model baselines.

## Phase F — Human recognition

1. Audit behavioral-data granularity and linkage.
2. Reproduce Feather's relevant accuracy summaries.
3. Run the predeclared condition-level feature-to-human analyses with appropriate confound controls and reliability/noise-ceiling assessment.
4. Generate model-family × layer × training-condition figures and tables.

## Phase G — Future brain/fMRI extension

Keep this separate until an appropriate brain dataset, alignment protocol, and analysis plan are selected. Do not imply that the existing visual behavioral data answer the brain-alignment research question.

---

# 12. Required experiment metadata and run identifiers

Each experiment/run must be uniquely described. At minimum, record:

| Field | Meaning |
|---|---|
| `run_id` | Unique run identifier |
| `experiment_track` | `pilot_parity`, `common_observer`, `generator_native`, `transfer`, `human_behavior`, or `brain_extension` |
| `generator_family` | `resnet50`, `alexnet`, `ipcl_alexnetgn`, or other verified family |
| `generator_model_id` | Exact Feather/source model identifier |
| `training_objective` | `supervised_standard`, `adversarial`, `random_perturbation`, `self_supervised_ipcl`, `supervised_matched_ipcl_arch`, etc. |
| `perturbation_norm` | e.g. `L2`, `Linf`, or null |
| `perturbation_epsilon` | Exact value and units, or null |
| `generator_checkpoint_hash` | Hash of exact model weights |
| `generator_stage` | Official Feather stage and actual module path |
| `observer_model_id` | Exact observer checkpoint/model |
| `observer_stage` | Actual observer feature module/stage |
| `recognition_readout` | Native head, linear readout, or not applicable |
| `natural_dataset_id` | Natural-image training dataset version |
| `natural_split_hash` | Exact train/validation split provenance |
| `stimulus_manifest_hash` | Exact Feather stimulus set/version |
| `source_id` | Stable reference-image identity |
| `condition_id` | Exact generator × training × stage condition |
| `split` | Development / primary / reserve / other declared split |
| `representation_id` | Exact PCA/SAE run used |
| `representation_training_data` | Natural activation corpus only; include hash |
| `sae_seed` | Random seed where relevant |
| `software_environment` | Python/framework/package versions |
| `code_commit` | Repository commit used for the run |
| `status` | Planned / available / validated / executed / reviewed / failed |

Use machine-readable YAML/JSON configs rather than relying on notebook variables or folder names to convey model identity.

---

# 13. Results and reporting structure

The repository must produce integrated tables and publication-ready visualizations that allow the results to be inspected across:

- Generator family: ResNet-50, AlexNet, IPCL.
- Training objective: standard/supervised, adversarial, random perturbation, self-supervised IPCL, matched supervised comparator.
- Generator stage/layer.
- Observer model and stage.
- Representation: raw, PCA, SAE.
- SAE seed and architecture/configuration.
- Stimulus condition and source identity.
- Behavioral recognition condition, where linked.

Required outputs:

- Model/checkpoint inventory and coverage heatmap.
- Model × training condition × stage × stimulus availability table.
- Natural-image training/validation activation summaries.
- PCA explained-variance and reconstruction reports.
- SAE training curves, reconstruction/sparsity trade-offs, dead-feature and feature-density reports.
- Feature galleries and stability/matching reports.
- Paired source–metamer metrics per source, model, stage, observer, and representation.
- Image-space baseline tables (pixel, SSIM, LPIPS).
- Recognition-model accuracy tables.
- Human-recognition linkage and analysis tables, when activated.
- Statistical contrasts with sample counts, effect sizes, uncertainty intervals, exact p-values, and multiplicity correction.
- Figures in high-resolution PNG and vector PDF/SVG where appropriate.
- Machine-readable run configuration, provenance manifest, and closure report for every run.

Use Matplotlib for the plotting system; do not use Seaborn. Make comparisons legible with matched axes/units where appropriate, source-linked paired plots, clear uncertainty, and explicit model/stage/condition labels.

---

# 14. Repository engineering requirements

The actual Python package, scripts, and tests are specified in the full-scale build guide. This README adds the following experimental requirements to that implementation:

- Model families and training conditions must be declared in configuration/registry data, not hard-coded in analysis logic.
- New architectures must implement a common model adapter interface for checkpoint loading, preprocessing, stage discovery, forward hooks, and optional readout.
- Layer inventories must be discovered and validated against actual model objects.
- Stimulus manifests must support missing metamer conditions without losing valid images from other conditions.
- Source IDs must support grouped splits and repeated-measures analysis across model families and stages.
- Activation caches must be keyed by the exact observer/generator checkpoint, stage, preprocessing, pooling, and source-image hashes.
- PCA/SAE artifacts must record their natural-image data provenance and preprocessing.
- Run results must be immutable once marked reviewed; reruns should receive new run IDs.
- The pipeline must resume from validated intermediate artifacts rather than repeat all downloads/extractions/training unnecessarily.
- Unit, integration, data-integrity, and pilot-parity tests must include at least ResNet-50 and AlexNet adapters; the IPCL adapter must be tested when its checkpoint is available.

---

# 15. Completion criteria

The full repository is not complete merely because the pilot works or the pipeline can technically load three model classes.

## Minimum acceptance criteria for the expanded research scope

- [ ] Exact ResNet-50 pilot parity demonstrated against the prototype outputs.
- [ ] Standard AlexNet generator checkpoint/configuration audited and integrated.
- [ ] Adversarial AlexNet generator checkpoint/configuration audited and integrated, including the proposal's L2 \(\epsilon=3\) condition if the official files are available.
- [ ] AlexNet random-perturbation control audited and included when available.
- [ ] IPCL AlexNetGN self-supervised checkpoint/configuration audited and integrated.
- [ ] Matched supervised AlexNetGN comparison audited and integrated.
- [ ] Adversarial-IPCL availability explicitly resolved: official condition found and included, or marked unavailable with a separate proposed new-experiment protocol.
- [ ] Model/layer/stimulus inventory covers the required families and every selected eligible stage.
- [ ] Source identity, missing conditions, and grouped data splits are recorded and validated.
- [ ] Natural-image training/validation activation extraction works for each selected generator-native and observer configuration.
- [ ] Shared common-observer and generator-native representation tracks are implemented and clearly distinguished.
- [ ] PCA and SAE training are restricted to natural-image activations and produce reproducible artifacts.
- [ ] All held-out source/metamer analyses reuse frozen representations.
- [ ] Image-space baselines and recognition-model baselines are implemented and independently validated.
- [ ] Human-recognition analysis can be run only after the behavioral-data schema and condition linkage are verified.
- [ ] Multi-model/multi-layer statistics account for repeated sources, uncertainty, and multiple comparisons.
- [ ] Figures, integrated tables, configuration manifests, and closure reports are generated and visually reviewed.

---

# 16. How this README relates to the other project guides

Use the project documents together, with clearly separated roles:

1. **`METAMER_SAE_FULL_SCALE_BUILD_GUIDE.md`** — authoritative detailed technical and methodological specification for the modular full-scale repository; contains the prototype handover, target architecture, implementation phases, contracts, and acceptance tests.
2. **`METAMER_SAE_AGENT_QUICKSTART.md`** — concise agent entry point; directs the agent to the master guide and initial repository audit.
3. **This README (`METAMER_SAE_EXPERIMENTAL_SCOPE_AND_SCALE_README.md`)** — authoritative update to the intended experimental scale: ResNet-50 + AlexNet + IPCL, their training conditions, layer coverage, data requirements, and analysis tracks. It supersedes any earlier wording that makes AlexNet or IPCL merely optional extensions, while retaining all pilot facts and safeguards.
4. **Original proposal and comprehensive project plan** — scientific motivation, research questions, hypotheses, and prior methodological decisions.
5. **Prototype notebooks 01–09 and their result summaries** — executable evidence of what was actually tested; use them for pilot parity, not as a substitute for the expanded scope.
6. **Feather et al. source paper, supplementary material, and official repository** — authoritative source for exact model configurations, checkpoints, generator stages, stimulus availability, optimization records, and behavioral data.

Before modifying the live repository, audit its actual contents and branch. Do not invent or assume its GitHub URL, existing file layout, model availability, or checkpoint paths.

---

# 17. Primary source references

- Feather, J., Leclerc, G., Mądry, A., & McDermott, J. H. (2023). **Model metamers reveal divergent invariances between biological and artificial neural networks.** *Nature Neuroscience*, 26, 2017–2034. [Nature article](https://www.nature.com/articles/s41593-023-01442-0) · [Open full text at PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620097/).
- Feather et al. official model/metamer code and data: [https://github.com/jenellefeather/model_metamers_pytorch](https://github.com/jenellefeather/model_metamers_pytorch).
- Original project proposal: `Project 08 Proposal_ Metamers and SAEs.pdf`.
- Consolidated research plan: `Metamer_SAE_Comprehensive_Project_Plan.docx`.
- Full-scale implementation guide: `METAMER_SAE_FULL_SCALE_BUILD_GUIDE.md`.
- Prototype results: `METAMER_SAE_Notebook_08_Results_Summary.md` and the relevant Notebook 01–09 handovers.

**Source-grounding note:** The AlexNet standard/adversarial/random-perturbation conditions and the IPCL self-supervised plus matched supervised AlexNetGN comparison are grounded in the project proposal and Feather et al. (2023). An adversarially trained IPCL counterpart is not assumed to exist; it is an explicit availability check/new-experiment decision.

---

## Final instruction

Build the repository for the **full intended Metamer-SAE study**, not only the existing ResNet-50 pilot. Reproduce the pilot without changing it; then make ResNet-50, AlexNet, and IPCL first-class model families, with their correct training-condition distinctions and all eligible Feather stages/stimuli. Keep source data, preprocessing, observer roles, training data, learned representations, and behavioral measurements traceable in every run. When a requested condition is unavailable in the released Feather resources, report the gap and specify the additional work required rather than silently omitting it or inventing a substitute.
