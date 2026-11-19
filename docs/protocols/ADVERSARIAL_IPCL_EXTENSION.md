# Adversarial-IPCL extension protocol (Phase D) — NOT executed

**Status:** specification only. Run `metamer-sae resolve-registry` first. If the audit
(`data/manifests/adversarial_ipcl_audit.json`) finds an official adversarially trained IPCL checkpoint with matching
metamers, add it to `configs/models/registry.yaml` as `ipcl_adversarial` (pin `model_key`, set `verified: true`) and stop
here. Otherwise this is a **new experiment**; self-supervised IPCL must never be relabelled adversarial and robust supervised
AlexNet must never be substituted.

## Required before any result may be claimed
1. **Training recipe:** AlexNetGN backbone, IPCL objective, robust objective (e.g. PGD against the IPCL instance-contrast loss
   or against a linear-readout loss — choose and justify), norm (L2/Linf), epsilon (incl. the L2 eps=3 comparison point),
   attack steps/step size, data, augmentations, schedule, optimiser, seeds.
2. **Checkpoint validation:** clean and robust accuracy of the readout, comparison with the non-adversarial IPCL and the
   matched supervised model trained under identical data/augmentation.
3. **Metamer generation:** per selected stage, Feather's optimisation recipe (~24,000 steps/image) on the same source IDs,
   with optimisation-fidelity and null-distribution checks (metamer matches its own source's activation better than other
   images' activations).
4. **Versioning:** a separately versioned dataset (`adversarial_ipcl_v1`), checksums, manifests, split inherited from the
   source-level split.
5. **Limits:** no human behavioural data exists for this condition; any human-recognition claim requires a new, separately
   approved psychophysics experiment.
6. Registry entry with `availability: available` only after items 1–4 are executed and reviewed.
