# Adding a model/condition

1. Add a `ModelCondition` to `configs/models/registry.yaml` (objective, norm, epsilon, matched comparator). No code branch.
2. `metamer-sae resolve-registry --config <matrix.yaml>` → inspect candidates → pin `model_key`, `experiment_key`; `verified: true`.
3. Add an adapter spec (`kind: feather`) with the audited build script, stage→module map and the model's OWN preprocessing.
4. `matrix-inventory` → stages, availability, behaviour links; gaps stay in the manifests.
5. Extract natural + Feather activations per activation space; `fit-space` (natural only); evaluate; image/recognition baselines.
6. Human stage only after `human-link` passes.
