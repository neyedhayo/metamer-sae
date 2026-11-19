# Data dictionary (key tables)

| Column | Meaning | Direction / unit |
|---|---|---|
| `stimulus_id` | Feather source image index (0–399); the paired unit | stable ID, never row order |
| `split` | `development` / `primary_evaluation` / `reserve` | reserve rows never loaded |
| `metamer_condition` | `standard` or `adversarial` generator | — |
| `representation` | `raw_384`, `pca_16`, `pca_64`, `sae_seed123`, `sae_seed42` | spaces are NOT on a common scale |
| `distance_l2` / `distance_l1` | ‖metamer − source‖ in the representation's coordinates (raw activations, PCA scores, or SAE codes) | higher = farther from source |
| `relative_l2_source_norm` | `distance_l2 / ‖source‖` | higher = larger relative displacement |
| `cosine_similarity` | cos(source, metamer) in representation coordinates | higher = more aligned |
| `support_jaccard` | |S∩M|/|S∪M| over nonzero SAE codes (SAE only) | higher = more shared active features |
| `*_reconstruction_mse` | mean squared residual in the 384-D activation space | lower = better reconstruction |
| `delta_reconstruction_error` | metamer MSE − source MSE | positive = metamer reconstructs worse |
| `wilcoxon_p_two_sided`, `wilcoxon_p_holm` | exact p and Holm-adjusted p over the 17-contrast family | report in scientific notation, never "0" |
| `cohens_dz` | mean paired difference / SD of differences | sign follows `difference_definition` |

SAE feature index *i* is meaningful only within one seed. Cross-seed comparisons require `features.analysis.match_dictionaries`.
