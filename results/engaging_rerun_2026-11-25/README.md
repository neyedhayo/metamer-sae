# Engaging rerun, 2026-11-25

From-scratch run of the pilot pipeline on MIT Engaging (one H100, about five minutes), produced by
`scripts/engaging_full_pilot.sbatch` at repo commit `95e8aed`. It is a **refit, not a parity run**: the archived
Colab PCA and SAE checkpoints were not available, so a new PCA and new TopK SAEs (seeds 123 and 42) were fit.

| Path | Contents |
|---|---|
| `runs/eval_20261125T164037Z/` | Paired evaluation: tables, figures, `parity_report.csv`, provenance and closure report |
| `runs/pca_20261125T163939Z/` | PCA retention table and figure |
| `runs/topk_seed123_*/`, `runs/topk_seed42_*/` | SAE training history and result (both stopped early at epoch 78) |
| `runs/inventory_*/` | Expanded-study audit attempts; `matrix-inventory` stopped with `KeyError: nan` on unresolved registry conditions |
| `configs/` | The derived config used for evaluation (points at the new PCA and checkpoints) |
| `stimulus_allocation.csv` | Split ledger written by `write-pilot-ledger` |
| `followup/` | Exploratory descriptives on the 32 primary triplets (`followup.py`, `followup.json`) |

Not included, to respect `.gitignore`: SAE checkpoints (`*.pt`), the PCA basis (`*.npz`), activations and the Feather
images. The closure reports still list hashes for those files. Rerunning the job script regenerates everything.

Headline: standard metamers are farther from the source than adversarial ones in 32 of 32 primary triplets in all five
spaces. Raw and PCA distances match the pilot reference to four digits; SAE distances are a few percent higher because
of the shorter training. The follow-up numbers are exploratory and used the primary split; the reserve split was not loaded.
A summary with figures is in `docs/next_steps/index.html`.

Environment notes: `overcomplete 0.3.0` needs `matplotlib < 3.9`; importing Feather's metadata also needs `h5py`,
`dill`, `torchaudio` (matching the torch version) and `chcochleagram` (from GitHub).
