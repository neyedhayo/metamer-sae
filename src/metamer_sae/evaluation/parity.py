"""Reference-parity checks against the Colab prototype (Notebooks 04/06/08/09)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "reference_values.json"


def _row(name, got, want, tol):
    return {"check": name, "got": got, "reference": want, "tolerance": tol, "pass": bool(abs(got - want) <= tol)}


def check_parity(cfg, pair_df: pd.DataFrame, contrasts: pd.DataFrame, ids, reps) -> pd.DataFrame:
    ref = json.loads(REF.read_text()); rows = []
    mean = np.load(cfg.sae.training_mean)
    Xva = np.asarray(np.load(Path(cfg.paths.natural_activations) / "validation" / "pooled_activations.npy", mmap_mode="r"))
    for k in (16, 64):
        mse = float(reps[f"pca_{k}"].reconstruction_mse(Xva).mean())
        rows.append(_row(f"PCA-{k} natural validation MSE", mse, ref["pca_val_mse"][str(k)], ref["tol"]["pca_mse"]))
    for seed in ("seed123", "seed42"):
        mse = float(reps[f"sae_{seed}"].reconstruction_mse(Xva).mean())
        rows.append(_row(f"SAE {seed} natural validation MSE", mse, ref["sae_val_mse"][seed], ref["tol"]["sae_mse"]))
    l2 = pair_df.pivot_table(index=["representation", "stimulus_id"], columns="metamer_condition", values="distance_l2")
    for rep, vals in ref["primary_l2_means"].items():
        for cond, want in vals.items():
            got = float(pair_df[(pair_df.representation == rep) & (pair_df.metamer_condition == cond)].distance_l2.mean())
            rows.append(_row(f"primary mean L2 {rep}/{cond}", got, want, ref["tol"]["l2_mean"]))
    for rep in reps:
        n = int((l2.loc[rep].standard > l2.loc[rep].adversarial).sum())
        rows.append(_row(f"32/32 L2 direction {rep}", n, 32, 0))
    rows.append(_row("number of paired contrasts", len(contrasts), 17, 0))
    rows.append(_row("max Holm-adjusted p < 0.001", float(contrasts.wilcoxon_p_holm.max() < 1e-3), 1.0, 0))
    rows.append(_row("max Holm-adjusted p (≈1.731e-06)", float(contrasts.wilcoxon_p_holm.max()), 1.731e-06, 2e-8))
    return pd.DataFrame(rows)
