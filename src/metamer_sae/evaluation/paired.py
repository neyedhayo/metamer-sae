"""Paired source-vs-metamer evaluation: one row per source x metamer condition x representation."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..representations.base import Representation
from .metrics import support_jaccard, vector_distance_metrics

def _is_sae(rep) -> bool:
    return hasattr(rep, "dictionary")  # SAERepresentation exposes decoder atoms (avoids a torch import)


PAIR_COLS = ["split", "stimulus_id", "metamer_condition", "representation", "distance_l2", "distance_l1",
             "relative_l2_source_norm", "cosine_similarity", "support_jaccard", "source_active_features",
             "metamer_active_features", "source_reconstruction_mse", "metamer_reconstruction_mse",
             "delta_reconstruction_error"]


def evaluate_pairs(split: str, ids: list[int], acts: dict[str, np.ndarray],
                   reps: dict[str, Representation], metamer_conditions=("standard", "adversarial"),
                   run_meta: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """acts[role] has shape (n_ids, D) in `ids` order (raw pooled activations)."""
    n = len(ids)
    for r, a in acts.items():
        assert a.shape[0] == n and np.isfinite(a).all(), f"bad activations for {r}"
    emb, mse = {}, {}
    for name, rep in reps.items():
        emb[name] = {c: rep.embed(acts[c]) for c in acts}
        mse[name] = {c: rep.reconstruction_mse(acts[c]) for c in acts}
    stim_rows, pair_rows = [], []
    for name in reps:
        for c in acts:
            for i, sid in enumerate(ids):
                z = emb[name][c][i]
                stim_rows.append({"split": split, "stimulus_id": sid, "condition": c, "representation": name,
                                  "reconstruction_mse": mse[name][c][i],
                                  "active_features": int(np.count_nonzero(z)) if _is_sae(reps[name]) else np.nan})
        for mc in metamer_conditions:
            for i, sid in enumerate(ids):
                s, m = emb[name]["source"][i], emb[name][mc][i]
                sae = _is_sae(reps[name])
                sm, mm = mse[name]["source"][i], mse[name][mc][i]
                pair_rows.append({"split": split, "stimulus_id": sid, "metamer_condition": mc,
                                  "representation": name, **vector_distance_metrics(s, m),
                                  "support_jaccard": support_jaccard(s, m) if sae else np.nan,
                                  "source_active_features": int(np.count_nonzero(s)) if sae else np.nan,
                                  "metamer_active_features": int(np.count_nonzero(m)) if sae else np.nan,
                                  "source_reconstruction_mse": sm, "metamer_reconstruction_mse": mm,
                                  "delta_reconstruction_error": mm - sm})
    stim, pair = pd.DataFrame(stim_rows), pd.DataFrame(pair_rows)
    assert len(pair) == n * len(metamer_conditions) * len(reps)
    for df in (stim, pair):
        for k, v in (run_meta or {}).items():
            df[k] = v
    return stim, pair
