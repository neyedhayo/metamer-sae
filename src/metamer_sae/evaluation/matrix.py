"""Generalised paired evaluation: one frozen representation, many metamer conditions, source-level pairing.

Distances are only comparable WITHIN a representation. `natural_scale` (median natural-validation embedding norm)
gives a within-representation normalised column; cross-architecture raw distances are never ranked.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from ..representations.base import Representation
from .metrics import support_jaccard, vector_distance_metrics


def natural_scale(rep: Representation, natural_val: np.ndarray) -> float:
    """Median L2 norm of natural-validation embeddings (centred coordinates for PCA/SAE as embedded)."""
    return float(np.median(np.linalg.norm(rep.embed(natural_val), axis=1)))


def evaluate_conditions(split: str, ids: list[int], source: np.ndarray, metamers: dict[str, np.ndarray],
                        reps: dict[str, Representation], scales: dict[str, float] | None = None,
                        cond_meta: dict[str, dict] | None = None, run_meta: dict | None = None) -> pd.DataFrame:
    """source: (n, D); metamers[condition_id]: (n, D) in `ids` order. Returns one row per
    source x condition x representation."""
    n = len(ids)
    assert source.shape[0] == n and np.isfinite(source).all()
    rows = []
    for rn, rep in reps.items():
        es, ms = rep.embed(source), rep.reconstruction_mse(source)
        sae = hasattr(rep, "dictionary")
        for cid, X in metamers.items():
            assert X.shape == source.shape and np.isfinite(X).all(), f"bad metamer block {cid}"
            em, mm = rep.embed(X), rep.reconstruction_mse(X)
            for i, sid in enumerate(ids):
                d = vector_distance_metrics(es[i], em[i])
                rows.append({"split": split, "source_image_id": sid, "condition_id": cid, "representation": rn, **d,
                             "distance_l2_natural_scaled": d["distance_l2"] / scales[rn] if scales else np.nan,
                             "support_jaccard": support_jaccard(es[i], em[i]) if sae else np.nan,
                             "source_reconstruction_mse": ms[i], "metamer_reconstruction_mse": mm[i],
                             "delta_reconstruction_error": mm[i] - ms[i], **((cond_meta or {}).get(cid, {}))})
    df = pd.DataFrame(rows)
    assert len(df) == n * len(metamers) * len(reps)
    for k, v in (run_meta or {}).items():
        df[k] = v
    return df


def cross_model_matrix(native: pd.DataFrame, observer: pd.DataFrame, metric: str = "relative_l2_source_norm") -> pd.DataFrame:
    """Is a metamer unusual only to its generator, or also to other models? Per generator condition, the
    source-level Spearman between the generator-native effect and each observer's effect (same sources)."""
    rows = []
    for cid, g in native.groupby("condition_id"):
        a = g.groupby("source_image_id")[metric].mean()
        for (oid, rep), o in observer[observer.condition_id == cid].groupby(["observer_id", "representation"]):
            b = o.groupby("source_image_id")[metric].mean()
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            rho = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0] if len(j) > 2 else np.nan
            rows.append({"condition_id": cid, "observer_id": oid, "representation": rep, "metric": metric,
                         "n_sources": len(j), "spearman_native_vs_observer": rho})
    return pd.DataFrame(rows)
