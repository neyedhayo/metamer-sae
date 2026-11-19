"""Metric definitions (directions/units: see docs/DATA_DICTIONARY.md)."""
from __future__ import annotations
import numpy as np


def cosine_similarity(a, b, eps=1e-12) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float("nan") if d < eps else float(a @ b / d)


def vector_distance_metrics(src, met) -> dict:
    a, b = np.asarray(src, np.float64).ravel(), np.asarray(met, np.float64).ravel()
    d = b - a; l2 = float(np.linalg.norm(d))
    return {"distance_l2": l2, "distance_l1": float(np.abs(d).sum()),
            "relative_l2_source_norm": float(l2 / max(np.linalg.norm(a), 1e-12)),
            "cosine_similarity": cosine_similarity(a, b)}


def support_jaccard(a, b) -> float:
    sa, sb = set(np.flatnonzero(np.asarray(a) != 0)), set(np.flatnonzero(np.asarray(b) != 0))
    u = sa | sb
    return 1.0 if not u else len(sa & sb) / len(u)
