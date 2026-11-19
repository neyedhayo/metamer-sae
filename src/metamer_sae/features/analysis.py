"""Natural-image feature analysis for a frozen TopK dictionary (declared natural data only)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def feature_density(codes: np.ndarray) -> dict:
    active = codes != 0
    count = active.sum(0)
    return {"n_images": int(len(codes)), "active_per_image_unique": np.unique(active.sum(1)).tolist(),
            "active_features": int((count > 0).sum()), "dead_features": int((count == 0).sum()),
            "dead_fraction": float((count == 0).mean()), "support_count": count}


def activation_stats(codes: np.ndarray) -> dict:
    """Per-feature mean/median/max over ACTIVE entries, plus total activation mass."""
    A = np.where(codes != 0, codes, np.nan)
    with np.errstate(all="ignore"):
        return {"mean_active": np.nanmean(A, 0), "median_active": np.nanmedian(A, 0),
                "max_active": np.nanmax(A, 0), "mass": np.nansum(A, 0)}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, np.float64)); n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def mass_concentration(mass: np.ndarray, tops=(10, 20, 50, 100)) -> dict:
    order = np.argsort(-mass); share = np.cumsum(mass[order]) / mass.sum()
    return {"gini": gini(mass), **{f"top_{t}_share": float(share[t - 1]) for t in tops}, "rank_order": order}


def support_groups(support_count: np.ndarray, min_support: int = 47) -> dict:
    """Pilot convention: >=47 training images = supported; 1..46 = rare; 0 = dead."""
    return {"supported": int((support_count >= min_support).sum()),
            "rare": int(((support_count >= 1) & (support_count < min_support)).sum()),
            "dead": int((support_count == 0).sum()), "min_support": min_support}


def decoder_norm_stats(D: np.ndarray) -> dict:
    n = np.linalg.norm(D, axis=1)
    return {"min": float(n.min()), "max": float(n.max()), "mean": float(n.mean())}


def match_dictionaries(Da: np.ndarray, Db: np.ndarray, absolute: bool = False) -> dict:
    """One-to-one (Hungarian) decoder-atom matching by cosine. Feature IDs are NEVER equated by index."""
    A = Da / np.linalg.norm(Da, axis=1, keepdims=True); B = Db / np.linalg.norm(Db, axis=1, keepdims=True)
    S = A @ B.T
    r, c = linear_sum_assignment(-(np.abs(S) if absolute else S))
    m = S[r, c]
    return {"row": r, "col": c, "cosine": m, "mean": float(m.mean()), "median": float(np.median(m)),
            "p10": float(np.quantile(m, .1)), "p90": float(np.quantile(m, .9)),
            "frac_ge_0.8": float((m >= .8).mean()), "frac_ge_0.9": float((m >= .9).mean())}


def top_activating(codes: np.ndarray, feature: int, n: int = 16) -> np.ndarray:
    """Row indices of the n strongest activations of one feature (for galleries)."""
    return np.argsort(-codes[:, feature])[:n]


def feature_changes(src_codes: np.ndarray, met_codes: np.ndarray) -> dict:
    d = met_codes - src_codes
    return {"signed_mean": d.mean(0), "abs_mean": np.abs(d).mean(0)}
