"""Paired statistics: bootstrap CI, d_z, Wilcoxon, Holm; plus later-stage helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

REPRESENTATIONS = ["raw_384", "pca_16", "pca_64", "sae_seed123", "sae_seed42"]


def bootstrap_mean_ci(values, n_boot=10000, seed=20261119, alpha=0.05):
    """Percentile CI of the mean; resamples PAIRED units (triplets), not image-condition rows."""
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_test_summary(x, y, name, note, first, second, n_boot=10000, seed=20261119) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y); x, y = x[v], y[v]; d = x - y
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    try:
        p = float(stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:  # noqa: BLE001
        p = np.nan
    return {"contrast": name, "n_triplets": int(len(d)), "first_series": first, "second_series": second,
            "mean_first": float(x.mean()), "mean_second": float(y.mean()),
            "difference_definition": f"{first} - {second}", "mean_paired_difference": float(d.mean()),
            "median_paired_difference": float(np.median(d)), "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi,
            "cohens_dz": float(d.mean() / sd) if sd and sd > 0 else np.nan, "wilcoxon_p_two_sided": p,
            "n_standard_farther": int((d > 0).sum()), "direction_interpretation": note}


def holm_bonferroni(p) -> np.ndarray:
    p = np.asarray(p, float); adj = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return adj
    order = idx[np.argsort(p[idx])]; m = len(order); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def contrast_family(pair_df: pd.DataFrame, ids: list[int], representations=REPRESENTATIONS,
                    n_boot=10000, seed=20261119) -> pd.DataFrame:
    """The pilot's 17 contrasts: L2, relative L2, cosine for every space + support Jaccard for SAEs.
    Holm is applied across the whole family. These are related metrics on the same triplets."""
    rows = []
    for rep in representations:
        sub = pair_df[pair_df.representation == rep]
        def wide(col):
            return sub.pivot(index="stimulus_id", columns="metamer_condition", values=col).loc[ids]
        w = wide("distance_l2")
        rows.append(paired_test_summary(w.standard, w.adversarial, f"{rep}: L2 distance (standard − adversarial)",
                                        "Positive: standard metamer is farther from its source.", "standard", "adversarial", n_boot, seed))
        w = wide("relative_l2_source_norm")
        rows.append(paired_test_summary(w.standard, w.adversarial, f"{rep}: relative L2 (standard − adversarial)",
                                        "Positive: standard has greater source-normalised displacement.", "standard", "adversarial", n_boot, seed))
        w = wide("cosine_similarity")
        rows.append(paired_test_summary(w.adversarial, w.standard, f"{rep}: cosine (adversarial − standard)",
                                        "Positive: adversarial has greater cosine alignment with its source.", "adversarial", "standard", n_boot, seed))
        if rep.startswith("sae_"):
            w = wide("support_jaccard")
            rows.append(paired_test_summary(w.adversarial, w.standard, f"{rep}: support Jaccard (adversarial − standard)",
                                            "Positive: adversarial shares more active SAE features with its source.", "adversarial", "standard", n_boot, seed))
    df = pd.DataFrame(rows)
    df["wilcoxon_p_holm"] = holm_bonferroni(df.wilcoxon_p_two_sided)
    df["p_value_family"] = "All contrasts in this table; Holm family-wise adjustment"
    return df


# ---- Later (human-recognition) stage helpers: condition-level, depth-controlled ----
def partial_spearman(x, y, z, n_boot=10000, seed=0) -> dict:
    """Spearman partial correlation of x,y controlling z (rank-residualisation); bootstrap over conditions."""
    def pr(x, y, z):
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        Z = np.column_stack([np.ones(len(x)), np.atleast_2d(stats.rankdata(z)).reshape(len(x), -1)])
        res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]  # noqa: E731
        return float(np.corrcoef(res(rx), res(ry))[0, 1])
    x, y, z = map(np.asarray, (x, y, z)); est = pr(x, y, z)
    rng = np.random.default_rng(seed); n = len(x)
    boots = [pr(x[i], y[i], z[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    boots = [b for b in boots if np.isfinite(b)]
    return {"rho_partial": est, "ci_low": float(np.quantile(boots, .025)), "ci_high": float(np.quantile(boots, .975))}


def split_half_reliability(responses: np.ndarray, n_splits=1000, seed=0) -> float:
    """responses: (participants, conditions) accuracy. Spearman-Brown corrected split-half noise ceiling."""
    rng = np.random.default_rng(seed); P = responses.shape[0]; rs = []
    for _ in range(n_splits):
        perm = rng.permutation(P); a, b = responses[perm[:P // 2]].mean(0), responses[perm[P // 2:]].mean(0)
        r = stats.spearmanr(a, b)[0]; rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs))
