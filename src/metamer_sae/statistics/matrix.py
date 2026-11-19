"""Multi-model, multi-layer statistics: source-level pairing, declared comparison families, Holm per family.

Unit of replication = SOURCE image. Conditions/stages/representations from one source are repeated measures, and
correlated metrics on the same sources are not independent replications (reported as `n_sources`, never as n rows).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .paired import holm_bonferroni, paired_test_summary


def source_paired_contrasts(pair_df: pd.DataFrame, family: str, confirmatory: bool, metrics: list[str],
                            contrasts: list[tuple[str, str]], representations: list[str] | None = None,
                            n_boot: int = 10000, seed: int = 20261120) -> pd.DataFrame:
    """metric(cond_a) - metric(cond_b), paired by source_image_id, within each representation (and stage group).
    Only sources present in BOTH conditions are used; the count is reported."""
    reps = representations or sorted(pair_df.representation.unique())
    rows = []
    for rep in reps:
        sub = pair_df[pair_df.representation == rep]
        for m in metrics:
            if m not in sub or sub[m].isna().all():
                continue
            w = sub.pivot_table(index="source_image_id", columns="condition_id", values=m, aggfunc="mean")
            for a, b in contrasts:
                if a not in w or b not in w:
                    rows.append({"family": family, "representation": rep, "metric": m, "condition_a": a,
                                 "condition_b": b, "n_sources": 0, "status": "condition_missing"})
                    continue
                j = w[[a, b]].dropna()
                s = paired_test_summary(j[a], j[b], f"{rep}: {m} ({a} − {b})", "see difference_definition", a, b, n_boot, seed)
                s.pop("n_standard_farther", None)
                s["n_sources"] = s.pop("n_triplets")
                rows.append({"family": family, "representation": rep, "metric": m, "condition_a": a, "condition_b": b,
                             "status": "ok", "confirmatory": confirmatory, **s})
    df = pd.DataFrame(rows)
    if "wilcoxon_p_two_sided" in df:
        df["p_holm_within_family"] = holm_bonferroni(df.wilcoxon_p_two_sided)
    return df


def combine_families(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Holm is applied per declared family; exploratory families are labelled and never mixed into the
    confirmatory correction."""
    df = pd.concat(tables, ignore_index=True)
    df["label"] = np.where(df.get("confirmatory", False) == True, "confirmatory", "exploratory")  # noqa: E712
    return df


def clustered_bootstrap(df: pd.DataFrame, value: str, cluster: str = "source_image_id", n_boot=10000, seed=0) -> tuple:
    """Percentile CI of the mean resampling SOURCES (all of a source's rows travel together)."""
    g = [x[value].to_numpy() for _, x in df.groupby(cluster)]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(n_boot)]
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def sample_size_report(pairs: pd.DataFrame, used_sources: list[int]) -> dict:
    return {"n_sources_used": len(used_sources), "n_incomplete_pairs": int((~pairs.pair_complete).sum()),
            "n_conditions": int(pairs.condition_id.nunique()),
            "note": "n counts independent sources; condition/representation/metric rows are repeated measures"}


def expand_contrasts(contrasts: list[tuple[str, str]], condition_ids: list[str]) -> list[tuple[str, str]]:
    """Registry-level contrast (a, b) -> per-stage contrasts between `a|exp|stage` and `b|exp|stage` with the SAME
    stage identifier. Stages are matched by exact name only (never by ordinal position across models); stages
    present for one side only are left out of the family and should be reported as gaps by the caller."""
    by: dict[str, dict[str, str]] = {}
    for c in condition_ids:
        reg, _exp, stage = c.split("|")
        by.setdefault(reg, {})[stage] = c
    out = []
    for a, b in contrasts:
        out += [(by[a][s], by[b][s]) for s in sorted(set(by.get(a, {})) & set(by.get(b, {})))]
    return out
