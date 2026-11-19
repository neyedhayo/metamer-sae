"""Human-behaviour stage: inspect -> link -> validate -> (only then) analyse. Separate from model-only runs.

Nothing here assumes the released file layout. `inspect_behavior_file` reports what is actually in a file;
the analysis refuses to run until `validate_linkage` has produced a passing report, and respects the real
aggregation unit (condition means cannot be bootstrapped over participants or turned into per-image labels).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.recognition import CHANCE
from ..statistics.paired import holm_bonferroni, partial_spearman, split_half_reliability

GRANULARITY = ("trial", "participant_by_condition", "condition_mean")
REQUIRED = ["experiment_key", "model_key", "stage", "accuracy", "granularity"]


def inspect_behavior_file(path: str | Path) -> dict:
    """Structure report for .mat / .csv / .json. Pure inspection; no analysis."""
    p = Path(path); rep = {"file": str(p), "suffix": p.suffix}
    if p.suffix == ".mat":
        from scipy.io import loadmat
        m = loadmat(p, squeeze_me=True, struct_as_record=False)
        rep["variables"] = {k: {"type": type(v).__name__, "shape": list(getattr(v, "shape", ()))}
                            for k, v in m.items() if not k.startswith("__")}
    elif p.suffix == ".csv":
        d = pd.read_csv(p, nrows=5000); rep["columns"] = list(d.columns); rep["n_rows_sampled"] = len(d)
    elif p.suffix == ".json":
        j = json.loads(p.read_text()); rep["top_level_keys"] = list(j) if isinstance(j, dict) else f"list[{len(j)}]"
    else:
        raise ValueError("unsupported behaviour file type")
    rep["granularity_inferred"] = "must be set by a human after inspection"
    return rep


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in REQUIRED if c not in df]
    if miss:
        raise ValueError(f"behaviour table missing columns {miss}")
    if not set(df.granularity).issubset(GRANULARITY):
        raise ValueError(f"granularity must be in {GRANULARITY}")
    if not df.accuracy.between(0, 1).all():
        raise ValueError("accuracy must be a proportion in [0, 1]")
    if (df.granularity != "condition_mean").any() and "participant_id" not in df:
        raise ValueError("participant-level rows need participant_id")
    return df


def validate_linkage(behavior: pd.DataFrame, condition_manifest: pd.DataFrame, out: str | Path) -> dict:
    """Behaviour rows must match (experiment, model_key, stage) condition records, both ways. Writes the gate file."""
    validate_schema(behavior)
    key = ["experiment_key", "model_key", "stage"]
    b = behavior[key].drop_duplicates(); c = condition_manifest[key].dropna().drop_duplicates()
    m = b.merge(c, on=key, how="outer", indicator=True)
    rep = {"n_behaviour_conditions": len(b), "n_manifest_conditions": len(c),
           "behaviour_without_stimuli": m[m._merge == "left_only"][key].to_dict("records"),
           "stimuli_without_behaviour": m[m._merge == "right_only"][key].to_dict("records"),
           "granularity": sorted(behavior.granularity.unique()), "chance_level": CHANCE}
    rep["linkage_validated"] = not rep["behaviour_without_stimuli"]
    Path(out).write_text(json.dumps(rep, indent=2)); return rep


def require_validated(report_path: str | Path) -> dict:
    p = Path(report_path)
    if not p.exists():
        raise PermissionError("human analysis blocked: run validate_linkage first")
    r = json.loads(p.read_text())
    if not r.get("linkage_validated"):
        raise PermissionError("human analysis blocked: linkage not validated")
    return r


def condition_table(behavior: pd.DataFrame, metrics: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    """Condition-level join. behavior -> mean accuracy per (experiment, model_key, stage); metrics (per condition)
    must already be averaged over SOURCES; depth gives depth_rank within model (not comparable across models)."""
    key = ["experiment_key", "model_key", "stage"]
    acc = behavior.groupby(key).accuracy.mean().rename("human_accuracy").reset_index()
    return acc.merge(metrics, on=key).merge(depth, on=key)


def run_condition_analysis(table: pd.DataFrame, score_cols: list[str], behavior: pd.DataFrame, gate: str | Path,
                           n_boot: int = 10000, seed: int = 0) -> dict:
    """Depth-controlled partial Spearman per score + Holm across scores + split-half ceiling when participant data
    exist. Within a single model family only (depth ranks are not comparable across architectures)."""
    require_validated(gate)
    if table.model_key.nunique() < 2 and table.stage.nunique() < 4:
        raise ValueError("too few conditions for a condition-level correlation")
    rows = []
    for s in score_cols:
        d = table[[s, "human_accuracy", "depth_rank"]].dropna()
        r = partial_spearman(d[s], d.human_accuracy, d.depth_rank, n_boot, seed)
        rows.append({"score": s, "n_conditions": len(d), **r})
    out = pd.DataFrame(rows)
    ceiling = None
    if (behavior.granularity == "participant_by_condition").all():
        piv = behavior.pivot_table(index="participant_id", columns=["model_key", "stage"], values="accuracy")
        ceiling = split_half_reliability(piv.dropna(axis=1).to_numpy(), seed=seed)
    return {"correlations": out, "split_half_ceiling": ceiling, "chance": CHANCE,
            "note": "Condition-level association; not a per-image prediction; no causal claim."}
