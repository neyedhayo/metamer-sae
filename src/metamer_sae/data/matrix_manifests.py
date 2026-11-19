"""Manifests for the full model x condition x stage matrix. Missing conditions are rows, never silent drops.

Written set (README scope §5.1): model_inventory, layer_inventory, stimulus_availability, stimulus_manifest,
metamer_pair_manifest, condition_manifest, human_behavior_manifest (metadata only), download_validation_report,
split_manifest.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..provenance.hashing import sha256_file, sha256_json
from ..registry import Registry

PAIR_LONG_COLS = ["pair_id", "source_image_id", "condition_id", "family", "training_objective", "stage",
                  "source_stimulus_id", "metamer_stimulus_id", "split", "pair_complete", "exclusion_reason"]


def make_condition_id(registry_condition_id: str, experiment_key: str, stage: str) -> str:
    """Exact generator x training x stage condition id (stage kept verbatim, never re-indexed)."""
    return f"{registry_condition_id}|{experiment_key}|{stage}"


def condition_manifest(reg: Registry, resolved: pd.DataFrame, stages_by_condition: dict[str, list[str]]) -> pd.DataFrame:
    res = resolved.set_index("condition_id")
    rows = []
    for c in reg.conditions:
        r = res.loc[c.condition_id]
        stages = stages_by_condition.get(c.condition_id, [])
        for s in stages or [None]:
            rows.append({"condition_id": make_condition_id(c.condition_id, c.experiment_key or "unresolved", s or "none"),
                         "registry_condition_id": c.condition_id, "family": c.family,
                         "training_objective": c.training_objective.value, "perturbation_norm": c.perturbation_norm,
                         "perturbation_epsilon": c.perturbation_epsilon, "experiment_key": c.experiment_key,
                         "model_key": r.resolved_model_key, "stage": s, "resolution_status": r.status,
                         "verified": c.verified, "availability": "gap" if r.resolved_model_key is None else c.availability})
    return pd.DataFrame(rows)


def stimulus_availability(probe: pd.DataFrame) -> pd.DataFrame:
    """probe: rows (condition_id, source_image_id, role, available). One summary row per condition/stage."""
    g = probe[probe.role != "source"].groupby("condition_id")
    src = probe[probe.role == "source"].groupby("condition_id").available.sum() if (probe.role == "source").any() else None
    out = g.agg(n_candidates=("source_image_id", "nunique"), n_metamers_available=("available", "sum")).reset_index()
    out["n_missing"] = out.n_candidates - out.n_metamers_available
    return out


def metamer_pair_manifest(stim: pd.DataFrame, cond_meta: pd.DataFrame, split_df: pd.DataFrame | None) -> pd.DataFrame:
    """One row per source x condition. A source keeps ONE split across all conditions."""
    ok = stim[stim.availability_status == "ok"]
    src = {int(r.source_image_id): r.stimulus_id for r in ok[ok.stimulus_role == "source"].itertuples()}
    have = {(r.generator_condition, int(r.source_image_id)): r.stimulus_id
            for r in ok[ok.stimulus_role != "source"].itertuples()}
    split_of = dict(zip(split_df.stimulus_id.astype(int), split_df.split)) if split_df is not None else {}
    meta = cond_meta.set_index("condition_id")
    rows = []
    for cid, m in meta.iterrows():
        for sid in sorted(set(int(i) for i in stim.source_image_id)):
            s, mm = src.get(sid), have.get((cid, sid))
            miss = [n for n, v in (("source", s), ("metamer", mm)) if v is None]
            rows.append({"pair_id": f"{cid}:{sid:04d}", "source_image_id": sid, "condition_id": cid,
                         "family": m.family, "training_objective": m.training_objective, "stage": m.stage,
                         "source_stimulus_id": s, "metamer_stimulus_id": mm, "split": split_of.get(sid),
                         "pair_complete": not miss, "exclusion_reason": ";".join(f"missing_{x}" for x in miss) or None})
    return pd.DataFrame(rows, columns=PAIR_LONG_COLS)


def assert_split_consistency(pairs: pd.DataFrame) -> None:
    """Every source must have exactly one split across all conditions (no leakage between model families)."""
    n = pairs.dropna(subset=["split"]).groupby("source_image_id").split.nunique()
    if (n > 1).any():
        raise ValueError(f"sources with conflicting splits: {n[n > 1].index.tolist()[:10]}")


def complete_case_subset(pairs: pd.DataFrame, conditions: list[str]) -> tuple[list[int], pd.DataFrame]:
    """Sources complete in ALL listed conditions + a transparent exclusion table (incomplete groups are kept)."""
    sub = pairs[pairs.condition_id.isin(conditions)]
    comp = sub.pivot_table(index="source_image_id", columns="condition_id", values="pair_complete", aggfunc="all")
    comp = comp.reindex(columns=conditions)
    keep = comp.fillna(False).all(axis=1)
    excl = comp[~keep].reset_index().melt(id_vars="source_image_id", var_name="condition_id", value_name="complete")
    return sorted(int(i) for i in comp.index[keep]), excl[~excl.complete.fillna(False).astype(bool)]


def coverage_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    return (pairs.groupby(["family", "training_objective", "condition_id", "stage"], dropna=False)
            .agg(n_sources=("source_image_id", "nunique"), n_complete=("pair_complete", "sum")).reset_index())


def split_manifest(split_df: pd.DataFrame, policy: str, seed: int, stratified: bool, rule: str = "") -> dict:
    from .splits import split_hash
    return {"policy": policy, "seed": seed, "stratified": stratified, "rule": rule,
            "split_sha256": split_hash(split_df), "counts": split_df.split.value_counts().to_dict(),
            "ids": {s: sorted(g.stimulus_id.astype(int).tolist()) for s, g in split_df.groupby("split")}}


def write_manifests(out_dir: str | Path, tables: dict[str, pd.DataFrame], jsons: dict[str, dict]) -> dict[str, str]:
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True); hashes = {}
    for name, df in tables.items():
        p = d / f"{name}.csv"; df.to_csv(p, index=False); hashes[p.name] = sha256_file(p)
        if name in ("model_inventory", "layer_inventory"):
            (d / f"{name}.json").write_text(df.to_json(orient="records", indent=2))
    for name, obj in jsons.items():
        p = d / f"{name}.json"; p.write_text(json.dumps(obj, indent=2, default=str)); hashes[p.name] = sha256_file(p)
    return hashes
