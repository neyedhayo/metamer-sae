"""Matrix-wide inventory against the pinned Feather metadata: models, layers, stimulus availability, behaviour links.
Every function reports gaps instead of skipping them."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..registry import Registry, resolve_registry
from .adapter import FeatherMetadata, probe_url


def model_inventory(meta: FeatherMetadata, reg: Registry) -> pd.DataFrame:
    keys = sorted(meta.networks)
    res = resolve_registry(reg, keys)
    exps = meta.enumerate_conditions()
    used_in = exps.groupby("model_key").experiment_key.agg(lambda s: sorted(set(s))).to_dict()
    res["feather_experiments"] = res.resolved_model_key.map(lambda k: used_in.get(k, []) if k else [])
    return res


def layer_inventory(meta: FeatherMetadata, inv: pd.DataFrame, loaded_shapes: dict | None = None) -> pd.DataFrame:
    """One row per (condition, Feather stage). `loaded_shapes[(model_key, stage)]` (from an adapter probe) fills
    tensor shapes; `valid_metamer_stage` is True only for stages Feather lists for that network in that experiment."""
    rows = []
    for r in inv.itertuples():
        mk = r.resolved_model_key
        if not mk:
            rows.append({"condition_id": r.condition_id, "model_key": None, "stage": None, "status": "model_unresolved"})
            continue
        for ek in r.feather_experiments or [None]:
            for i, layer in enumerate(meta.layers(mk)):
                rows.append({"condition_id": r.condition_id, "model_key": mk, "experiment_key": ek, "stage": layer,
                             "depth_rank": i, "valid_metamer_stage": ek is not None,
                             "tensor_shape": (loaded_shapes or {}).get((mk, layer)),
                             "status": "listed_in_feather_metadata"})
    return pd.DataFrame(rows)


def stimulus_probe(meta: FeatherMetadata, selections: list[dict], candidate_ids, workers: int = 16) -> pd.DataFrame:
    """selections: dicts with condition_id, experiment_key, model_key, stage. Probes source + metamer URLs."""
    jobs = []
    for s in selections:
        ctx = meta.context(s["experiment_key"], s["model_key"], s["stage"])
        for i in candidate_ids:
            jobs.append((s["condition_id"], "source", i, s["experiment_key"], meta.source_url(s["experiment_key"], i)))
            jobs.append((s["condition_id"], "metamer", i, s["experiment_key"], meta.metamer_url(ctx, i)))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: probe_url(j[4]), jobs))
    return pd.DataFrame([{"condition_id": j[0], "role": j[1], "source_image_id": j[2], "experiment_key": j[3],
                          "url": j[4], **r} for j, r in zip(jobs, res)])


def human_behavior_manifest(meta: FeatherMetadata, experiment_keys: list[str]) -> pd.DataFrame:
    """METADATA/LINKS ONLY. No behavioural file is read here; linkage_validated stays False until
    human.behavior.validate_linkage passes on the actual files."""
    rows = []
    for ek in experiment_keys:
        p = meta.experiments[ek]
        rows.append({"experiment_key": ek, "experiment_params_url": p.get("experiment_params_web"),
                     "behavior_file": None, "granularity": "unknown_until_inspected", "linkage_validated": False,
                     "chance_level": 1 / 16, "task": "16-way object categorisation"})
    return pd.DataFrame(rows)
