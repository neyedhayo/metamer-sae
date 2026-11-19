"""Audit registry entries against the REAL Feather metadata. Nothing is guessed: zero matches -> `unresolved`
(an availability gap), several -> `ambiguous` (a human must pin `model_key`)."""
from __future__ import annotations
import re

import pandas as pd

from .schema import Registry


def classify_model_key(model_key: str) -> dict:
    """Heuristic label for a Feather key. REPORTING only; it never selects a model."""
    k = model_key.lower()
    return {"model_key": model_key, "mentions_ipcl": "ipcl" in k,
            "mentions_robust": bool(re.search(r"robust|adv|l2_|linf", k)),
            "mentions_random": "random" in k, "heuristic_only": True}


def resolve_registry(reg: Registry, model_keys: list[str]) -> pd.DataFrame:
    rows = []
    for c in reg.conditions:
        base = {"condition_id": c.condition_id, "family": c.family, "training_objective": c.training_objective.value}
        if c.model_key:
            ok = c.model_key in model_keys
            rows.append({**base, "status": "resolved_pinned" if ok else "pinned_key_missing_in_metadata",
                         "resolved_model_key": c.model_key if ok else None,
                         "candidates": [c.model_key] if ok else [], "verified": c.verified})
            continue
        cands = [k for k in model_keys if any(re.search(p, k) for p in c.model_key_patterns)
                 and not any(re.search(p, k) for p in c.exclude_patterns)]
        status = "unresolved" if not cands else ("resolved_unique_candidate" if len(cands) == 1 else "ambiguous")
        rows.append({**base, "status": status, "resolved_model_key": cands[0] if len(cands) == 1 else None,
                     "candidates": cands, "verified": False})
    df = pd.DataFrame(rows)
    df["action_required"] = df.status.map({
        "resolved_pinned": "none", "resolved_unique_candidate": "inspect, then pin model_key + set verified: true",
        "ambiguous": "pin the correct model_key by hand", "unresolved": "record availability gap",
        "pinned_key_missing_in_metadata": "pinned key not in pinned Feather metadata: investigate"})
    return df


def adversarial_ipcl_audit(model_keys: list[str]) -> dict:
    """Phase D gate: is there an IPCL key that is also adversarially trained? Never relabels IPCL."""
    info = {k: classify_model_key(k) for k in model_keys}
    hits = [k for k, v in info.items() if v["mentions_ipcl"] and v["mentions_robust"]]
    found = bool(hits)
    return {"status": "candidate_found_requires_manual_verification" if found else "unavailable_in_feather_metadata",
            "candidate_keys": hits, "all_ipcl_keys": [k for k, v in info.items() if v["mentions_ipcl"]],
            "must_not": ["relabel self-supervised IPCL as adversarial", "substitute robust supervised AlexNet"],
            "if_unavailable_requires": ["robust-training recipe (objective, norm, epsilon, schedule)",
                                        "trained AlexNetGN-IPCL checkpoint + validation",
                                        "metamer generation per stage (~24k optimisation steps/image)",
                                        "optimisation-fidelity / null-distribution validation",
                                        "separately versioned dataset; no human data exists for it"],
            "protocol_doc": "docs/protocols/ADVERSARIAL_IPCL_EXTENSION.md"}
