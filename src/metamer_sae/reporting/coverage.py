"""Coverage heatmap (Matplotlib only) and acceptance audit against README scope §15."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd


def coverage_figure(cov: pd.DataFrame, base: str | Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = cov.assign(frac=cov.n_complete / cov.n_sources.clip(lower=1))
    cov["label"] = cov.family + " / " + cov.training_objective
    piv = cov.pivot_table(index="label", columns="stage", values="frac", aggfunc="max")
    fig, ax = plt.subplots(figsize=(1 + 0.8 * piv.shape[1], 1 + 0.45 * piv.shape[0]))
    im = ax.imshow(piv.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=60, ha="right"); ax.set_yticks(range(piv.shape[0]), piv.index)
    fig.colorbar(im, label="fraction of candidate sources with complete pair")
    ax.set_title("Stimulus coverage (white = no data / gap)"); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def _exists(p): return Path(p).exists()


def acceptance_report(root: str | Path = ".") -> pd.DataFrame:
    """Evidence-based: an item is `evidenced` only if its artifact exists/passes. Code executing is not evidence."""
    r = Path(root); man = r / "data" / "manifests"
    def parity():
        fs = list((r / "outputs" / "runs").glob("*/parity_report.csv"))
        return bool(fs) and all(pd.read_csv(f)["pass"].all() for f in fs), "parity_report.csv in outputs/runs"
    def cond_has(family):
        p = man / "condition_manifest.csv"
        if not p.exists(): return False, "condition_manifest.csv missing"
        d = pd.read_csv(p); return bool(((d.family == family) & (d.availability != "gap")).any()), "condition_manifest.csv"
    def obj_has(obj):
        p = man / "condition_manifest.csv"
        if not p.exists(): return False, "condition_manifest.csv missing"
        d = pd.read_csv(p); return bool(((d.training_objective == obj) & (d.availability != "gap")).any()), "condition_manifest.csv"
    def ipcl_resolved():
        p = man / "adversarial_ipcl_audit.json"
        return _exists(p), "adversarial_ipcl_audit.json (found, or unavailable + protocol)"
    def f(p): return (lambda: (_exists(r / p), p))
    items = [
        ("Exact ResNet-50 pilot parity vs prototype outputs", parity),
        ("Standard AlexNet audited and integrated", lambda: obj_has("supervised_standard") ),
        ("Adversarial AlexNet (L2 eps=3) audited and integrated", lambda: obj_has("adversarial")),
        ("AlexNet random-perturbation control audited (or gap recorded)", lambda: obj_has("random_perturbation")),
        ("IPCL self-supervised audited and integrated", lambda: obj_has("self_supervised_ipcl")),
        ("Matched supervised AlexNetGN audited and integrated", lambda: obj_has("supervised_matched_ipcl_arch")),
        ("Adversarial-IPCL availability resolved", ipcl_resolved),
        ("Model/layer/stimulus inventory", lambda: (all(_exists(man / n) for n in ("model_inventory.csv", "layer_inventory.csv", "stimulus_availability.csv")), "inventory csvs")),
        ("Source identity + grouped splits recorded", f("data/manifests/split_manifest.json")),
        ("Pair manifest present", f("data/manifests/metamer_pair_manifest.csv")),
        ("Track A and Track B runs executed", lambda: (any((r / "outputs" / "runs").glob("*common_observer*")) and any((r / "outputs" / "runs").glob("*generator_native*")), "runs dirs")),
        ("Image baselines (pixel/SSIM/LPIPS) computed", lambda: (any((r / "outputs").rglob("image_baselines.csv")), "image_baselines.csv")),
        ("Recognition baselines computed", lambda: (any((r / "outputs").rglob("recognition_accuracy.csv")), "recognition_accuracy.csv")),
        ("Human linkage validated", lambda: (any((r / "outputs").rglob("linkage_report.json")), "linkage_report.json")),
        ("Statistics with multiplicity correction", lambda: (any((r / "outputs").rglob("family_contrasts.csv")), "family_contrasts.csv")),
        ("Closure reports + figures reviewed", lambda: (any((r / "outputs" / "runs").glob("*/REVIEWED.json")), "REVIEWED.json")),
    ]
    rows = []
    for name, fn in items:
        try:
            ok, ev = fn()
        except Exception as e:  # noqa: BLE001
            ok, ev = False, f"error: {e}"
        rows.append({"criterion": name, "status": "evidenced" if ok else "not_evidenced", "evidence": ev})
    return pd.DataFrame(rows)
