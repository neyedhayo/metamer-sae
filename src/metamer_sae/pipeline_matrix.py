"""Orchestration for the expanded study. Each function is one stage inside an immutable Run directory.

Directory convention (spaces are keyed by ActivationSpace.space_id, never by layer name):
  outputs/activations/<space_id>/{natural_train,natural_validation,feather/<role>}/...
  outputs/representations/<space_id>/{pca,sae_seed<k>}/...
Stages that need torch (extraction, SAE training) import it lazily. Stages that need the network are labelled.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config.matrix import MatrixConfig
from .data import matrix_manifests as mm
from .data.splits import matrix_split
from .feather.adapter import FeatherMetadata, ensure_checkout
from .feather import inventory as inv
from .provenance.run import Run
from .registry import adversarial_ipcl_audit, load_registry, resolve_registry


def _meta(cfg: MatrixConfig) -> FeatherMetadata:
    return FeatherMetadata(ensure_checkout(cfg.feather_repository, cfg.feather_dir, cfg.feather_commit))


def run_inventory(cfg: MatrixConfig, run_root: str, probe: bool = True) -> Path:
    """Phase A/B/C/D audit: model + layer inventory, registry resolution, adversarial-IPCL gate, availability scan."""
    reg, meta = load_registry(cfg.registry), _meta(cfg)
    run = Run(run_root, f"inventory_{cfg.experiment_id}", cfg.model_dump(), extra={"feather_commit": cfg.feather_commit})
    models = inv.model_inventory(meta, reg)
    layers = inv.layer_inventory(meta, models)
    ipcl = adversarial_ipcl_audit(sorted(meta.networks))
    tables = {"model_inventory": models.astype({"candidates": str, "feather_experiments": str}), "layer_inventory": layers}
    sel = []
    for c in cfg.conditions:
        r = models[models.condition_id == c.condition_id].iloc[0]
        if not r.resolved_model_key or not r.feather_experiments:
            continue
        L = layers[(layers.condition_id == c.condition_id) & layers.valid_metamer_stage.fillna(False)]
        stages = L.stage.tolist() if c.stages.mode == "all_eligible" else c.stages.stages
        bad = set(stages) - set(L.stage)
        if bad:
            raise ValueError(f"{c.condition_id}: declared stages not listed by Feather: {sorted(bad)}")
        for s in stages:
            sel.append({"condition_id": mm.make_condition_id(c.condition_id, r.feather_experiments[0], s),
                        "experiment_key": r.feather_experiments[0], "model_key": r.resolved_model_key, "stage": s})
    stages_by = {}
    for s in sel:
        stages_by.setdefault(s["condition_id"].split("|")[0], []).append(s["stage"])
    cond = mm.condition_manifest(reg, models, stages_by)
    tables["condition_manifest"] = cond
    tables["human_behavior_manifest"] = inv.human_behavior_manifest(meta, sorted({s["experiment_key"] for s in sel}))
    if probe and sel:                                    # NETWORK: probes every candidate id x condition
        pr = inv.stimulus_probe(meta, sel, cfg.candidate_ids)
        pr.to_csv(run.path("tables", "probe_raw.csv"), index=False)
        tables["stimulus_availability"] = mm.stimulus_availability(pr)
    jsons = {"adversarial_ipcl_audit": ipcl, "download_validation_report": {"status": "not_run_yet"}}
    hashes = mm.write_manifests(cfg.manifests_dir, tables, jsons)
    for n, df in tables.items():
        df.to_csv(run.path("tables", f"{n}.csv"), index=False)
    run.close({"manifest_sha256": hashes, "unresolved": models[models.status != "resolved_pinned"].condition_id.tolist(),
               "adversarial_ipcl": ipcl["status"], "note": "audit only; no stimuli downloaded"})
    return run.dir


def run_split(cfg: MatrixConfig, pairs_csv: str, require_conditions: list[str] | None = None) -> Path:
    """Grouped source-level split over the matrix (every condition of a source inherits its split)."""
    pairs = pd.read_csv(pairs_csv)
    s = cfg.split
    df = matrix_split(pairs, s.n_development, s.n_primary, s.n_reserve, s.seed, None, require_conditions)
    out = Path(cfg.manifests_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "matrix_split_ledger.csv", index=False)
    man = mm.split_manifest(df, s.policy, s.seed, False, "source-level; complete-case across required conditions")
    (out / "split_manifest.json").write_text(json.dumps(man, indent=2))
    pairs["split"] = pairs.source_image_id.map(dict(zip(df.stimulus_id, df.split)))
    mm.assert_split_consistency(pairs)
    pairs.to_csv(pairs_csv, index=False)
    return out / "split_manifest.json"


def run_extract_space(adapter, stage: str, natural_train: pd.DataFrame, natural_val: pd.DataFrame,
                      feather_items: dict[str, pd.DataFrame], out_root: str, device: str = "cpu",
                      batch_size: int = 64) -> dict:
    """Extract natural train/val and Feather blocks for ONE activation space (needs torch + checkpoints)."""
    from .extraction.engine import extract, verify
    from .representations.tracks import ActivationSpace
    ex = adapter.extractor(stage, device)
    sp = ActivationSpace.from_meta(ex.meta); base = Path(out_root) / "activations" / sp.space_id
    dim = int(np.prod(ex.meta["output_shape"][:1]))
    out = {"space_id": sp.space_id, "meta": ex.meta}
    for name, items in {"natural_train": natural_train, "natural_validation": natural_val, **{f"feather/{k}": v for k, v in feather_items.items()}}.items():
        d = extract(ex, items, base / name, "sample_id", batch_size, device)
        out[name] = verify(d, len(items), dim)
    (base / "space_meta.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_fit_space(space_dir: str, cfg: MatrixConfig, run_root: str, device: str = "cpu") -> Path:
    """Fit PCA + SAE seeds in ONE activation space from natural activations only (guarded)."""
    from .config.schema import SAECfg
    from .representations.pca import PCABasis
    from .representations.tracks import assert_natural_only
    from .training.train_topk import train_topk
    base = Path(space_dir)
    tr = np.asarray(np.load(base / "natural_train" / "pooled_activations.npy", mmap_mode="r"))
    va = np.asarray(np.load(base / "natural_validation" / "pooled_activations.npy", mmap_mode="r"))
    meta_tr = pd.read_csv(base / "natural_train" / "metadata.csv")
    feather_hashes = set()
    for p in (base / "feather").glob("*/metadata.csv"):
        feather_hashes |= set(pd.read_csv(p).image_sha256)
    assert_natural_only(meta_tr, feather_hashes, cfg.natural.dataset_id, cfg.natural.dataset_id)
    D = tr.shape[1]
    run = Run(run_root, f"fit_{base.name}", {**cfg.model_dump(), "space_dir": str(base), "dim": D})
    k0 = cfg.pca.fitted_components or D
    basis = PCABasis.fit(tr, min(k0, D, len(tr))); basis.save(run.path("pca"))
    mean = tr.mean(0).astype(np.float32); np.save(run.path("sae", "training_mean.npy"), mean)
    results = {}
    for exp in cfg.sae.expansions:
        for k in cfg.sae.top_k:
            for seed in cfg.sae.seeds:
                sc = SAECfg(input_dim=D, dictionary_width=D * exp, top_k=min(k, D * exp), training_mean="x",
                            primary_checkpoint="x", replicate_checkpoint="x", lr=cfg.sae.lr,
                            batch_size=cfg.sae.batch_size, max_epochs=cfg.sae.max_epochs)
                results[f"x{exp}_k{k}_seed{seed}"] = train_topk(sc, tr - mean, va - mean,
                                                                run.path("sae", f"x{exp}_k{k}_seed{seed}"), seed, device=device)
    run.close({"dim": D, "n_fit": len(tr), "fit_data": "natural_train only", "sae": results})
    return run.dir
