"""Orchestration: each function is one stage, executed inside an immutable Run directory."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .config.schema import RunConfig
from .data.splits import load_ledger, split_hash
from .evaluation.paired import evaluate_pairs
from .evaluation.pools import FeatherPools
from .provenance.hashing import sha256_file
from .provenance.run import Run
from .representations.base import RawRepresentation
from .representations.pca import PCABasis, load_pca
from .representations.sae import SAERepresentation, load_topk
from .statistics.paired import contrast_family
from .visualization import figures


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(path):  # natural activations
    return np.load(path, mmap_mode="r")


def build_representations(cfg: RunConfig, device: str | None = None) -> dict:
    device = device or _device()
    basis = load_pca(cfg.pca.artifact, cfg.pca.artifact_sha256)
    mean = np.load(cfg.sae.training_mean)
    s = cfg.sae
    kw = dict(input_dim=s.input_dim, width=s.dictionary_width, top_k=s.top_k, device=device, version=s.version)
    m123, _ = load_topk(s.primary_checkpoint, **kw)
    m42, _ = load_topk(s.replicate_checkpoint, **kw)
    reps = {"raw_384": RawRepresentation(), f"pca_{cfg.pca.primary_components}": basis.view(cfg.pca.primary_components)}
    for k in cfg.pca.sensitivity_components:
        reps[f"pca_{k}"] = basis.view(k)
    reps["sae_seed123"] = SAERepresentation(m123, mean, "sae_seed123", device)
    reps["sae_seed42"] = SAERepresentation(m42, mean, "sae_seed42", device)
    return reps


def open_pools(cfg: RunConfig) -> FeatherPools:
    ledger = load_ledger(cfg.stimuli.split_manifest)
    dirs = {r: Path(cfg.paths.feather_activations) / d for r, d in cfg.paths.feather_role_dirs.items()}
    return FeatherPools(dirs, ledger)


def run_fit_pca(cfg: RunConfig, run_root: str) -> Path:
    """Fit a NEW pooled PCA on natural TRAIN activations (the pilot PCA is never refit)."""
    Xtr = np.asarray(_load(Path(cfg.paths.natural_activations) / "train" / "pooled_activations.npy"))
    Xva = np.asarray(_load(Path(cfg.paths.natural_activations) / "validation" / "pooled_activations.npy"))
    run = Run(run_root, "pca", cfg.model_dump())
    basis = PCABasis.fit(Xtr, cfg.pca.fitted_components)
    basis.save(run.path("basis"))
    ks = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384]
    tab = pd.DataFrame(basis.retention_table(Xva, [k for k in ks if k <= cfg.pca.fitted_components]))
    tab.to_csv(run.path("component_retention.csv"), index=False)
    figures.pca_retention_figure(tab, run.path("figures", "pca_retention"))
    run.close({"fit_samples": len(Xtr)})
    return run.dir


def run_train_sae(cfg: RunConfig, run_root: str, seed: int, width: int | None, top_k: int | None) -> Path:
    from .training.train_topk import train_topk
    mean = np.load(cfg.sae.training_mean)
    Xtr = np.asarray(_load(Path(cfg.paths.natural_activations) / "train" / "pooled_activations.npy")) - mean
    Xva = np.asarray(_load(Path(cfg.paths.natural_activations) / "validation" / "pooled_activations.npy")) - mean
    run = Run(run_root, f"topk_seed{seed}", {**cfg.model_dump(), "train_seed": seed})
    res = train_topk(cfg.sae, Xtr, Xva, run.path("model"), seed, width, top_k, _device())
    run.close(res)
    return run.dir


def run_evaluate(cfg: RunConfig, run_root: str) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Development first, then primary evaluation. Reserve rows are never touched."""
    reps = build_representations(cfg)
    pools = open_pools(cfg)
    ev = cfg.evaluation
    run = Run(run_root, "eval", cfg.model_dump(), extra={
        "split_sha256": split_hash(pools.ledger), "pca_sha256": sha256_file(cfg.pca.artifact),
        "sae_checkpoint_sha256": {"seed123": sha256_file(cfg.sae.primary_checkpoint),
                                  "seed42": sha256_file(cfg.sae.replicate_checkpoint)}})
    out = {}
    for split in (ev.development_split, ev.primary_split):
        ids, acts = pools.rows(split)
        stim, pair = evaluate_pairs(split, ids, acts, reps, run_meta={"run_id": run.run_id,
                                    "experiment_id": cfg.experiment.id})
        stim.to_csv(run.path("tables", f"{split}_stimulus_metrics.csv"), index=False)
        pair.to_csv(run.path("tables", f"{split}_paired_metrics.csv"), index=False)
        out[split] = (ids, pair)
    ids, pair = out[ev.primary_split]
    contrasts = contrast_family(pair, ids, [r for r in reps], ev.bootstrap_resamples, ev.bootstrap_seed)
    contrasts.to_csv(run.path("tables", "primary_paired_contrasts_bootstrap.csv"), index=False)
    figures.paired_distance_figure(pair, "distance_l2", list(reps), run.path("figures", "primary_l2_paired"))
    figures.contrast_forest(contrasts, run.path("figures", "primary_contrast_forest"))
    l2 = pair.pivot_table(index=["representation", "stimulus_id"], columns="metamer_condition", values="distance_l2")
    run.close({"n_primary": len(ids), "n_contrasts": len(contrasts),
               "l2_standard_gt_adversarial": {r: int((l2.loc[r].standard > l2.loc[r].adversarial).sum()) for r in reps},
               "max_holm_p": float(contrasts.wilcoxon_p_holm.max()),
               "note": "Model-representation findings only; no human behaviour was loaded."})
    return run.dir, pair, contrasts
