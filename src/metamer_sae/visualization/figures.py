"""Matplotlib-only figures generated from saved tables. PNG + vector PDF; independent scales per space."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"source": "#4d4d4d", "standard": "#d95f02", "adversarial": "#1b9e77"}


def _save(fig, base: str | Path):
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def paired_distance_figure(pair_df: pd.DataFrame, metric: str, representations: list[str], base,
                           label: str = "source→metamer distance (L2 of representation coordinates)"):
    """One panel per representation (independent y-scales); lines link the two metamers of each source."""
    fig, axes = plt.subplots(1, len(representations), figsize=(3.2 * len(representations), 3.6))
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, representations):
        w = pair_df[pair_df.representation == rep].pivot(index="stimulus_id", columns="metamer_condition", values=metric)
        for _, r in w.iterrows():
            ax.plot([0, 1], [r["standard"], r["adversarial"]], color="0.8", lw=.7, zorder=1)
        for x, c in enumerate(("standard", "adversarial")):
            ax.scatter(np.full(len(w), x), w[c], s=14, color=COLORS[c], zorder=2)
            ax.hlines(w[c].mean(), x - .2, x + .2, color="k", lw=1.5)
        ax.set_xticks([0, 1], ["standard", "adversarial"]); ax.set_title(f"{rep}  (n={len(w)})", fontsize=9)
    axes[0].set_ylabel(label, fontsize=8)
    fig.suptitle("Paired source-to-metamer displacement; each line = one source triplet. Scales differ by space.", fontsize=8)
    _save(fig, base)


def contrast_forest(contrast_df: pd.DataFrame, base):
    """Mean paired difference + bootstrap CI per contrast. Units differ across metrics: compare within a row only."""
    d = contrast_df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    ax.errorbar(d.mean_paired_difference, range(len(d)),
                xerr=[d.mean_paired_difference - d.bootstrap_95ci_low, d.bootstrap_95ci_high - d.mean_paired_difference],
                fmt="o", color="#333", ecolor="#888", capsize=2)
    ax.axvline(0, color="k", lw=.6); ax.set_yticks(range(len(d)), d.contrast, fontsize=6)
    ax.set_xlabel("mean paired difference (95% triplet bootstrap CI); definition is in each label", fontsize=7)
    _save(fig, base)


def pca_retention_figure(table: pd.DataFrame, base):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].plot(table.k, table.cumulative_variance * 100, "o-"); ax[0].set_xlabel("components"); ax[0].set_ylabel("cumulative variance (%)")
    ax[1].semilogy(table.k, table.validation_mse.clip(lower=1e-12), "o-"); ax[1].set_xlabel("components"); ax[1].set_ylabel("natural validation MSE")
    _save(fig, base)


def feature_mass_figure(mass: np.ndarray, base):
    m = np.sort(mass)[::-1]; fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(np.cumsum(m) / m.sum()); ax.set_xlabel("features ranked by activation mass"); ax.set_ylabel("cumulative mass share")
    _save(fig, base)
