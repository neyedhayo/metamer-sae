"""Pair-level image-space baselines + smoothness confounds, driven by the metamer pair manifest.

Protocol (record with results): RGB; shorter side -> 236 bilinear; centre crop 224; range [0,1]; pixel L2/RMSE
(lower = closer), SSIM (higher = more similar, channel_axis=2, data_range=1), LPIPS (lower = more similar;
backbone/weights recorded). Smoothness covariates are computed on the METAMER image only. None of these
is a measure of human recognisability.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .baselines import LPIPS, load_image, pixel_metrics, ssim


def smoothness(img: np.ndarray) -> dict:
    g = img.mean(axis=2)
    gx, gy = np.diff(g, axis=1), np.diff(g, axis=0)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    f = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.indices(f.shape); r = np.hypot(yy - f.shape[0] / 2, xx - f.shape[1] / 2)
    hf = f[r > f.shape[0] / 4].sum() / max(f.sum(), 1e-12)
    return {"gradient_energy": float(np.sqrt((gx ** 2).mean() + (gy ** 2).mean())),
            "laplacian_variance": float(lap.var()), "high_freq_power_ratio": float(hf)}


def score_pairs(pairs: pd.DataFrame, stim: pd.DataFrame, use_ssim=True, lpips_net: str | None = "alex",
                device: str = "cpu") -> pd.DataFrame:
    path = dict(zip(stim.stimulus_id, stim.image_path))
    lp = LPIPS(lpips_net, device) if lpips_net else None
    rows = []
    for r in pairs[pairs.pair_complete].itertuples():
        a, b = load_image(path[r.source_stimulus_id]), load_image(path[r.metamer_stimulus_id])
        row = {"pair_id": r.pair_id, "source_image_id": r.source_image_id, "condition_id": r.condition_id,
               **pixel_metrics(a, b), **{f"metamer_{k}": v for k, v in smoothness(b).items()}}
        if use_ssim:
            row["ssim"] = ssim(a, b)
        if lp:
            row["lpips"] = lp(a, b); row["lpips_net"] = lpips_net
        rows.append(row)
    return pd.DataFrame(rows)
