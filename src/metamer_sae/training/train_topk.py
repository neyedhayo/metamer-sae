"""TopK-SAE training with the pilot's checkpoint policy.

Policy: validate every epoch; save a checkpoint on every strict absolute validation-MSE decrease;
`min_delta` is used only for early-stopping patience; restore best weights; Adam + MSE; clip grad.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config.schema import SAECfg
from ..representations.sae import build_topk


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _val(model, X, bs, device):
    model.eval(); tot = 0.0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        tot += float(((model(xb)[2] - xb) ** 2).sum())
    return tot / X.numel()


def train_topk(cfg: SAECfg, Xtr_centered: np.ndarray, Xva_centered: np.ndarray, out_dir: str | Path,
               seed: int, width: int | None = None, top_k: int | None = None, device: str = "cpu") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)  # never overwrite checkpoints
    width, top_k = width or cfg.dictionary_width, top_k or cfg.top_k
    seed_everything(seed)
    model = build_topk(cfg.input_dim, width, top_k, device, cfg.version)
    Xtr, Xva = torch.from_numpy(Xtr_centered.astype(np.float32)), torch.from_numpy(Xva_centered.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr), batch_size=cfg.batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience,
                                                       min_lr=cfg.min_lr)
    best, best_epoch, stall, hist = float("inf"), 0, 0, []
    ck = out / "best_model.pt"
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); tl, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = torch.nn.functional.mse_loss(model(xb)[2], xb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            tl += float(loss) * len(xb); n += len(xb)
        v = _val(model, Xva, cfg.batch_size, device); sched.step(v)
        hist.append({"epoch": epoch, "train_mse": tl / n, "val_mse": v, "lr": opt.param_groups[0]["lr"]})
        if v < best:                                  # strict absolute decrease -> checkpoint
            improved_enough = (best - v) > cfg.min_delta
            best, best_epoch = v, epoch
            torch.save({"model_state_dict": {k: t.cpu() for k, t in model.state_dict().items()},
                        "config": {"input_dim": cfg.input_dim, "width": width, "top_k": top_k, "seed": seed},
                        "best_epoch": epoch, "best_val_mse": v}, ck)
            stall = 0 if improved_enough else stall + 1
        else:
            stall += 1
        if stall >= cfg.early_stop_patience:
            break
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    res = {"best_epoch": best_epoch, "best_val_mse": best, "epochs_completed": len(hist), "seed": seed,
           "width": width, "top_k": top_k, "stopped_early": len(hist) < cfg.max_epochs}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res
