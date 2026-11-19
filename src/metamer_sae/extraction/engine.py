"""Resumable batch activation extraction with cache keys and integrity assertions."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import EXTRACTION_CODE_VERSION
from ..models.observer import Observer
from ..provenance.hashing import sha256_file, sha256_json


class _Imgs(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.tf = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def cache_key(image_hashes: list[str], obs_meta: dict) -> str:
    key = {"images": sha256_json(image_hashes), "weights": obs_meta["weights_sha256"],
           "layer": obs_meta["layer"], "preproc": obs_meta["preprocessing_sha256"],
           "pooling": obs_meta["pooling"], "code": EXTRACTION_CODE_VERSION}
    # model identity / module path join the key only when present, so pilot cache keys are unchanged
    for k in ("model_id", "module_path"):
        if k in obs_meta:
            key[k] = obs_meta[k]
    return sha256_json(key)


def extract(observer: Observer, items: pd.DataFrame, out_dir: str | Path, id_col: str = "sample_id",
            batch_size: int = 64, device: str = "cpu", num_workers: int = 0,
            save_spatial: bool = True, resume: bool = True) -> Path:
    """items needs [id_col, image_path] (+ optional metadata). Order of `items` defines row order.

    Writes pooled_activations.npy (float32), spatial_activations.npy (float16), metadata.csv,
    progress.json, cache_key.json. Re-running resumes; a changed cache key refuses to reuse the dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = items.reset_index(drop=True)
    hashes = [sha256_file(p) for p in items.image_path]
    key = cache_key(hashes, observer.meta)
    n, shp = len(items), tuple(observer.meta["output_shape"])
    c = shp[0]                                  # channels or vector length; (h, w) only for spatial stages
    spatial_shape = (n,) + shp if len(shp) == 3 else None
    save_spatial = save_spatial and spatial_shape is not None
    kf, pf = out / "cache_key.json", out / "progress.json"
    if kf.exists() and json.loads(kf.read_text())["cache_key"] != key:
        raise RuntimeError(f"Cache key mismatch in {out}; configuration or inputs changed. Use a new directory.")
    kf.write_text(json.dumps({"cache_key": key, "n": n, "observer": observer.meta,
                              "code_version": EXTRACTION_CODE_VERSION}, indent=2))
    done = json.loads(pf.read_text())["n_done"] if (resume and pf.exists()) else 0
    mode = "r+" if done and (out / "pooled_activations.npy").exists() else "w+"
    pooled = np.lib.format.open_memmap(out / "pooled_activations.npy", mode=mode, dtype=np.float32, shape=(n, c))
    spatial = (np.lib.format.open_memmap(out / "spatial_activations.npy", mode=mode, dtype=np.float16,
                                         shape=spatial_shape) if save_spatial else None)
    loader = DataLoader(_Imgs(items.image_path[done:], observer.transform), batch_size=batch_size,
                        num_workers=num_workers)
    pos = done
    for batch in loader:
        s = observer.spatial(batch.to(device))
        p = observer.pool(s).cpu().numpy()
        assert tuple(s.shape[1:]) == shp and np.isfinite(p).all(), "bad shape or non-finite activations"
        pooled[pos:pos + len(p)] = p
        if spatial is not None:
            spatial[pos:pos + len(p)] = s.cpu().numpy().astype(np.float16)
        pos += len(p)
        pooled.flush()
        if spatial is not None:
            spatial.flush()
        pf.write_text(json.dumps({"n_done": pos}))
    assert pos == n, f"extraction incomplete: {pos}/{n}"
    meta = items.assign(image_sha256=hashes, row_index=range(n))
    meta.to_csv(out / "metadata.csv", index=False)
    pf.write_text(json.dumps({"n_done": n, "complete": True}))
    return out


def verify(out_dir: str | Path, expected_n: int, dim: int = 384) -> dict:
    """`dim` is the pooled dimension of THIS observer/stage (384 only for the pilot ConvNeXt stage 3)."""
    out = Path(out_dir)
    a = np.load(out / "pooled_activations.npy", mmap_mode="r")
    ok = a.shape == (expected_n, dim) and a.dtype == np.float32 and bool(np.isfinite(a).all())
    meta_n = len(pd.read_csv(out / "metadata.csv"))
    return {"dir": str(out), "shape": list(a.shape), "dtype": str(a.dtype), "finite": bool(np.isfinite(a).all()),
            "metadata_rows": meta_n, "ok": bool(ok and meta_n == expected_n)}
