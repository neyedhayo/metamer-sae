"""Natural-image corpus listing (Imagenette layout: <root>/{train,val}/<wnid>/*.JPEG)."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

_EXT = {".jpeg", ".jpg", ".png"}


def list_natural_images(root: str | Path, split: str) -> pd.DataFrame:
    sub = {"train": "train", "validation": "val"}[split]
    paths = sorted(p for p in (Path(root) / sub).rglob("*") if p.suffix.lower() in _EXT)
    return pd.DataFrame({"sample_id": [f"{split}:{i:05d}" for i in range(len(paths))],
                         "image_path": [str(p) for p in paths], "class_wnid": [p.parent.name for p in paths],
                         "split": split})
