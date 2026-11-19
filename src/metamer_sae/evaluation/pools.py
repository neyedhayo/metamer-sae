"""Split-guarded access to Feather activation pools. Reserve rows are never loaded."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

GUARDED = {"reserve"}


class FeatherPools:
    def __init__(self, role_dirs: dict[str, Path], ledger: pd.DataFrame, expected_rows: int | None = None):
        self.ledger = ledger
        self.arrays = {r: np.load(Path(d) / "pooled_activations.npy", mmap_mode="r") for r, d in role_dirs.items()}
        n = expected_rows or len(ledger)
        for r, a in self.arrays.items():
            if a.ndim != 2 or a.shape[0] != n:
                raise ValueError(f"pool '{r}' has shape {a.shape}, expected ({n}, D)")

    def split_ids(self, split: str) -> list[int]:
        return self.ledger[self.ledger.split == split].stimulus_id.astype(int).tolist()

    def rows(self, split: str, allow_guarded: bool = False) -> tuple[list[int], dict[str, np.ndarray]]:
        if split in GUARDED and not allow_guarded:
            raise PermissionError(f"Split '{split}' activation rows must not be loaded in the pilot.")
        sub = self.ledger[self.ledger.split == split]
        idx = sub.activation_index.to_numpy()
        return sub.stimulus_id.astype(int).tolist(), {r: np.asarray(a[idx], np.float32) for r, a in self.arrays.items()}
