"""Pooled PCA baseline. Fitted on natural-image TRAIN activations only; mean-centred, no whitening."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..provenance.hashing import sha256_file
from .base import Representation


class PCABasis:
    """Frozen full basis; `view(k)` gives a k-component Representation sharing it."""

    def __init__(self, mean, components, explained_variance, explained_variance_ratio, meta=None):
        self.mean = np.asarray(mean, np.float64)
        self.components = np.asarray(components, np.float64)
        self.explained_variance = np.asarray(explained_variance, np.float64)
        self.explained_variance_ratio = np.asarray(explained_variance_ratio, np.float64)
        self.meta = meta or {"centering": "mean", "scaling": None, "whiten": False}

    @classmethod
    def fit(cls, X_train: np.ndarray, n_components: int = 384) -> "PCABasis":
        from sklearn.decomposition import PCA
        p = PCA(n_components=n_components, svd_solver="full", whiten=False).fit(X_train)
        return cls.from_sklearn(p, {"fit_samples": int(len(X_train)), "solver": "full", **{
            "centering": "mean", "scaling": None, "whiten": False}})

    @classmethod
    def from_sklearn(cls, p, meta=None) -> "PCABasis":
        if getattr(p, "whiten", False):
            raise ValueError("Pilot PCA must not be whitened")
        return cls(p.mean_, p.components_, p.explained_variance_, p.explained_variance_ratio_, meta)

    def scores(self, X, k: int | None = None) -> np.ndarray:
        k = k or len(self.components)
        return (np.asarray(X, np.float64) - self.mean) @ self.components[:k].T

    def view(self, k: int) -> "PCAView":
        return PCAView(self, k)

    def retention_table(self, X_val, ks) -> list[dict]:
        cum = np.cumsum(self.explained_variance_ratio)
        rows = []
        for k in ks:
            R = self.mean + self.scores(X_val, k) @ self.components[:k]
            rows.append({"k": int(k), "cumulative_variance": float(cum[k - 1]),
                         "validation_mse": float(((X_val - R) ** 2).mean())})
        return rows

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=False)  # immutable
        np.savez(d / "pca_basis.npz", mean=self.mean, components=self.components,
                 explained_variance=self.explained_variance,
                 explained_variance_ratio=self.explained_variance_ratio)
        (d / "pca_meta.json").write_text(json.dumps({**self.meta, "sha256": sha256_file(d / "pca_basis.npz")}, indent=2))
        return d / "pca_basis.npz"


class PCAView(Representation):
    def __init__(self, basis: PCABasis, k: int):
        self.basis, self.k, self.name = basis, k, f"pca_{k}"

    def embed(self, X):
        return self.basis.scores(X, self.k)

    def reconstruct(self, X):
        return self.basis.mean + self.embed(X) @ self.basis.components[: self.k]

    def metadata(self):
        return {**super().metadata(), "k": self.k, **self.basis.meta}


def load_pca(path: str | Path, expected_sha256: str | None = None) -> PCABasis:
    """Load .npz (preferred) or a TRUSTED local scikit-learn pickle (pilot artifact)."""
    path = Path(path)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"PCA artifact hash mismatch for {path}")
    if path.suffix == ".npz":
        z = np.load(path)
        return PCABasis(z["mean"], z["components"], z["explained_variance"], z["explained_variance_ratio"])
    import joblib  # trusted artifact only; hash-check it via expected_sha256
    return PCABasis.from_sklearn(joblib.load(path))
