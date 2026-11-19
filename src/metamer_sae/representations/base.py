"""Common representation interface (fit / embed / reconstruct / error / save / load)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Representation(ABC):
    name: str

    @abstractmethod
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Coordinates in which source-to-metamer distances are measured (raw activations in)."""

    @abstractmethod
    def reconstruct(self, X: np.ndarray) -> np.ndarray | None:
        """Reconstruction in the ORIGINAL (uncentred) activation space, or None if undefined."""

    def reconstruction_mse(self, X: np.ndarray) -> np.ndarray:
        R = self.reconstruct(X)
        if R is None:
            return np.full(len(X), np.nan)
        return ((np.asarray(X, np.float64) - R) ** 2).mean(axis=1)

    def fit(self, X: np.ndarray):  # frozen representations may not support fitting
        raise NotImplementedError(f"{type(self).__name__} is frozen / loaded from an artifact")

    def save(self, directory: str | Path):
        raise NotImplementedError

    def metadata(self) -> dict:
        return {"name": self.name, "type": type(self).__name__}


class RawRepresentation(Representation):
    name = "raw_384"

    def embed(self, X):
        return np.asarray(X, np.float64)

    def reconstruct(self, X):
        return None
