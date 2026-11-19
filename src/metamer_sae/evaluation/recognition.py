"""Recognition-model baselines (model accuracy on the 16-way task). NEVER human accuracy.

* Supervised models: native 1000-way head -> 16 categories through an explicit CategoryMap supplied by the user
  (ImageNet-class -> category table from Feather's code; this repo does not invent it).
* IPCL / self-supervised: a linear readout trained on frozen features (backbone untouched); the training split
  and method are recorded. Confidences are not compared across models; only top-1 accuracy per condition.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..provenance.hashing import sha256_json

CATEGORIES_16 = ["airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat", "chair", "clock", "dog",
                 "elephant", "keyboard", "knife", "oven", "truck"]
CHANCE = 1 / 16


class CategoryMap:
    """class index -> category index in CATEGORIES_16 (classes absent from the map are ignored in pooling)."""
    def __init__(self, mapping: dict[int, int]):
        bad = [v for v in mapping.values() if not 0 <= v < 16]
        if not mapping or bad:
            raise ValueError("CategoryMap must be non-empty with category ids in 0..15")
        self.mapping = {int(k): int(v) for k, v in mapping.items()}

    @classmethod
    def from_csv(cls, path: str):
        d = pd.read_csv(path)    # columns: class_index, category (name in CATEGORIES_16)
        return cls({int(r.class_index): CATEGORIES_16.index(r.category) for r in d.itertuples()})

    def sha256(self) -> str:
        return sha256_json(sorted(self.mapping.items()))

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Sum class probabilities within each category, then argmax. (Assumption: verify vs Feather's own rule.)"""
        out = np.zeros((len(probs), 16))
        for k, v in self.mapping.items():
            out[:, v] += probs[:, k]
        return out.argmax(1)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def fit_linear_readout(train_feats: np.ndarray, train_labels: np.ndarray, seed: int = 0, C: float = 1.0) -> dict:
    """Frozen-backbone readout. Returns the sklearn model + provenance. Train on NATURAL images only."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(train_feats, train_labels)
    return {"model": m, "classes": m.classes_, "n_train": len(train_labels),
            "train_data_sha256": sha256_json([float(train_feats.sum()), list(map(int, train_labels[:1000]))]),
            "method": f"sklearn LogisticRegression(C={C}) on frozen pooled features"}


def accuracy_by_condition(pred: np.ndarray, truth: np.ndarray, source_ids: np.ndarray, condition_id: str) -> dict:
    ok = pred == truth
    return {"condition_id": condition_id, "n": int(len(ok)), "accuracy": float(ok.mean()), "chance": CHANCE,
            "n_sources": int(len(set(source_ids))), "kind": "recognition_model_accuracy_not_human"}
