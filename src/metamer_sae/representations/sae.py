"""Overcomplete TopK-SAE adapter: safe checkpoint loading, frozen inference, decoder access."""
from __future__ import annotations
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from .base import Representation


def _overcomplete(expected: str | None = "0.3.0"):
    try:
        v = metadata.version("overcomplete")
    except metadata.PackageNotFoundError as e:
        raise ImportError("Install the validated dependency: pip install overcomplete==0.3.0") from e
    if expected and v != expected:
        raise RuntimeError(f"Overcomplete {v} != validated {expected}. Inspect the API; do not silently switch.")
    from overcomplete.sae import TopKSAE
    return TopKSAE


def build_topk(input_dim: int, width: int, top_k: int, device="cpu", version="0.3.0"):
    return _overcomplete(version)(input_shape=input_dim, nb_concepts=width, top_k=top_k, device=device)


def safe_load_checkpoint(path: str | Path) -> dict:
    """Restricted deserialisation (no unrestricted pickle)."""
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_topk(path, input_dim=384, width=1536, top_k=64, device="cpu", version="0.3.0"):
    ck = safe_load_checkpoint(path)
    model = build_topk(input_dim, width, top_k, "cpu", version)
    state = ck.get("model_state_dict", ck.get("state_dict", ck))
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


class SAERepresentation(Representation):
    """Input: RAW pooled activations. Centres with the saved natural-TRAIN mean only (no scaling)."""

    def __init__(self, model, mean: np.ndarray, name: str, device="cpu", batch_size=128):
        self.model, self.mean, self.name = model, np.asarray(mean, np.float32), name
        self.device, self.bs = device, batch_size
        self.top_k = getattr(model, "top_k", None)

    @torch.inference_mode()
    def _run(self, X):
        Xc = (np.asarray(X, np.float32) - self.mean[None])
        codes, recon = [], []
        self.model.eval()
        for i in range(0, len(Xc), self.bs):
            xb = torch.from_numpy(np.ascontiguousarray(Xc[i:i + self.bs])).to(self.device)
            _, z, r = self.model(xb)
            codes.append(z.cpu().numpy()); recon.append(r.cpu().numpy())
        return np.concatenate(codes), np.concatenate(recon)

    def embed(self, X):
        codes, _ = self._run(X)
        if self.top_k is not None:
            n = np.count_nonzero(codes, axis=1)
            assert np.all(n == self.top_k), f"expected exactly {self.top_k} active codes, got {np.unique(n)}"
        return codes.astype(np.float64)

    def encode_and_reconstruct(self, X):
        codes, recon = self._run(X)
        return codes, recon + self.mean[None]

    def reconstruct(self, X):
        return self.encode_and_reconstruct(X)[1].astype(np.float64)

    def dictionary(self) -> np.ndarray:
        """Decoder atoms, shape (width, input_dim)."""
        if hasattr(self.model, "get_dictionary"):
            return self.model.get_dictionary().detach().cpu().numpy()
        for p in self.model.parameters():
            if p.ndim == 2 and p.shape[0] == self.model.nb_concepts:
                return p.detach().cpu().numpy()
        raise RuntimeError("Cannot locate decoder dictionary on this Overcomplete version")
