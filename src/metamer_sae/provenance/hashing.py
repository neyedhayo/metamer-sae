"""Hashing helpers used for manifests, caches and artifact records."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode())


def sha256_state_dict(state: dict) -> str:
    """Deterministic hash of a torch state dict (names + raw tensor bytes)."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode() + str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()
