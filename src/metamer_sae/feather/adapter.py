"""Thin, read-only adapter over Feather et al.'s public repository.

Metadata (experiment groups, network order, layer lists) is read from the pinned
checkout; image URLs follow the scheme validated in Notebook 02. Nothing is guessed:
if a network/layer is not in Feather's metadata a ValueError is raised.
"""
from __future__ import annotations
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

_UA = {"User-Agent": "Mozilla/5.0"}


def ensure_checkout(repo_url: str, local_dir: str | Path, commit: str) -> Path:
    local_dir = Path(local_dir)
    if not local_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(local_dir)], check=True)
    elif not (local_dir / ".git").exists():
        raise RuntimeError(f"{local_dir} exists but is not a git repository")
    subprocess.run(["git", "-C", str(local_dir), "checkout", "--detach", commit], check=True)
    head = subprocess.check_output(["git", "-C", str(local_dir), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise RuntimeError(f"Feather commit mismatch: {head} != {commit}")
    return local_dir


def read_remote_json(url: str, timeout: int = 40):
    with urlopen(Request(url, headers=_UA), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def probe_url(url: str, timeout: int = 25) -> dict:
    """HEAD probe with a tiny ranged-GET fallback (as in Notebook 02)."""
    def _try(method, headers):
        with urlopen(Request(url, headers=headers, method=method), timeout=timeout) as r:
            return r.status
    try:
        s = _try("HEAD", _UA)
        return {"available": s == 200, "status_code": s, "probe_method": "HEAD", "error": None}
    except HTTPError as e:
        if e.code not in (403, 405, 501):
            return {"available": False, "status_code": e.code, "probe_method": "HEAD", "error": str(e)}
    except (URLError, TimeoutError, OSError) as e:
        return {"available": False, "status_code": None, "probe_method": "HEAD", "error": str(e)}
    try:
        s = _try("GET", {**_UA, "Range": "bytes=0-0"})
        return {"available": s in (200, 206), "status_code": s, "probe_method": "GET-range", "error": None}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "status_code": getattr(e, "code", None),
                "probe_method": "GET-range", "error": str(e)}


@dataclass(frozen=True)
class ConditionContext:
    experiment_key: str
    model_key: str
    layer: str
    base: str
    model_index: int
    layer_index: int


class FeatherMetadata:
    def __init__(self, repo_dir: str | Path):
        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from model_analysis_folders import all_model_info  # type: ignore  # noqa: PLC0415
        self.experiments = all_model_info.TURK_IMAGE_EXPERIMENTS_GROUPINGS
        self.networks = all_model_info.ALL_NETWORKS_AND_LAYERS_IMAGES
        self._params: dict[str, dict] = {}

    def _experiment_params(self, key: str) -> dict:
        if key not in self.experiments:
            raise ValueError(f"Unknown Feather visual experiment: {key}")
        if key not in self._params:
            self._params[key] = read_remote_json(self.experiments[key]["experiment_params_web"])
        return self._params[key]

    def layers(self, model_key: str) -> list[str]:
        return [l for l in self.networks[model_key]["layers"] if "input_after_preproc" not in l]

    def context(self, experiment_key: str, model_key: str, layer: str) -> ConditionContext:
        order = self._experiment_params(experiment_key)["experiment_info"]["networks"]
        if model_key not in order:
            raise ValueError(f"{model_key} not listed in {experiment_key}")
        layers = self.layers(model_key)
        if layer not in layers:
            raise ValueError(f"{layer} unavailable for {model_key}")
        return ConditionContext(experiment_key, model_key, layer,
                                self.experiments[experiment_key]["experiment_folder_web"].rstrip("/"),
                                order.index(model_key), layers.index(layer))

    def enumerate_conditions(self) -> pd.DataFrame:
        """Every (experiment, network, layer) Feather lists for visual experiments."""
        rows = []
        for ek in self.experiments:
            nets = self._experiment_params(ek)["experiment_info"]["networks"]
            for mk in nets:
                if mk not in self.networks:
                    continue
                for li, layer in enumerate(self.layers(mk)):
                    rows.append({"experiment_key": ek, "model_key": mk, "layer": layer, "layer_index": li})
        return pd.DataFrame(rows)

    def source_url(self, experiment_key: str, image_index: int) -> str:
        base = self.experiments[experiment_key]["experiment_folder_web"].rstrip("/")
        return f"{base}/0_orig/0_orig/{image_index}_orig_orig.png"

    @staticmethod
    def metamer_url(ctx: ConditionContext, image_index: int) -> str:
        return (f"{ctx.base}/{ctx.model_index + 1}_{ctx.model_key}/{ctx.layer_index}_{ctx.layer}/"
                f"{image_index}_{ctx.model_key}_{ctx.layer}.png")


def inventory_availability(meta: FeatherMetadata, conditions, candidate_ids, workers: int = 16) -> pd.DataFrame:
    """Probe source + metamer availability for each candidate id and condition."""
    jobs = []
    ctxs = {c.id: meta.context(c.experiment_key, c.model_key, c.layer) for c in conditions}
    src_exp = conditions[0].experiment_key
    for i in candidate_ids:
        jobs.append(("source", "source", i, meta.source_url(src_exp, i), None))
        for c in conditions:
            jobs.append((c.id, c.role, i, meta.metamer_url(ctxs[c.id], i), c))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: probe_url(j[3]), jobs))
    rows = [{"condition_id": j[0], "role": j[1], "source_image_id": j[2], "url": j[3], **r}
            for j, r in zip(jobs, res)]
    return pd.DataFrame(rows)
