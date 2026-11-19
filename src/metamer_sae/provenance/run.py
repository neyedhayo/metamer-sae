"""Immutable run directories, provenance records and closure reports."""
from __future__ import annotations
import datetime as dt, json, platform, subprocess, sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json

_PKGS = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision",
         "overcomplete", "pillow", "pydantic", "matplotlib"]


def git_commit(path: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    out = {}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not-installed"
    return out


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"


class Run:
    """An immutable run directory. A changed config must create a new run."""

    def __init__(self, root: str | Path, prefix: str, config: dict, run_id: str | None = None,
                 extra: dict | None = None):
        self.run_id = run_id or new_run_id(prefix)
        self.dir = Path(root) / self.run_id
        if self.dir.exists():
            raise FileExistsError(f"Run directory exists and runs are immutable: {self.dir}")
        self.dir.mkdir(parents=True)
        self.config = config
        self.artifacts: dict[str, str] = {}
        (self.dir / "config.resolved.json").write_text(json.dumps(config, indent=2, default=str))
        prov = {
            "run_id": self.run_id,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_sha256": sha256_json(config),
            "metamer_sae_commit": git_commit(Path(__file__).parents[3]),
            "feather_commit": (config.get("feather") or {}).get("commit"),
            "packages": package_versions(), "runtime": runtime_info(), **(extra or {}),
        }
        (self.dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def register(self, path: str | Path) -> None:
        path = Path(path)
        self.artifacts[str(path.relative_to(self.dir))] = sha256_file(path)

    def close(self, summary: dict | None = None, status: str = "complete") -> Path:
        for p in self.dir.rglob("*"):
            if p.is_file() and p.name != "closure_report.json":
                self.register(p)
        report = {"run_id": self.run_id, "status": status,
                  "closed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "summary": summary or {}, "artifact_sha256": self.artifacts}
        out = self.dir / "closure_report.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        return out


def verify_run(run_dir: str | Path) -> dict:
    """Re-hash every artifact against closure_report.json. Used before a run may be marked reviewed."""
    d = Path(run_dir); rep = json.loads((d / "closure_report.json").read_text())
    bad = [k for k, h in rep["artifact_sha256"].items() if not (d / k).exists() or sha256_file(d / k) != h]
    return {"run_id": rep["run_id"], "n_artifacts": len(rep["artifact_sha256"]), "mismatched": bad, "ok": not bad}


def mark_reviewed(run_dir: str | Path, reviewer: str, note: str = "") -> Path:
    """Freeze a verified run. After this a run is immutable; a rerun must receive a new run_id."""
    d = Path(run_dir); v = verify_run(d)
    if not v["ok"]:
        raise RuntimeError(f"cannot mark reviewed; artifacts changed: {v['mismatched']}")
    p = d / "REVIEWED.json"
    if p.exists():
        raise FileExistsError("run already reviewed")
    p.write_text(json.dumps({"reviewer": reviewer, "note": note, "closure_sha256": sha256_file(d / "closure_report.json"),
                             "reviewed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2))
    return p
