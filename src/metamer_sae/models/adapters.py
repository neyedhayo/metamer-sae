"""Common model-adapter interface: checkpoint loading, preprocessing, stage discovery, forward hooks, readout.

Roles are kept apart by *use*, not by class: the same adapter can serve as generator-native space, common
observer, or recognition baseline, and the role is recorded in run metadata. Torch is imported lazily so the
registry/config/statistics layers work without it.

Stage names for torchvision models are OBSERVER aliases (documented below); they are NOT Feather's generator
stage identifiers and are never equated with them. Feather stages come from the audited Feather metadata.
"""
from __future__ import annotations
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from ..provenance.hashing import sha256_json


class PreprocSpec(BaseModel):
    """Per-model input preprocessing. Never shared implicitly between architectures."""
    model_config = ConfigDict(extra="forbid")
    resize: int
    crop: int
    mean: list[float]
    std: list[float]
    interpolation: str = "bilinear"
    channel_order: str = "RGB"
    source: str = ""        # where this came from (torchvision weights / Feather code path)


def build_transform(p: PreprocSpec):
    from torchvision import transforms as T
    interp = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}[p.interpolation]
    return T.Compose([T.Resize(p.resize, interpolation=interp), T.CenterCrop(p.crop), T.ToTensor(),
                      T.Normalize(p.mean, p.std)])


class _Stop(Exception):
    """Raised inside a forward hook to truncate the forward pass after the requested stage."""


# Observer-alias stage maps for torchvision backbones (module paths inside the loaded model).
TORCHVISION_STAGES: dict[str, dict[str, str]] = {
    "convnext_tiny": {"stage1": "features.1", "stage2": "features.3", "stage3": "features.5", "stage4": "features.7"},
    "alexnet": {"relu0": "features.1", "relu1": "features.4", "relu2": "features.7", "relu3": "features.9",
                "relu4": "features.11", "fc0_relu": "classifier.2", "fc1_relu": "classifier.5", "final": "classifier.6"},
    "resnet50": {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"},
}


@dataclass
class Extractor:
    """Observer-compatible object consumed by `extraction.engine.extract`."""
    model: Any
    stage: str
    module_path: str
    transform: Any
    meta: dict
    device: str = "cpu"
    _store: dict = field(default_factory=dict)

    def spatial(self, batch):
        import torch
        mod = dict(self.model.named_modules())[self.module_path]

        def hook(_m, _i, out):
            self._store["out"] = out.detach()
            raise _Stop()
        h = mod.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                try:
                    self.model(batch.to(self.device))
                except _Stop:
                    pass
        finally:
            h.remove()
        return self._store.pop("out")

    @staticmethod
    def pool(x):
        x = x.float()
        return x.mean(dim=(2, 3)) if x.ndim == 4 else x.reshape(len(x), -1)   # vectors (fc stages) are kept as-is


class ModelAdapter(ABC):
    model_id: str
    family: str
    role_hint: str = "unspecified"

    @abstractmethod
    def load(self, device: str = "cpu"): ...
    @abstractmethod
    def preprocessing(self) -> PreprocSpec: ...
    @abstractmethod
    def stage_modules(self) -> dict[str, str]: ...
    @abstractmethod
    def checkpoint_hash(self, model) -> str: ...

    def validate_stages(self, model) -> dict[str, bool]:
        names = {n for n, _ in model.named_modules()}
        return {s: p in names for s, p in self.stage_modules().items()}

    def probe_shapes(self, model, stage_filter=None, device="cpu") -> dict[str, list[int]]:
        """Run a dummy batch through every stage to record real tensor shapes (nothing is assumed)."""
        import torch
        pre = self.preprocessing()
        out = {}
        for s, path in self.stage_modules().items():
            if stage_filter and s not in stage_filter:
                continue
            ex = Extractor(model, s, path, None, {}, device)
            out[s] = list(ex.spatial(torch.zeros(1, 3, pre.crop, pre.crop)).shape[1:])
        return out

    def extractor(self, stage: str, device: str = "cpu", pooling: str = "global_average_pool") -> Extractor:
        model = self.load(device)
        if stage not in self.stage_modules():
            raise KeyError(f"{self.model_id}: unknown stage '{stage}'. Known: {sorted(self.stage_modules())}")
        ok = self.validate_stages(model)
        if not ok[stage]:
            raise ValueError(f"{self.model_id}: stage '{stage}' -> '{self.stage_modules()[stage]}' not in loaded model")
        pre = self.preprocessing()
        shape = self.probe_shapes(model, [stage], device)[stage]
        meta = {"model_id": self.model_id, "family": self.family, "layer": stage,
                "module_path": self.stage_modules()[stage], "weights_sha256": self.checkpoint_hash(model),
                "preprocessing": pre.model_dump(), "preprocessing_sha256": sha256_json(pre.model_dump()),
                "pooling": pooling, "output_shape": shape}
        return Extractor(model, stage, self.stage_modules()[stage], build_transform(pre), meta, device)


class TorchvisionAdapter(ModelAdapter):
    """ImageNet-pretrained torchvision backbone (observer / recognition baseline roles)."""
    def __init__(self, architecture: str, weights: str = "DEFAULT", family: str | None = None):
        if architecture not in TORCHVISION_STAGES:
            raise KeyError(f"unsupported torchvision architecture {architecture}")
        self.arch, self.weights_name, self.family = architecture, weights, family or architecture
        self.model_id = f"torchvision:{architecture}:{weights}"

    def _weights(self):
        import torchvision.models as M
        enum = {"convnext_tiny": M.ConvNeXt_Tiny_Weights, "alexnet": M.AlexNet_Weights,
                "resnet50": M.ResNet50_Weights}[self.arch]
        return getattr(enum, self.weights_name)

    def load(self, device="cpu"):
        import torchvision.models as M
        m = getattr(M, self.arch)(weights=self._weights()).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def preprocessing(self) -> PreprocSpec:
        t = self._weights().transforms()
        return PreprocSpec(resize=t.resize_size[0], crop=t.crop_size[0], mean=list(t.mean), std=list(t.std),
                           interpolation="bicubic" if "BICUBIC" in str(t.interpolation).upper() else "bilinear",
                           source=f"torchvision {self.arch} {self.weights_name}")

    def stage_modules(self):
        return dict(TORCHVISION_STAGES[self.arch])

    def checkpoint_hash(self, model):
        from ..provenance.hashing import sha256_state_dict
        return sha256_state_dict(model.state_dict())


class FeatherAdapter(ModelAdapter):
    """Generator-native adapter for a Feather network (ResNet-50, AlexNet, IPCL AlexNetGN, ...).

    NOT EXECUTED against real Feather checkpoints in this repository's authoring environment: the loader
    contract (build script + function + stage->module map + preprocessing) is configuration, and must be
    verified against the pinned Feather commit. Feather networks usually embed input normalisation inside the
    model; only set mean=0/std=1 after confirming that in their code. Never reuse another model's PreprocSpec.
    """
    def __init__(self, model_id: str, family: str, repo_dir: str | Path, build_script: str,
                 stage_to_module: dict[str, str], preprocessing: PreprocSpec, build_fn: str = "build_net",
                 checkpoint_path: str | None = None, build_kwargs: dict | None = None):
        self.model_id, self.family = model_id, family
        self.repo, self.script, self.fn = Path(repo_dir), build_script, build_fn
        self._stages, self._pre = dict(stage_to_module), preprocessing
        self.ckpt, self.kw = checkpoint_path, build_kwargs or {}

    def load(self, device="cpu"):
        path = self.repo / self.script
        if not path.exists():
            raise FileNotFoundError(f"Feather build script not found: {path} (audit the pinned checkout)")
        spec = importlib.util.spec_from_file_location(f"_feather_{abs(hash(str(path)))}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)           # executes trusted pinned Feather code
        built = getattr(mod, self.fn)(**self.kw)
        model = built[0] if isinstance(built, tuple) else built
        if self.ckpt:
            from ..representations.sae import safe_load_checkpoint
            state = safe_load_checkpoint(self.ckpt)
            model.load_state_dict(state.get("state_dict", state), strict=True)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def preprocessing(self):
        return self._pre

    def stage_modules(self):
        return dict(self._stages)

    def checkpoint_hash(self, model):
        from ..provenance.hashing import sha256_state_dict
        return sha256_state_dict(model.state_dict())


def adapter_from_spec(spec: dict, feather_dir: str | None = None) -> ModelAdapter:
    """Build an adapter from a YAML observer/generator spec (see configs/observers/*.yaml)."""
    kind = spec["kind"]
    if kind == "torchvision":
        return TorchvisionAdapter(spec["architecture"], spec.get("weights", "DEFAULT"), spec.get("family"))
    if kind == "feather":
        return FeatherAdapter(spec["model_id"], spec["family"], feather_dir or spec["repo_dir"], spec["build_script"],
                              spec["stage_to_module"], PreprocSpec(**spec["preprocessing"]),
                              spec.get("build_fn", "build_net"), spec.get("checkpoint_path"), spec.get("build_kwargs"))
    raise ValueError(f"unknown adapter kind '{kind}'")
