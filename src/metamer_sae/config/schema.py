"""Strict, validated run configuration. Unknown keys are rejected, never ignored."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentCfg(_Strict):
    id: str
    description: str = ""


class ConditionCfg(_Strict):
    id: str
    role: Literal["standard", "adversarial", "control"]
    experiment_key: str
    model_key: str
    layer: str


class FeatherCfg(_Strict):
    repository: str
    commit: str = Field(min_length=40, max_length=40)
    local_dir: str = "data/external/model_metamers_pytorch"
    candidate_ids: list[int] = Field(default_factory=lambda: list(range(400)))
    conditions: list[ConditionCfg]


class StimuliCfg(_Strict):
    image_dir: str = "data/images"
    manifest: str = "data/manifests/stimulus_manifest.csv"
    split_manifest: str = "data/manifests/stimulus_allocation.csv"
    split_policy: Literal["pilot_fixed", "source_level"] = "pilot_fixed"


class NaturalCfg(_Strict):
    dataset: str = "imagenette2-320"
    root: str = "data/external/imagenette2-320"
    train_samples: int
    validation_samples: int


class ObserverCfg(_Strict):
    architecture: Literal["convnext_tiny"] = "convnext_tiny"
    weights: str = "IMAGENET1K_V1"
    layer: Literal["stage1", "stage2", "stage3", "stage4"] = "stage3"
    frozen: bool = True
    input_resolution: int = 224
    resize_size: int = 236
    output_channels: int = 384
    spatial_shape: tuple[int, int, int] = (384, 14, 14)
    pooling: Literal["global_average_pool"] = "global_average_pool"


class PCACfg(_Strict):
    run_id: str
    artifact: str
    artifact_sha256: str | None = None
    fit_split: Literal["imagenette_train"] = "imagenette_train"
    fitted_components: int = 384
    primary_components: int = 16
    sensitivity_components: list[int] = [64]
    whiten: bool = False
    channel_standardize: bool = False


class SAECfg(_Strict):
    framework: Literal["overcomplete"] = "overcomplete"
    version: str = "0.3.0"
    variant: Literal["topk"] = "topk"
    input_dim: int = 384
    dictionary_width: int = 1536
    top_k: int = 64
    training_mean: str
    primary_seed: int = 123
    replicate_seed: int = 42
    primary_checkpoint: str
    replicate_checkpoint: str
    # training hyper-parameters (used by `train-sae` only)
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    grad_clip: float = 1.0
    early_stop_patience: int = 12
    min_delta: float = 1e-5
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 1e-5


class PathsCfg(_Strict):
    outputs: str = "outputs"
    natural_activations: str = "outputs/activations/natural_images"
    feather_activations: str = "outputs/activations/feather"
    # directory name for each stimulus role under feather_activations
    feather_role_dirs: dict[str, str] = {"source": "source",
                                         "standard": "standard_resnet50",
                                         "adversarial": "adversarial_resnet50"}


class EvaluationCfg(_Strict):
    primary_split: str = "primary_evaluation"
    development_split: str = "development"
    reserve_policy: Literal["never_load_activation_rows"] = "never_load_activation_rows"
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20261119
    primary_metrics: list[str] = ["distance_l2", "relative_l2", "cosine_similarity",
                                  "support_jaccard", "reconstruction_mse", "feature_change"]


class RunConfig(_Strict):
    experiment: ExperimentCfg
    feather: FeatherCfg
    stimuli: StimuliCfg = StimuliCfg()
    natural_images: NaturalCfg
    observer: ObserverCfg = ObserverCfg()
    pca: PCACfg
    sae: SAECfg
    paths: PathsCfg = PathsCfg()
    evaluation: EvaluationCfg = EvaluationCfg()

    @model_validator(mode="after")
    def _check(self):
        o, s = self.observer, self.sae
        if s.input_dim != o.output_channels:
            raise ValueError("sae.input_dim must equal observer.output_channels")
        if s.top_k > s.dictionary_width:
            raise ValueError("sae.top_k cannot exceed dictionary_width")
        if self.pca.primary_components > self.pca.fitted_components:
            raise ValueError("pca.primary_components exceeds fitted_components")
        roles = [c.role for c in self.feather.conditions]
        if len(set(c.id for c in self.feather.conditions)) != len(roles):
            raise ValueError("duplicate feather condition ids")
        return self


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
