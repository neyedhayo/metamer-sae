"""Schema for the expanded multi-family study. Separate from the pilot RunConfig so the pilot config is never
altered. Unknown keys are errors. Layer subsets must be declared with a justification BEFORE confirmatory runs."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageSelection(_S):
    mode: Literal["all_eligible", "declared_subset"]
    stages: list[str] = Field(default_factory=list)
    justification: str = ""            # mandatory for declared_subset
    declared_before_confirmatory: bool = False


class ConditionSel(_S):
    condition_id: str                  # key into configs/models/registry.yaml
    stages: StageSelection


class NaturalDataset(_S):
    dataset_id: str = "imagenette2-320"
    version: str = "v1"
    root: str = "data/external/imagenette2-320"
    train_samples: int = 9469
    validation_samples: int = 3925
    license_note: str = "Imagenette (fast.ai), subset of ImageNet; see dataset page"


class SplitCfg(_S):
    policy: Literal["pilot_fixed", "source_level"] = "source_level"
    n_development: int = 8
    n_primary: int = 32
    n_reserve: int = 8
    seed: int = 20261120
    stratify_by_category: bool = False  # only if reliable category metadata exists


class SAESweep(_S):
    variant: Literal["topk"] = "topk"
    expansions: list[int] = [4]
    top_k: list[int] = [64]
    seeds: list[int] = [123, 42]
    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200
    selection_criterion: Literal["natural_validation_mse"] = "natural_validation_mse"


class PCACfg2(_S):
    fitted_components: int | None = None   # None -> full dimensionality of the space
    primary_components: int = 16
    sensitivity_components: list[int] = [64]


class ComparisonFamilyCfg(_S):
    name: str
    confirmatory: bool = False
    metrics: list[str]
    contrasts: list[tuple[str, str]]       # (condition_a, condition_b): metric_a - metric_b, paired by source


class MatrixConfig(_S):
    experiment_id: str
    track: Literal["common_observer", "generator_native", "transfer"]
    description: str = ""
    registry: str = "configs/models/registry.yaml"
    observer_panel: str = "configs/observers/panel_default.yaml"
    observer_ids: list[str] = Field(default_factory=list)   # Track A only
    feather_commit: str = Field(min_length=40, max_length=40)
    feather_repository: str = "https://github.com/jenellefeather/model_metamers_pytorch.git"
    feather_dir: str = "data/external/model_metamers_pytorch"
    candidate_ids: list[int] = Field(default_factory=lambda: list(range(400)))
    conditions: list[ConditionSel]
    natural: NaturalDataset = NaturalDataset()
    split: SplitCfg = SplitCfg()
    pca: PCACfg2 = PCACfg2()
    sae: SAESweep = SAESweep()
    families: list[ComparisonFamilyCfg] = Field(default_factory=list)
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20261120
    manifests_dir: str = "data/manifests"
    outputs: str = "outputs"
    human_behavior_enabled: bool = False   # human stage is gated; see human/behavior.py

    @model_validator(mode="after")
    def _chk(self):
        for c in self.conditions:
            if c.stages.mode == "declared_subset" and not (c.stages.stages and c.stages.justification):
                raise ValueError(f"{c.condition_id}: declared_subset needs stages and a justification")
        ids = [c.condition_id for c in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate condition in matrix")
        if self.track == "common_observer" and not self.observer_ids:
            raise ValueError("common_observer track needs observer_ids")
        for f in self.families:
            for a, b in f.contrasts:
                if a not in ids or b not in ids:
                    raise ValueError(f"family {f.name}: contrast references unselected condition")
        return self


def load_matrix(path: str | Path) -> MatrixConfig:
    with open(path) as f:
        return MatrixConfig.model_validate(yaml.safe_load(f))
