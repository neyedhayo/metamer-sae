"""Model-family / training-condition registry. Model identity lives in DATA (configs/models/registry.yaml),
never in `if model == ...` branches. Unverified entries are `verified: false` until audited against the
pinned Feather metadata (`metamer-sae resolve-registry`)."""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingObjective(str, Enum):
    supervised_standard = "supervised_standard"
    adversarial = "adversarial"
    random_perturbation = "random_perturbation"
    self_supervised_ipcl = "self_supervised_ipcl"
    supervised_matched_ipcl_arch = "supervised_matched_ipcl_arch"
    adversarial_ipcl = "adversarial_ipcl"      # availability must be audited; never relabel IPCL as this


Availability = Literal["pilot_validated", "expected_from_proposal", "audit_required", "unavailable", "available"]


class ModelCondition(_S):
    condition_id: str
    family: str
    training_objective: TrainingObjective
    perturbation_norm: Literal["L2", "Linf"] | None = None
    perturbation_epsilon: float | None = None
    experiment_key: str | None = None         # Feather visual experiment (None until audited)
    model_key: str | None = None              # exact Feather network key (None until audited)
    model_key_patterns: list[str] = Field(default_factory=list)   # candidates used ONLY by the audit
    exclude_patterns: list[str] = Field(default_factory=list)
    matched_comparator: str | None = None     # condition_id of the architecture/augmentation-matched model
    availability: Availability = "audit_required"
    verified: bool = False
    pilot: bool = False
    readout: Literal["native_head", "linear_readout", "not_applicable"] = "native_head"
    notes: str = ""

    @model_validator(mode="after")
    def _chk(self):
        pert = self.training_objective in (TrainingObjective.adversarial, TrainingObjective.adversarial_ipcl,
                                           TrainingObjective.random_perturbation)
        if pert and (self.perturbation_norm is None or self.perturbation_epsilon is None) and self.verified:
            raise ValueError(f"{self.condition_id}: verified perturbation-trained conditions need norm and epsilon")
        if not pert and (self.perturbation_norm or self.perturbation_epsilon is not None):
            raise ValueError(f"{self.condition_id}: perturbation metadata on a non-perturbation objective")
        if self.verified and not self.model_key:
            raise ValueError(f"{self.condition_id}: verified requires an exact model_key")
        if self.training_objective == TrainingObjective.self_supervised_ipcl and self.readout == "native_head":
            raise ValueError(f"{self.condition_id}: self-supervised IPCL has no native classifier head")
        return self


class Family(_S):
    family: str
    architecture: str
    notes: str = ""


class Registry(_S):
    version: str
    families: list[Family]
    conditions: list[ModelCondition]

    @model_validator(mode="after")
    def _chk(self):
        ids = [c.condition_id for c in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate condition_id in registry")
        fam = {f.family for f in self.families}
        for c in self.conditions:
            if c.family not in fam:
                raise ValueError(f"{c.condition_id}: unknown family {c.family}")
            if c.matched_comparator and c.matched_comparator not in ids:
                raise ValueError(f"{c.condition_id}: unknown matched_comparator")
        return self

    def get(self, condition_id: str) -> ModelCondition:
        for c in self.conditions:
            if c.condition_id == condition_id:
                return c
        raise KeyError(f"condition '{condition_id}' not in registry")

    def family_conditions(self, family: str) -> list[ModelCondition]:
        return [c for c in self.conditions if c.family == family]


def load_registry(path: str | Path = "configs/models/registry.yaml") -> Registry:
    with open(path) as f:
        return Registry.model_validate(yaml.safe_load(f))
