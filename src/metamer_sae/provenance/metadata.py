"""Run metadata (README scope §12) with a guarded status lifecycle."""
from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, ConfigDict

Status = Literal["planned", "available", "validated", "executed", "reviewed", "failed"]
_NEXT = {"planned": {"available", "failed"}, "available": {"validated", "failed"},
         "validated": {"executed", "failed"}, "executed": {"reviewed", "failed"}, "reviewed": set(), "failed": set()}
TRACKS = ("pilot_parity", "common_observer", "generator_native", "transfer", "human_behavior", "brain_extension")


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    experiment_track: Literal[TRACKS]  # type: ignore[valid-type]
    generator_family: str | None = None
    generator_model_id: str | None = None
    training_objective: str | None = None
    perturbation_norm: str | None = None
    perturbation_epsilon: float | None = None
    generator_checkpoint_hash: str | None = None
    generator_stage: str | None = None
    generator_module_path: str | None = None
    observer_model_id: str | None = None
    observer_stage: str | None = None
    recognition_readout: str | None = None
    natural_dataset_id: str | None = None
    natural_split_hash: str | None = None
    stimulus_manifest_hash: str | None = None
    source_id: str | None = None          # per-row in tables; null at run level
    condition_id: str | None = None
    split: str | None = None
    representation_id: str | None = None
    representation_training_data: str | None = None
    sae_seed: int | None = None
    software_environment: dict | None = None
    code_commit: str | None = None
    status: Status = "planned"

    def advance(self, new: Status) -> "RunMetadata":
        if new not in _NEXT[self.status]:
            raise ValueError(f"illegal status transition {self.status} -> {new}")
        return self.model_copy(update={"status": new})
