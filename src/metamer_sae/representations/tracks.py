"""Experiment tracks and the dictionary-sharing rules (README scope §6.3). Pure bookkeeping + guards.

Track A  common_observer : ONE natural-trained dictionary per observer x stage x preprocessing, reused for
                           source + every generator condition being compared.
Track B  generator_native: ONE dictionary per generator checkpoint/activation space x stage, reused for the
                           source and its paired metamers.
Track C  transfer        : a dictionary applied to a different activation space; always labelled, always
                           reports distribution mismatch.
A dictionary is never trained per image label (source / standard / adversarial) and never on Feather images.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..provenance.hashing import sha256_json


class Track(str, Enum):
    pilot_parity = "pilot_parity"
    common_observer = "common_observer"
    generator_native = "generator_native"
    transfer = "transfer"
    human_behavior = "human_behavior"
    brain_extension = "brain_extension"


@dataclass(frozen=True)
class ActivationSpace:
    """Identity of a coordinate system. Two stages with equal names/indices are NOT the same space."""
    model_id: str
    checkpoint_hash: str
    stage: str
    module_path: str
    preprocessing_sha256: str
    pooling: str
    shape: tuple[int, ...]

    @property
    def space_id(self) -> str:
        return sha256_json(asdict(self))[:16]

    @classmethod
    def from_meta(cls, meta: dict) -> "ActivationSpace":
        return cls(meta["model_id"], meta["weights_sha256"], meta["layer"], meta["module_path"],
                   meta["preprocessing_sha256"], meta["pooling"], tuple(meta["output_shape"]))


@dataclass(frozen=True)
class RepresentationRecord:
    representation_id: str            # PCA / SAE run id
    kind: str                         # pca | sae
    space_id: str                     # activation space it was fitted in
    natural_dataset_id: str
    training_data_sha256: str         # hash of the natural activation corpus (never Feather)
    seed: int | None = None
    track: Track = Track.common_observer


def assert_natural_only(corpus_metadata: pd.DataFrame, feather_image_sha256: set[str],
                        natural_dataset_id: str, expected_dataset_id: str) -> None:
    """Fit guard. `corpus_metadata` is the extraction metadata.csv of the fitting corpus (needs image_sha256)."""
    if natural_dataset_id != expected_dataset_id:
        raise ValueError(f"fitting corpus '{natural_dataset_id}' != declared natural dataset '{expected_dataset_id}'")
    if "image_sha256" not in corpus_metadata:
        raise ValueError("corpus metadata lacks image_sha256; cannot prove it excludes Feather images")
    overlap = set(corpus_metadata.image_sha256) & set(feather_image_sha256)
    if overlap:
        raise PermissionError(f"{len(overlap)} Feather source/metamer images are in the fitting corpus")


def validate_common_observer(records: dict[str, RepresentationRecord], observer_space: ActivationSpace) -> str:
    """records: {condition_id|'source': record}. All must be the SAME frozen representation in the observer space."""
    ids = {r.representation_id for r in records.values()}
    if len(ids) != 1:
        raise ValueError(f"common_observer requires one shared representation; got {sorted(ids)}")
    for name, r in records.items():
        if r.space_id != observer_space.space_id:
            raise ValueError(f"{name}: representation fitted in space {r.space_id}, observer space is {observer_space.space_id}")
    return ids.pop()


def validate_generator_native(record: RepresentationRecord, generator_space: ActivationSpace) -> None:
    if record.space_id != generator_space.space_id:
        raise ValueError("generator_native dictionary must be fitted in the generator's own checkpoint/stage space; "
                         "use an explicitly labelled transfer experiment otherwise")


def distribution_mismatch(natural: np.ndarray, target: np.ndarray) -> dict:
    """Transfer-track diagnostic: how far target activations sit from the natural fitting distribution."""
    mu, sd = natural.mean(0), natural.std(0) + 1e-8
    z = (target.mean(0) - mu) / sd
    return {"mean_abs_standardised_shift": float(np.abs(z).mean()),
            "norm_ratio_target_over_natural": float(np.linalg.norm(target, axis=1).mean() / np.linalg.norm(natural, axis=1).mean()),
            "dim_natural": int(natural.shape[1]), "dim_target": int(target.shape[1])}


def require_matching_report(report: dict | None) -> dict:
    """Cross-seed/cross-dictionary feature-index comparison is only legal with a matching quality report."""
    if not report or "mean_matched_cosine" not in report:
        raise ValueError("compare SAE feature indices only via match_dictionaries(); report matching quality")
    return report
