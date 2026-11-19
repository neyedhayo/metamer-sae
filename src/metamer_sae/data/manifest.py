"""Stimulus and pair manifests keyed by stable IDs (never by file order)."""
from __future__ import annotations
import pandas as pd

STIMULUS_COLUMNS = [
    "stimulus_id", "source_image_id", "category_id", "category_name", "feather_commit",
    "feather_experiment_key", "generator_condition", "generator_model_key", "generator_layer_key",
    "generator_layer_index", "stimulus_role", "image_path", "image_sha256", "image_width",
    "image_height", "image_format", "availability_status", "retrieval_timestamp", "retrieval_source",
]
PAIR_COLUMNS = ["pair_id", "source_image_id", "experiment_family", "generator_layer",
                "source_stimulus_id", "standard_stimulus_id", "adversarial_stimulus_id",
                "control_stimulus_id", "split", "pair_complete", "exclusion_reason"]


def stimulus_id(role: str, condition: str, source_image_id: int) -> str:
    return f"{role}:{condition}:{int(source_image_id):04d}" if role != "source" else f"source:{int(source_image_id):04d}"


def build_stimulus_manifest(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records).reindex(columns=STIMULUS_COLUMNS)
    if df.stimulus_id.duplicated().any():
        raise ValueError("duplicate stimulus_id in manifest")
    return df


def build_pair_manifest(stim: pd.DataFrame, split_df: pd.DataFrame, family: str, layer: str,
                        standard_cond: str, adversarial_cond: str, control_cond: str | None = None) -> pd.DataFrame:
    """One row per source. Incomplete groups are KEPT with pair_complete=False + reason."""
    ok = stim[stim.availability_status == "ok"]
    have = {(r.stimulus_role, r.generator_condition, int(r.source_image_id)): r.stimulus_id
            for r in ok.itertuples()}
    split_of = dict(zip(split_df.stimulus_id.astype(int), split_df.split))
    rows = []
    for sid in sorted(set(int(i) for i in stim.source_image_id)):
        src = have.get(("source", "source", sid)) or have.get(("source", "", sid))
        std = have.get(("standard", standard_cond, sid))
        adv = have.get(("adversarial", adversarial_cond, sid))
        ctl = have.get(("control", control_cond, sid)) if control_cond else None
        missing = [n for n, v in (("source", src), ("standard", std), ("adversarial", adv)) if v is None]
        if control_cond and ctl is None:
            missing.append("control")
        rows.append({"pair_id": f"{family}:{layer}:{sid:04d}", "source_image_id": sid,
                     "experiment_family": family, "generator_layer": layer, "source_stimulus_id": src,
                     "standard_stimulus_id": std, "adversarial_stimulus_id": adv,
                     "control_stimulus_id": ctl, "split": split_of.get(sid), "pair_complete": not missing,
                     "exclusion_reason": ";".join(f"missing_{m}" for m in missing) or None})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def availability_report(pairs: pd.DataFrame) -> pd.DataFrame:
    return (pairs.groupby(["experiment_family", "generator_layer"])
            .agg(n_sources=("source_image_id", "size"), n_complete=("pair_complete", "sum"))
            .assign(n_incomplete=lambda d: d.n_sources - d.n_complete).reset_index())
