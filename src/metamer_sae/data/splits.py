"""Fixed pilot split (immutable) and source-level split generation for full-scale runs."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..provenance.hashing import sha256_json

PILOT_SEED = 42
PILOT_SPLITS: dict[str, list[int]] = {
    "development": [0, 260, 308, 84, 199, 119, 97, 27],
    "primary_evaluation": [116, 101, 133, 125, 232, 362, 227, 326, 398, 358, 54, 78, 288, 45, 295, 249,
                           329, 197, 108, 393, 234, 59, 205, 245, 102, 343, 52, 241, 286, 81, 383, 361],
    "reserve": [46, 255, 186, 169, 395, 134, 194, 62],
}
PILOT_EXCLUDED_NO_STANDARD = (207, 240, 336)
SPLIT_ORDER = ("development", "primary_evaluation", "reserve")


def assert_disjoint(splits: dict[str, list[int]]) -> None:
    seen: dict[int, str] = {}
    for name, ids in splits.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids inside split {name}")
        for i in ids:
            if i in seen:
                raise ValueError(f"source {i} in both {seen[i]} and {name}")
            seen[i] = name


def pilot_ledger() -> pd.DataFrame:
    """Ledger of the fixed 8/32/8 split. activation_index = position in dev+primary+reserve order.

    NOTE: when reusing the notebook-era activation pools, load the notebook's own
    `fixed_triplet_split_ledger.csv` instead (see `load_ledger`) so row indices match.
    """
    assert_disjoint(PILOT_SPLITS)
    rows, k = [], 0
    for split in SPLIT_ORDER:
        for sid in PILOT_SPLITS[split]:
            rows.append({"stimulus_id": sid, "split": split, "activation_index": k})
            k += 1
    df = pd.DataFrame(rows)
    assert df.split.value_counts().to_dict() == {"primary_evaluation": 32, "development": 8, "reserve": 8}
    return df


def load_ledger(path: str | None = None) -> pd.DataFrame:
    if path:
        from pathlib import Path
        if Path(path).exists():
            df = pd.read_csv(path)
            need = {"stimulus_id", "split", "activation_index"}
            if not need <= set(df.columns):
                raise ValueError(f"ledger missing columns {need - set(df.columns)}")
            got = {s: sorted(g.stimulus_id.tolist()) for s, g in df.groupby("split")}
            for s, ids in PILOT_SPLITS.items():
                if sorted(ids) != got.get(s):
                    raise ValueError(f"ledger split '{s}' differs from the fixed pilot split")
            return df
    return pilot_ledger()


def make_source_level_split(source_ids, n_dev: int, n_primary: int, n_reserve: int, seed: int,
                            categories: dict[int, str] | None = None) -> pd.DataFrame:
    """Split by source identity (all conditions of a source stay together).

    Stratified round-robin by category only if reliable `categories` metadata is supplied.
    Decide counts and seed BEFORE looking at any test metric; store the returned frame + hash.
    """
    ids = sorted(set(int(i) for i in source_ids))
    need = n_dev + n_primary + n_reserve
    if need > len(ids):
        raise ValueError(f"requested {need} sources but only {len(ids)} eligible")
    rng = np.random.RandomState(seed)
    if categories:
        by_cat: dict[str, list[int]] = {}
        for i in ids:
            by_cat.setdefault(categories[i], []).append(i)
        for v in by_cat.values():
            rng.shuffle(v)
        order = []
        while any(by_cat.values()):
            for c in sorted(by_cat):
                if by_cat[c]:
                    order.append(by_cat[c].pop())
        chosen = order[:need]
    else:
        chosen = list(rng.permutation(ids)[:need])
    parts = {"development": chosen[:n_dev], "primary_evaluation": chosen[n_dev:n_dev + n_primary],
             "reserve": chosen[n_dev + n_primary:need]}
    assert_disjoint(parts)
    df = pd.DataFrame([{"stimulus_id": int(i), "split": s, "activation_index": k}
                       for k, (s, i) in enumerate((s, i) for s in SPLIT_ORDER for i in parts[s])])
    df.attrs["split_sha256"] = split_hash(df)
    df.attrs["stratified"] = bool(categories)
    return df


def split_hash(df: pd.DataFrame) -> str:
    return sha256_json(sorted(zip(df.stimulus_id.astype(int), df.split)))


def matrix_split(pairs: pd.DataFrame, n_dev: int, n_primary: int, n_reserve: int, seed: int,
                 categories: dict[int, str] | None = None, require_conditions: list[str] | None = None) -> pd.DataFrame:
    """Source-level grouped split over the matrix. Eligible sources = those complete in `require_conditions`
    (default: any complete pair). All conditions of a source inherit its split; the pilot split is untouched."""
    ok = pairs[pairs.pair_complete]
    if require_conditions:
        ok = ok[ok.condition_id.isin(require_conditions)]
        cnt = ok.groupby("source_image_id").condition_id.nunique()
        elig = cnt[cnt == len(require_conditions)].index
    else:
        elig = ok.source_image_id.unique()
    return make_source_level_split(elig, n_dev, n_primary, n_reserve, seed, categories)
