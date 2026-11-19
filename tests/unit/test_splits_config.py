import pytest
from pydantic import ValidationError

from metamer_sae.config import load_config
from metamer_sae.data.splits import PILOT_SPLITS, assert_disjoint, make_source_level_split, pilot_ledger, split_hash

CFG = "configs/experiments/pilot_resnet50_convnext_stage3.yaml"


def test_pilot_split_counts_and_disjoint():
    assert [len(PILOT_SPLITS[k]) for k in ("development", "primary_evaluation", "reserve")] == [8, 32, 8]
    assert_disjoint(PILOT_SPLITS)
    assert 207 not in sum(PILOT_SPLITS.values(), [])
    led = pilot_ledger()
    assert len(led) == 48 and led.activation_index.is_unique


def test_overlap_detected():
    with pytest.raises(ValueError):
        assert_disjoint({"a": [1, 2], "b": [2, 3]})


def test_source_level_split_deterministic_and_hashed():
    a = make_source_level_split(range(100), 5, 20, 5, seed=1)
    b = make_source_level_split(range(100), 5, 20, 5, seed=1)
    assert a.equals(b) and split_hash(a) == split_hash(b) and a.stimulus_id.is_unique


def test_pilot_config_valid_and_strict(tmp_path):
    cfg = load_config(CFG)
    assert cfg.sae.top_k == 64 and cfg.observer.layer == "stage3" and len(cfg.feather.candidate_ids) == 400
    bad = open(CFG).read().replace("  top_k: 64", "  top_k: 64\n  topk_typo: 1")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValidationError):
        load_config(p)
