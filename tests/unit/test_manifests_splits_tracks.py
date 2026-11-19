import numpy as np
import pandas as pd
import pytest

from metamer_sae.data import matrix_manifests as mm
from metamer_sae.data.splits import matrix_split, PILOT_SPLITS, pilot_ledger
from metamer_sae.provenance.metadata import RunMetadata
from metamer_sae.representations.tracks import (ActivationSpace, RepresentationRecord, Track, assert_natural_only,
                                                distribution_mismatch, require_matching_report,
                                                validate_common_observer, validate_generator_native)


def _stim():
    rows = []
    for i in range(10):
        rows.append(dict(stimulus_id=f"source:{i:04d}", source_image_id=i, stimulus_role="source", generator_condition="source", availability_status="ok"))
        for c in ("A|e|s1", "B|e|s1"):
            if c == "B|e|s1" and i in (3, 4):
                continue                                           # missing B metamers
            rows.append(dict(stimulus_id=f"m:{c}:{i}", source_image_id=i, stimulus_role="metamer", generator_condition=c, availability_status="ok"))
    return pd.DataFrame(rows)


def _cond():
    return pd.DataFrame({"condition_id": ["A|e|s1", "B|e|s1"], "family": "f", "training_objective": ["x", "y"], "stage": "s1"})


def test_missing_conditions_kept_not_dropped():
    p = mm.metamer_pair_manifest(_stim(), _cond(), None)
    assert len(p) == 20 and (~p.pair_complete).sum() == 2
    assert set(p[~p.pair_complete].source_image_id) == {3, 4}
    keep, excl = mm.complete_case_subset(p, ["A|e|s1", "B|e|s1"])
    assert keep == [0, 1, 2, 5, 6, 7, 8, 9] and set(excl.source_image_id) == {3, 4}


def test_grouped_split_inherited_across_conditions():
    p = mm.metamer_pair_manifest(_stim(), _cond(), None)
    s = matrix_split(p, 1, 3, 1, seed=1, require_conditions=["A|e|s1", "B|e|s1"])
    assert not set(s.stimulus_id) & {3, 4}
    p["split"] = p.source_image_id.map(dict(zip(s.stimulus_id, s.split)))
    mm.assert_split_consistency(p)
    p.loc[p.condition_id == "B|e|s1", "split"] = "reserve"; p.loc[p.condition_id == "A|e|s1", "split"] = "development"
    with pytest.raises(ValueError):
        mm.assert_split_consistency(p)


def test_pilot_split_untouched():
    assert [len(PILOT_SPLITS[k]) for k in PILOT_SPLITS] == [8, 32, 8] and len(pilot_ledger()) == 48


def test_condition_id_keeps_stage_verbatim():
    assert mm.make_condition_id("alexnet_standard", "experiment_4", "relu2") == "alexnet_standard|experiment_4|relu2"


def _sp(stage, h="h"):
    return ActivationSpace("m", h, stage, f"mod.{stage}", "p", "gap", (4,))


def test_tracks_guards():
    obs, other = _sp("s3"), _sp("s3", "different-ckpt")
    assert obs.space_id != other.space_id                       # same stage name, different checkpoint -> different space
    r = RepresentationRecord("sae1", "sae", obs.space_id, "ds", "h")
    assert validate_common_observer({"source": r, "A": r, "B": r}, obs) == "sae1"
    r2 = RepresentationRecord("sae2", "sae", obs.space_id, "ds", "h")
    with pytest.raises(ValueError):                               # separate dictionaries per condition forbidden
        validate_common_observer({"source": r, "A": r2}, obs)
    with pytest.raises(ValueError):
        validate_generator_native(r, other)
    with pytest.raises(PermissionError):
        assert_natural_only(pd.DataFrame({"image_sha256": ["a", "b"]}), {"b"}, "ds", "ds")
    assert_natural_only(pd.DataFrame({"image_sha256": ["a"]}), {"b"}, "ds", "ds")
    with pytest.raises(ValueError):
        require_matching_report(None)
    d = distribution_mismatch(np.random.default_rng(0).normal(size=(200, 5)), np.random.default_rng(1).normal(2, 1, size=(50, 5)))
    assert d["mean_abs_standardised_shift"] > 1


def test_run_metadata_lifecycle():
    m = RunMetadata(run_id="r", experiment_track="generator_native")
    m = m.advance("available").advance("validated").advance("executed").advance("reviewed")
    with pytest.raises(ValueError):
        m.advance("planned")
    with pytest.raises(ValueError):
        RunMetadata(run_id="r", experiment_track="generator_native").advance("reviewed")
    with pytest.raises(Exception):
        RunMetadata(run_id="r", experiment_track="bogus")
