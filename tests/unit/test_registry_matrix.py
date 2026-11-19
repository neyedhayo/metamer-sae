import pandas as pd
import pytest
from pydantic import ValidationError

from metamer_sae.config.matrix import MatrixConfig, load_matrix
from metamer_sae.registry import adversarial_ipcl_audit, load_registry, resolve_registry
from metamer_sae.registry.schema import ModelCondition


def test_registry_loads_and_pilot_pair_verified():
    r = load_registry()
    assert r.get("resnet50_standard").verified and r.get("resnet50_l2_3_robust").perturbation_epsilon == 3.0
    assert not r.get("alexnet_l2_3_robust").verified            # nothing unverified is silently trusted
    assert {c.family for c in r.conditions} == {"resnet50", "alexnet", "ipcl_alexnetgn"}


def test_ipcl_not_adversarial_and_needs_readout():
    r = load_registry()
    assert r.get("ipcl_selfsup").training_objective.value == "self_supervised_ipcl"
    assert r.get("ipcl_selfsup").perturbation_norm is None and r.get("ipcl_selfsup").readout == "linear_readout"
    with pytest.raises(ValidationError):
        ModelCondition(condition_id="x", family="f", training_objective="self_supervised_ipcl")  # native head
    with pytest.raises(ValidationError):
        ModelCondition(condition_id="x", family="f", training_objective="self_supervised_ipcl",
                       readout="linear_readout", perturbation_norm="L2", perturbation_epsilon=3)


def test_resolution_reports_gaps_and_ambiguity():
    r = load_registry()
    keys = ["resnet50", "resnet50_l2_3_robust", "alexnet", "alexnet_l2_3_robust", "alexnet_random_l2_3_robust"]
    t = resolve_registry(r, keys).set_index("condition_id")
    assert t.loc["resnet50_standard", "status"] == "resolved_pinned"
    assert t.loc["alexnet_standard", "resolved_model_key"] == "alexnet"
    assert t.loc["alexnet_l2_3_robust", "resolved_model_key"] == "alexnet_l2_3_robust"   # random excluded
    assert t.loc["alexnet_random_perturbation", "resolved_model_key"] == "alexnet_random_l2_3_robust"
    assert t.loc["ipcl_selfsup", "status"] == "unresolved"
    amb = resolve_registry(r, keys + ["alexnet_l2_3_robust_v2"]).set_index("condition_id")
    assert amb.loc["alexnet_l2_3_robust", "status"] == "ambiguous" and pd.isna(amb.loc["alexnet_l2_3_robust", "resolved_model_key"])


def test_adversarial_ipcl_gate():
    none = adversarial_ipcl_audit(["alexnet", "ipcl_alexnet_gn", "resnet50_l2_3_robust"])
    assert none["status"] == "unavailable_in_feather_metadata" and none["candidate_keys"] == []
    assert "relabel self-supervised IPCL as adversarial" in none["must_not"]
    hit = adversarial_ipcl_audit(["ipcl_alexnet_gn_l2_3_robust"])
    assert hit["status"] == "candidate_found_requires_manual_verification"


@pytest.mark.parametrize("f", ["alexnet_core_generator_native", "ipcl_core_generator_native", "common_observer_panel_core"])
def test_expanded_configs_valid(f):
    c = load_matrix(f"configs/experiments/{f}.yaml")
    reg = load_registry(c.registry)
    for s in c.conditions:
        reg.get(s.condition_id)


def test_matrix_validation_rules():
    base = dict(experiment_id="e", track="generator_native", feather_commit="a" * 40)
    with pytest.raises(ValidationError):   # subset without justification
        MatrixConfig(**base, conditions=[{"condition_id": "c", "stages": {"mode": "declared_subset", "stages": ["s"]}}])
    with pytest.raises(ValidationError):   # common observer without observers
        MatrixConfig(**{**base, "track": "common_observer"}, conditions=[{"condition_id": "c", "stages": {"mode": "all_eligible"}}])
    with pytest.raises(ValidationError):   # unknown key
        MatrixConfig(**base, conditions=[], bogus=1)
