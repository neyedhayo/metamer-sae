import json

import numpy as np
import pandas as pd
import pytest

from metamer_sae.evaluation.image_baselines import smoothness
from metamer_sae.evaluation.matrix import cross_model_matrix, evaluate_conditions, natural_scale
from metamer_sae.evaluation.recognition import CategoryMap, accuracy_by_condition, softmax
from metamer_sae.human.behavior import condition_table, require_validated, run_condition_analysis, validate_linkage, validate_schema
from metamer_sae.provenance.run import Run, mark_reviewed, verify_run
from metamer_sae.representations.base import RawRepresentation
from metamer_sae.representations.pca import PCABasis
from metamer_sae.statistics.matrix import clustered_bootstrap, expand_contrasts, source_paired_contrasts


def _data(rng, n=24, d=10):
    src = rng.normal(size=(n, d))
    return src, {"A|e|s1": src + rng.normal(0, 1.5, src.shape), "B|e|s1": src + rng.normal(0, .5, src.shape)}


def test_evaluate_conditions_and_contrasts(rng):
    src, mets = _data(rng)
    nat = rng.normal(size=(300, 10)); basis = PCABasis.fit(nat, 10)
    reps = {"raw": RawRepresentation(), "pca_4": basis.view(4)}
    sc = {k: natural_scale(v, nat) for k, v in reps.items()}
    df = evaluate_conditions("primary_evaluation", list(range(24)), src, mets, reps, sc, {"A|e|s1": {"family": "f"}})
    assert len(df) == 24 * 2 * 2 and df.distance_l2_natural_scaled.notna().all()
    t = source_paired_contrasts(df, "fam", False, ["distance_l2"], [("A|e|s1", "B|e|s1"), ("A|e|s1", "MISSING")], n_boot=200)
    ok = t[t.status == "ok"]
    assert (ok.mean_paired_difference > 0).all() and (ok.n_sources == 24).all() and "p_holm_within_family" in t
    assert (t.status == "condition_missing").sum() == 2           # gaps are reported, not dropped
    lo, hi = clustered_bootstrap(df[df.representation == "raw"], "distance_l2", n_boot=200)
    assert lo < hi


def test_expand_contrasts_matches_stage_names_exactly():
    ids = ["a|e|relu2", "a|e|relu3", "b|e|relu2", "b|e|fc0"]
    assert expand_contrasts([("a", "b")], ids) == [("a|e|relu2", "b|e|relu2")]


def test_cross_model_matrix(rng):
    base = rng.uniform(1, 2, 30)
    nat = pd.DataFrame({"condition_id": "A", "source_image_id": range(30), "relative_l2_source_norm": base})
    obs = pd.DataFrame({"condition_id": "A", "source_image_id": range(30), "observer_id": "o", "representation": "pca",
                        "relative_l2_source_norm": base * 2 + rng.normal(0, .01, 30)})
    assert cross_model_matrix(nat, obs).spearman_native_vs_observer.iloc[0] > 0.9


def test_category_map_and_recognition():
    cm = CategoryMap({0: 3, 1: 3, 2: 7})
    p = softmax(np.array([[5., 5., 0.], [0., 0., 9.]]))
    assert cm.predict(p).tolist() == [3, 7]
    r = accuracy_by_condition(cm.predict(p), np.array([3, 7]), np.array([0, 1]), "c")
    assert r["accuracy"] == 1 and r["chance"] == 1 / 16 and "not_human" in r["kind"]
    with pytest.raises(ValueError):
        CategoryMap({0: 99})


def test_smoothness_orders_blur():
    rng = np.random.default_rng(0); noisy = rng.uniform(size=(64, 64, 3)).astype(np.float32)
    blur = noisy.copy()
    for _ in range(6):
        blur = (blur + np.roll(blur, 1, 0) + np.roll(blur, 1, 1)) / 3
    assert smoothness(noisy)["gradient_energy"] > smoothness(blur)["gradient_energy"]


def _beh(rng):
    rows = []
    for st, p in zip(["relu0", "relu1", "relu2", "relu3", "relu4", "fc0"], [.9, .8, .6, .4, .2, .1]):
        for pid in range(12):
            rows.append(dict(experiment_key="experiment_4", model_key="alexnet", stage=st, participant_id=pid,
                             accuracy=float(np.clip(p + rng.normal(0, .05), 0, 1)), granularity="participant_by_condition"))
    return pd.DataFrame(rows)


def test_human_gate_and_analysis(rng, tmp_path):
    beh = _beh(rng)
    cm = pd.DataFrame({"experiment_key": "experiment_4", "model_key": "alexnet", "stage": beh.stage.unique()})
    gate = tmp_path / "linkage.json"
    with pytest.raises(PermissionError):
        require_validated(gate)                                   # blocked before linkage
    bad = pd.concat([beh, beh.assign(stage="ghost")])
    assert not validate_linkage(bad, cm, gate)["linkage_validated"]
    with pytest.raises(PermissionError):
        require_validated(gate)
    assert validate_linkage(beh, cm, gate)["linkage_validated"]
    metrics = cm.assign(score=[.1, .2, .4, .6, .8, .9]); depth = cm.assign(depth_rank=range(6))
    tab = condition_table(beh, metrics, depth)
    out = run_condition_analysis(tab, ["score"], beh, gate, n_boot=200)
    assert out["split_half_ceiling"] > 0.5 and out["chance"] == 1 / 16
    with pytest.raises(ValueError):
        validate_schema(beh.drop(columns="participant_id").assign(granularity="trial"))


def test_run_immutability(tmp_path):
    run = Run(str(tmp_path), "t", {"a": 1}); run.path("x.txt").write_text("hi"); run.close({})
    assert verify_run(run.dir)["ok"]
    mark_reviewed(run.dir, "me")
    with pytest.raises(FileExistsError):
        mark_reviewed(run.dir, "me")
    (run.dir / "x.txt").write_text("changed")
    assert not verify_run(run.dir)["ok"]
