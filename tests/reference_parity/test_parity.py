"""Needs the archived pilot artifacts under outputs/ (see README). Skips cleanly otherwise."""
from pathlib import Path

import pytest

CFG = "configs/experiments/pilot_resnet50_convnext_stage3.yaml"
pytestmark = pytest.mark.parity


def test_pilot_parity(tmp_path):
    from metamer_sae.config import load_config
    cfg = load_config(CFG)
    if not all(Path(p).exists() for p in [cfg.pca.artifact, cfg.sae.training_mean,
                                          cfg.sae.primary_checkpoint, cfg.sae.replicate_checkpoint]):
        pytest.skip("pilot artifacts not present")
    pytest.importorskip("overcomplete")
    from metamer_sae.evaluation.parity import check_parity
    from metamer_sae.pipeline import build_representations, open_pools, run_evaluate
    _, pair, con = run_evaluate(cfg, str(tmp_path))
    tab = check_parity(cfg, pair, con, open_pools(cfg).split_ids("primary_evaluation"), build_representations(cfg))
    assert tab["pass"].all(), tab[~tab["pass"]].to_string()
