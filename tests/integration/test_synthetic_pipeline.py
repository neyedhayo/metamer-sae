import numpy as np

from metamer_sae.evaluation.paired import evaluate_pairs
from metamer_sae.representations.base import RawRepresentation
from metamer_sae.representations.pca import PCABasis
from metamer_sae.statistics.paired import contrast_family


def test_end_to_end_synthetic(rng):
    basis = PCABasis.fit(rng.normal(size=(500, 12)), 12)
    reps = {"raw_384": RawRepresentation(), "pca_16": basis.view(4), "pca_64": basis.view(8)}
    n, src = 32, rng.normal(size=(32, 12))
    acts = {"source": src, "standard": src + rng.normal(0, 1.5, src.shape), "adversarial": src + rng.normal(0, .5, src.shape)}
    stim, pair = evaluate_pairs("primary_evaluation", list(range(n)), acts, reps)
    assert len(pair) == n * 2 * 3 and len(stim) == n * 3 * 3
    con = contrast_family(pair, list(range(n)), ["raw_384", "pca_16", "pca_64"], n_boot=100)
    assert len(con) == 9 and (con.mean_paired_difference[con.contrast.str.contains("L2 distance")] > 0).all()
