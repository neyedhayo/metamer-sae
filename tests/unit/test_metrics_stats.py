import numpy as np
import pandas as pd

from metamer_sae.evaluation.metrics import cosine_similarity, support_jaccard, vector_distance_metrics
from metamer_sae.statistics.paired import bootstrap_mean_ci, contrast_family, holm_bonferroni, partial_spearman


def test_metrics_basic():
    a, b = np.array([3.0, 4.0]), np.array([3.0, 0.0])
    m = vector_distance_metrics(a, b)
    assert np.isclose(m["distance_l2"], 4) and np.isclose(m["relative_l2_source_norm"], 0.8)
    assert np.isclose(cosine_similarity(a, a), 1)
    assert support_jaccard([1, 0, 2], [0, 0, 3]) == 1 / 2


def test_holm():
    adj = holm_bonferroni(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_bootstrap_ci_contains_mean(rng):
    x = rng.normal(1, 1, 50)
    lo, hi = bootstrap_mean_ci(x, 2000)
    assert lo < x.mean() < hi


def test_contrast_family_has_17_rows(rng):
    ids, rows = list(range(32)), []
    for rep in ["raw_384", "pca_16", "pca_64", "sae_seed123", "sae_seed42"]:
        for i in ids:
            base = rng.uniform(1, 2)
            for c, add in (("standard", 1.5), ("adversarial", 0.0)):
                rows.append({"representation": rep, "stimulus_id": i, "metamer_condition": c,
                             "distance_l2": base + add + rng.normal(0, .1), "relative_l2_source_norm": base + add,
                             "cosine_similarity": 1 - (base + add) / 10, "support_jaccard": 0.5 - add / 10})
    df = contrast_family(pd.DataFrame(rows), ids, n_boot=200)
    assert len(df) == 17 and (df.wilcoxon_p_holm >= df.wilcoxon_p_two_sided).all()
    assert 0 < df.wilcoxon_p_two_sided.min() < 1e-8   # exact minimum for 32/32 same sign, never rounded to 0


def test_partial_spearman_removes_depth(rng):
    z = np.arange(40.0)
    assert abs(partial_spearman(z + rng.normal(0, 1, 40), z + rng.normal(0, 1, 40), z, n_boot=100)["rho_partial"]) < 0.6
