import numpy as np
import pandas as pd
import pytest

from metamer_sae.evaluation.pools import FeatherPools
from metamer_sae.features.analysis import gini, match_dictionaries, support_groups
from metamer_sae.representations.pca import PCABasis


def test_pca_matches_sklearn_and_reconstruction_monotone(rng):
    from sklearn.decomposition import PCA
    X = rng.normal(size=(300, 20)) @ rng.normal(size=(20, 20))
    b, sk = PCABasis.fit(X, 20), PCA(20, svd_solver="full").fit(X)
    assert np.allclose(b.scores(X), sk.transform(X), atol=1e-8)
    mse = [b.view(k).reconstruction_mse(X).mean() for k in (2, 5, 10, 20)]
    assert all(a >= c for a, c in zip(mse, mse[1:])) and mse[-1] < 1e-18


def test_pca_save_is_immutable(tmp_path, rng):
    b = PCABasis.fit(rng.normal(size=(50, 6)), 6)
    b.save(tmp_path / "p")
    with pytest.raises(FileExistsError):
        b.save(tmp_path / "p")


def test_gini_and_support():
    assert gini(np.ones(10)) < 1e-9 and gini(np.array([0] * 9 + [1.0])) > 0.85
    assert support_groups(np.array([0, 5, 60]))["rare"] == 1


def test_cross_seed_matching_identity(rng):
    D = rng.normal(size=(30, 8))
    m = match_dictionaries(D, D[rng.permutation(30)])
    assert np.allclose(m["cosine"], 1) and m["frac_ge_0.9"] == 1.0


def test_reserve_rows_are_guarded(tmp_path):
    np.save(tmp_path / "pooled_activations.npy", np.zeros((3, 4), np.float32))
    led = pd.DataFrame({"stimulus_id": [1, 2, 3], "split": ["development", "primary_evaluation", "reserve"],
                        "activation_index": [0, 1, 2]})
    pools = FeatherPools({"source": tmp_path}, led)
    pools.rows("development")
    with pytest.raises(PermissionError):
        pools.rows("reserve")
