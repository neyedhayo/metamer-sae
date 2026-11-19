import numpy as np
import pytest

pytest.importorskip("overcomplete")
from metamer_sae.config.schema import SAECfg  # noqa: E402
from metamer_sae.representations.sae import SAERepresentation, build_topk  # noqa: E402
from metamer_sae.training.train_topk import train_topk  # noqa: E402


def test_topk_exact_activity_and_training_smoke(tmp_path, rng):
    cfg = SAECfg(input_dim=16, dictionary_width=32, top_k=4, training_mean="m.npy", primary_checkpoint="a",
                 replicate_checkpoint="b", max_epochs=3, batch_size=32)
    X = rng.normal(size=(200, 16)).astype(np.float32)
    assert train_topk(cfg, X[:150], X[150:], tmp_path / "run", seed=1)["best_epoch"] >= 1
    rep = SAERepresentation(build_topk(16, 32, 4), np.zeros(16), "sae")
    assert np.all(np.count_nonzero(rep.embed(X), axis=1) == 4)
    with pytest.raises(FileExistsError):
        train_topk(cfg, X[:150], X[150:], tmp_path / "run", seed=1)
