import pytest

try:                                   # a broken/partial torch install raises OSError, not ImportError
    import torch
    import torchvision  # noqa: F401
except Exception as e:  # noqa: BLE001
    pytest.skip(f"torch unavailable: {e}", allow_module_level=True)

from metamer_sae.models.adapters import PreprocSpec, TorchvisionAdapter


@pytest.mark.parametrize("arch,stage", [("alexnet", "relu2"), ("alexnet", "fc0_relu"), ("resnet50", "layer2"), ("convnext_tiny", "stage3")])
def test_stage_hooks_shapes(arch, stage, monkeypatch):
    a = TorchvisionAdapter(arch, "DEFAULT")
    from torchvision import models as M
    monkeypatch.setattr(a, "load", lambda device="cpu": getattr(M, arch)(weights=None).eval())
    monkeypatch.setattr(a, "preprocessing", lambda: PreprocSpec(resize=256, crop=224, mean=[0, 0, 0], std=[1, 1, 1]))
    monkeypatch.setattr(a, "checkpoint_hash", lambda m: "random-init")
    ex = a.extractor(stage)
    out = ex.spatial(torch.zeros(2, 3, 224, 224)); pooled = ex.pool(out)
    assert pooled.shape == (2, ex.meta["output_shape"][0]) and all(a.validate_stages(a.load()).values())
    assert ex.meta["layer"] == stage and ex.meta["family"] == arch
