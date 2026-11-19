"""Frozen common observer (ConvNeXt-Tiny, ImageNet-1K) truncated at a stage output."""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

from ..config.schema import ObserverCfg
from ..provenance.hashing import sha256_json, sha256_state_dict

# torchvision convnext_tiny.features: [0] stem, [1] stage1, [2] down, [3] stage2, [4] down,
# [5] stage3 (384 ch, 14x14 @224), [6] down, [7] stage4.
STAGE_TO_FEATURE_INDEX = {"stage1": 1, "stage2": 3, "stage3": 5, "stage4": 7}


@dataclass
class Observer:
    module: nn.Module
    transform: nn.Module
    meta: dict

    @torch.inference_mode()
    def spatial(self, batch: torch.Tensor) -> torch.Tensor:
        return self.module(batch)

    @staticmethod
    def pool(spatial: torch.Tensor) -> torch.Tensor:
        return spatial.float().mean(dim=(2, 3))  # global average pool, float32


def build_observer(cfg: ObserverCfg, device: str = "cpu") -> Observer:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    weights = getattr(ConvNeXt_Tiny_Weights, cfg.weights)
    model = convnext_tiny(weights=weights)
    tf = weights.transforms()
    pre = {"resize": tf.resize_size[0], "crop": tf.crop_size[0], "interpolation": str(tf.interpolation),
           "mean": list(tf.mean), "std": list(tf.std)}
    if pre["resize"] != cfg.resize_size or pre["crop"] != cfg.input_resolution:
        raise ValueError(f"Official preprocessing {pre} disagrees with config; refusing to substitute.")
    idx = STAGE_TO_FEATURE_INDEX[cfg.layer]
    trunk = nn.Sequential(*list(model.features.children())[: idx + 1]).to(device).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        out = trunk(torch.zeros(1, 3, cfg.input_resolution, cfg.input_resolution, device=device))
    if tuple(out.shape[1:]) != tuple(cfg.spatial_shape):
        raise ValueError(f"Observer output {tuple(out.shape[1:])} != configured {cfg.spatial_shape}")
    meta = {"architecture": cfg.architecture, "weights": cfg.weights, "layer": cfg.layer,
            "feature_index": idx, "preprocessing": pre, "preprocessing_sha256": sha256_json(pre),
            "weights_sha256": sha256_state_dict(trunk.state_dict()), "pooling": cfg.pooling,
            "output_shape": list(out.shape[1:])}
    return Observer(trunk, tf, meta)
