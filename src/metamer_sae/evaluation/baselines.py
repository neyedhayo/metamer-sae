"""Image-space baselines from the proposal: pixel distance, SSIM, LPIPS (optional deps).

Protocol (fixed here, record it with results): RGB, resize shorter side to 236 then centre-crop 224
(bilinear, same as the observer), pixel range [0,1]. LPIPS expects [-1,1]. Pairing is by stimulus ID.
Recognition-model accuracy (other networks) is a distinct baseline, not implemented here.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def load_image(path, size: int = 236, crop: int = 224) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB"); w, h = im.size; s = size / min(w, h)
        im = im.resize((max(round(w * s), crop), max(round(h * s), crop)), Image.BILINEAR)
        w, h = im.size; l, t = (w - crop) // 2, (h - crop) // 2
        return np.asarray(im.crop((l, t, l + crop, t + crop)), np.float32) / 255.0


def pixel_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = (a - b).ravel()
    return {"pixel_l2": float(np.linalg.norm(d)), "pixel_rmse": float(np.sqrt((d ** 2).mean()))}


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


class LPIPS:
    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips, torch  # noqa: E401
        self.torch, self.device = torch, device
        self.fn = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = lambda x: self.torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device) * 2 - 1  # noqa: E731
        with self.torch.no_grad():
            return float(self.fn(t(a), t(b)))
