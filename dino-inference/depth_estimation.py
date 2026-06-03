"""Monocular depth from a single RGB image with Depth Anything V2.

Depth Anything V2 = a DINOv2 backbone + DPT head, trained on ~62M depth images.
Ungated, runs on Apple Silicon (MPS). Writes a side-by-side [original | depth] PNG.

Usage:
    python depth_estimation.py path/to/img.jpg
    python depth_estimation.py path/to/img.jpg --out outputs/depth.png

Env:
    DEPTH_MODEL  override model id (default: Depth-Anything-V2-Small-hf)
                 bigger: depth-anything/Depth-Anything-V2-Base-hf / -Large-hf
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

MODEL_ID = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")

# perceptually-uniform-ish colormap (turbo) as 8 anchor stops, linearly interpolated
_TURBO = np.array(
    [
        [48, 18, 59], [70, 134, 251], [27, 207, 212], [90, 240, 110],
        [220, 220, 40], [250, 152, 40], [220, 60, 30], [122, 4, 3],
    ],
    dtype=np.float32,
)


def colorize(depth: np.ndarray) -> Image.Image:
    """Normalize a HxW depth array to [0,1] and apply the turbo colormap."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    idx = d * (len(_TURBO) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(_TURBO) - 1)
    frac = (idx - lo)[..., None]
    rgb = _TURBO[lo] * (1 - frac) + _TURBO[hi] * frac
    return Image.fromarray(rgb.astype("uint8"))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: python depth_estimation.py <image> [--out path.png]")
        sys.exit(1)
    img_path = args[0]
    out_path = "outputs/depth.png"
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model: {MODEL_ID} | device: {device}")

    pipe = pipeline("depth-estimation", model=MODEL_ID, device=device)

    img = Image.open(img_path).convert("RGB")
    result = pipe(img)
    # relative inverse-depth: brighter/warmer = nearer
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    print(f"depth map: {depth.shape}, raw range [{depth.min():.2f}, {depth.max():.2f}]")

    depth_img = colorize(depth).resize(img.size, Image.BILINEAR)
    canvas = Image.new("RGB", (img.width * 2 + 10, img.height), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(depth_img, (img.width + 10, 0))
    canvas.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
