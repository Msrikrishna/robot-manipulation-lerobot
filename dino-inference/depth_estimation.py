"""Monocular depth from a single RGB image with a DPT depth head.

Several heads are selectable by short key (see MODELS). Each is a frozen ViT
backbone (DINOv3 / DINOv2) with a DPT depth head, run through the HF
depth-estimation pipeline. Runs on Apple Silicon (MPS). Writes a side-by-side
[original | depth] PNG.

Some DINOv3 heads are gated: accept the license on the model page and run
`huggingface-cli login` once before first use.

Usage:
    python depth_estimation.py img.jpg                       # default model
    python depth_estimation.py img.jpg --model da2-indoor    # pick by key
    python depth_estimation.py img.jpg --model facebook/...  # or raw repo id
    python depth_estimation.py img.jpg --out outputs/depth.png
    python depth_estimation.py --list                        # list model keys

Env:
    DEPTH_MODEL  default model key or repo id (default: dinov2-nyu-base)
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

# key -> (repo id, metric?). Metric heads output meters (larger = farther);
# the rest are relative inverse depth (larger = nearer).
MODELS = {
    "dinov3-chmv2": ("facebook/dinov3-vitl16-chmv2-dpt-head", True),
    "dinov2-nyu-small": ("facebook/dpt-dinov2-small-nyu", True),
    "dinov2-nyu-base": ("facebook/dpt-dinov2-base-nyu", True),
    "dinov2-nyu-giant": ("facebook/dpt-dinov2-giant-nyu", True),
    "da2-large": ("depth-anything/Depth-Anything-V2-Large-hf", False),
    "da2-indoor": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", True),
    "da2-outdoor": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", True),
}


def resolve_model(name: str) -> tuple[str, bool]:
    """Map a short key or raw repo id to (repo_id, metric?)."""
    if name in MODELS:
        return MODELS[name]
    return name, False  # raw repo id; assume relative unless known


DEFAULT_MODEL = os.environ.get("DEPTH_MODEL", "dinov2-nyu-base")

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
    if "--list" in sys.argv:
        print("available --model keys:")
        for k, (repo, metric) in MODELS.items():
            print(f"  {k:18s} {repo}  ({'metric' if metric else 'relative'})")
        return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: python depth_estimation.py <image> [--model key] [--out path.png]")
        print("       python depth_estimation.py --list")
        sys.exit(1)
    img_path = args[0]
    out_path = "outputs/depth.png"
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    model_name = DEFAULT_MODEL
    if "--model" in sys.argv:
        model_name = sys.argv[sys.argv.index("--model") + 1]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    repo, metric = resolve_model(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model: {repo} ({'metric' if metric else 'relative'}) | device: {device}")

    pipe = pipeline("depth-estimation", model=repo, device=device)

    img = Image.open(img_path).convert("RGB")
    result = pipe(img)
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    unit = "meters" if metric else "relative inverse-depth"
    print(f"depth map: {depth.shape}, raw range [{depth.min():.2f}, {depth.max():.2f}] ({unit})")

    # Colorize so warmer = nearer regardless of model convention. Metric heads
    # give distance (larger = farther), so invert those before colorizing.
    depth_for_color = -depth if metric else depth
    depth_img = colorize(depth_for_color).resize(img.size, Image.BILINEAR)
    canvas = Image.new("RGB", (img.width * 2 + 10, img.height), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(depth_img, (img.width + 10, 0))
    canvas.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
