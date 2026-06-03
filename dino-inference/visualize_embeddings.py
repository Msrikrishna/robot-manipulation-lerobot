"""Visualize DINOv3 patch embeddings via PCA -> RGB.

Projects each patch token to 3 PCA components, maps them to R/G/B, and writes a
side-by-side [original | PCA feature map] PNG. Similar colors = similar features.

Usage:
    python visualize_embeddings.py path/to/img.jpg
    python visualize_embeddings.py path/to/img.jpg --size 896 --out outputs/vis.png

Flags:
    --size N   longest side fed to the model (default 448). Bigger = finer map,
               more patches, more memory. Must be >= 16; rounded to /16.
    --out P    output PNG path (default outputs/vis.png)

Env:
    DINOV3_MODEL  override model id (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: python visualize_embeddings.py <image> [--out path.png]")
        sys.exit(1)
    img_path = args[0]
    out_path = "outputs/vis.png"
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    size = 448
    if "--size" in sys.argv:
        size = int(sys.argv[sys.argv.index("--size") + 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model: {MODEL_ID} | device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    img = Image.open(img_path).convert("RGB")
    # resize so the longest side ~= --size, both sides multiples of 16 (patch),
    # aspect preserved, and skip the processor's own resize/crop so the grid is
    # exactly (H/16)x(W/16).
    patch = model.config.patch_size
    w, h = img.size
    scale = size / max(w, h)
    nw = max(patch, round(w * scale / patch) * patch)
    nh = max(patch, round(h * scale / patch) * patch)
    img_in = img.resize((nw, nh), Image.BICUBIC)
    inputs = processor(
        images=img_in, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**inputs)
    tokens = out.last_hidden_state[0]  # (num_tokens, D)

    # patch grid from the actual preprocessed tensor + model patch size
    _, _, H, W = inputs["pixel_values"].shape
    gh, gw = H // patch, W // patch
    n_patches = gh * gw
    # DINOv3 token order is [CLS, registers..., patches] -> patches are the tail
    patch_tokens = tokens[-n_patches:].float().cpu()  # (n_patches, D)
    print(f"patch grid: {gh}x{gw} = {n_patches} tokens, dim {patch_tokens.shape[1]}")

    # PCA -> 3 components
    feats = patch_tokens - patch_tokens.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(feats, q=3)
    proj = feats @ V[:, :3]                       # (n_patches, 3)
    proj = proj.reshape(gh, gw, 3).numpy()

    # per-channel robust min-max to [0,1] -> 0..255
    lo = np.percentile(proj, 2, axis=(0, 1))
    hi = np.percentile(proj, 98, axis=(0, 1))
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    pca_img = Image.fromarray((rgb * 255).astype("uint8")).resize(img.size, Image.BILINEAR)

    # side-by-side canvas
    canvas = Image.new("RGB", (img.width * 2 + 10, img.height), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(pca_img, (img.width + 10, 0))
    canvas.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
