"""Minimal DINOv3 inference on Apple Silicon (MPS).

Usage:
    python dinov3_infer.py                 # runs on a synthetic image
    python dinov3_infer.py path/to/img.jpg # runs on your image

Env vars:
    DINOV3_MODEL   override the model id (default: ViT-S/16)
    HF_HOME        override the HF cache location
"""
import os
import sys

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Official gated DINOv3 ViT-B/16 (768-dim). Requires HF access: accept the
# license on the model page and run `huggingface-cli login` once.
MODEL_ID = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")


def load_image(path: str | None) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    # no image given -> synthetic 224x224 so the script runs end-to-end
    import numpy as np

    arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def main() -> None:
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model:  {MODEL_ID}")
    print(f"device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    img = load_image(img_path)
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    cls = out.pooler_output            # (1, D) global image embedding
    tokens = out.last_hidden_state     # (1, 1 + num_patches, D)
    patches = tokens[:, 1:]            # (1, num_patches, D) spatial grid

    print(f"global embedding: {tuple(cls.shape)}")
    print(f"patch tokens:     {tuple(patches.shape)}")
    print(f"embedding dim:    {cls.shape[-1]}")


if __name__ == "__main__":
    main()
