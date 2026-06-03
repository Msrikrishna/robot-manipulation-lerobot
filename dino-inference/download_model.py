"""Pre-download a DINOv3 checkpoint into a local folder (optional).

By default `from_pretrained` lazily downloads to ~/.cache/huggingface on first
run. Use this only if you want the weights in an explicit, self-contained dir.

Usage:
    python download_model.py                         # ViT-S/16 -> ./models/<id>
    python download_model.py facebook/dinov3-vitb16-pretrain-lvd1689m

Note: DINOv3 repos are gated. Accept the license on the model page and run
`huggingface-cli login` once before downloading.
"""
import sys

from huggingface_hub import snapshot_download

DEFAULT = "facebook/dinov3-vits16-pretrain-lvd1689m"


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    local_dir = f"./models/{model_id.split('/')[-1]}"
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        # grab only what inference needs, skip duplicate weight formats
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    print(f"downloaded {model_id} -> {path}")
    print("point the inference script at it with:")
    print(f'    DINOV3_MODEL="{local_dir}" python dinov3_infer.py')


if __name__ == "__main__":
    main()
