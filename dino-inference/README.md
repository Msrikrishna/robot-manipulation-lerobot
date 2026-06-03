# dino-inference

Minimal DINOv3 inference on Apple Silicon (M2 Pro, MPS backend).

## Setup

PyTorch has no wheels for Python 3.14 yet — use a 3.12 env:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

(No `uv`? `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)

## Authenticate (weights are gated)

Accept the license on the model page, then:

```bash
huggingface-cli login
```

## Run

```bash
python dinov3_infer.py                 # synthetic image, just verifies it works
python dinov3_infer.py path/to/img.jpg # your image
```

`pooler_output` is the global image embedding; `last_hidden_state[:, 1:]` is the
patch-token grid (what you'd feed a diffusion policy).

## Do I need to download the model first?

No. `from_pretrained` downloads on first run and caches to
`~/.cache/huggingface/hub`, reused on every later run. Pre-download only if you
want the weights in an explicit folder:

```bash
python download_model.py                                  # ViT-S/16
python download_model.py facebook/dinov3-vitb16-pretrain-lvd1689m
```

## Model sizes (download ≈ fp32 safetensors)

| Model id (`facebook/...`)              | Params | Download | Embed dim |
|----------------------------------------|--------|----------|-----------|
| `dinov3-vits16-pretrain-lvd1689m`      | 21 M   | ~85 MB   | 384       |
| `dinov3-vits16plus-pretrain-lvd1689m`  | 29 M   | ~115 MB  | 384       |
| `dinov3-vitb16-pretrain-lvd1689m`      | 86 M   | ~345 MB  | 768       |
| `dinov3-vitl16-pretrain-lvd1689m`      | 300 M  | ~1.2 GB  | 1024      |
| `dinov3-vith16plus-pretrain-lvd1689m`  | 840 M  | ~3.4 GB  | 1280      |
| `dinov3-vit7b16-pretrain-lvd1689m`     | 6.7 B  | ~27 GB   | 4096      |

ConvNeXt variants (`dinov3-convnext-{tiny,small,base,large}-...`) range ~29–198 M.

Start with **ViT-S/16** on the M2 Pro — fast and plenty for prototyping. Move to
ViT-B/16 for stronger features.
