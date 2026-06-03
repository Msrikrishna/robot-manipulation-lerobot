# robot-manipulation-lerobot

DINO-based robot manipulation built on [LeRobot](https://github.com/huggingface/lerobot) — a monorepo bringing together a customized policy stack, a control/inference app, hardware + simulation, and a small toolkit for inspecting DINOv3 vision features.

## Repository layout

| Folder | What it is |
|---|---|
| [`lerobot-MakerMods/`](lerobot-MakerMods) | Fork of 🤗 LeRobot with custom policies, including a DINO + diffusion policy and a `dino_dt` policy for SO-101 manipulation. |
| [`MakerMods-App/`](MakerMods-App) | Frontend app for recording episodes, running inference, and driving the robot. |
| [`XLeRobot-hardware-sim/`](XLeRobot-hardware-sim) | Hardware designs and simulation (MuJoCo, ManiSkill) based on [XLeRobot](https://github.com/Vector-Wangel/XLeRobot). |
| [`dino-inference/`](dino-inference) | Standalone tools for running DINOv3 inference, visualizing patch embeddings, and monocular depth estimation on Apple Silicon (MPS). |

> Note: the three subprojects were merged in from separate repositories with full history via `git subtree`, so their commit history is preserved in this repo's log.

## `dino-inference/` toolkit

Minimal scripts for understanding what DINOv3 "sees" — useful when designing vision features for a manipulation policy. Runs on Apple Silicon via the PyTorch MPS backend.

```bash
cd dino-inference
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

python dinov3_infer.py         IMAGE              # print embedding shapes
python visualize_embeddings.py IMAGE --size 896   # PCA-of-patches -> RGB feature map
python depth_estimation.py     IMAGE              # monocular depth (Depth Anything V2)
```

By default the scripts use an **ungated** DINOv3 ViT-B/16 mirror so no Hugging Face access approval is needed. See [`dino-inference/README.md`](dino-inference/README.md) for model sizes, the gated `facebook/dinov3-*` weights, and flags.

### Example: PCA feature map (`visualize_embeddings.py --size 896`)

Each 16×16 patch becomes a 768-d DINOv3 vector; PCA projects it to 3 dims shown as RGB. Similar colors ≈ similar features — object/part structure emerges.

![PCA feature map](dino-inference/outputs/thor_vis_896.png)

### Example: monocular depth (`depth_estimation.py`)

Relative depth from a single RGB image (red = nearer, blue = farther).

![Depth map](dino-inference/outputs/thor_depth.png)

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot)
- [DINOv3](https://github.com/facebookresearch/dinov3) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
