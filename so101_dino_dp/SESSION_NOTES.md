# SO101 + DINOv3 + Diffusion Policy — Session Notes

Goal: drive an **SO101 arm** for robust manipulation (e.g. **cloth folding**) using
**frozen DINOv3 image embeddings** as the perceptual front-end feeding a
**diffusion policy** action head. Training on episodes; inference separately on a Mac (MPS).

## What we decided (the design)

- **Backbone**: frozen DINOv3 ViT-S/16 (~21M), self-supervised, dense patch tokens.
  Frozen — not fine-tuned — to preserve the billion-image invariances and stay
  trainable on a few hundred demos. (LoRA/partial unfreeze only once you have 1000s of demos.)
- **Action head**: **Diffusion Transformer** (transformer-decoder denoiser) that
  **cross-attends to DINOv3 patch tokens** ("option B" — no spatial pooling, so
  cloth-edge detail survives). Predicts noise (epsilon).
- **Diffusion**: DDPM (100 steps, cosine beta) for training; DDIM (16 steps) for inference.
- **Action chunking**: predict horizon=16, execute n_action_steps=8, re-plan.
- **Resolution**: 256x256 (multiple of 16 -> 16x16 = 256 patch tokens).
- **Cameras**: overhead + wrist, shared encoder, tagged with learned cam/timestep embeddings.

## Why it should work (the reasoning)

1. Frozen DINOv3 supplies **deformation/lighting/viewpoint-invariant dense features**
   that scarce demos can't learn themselves -> reduces the problem to features->actions.
2. Freezing preserves the prior (joint fine-tuning would overfit 21M params on ~200 demos).
3. Diffusion head models the **multimodal** action distribution (grab left OR right)
   instead of averaging modes like MSE behavior cloning -> avoids invalid mean actions.
4. The two halves fix the two distinct IL failure modes (brittle perception; mode-averaging).

### Honest caveats
- Pooling can bottleneck DINOv3's spatial detail -> we use cross-attention (option B), not pooling.
- DINOv3 saw web images, not top-down fabric on your table — features are good but not optimal.
- Diffusion only covers states the demos visited: **data diversity is the hard floor**
  (plan 150-300+ demos, varied cloth pose/color/lighting). Backbone raises the ceiling, not the floor.
- The +~10% success figure is from RoboMimic, not SO101 folding — treat as a hypothesis to test
  vs your existing SmolVLA baseline.
- Folding on a low-DOF SO101 is genuinely frontier; expect limited robustness early.

## Reference
- DINOv3-Diffusion Policy paper: arXiv 2509.17684
- facebookresearch/dinov3 (gated weights, non-Apache DINOv3 license)
- DINOv3 model suite: ViT T/S/B/L/H+/7B (patch16); ConvNeXt T/S/B/L for on-device

## Files in this folder
- `model.py`  — shared architecture (frozen DINOv3 via HF transformers + Diffusion Transformer).
- `train.py`  — trains on LeRobotDataset episodes. Run on a GPU box (Nebius/Brev).
- `infer.py`  — receding-horizon control on SO101, Mac/MPS. Has `--dry_run` (no hardware).
- `requirements.txt`

## How to run
```bash
pip install -r requirements.txt
huggingface-cli login                 # DINOv3 weights are gated
pip install -e /path/to/lerobot-MakerMods

# Train (GPU)
python train.py --repo_id <you>/so101_fold_v1 \
  --cam_keys observation.images.overhead observation.images.wrist \
  --steps 200000 --out ckpt/fold.pt

# Test pipeline on Mac, no robot
python infer.py --ckpt ckpt/fold.pt --dry_run

# Run on the SO101
python infer.py --ckpt ckpt/fold.pt --inference_steps 16
```

## 3 things to adjust before it runs (flagged in code)
1. **DINOv3 model id** — default `facebook/dinov3-vits16-pretrain-lvd1689m`; confirm on the
   HF model card + accept the gated license.
2. **Robot I/O** — wire `make_robot()` / `read_observation()` / `send_action()` in `infer.py`
   to your lerobot-MakerMods 0.3.4 fork's SO101 API (this API drifts between versions).
3. **lerobot import + key names** — `train.py`'s `LeRobotDataset` import path and the
   `cam_keys`/`state_key`/`action_key` defaults must match your dataset.

## Verify on your data
- `delta_timestamps` window: obs = [-1/fps, 0], action = [0..15]/fps.
- `dataset.meta.stats` must carry `action` + `observation.state` mean/std (used for
  normalization; baked into the checkpoint for inference).
- If `transformers` is too old to know DINOv3, `AutoModel` errors -> upgrade.

## Open follow-ups (not yet built)
- `eval_episodes.py` to replay held-out episodes and score success rate (mirror your eval_fixed.py).
- Adapting the robot I/O adapters to the actual SO101 fork API.
- Optional: feature caching to speed up training (frozen backbone makes this safe).
