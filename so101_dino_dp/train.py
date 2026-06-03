"""
Train the DINOv3 + Diffusion Transformer policy on episodes recorded in
LeRobotDataset format.

Run this on a GPU box (Nebius / Brev), not the Mac.

Example
-------
python train.py \
  --repo_id <you>/so101_fold_v1 \
  --cam_keys observation.images.overhead observation.images.wrist \
  --dino_model facebook/dinov3-vits16-pretrain-lvd1689m \
  --n_obs_steps 2 --horizon 16 --n_action_steps 8 \
  --batch_size 64 --steps 200000 --out ckpt/fold.pt

API note: LeRobotDataset import paths drift between lerobot versions. If the
import below fails on your fork, change it to match (e.g. `from lerobot.datasets
.lerobot_dataset import LeRobotDataset`).
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # adjust per fork
from model import DiffusionTransformerPolicy


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_delta_timestamps(fps, n_obs, horizon, cam_keys, state_key, action_key):
    # observations look back: e.g. n_obs=2 -> [-1/fps, 0.0]
    obs_t = [-(n_obs - 1 - i) / fps for i in range(n_obs)]
    # actions look forward: [0, 1/fps, ..., (horizon-1)/fps]
    act_t = [i / fps for i in range(horizon)]
    dt = {action_key: act_t, state_key: obs_t}
    for k in cam_keys:
        dt[k] = obs_t
    return dt


def stat_tensor(stats, key, field, device):
    return torch.as_tensor(np.array(stats[key][field]), dtype=torch.float32, device=device)


@torch.no_grad()
def ema_update(ema_params, model, decay):
    for e, p in zip(ema_params, (p for p in model.parameters() if p.requires_grad)):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None, help="local dataset root (optional)")
    ap.add_argument("--cam_keys", nargs="+",
                    default=["observation.images.overhead", "observation.images.wrist"])
    ap.add_argument("--state_key", default="observation.state")
    ap.add_argument("--action_key", default="action")
    ap.add_argument("--dino_model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--n_obs_steps", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--train_timesteps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default="ckpt/policy.pt")
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)

    # --- dataset -----------------------------------------------------------
    # peek fps from metadata first (delta_timestamps needs it)
    meta_ds = LeRobotDataset(args.repo_id, root=args.root)
    fps = meta_ds.fps if hasattr(meta_ds, "fps") else meta_ds.meta.fps
    dt = make_delta_timestamps(fps, args.n_obs_steps, args.horizon,
                               args.cam_keys, args.state_key, args.action_key)
    dataset = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=dt)
    stats = dataset.meta.stats

    state_dim = len(np.array(stats[args.state_key]["mean"]).reshape(-1))
    action_dim = len(np.array(stats[args.action_key]["mean"]).reshape(-1))
    print(f"fps={fps} state_dim={state_dim} action_dim={action_dim} episodes via {args.repo_id}")

    a_mean = stat_tensor(stats, args.action_key, "mean", device)
    a_std = stat_tensor(stats, args.action_key, "std", device).clamp_min(1e-6)
    s_mean = stat_tensor(stats, args.state_key, "mean", device)
    s_std = stat_tensor(stats, args.state_key, "std", device).clamp_min(1e-6)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True,
    )

    # --- model -------------------------------------------------------------
    model = DiffusionTransformerPolicy(
        action_dim=action_dim, state_dim=state_dim, cam_keys=args.cam_keys,
        n_obs_steps=args.n_obs_steps, horizon=args.horizon, model_id=args.dino_model,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        image_size=args.image_size,
    ).to(device)
    model.train()

    scheduler = DDPMScheduler(num_train_timesteps=args.train_timesteps,
                              beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_params/1e6:.2f}M (DINOv3 backbone frozen)")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    ema_params = [p.detach().clone() for p in trainable]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    step = 0
    while step < args.steps:
        for batch in loader:
            # move + cast
            for k in args.cam_keys:
                batch[k] = batch[k].to(device, non_blocking=True).float()
                if batch[k].max() > 1.5:  # uint8 -> [0,1] safety
                    batch[k] = batch[k] / 255.0
            batch[args.state_key] = batch[args.state_key].to(device).float()
            actions = batch[args.action_key].to(device).float()  # [B, horizon, action_dim]

            # normalize
            batch["observation.state"] = (batch[args.state_key] - s_mean) / s_std
            actions = (actions - a_mean) / a_std

            memory = model.encode_obs(batch)

            noise = torch.randn_like(actions)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (actions.shape[0],),
                              device=device)
            noisy = scheduler.add_noise(actions, noise, t)

            pred = model(noisy, t, memory)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ema_update(ema_params, model, args.ema_decay)

            if step % 50 == 0:
                print(f"step {step:>7} loss {loss.item():.4f}")
            step += 1

            if step % 5000 == 0 or step >= args.steps:
                save_ckpt(args, model, ema_params, trainable, fps, state_dim, action_dim,
                          stats, scheduler)
                print(f"saved -> {args.out}")
            if step >= args.steps:
                break


def save_ckpt(args, model, ema_params, trainable, fps, state_dim, action_dim, stats, scheduler):
    # write EMA weights into a copy of the state_dict for the trainable params
    ema_state = {}
    ema_iter = iter(ema_params)
    for name, p in model.named_parameters():
        ema_state[name] = next(ema_iter).clone() if p.requires_grad else p.detach().clone()
    torch.save(
        {
            "ema_state_dict": ema_state,
            "config": {
                "action_dim": action_dim, "state_dim": state_dim,
                "cam_keys": args.cam_keys, "state_key": args.state_key,
                "action_key": args.action_key, "n_obs_steps": args.n_obs_steps,
                "horizon": args.horizon, "n_action_steps": args.n_action_steps,
                "dino_model": args.dino_model, "d_model": args.d_model,
                "n_heads": args.n_heads, "n_layers": args.n_layers,
                "image_size": args.image_size, "fps": fps,
                "train_timesteps": args.train_timesteps,
            },
            "stats": {
                "action_mean": np.array(stats[args.action_key]["mean"]).reshape(-1).tolist(),
                "action_std": np.array(stats[args.action_key]["std"]).reshape(-1).tolist(),
                "state_mean": np.array(stats[args.state_key]["mean"]).reshape(-1).tolist(),
                "state_std": np.array(stats[args.state_key]["std"]).reshape(-1).tolist(),
            },
        },
        args.out,
    )


if __name__ == "__main__":
    main()
