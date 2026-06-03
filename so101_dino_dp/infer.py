"""
Run the trained DINOv3 + Diffusion Transformer policy on the SO101, on your Mac
(MPS). Receding-horizon control: sample `horizon` actions, execute the first
`n_action_steps`, then re-plan.

Two robot-I/O functions are intentionally isolated at the top
(`read_observation` / `send_action`). lerobot's robot API differs across
versions/forks, so wire THOSE two to your lerobot-MakerMods fork and leave the
rest alone.

Quick pipeline test with NO hardware:
    python infer.py --ckpt ckpt/fold.pt --dry_run

On the robot:
    python infer.py --ckpt ckpt/fold.pt --inference_steps 16
"""

import argparse
import time
from collections import deque

import numpy as np
import torch
from diffusers import DDIMScheduler

from model import DiffusionTransformerPolicy


# ============================================================================
# ROBOT I/O ADAPTER  — wire these two to your lerobot fork's SO101 API.
# ============================================================================
def make_robot():
    """Return a connected SO101 follower object (with cameras)."""
    # Example for recent lerobot; adjust import + config to your 0.3.4 fork:
    #   from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    #   robot = SO101Follower(SO101FollowerConfig(port="/dev/tty.usbmodemXXXX",
    #                                             cameras={...}))
    #   robot.connect()
    #   return robot
    raise NotImplementedError("Wire make_robot() to your lerobot fork's SO101 API.")


def read_observation(robot, cam_keys, state_key):
    """Return {cam_key: HxWx3 uint8 np.array, ..., state_key: 1D np.array float}."""
    obs = robot.get_observation()  # dict in most lerobot versions
    out = {}
    for k in cam_keys:
        out[k] = np.asarray(obs[k])          # HxWx3, uint8
    out[state_key] = np.asarray(obs[state_key], dtype=np.float32).reshape(-1)
    return out


def send_action(robot, action_vec, action_key):
    """Send one action (1D np.array, already un-normalized) to the robot."""
    robot.send_action({action_key: action_vec})
# ============================================================================


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def img_to_chw(img_uint8, device):
    """HxWx3 uint8 -> [1,3,H,W] float in [0,1]."""
    t = torch.from_numpy(img_uint8).to(device).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inference_steps", type=int, default=16)
    ap.add_argument("--max_cycles", type=int, default=10000)
    ap.add_argument("--dry_run", action="store_true",
                    help="run the policy on random images, no hardware")
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    st = ckpt["stats"]

    a_mean = torch.tensor(st["action_mean"], device=device)
    a_std = torch.tensor(st["action_std"], device=device)
    s_mean = torch.tensor(st["state_mean"], device=device)
    s_std = torch.tensor(st["state_std"], device=device)

    model = DiffusionTransformerPolicy(
        action_dim=cfg["action_dim"], state_dim=cfg["state_dim"], cam_keys=cfg["cam_keys"],
        n_obs_steps=cfg["n_obs_steps"], horizon=cfg["horizon"], model_id=cfg["dino_model"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        image_size=cfg["image_size"],
    ).to(device)
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=cfg["train_timesteps"],
                              beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    scheduler.set_timesteps(args.inference_steps)

    cam_keys = cfg["cam_keys"]
    state_key = cfg["state_key"]
    n_obs = cfg["n_obs_steps"]
    horizon = cfg["horizon"]
    n_act = cfg["n_action_steps"]
    fps = cfg["fps"]
    period = 1.0 / fps

    robot = None if args.dry_run else make_robot()

    # rolling history of the last n_obs observations
    img_hist = {k: deque(maxlen=n_obs) for k in cam_keys}
    state_hist = deque(maxlen=n_obs)

    def grab_obs():
        if args.dry_run:
            obs = {k: (np.random.rand(cfg["image_size"], cfg["image_size"], 3) * 255).astype(np.uint8)
                   for k in cam_keys}
            obs[state_key] = np.zeros(cfg["state_dim"], dtype=np.float32)
            return obs
        return read_observation(robot, cam_keys, state_key)

    # prime history (repeat first frame)
    first = grab_obs()
    for _ in range(n_obs):
        for k in cam_keys:
            img_hist[k].append(first[k])
        state_hist.append(first[state_key])

    print("running. Ctrl-C to stop.")
    try:
        for cycle in range(args.max_cycles):
            # build batch [1, n_obs, ...]
            batch = {}
            for k in cam_keys:
                frames = [img_to_chw(img_hist[k][t], device) for t in range(n_obs)]
                batch[k] = torch.stack(frames, dim=1)  # [1, n_obs, 3, H, W]
            state = torch.stack(
                [torch.tensor(state_hist[t], device=device).float() for t in range(n_obs)], dim=0
            )[None]  # [1, n_obs, state_dim]
            batch["observation.state"] = (state - s_mean) / s_std

            with torch.no_grad():
                memory = model.encode_obs(batch)
                a = torch.randn(1, horizon, cfg["action_dim"], device=device)
                for t in scheduler.timesteps:
                    pred = model(a, t.expand(1).to(device), memory)
                    a = scheduler.step(pred, t, a).prev_sample
            actions = a[0] * a_std + a_mean  # un-normalize -> [horizon, action_dim]

            # execute the first n_act steps open-loop, then re-plan
            for i in range(n_act):
                t0 = time.time()
                act = actions[i].cpu().numpy()
                if not args.dry_run:
                    send_action(robot, act, cfg["action_key"])
                obs = grab_obs()
                for k in cam_keys:
                    img_hist[k].append(obs[k])
                state_hist.append(obs[state_key])
                if args.dry_run and cycle == 0 and i == 0:
                    print("dry-run action[0]:", np.round(act, 3))
                time.sleep(max(0.0, period - (time.time() - t0)))

            if args.dry_run:
                print(f"cycle {cycle}: planned {horizon} actions, sampled in "
                      f"{args.inference_steps} steps — pipeline OK")
                break
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if robot is not None and hasattr(robot, "disconnect"):
            robot.disconnect()


if __name__ == "__main__":
    main()
