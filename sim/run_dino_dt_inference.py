"""Run a trained LeRobot dino_dt policy in the BooksSO101 sim and save a video.

Same flow as run_act_inference.py, but for the DINOv3 diffusion-transformer
policy (`dino_dt`). The checkpoint here lives in an HF *subfolder*
(e.g. srik410/dino_dt_checkpoints/030000/pretrained_model), so we snapshot just
that subfolder to a local dir and load the policy + its processors from it.

Examples:
  python sim/run_dino_dt_inference.py --view --steps 1000
  python sim/run_dino_dt_inference.py \
      --repo-id srik410/dino_dt_checkpoints --subfolder 030000/pretrained_model \
      --steps 300 --out sim/dino_dt_rollout.mp4
"""

import argparse
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import books_env  # noqa: F401  (registers BooksSO101-v1)
import gymnasium as gym

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


def _lerobot_imports():
    from lerobot.policies.dino_dt.modeling_dino_dt import DinoDTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    try:
        from lerobot.utils.control_utils import predict_action  # lerobot 0.4.x
    except ModuleNotFoundError:
        from lerobot.common.control_utils import predict_action  # lerobot 0.5.x
    return DinoDTPolicy, make_pre_post_processors, predict_action


def _resolve_checkpoint(repo_id, subfolder):
    """Return a local path containing config.json/model weights.

    If subfolder is given, snapshot only that subfolder from the Hub and return
    its local path; otherwise return repo_id unchanged (a Hub id or local dir).
    """
    if not subfolder:
        return repo_id
    from huggingface_hub import snapshot_download

    local_root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
    )
    return osp.join(local_root, subfolder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="srik410/dino_dt_checkpoints")
    ap.add_argument("--subfolder", default="030000/pretrained_model",
                    help="HF subfolder holding config.json; empty for repo root/local dir")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--task", default="", help="task string passed to predict_action")
    ap.add_argument("--out", default="sim/dino_dt_rollout.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="live SAPIEN viewer instead of mp4")
    args = ap.parse_args()

    device = torch.device(args.device)
    DinoDTPolicy, make_pre_post_processors, predict_action = _lerobot_imports()
    ckpt = _resolve_checkpoint(args.repo_id, args.subfolder)

    env = gym.make(
        "BooksSO101-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="human" if args.view else "rgb_array",
        max_episode_steps=args.steps,
    )
    obs, _ = env.reset(seed=args.seed)

    policy = DinoDTPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    agent = env.unwrapped.agent
    action_low = env.action_space.low
    action_high = env.action_space.high

    frames = []
    for t in range(args.steps):
        qpos = agent.robot.get_qpos()[0, :6].detach().cpu().numpy()
        rgb = obs["sensor_data"]["front_cam"]["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        observation = {
            "observation.state": (qpos * RAD2DEG).astype(np.float32),
            "observation.images.front_cam": rgb,
        }

        action = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=False,
            task=args.task,
        )
        action = np.asarray(action.detach().cpu()).reshape(-1)[:6]

        sim_action = np.clip(action * DEG2RAD, action_low, action_high).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(sim_action)

        if args.view:
            env.render()
        else:
            frames.append(env.render()[0].cpu().numpy().astype(np.uint8))
        if bool(terminated) or bool(truncated):
            obs, _ = env.reset()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

    env.close()

    if not args.view:
        import imageio.v3 as iio
        iio.imwrite(args.out, np.stack(frames), fps=30)
        print(f"saved rollout -> {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
