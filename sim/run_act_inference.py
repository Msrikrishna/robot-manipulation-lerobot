"""Run a trained LeRobot ACT policy in the BooksSO101 sim and save a video.

This mirrors how `lerobot-record` runs a policy. Per lerobot's record loop it:
  1. builds a raw observation dict (numpy), keyed like the dataset features,
  2. loads the policy AND its processor pipelines (make_pre_post_processors),
  3. calls lerobot's own `predict_action`, which does
        prepare_observation_for_inference -> preprocessor -> policy.select_action
        -> postprocessor
     i.e. normalization/un-normalization happen in the loaded processors (the
     policy_preprocessor_* / policy_postprocessor_* files), not by hand.

SIM<->ROBOT UNITS: `predict_action` returns an action in the robot's native units
(what the real SO101 would receive). The real robot uses those units directly;
our sim uses radians, so we bridge with rad<->deg around predict_action. This
bridge is the only sim-specific bit (lerobot-record has no such step because it
talks to the physical robot). It is approximate for the gripper.

Policy (default hdhruva/act_pick_book_place) I/O:
  observation.state (6) | observation.images.front_cam (3x480x640) | action (6)

Examples:
  python sim/run_act_inference.py --view --steps 1000
  python sim/run_act_inference.py --steps 300 --out sim/act_rollout.mp4
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
    """Import the lerobot inference pieces across 0.4/0.5 module layouts."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    try:
        from lerobot.utils.control_utils import predict_action  # lerobot 0.4.x
    except ModuleNotFoundError:
        from lerobot.common.control_utils import predict_action  # lerobot 0.5.x
    return ACTPolicy, make_pre_post_processors, predict_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="hdhruva/act_pick_book_place")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--task", default="", help="task string passed to predict_action")
    ap.add_argument("--out", default="sim/act_rollout.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="live SAPIEN viewer instead of mp4")
    args = ap.parse_args()

    device = torch.device(args.device)
    ACTPolicy, make_pre_post_processors, predict_action = _lerobot_imports()

    # max_episode_steps overrides the env default (200) so the TimeLimit wrapper
    # doesn't truncate/reset mid-run.
    env = gym.make(
        "BooksSO101-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="human" if args.view else "rgb_array",
        max_episode_steps=args.steps,
    )
    obs, _ = env.reset(seed=args.seed)

    # Policy + its processor pipelines, exactly as lerobot-record loads them.
    policy = ACTPolicy.from_pretrained(args.repo_id)
    policy.to(device)
    policy.eval()
    policy.reset()
    # Override the baked-in device (the checkpoint was saved with device_processor
    # set to "cuda"); point it at our device like lerobot-record does.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.repo_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    agent = env.unwrapped.agent
    action_low = env.action_space.low
    action_high = env.action_space.high

    frames = []
    for t in range(args.steps):
        # Build the raw observation dict the way lerobot's record loop does:
        # numpy, dataset-feature keys; images as HxWxC uint8. State goes in the
        # robot's units (sim radians -> degrees).
        qpos = agent.robot.get_qpos()[0, :6].detach().cpu().numpy()
        rgb = obs["sensor_data"]["front_cam"]["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        observation = {
            "observation.state": (qpos * RAD2DEG).astype(np.float32),
            "observation.images.front_cam": rgb,
        }

        # lerobot's own inference: prepare -> preprocess -> select_action -> postprocess
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

        # robot units (degrees) -> sim radians, clipped to the action space
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
