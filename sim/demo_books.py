"""View / sanity-check the BooksSO101-v1 environment.

Examples:
  # interactive viewer (don't click entities unless pinocchio is installed)
  python sim/demo_books.py

  # headless: settle the physics and save a rendered frame to sim/books_render.png
  python sim/demo_books.py --save sim/books_render.png

  # apply random arm actions instead of holding still
  python sim/demo_books.py --random
"""

import argparse
import os.path as osp
import sys

import gymnasium as gym
import numpy as np

# make `import books_env` work regardless of cwd, and register the env
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import books_env  # noqa: F401  (registers BooksSO101-v1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env-id", default="BooksSO101-v1")
    ap.add_argument("--random", action="store_true", help="random arm actions")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--save", default=None, help="save a rendered frame to this path")
    ap.add_argument(
        "--cam",
        choices=["front", "top", "wrist", "both"],
        default="front",
        help="which render camera for --save (front, top-down, wrist/ego, or both)",
    )
    args = ap.parse_args()

    render_mode = "rgb_array" if args.save else "human"
    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        render_view=args.cam,
    )

    obs, _ = env.reset(seed=0)
    for _ in range(args.steps):
        if args.random:
            action = env.action_space.sample()
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if render_mode == "human":
            env.render()

    if args.save:
        frame = env.render()  # (1, H, W, 3) tensor for the human render camera
        try:
            frame = frame[0].cpu().numpy()
        except Exception:
            frame = np.asarray(frame).squeeze()
        try:
            import imageio.v3 as iio

            iio.imwrite(args.save, frame.astype(np.uint8))
        except Exception:
            from PIL import Image

            Image.fromarray(frame.astype(np.uint8)).save(args.save)
        print(f"saved rendered frame -> {args.save}")

    env.close()


if __name__ == "__main__":
    main()
