"""Record an mp4 of blind maze navigation (GT contact or estimator).

    MUJOCO_GL=egl .../python experiments/maze/record_video.py --ckpt <model.pt> \
        --maze-seed 0 --rows 3 --cols 3 --time-limit 300 \
        --out experiments/maze/results/nav_seed0.mp4 [--sensor experiments/maze/data/contact_gru.pt]
"""
from __future__ import annotations
import argparse
import os
import sys

import imageio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.maze.maze_env import shifted_maze, build_env, load_policy, MazeInterface
from experiments.maze.pledge import PledgeController, PledgeCfg
from experiments.maze.features import FeatureExtractor

CTRL_DT = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sensor", type=str, default=None,
                    help="contact_gru.pt -> use the learned estimator instead of GT")
    ap.add_argument("--maze-seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--v-walk", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=str, default="experiments/maze/results/nav.mp4")
    ap.add_argument("--ctl-seed", type=int, default=0, help="Pledge RNG seed (random kicks)")
    ap.add_argument("--chase-cam", action="store_true",
                    help="use the default body-tracking camera instead of top-down")
    args = ap.parse_args()
    dev = "cuda:0"

    maze = shifted_maze(args.rows, args.cols, seed=args.maze_seed)
    env = build_env(maze, num_envs=1, device=dev, episode_s=args.time_limit + 5,
                    render_mode="rgb_array", topdown=not args.chase_cam)
    mi = MazeInterface(env, maze)
    policy = load_policy(env, args.ckpt, dev)

    est = None
    fx = None
    if args.sensor:
        from experiments.maze.sensor_model import RollingEstimator
        est = RollingEstimator(args.sensor, device=dev)
        fx = FeatureExtractor(env.unwrapped)

    ctl = PledgeController(PledgeCfg(dt=CTRL_DT, v_walk=args.v_walk, seed=args.ctl_seed))
    obs, _ = env.reset()
    pose = mi.pose2d()
    ctl.reset(float(pose[0, 2]), (float(pose[0, 0]), float(pose[0, 1])))
    ctl.phi0 = 0.0

    frames = []
    every = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
    steps = int(args.time_limit / CTRL_DT)
    status = "timeout"
    with torch.inference_mode():
        for k in range(steps):
            act = policy(obs)
            if est is None:
                sec = mi.gt_sectors()
            else:
                sec = est.world_bins(fx.proprio(), mi.pose2d()[:, 2])
            pose = mi.pose2d()
            vx, vy, wz, st = ctl.step([bool(x) for x in sec[0]], float(pose[0, 2]),
                                      (float(pose[0, 0]), float(pose[0, 1])))
            mi.set_cmd(vx, vy, wz)
            obs, _, _, _ = env.step(act)
            if k % every == 0:
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(frame)
            if bool(mi.success()[0]):
                status = "SUCCESS"
                for _ in range(args.fps):        # linger a second on the win
                    frames.append(frames[-1])
                break
            if bool(mi.fell()[0]):
                status = "fell"
                break
            if k % 500 == 0:
                print(f"[rec] t={k*CTRL_DT:.0f}s pose=({pose[0,0]:.1f},{pose[0,1]:.1f}) "
                      f"{ctl.state} frames={len(frames)}", flush=True)

    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"[rec] {status} at t={k*CTRL_DT:.0f}s  {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
