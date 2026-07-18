"""Collect (proprioception, wall-contact label) pairs: random walk in mazes.

Robots walk with randomly resampled velocity commands inside mazes, naturally
bumping/leaning/sliding into walls — the *sustained walking contact*
distribution the maze task needs (unexplored by prior work, cf. research doc).

    MUJOCO_GL=egl .../python experiments/maze/collect_sensor_data.py \
        --ckpt <model.pt> --n-envs 32 --steps 15000 --maze-seed 0 \
        --out experiments/maze/data/sensor_seed0.h5
"""
from __future__ import annotations
import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.maze.maze_env import shifted_maze, build_env, load_policy, MazeInterface
from experiments.maze.features import FeatureExtractor, contact_label, FEAT_DIM

CTRL_DT = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--maze-seed", type=int, default=0)
    ap.add_argument("--cmd-resample-s", type=float, default=3.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    maze = shifted_maze(args.rows, args.cols, seed=args.maze_seed)
    # long episodes; fall termination still resets individual envs (fresh data)
    env = build_env(maze, num_envs=args.n_envs, device=dev, episode_s=10_000.0)
    mi = MazeInterface(env, maze)
    fx = FeatureExtractor(env.unwrapped)
    policy = load_policy(env, args.ckpt, dev)

    N, T = args.n_envs, args.steps
    X = np.zeros((T, N, FEAT_DIM), dtype=np.float32)
    Yc = np.zeros((T, N), dtype=bool)          # contact
    Ya = np.zeros((T, N), dtype=np.float32)    # robot-frame azimuth
    Ym = np.zeros((T, N), dtype=np.float32)    # magnitude (N)
    Er = np.zeros((T, N), dtype=bool)          # env was reset at this step (window break)

    obs, _ = env.reset()
    g = torch.Generator(device="cpu").manual_seed(args.maze_seed * 131 + 7)
    resample_every = int(args.cmd_resample_s / CTRL_DT)

    def sample_cmds():
        # biased toward forward walking so robots cruise into walls; some pure turns
        vx = torch.rand(N, generator=g) * 1.0 - 0.3          # [-0.3, 0.7]
        wz = torch.rand(N, generator=g) * 1.4 - 0.7          # [-0.7, 0.7]
        vy = torch.rand(N, generator=g) * 0.2 - 0.1
        return torch.stack([vx, vy, wz], -1).to(dev)

    cmds = sample_cmds()
    with torch.inference_mode():
        for k in range(T):
            if k % resample_every == 0:
                cmds = sample_cmds()
            mi.set_cmd_batch(cmds)
            act = policy(obs)
            obs, _, dones, _ = env.step(act)
            X[k] = fx.proprio().cpu().numpy()
            c, a, m = contact_label(mi)
            Yc[k] = c.cpu().numpy(); Ya[k] = a.cpu().numpy(); Ym[k] = m.cpu().numpy()
            if dones is not None:
                d = dones.squeeze(-1) if dones.ndim > 1 else dones
                Er[k] = d.bool().cpu().numpy()
            if k % 2000 == 0:
                print(f"[collect] {k}/{T}  contact_rate={Yc[:k+1].mean():.3f}", flush=True)

    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h.create_dataset("X", data=X, compression="lzf")
        h.create_dataset("contact", data=Yc)
        h.create_dataset("az_r", data=Ya)
        h.create_dataset("mag", data=Ym)
        h.create_dataset("reset", data=Er)
        h.attrs["feat_dim"] = FEAT_DIM
        h.attrs["ctrl_dt"] = CTRL_DT
        h.attrs["maze_seed"] = args.maze_seed
    print(f"[collect] saved {T}x{N}x{FEAT_DIM} -> {args.out}  "
          f"contact_rate={Yc.mean():.3f}  mean|F|@contact={Ym[Yc].mean() if Yc.any() else 0:.1f}N")


if __name__ == "__main__":
    main()
