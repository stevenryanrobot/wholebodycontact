"""Closed-loop blind maze navigation: walking policy + Pledge controller.

    MUJOCO_GL=egl .../python experiments/maze/nav_run.py \
        --ckpt experiments/maze/logs/g1_velocity/<run>/model_XXXX.pt \
        --rows 4 --cols 4 --n-mazes 8 --time-limit 240

Per-episode: robot spawns at the maze start cell, PledgeController consumes GT
(or estimated) contact sectors and drives twist commands until it crosses the
exit, falls, or times out. One maze per run (walls are baked into the scene),
episodes vary by maze seed across runs; use --n-envs>1 for repeated attempts
of the SAME maze in parallel.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.maze.maze_env import shifted_maze, build_env, load_policy, MazeInterface
from experiments.maze.pledge import PledgeController, PledgeCfg

CTRL_DT = 0.02   # velocity task control period (50 Hz)


def run_maze(ckpt, maze_seed, rows, cols, n_envs, time_limit, device="cuda:0",
             estimator=None, v_walk=0.5, force_thresh=3.0, verbose=True, trace_path=None):
    maze = shifted_maze(rows, cols, seed=maze_seed)
    env = build_env(maze, num_envs=n_envs, device=device, episode_s=time_limit + 5)
    mi = MazeInterface(env, maze)
    policy = load_policy(env, ckpt, device)

    if isinstance(estimator, str):        # path to contact_gru ckpt -> learned sensing
        from experiments.maze.sensor_model import RollingEstimator
        from experiments.maze.features import FeatureExtractor
        est = RollingEstimator(estimator, device=device)
        fx = FeatureExtractor(env.unwrapped)
        estimator = lambda m: est.world_bins(fx.proprio(), m.pose2d()[:, 2])

    cfg = PledgeCfg(dt=CTRL_DT, v_walk=v_walk, v_slow=0.3 * v_walk / 0.5)
    ctls = [PledgeController(PledgeCfg(**{**cfg.__dict__, "seed": i})) for i in range(n_envs)]

    obs, _ = env.reset()
    pose = mi.pose2d()
    for i, c in enumerate(ctls):
        c.reset(float(pose[i, 2]), (float(pose[i, 0]), float(pose[i, 1])))
        c.phi0 = 0.0        # compass assumption: exit is on the +x (East) side
    # maze bounding box (shifted frame): [-cell/2-t, span+cell/2+t]; anything well
    # outside without success = punched through a wall -> hard failure
    span_x = cols * 2.0 - 1.0 + 1.0
    span_y = rows * 2.0 - 1.0 + 1.0

    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    result = [{"success": False, "fell": False, "t": None} for _ in range(n_envs)]
    steps = int(time_limit / CTRL_DT)
    t0 = time.time()

    for k in range(steps):
        with torch.inference_mode():
            act = policy(obs)
            # sectors: GT or estimator
            if estimator is None:
                sec = mi.gt_sectors(force_thresh)
            else:
                sec = estimator(mi)                      # [N,12] bool world bins
            pose = mi.pose2d()
            cmds = torch.zeros(n_envs, 3, device=device)
            for i, c in enumerate(ctls):
                if done_mask[i]:
                    continue
                vx, vy, wz, st = c.step(
                    [bool(x) for x in sec[i]], float(pose[i, 2]),
                    (float(pose[i, 0]), float(pose[i, 1])))
                cmds[i, 0], cmds[i, 1], cmds[i, 2] = vx, vy, wz
            mi.set_cmd_batch(cmds)
            obs, _, dones, _ = env.step(act)

        if trace_path is not None and k % 5 == 0:
            with open(trace_path, "a") as tf:
                tf.write(json.dumps({
                    "t": round(k * CTRL_DT, 2),
                    "pose": [round(float(v), 3) for v in pose[0]],
                    "bins": [int(b) for b in sec[0]],
                    "cmd": [round(float(v), 2) for v in cmds[0]],
                    "state": ctls[0].state,
                }) + "\n")

        succ = mi.success(); fell = mi.fell()
        env_done = dones.squeeze(-1) if (dones is not None and dones.ndim > 1) else dones
        for i in range(n_envs):
            if done_mask[i]:
                continue
            if bool(succ[i]):
                result[i] = {"success": True, "fell": False, "t": k * CTRL_DT}
                done_mask[i] = True
            elif bool(fell[i]) or (env_done is not None and bool(env_done[i])):
                # our height check OR the env's own termination (which silently
                # auto-resets the robot to the start) — count as a fall-fail
                result[i] = {"success": False, "fell": True, "t": k * CTRL_DT,
                             "state": ctls[i].state}
                done_mask[i] = True
            elif (pose[i, 0] < -2.0 or pose[i, 0] > span_x + 1.5
                  or pose[i, 1] < -2.0 or pose[i, 1] > span_y):
                # outside the maze without crossing the exit window: wall breach
                result[i] = {"success": False, "fell": False, "escaped": True,
                             "t": k * CTRL_DT}
                done_mask[i] = True
        if bool(done_mask.all()):
            break
        if verbose and k % 500 == 0:
            states = [c.state for c in ctls]
            print(f"  [nav] t={k*CTRL_DT:5.1f}s pose0=({pose[0,0]:.2f},{pose[0,1]:.2f}) "
                  f"states={states[:4]} done={int(done_mask.sum())}/{n_envs}", flush=True)

    env.close()
    wall = time.time() - t0
    return maze, result, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-mazes", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=240.0)
    ap.add_argument("--v-walk", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="experiments/maze/results/nav_gt.jsonl")
    ap.add_argument("--trace", type=str, default=None)
    ap.add_argument("--sensor", type=str, default=None,
                    help="contact_gru.pt -> learned proprioceptive sensing instead of GT")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_res = []
    for ms in range(args.n_mazes):
        print(f"[nav] maze seed {ms} ({args.rows}x{args.cols}) ...", flush=True)
        maze, res, wall = run_maze(args.ckpt, ms, args.rows, args.cols,
                                   args.n_envs, args.time_limit, v_walk=args.v_walk,
                                   trace_path=args.trace, estimator=args.sensor)
        ok = sum(r["success"] for r in res)
        fell = sum(r["fell"] for r in res)
        print(f"[nav] maze {ms}: success {ok}/{len(res)}  fell {fell}  wall={wall:.0f}s", flush=True)
        all_res.append({"maze_seed": ms, "results": res})
        with open(args.out, "w") as f:
            for row in all_res:
                f.write(json.dumps(row) + "\n")

    n = sum(len(r["results"]) for r in all_res)
    ok = sum(x["success"] for r in all_res for x in r["results"])
    fell = sum(x["fell"] for r in all_res for x in r["results"])
    print(f"[nav] TOTAL success {ok}/{n} ({100*ok/max(n,1):.0f}%)  fell {fell}  -> {args.out}")


if __name__ == "__main__":
    main()
