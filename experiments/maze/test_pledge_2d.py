"""Kinematic 2D validation of the Pledge controller BEFORE the humanoid.

A unicycle 'robot' (circle, radius R) drives through generated mazes. Contact
sectors are computed geometrically: any wall within (R + skin) of the body
produces a contact whose direction is the outward normal from robot to wall.
Optionally injects the estimator's noise profile (miss rate + false alarms)
to test the debounce chain.

    python3 experiments/maze/test_pledge_2d.py            # clean sensing
    python3 experiments/maze/test_pledge_2d.py --noisy    # det 0.85, FA 0.15/frame
"""
from __future__ import annotations
import math
import random
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.maze.maze_gen import generate
from experiments.maze.pledge import PledgeController, PledgeCfg, world_bin_of, N_WORLD_BINS

R = 0.35          # robot body radius (m) ~ humanoid with arms
SKIN = 0.05       # contact detection shell
DT = 0.02


def contact_sectors(x, y, yaw, boxes, noisy=False, rng=None):
    """N_WORLD_BINS bools: wall intrusion into the shell, per world azimuth bin."""
    raw = [False] * N_WORLD_BINS
    for (cx, cy, hx, hy) in boxes:
        # closest point on box to robot center
        px = max(cx - hx, min(x, cx + hx))
        py = max(cy - hy, min(y, cy + hy))
        d = math.hypot(px - x, py - y)
        if d < R + SKIN:
            az_w = math.atan2(py - y, px - x)          # world azimuth robot->wall
            raw[world_bin_of(az_w)] = True             # world bin (yaw-independent)
    if noisy and rng is not None:
        out = []
        for v in raw:
            if v:
                out.append(rng.random() < 0.85)         # 15% miss
            else:
                out.append(rng.random() < 0.15 / N_WORLD_BINS)  # ~15% FA split over bins
        return out
    return raw


def run_episode(seed, noisy=False, rows=4, cols=4, max_t=None, trace=False):
    if max_t is None:
        max_t = rows * cols * 15.0          # wall-follow path length ~ O(cells)
    spec = generate(rows, cols, seed=seed)
    rng = random.Random(seed * 7 + 1)
    x, y = spec.start_xy
    yaw = 0.0                                           # face +x (toward exit side E)
    ctl = PledgeController(PledgeCfg(dt=DT))
    ctl.reset(yaw, (x, y))
    exit_x, exit_y = spec.exit_xy
    t, path = 0.0, []
    while t < max_t:
        raw = contact_sectors(x, y, yaw, spec.boxes, noisy, rng)
        vx, vy, wz, st = ctl.step(raw, yaw, (x, y))
        # unicycle integration + crude wall collision response (slide)
        nx = x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT
        ny = y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
        # forbid penetration: push out of any box the center would enter (radius R)
        for (cx, cy, hx, hy) in spec.boxes:
            px = max(cx - hx, min(nx, cx + hx)); py = max(cy - hy, min(ny, cy + hy))
            d = math.hypot(px - nx, py - ny)
            if d < R and d > 1e-9:
                nx = px + (nx - px) / d * R; ny = py + (ny - py) / d * R
        x, y = nx, ny
        yaw = math.atan2(math.sin(yaw + wz * DT), math.cos(yaw + wz * DT))
        t += DT
        if trace and int(t / DT) % 25 == 0:
            path.append((round(x, 2), round(y, 2), st))
        if x > exit_x - 0.4 and abs(y - exit_y) < 1.2:  # crossed the opened E wall
            return True, t, path
    return False, t, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    ok, times = 0, []
    fails = []
    for s in range(args.n):
        succ, t, _ = run_episode(s, args.noisy, args.rows, args.cols)
        ok += succ; times.append(t)
        if not succ:
            fails.append(s)
    mode = "noisy(det .85/FA .15)" if args.noisy else "clean"
    print(f"[pledge-2d] {mode}  {args.rows}x{args.cols}  success {ok}/{args.n} "
          f"({100*ok/args.n:.0f}%)  mean_t={sum(times)/len(times):.1f}s  fails={fails[:10]}")


if __name__ == "__main__":
    main()
