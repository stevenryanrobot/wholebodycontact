"""Whole-Body Contact (Plan A) — v3 sweep driver.

Loads the 8 GB dataset ONCE onto the GPU as a [T,E,D] grid, then runs a curated
list of v3 experiments that all reuse that grid (an epoch is pure-GPU, ~seconds).
Explores the levers v3 adds over the single-frame v1 baseline: temporal window W,
localization granularity (links vs regions), and train-label cleaning.

    python forcesense/train/sweep_v3.py --data data/wbc/datasets/wbc_train.h5 --out_dir data/wbc/sweep_v3

Score = active_acc + det_f1  (localize the contact AND don't false-alarm), so the
leaderboard is directly comparable to the v1 sweep.
"""
import os
import sys
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from forcesense.common.data import load_grid
from forcesense.train.core import run_experiment


def base(**kw):
    cfg = dict(epochs=40, batch_size=16384, lr=1e-3, eval_every=5,
               hidden=[512, 512, 256], dropout=0.0, val_envs=16)
    cfg.update(kw)
    return cfg


# Curated list. Phase 1: how much does a temporal window buy us (the core v3 bet)?
# Phase 2: granularity x label-cleaning at the best window. Phase 3: capacity.
EXPERIMENTS = [
    # --- phase 1: window sweep, link granularity, raw labels ---
    base(label="w1_links", window=1),
    base(label="w3_links", window=3),
    base(label="w6_links", window=6),
    base(label="w10_links", window=10),
    # --- phase 2: regions + label cleaning at a mid window ---
    base(label="w6_regions", window=6, regions=True),
    base(label="w6_links_dropramp", window=6, drop_ramp=True),
    base(label="w6_links_dropramp_mm0.3", window=6, drop_ramp=True, min_mag=0.3),
    base(label="w6_regions_dropramp", window=6, regions=True, drop_ramp=True),
    base(label="w6_regions_dropramp_mm0.3", window=6, regions=True, drop_ramp=True, min_mag=0.3),
    # --- phase 3: capacity on the cleaned settings ---
    base(label="w6_links_big", window=6, drop_ramp=True, min_mag=0.25,
         hidden=[768, 768, 512, 256], dropout=0.1, epochs=55),
    base(label="w6_regions_big", window=6, regions=True, drop_ramp=True,
         hidden=[768, 768, 512, 256], dropout=0.1, epochs=55),
    base(label="w10_regions_dropramp", window=10, regions=True, drop_ramp=True),
]


def score(m):
    return (m.get("active_acc", 0) or 0) + (m.get("det_f1", 0) or 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/wbc/datasets/wbc_train.h5")
    p.add_argument("--out_dir", type=str, default="data/wbc/sweep_v3")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    board_path = os.path.join(args.out_dir, "leaderboard.jsonl")
    best_path = os.path.join(args.out_dir, "force_sensor_v3_best.pt")

    print(f"[sweep-v3] loading grid from {args.data} (once) ...", flush=True)
    t_load = time.time()
    grid = load_grid(args.data, args.device)
    print(f"[sweep-v3] grid loaded T={grid['T']} E={grid['E']} D={grid['D']} "
          f"in {time.time()-t_load:.1f}s; GPU {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    best_score = -1.0
    results = []
    t0 = time.time()
    for i, exp in enumerate(EXPERIMENTS):
        exp = dict(exp, data=args.data, device=args.device,
                   out=os.path.join(args.out_dir, f"model_{exp['label']}.pt"))
        print(f"\n===== [{i+1}/{len(EXPERIMENTS)}] {exp['label']} =====", flush=True)
        try:
            m = run_experiment(exp, grid=grid)
        except Exception:
            print(f"[sweep-v3] {exp['label']} FAILED:\n{traceback.format_exc()}", flush=True)
            with open(board_path, "a") as f:
                f.write(json.dumps({"label": exp["label"], "error": True}) + "\n")
            torch.cuda.empty_cache()
            continue
        m["score"] = score(m)
        m["cfg"] = {k: v for k, v in exp.items() if k not in ("data", "device", "out")}
        results.append(m)
        with open(board_path, "a") as f:
            rec = {k: v for k, v in m.items() if k != "cm"}
            f.write(json.dumps(rec) + "\n")
        print(f"[sweep-v3] {exp['label']}: score={m['score']:.3f} act_acc={m['active_acc']:.3f} "
              f"det_f1={m['det_f1']:.3f} cos={m['force_cos']:.3f} W={m['window']} (K={m['K']})", flush=True)
        if m["score"] > best_score:
            best_score = m["score"]
            import shutil
            shutil.copy(exp["out"], best_path)
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump({k: v for k, v in m.items() if k != "cm"}, f, indent=2)
            print(f"[sweep-v3] >>> NEW BEST: {exp['label']} score={best_score:.3f} -> {best_path}", flush=True)
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n===== SWEEP-V3 DONE in {(time.time()-t0)/60:.1f} min =====", flush=True)
    print(f"{'label':<30} {'score':>6} {'act_acc':>8} {'det_f1':>7} {'cos':>6} {'W':>3} {'K':>3}", flush=True)
    for r in results:
        print(f"{r['label']:<30} {r['score']:>6.3f} {r['active_acc']:>8.3f} "
              f"{r['det_f1']:>7.3f} {r['force_cos']:>6.3f} {r['window']:>3} {r['K']:>3}", flush=True)
    print(f"\nbest model -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
