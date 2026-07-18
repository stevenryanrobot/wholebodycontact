"""Whole-Body Contact — v4 sweep driver.

Loads the v4 dataset(s) ONCE onto the GPU (motion + static h5 concatenated
along time by wbc_train_v3.load_grid) and runs a small curated sweep:

  - MLP baselines (ForceSensorV3): windows {6, 10} x {links, regions}
  - GRU H=50 (ContactGRUv4):       {links, regions}
  - GRU H=50 + impedance-aware torque normalization (SixthSense Eq. 10):
                                    {links, regions}

No label cleaning (drop_ramp/min_mag hurt in the v3 sweep). Score =
active_acc + det_f1 on the NATURAL validation distribution (comparable to the
v1/v3 leaderboards); GRU runs additionally log balanced-val (bal_*) metrics.

Outputs in --out_dir:
  leaderboard.jsonl, best.json         per-run records / best summary
  model_<label>.pt                     every run's best checkpoint
  force_sensor_v4_best.pt              best overall (any arch)
  force_sensor_v4_best_mlp.pt          best MLP (deployment fallback)
  region_map.json                      link list + link->region aggregation map
                                       (web demo: 30-link probs -> region heatmap)

    python forcesense/train/sweep_v4.py --data data/wbc/datasets/wbc_v4_motion.h5 \
        data/wbc/datasets/wbc_v4_static.h5 --out_dir data/wbc/sweep_v4
"""
import os
import sys
import json
import time
import shutil
import argparse
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from forcesense.common.data import load_grid
from forcesense.train.core import run_experiment
from forcesense.common.regions import region_of


def mlp(**kw):
    cfg = dict(arch="mlp", epochs=40, batch_size=16384, lr=1e-3, eval_every=5,
               hidden=[512, 512, 256], dropout=0.0, val_envs=16)
    cfg.update(kw)
    return cfg


def gru(**kw):
    # bs 2048 x 1500 steps x 24 epochs ~ 74M windows seen (~11 passes);
    # [2048, 50, 320] batches are gathered on the fly (~130 MB each).
    cfg = dict(arch="gru", window=50, epochs=24, eval_every=3, batch_size=2048,
               steps_per_epoch=1500, lr=1e-3, gru_hidden=384, gru_layers=2,
               dropout=0.0, input_noise=0.01, val_envs=16)
    cfg.update(kw)
    return cfg


EXPERIMENTS = [
    # --- MLP baselines (v3 architecture on v4 data) ---
    mlp(label="mlp_w6_links", window=6),
    mlp(label="mlp_w10_links", window=10),
    mlp(label="mlp_w6_regions", window=6, regions=True),
    mlp(label="mlp_w10_regions", window=10, regions=True),
    # --- GRU H=50, raw torque channels ---
    gru(label="gru50_links"),
    gru(label="gru50_regions", regions=True),
    # --- GRU H=50, impedance-normalized torque channels ---
    gru(label="gru50imp_links", imp_norm=True),
    gru(label="gru50imp_regions", imp_norm=True, regions=True),
]


def score(m):
    return (m.get("active_acc", 0) or 0) + (m.get("det_f1", 0) or 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+",
                   default=["data/wbc/datasets/wbc_v4_motion.h5", "data/wbc/datasets/wbc_v4_static.h5"])
    p.add_argument("--out_dir", type=str, default="data/wbc/sweep_v4")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_envs", type=int, default=128,
                   help="Collection env count (h5 row-major reshape factor).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for pipeline validation (few epochs/steps).")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    board_path = os.path.join(args.out_dir, "leaderboard.jsonl")
    best_path = os.path.join(args.out_dir, "force_sensor_v4_best.pt")
    best_mlp_path = os.path.join(args.out_dir, "force_sensor_v4_best_mlp.pt")

    print(f"[sweep-v4] loading grid from {args.data} (once) ...", flush=True)
    t_load = time.time()
    grid = load_grid(args.data, args.device, n_envs=args.n_envs)
    print(f"[sweep-v4] grid loaded T={grid['T']} E={grid['E']} D={grid['D']} "
          f"K={grid['K']} segs={grid['seg_bounds']} in {time.time()-t_load:.1f}s; "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # link -> region aggregation map for the web demo (independent of winner)
    body_names = grid["body_names"]
    regions = sorted(set(region_of(n) for n in body_names))
    with open(os.path.join(args.out_dir, "region_map.json"), "w") as f:
        json.dump({"body_names": body_names,
                   "regions": regions,
                   "body_to_region": {n: region_of(n) for n in body_names},
                   "region_index": {r: i for i, r in enumerate(regions)}}, f, indent=2)
    print(f"[sweep-v4] region_map.json written ({len(body_names)} links -> {regions})", flush=True)

    experiments = EXPERIMENTS
    if args.smoke:
        experiments = [dict(e, epochs=2, eval_every=1, val_envs=max(args.n_envs // 4, 1),
                            **({"steps_per_epoch": 30, "batch_size": 512} if e["arch"] == "gru" else {}))
                       for e in EXPERIMENTS]

    best_score, best_mlp_score = -1.0, -1.0
    results = []
    t0 = time.time()
    for i, exp in enumerate(experiments):
        exp = dict(exp, data=args.data, device=args.device,
                   out=os.path.join(args.out_dir, f"model_{exp['label']}.pt"))
        print(f"\n===== [{i+1}/{len(experiments)}] {exp['label']} =====", flush=True)
        try:
            m = run_experiment(exp, grid=grid)
        except Exception:
            print(f"[sweep-v4] {exp['label']} FAILED:\n{traceback.format_exc()}", flush=True)
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
        print(f"[sweep-v4] {exp['label']}: score={m['score']:.3f} act_acc={m['active_acc']:.3f} "
              f"det_f1={m['det_f1']:.3f} cos={m['force_cos']:.3f} "
              f"arch={m['arch']} W={m['window']} K={m['K']}", flush=True)
        if m["score"] > best_score:
            best_score = m["score"]
            shutil.copy(exp["out"], best_path)
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump({k: v for k, v in m.items() if k != "cm"}, f, indent=2)
            print(f"[sweep-v4] >>> NEW BEST: {exp['label']} score={best_score:.3f} -> {best_path}", flush=True)
        if m["arch"] == "mlp" and m["score"] > best_mlp_score:
            best_mlp_score = m["score"]
            shutil.copy(exp["out"], best_mlp_path)
            print(f"[sweep-v4] >>> NEW BEST MLP: {exp['label']} -> {best_mlp_path}", flush=True)
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n===== SWEEP-V4 DONE in {(time.time()-t0)/60:.1f} min =====", flush=True)
    print(f"{'label':<26} {'score':>6} {'act_acc':>8} {'det_f1':>7} {'cos':>6} "
          f"{'mag_err':>7} {'bal_f1':>7} {'W':>3} {'K':>3}", flush=True)
    for r in results:
        print(f"{r['label']:<26} {r['score']:>6.3f} {r['active_acc']:>8.3f} "
              f"{r['det_f1']:>7.3f} {r['force_cos']:>6.3f} {r['mag_rel_err']:>7.2f} "
              f"{r.get('bal_det_f1', float('nan')):>7.3f} {r['window']:>3} {r['K']:>3}", flush=True)
    print(f"\nbest model -> {best_path}\nbest MLP  -> {best_mlp_path}", flush=True)


if __name__ == "__main__":
    main()
