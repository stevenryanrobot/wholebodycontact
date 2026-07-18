"""Whole-Body Contact (Plan A) — autonomous training sweep.

Runs a curated list of experiments over the existing dataset, exploring the levers
that matter (localization granularity, class weighting, label filtering, arch/hp),
on an HONEST env-held-out split. Appends every result to a leaderboard and keeps the
best model. Designed to run unattended in tmux.

    python scripts/wbc_sweep.py --data data/wbc/wbc_train.h5 --out_dir data/wbc/sweep

Score = active_acc + det_f1  (want both: localize the contact AND not false-alarm).
"""
# ARCHIVED v1 — superseded by forcesense/ (kept for reference; not maintained).
import os
import json
import time
import argparse
import traceback

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.wbc_train import run_experiment


def base(**kw):
    cfg = dict(epochs=40, batch_size=8192, lr=1e-3, eval_every=5,
               hidden=[512, 512, 256], dropout=0.0, val_envs=16, collect_envs=128)
    cfg.update(kw)
    return cfg


# Curated experiment list. Phase 1: granularity x weighting x label-cleaning.
# Phase 2: arch/hp on the promising settings.
EXPERIMENTS = [
    # --- phase 1: links vs regions, weighting, ramp filtering ---
    base(label="links_raw"),
    base(label="links_nw0.3", none_weight=0.3),
    base(label="links_nw0.3_droptamp", none_weight=0.3, drop_ramp=True),
    base(label="regions_raw", regions=True),
    base(label="regions_nw0.3", regions=True, none_weight=0.3),
    base(label="regions_nw0.3_dropramp", regions=True, none_weight=0.3, drop_ramp=True),
    base(label="regions_nw0.5_dropramp", regions=True, none_weight=0.5, drop_ramp=True),
    base(label="regions_dropramp_minmag0.3", regions=True, none_weight=0.4, drop_ramp=True, min_mag=0.3),
    base(label="links_nw0.4_dropramp_minmag0.3", none_weight=0.4, drop_ramp=True, min_mag=0.3),
    # --- phase 2: arch / hp on the cleaned settings ---
    base(label="regions_big", regions=True, none_weight=0.4, drop_ramp=True,
         hidden=[768, 768, 512, 256], dropout=0.1, epochs=60),
    base(label="regions_lrlow", regions=True, none_weight=0.4, drop_ramp=True,
         lr=5e-4, epochs=70),
    base(label="links_big", none_weight=0.4, drop_ramp=True, min_mag=0.25,
         hidden=[768, 768, 512, 256], dropout=0.1, epochs=60),
    base(label="regions_minmag0.4_big", regions=True, none_weight=0.4, drop_ramp=True, min_mag=0.4,
         hidden=[1024, 768, 512], dropout=0.1, epochs=70),
    base(label="links_nw0.5_big", none_weight=0.5, drop_ramp=True,
         hidden=[768, 768, 512, 256], dropout=0.1, epochs=60),
]


def score(m):
    a = m.get("active_acc", 0) or 0
    f = m.get("det_f1", 0) or 0
    return a + f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/wbc/wbc_train.h5")
    p.add_argument("--out_dir", type=str, default="data/wbc/sweep")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    board_path = os.path.join(args.out_dir, "leaderboard.jsonl")
    best_path = os.path.join(args.out_dir, "force_sensor_best.pt")

    best_score = -1
    results = []
    t0 = time.time()
    for i, exp in enumerate(EXPERIMENTS):
        exp = dict(exp, data=args.data, device=args.device,
                   out=os.path.join(args.out_dir, f"model_{exp['label']}.pt"))
        print(f"\n===== [{i+1}/{len(EXPERIMENTS)}] {exp['label']} =====", flush=True)
        try:
            m = run_experiment(exp)
        except Exception:
            print(f"[sweep] {exp['label']} FAILED:\n{traceback.format_exc()}", flush=True)
            with open(board_path, "a") as f:
                f.write(json.dumps({"label": exp["label"], "error": True}) + "\n")
            continue
        m["score"] = score(m)
        m["cfg"] = {k: v for k, v in exp.items() if k not in ("data", "device", "out")}
        results.append(m)
        with open(board_path, "a") as f:
            rec = {k: v for k, v in m.items() if k != "cm"}
            f.write(json.dumps(rec) + "\n")
        print(f"[sweep] {exp['label']}: score={m['score']:.3f} act_acc={m['active_acc']:.3f} "
              f"det_f1={m['det_f1']:.3f} cos={m['force_cos']:.3f} (K={m['K']})", flush=True)
        if m["score"] > best_score:
            best_score = m["score"]
            import shutil
            shutil.copy(exp["out"], best_path)
            with open(os.path.join(args.out_dir, "best.json"), "w") as f:
                json.dump({k: v for k, v in m.items() if k != "cm"}, f, indent=2)
            print(f"[sweep] >>> NEW BEST: {exp['label']} score={best_score:.3f} -> {best_path}", flush=True)

    # final ranking
    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n===== SWEEP DONE in {(time.time()-t0)/60:.1f} min =====", flush=True)
    print(f"{'label':<32} {'score':>6} {'act_acc':>8} {'det_f1':>7} {'cos':>6} {'K':>3}", flush=True)
    for r in results:
        print(f"{r['label']:<32} {r['score']:>6.3f} {r['active_acc']:>8.3f} "
              f"{r['det_f1']:>7.3f} {r['force_cos']:>6.3f} {r['K']:>3}", flush=True)
    print(f"\nbest model -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
