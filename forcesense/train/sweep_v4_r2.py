"""Round-2 Track B: GRU recipe-fix retrains (regions granularity).

Round-1 finding: GRU loc/dir heads DEGRADE over epochs (act 0.12->0.09,
cos 0.23->0.15 on links) while detection improves — the 1:4 pos:neg batch
sampling skews/starves the loc head, and lr 1e-3 is likely too hot for the
shared trunk. Variants (all lr 3e-4, 60 epochs, model selection by the
DEPLOYMENT metric = debounced honest F1 on val, wbc_deploy_metrics):

  1. gru50r2_nat_regions : natural-distribution sampling (no 1:4).
  2. gru50r2_sep_regions : keep 1:4 for the det head, but loc/dir/mag are
     trained on their OWN uniformly-sampled true-contact batch each step.
  3. gru50r2_sep_big_regions : variant 2 with hidden 512.

Appends to the round-1 leaderboard (data/wbc/sweep_v4/leaderboard.jsonl).

    python forcesense/train/sweep_v4_r2.py --data data/wbc/datasets/wbc_v4_motion.h5 \
        data/wbc/datasets/wbc_v4_static.h5 --out_dir data/wbc/sweep_v4
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


def gru_r2(**kw):
    cfg = dict(arch="gru", window=50, regions=True, epochs=60, eval_every=3,
               batch_size=2048, steps_per_epoch=1500, lr=3e-4,
               gru_hidden=384, gru_layers=2, dropout=0.0, input_noise=0.01,
               val_envs=16, deploy_select=True)
    cfg.update(kw)
    return cfg


EXPERIMENTS = [
    gru_r2(label="gru50r2_nat_regions", pn_sample=False),
    gru_r2(label="gru50r2_sep_regions", pn_sample=True, sep_loc_batch=True),
    gru_r2(label="gru50r2_sep_big_regions", pn_sample=True, sep_loc_batch=True,
           gru_hidden=512),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+",
                   default=["data/wbc/datasets/wbc_v4_motion.h5", "data/wbc/datasets/wbc_v4_static.h5"])
    p.add_argument("--out_dir", type=str, default="data/wbc/sweep_v4")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_envs", type=int, default=128)
    args = p.parse_args()
    board_path = os.path.join(args.out_dir, "leaderboard.jsonl")

    print(f"[sweep-r2] loading grid from {args.data} (once) ...", flush=True)
    grid = load_grid(args.data, args.device, n_envs=args.n_envs)
    print(f"[sweep-r2] grid T={grid['T']} E={grid['E']} K={grid['K']}; "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    t0 = time.time()
    for i, exp in enumerate(EXPERIMENTS):
        exp = dict(exp, data=args.data, device=args.device,
                   out=os.path.join(args.out_dir, f"model_{exp['label']}.pt"))
        print(f"\n===== [r2 {i+1}/{len(EXPERIMENTS)}] {exp['label']} =====", flush=True)
        try:
            m = run_experiment(exp, grid=grid)
        except Exception:
            print(f"[sweep-r2] {exp['label']} FAILED:\n{traceback.format_exc()}", flush=True)
            with open(board_path, "a") as f:
                f.write(json.dumps({"label": exp["label"], "error": True}) + "\n")
            torch.cuda.empty_cache()
            continue
        m["cfg"] = {k: v for k, v in exp.items() if k not in ("data", "device", "out")}
        with open(board_path, "a") as f:
            f.write(json.dumps({k: v for k, v in m.items() if k != "cm"}) + "\n")
        print(f"[sweep-r2] {exp['label']}: deployF1={m.get('deploy_f1', float('nan')):.3f} "
              f"act_acc={m['active_acc']:.3f} det_f1={m['det_f1']:.3f} "
              f"cos={m['force_cos']:.3f} (selected ep{m.get('epoch','?')})", flush=True)
        torch.cuda.empty_cache()
    print(f"\n===== SWEEP-R2 DONE in {(time.time()-t0)/60:.1f} min =====", flush=True)


if __name__ == "__main__":
    main()
