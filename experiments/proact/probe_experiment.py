"""ProACT existence proof --- does an ACTIVE probe beat the PASSIVE baseline?

Trains the *same* v3 region decoder (W=6, 5 regions) on two matched datasets:
  STATIC  : frozen policy stands still under sustained net_pull  (passive)
  WIGGLE  : same, but the robot squats (root-height oscillation) during the
            hold, sweeping the leg configuration under load               (active)

and compares per-region localization recall. The thesis: the squat changes the
contact Jacobian, so the otherwise-unobservable LEG/TRUNK contacts become
observable -> their recall rises. ARMS are the control (already observable
passively) and should move little.

    python experiments/proact/probe_experiment.py \
        --static data/wbc/probe/probe_static.h5 \
        --wiggle data/wbc/probe/probe_wiggle.h5 --n_envs 64
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from forcesense.train.core import run_experiment


def train_one(h5, n_envs, label, epochs, arch="mlp", window=6):
    cfg = dict(data=[h5], n_envs=n_envs, window=window, regions=True, epochs=epochs,
               batch_size=16384, lr=1e-3, val_envs=16, hidden=[512, 512, 256],
               eval_every=5, label=label, arch=arch)
    if arch == "gru":
        # small batches: the GPU is shared with other users' jobs (~10 GB), and
        # a 50-step GRU forward on a big batch OOMs the remaining ~13 GB.
        cfg.update(gru_hidden=384, gru_layers=2, lr=3e-4,
                   batch_size=2048, eval_bs=2048)
    return run_experiment(cfg)


def summarize(res):
    d = dict(zip(res["names"], res["recall"]))
    arm = [d[k] for k in ("left_arm", "right_arm") if k in d]
    legtrunk = [d[k] for k in ("left_leg", "right_leg", "trunk") if k in d]
    return d, sum(arm) / max(len(arm), 1), sum(legtrunk) / max(len(legtrunk), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--static", required=True)
    p.add_argument("--wiggle", required=True)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--arch", default="mlp", choices=["mlp", "gru"])
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--out", default="data/wbc/probe/probe_result.json")
    args = p.parse_args()

    print("=" * 64 + f"\nSTATIC  (passive baseline)  arch={args.arch} W={args.window}\n" + "=" * 64, flush=True)
    rs = train_one(args.static, args.n_envs, "static", args.epochs, args.arch, args.window)
    print("=" * 64 + f"\nWIGGLE  (active probe)  arch={args.arch} W={args.window}\n" + "=" * 64, flush=True)
    rw = train_one(args.wiggle, args.n_envs, "wiggle", args.epochs, args.arch, args.window)

    ds, arm_s, lt_s = summarize(rs)
    dw, arm_w, lt_w = summarize(rw)

    print("\n" + "=" * 64)
    print("ProACT RESULT --- per-region recall   (STATIC -> WIGGLE)")
    print("=" * 64)
    for n in rs["names"]:
        print(f"  {n:11s} {ds[n]:.3f} -> {dw[n]:.3f}   ({dw[n]-ds[n]:+.3f})")
    print("  " + "-" * 46)
    print(f"  {'ARMS avg':11s} {arm_s:.3f} -> {arm_w:.3f}   ({arm_w-arm_s:+.3f})   [control]")
    print(f"  {'LEG/TRUNK':11s} {lt_s:.3f} -> {lt_w:.3f}   ({lt_w-lt_s:+.3f})   [treatment]")
    print(f"\n  active_acc   {rs['active_acc']:.3f} -> {rw['active_acc']:.3f}")
    print(f"  det_f1       {rs['det_f1']:.3f} -> {rw['det_f1']:.3f}")
    verdict = ("PROBE HELPS legs/trunk" if lt_w - lt_s > 0.03
               else "no clear effect")
    print(f"\n  VERDICT: {verdict}  (leg/trunk {lt_s:.3f} -> {lt_w:.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({
        "static": {"per_region": ds, "arms_avg": arm_s, "legtrunk_avg": lt_s,
                   "active_acc": rs["active_acc"], "det_f1": rs["det_f1"]},
        "wiggle": {"per_region": dw, "arms_avg": arm_w, "legtrunk_avg": lt_w,
                   "active_acc": rw["active_acc"], "det_f1": rw["det_f1"]},
        "legtrunk_gain": lt_w - lt_s, "arms_gain": arm_w - arm_s,
    }, open(args.out, "w"), indent=2)
    print(f"\n[probe-exp] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
