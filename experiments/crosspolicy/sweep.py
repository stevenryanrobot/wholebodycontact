"""Leave-one-out cross-controller robustness sweep for the plug-and-play sensor.

For each held-out controller, train on ALL the others (controller randomization)
and evaluate on the unseen one. Do this for feat in {proprio, resid, both}, and
report the WORST-CASE and mean held-out performance across controllers — the
honest measure of "does this plug into an unseen controller?".

    python experiments/crosspolicy/sweep.py \
        --data data/wbc/cross/cross_A_base.h5 ... --epochs 12
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.crosspolicy.base import load, train, evaluate
import torch


def combine(dss):
    T = min(d["T"] for d in dss)
    return dict(
        F3=torch.cat([d["F3"][:T] for d in dss], dim=1),
        region=torch.cat([d["region"][:T] for d in dss], dim=1),
        pos=torch.cat([d["pos"][:T] for d in dss], dim=1),
        bdim=dss[0]["bdim"], T=T, W=sum(d["W"] for d in dss),
        fmax=dss[0]["fmax"], kp=-1, path="+".join(d["path"] for d in dss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--feats", nargs="+", default=["proprio", "resid", "both"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="data/wbc/cross/sweep_result.json")
    args = ap.parse_args()

    names = [os.path.basename(p).replace("cross_", "").replace(".h5", "")
             for p in args.data]
    summary = {}
    for feat in args.feats:
        print(f"\n########## feat = {feat} ##########", flush=True)
        cache = {n: load(p, feat) for n, p in zip(names, args.data)}
        folds = []
        for held in names:
            train_ds = combine([cache[n] for n in names if n != held])
            net, mean, std = train(train_ds, args.epochs, list(range(train_ds["W"])))
            r = evaluate(net, cache[held], mean, std, cols=None)
            folds.append(dict(held=held, regAcc=r["region_acc"], actAcc=r["active_acc"],
                              detF1=r["det_f1"], prec=r["det_prec"], rec=r["det_rec"]))
            print(f"  held-out {held:10s}: regAcc={r['region_acc']:.3f} "
                  f"prec={r['det_prec']:.3f} detF1={r['det_f1']:.3f} "
                  f"actAcc={r['active_acc']:.3f}", flush=True)
            del net, train_ds
            torch.cuda.empty_cache()
        reg = [f["regAcc"] for f in folds]; pr = [f["prec"] for f in folds]
        summary[feat] = dict(folds=folds,
                             worst_regAcc=min(reg), mean_regAcc=sum(reg) / len(reg),
                             worst_prec=min(pr), mean_prec=sum(pr) / len(pr))
        del cache
        torch.cuda.empty_cache()

    print("\n==================== ROBUSTNESS SUMMARY ====================")
    print(f"{'feat':<8} {'worst_regAcc':>13} {'mean_regAcc':>12} "
          f"{'worst_prec':>11} {'mean_prec':>10}")
    for feat, s in summary.items():
        print(f"{feat:<8} {s['worst_regAcc']:13.3f} {s['mean_regAcc']:12.3f} "
              f"{s['worst_prec']:11.3f} {s['mean_prec']:10.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
