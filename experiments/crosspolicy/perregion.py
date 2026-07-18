"""Per-region recall of the cross-controller sensor — how good are the LEGS?

The sweep's regAcc is a 5-region average that can hide weak legs. Here we break
recall down per region, in the plug-and-play setting (leave-one-out: train on 6
controllers, eval on the unseen one), averaged over all 7 held-outs.

Region idx: 0 left_arm, 1 right_arm, 2 left_leg, 3 right_leg, 4 trunk.
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from experiments.crosspolicy.base import load, train, valid_pairs, gather, DEV
from experiments.crosspolicy.sweep import combine

RNAME = ["left_arm", "right_arm", "left_leg", "right_leg", "trunk"]


@torch.no_grad()
def per_region_recall(net, ds, mean, std, batch=16384):
    net.eval()
    t, w = valid_pairs(ds["T"], ds["W"], None)
    reg_t = ds["region"][t, w]; pos = ds["pos"][t, w]
    pred = torch.empty(t.shape[0], dtype=torch.long, device=DEV)
    for i in range(0, t.shape[0], batch):
        sl = slice(i, i + batch)
        feat = gather(ds["F3"], t[sl], w[sl], mean, std)
        _, rlog = net(feat)
        pred[sl] = rlog.argmax(-1)
    out = {}
    for r in range(5):
        m = pos & (reg_t == r)
        out[r] = (pred[m] == r).float().mean().item() if m.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--feats", nargs="+", default=["proprio", "resid"])
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    names = [os.path.basename(p).replace("cross_", "").replace(".h5", "")
             for p in args.data]
    print(f"{'feat':<8} " + " ".join(f"{n:>10}" for n in RNAME) +
          f"{'LEGS':>8}{'ARMS':>8}")
    for feat in args.feats:
        cache = {n: load(p, feat) for n, p in zip(names, args.data)}
        acc = {r: [] for r in range(5)}
        for held in names:
            tr = combine([cache[n] for n in names if n != held])
            net, mean, std = train(tr, args.epochs, list(range(tr["W"])))
            pr = per_region_recall(net, cache[held], mean, std)
            for r in range(5):
                if not np.isnan(pr[r]):
                    acc[r].append(pr[r])
            del net, tr; torch.cuda.empty_cache()
        mean_r = {r: float(np.mean(acc[r])) for r in range(5)}
        legs = np.mean([mean_r[2], mean_r[3]]); arms = np.mean([mean_r[0], mean_r[1]])
        print(f"{feat:<8} " + " ".join(f"{mean_r[r]:>10.3f}" for r in range(5)) +
              f"{legs:>8.3f}{arms:>8.3f}", flush=True)
        del cache; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
