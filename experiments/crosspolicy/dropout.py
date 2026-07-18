"""Optimization: 'both' feature with PROPRIO-CHANNEL DROPOUT.

'both' (proprio+resid) has the highest in-domain accuracy but re-inherits the
proprio shortcut and degrades on extreme unseen controllers (worst regAcc 0.78,
prec 0.67). Fix: during training, randomly zero the proprio part of each frame
with prob p, so the net must also work from the controller-invariant residual
alone -> keep 'both' peak AND resid robustness.

Leave-one-out over all controllers; reports worst/mean like the sweep.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
import torch.nn as nn
from experiments.crosspolicy.base import (load, evaluate, valid_pairs, gather, Net,
                                     WIN, DEV)
from experiments.crosspolicy.sweep import combine

PROPRIO_DIM = 320   # X width; resid is the remaining 29 of the 349 'both' frame


def train_drop(ds, epochs, p_drop, proprio_dim=PROPRIO_DIM):
    t, w = valid_pairs(ds["T"], ds["W"], list(range(ds["W"])))
    idx = torch.randperm(t.shape[0], device=DEV)[:50000]
    sample = gather(ds["F3"], t[idx], w[idx])
    mean = sample.mean(0, keepdim=True); std = sample.std(0, keepdim=True) + 1e-6
    reg_t = ds["region"][t, w]; pos_t = ds["pos"][t, w].float()
    pos_w = ((pos_t.numel() - pos_t.sum()) / (pos_t.sum() + 1e-9)).clamp(1, 20)
    bdim = ds["bdim"]
    # mask of the proprio positions within the flattened WIN*bdim vector
    proprio_mask = torch.zeros(WIN * bdim, device=DEV, dtype=torch.bool)
    for f in range(WIN):
        proprio_mask[f * bdim: f * bdim + proprio_dim] = True
    net = Net(WIN * bdim).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w); ce = nn.CrossEntropyLoss()
    B = 8192
    for ep in range(epochs):
        net.train(); perm = torch.randperm(t.shape[0], device=DEV)
        for i in range(0, t.shape[0], B):
            b = perm[i:i + B]
            feat = gather(ds["F3"], t[b], w[b], mean, std)
            # per-sample: with prob p_drop, zero the proprio channels
            drop = (torch.rand(feat.shape[0], 1, device=DEV) < p_drop) & proprio_mask
            feat = feat.masked_fill(drop, 0.0)
            dlog, rlog = net(feat)
            ld = bce(dlog, pos_t[b])
            pm = pos_t[b] > 0.5
            lr = ce(rlog[pm], reg_t[b][pm]) if pm.any() else 0.0 * dlog.sum()
            (ld + lr).backward(); opt.step(); opt.zero_grad()
    return net, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--p_drop", type=float, default=0.5)
    ap.add_argument("--out", default="data/wbc/cross/dropout_result.json")
    args = ap.parse_args()
    names = [os.path.basename(p).replace("cross_", "").replace(".h5", "")
             for p in args.data]
    cache = {n: load(p, "both") for n, p in zip(names, args.data)}
    print(f"########## both + proprio_dropout p={args.p_drop} ##########", flush=True)
    folds = []
    for held in names:
        tr = combine([cache[n] for n in names if n != held])
        net, mean, std = train_drop(tr, args.epochs, args.p_drop)
        r = evaluate(net, cache[held], mean, std, cols=None)
        folds.append(dict(held=held, regAcc=r["region_acc"], prec=r["det_prec"],
                          detF1=r["det_f1"], actAcc=r["active_acc"]))
        print(f"  held-out {held:10s}: regAcc={r['region_acc']:.3f} "
              f"prec={r['det_prec']:.3f} detF1={r['det_f1']:.3f}", flush=True)
        del net, tr; torch.cuda.empty_cache()
    reg = [f["regAcc"] for f in folds]; pr = [f["prec"] for f in folds]
    print(f"\nboth+dropout p={args.p_drop}: worst_regAcc={min(reg):.3f} "
          f"mean_regAcc={sum(reg)/len(reg):.3f} worst_prec={min(pr):.3f} "
          f"mean_prec={sum(pr)/len(pr):.3f}", flush=True)
    json.dump(dict(p_drop=args.p_drop, folds=folds, worst_regAcc=min(reg),
                   mean_regAcc=sum(reg)/len(reg), worst_prec=min(pr),
                   mean_prec=sum(pr)/len(pr)), open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
