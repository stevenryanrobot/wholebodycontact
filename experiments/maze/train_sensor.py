"""Train the v4 contact sensor (GRU H=50) on maze random-walk data.

    .../python experiments/maze/train_sensor.py --data experiments/maze/data/sensor_*.h5 \
        --out experiments/maze/data/contact_gru.pt

Honest eval on held-out envs: detection P/R/F1, azimuth error (deg) on contact
frames, 12-bin accuracy, and the same after a k=3 debounce (what nav sees).
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.maze.sensor_model import ContactGRU


def load_grids(patterns, device):
    files = sorted(sum([glob.glob(p) for p in patterns], []))
    assert files, f"no files match {patterns}"
    grids = []
    for f in files:
        with h5py.File(f, "r") as h:
            g = {
                "X": torch.from_numpy(h["X"][:]).to(device),          # [T,E,D]
                "c": torch.from_numpy(h["contact"][:]).to(device),
                "az": torch.from_numpy(h["az_r"][:]).to(device),
                "mag": torch.from_numpy(h["mag"][:]).to(device),
                "rst": torch.from_numpy(h["reset"][:]).to(device),
            }
        grids.append(g)
        print(f"[data] {f}: T={g['X'].shape[0]} E={g['X'].shape[1]} "
              f"contact_rate={g['c'].float().mean():.3f}")
    return grids


def build_index(grids, H, val_frac=0.25):
    """(gi, t, e) tuples with a clean window (no reset inside)."""
    tr, va = [], []
    for gi, g in enumerate(grids):
        T, E = g["c"].shape
        # windows ending at t are invalid if any reset in (t-H+1..t]
        bad = g["rst"].float()
        csum = torch.cumsum(bad, dim=0)
        n_val = max(1, int(E * val_frac))
        for e in range(E):
            valid_t = torch.arange(H, T, device=bad.device)
            w_bad = csum[valid_t, e] - csum[valid_t - H, e]
            ok_t = valid_t[w_bad == 0]
            idx = torch.stack([torch.full_like(ok_t, gi), ok_t,
                               torch.full_like(ok_t, e)], -1)
            (va if e >= E - n_val else tr).append(idx)
    empty = torch.zeros(0, 3, dtype=torch.long)
    return (torch.cat(tr) if tr else empty), (torch.cat(va) if va else empty)


def gather(grids, idx, H):
    """idx [B,3] -> X [B,H,D], labels."""
    gi = idx[:, 0]
    out_x, out_c, out_az, out_m = [], [], [], []
    for g_id in gi.unique():
        m = gi == g_id
        g = grids[int(g_id)]
        t, e = idx[m, 1], idx[m, 2]
        steps = torch.arange(-H + 1, 1, device=t.device)
        tt = t.unsqueeze(1) + steps.unsqueeze(0)                  # [b,H]
        out_x.append(g["X"][tt, e.unsqueeze(1).expand_as(tt)])    # [b,H,D]
        out_c.append(g["c"][t, e]); out_az.append(g["az"][t, e]); out_m.append(g["mag"][t, e])
    return (torch.cat(out_x), torch.cat(out_c), torch.cat(out_az), torch.cat(out_m))


@torch.no_grad()
def evaluate(model, grids, idx, H, mean, std, bs=8192, det_thresh=0.5):
    model.eval()
    tp = fp = fn = tn = 0
    az_err, nact = 0.0, 0
    bin_ok = 0
    for s in range(0, idx.shape[0], bs):
        x, c, az, mg = gather(grids, idx[s:s + bs], H)
        det, azp, magp = model((x - mean) / std)
        p = torch.sigmoid(det) > det_thresh
        tp += int((p & c).sum()); fp += int((p & ~c).sum())
        fn += int((~p & c).sum()); tn += int((~p & ~c).sum())
        if c.any():
            az_hat = torch.atan2(azp[c][:, 0], azp[c][:, 1])
            d = torch.atan2(torch.sin(az_hat - az[c]), torch.cos(az_hat - az[c])).abs()
            az_err += float(torch.rad2deg(d).sum()); nact += int(c.sum())
            b_true = ((torch.rad2deg(az[c]) % 360 + 15) % 360 // 30).long()
            b_hat = ((torch.rad2deg(az_hat) % 360 + 15) % 360 // 30).long()
            bin_ok += int((b_true == b_hat).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"prec": prec, "rec": rec, "f1": f1,
            "az_deg": az_err / max(nact, 1), "bin_acc": bin_ok / max(nact, 1),
            "fa_rate": fp / max(fp + tn, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="experiments/maze/data/contact_gru.pt")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.01, help="train-time input noise (post-norm)")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    H = args.window

    grids = load_grids(args.data, dev)
    tr_idx, va_idx = build_index(grids, H)
    print(f"[data] train windows {tr_idx.shape[0]}  val windows {va_idx.shape[0]}")

    # feature normalization from a train subsample
    sub = tr_idx[torch.randperm(tr_idx.shape[0], device=dev)[:100_000]]
    xs, cs, _, _ = gather(grids, sub, H)
    mean = xs.reshape(-1, xs.shape[-1]).mean(0)
    std = xs.reshape(-1, xs.shape[-1]).std(0) + 1e-5
    pos_rate = float(cs.float().mean())
    pos_w = torch.tensor([min((1 - pos_rate) / max(pos_rate, 1e-3), 8.0)], device=dev)
    print(f"[train] contact rate {pos_rate:.3f}  pos_weight {float(pos_w):.2f}")

    model = ContactGRU(xs.shape[-1], args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    n = tr_idx.shape[0]

    best = {"f1": -1}
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, args.bs):
            x, c, az, mg = gather(grids, tr_idx[perm[s:s + args.bs]], H)
            x = (x - mean) / std
            if args.noise > 0:
                x = x + torch.randn_like(x) * args.noise
            det, azp, magp = model(x)
            l_det = F.binary_cross_entropy_with_logits(det, c.float(), pos_weight=pos_w)
            if c.any():
                target = torch.stack([torch.sin(az[c]), torch.cos(az[c])], -1)
                l_az = F.mse_loss(azp[c], target)
                l_mag = F.smooth_l1_loss(magp[c], mg[c] / 50.0)      # scale ~50N
            else:
                l_az = l_mag = torch.zeros((), device=dev)
            loss = l_det + 2.0 * l_az + 0.5 * l_mag
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * x.shape[0]
        m = evaluate(model, grids, va_idx, H, mean, std)
        print(f"[ep{ep}] loss {tot/n:.4f} | det P {m['prec']:.3f} R {m['rec']:.3f} "
              f"F1 {m['f1']:.3f} FA {m['fa_rate']:.3f} | az {m['az_deg']:.1f}° "
              f"bin12 {m['bin_acc']:.3f}", flush=True)
        if m["f1"] > best["f1"]:
            best = {**m, "ep": ep}
            torch.save({"state_dict": model.state_dict(), "in_dim": xs.shape[-1],
                        "hidden": args.hidden, "layers": args.layers, "window": H,
                        "x_mean": mean.cpu(), "x_std": std.cpu(), "det_thresh": 0.5,
                        "metrics": m}, args.out)
    print(f"[train] best ep{best['ep']}: F1 {best['f1']:.3f} az {best['az_deg']:.1f}° "
          f"bin12 {best['bin_acc']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
