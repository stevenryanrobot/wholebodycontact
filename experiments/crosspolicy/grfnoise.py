"""Tier-3: how noise-robust are the LEGS to the ground-reaction estimate?

The residual's only sim-privileged term is the GRF (qfrc_constraint). A real
robot supplies it from a foot F/T sensor / floating-base MOB, WITH some error.
We build resid_noisy = resid_real - (G + alpha * eps * std(G)) and sweep alpha:
  alpha = 0    -> perfect GRF = the privileged residual (legs ~0.89)
  alpha > 0    -> noisy foot-sensor GRF
and report per-region recall (esp. legs) vs alpha, leave-one-out over controllers.

Needs cross_grf_*.h5 (collected with --resid_mode realizable, which stores both
R = resid_real and G = the GRF projection).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import h5py
import torch
from experiments.crosspolicy.base import (link_to_region, train, valid_pairs, gather,
                                     Net, WIN, DEV, LABEL_PREFIX, POS_MAG_N)
from experiments.crosspolicy.sweep import combine
from experiments.crosspolicy.perregion import per_region_recall, RNAME


def load_grf(h5path, alpha, gstd, rng):
    """feature = resid_real - (G + alpha*eps*gstd); ds dict like load()."""
    with h5py.File(h5path, "r") as h:
        R = np.asarray(h["R"], dtype=np.float32)      # resid_real
        G = np.asarray(h["G"], dtype=np.float32)      # GRF projection
        Y = np.asarray(h["Y"], dtype=np.float32)
        W = int(h.attrs["n_envs"])
        names = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in h.attrs["body_names"]]
        fmax = float(h.attrs["force_max"])
    noise = alpha * rng.standard_normal(G.shape).astype(np.float32) * gstd
    F = R - (G + noise)                                # 29-dim
    K = len(names); T = F.shape[0] // W
    F3 = torch.from_numpy(F[:T * W].reshape(T, W, 29)).to(DEV)
    Y3 = torch.from_numpy(Y[:T * W].reshape(T, W, Y.shape[1])).to(DEV)
    l2r = torch.from_numpy(link_to_region(names)).to(DEV)
    link = Y3[:, :, LABEL_PREFIX:LABEL_PREFIX + K].argmax(-1)
    region = l2r[link]
    pos = (Y3[:, :, -1] * fmax) > POS_MAG_N
    return dict(F3=F3, bdim=29, region=region, pos=pos, T=T, W=W, fmax=fmax,
                kp=-1, path=os.path.basename(h5path))


def gstd_of(paths):
    """per-joint std of the GRF projection over the first dataset (scale ref)."""
    with h5py.File(paths[0], "r") as h:
        G = np.asarray(h["G"], dtype=np.float32)
    return G.std(0, keepdims=True) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    names = [os.path.basename(p).replace("cross_grf_", "").replace(".h5", "")
             for p in args.data]
    gstd = gstd_of(args.data)
    print(f"{'alpha':>6} " + " ".join(f"{n:>9}" for n in RNAME) +
          f"{'LEGS':>8}{'ARMS':>8}{'ALL':>8}")
    for alpha in args.alphas:
        rng = np.random.default_rng(0)
        cache = {n: load_grf(p, alpha, gstd, rng) for n, p in zip(names, args.data)}
        acc = {r: [] for r in range(5)}
        for held in names:
            tr = combine([cache[n] for n in names if n != held])
            net, mean, std = train(tr, args.epochs, list(range(tr["W"])))
            pr = per_region_recall(net, cache[held], mean, std)
            for r in range(5):
                if not np.isnan(pr[r]):
                    acc[r].append(pr[r])
            del net, tr; torch.cuda.empty_cache()
        m = {r: float(np.mean(acc[r])) for r in range(5)}
        legs = np.mean([m[2], m[3]]); arms = np.mean([m[0], m[1]])
        allm = np.mean([m[r] for r in range(5)])
        print(f"{alpha:>6.2f} " + " ".join(f"{m[r]:>9.3f}" for r in range(5)) +
              f"{legs:>8.3f}{arms:>8.3f}{allm:>8.3f}", flush=True)
        del cache; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
