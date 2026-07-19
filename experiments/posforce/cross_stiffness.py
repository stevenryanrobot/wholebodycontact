"""Task 1 extension: does position-reconstructed torque hold across controller
stiffness? (Physics intuition: a stiffer controller -> a given contact force
produces a SMALLER joint deviation, so reconstructing torque from position
should be harder. In noise-free sim the reconstruction uses the known scaled
gains so it may still hold; --pos_noise adds encoder noise to expose the
real-hardware degradation.)

Per controller dataset (data/wbc/cross/cross_*.h5, each with its kp_scale/
kd_scale attr), train:
  A  measured torque              (current proxy)
  B  PD reconstruction from pos    torque = (kp*s_kp)*(q_des-q) - (kd*s_kd)*qd
and report the A-B detection-F1 gap vs stiffness.

Usage:
  python experiments/posforce/cross_stiffness.py --epochs 20 --pos_noise 0.0
"""
import os
import sys
import json
import argparse

import h5py
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from forcesense.common.data import load_grid, impedance_constants, IMP_LAYOUT, CTRL_DT
from forcesense.train.core import run_experiment

TQ = IMP_LAYOUT["torque"]; JP0 = IMP_LAYOUT["jp0"]; JP1 = IMP_LAYOUT["jp1"]; QD = IMP_LAYOUT["q_des"]

CTRLS = [("D_softud", "cross_D_softud.h5"), ("B_soft", "cross_B_soft.h5"),
         ("A_base", "cross_A_base.h5"), ("C_stiff", "cross_C_stiff.h5"),
         ("E_vstiff", "cross_E_vstiff.h5")]


def pd_torque(X, kp, kd, qdef, pos_noise=0.0):
    jp0 = X[..., JP0[0]:JP0[1]]
    jp1 = X[..., JP1[0]:JP1[1]]
    qdes = X[..., QD[0]:QD[1]]
    if pos_noise > 0:
        jp0 = jp0 + torch.randn_like(jp0) * pos_noise
        jp1 = jp1 + torch.randn_like(jp1) * pos_noise
    q_err = qdes - qdef - jp0
    qd = (jp0 - jp1) / CTRL_DT
    return kp * q_err - kd * qd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--pos_noise", type=float, default=0.0, help="std of gaussian noise on joint pos (rad)")
    p.add_argument("--out", type=str, default="data/wbc/posforce")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    kp0, kd0, qdef = impedance_constants(dev)
    base_cfg = dict(window=args.window, regions=True, arch="mlp", epochs=args.epochs,
                    imp_norm=False, device=dev, val_envs=16, eval_every=5,
                    hidden=[512, 512, 256], lr=1e-3, batch_size=16384)
    tag = f"_noise{args.pos_noise}" if args.pos_noise > 0 else ""
    outp = os.path.join(args.out, f"cross_stiffness{tag}.json")
    rows = {}
    for label, fname in CTRLS:
        path = os.path.join("data/wbc/cross", fname)
        if not os.path.exists(path):
            print("skip missing", path); continue
        with h5py.File(path, "r") as h:
            s_kp = float(h.attrs.get("kp_scale", 1.0)); s_kd = float(h.attrs.get("kd_scale", 1.0))
        kp, kd = kp0 * s_kp, kd0 * s_kd
        grid = load_grid(path, dev)
        X = grid["X"]
        tq_orig = X[..., TQ[0]:TQ[1]].clone()             # keep measured torque
        res = {"kp_scale": s_kp, "kd_scale": s_kd}
        # A measured
        X[..., TQ[0]:TQ[1]] = tq_orig
        mA = run_experiment({**base_cfg, "label": f"{label}_A"}, grid=grid)
        # B pd reconstruction (correct scaled gains, optional pos noise)
        X[..., TQ[0]:TQ[1]] = pd_torque(X, kp, kd, qdef, args.pos_noise)
        mB = run_experiment({**base_cfg, "label": f"{label}_B"}, grid=grid)
        res.update(A_det_f1=round(mA["det_f1"], 3), B_det_f1=round(mB["det_f1"], 3),
                   A_act=round(mA["active_acc"], 3), B_act=round(mB["active_acc"], 3),
                   gap_det_f1=round(mA["det_f1"] - mB["det_f1"], 3),
                   gap_act=round(mA["active_acc"] - mB["active_acc"], 3))
        rows[label] = res
        json.dump(rows, open(outp, "w"), indent=2)
        del grid, X, tq_orig; torch.cuda.empty_cache()
        print(f"[{label}] kp_scale={s_kp} A_f1={res['A_det_f1']} B_f1={res['B_det_f1']} "
              f"gap={res['gap_det_f1']:+.3f}", flush=True)

    print(f"\n===== POSITION-reconstruct vs MEASURED torque across stiffness (pos_noise={args.pos_noise}) =====")
    print(f"{'controller':12s} {'kp_scale':>8s} {'A(meas)_f1':>10s} {'B(pos)_f1':>10s} {'gap A-B':>8s}")
    for label, _ in CTRLS:
        if label in rows:
            r = rows[label]
            print(f"{label:12s} {r['kp_scale']:8.2f} {r['A_det_f1']:10.3f} {r['B_det_f1']:10.3f} {r['gap_det_f1']:+8.3f}")
    print("larger gap at higher kp => position estimate degrades for stiff controllers")
    print("wrote ->", outp)


if __name__ == "__main__":
    main()
