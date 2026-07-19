"""Position-based vs current-based force sensing (Task 1).

Our force sensor's proprioception includes the applied joint TORQUE (channels
[0:29]). On real hardware that torque is only available via motor current
(tau ~= K_t * i, noisy). Can we instead reconstruct it from POSITION (the PD law,
needs no current) with little accuracy loss?

Three input variants, same everything else, compared on the same data:
  A_measured    raw applied_torque (the current proxy)         -- baseline
  B_pd_position torque = kp*(q_des-q) - kd*qd                   -- position only, NO current
  C_none        torque channels zeroed (pure kinematics)        -- net learns implicitly

Channels (forcesense/common/data.py IMP_LAYOUT): torque[0:29], jp0=q-qdef[29:58],
jp1[58:87], q_des(abs)[174:203]; q_err = q_des - q = X[q_des]-qdef-X[jp0].

Memory: variants are applied IN PLACE on the loaded grid, in the order
measured -> pd -> none. pd/none only READ the position channels and WRITE the
torque channels, so no copy of the (multi-GB) grid is needed.

Usage:
  python experiments/posforce/run.py --data data/wbc/datasets/wbc_v4_motion.h5 \
      data/wbc/datasets/wbc_v4_static.h5 --epochs 30 --out data/wbc/posforce
"""
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from forcesense.common.data import load_grid, impedance_constants, IMP_LAYOUT, CTRL_DT
from forcesense.train.core import run_experiment

TQ = IMP_LAYOUT["torque"]; JP0 = IMP_LAYOUT["jp0"]; JP1 = IMP_LAYOUT["jp1"]; QD = IMP_LAYOUT["q_des"]


def set_torque(X, mode, kp, kd, qdef):
    """Mutate the torque channels of X [T,E,D] in place (positions untouched)."""
    if mode == "measured":
        return                                            # leave original torque
    if mode == "none":
        X[..., TQ[0]:TQ[1]] = 0.0
        return
    if mode == "pd":                                      # reconstruct from position
        q_err = X[..., QD[0]:QD[1]] - qdef - X[..., JP0[0]:JP0[1]]
        qd = (X[..., JP0[0]:JP0[1]] - X[..., JP1[0]:JP1[1]]) / CTRL_DT
        X[..., TQ[0]:TQ[1]] = kp * q_err - kd * qd
        return
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+",
                   default=["data/wbc/datasets/wbc_v4_motion.h5",
                            "data/wbc/datasets/wbc_v4_static.h5"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--out", type=str, default="data/wbc/posforce")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    data = args.data if len(args.data) > 1 else args.data[0]

    grid = load_grid(data, dev)
    X = grid["X"]                                         # mutated in place per variant
    kp, kd, qdef = impedance_constants(dev)

    base_cfg = dict(window=args.window, regions=True, arch="mlp", epochs=args.epochs,
                    imp_norm=False, device=dev, val_envs=16, eval_every=5,
                    hidden=[512, 512, 256], lr=1e-3, batch_size=16384)

    modes = [("A_measured", "measured"), ("B_pd_position", "pd"), ("C_none", "none")]
    results = {}
    for label, mode in modes:
        print(f"\n=========== variant {label} (torque={mode}) ===========", flush=True)
        set_torque(X, mode, kp, kd, qdef)                # in-place; order matters
        cfg = {**base_cfg, "label": label,
               "out": os.path.join(args.out, f"sensor_{label}.pt")}
        m = run_experiment(cfg, grid=grid)
        keep = {k: m[k] for k in ("active_acc", "det_f1", "det_prec", "det_rec",
                                  "overall_acc", "force_cos", "mag_rel_err", "K")
                if k in m}
        keep["region_recall"] = {n: round(r, 3) for n, r in zip(m.get("names", []), m.get("recall", []))}
        results[label] = keep
        json.dump(results, open(os.path.join(args.out, "compare.json"), "w"), indent=2)
        print(f"[{label}] act_acc={m['active_acc']:.3f} det_f1={m['det_f1']:.3f} "
              f"prec={m['det_prec']:.3f} cos={m['force_cos']:.3f}", flush=True)

    print("\n\n================ POSITION vs CURRENT force sensing ================")
    print(f"{'variant':16s} {'act_acc':>8s} {'det_f1':>8s} {'det_prec':>9s} {'force_cos':>10s}")
    for label, _ in modes:
        r = results[label]
        print(f"{label:16s} {r['active_acc']:8.3f} {r['det_f1']:8.3f} "
              f"{r['det_prec']:9.3f} {r['force_cos']:10.3f}")
    print("\nper-region recall:")
    for label, _ in modes:
        print(f"  {label:16s} {results[label]['region_recall']}")
    a, b = results["A_measured"], results["B_pd_position"]
    print(f"\nGAP A(measured/current) - B(PD/position):  act_acc {a['active_acc']-b['active_acc']:+.3f}  "
          f"det_f1 {a['det_f1']-b['det_f1']:+.3f}  force_cos {a['force_cos']-b['force_cos']:+.3f}")
    print("small gap => can drop current sensing, reconstruct torque from position.")
    print("wrote ->", os.path.join(args.out, "compare.json"))


if __name__ == "__main__":
    main()
