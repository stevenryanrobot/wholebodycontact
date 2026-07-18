"""Round-2 Track A: post-hoc DEPLOYMENT-semantics re-evaluation of v4 sensors.

Re-evaluates checkpoints on the held-out val envs with what the demo actually
perceives (see forcesense/common/metrics.py):
  - honest labels (pos = |F| > 5 N; sub-dead-zone contact and 1 s post-release
    ring-down excluded from negatives),
  - detection-threshold sweep (0.30..0.95),
  - temporal debouncing (k in {3,5} persistence + hysteresis th_lo=th_hi-0.15),
  - region accuracy during confirmed-contact frames,
  - motion vs static split (locates the false-positive source).

    python forcesense/eval/eval_deploy.py \
        --data data/wbc/datasets/wbc_v4_motion.h5 data/wbc/datasets/wbc_v4_static.h5 \
        --ckpts 'data/wbc/sweep_v4/model_*.pt' \
        --out data/wbc/sweep_v4/deploy_leaderboard.jsonl
"""
import os
import sys
import json
import glob
import time
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from forcesense.common.regions import region_of
from forcesense.models import ForceSensorV3, ContactGRUv4
from forcesense.common.data import (load_grid, gather_frames, impedance_constants,
                                    impedance_normalize)
from forcesense.common.metrics import (honest_labels, prf, frame_threshold_sweep,
                                        debounce_scan, debounce_sweep, predict_grids)


def build_model_featurize(ckpt, grid, device):
    """Reconstruct model + featurize(ti, ei) from a v4 checkpoint."""
    arch = ckpt.get("arch", "v3")
    K = ckpt["num_bodies"]
    W = ckpt["window"]
    D = ckpt["base_dim"]
    imp = bool(ckpt.get("imp_norm", False))
    kp_t, kd_t, qdef_t = impedance_constants(device)
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    Xg = grid["X"]
    if arch == "gru_v4":
        model = ContactGRUv4(D, K, ckpt["gru_hidden"], ckpt["gru_layers"]).to(device)

        def featurize(ti, ei):
            f = gather_frames(Xg, ti, ei, W)
            if imp:
                f = impedance_normalize(f, kp_t, kd_t, qdef_t)
            return (f - x_mean) / x_std
    else:
        model = ForceSensorV3(ckpt["in_dim"], K, ckpt["hidden"]).to(device)
        x_mean_w = x_mean.repeat(1, W); x_std_w = x_std.repeat(1, W)

        def featurize(ti, ei):
            f = gather_frames(Xg, ti, ei, W)
            if imp:
                f = impedance_normalize(f, kp_t, kd_t, qdef_t)
            return (f.reshape(ti.shape[0], -1) - x_mean_w) / x_std_w
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, featurize, W, K


def region_maps(grid, ckpt, device):
    """true region id [T,E] and (model class -> region id) mapping tensor."""
    body_names = grid["body_names"]
    regions = sorted(set(region_of(n) for n in body_names))
    reg_id = {r: i for i, r in enumerate(regions)}
    body_to_reg = torch.tensor([reg_id[region_of(n)] for n in body_names],
                               device=device, dtype=torch.long)
    true_reg = body_to_reg[grid["body_idx"]]                     # [T,E]
    cls_names = ckpt["body_names"]                               # model classes
    if ckpt.get("regions", False):
        cls_to_reg = torch.tensor([reg_id[c] for c in cls_names], device=device)
    else:
        cls_to_reg = torch.tensor([reg_id[region_of(c)] for c in cls_names], device=device)
    return true_reg, cls_to_reg, regions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+",
                   default=["data/wbc/datasets/wbc_v4_motion.h5", "data/wbc/datasets/wbc_v4_static.h5"])
    p.add_argument("--ckpts", type=str, nargs="+",
                   default=["data/wbc/sweep_v4/model_*.pt"])
    p.add_argument("--out", type=str, default="data/wbc/sweep_v4/deploy_leaderboard.jsonl")
    p.add_argument("--val_envs", type=int, default=16)
    p.add_argument("--n_envs", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    paths = sorted(sum([glob.glob(g) for g in args.ckpts], []))
    assert paths, f"no checkpoints match {args.ckpts}"
    dev = args.device

    print(f"[deploy-eval] loading grid {args.data} ...", flush=True)
    grid = load_grid(args.data, dev, n_envs=args.n_envs)
    V = args.val_envs
    pos_all, valid_all = honest_labels(grid)
    pos, valid = pos_all[:, :V], valid_all[:, :V]
    seg1 = grid["seg_bounds"][1]
    T = grid["T"]
    motion_mask = (torch.arange(T, device=dev) < seg1).unsqueeze(1).expand(T, V)
    print(f"[deploy-eval] honest labels: pos={pos[valid].float().mean():.3f} of valid, "
          f"excluded={1 - valid.float().mean():.3f} of frames", flush=True)

    results = []
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0].replace("model_", "")
        t0 = time.time()
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        model, featurize, W, K = build_model_featurize(ckpt, grid, dev)
        true_reg, cls_to_reg, regions = region_maps(grid, ckpt, dev)
        det_p, loc_arg = predict_grids(model, grid, featurize, W, V)

        frame_best = frame_threshold_sweep(det_p, pos, valid)
        deb_best = debounce_sweep(det_p, pos, valid, frame_best["th"])
        state = debounce_scan(det_p, deb_best["th_hi"], deb_best["th_lo"], deb_best["k"])
        ok = valid & ~torch.isnan(det_p)
        deb_motion = prf(state, pos, ok, motion_mask)
        deb_static = prf(state, pos, ok, ~motion_mask)

        # region accuracy on confirmed-contact frames (state ON & true pos)
        conf = state & pos & ok
        pred_reg = cls_to_reg[loc_arg]
        tr = true_reg[:, :V]
        reg_acc = float((pred_reg[conf] == tr[conf]).float().mean()) if conf.any() else float("nan")
        conf_m, conf_s = conf & motion_mask, conf & ~motion_mask
        reg_acc_m = float((pred_reg[conf_m] == tr[conf_m]).float().mean()) if conf_m.any() else float("nan")
        reg_acc_s = float((pred_reg[conf_s] == tr[conf_s]).float().mean()) if conf_s.any() else float("nan")

        rec = {"label": label, "arch": ckpt.get("arch", "v3"), "K": K, "window": W,
               "regions": bool(ckpt.get("regions", False)),
               "frame_best": frame_best, "debounce_best": deb_best,
               "debounce_motion": deb_motion, "debounce_static": deb_static,
               "region_acc_confirmed": round(reg_acc, 4),
               "region_acc_confirmed_motion": round(reg_acc_m, 4),
               "region_acc_confirmed_static": round(reg_acc_s, 4),
               "ckpt": path}
        results.append(rec)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[deploy-eval] {label:<24} frameF1={frame_best['f1']:.3f}@{frame_best['th']:.2f}  "
              f"debF1={deb_best['f1']:.3f} (k={deb_best['k']} hi={deb_best['th_hi']:.2f}) "
              f"P={deb_best['prec']:.3f} R={deb_best['rec']:.3f}  "
              f"motF1={deb_motion['f1']:.3f} statF1={deb_static['f1']:.3f}  "
              f"regAcc={reg_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda r: r["debounce_best"]["f1"], reverse=True)
    print("\n===== DEPLOYMENT LEADERBOARD (honest labels, debounced) =====", flush=True)
    print(f"{'label':<24} {'debF1':>6} {'prec':>6} {'rec':>6} {'motF1':>6} {'statF1':>7} "
          f"{'regAcc':>7} {'frameF1':>8} {'th':>5} {'k':>2}", flush=True)
    for r in results:
        d = r["debounce_best"]
        print(f"{r['label']:<24} {d['f1']:>6.3f} {d['prec']:>6.3f} {d['rec']:>6.3f} "
              f"{r['debounce_motion']['f1']:>6.3f} {r['debounce_static']['f1']:>7.3f} "
              f"{r['region_acc_confirmed']:>7.3f} {r['frame_best']['f1']:>8.3f} "
              f"{d['th_hi']:>5.2f} {d['k']:>2}", flush=True)
    print(f"\nappended {len(results)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
