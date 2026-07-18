"""Visualize a trained whole-body force-sensing MLP (Plan A, **v3** dual-head) — offline plots.

    source scripts/start_gentle_local.sh
    python scripts/viz_force_sensor_v3.py \
        --data data/wbc/datasets/wbc_train.h5 \
        --model data/wbc/sweep_v3/force_sensor_v3_best.pt \
        --out_dir data/wbc/viz_v3

Unlike viz_force_sensor.py (v1 per-link single-softmax), this loads the v3 grid,
stacks W frames, and uses the detection / localization / dir / mag heads. Eval is
on the HELD-OUT val envs (0..val_envs-1) so numbers match the leaderboard.

Produces (PNG):
  - confusion_matrix.png : row-normalized region confusion (incl. <none>)
  - per_region.png       : per-region detection rate + correct-region recall
  - force_timeline.png    : true vs predicted |F| over time for one val env,
                            with true/predicted contact-region strips
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from forcesense.models import ForceSensorV3
from forcesense.common.data import load_grid, make_targets, build_index, gather_window
from forcesense.train.core import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/wbc/datasets/wbc_train.h5")
    p.add_argument("--model", type=str, default="data/wbc/sweep_v3/force_sensor_v3_best.pt")
    p.add_argument("--out_dir", type=str, default="data/wbc/viz_v3")
    p.add_argument("--n_envs", type=int, default=128, help="collection envs (grid reshape)")
    p.add_argument("--val_envs", type=int, default=16, help="held-out envs used at train time")
    p.add_argument("--timeline_env", type=int, default=0, help="which val env to draw (0..val_envs-1)")
    p.add_argument("--timeline_steps", type=int, default=600)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    assert ckpt.get("arch") == "v3", f"not a v3 checkpoint: arch={ckpt.get('arch')}"
    model = ForceSensorV3(ckpt["in_dim"], ckpt["num_bodies"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_env(model, grid, W, e0, x_mean_w, x_std_w, det_thresh, bs=65536):
    """Per-frame prediction for a single env across all time. Returns dict of np arrays."""
    Xg = grid["X"]
    T = grid["T"]
    dev = Xg.device
    t_idx = torch.arange(W - 1, T, device=dev)
    e_idx = torch.full_like(t_idx, e0)
    det_p, loc_arg, mag_pred = [], [], []
    for s in range(0, t_idx.shape[0], bs):
        ti, ei = t_idx[s:s + bs], e_idx[s:s + bs]
        x = (gather_window(Xg, ti, ei, W) - x_mean_w) / x_std_w
        det, loc, dirh, magh = model(x)
        det_p.append(torch.sigmoid(det.squeeze(-1)))
        loc_arg.append(loc.argmax(-1))
        mag_pred.append(magh.squeeze(-1).clamp_min(0))
    det_p = torch.cat(det_p)
    loc_arg = torch.cat(loc_arg)
    mag_pred = torch.cat(mag_pred)
    return {
        "t": t_idx.cpu().numpy(),
        "det_p": det_p.cpu().numpy(),
        "loc_arg": loc_arg.cpu().numpy(),
        "mag_pred": mag_pred.cpu().numpy(),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    model, ckpt = load_model(args.model, device)
    W = ckpt["window"]
    det_thresh = ckpt["det_thresh"]
    force_max = ckpt["force_max"]
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    x_mean_w = x_mean.repeat(1, W)
    x_std_w = x_std.repeat(1, W)

    grid = load_grid(args.data, device, args.n_envs)
    cls, names, K, active = make_targets(grid, ckpt["regions"], ckpt["contact_thresh"], device)
    assert names == ckpt["body_names"], f"class names mismatch: {names} vs {ckpt['body_names']}"
    labels = names + ["<none>"]
    T, E = grid["T"], grid["E"]

    # ---- honest eval on held-out val envs ----
    t_all, e_all, is_val = build_index(T, E, W, args.val_envs)
    t_va = t_all[is_val].to(device)
    e_va = e_all[is_val].to(device)
    m = evaluate(model, grid, cls, active, K, W, t_va, e_va, det_thresh, x_mean_w, x_std_w)
    cm = np.array(m["cm"], dtype=np.int64)
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)

    print(f"[viz-v3] window={W} regions={ckpt['regions']} det_thresh={det_thresh} "
          f"classes={labels}")
    print(f"[viz-v3] active_acc={m['active_acc']:.3f} det_f1={m['det_f1']:.3f} "
          f"(prec={m['det_prec']:.2f} rec={m['det_rec']:.2f}) "
          f"force_cos={m['force_cos']:.3f} mag_rel_err={m['mag_rel_err']:.2f}")

    # ---- confusion matrix ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K + 1)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K + 1)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"v3 region confusion (row-normalized)  W={W}")
    for i in range(K + 1):
        for j in range(K + 1):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="w" if cm_norm[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ---- per-region detection + recall ----
    recall = np.array([cm[i, i] / max(cm[i].sum(), 1) for i in range(K)])
    detect = np.array([1.0 - cm[i, K] / max(cm[i].sum(), 1) for i in range(K)])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(K)
    ax.bar(x - 0.2, detect, 0.4, label="detected as contact", color="#7aa8d6")
    ax.bar(x + 0.2, recall, 0.4, label="correct region (recall)", color="#c2603f")
    for i in range(K):
        ax.text(i - 0.2, detect[i] + 0.02, f"{detect[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, recall[i] + 0.02, f"{recall[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("rate"); ax.legend()
    ax.set_title("v3 per-region: contact detection vs correct localization")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "per_region.png"), dpi=120)
    plt.close(fig)

    # ---- force timeline for one val env ----
    e0 = args.timeline_env
    pe = predict_env(model, grid, W, e0, x_mean_w, x_std_w, det_thresh)
    true_cls = cls[:, e0].cpu().numpy()[pe["t"]]
    true_mag = grid["mag"][:, e0].cpu().numpy()[pe["t"]] * force_max
    det_on = pe["det_p"] > det_thresh
    # the magnitude head is only meaningful when contact is detected; gate it by
    # the detection head exactly as a deployed sensor would (no contact -> 0 N).
    pred_mag = pe["mag_pred"] * force_max * det_on
    pred_cls = np.where(det_on, pe["loc_arg"], K)

    # choose a slice that actually contains contact events
    act_idx = np.where(true_cls < K)[0]
    start = max(0, int(act_idx[0]) - 40) if act_idx.size else 0
    sl = slice(start, start + args.timeline_steps)
    t = np.arange(start, start + len(true_cls[sl]))

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1], sharex=True)
    a0.plot(t, true_mag[sl], color="#c2603f", lw=1.8, label="true |F| (N)")
    a0.plot(t, pred_mag[sl], color="#2e7d32", lw=1.1, alpha=0.85, label="predicted |F| (N), det-gated")
    a0.set_ylabel("force magnitude (N)"); a0.legend(loc="upper right")
    a0.set_title(f"v3 env {e0}: true vs predicted external force  (det_thresh={det_thresh})")
    a1.plot(t, true_cls[sl], color="#c2603f", lw=0, marker="s", ms=4, label="true region")
    a1.plot(t, pred_cls[sl], color="#2e7d32", lw=0, marker="x", ms=4, label="predicted region")
    a1.set_yticks(range(K + 1)); a1.set_yticklabels(labels, fontsize=8)
    a1.set_ylabel("contact region"); a1.set_xlabel("step"); a1.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "force_timeline.png"), dpi=120)
    plt.close(fig)

    print(f"[viz-v3] per-region:")
    for i in range(K):
        print(f"    {names[i]:<12} detect={detect[i]:.2f}  recall={recall[i]:.2f}")
    print(f"[viz-v3] saved 3 PNGs -> {args.out_dir}")


if __name__ == "__main__":
    main()
