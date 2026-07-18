"""Visualize a trained whole-body force-sensing MLP (Plan A) — offline plots.

    python scripts/viz_force_sensor.py --data data/wbc/wbc_train.h5 \
        --model data/wbc/force_sensor.pt --out_dir data/wbc/viz

Produces (PNG):
  - confusion_matrix.png : row-normalized link confusion (incl. <none>)
  - per_body.png         : per-body localization recall + detection rate
  - force_timeline.png   : true vs predicted force magnitude over time for one env,
                           with true/predicted contact-link strips
"""
# ARCHIVED v1 — superseded by forcesense/ (kept for reference; not maintained).
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train_force_sensor import ForceSensorMLP, load_data, build_targets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/wbc/viz")
    p.add_argument("--num_envs", type=int, default=128, help="envs used during collection (for timeline de-interleave)")
    p.add_argument("--max_eval", type=int, default=400000, help="cap samples scored for the confusion/per-body plots")
    p.add_argument("--timeline_env", type=int, default=0)
    p.add_argument("--timeline_steps", type=int, default=600)
    p.add_argument("--contact_thresh", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ForceSensorMLP(ckpt["in_dim"], ckpt["num_classes"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, X, x_mean, x_std, device, bs=16384):
    logits_all, reg_all = [], []
    for s in range(0, X.shape[0], bs):
        xb = torch.from_numpy((X[s:s+bs] - x_mean) / x_std).float().to(device)
        lo, re = model(xb)
        logits_all.append(lo.argmax(1).cpu().numpy())
        reg_all.append(re.cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(reg_all)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    model, ckpt = load_model(args.model, device)
    body_names = ckpt["body_names"]
    K = ckpt["num_bodies"]
    force_max = ckpt["force_max"]
    x_mean = ckpt["x_mean"].cpu().numpy()
    x_std = ckpt["x_std"].cpu().numpy()
    labels = body_names + ["<none>"]

    X, Y, meta = load_data([args.data])
    cls, force_b, active = build_targets(Y, meta, args.contact_thresh)

    # ---- confusion + per-body on a capped slice ----
    n = min(args.max_eval, X.shape[0])
    pred, reg = predict(model, X[:n], x_mean, x_std, device)
    true = cls[:n]

    cm = np.zeros((K + 1, K + 1), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K + 1)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(K + 1)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Force-link confusion (row-normalized)")
    for i in range(K + 1):
        ax.text(i, i, f"{cm_norm[i,i]:.2f}", ha="center", va="center",
                color="w" if cm_norm[i, i] < 0.6 else "k", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ---- per-body recall + detection rate ----
    recall = np.array([cm[i, i] / max(cm[i].sum(), 1) for i in range(K)])
    detect = np.array([1.0 - cm[i, K] / max(cm[i].sum(), 1) for i in range(K)])  # not predicted <none>
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(K)
    ax.bar(x - 0.2, detect, 0.4, label="detected as contact", color="#7aa8d6")
    ax.bar(x + 0.2, recall, 0.4, label="correct link (recall)", color="#c2603f")
    ax.set_xticks(x); ax.set_xticklabels(body_names, rotation=90, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("rate"); ax.legend()
    ax.set_title("Per-body: contact detection vs correct localization")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "per_body.png"), dpi=120)
    plt.close(fig)

    # ---- force timeline for one env (rows are step-major: row = step*num_envs + env) ----
    ne = args.num_envs
    idx = args.timeline_env + ne * np.arange(args.timeline_steps)
    idx = idx[idx < X.shape[0]]
    p_idx, r_idx = predict(model, X[idx], x_mean, x_std, device)
    true_mag = np.linalg.norm(force_b[idx], axis=1) * force_max
    pred_mag = np.linalg.norm(r_idx, axis=1) * force_max
    t = np.arange(len(idx))

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1], sharex=True)
    a0.plot(t, true_mag, color="#c2603f", lw=1.6, label="true |F| (N)")
    a0.plot(t, pred_mag, color="#2e7d32", lw=1.2, alpha=0.8, label="predicted |F| (N)")
    a0.set_ylabel("force magnitude (N)"); a0.legend(loc="upper right")
    a0.set_title(f"env {args.timeline_env}: true vs predicted external force")
    # contact-link strips
    a1.plot(t, cls[idx], color="#c2603f", lw=0, marker="s", ms=3, label="true link")
    a1.plot(t, p_idx, color="#2e7d32", lw=0, marker="x", ms=3, label="predicted link")
    a1.set_yticks(range(K + 1)); a1.set_yticklabels(labels, fontsize=6)
    a1.set_ylabel("contact link"); a1.set_xlabel("step"); a1.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "force_timeline.png"), dpi=120)
    plt.close(fig)

    # ---- console summary ----
    am = active[:n]
    print(f"[viz] scored {n} samples | active_acc={(pred[am]==true[am]).mean():.3f} "
          f"overall_acc={(pred==true).mean():.3f}")
    print("[viz] per-body recall:")
    for i in range(K):
        print(f"    {body_names[i]:<26} detect={detect[i]:.2f}  recall={recall[i]:.2f}")
    print(f"[viz] saved 3 PNGs -> {args.out_dir}")


if __name__ == "__main__":
    main()
