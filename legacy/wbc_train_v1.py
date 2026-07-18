"""Whole-Body Contact (Plan A) — advanced trainer, callable from the sweep.

Adds the levers the first trainer lacked:
  - honest eval: hold out whole collection envs (no temporally-adjacent leakage)
  - class weighting / down-weight the dominant <none> class
  - body-region grouping (left/right arm, left/right leg, trunk) -> better observability
  - drop ramp-phase / weak-force frames (cleaner labels)
  - configurable arch / lr / epochs / dropout

Reuses ForceSensorMLP + load_data from train_force_sensor so the live viewer and
viz scripts can load any model produced here.

CLI:  python scripts/wbc_train.py --data data/wbc/wbc_train.h5 --regions --balance ...
Lib:  from scripts.wbc_train import run_experiment; m = run_experiment({...})
"""
# ARCHIVED v1 — superseded by forcesense/ (kept for reference; not maintained).
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train_force_sensor import ForceSensorMLP, load_data

ORIG_K = 12  # number of force links in collect.yaml (label onehot width)


def region_of(name):
    side = "left" if name.startswith("left") else ("right" if name.startswith("right") else "")
    if any(k in name for k in ("shoulder", "elbow", "wrist", "hand")):
        return f"{side}_arm"
    if any(k in name for k in ("hip", "knee", "ankle")):
        return f"{side}_leg"
    return "trunk"


def parse_label(Y, body_names, contact_thresh, use_regions):
    """Return class_idx[N], class_names, force_b[N,3], mag[N], phase[N]."""
    K0 = ORIG_K
    force_b = Y[:, 3:6].astype(np.float32)
    onehot = Y[:, 9:9 + K0]
    phase = Y[:, 9 + K0:13 + K0].argmax(1)         # 0 rest,1 ramp_up,2 hold,3 ramp_down
    mag = Y[:, -1].astype(np.float32)
    body_idx = onehot.argmax(1)
    active = mag > contact_thresh

    if use_regions:
        regions = sorted(set(region_of(n) for n in body_names))
        reg_id = {r: i for i, r in enumerate(regions)}
        body_to_reg = np.array([reg_id[region_of(n)] for n in body_names], dtype=np.int64)
        loc = body_to_reg[body_idx]
        names = regions
    else:
        loc = body_idx.astype(np.int64)
        names = list(body_names)

    K = len(names)
    cls = np.where(active, loc, K).astype(np.int64)   # class K == no-contact
    return cls, names, force_b, mag, phase, active


def evaluate(model, X, cls, force_b, active, K, force_max, device, bs=16384):
    model.eval()
    preds, regs = [], []
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            lo, re = model(X[s:s + bs].to(device))
            preds.append(lo.argmax(1).cpu()); regs.append(re.cpu())
    pred = torch.cat(preds); reg = torch.cat(regs)
    cls_t = torch.from_numpy(cls); act = torch.from_numpy(active)
    overall = (pred == cls_t).float().mean().item()
    am = act.bool()
    active_acc = (pred[am] == cls_t[am]).float().mean().item() if am.any() else float("nan")
    # detection: contact (cls<K) vs none (cls==K)
    pred_contact = pred < K; true_contact = cls_t < K
    tp = (pred_contact & true_contact).sum().item()
    fp = (pred_contact & ~true_contact).sum().item()
    fn = (~pred_contact & true_contact).sum().item()
    det_prec = tp / max(tp + fp, 1); det_rec = tp / max(tp + fn, 1)
    det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-9)
    # force vector on active
    if am.any():
        fp_ = reg[am]; ft_ = torch.from_numpy(force_b)[am]
        cos = F.cosine_similarity(fp_, ft_, dim=1).mean().item()
        mr = ((fp_.norm(dim=1) - ft_.norm(dim=1)).abs()
              / ft_.norm(dim=1).clamp_min(1e-3)).mean().item()
    else:
        cos = mr = float("nan")
    # per-class recall
    cm = torch.zeros(K + 1, K + 1, dtype=torch.long)
    for t, p in zip(cls_t.tolist(), pred.tolist()):
        cm[t, p] += 1
    recall = [(cm[i, i].item() / max(cm[i].sum().item(), 1)) for i in range(K)]
    return {"overall_acc": overall, "active_acc": active_acc, "det_f1": det_f1,
            "det_prec": det_prec, "det_rec": det_rec, "force_cos": cos,
            "mag_rel_err": mr, "recall": recall, "cm": cm.tolist()}


def run_experiment(cfg):
    """cfg dict keys: data, out, regions, balance, none_weight, drop_ramp, min_mag,
    contact_thresh, hidden, dropout, lr, weight_decay, epochs, batch_size,
    collect_envs, val_envs, lambda_reg, device, seed, label(str)."""
    g = lambda k, d: cfg.get(k, d)
    device = g("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(g("seed", 0)); np.random.seed(g("seed", 0))

    X, Y, meta = load_data([cfg["data"]] if isinstance(cfg["data"], str) else cfg["data"])
    force_max = meta["force_max"]; body_names = meta["body_names"]
    cls, names, force_b, mag, phase, active = parse_label(
        Y, body_names, g("contact_thresh", 0.05), g("regions", False))
    K = len(names)

    # ---- optional filtering of ramp / weak active frames (keep all none) ----
    keep = np.ones(X.shape[0], dtype=bool)
    if g("drop_ramp", False):
        keep &= ~(active & (phase != 2))          # drop active non-hold
    if g("min_mag", 0.0) > 0:
        keep &= ~(active & (mag < g("min_mag")))  # drop weak active
    # ---- honest split: hold out whole collection envs ----
    ce = g("collect_envs", 128); ve = g("val_envs", 16)
    env_id = np.arange(X.shape[0]) % ce
    is_val = env_id < ve
    tr = keep & ~is_val
    va = keep & is_val
    Xtr_np = X[tr]
    x_mean = Xtr_np.mean(0, keepdims=True); x_std = Xtr_np.std(0, keepdims=True) + 1e-6

    def prep(mask):
        Xn = (X[mask] - x_mean) / x_std
        return (torch.from_numpy(Xn).float(), torch.from_numpy(cls[mask]),
                torch.from_numpy(force_b[mask]), torch.from_numpy(active[mask].astype(np.float32)))
    Xtr, cls_tr, f_tr, a_tr = prep(tr)
    Xva, cls_va, f_va, a_va = prep(va)

    # ---- class weights ----
    weight = torch.ones(K + 1)
    if g("balance", False):
        counts = np.bincount(cls[tr], minlength=K + 1).astype(np.float64)
        w = counts.sum() / (np.maximum(counts, 1) * (K + 1))
        weight = torch.tensor(w, dtype=torch.float32)
    if g("none_weight", None) is not None:
        weight[K] = g("none_weight")
    weight = weight.to(device)

    model = ForceSensorMLP(X.shape[1], K + 1, g("hidden", [512, 512, 256]), g("dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=g("lr", 1e-3), weight_decay=g("weight_decay", 1e-5))
    lam = g("lambda_reg", 1.0); bs = g("batch_size", 8192); n_tr = Xtr.shape[0]

    Xva_d = Xva  # keep on cpu; evaluate moves batches
    best = {"active_acc": -1}
    for ep in range(g("epochs", 40)):
        model.train(); idx = torch.randperm(n_tr)
        for s in range(0, n_tr, bs):
            b = idx[s:s + bs]
            logits, reg = model(Xtr[b].to(device))
            closs = F.cross_entropy(logits, cls_tr[b].to(device), weight=weight)
            m = a_tr[b].to(device)
            rloss = (((reg - f_tr[b].to(device)) ** 2).sum(1) * m).sum() / m.sum().clamp_min(1) if m.sum() > 0 else torch.zeros((), device=device)
            loss = closs + lam * rloss
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % g("eval_every", 5) == 0 or ep == g("epochs", 40) - 1:
            mtr = evaluate(model, Xva_d, cls_va.numpy(), f_va.numpy(), a_va.numpy().astype(bool), K, force_max, device)
            if mtr["active_acc"] > best["active_acc"]:
                best = {**mtr, "epoch": ep}
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{cfg.get('label','exp')}] ep{ep:3d} act_acc={mtr['active_acc']:.3f} "
                  f"det_f1={mtr['det_f1']:.3f} cos={mtr['force_cos']:.3f} overall={mtr['overall_acc']:.3f}", flush=True)

    if cfg.get("out"):
        os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
        torch.save({"state_dict": best_state, "x_mean": torch.from_numpy(x_mean).float(),
                    "x_std": torch.from_numpy(x_std).float(), "hidden": g("hidden", [512, 512, 256]),
                    "in_dim": X.shape[1], "num_classes": K + 1, "num_bodies": K,
                    "force_max": force_max, "body_names": names,
                    "contact_thresh": g("contact_thresh", 0.05), "regions": g("regions", False)},
                   cfg["out"])
    return {"label": cfg.get("label", "exp"), "names": names, "K": K, **best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, default="data/wbc/force_sensor_v2.pt")
    p.add_argument("--regions", action="store_true")
    p.add_argument("--balance", action="store_true")
    p.add_argument("--none_weight", type=float, default=None)
    p.add_argument("--drop_ramp", action="store_true")
    p.add_argument("--min_mag", type=float, default=0.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--val_envs", type=int, default=16)
    p.add_argument("--collect_envs", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    m = run_experiment(vars(args) | {"label": "cli"})
    print(json.dumps({k: v for k, v in m.items() if k not in ("cm", "recall", "names")}, indent=2))


if __name__ == "__main__":
    main()
