"""Whole-Body Contact (Plan A) — train the proprioceptive force-sensing MLP.

Supervised learning, no RL, no Isaac. Reads HDF5 produced by
scripts/collect_force_data.py and trains an MLP that maps proprioception ->
(which body is being pushed, force vector).

    python scripts/train_force_sensor.py --data data/wbc/wbc_train.h5 \
        --out data/wbc/force_sensor.pt --epochs 40

Targets parsed from net_pull_force_priv label layout:
    [ point_b(3) | force_b/Fmax(3) | force_w/Fmax(3) | body_onehot(K) | phase(4) | timer(1) | mag(1) ]

Two heads:
    - classification over K bodies + 1 "no-contact" class
    - regression of the base-frame force vector (normalized by Fmax), masked to
      samples where an external force is actually active.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+", required=True, help="HDF5 file(s) from collect_force_data.py")
    p.add_argument("--out", type=str, default="data/wbc/force_sensor.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--contact_thresh", type=float, default=0.05,
                   help="Normalized-magnitude threshold above which a sample counts as 'in contact'.")
    p.add_argument("--lambda_reg", type=float, default=1.0, help="Weight of the force-vector regression loss.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_data(paths):
    Xs, Ys = [], []
    meta = None
    for path in paths:
        with h5py.File(path, "r") as h:
            Xs.append(h["X"][:])
            Ys.append(h["Y"][:])
            m = {
                "num_bodies": int(h.attrs["num_bodies"]),
                "force_max": float(h.attrs["force_max"]),
                "body_names": [b.decode() if isinstance(b, bytes) else str(b) for b in h.attrs["body_names"]],
                "label_prefix": int(h.attrs["label_prefix"]),
                "input_dim": int(h.attrs["input_dim"]),
            }
        if meta is None:
            meta = m
        else:
            assert m["num_bodies"] == meta["num_bodies"], "datasets disagree on num_bodies"
            assert m["input_dim"] == meta["input_dim"], "datasets disagree on input_dim"
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y, meta


def build_targets(Y, meta, contact_thresh):
    """Return (class_idx [N], force_b [N,3] normalized, active_mask [N])."""
    K = meta["num_bodies"]
    pfx = meta["label_prefix"]  # 9
    force_b = Y[:, 3:6]                     # base-frame force / Fmax
    body_onehot = Y[:, pfx:pfx + K]
    mag = Y[:, -1]                          # normalized magnitude
    active = mag > contact_thresh
    body_idx = body_onehot.argmax(axis=1)
    cls = np.where(active, body_idx, K).astype(np.int64)   # class K == "no contact"
    return cls, force_b.astype(np.float32), active


class ForceSensorMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(d, num_classes)
        self.reg_head = nn.Linear(d, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.cls_head(z), self.reg_head(z)


def evaluate(model, Xv, cls_v, force_v, active_v, K, force_max, device, body_names):
    model.eval()
    with torch.no_grad():
        logits, reg = model(Xv.to(device))
        logits, reg = logits.cpu(), reg.cpu()
    pred = logits.argmax(1)

    overall_acc = (pred == cls_v).float().mean().item()
    am = active_v.bool()
    if am.any():
        active_acc = (pred[am] == cls_v[am]).float().mean().item()
        # force vector metrics on active samples
        f_pred = reg[am]
        f_true = force_v[am]
        cos = F.cosine_similarity(f_pred, f_true, dim=1).mean().item()
        mag_pred = f_pred.norm(dim=1) * force_max
        mag_true = f_true.norm(dim=1) * force_max
        mag_rel = ((mag_pred - mag_true).abs() / mag_true.clamp_min(1e-3)).mean().item()
    else:
        active_acc = cos = mag_rel = float("nan")

    # confusion matrix over K+1 classes
    cm = torch.zeros(K + 1, K + 1, dtype=torch.long)
    for t, pdc in zip(cls_v.tolist(), pred.tolist()):
        cm[t, pdc] += 1
    return {"overall_acc": overall_acc, "active_acc": active_acc,
            "force_cos": cos, "mag_rel_err": mag_rel, "cm": cm}


def print_confusion(cm, body_names):
    labels = body_names + ["<none>"]
    width = max(len(l) for l in labels) + 1
    print("\nconfusion matrix (row=true, col=pred), per-row recall on right:")
    header = " " * width + "".join(f"{i:>5}" for i in range(len(labels)))
    print(header)
    for i, l in enumerate(labels):
        row = cm[i]
        total = row.sum().item()
        recall = (row[i].item() / total) if total else float("nan")
        cells = "".join(f"{row[j].item():>5}" for j in range(len(labels)))
        print(f"{l:<{width}}{cells}   recall={recall:.2f}  (n={total})")
    print("legend:", {i: l for i, l in enumerate(labels)})


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, Y, meta = load_data(args.data)
    K = meta["num_bodies"]
    force_max = meta["force_max"]
    body_names = meta["body_names"]
    print(f"[train] {X.shape[0]} samples, input_dim={X.shape[1]}, K={K} bodies, force_max={force_max}")

    cls, force_b, active = build_targets(Y, meta, args.contact_thresh)
    print(f"[train] contact-active fraction = {active.mean():.3f}")

    # shuffle + split
    N = X.shape[0]
    perm = np.random.permutation(N)
    X, cls, force_b, active = X[perm], cls[perm], force_b[perm], active[perm]
    n_val = int(N * args.val_frac)
    sl_tr, sl_va = slice(n_val, None), slice(0, n_val)

    # input normalization from train split
    x_mean = X[sl_tr].mean(0, keepdims=True)
    x_std = X[sl_tr].std(0, keepdims=True) + 1e-6
    Xn = (X - x_mean) / x_std

    to_t = lambda a: torch.from_numpy(a)
    Xtr, Xva = to_t(Xn[sl_tr]).float(), to_t(Xn[sl_va]).float()
    cls_tr, cls_va = to_t(cls[sl_tr]), to_t(cls[sl_va])
    f_tr, f_va = to_t(force_b[sl_tr]), to_t(force_b[sl_va])
    a_tr, a_va = to_t(active[sl_tr].astype(np.float32)), to_t(active[sl_va].astype(np.float32))

    device = args.device
    model = ForceSensorMLP(X.shape[1], K + 1, args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_tr = Xtr.shape[0]
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(n_tr)
        tot_cls = tot_reg = 0.0
        for s in range(0, n_tr, args.batch_size):
            b = idx[s:s + args.batch_size]
            xb = Xtr[b].to(device)
            logits, reg = model(xb)
            cls_loss = F.cross_entropy(logits, cls_tr[b].to(device))
            mask = a_tr[b].to(device)
            if mask.sum() > 0:
                reg_err = ((reg - f_tr[b].to(device)) ** 2).sum(1)
                reg_loss = (reg_err * mask).sum() / mask.sum()
            else:
                reg_loss = torch.zeros((), device=device)
            loss = cls_loss + args.lambda_reg * reg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_cls += cls_loss.item() * xb.shape[0]
            tot_reg += reg_loss.item() * xb.shape[0]
        m = evaluate(model, Xva, cls_va, f_va, a_va, K, force_max, device, body_names)
        print(f"[ep {epoch:3d}] cls_loss={tot_cls/n_tr:.4f} reg_loss={tot_reg/n_tr:.4f} | "
              f"val acc={m['overall_acc']:.3f} active_acc={m['active_acc']:.3f} "
              f"force_cos={m['force_cos']:.3f} mag_rel_err={m['mag_rel_err']:.3f}")

    m = evaluate(model, Xva, cls_va, f_va, a_va, K, force_max, device, body_names)
    print_confusion(m["cm"], body_names)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "x_mean": torch.from_numpy(x_mean).float(),
        "x_std": torch.from_numpy(x_std).float(),
        "hidden": args.hidden,
        "in_dim": X.shape[1],
        "num_classes": K + 1,
        "num_bodies": K,
        "force_max": force_max,
        "body_names": body_names,
        "contact_thresh": args.contact_thresh,
    }, args.out)
    print(f"[train] saved -> {args.out}")


if __name__ == "__main__":
    main()
