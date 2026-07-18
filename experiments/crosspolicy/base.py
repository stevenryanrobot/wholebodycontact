"""Cross-policy generalization test for the plug-and-play force estimator.

Question: does a proprioception->contact estimator trained under ONE controller
transfer to a DIFFERENT controller (different downstream policy)? We emulate
"different policy" with scaled deploy PD gains (base / soft / stiff), collected
by forcesense/collect/mujoco.py --kp_scale/--kd_scale. Same net_pull GT force
protocol across all three, so the ONLY difference is the controller.

We train a dual-head (detection + 5-region) MLP on ONE dataset and evaluate it
in-domain (held-out columns of the SAME controller) vs cross-domain (the OTHER
controllers). A large in-domain -> cross-domain drop = the plug-and-play gap is
real (naive estimator overfits the controller); a small drop = it transfers.

Usage:
    python experiments/crosspolicy/base.py \
        --train data/wbc/cross/cross_A_base.h5 \
        --eval  data/wbc/cross/cross_A_base.h5 data/wbc/cross/cross_B_soft.h5 \
                data/wbc/cross/cross_C_stiff.h5
"""
import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import h5py

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WIN = 6
BASE_DIM = 320
LABEL_PREFIX = 9
POS_MAG_N = 5.0          # a frame is "contact" if |F| > 5 N (honest label)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

REGION_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg", "trunk"]


def link_to_region(names):
    """Map each 24-link index -> region idx (0..4)."""
    reg = np.zeros(len(names), dtype=np.int64)
    for i, n in enumerate(names):
        side = 0 if n.startswith("left") else (1 if n.startswith("right") else -1)
        is_arm = ("shoulder" in n) or ("elbow" in n) or ("wrist" in n)
        is_leg = ("hip" in n) or ("knee" in n) or ("ankle" in n)
        if is_arm:
            reg[i] = 0 if side == 0 else 1
        elif is_leg:
            reg[i] = 2 if side == 0 else 3
        else:                                    # pelvis / torso
            reg[i] = 4
    return reg


def load(h5path, feat="proprio"):
    with h5py.File(h5path, "r") as h:
        X = np.asarray(h["X"], dtype=np.float32)
        Y = np.asarray(h["Y"], dtype=np.float32)
        R = np.asarray(h["R"], dtype=np.float32) if "R" in h else None
        W = int(h.attrs["n_envs"])
        names = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in h.attrs["body_names"]]
        fmax = float(h.attrs["force_max"])
        kp = float(h.attrs.get("kp_scale", 1.0))
    K = len(names)
    T = X.shape[0] // W
    # choose the input feature source: raw proprioception (320), the
    # policy-invariant residual channel (29), or both concatenated (349).
    if feat == "proprio":
        F = X
    elif feat == "resid":
        assert R is not None, f"{h5path} has no R channel"
        F = R
    elif feat == "both":
        assert R is not None, f"{h5path} has no R channel"
        F = np.concatenate([X, R], axis=1)
    else:
        raise ValueError(feat)
    bdim = F.shape[1]
    F3 = torch.from_numpy(F[:T * W].reshape(T, W, bdim)).to(DEV)
    Y3 = torch.from_numpy(Y[:T * W].reshape(T, W, Y.shape[1])).to(DEV)
    l2r = torch.from_numpy(link_to_region(names)).to(DEV)
    onehot = Y3[:, :, LABEL_PREFIX:LABEL_PREFIX + K]           # [T,W,K]
    link_idx = onehot.argmax(-1)                               # [T,W]
    region = l2r[link_idx]                                     # [T,W]
    mag_n = Y3[:, :, -1]                                       # mag / fmax
    pos = (mag_n * fmax) > POS_MAG_N                           # [T,W] contact
    return dict(F3=F3, bdim=bdim, region=region, pos=pos, T=T, W=W,
                fmax=fmax, kp=kp, path=os.path.basename(h5path))


def load_multi(paths, feat="proprio"):
    """Load several controller datasets and stack along the worker-column axis
    (windows are per-column, so this just adds more independent columns)."""
    dss = [load(p, feat) for p in paths]
    T = min(d["T"] for d in dss)
    F3 = torch.cat([d["F3"][:T] for d in dss], dim=1)
    region = torch.cat([d["region"][:T] for d in dss], dim=1)
    pos = torch.cat([d["pos"][:T] for d in dss], dim=1)
    return dict(F3=F3, bdim=dss[0]["bdim"], region=region, pos=pos, T=T,
                W=F3.shape[1], fmax=dss[0]["fmax"], kp=-1,
                path="+".join(d["path"].replace("cross_", "").replace(".h5", "")
                              for d in dss))


def valid_pairs(T, W, cols=None):
    """all (t,w) with a full window; cols restricts worker columns."""
    ts = torch.arange(WIN - 1, T)
    ws = torch.arange(W) if cols is None else torch.as_tensor(cols)
    tt, ww = torch.meshgrid(ts, ws, indexing="ij")
    return tt.reshape(-1).to(DEV), ww.reshape(-1).to(DEV)


def gather(F3, t, w, mean=None, std=None):
    off = torch.arange(WIN, device=DEV) - (WIN - 1)            # oldest..newest
    rows = t[:, None] + off[None, :]                          # [B,WIN]
    feat = F3[rows, w[:, None], :].reshape(t.shape[0], WIN * F3.shape[-1])
    if mean is not None:
        feat = (feat - mean) / std
    return feat


class Net(nn.Module):
    def __init__(self, din, nreg=5):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(din, 512), nn.ReLU(),
                                   nn.Linear(512, 256), nn.ReLU())
        self.det = nn.Linear(256, 1)
        self.reg = nn.Linear(256, nreg)

    def forward(self, x):
        h = self.trunk(x)
        return self.det(h).squeeze(-1), self.reg(h)


def evaluate(net, ds, mean, std, cols=None, batch=16384):
    net.eval()
    t, w = valid_pairs(ds["T"], ds["W"], cols)
    reg_t = ds["region"][t, w]
    pos_t = ds["pos"][t, w]
    det_p = torch.empty(t.shape[0], device=DEV)
    reg_pred = torch.empty(t.shape[0], dtype=torch.long, device=DEV)
    with torch.no_grad():
        for i in range(0, t.shape[0], batch):
            sl = slice(i, i + batch)
            feat = gather(ds["F3"], t[sl], w[sl], mean, std)
            dlog, rlog = net(feat)
            det_p[sl] = torch.sigmoid(dlog)
            reg_pred[sl] = rlog.argmax(-1)
    det = det_p > 0.5
    tp = (det & pos_t).sum().item(); fp = (det & ~pos_t).sum().item()
    fn = (~det & pos_t).sum().item()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    posm = pos_t
    correct_reg = (reg_pred == reg_t)
    region_acc = (correct_reg & posm).sum().item() / (posm.sum().item() + 1e-9)
    active_acc = (correct_reg & det & posm).sum().item() / (posm.sum().item() + 1e-9)
    return dict(det_f1=f1, det_prec=prec, det_rec=rec,
                region_acc=region_acc, active_acc=active_acc,
                n=int(t.shape[0]), n_pos=int(posm.sum().item()))


def train(ds, epochs, cols_train):
    t, w = valid_pairs(ds["T"], ds["W"], cols_train)
    # normalization from a random subset of train windows
    idx = torch.randperm(t.shape[0], device=DEV)[:50000]
    sample = gather(ds["F3"], t[idx], w[idx])
    mean = sample.mean(0, keepdim=True); std = sample.std(0, keepdim=True) + 1e-6
    reg_t = ds["region"][t, w]; pos_t = ds["pos"][t, w].float()
    pos_w = ((pos_t.numel() - pos_t.sum()) / (pos_t.sum() + 1e-9)).clamp(1, 20)
    net = Net(WIN * ds["bdim"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce = nn.CrossEntropyLoss()
    B = 8192
    for ep in range(epochs):
        net.train(); perm = torch.randperm(t.shape[0], device=DEV)
        tot = 0.0
        for i in range(0, t.shape[0], B):
            b = perm[i:i + B]
            feat = gather(ds["F3"], t[b], w[b], mean, std)
            dlog, rlog = net(feat)
            ld = bce(dlog, pos_t[b])
            pm = pos_t[b] > 0.5
            lr = ce(rlog[pm], reg_t[b][pm]) if pm.any() else 0.0 * dlog.sum()
            loss = ld + lr
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{epochs} loss {tot/(t.shape[0]//B+1):.3f}", flush=True)
    return net, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--feat", choices=["proprio", "resid", "both"],
                    default="proprio", help="input channel(s) for the estimator")
    ap.add_argument("--val_cols", type=int, default=2,
                    help="held-out worker columns of the train set for in-domain val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    train_ds = load_multi(args.train, args.feat)
    train_bases = {os.path.basename(p) for p in args.train}
    W = train_ds["W"]
    cols_train = list(range(W - args.val_cols))
    cols_val = list(range(W - args.val_cols, W))
    print(f"TRAIN on {train_ds['path']} ({W} cols) feat={args.feat} "
          f"bdim={train_ds['bdim']}, "
          f"{len(cols_train)} train / {len(cols_val)} in-domain-val cols", flush=True)
    net, mean, std = train(train_ds, args.epochs, cols_train)

    rows = []
    # in-domain held-out columns of the (possibly combined) train set
    r = evaluate(net, train_ds, mean, std, cols=cols_val)
    r["set"] = f"[{train_ds['path']}] IN-DOMAIN val"; rows.append(r)
    # eval sets: full dataset; tag whether it was part of training
    for ep in args.eval:
        ds = load(ep, args.feat)
        seen = os.path.basename(ep) in train_bases
        tag = "SEEN-controller" if seen else "UNSEEN-controller"
        r = evaluate(net, ds, mean, std, cols=None)
        r["set"] = f"{ds['path']} (kp{ds['kp']}) [{tag}]"; rows.append(r)

    print("\n==================== CROSS-POLICY RESULTS ====================")
    print(f"{'eval set':<48} {'detF1':>6} {'prec':>5} {'rec':>5} "
          f"{'regAcc':>7} {'actAcc':>7} {'n_pos':>8}")
    for r in rows:
        print(f"{r['set']:<48} {r['det_f1']:6.3f} {r['det_prec']:5.2f} "
              f"{r['det_rec']:5.2f} {r['region_acc']:7.3f} {r['active_acc']:7.3f} "
              f"{r['n_pos']:8d}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
