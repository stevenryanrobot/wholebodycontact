"""Whole-Body Contact (Plan A) — v3/v4 trainer core.

v3 core: load-once [T,E,D] GPU grid, temporal-window MLP (ForceSensorV3), dual
det/loc heads + decoupled dir/mag force heads.
v4 additions: dynamic K (force-link width from h5), multi-file data, GRU arch
(ContactGRUv4), impedance-aware torque normalization (SixthSense Eq. 10).

Models live in forcesense/models.py; data/feature utils in
forcesense/common/data.py; region map in forcesense/common/regions.py.

Lib:  from forcesense.train.core import load_grid, run_experiment
CLI:  python -m forcesense.train.core --data data/wbc/datasets/wbc_v4_motion.h5 \
          data/wbc/datasets/wbc_v4_static.h5 --window 6 --regions
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from forcesense.models import ForceSensorV3, ContactGRUv4
from forcesense.common.data import (
    load_grid, make_targets, gather_frames, gather_window, build_index,
    impedance_constants, impedance_normalize,
    IMP_EPS, CTRL_DT, IMP_KP, IMP_KD, IMP_QDEF, IMP_LAYOUT, ISAAC_JOINTS,
)

# re-exported for callers that used the old flat module (backward compat)
__all__ = ["load_grid", "make_targets", "gather_frames", "gather_window",
           "build_index", "impedance_constants", "impedance_normalize",
           "ForceSensorV3", "ContactGRUv4", "focal_bce", "evaluate",
           "run_experiment", "main"]


def focal_bce(logit, target, gamma, pos_weight):
    """Binary focal loss; logit/target are [B]."""
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none",
                                            pos_weight=pos_weight)
    pt = torch.where(target > 0.5, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, grid, cls, active, K, W, t_idx, e_idx, det_thresh, xm=None, xs=None,
             bs=65536, featurize=None):
    """featurize(ti, ei) -> model-ready input; default = legacy flat-MLP path
    using (xm, xs) window-tiled stats (keeps viz_force_sensor_v3 working)."""
    if featurize is None:
        featurize = lambda ti, ei: (gather_window(grid["X"], ti, ei, W) - xm) / xs
    model.eval()
    force_b, mag = grid["force_b"], grid["mag"]
    dev = grid["X"].device
    preds, dets = [], []
    cos_sum = mr_sum = mr_n = 0.0
    cm = torch.zeros(K + 1, K + 1, dtype=torch.long)
    n = t_idx.shape[0]
    for s in range(0, n, bs):
        ti, ei = t_idx[s:s + bs], e_idx[s:s + bs]
        x = featurize(ti, ei)
        det, loc, dirh, magh = model(x)
        det_p = torch.sigmoid(det.squeeze(-1))
        loc_arg = loc.argmax(-1)
        pred = torch.where(det_p > det_thresh, loc_arg, torch.full_like(loc_arg, K))
        c = cls[ti, ei]
        a = active[ti, ei]
        preds.append((pred == c))
        for tcl, pcl in zip(c.tolist(), pred.tolist()):
            cm[tcl, pcl] += 1
        # force metrics on active frames
        if a.any():
            d_unit = F.normalize(dirh[a], dim=-1)
            f_true = force_b[ti, ei][a]
            t_unit = F.normalize(f_true, dim=-1)
            cos_sum += F.cosine_similarity(d_unit, t_unit, dim=-1).sum().item()
            mag_pred = magh.squeeze(-1)[a].clamp_min(0)
            mag_true = mag[ti, ei][a]
            mr_sum += ((mag_pred - mag_true).abs() / mag_true.clamp_min(1e-3)).sum().item()
            mr_n += int(a.sum().item())
        dets.append((det_p, c < K))
    correct = torch.cat(preds).float()
    overall = correct.mean().item()
    # active accuracy
    a_all = torch.cat([active[t_idx[s:s+bs], e_idx[s:s+bs]] for s in range(0, n, bs)])
    active_acc = correct[a_all].mean().item() if a_all.any() else float("nan")
    # detection f1 at threshold
    det_p = torch.cat([d[0] for d in dets]); true_c = torch.cat([d[1] for d in dets])
    pred_c = det_p > det_thresh
    tp = (pred_c & true_c).sum().item(); fp = (pred_c & ~true_c).sum().item()
    fn = (~pred_c & true_c).sum().item()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    cos = cos_sum / max(mr_n, 1); mr = mr_sum / max(mr_n, 1)
    recall = [(cm[i, i].item() / max(cm[i].sum().item(), 1)) for i in range(K)]
    return {"overall_acc": overall, "active_acc": active_acc, "det_f1": f1,
            "det_prec": prec, "det_rec": rec, "force_cos": cos,
            "mag_rel_err": mr, "recall": recall, "cm": cm.tolist()}


def run_experiment(cfg, grid=None):
    g = lambda k, d: cfg.get(k, d)
    device = g("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(g("seed", 0)); np.random.seed(g("seed", 0))
    if grid is None:
        grid = load_grid(cfg["data"], device, g("n_envs", 128))
    T, E, D = grid["T"], grid["E"], grid["D"]
    W = g("window", 1)
    arch = g("arch", "mlp")                     # "mlp" | "gru"
    imp = g("imp_norm", False)
    kp_t, kd_t, qdef_t = impedance_constants(device)

    cls, names, K, active = make_targets(grid, g("regions", False),
                                         g("contact_thresh", 0.05), device)
    t_all, e_all, is_val = build_index(T, E, W, g("val_envs", 16),
                                       grid.get("seg_bounds"))
    t_all, e_all, is_val = t_all.to(device), e_all.to(device), is_val.to(device)

    # ---- optional label cleaning on TRAIN frames only (keep all none) ----
    phase = grid["phase"]; mag = grid["mag"]
    keep = torch.ones_like(t_all, dtype=torch.bool)
    a_at = active[t_all, e_all]
    if g("drop_ramp", False):
        keep &= ~(a_at & (phase[t_all, e_all] != 2))
    if g("min_mag", 0.0) > 0:
        keep &= ~(a_at & (mag[t_all, e_all] < g("min_mag", 0.0)))
    tr_mask = keep & ~is_val
    t_tr, e_tr = t_all[tr_mask], e_all[tr_mask]
    t_va, e_va = t_all[is_val], e_all[is_val]

    # ---- per-feature input normalization from train frames (on the base D) ----
    Xg = grid["X"]
    sample = Xg[t_tr[:200000], e_tr[:200000]]
    if imp:
        sample = impedance_normalize(sample, kp_t, kd_t, qdef_t)
    x_mean = sample.mean(0, keepdim=True); x_std = sample.std(0, keepdim=True) + 1e-6
    del sample

    noise = g("input_noise", 0.0)

    if arch == "gru":
        def featurize(ti, ei):
            f = gather_frames(Xg, ti, ei, W)                     # [B,W,D]
            if imp:
                f = impedance_normalize(f, kp_t, kd_t, qdef_t)
            return (f - x_mean) / x_std                          # [1,D] broadcasts
        model = ContactGRUv4(D, K, g("gru_hidden", 384), g("gru_layers", 2),
                             g("dropout", 0.0)).to(device)
        in_dim = D
        eval_bs = g("eval_bs", 8192)
    else:
        x_mean_w = x_mean.repeat(1, W); x_std_w = x_std.repeat(1, W)
        def featurize(ti, ei):
            f = gather_frames(Xg, ti, ei, W)
            if imp:
                f = impedance_normalize(f, kp_t, kd_t, qdef_t)
            return (f.reshape(ti.shape[0], -1) - x_mean_w) / x_std_w
        model = ForceSensorV3(W * D, K, g("hidden", [512, 512, 256]),
                              g("dropout", 0.0)).to(device)
        in_dim = W * D
        eval_bs = g("eval_bs", 65536)

    # ---- detection pos_weight from train balance ----
    n_pos = int(a_at[tr_mask].sum().item()); n_neg = int(tr_mask.sum().item()) - n_pos
    pos_w = torch.tensor([max(n_neg, 1) / max(n_pos, 1)], device=device) if g("det_balance", True) \
        else torch.tensor([1.0], device=device)
    pos_w = pos_w.clamp(max=g("pos_weight_cap", 4.0))

    opt = torch.optim.AdamW(model.parameters(), lr=g("lr", 1e-3), weight_decay=g("weight_decay", 1e-5))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=g("epochs", 40))

    lam_loc = g("lambda_loc", 1.0); lam_dir = g("lambda_dir", 1.0); lam_mag = g("lambda_mag", 1.0)
    gamma = g("focal_gamma", 1.5); bs = g("batch_size", 16384)
    n_tr = t_tr.shape[0]
    force_b = grid["force_b"]
    det_thresh = g("det_thresh", 0.5)

    # ---- 1:4 pos:neg frame sampling (GRU / SixthSense recipe) ----
    pn_sample = g("pn_sample", arch == "gru")
    sep_loc = g("sep_loc_batch", False)   # r2: loc/dir/mag on own uniform pos batch
    loc_bs = g("loc_batch", 1024)
    pos_idx = a_at[tr_mask].nonzero(as_tuple=False).squeeze(-1)
    neg_idx = (~a_at[tr_mask]).nonzero(as_tuple=False).squeeze(-1)
    if (pn_sample or sep_loc) and (pos_idx.numel() == 0 or neg_idx.numel() == 0):
        print(f"[{cfg.get('label','v3')}] WARNING: one-sided train labels, "
              "disabling pos:neg / separate-loc sampling", flush=True)
        pn_sample = sep_loc = False
    steps_per_epoch = g("steps_per_epoch", 1000 if pn_sample else None)
    n_bpos = max(bs // 5, 1)                                       # 1:4

    def train_batches():
        if pn_sample:
            for _ in range(steps_per_epoch):
                bp = pos_idx[torch.randint(0, pos_idx.shape[0], (n_bpos,), device=device)]
                bn = neg_idx[torch.randint(0, neg_idx.shape[0], (bs - n_bpos,), device=device)]
                yield torch.cat([bp, bn])
        elif steps_per_epoch:               # natural distribution, fixed compute
            for _ in range(steps_per_epoch):
                yield torch.randint(0, n_tr, (bs,), device=device)
        else:
            perm = torch.randperm(n_tr, device=device)
            for s in range(0, n_tr, bs):
                yield perm[s:s + bs]

    # ---- r2: deployment-metric model selection (debounced honest F1) ----
    deploy_select = g("deploy_select", False)
    if deploy_select:
        from forcesense.common.metrics import (honest_labels, deploy_score,
                                               default_eval_slices)
        d_pos, d_valid = honest_labels(grid)
        d_slices = default_eval_slices(grid, W)

    best = {"score": -1}
    best_state = None
    for ep in range(g("epochs", 40)):
        model.train()
        for b in train_batches():
            ti, ei = t_tr[b], e_tr[b]
            x = featurize(ti, ei)
            if noise > 0:
                x = x + torch.randn_like(x) * noise
            det, loc, dirh, magh = model(x)
            a = active[ti, ei]                      # contact mask
            cl = cls[ti, ei]
            # detection
            det_loss = focal_bce(det.squeeze(-1), a.float(), gamma, pos_w)
            # localization/force: either on the batch's contact frames, or
            # (r2 sep_loc_batch) on an independently sampled uniform pos batch
            # so the det-oriented 1:4 sampling cannot skew/starve these heads.
            if sep_loc:
                bl = pos_idx[torch.randint(0, pos_idx.shape[0], (loc_bs,), device=device)]
                tl, el = t_tr[bl], e_tr[bl]
                xl = featurize(tl, el)
                if noise > 0:
                    xl = xl + torch.randn_like(xl) * noise
                _, loc_l, dir_l, mag_l = model(xl)
                loc_loss = F.cross_entropy(loc_l, cls[tl, el])
                t_unit = F.normalize(force_b[tl, el], dim=-1)
                d_unit = F.normalize(dir_l, dim=-1)
                dir_loss = (1 - F.cosine_similarity(d_unit, t_unit, dim=-1)).mean()
                mag_loss = F.smooth_l1_loss(mag_l.squeeze(-1), grid["mag"][tl, el])
            elif a.any():
                loc_loss = F.cross_entropy(loc[a], cl[a])
                f_true = force_b[ti, ei][a]
                t_unit = F.normalize(f_true, dim=-1)
                d_unit = F.normalize(dirh[a], dim=-1)
                dir_loss = (1 - F.cosine_similarity(d_unit, t_unit, dim=-1)).mean()
                mag_true = grid["mag"][ti, ei][a]
                mag_loss = F.smooth_l1_loss(magh.squeeze(-1)[a], mag_true)
            else:
                loc_loss = dir_loss = mag_loss = torch.zeros((), device=device)
            loss = det_loss + lam_loc * loc_loss + lam_dir * dir_loss + lam_mag * mag_loss
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % g("eval_every", 5) == 0 or ep == g("epochs", 40) - 1:
            m = evaluate(model, grid, cls, active, K, W, t_va, e_va, det_thresh,
                         bs=eval_bs, featurize=featurize)
            if deploy_select:
                ds = deploy_score(model, grid, featurize, W, g("val_envs", 16),
                                  d_pos, d_valid, slices=d_slices, bs=eval_bs)
                m.update({"deploy_f1": ds["f1"], "deploy_th": ds["th_hi"],
                          "deploy_prec": ds["prec"], "deploy_rec": ds["rec"]})
                sc = ds["f1"]
            else:
                sc = (m["active_acc"] or 0) + (m["det_f1"] or 0)
            if sc > best["score"]:
                best = {**m, "score": sc, "epoch": ep}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            dep = f" depF1={m['deploy_f1']:.3f}@{m['deploy_th']:.2f}" if deploy_select else ""
            print(f"[{cfg.get('label','v3')}] ep{ep:3d} act_acc={m['active_acc']:.3f} "
                  f"det_f1={m['det_f1']:.3f} cos={m['force_cos']:.3f} "
                  f"mag_err={m['mag_rel_err']:.2f} overall={m['overall_acc']:.3f}{dep}", flush=True)

    # ---- balanced-validation metrics on the best weights (GRU / pn runs) ----
    if g("balanced_eval", arch == "gru") and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        a_va = active[t_va, e_va]
        pi = a_va.nonzero(as_tuple=False).squeeze(-1)
        ni = (~a_va).nonzero(as_tuple=False).squeeze(-1)
        nb = min(pi.shape[0], ni.shape[0], 100_000)
        sel = torch.cat([pi[torch.randperm(pi.shape[0], device=device)[:nb]],
                         ni[torch.randperm(ni.shape[0], device=device)[:nb]]])
        mb = evaluate(model, grid, cls, active, K, W, t_va[sel], e_va[sel],
                      det_thresh, bs=eval_bs, featurize=featurize)
        best.update({"bal_active_acc": mb["active_acc"], "bal_det_f1": mb["det_f1"],
                     "bal_overall_acc": mb["overall_acc"]})
        print(f"[{cfg.get('label','v3')}] balanced-val: act_acc={mb['active_acc']:.3f} "
              f"det_f1={mb['det_f1']:.3f}", flush=True)

    if cfg.get("out") and best_state is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
        ckpt = {"state_dict": best_state,
                "arch": "gru_v4" if arch == "gru" else "v3",
                "x_mean": x_mean.cpu(), "x_std": x_std.cpu(),
                "hidden": g("hidden", [512, 512, 256]), "window": W,
                "in_dim": in_dim, "base_dim": D, "num_classes": K + 1, "num_bodies": K,
                "force_max": grid["force_max"], "body_names": names,
                "regions": g("regions", False), "det_thresh": det_thresh,
                "contact_thresh": g("contact_thresh", 0.05),
                "imp_norm": bool(imp)}
        if arch == "gru":
            ckpt.update({"gru_hidden": g("gru_hidden", 384),
                         "gru_layers": g("gru_layers", 2)})
        if imp:
            ckpt.update({"imp_eps": IMP_EPS, "ctrl_dt": CTRL_DT,
                         "imp_kp": IMP_KP.tolist(), "imp_kd": IMP_KD.tolist(),
                         "imp_qdef": IMP_QDEF.tolist(), "imp_layout": IMP_LAYOUT,
                         "isaac_joints": ISAAC_JOINTS})
        torch.save(ckpt, cfg["out"])
    return {"label": cfg.get("label", "v3"), "names": names, "K": K, "window": W,
            "arch": arch, "imp_norm": bool(imp), **best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+", default=["data/wbc/datasets/wbc_train.h5"])
    p.add_argument("--out", type=str, default="data/wbc/models/force_sensor_v3.pt")
    p.add_argument("--arch", type=str, default="mlp", choices=["mlp", "gru"])
    p.add_argument("--imp_norm", action="store_true")
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--regions", action="store_true")
    p.add_argument("--drop_ramp", action="store_true")
    p.add_argument("--min_mag", type=float, default=0.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    p.add_argument("--gru_hidden", type=int, default=384)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--val_envs", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = vars(args) | {"label": "cli"}
    cfg["data"] = args.data if len(args.data) > 1 else args.data[0]
    m = run_experiment(cfg)
    print(json.dumps({k: v for k, v in m.items() if k not in ("cm", "recall", "names")}, indent=2))


if __name__ == "__main__":
    main()
