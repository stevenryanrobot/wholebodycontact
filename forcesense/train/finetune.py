"""Track C2: fine-tune the deployment champion on Isaac + MuJoCo mixed data.

Closes the sim2sim domain gap (v4 sensors read MuJoCo standing-at-rest as
det~0.7): fine-tunes the champion from data/wbc/sweep_v4/champion.json on
Isaac motion + Isaac static + MuJoCo (wbc_collect_mujoco.py) with the MuJoCo
file oversampled, lr 1e-4, selection by debounced honest deploy-F1 on a
HELD-OUT MuJoCo time slice (last 10%) — the metric that matches the demo.

Success gates (reported PASS/FAIL, before vs after):
  - MuJoCo rest det plateau < 0.2   (mean det_p, rest frames >1s after release)
  - MuJoCo push det >= 0.85 at th 0.5 (hold frames, |F| > 5 N)
  - trunk-push det recovered (same, trunk links only)
Isaac val metrics are re-reported to confirm no catastrophic forgetting.

Exports data/wbc/sweep_v4/force_sensor_v4c.pt + .onnx + meta (with the
recommended operating point).
"""
import os
import sys
import json
import copy
import argparse
import subprocess

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from forcesense.common.regions import region_of
from forcesense.common.data import (load_grid, make_targets, gather_frames,
                                     impedance_constants, impedance_normalize)
from forcesense.models import ForceSensorV3, ContactGRUv4
from forcesense.train.core import evaluate, focal_bce
from forcesense.common.metrics import (honest_labels, deploy_score, debounce_scan,
                                       predict_grids, prf)


def build(ckpt, grid, device):
    """Model + featurize(ti,ei) bound to one grid (stats frozen from ckpt)."""
    arch = ckpt.get("arch", "v3")
    K, W, D = ckpt["num_bodies"], ckpt["window"], ckpt["base_dim"]
    imp = bool(ckpt.get("imp_norm", False))
    kp_t, kd_t, qdef_t = impedance_constants(device)
    xm = ckpt["x_mean"].to(device); xs = ckpt["x_std"].to(device)
    Xg = grid["X"]
    if arch == "gru_v4":
        def feat(ti, ei):
            f = gather_frames(Xg, ti, ei, W)
            if imp:
                f = impedance_normalize(f, kp_t, kd_t, qdef_t)
            return (f - xm) / xs
    else:
        xmw, xsw = xm.repeat(1, W), xs.repeat(1, W)
        def feat(ti, ei):
            f = gather_frames(Xg, ti, ei, W)
            if imp:
                f = impedance_normalize(f, kp_t, kd_t, qdef_t)
            return (f.reshape(ti.shape[0], -1) - xmw) / xsw
    return feat


def make_model(ckpt, device):
    arch = ckpt.get("arch", "v3")
    K = ckpt["num_bodies"]
    if arch == "gru_v4":
        model = ContactGRUv4(ckpt["base_dim"], K, ckpt["gru_hidden"], ckpt["gru_layers"])
    else:
        model = ForceSensorV3(ckpt["in_dim"], K, ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


@torch.no_grad()
def mj_gate_metrics(model, grid, feat, W, t0, t1, device, th=0.5, deadzone_n=5.0,
                    cls_names=None, regions_flag=False):
    """Demo gates on a MuJoCo slice [t0,t1): rest plateau / push det / trunk det.
    If cls_names is given (the checkpoint's class list), also reports
    localization on hold frames: link-level accuracy (links models only) and
    region-aggregated accuracy (both — the heatmap's soft layer)."""
    E = grid["E"]
    det_p, loc_arg = predict_grids(model, grid, feat, W, E)    # full T (cheap: small grid)
    det_p, loc_arg = det_p[t0:t1], loc_arg[t0:t1]
    mag, phase = grid["mag"][t0:t1], grid["phase"][t0:t1]
    pos = mag > deadzone_n / grid["force_max"]
    # frames since last positive (for a clean rest plateau)
    T = pos.shape[0]
    tidx = torch.arange(T, device=device).unsqueeze(1).expand(T, E)
    marked = torch.where(pos, tidx, torch.full_like(tidx, -(10**9)))
    since = tidx - torch.cummax(marked, dim=0).values
    ok = ~torch.isnan(det_p)
    rest = (phase == 0) & (since > 50) & ok
    hold = (phase == 2) & pos & ok
    body_names = grid["body_names"]
    true_link = grid["body_idx"][t0:t1]
    trunk_mask = torch.tensor([0 if region_of(n) == "trunk" else 1
                               for n in body_names], device=device)
    trunk = hold & (trunk_mask[true_link] == 0)
    out = {
        "rest_det_plateau": float(det_p[rest].mean()) if rest.any() else float("nan"),
        "rest_fp_at_th": float((det_p[rest] > th).float().mean()) if rest.any() else float("nan"),
        "push_det_at_th": float((det_p[hold] > th).float().mean()) if hold.any() else float("nan"),
        "trunk_det_at_th": float((det_p[trunk] > th).float().mean()) if trunk.any() else float("nan"),
        "n_rest": int(rest.sum()), "n_hold": int(hold.sum()), "n_trunk": int(trunk.sum()),
    }
    if cls_names is not None and hold.any():
        reg_names = sorted(set(region_of(n) for n in body_names))
        reg_id = {r: i for i, r in enumerate(reg_names)}
        true_reg = torch.tensor([reg_id[region_of(n)] for n in body_names],
                                device=device)[true_link]
        if regions_flag:
            pred_reg = torch.tensor([reg_id[c] for c in cls_names], device=device)[loc_arg]
            out["link_acc_hold"] = float("nan")
        else:
            # links model: class ids == body ids (asserted upstream)
            out["link_acc_hold"] = float((loc_arg == true_link)[hold].float().mean())
            pred_reg = torch.tensor([reg_id[region_of(c)] for c in cls_names],
                                    device=device)[loc_arg]
        out["region_acc_hold"] = float((pred_reg == true_reg)[hold].float().mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", default="data/wbc/sweep_v4/champion.json")
    p.add_argument("--ckpt", default=None, help="override champion ckpt path")
    p.add_argument("--isaac_data", nargs="+",
                   default=["data/wbc/datasets/wbc_v4_motion.h5", "data/wbc/datasets/wbc_v4_static.h5"])
    p.add_argument("--mj_data", default="data/wbc/datasets/wbc_v4_mujoco.h5")
    p.add_argument("--out", default="data/wbc/sweep_v4/force_sensor_v4c.pt")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--steps_per_epoch", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mj_oversample", type=float, default=3.0)
    p.add_argument("--mj_holdout", type=float, default=0.10)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--mj_rest_frac", type=float, default=None,
                   help="If set (e.g. 0.6), this fraction of the mj sub-batch "
                        "is drawn from pure-rest frames (natural ~0.34).")
    p.add_argument("--pos_weight", type=float, default=4.0,
                   help="det BCE pos_weight; use ~1.0 when the goal is "
                        "suppressing rest false positives (mj-rest frames are "
                        "only ~9%% of a mixed batch; pos_weight=4 de-weights "
                        "their gradient a further 4x).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(0)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        champ = json.load(open(args.champion))
        ckpt_path = champ["champion"]["ckpt"]
    print(f"[v4c] fine-tuning champion: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    W = ckpt["window"]
    arch = ckpt.get("arch", "v3")
    bs = 2048 if arch == "gru_v4" else 16384
    eval_bs = 8192 if arch == "gru_v4" else 65536

    # ---- data ----
    isaac = load_grid(args.isaac_data, dev, n_envs=128)
    with h5py.File(args.mj_data, "r") as h:
        mj_envs = int(h.attrs["n_envs"])
    mj = load_grid([args.mj_data], dev, n_envs=mj_envs)
    assert mj["body_names"] == isaac["body_names"], "body onehot order mismatch!"
    print(f"[v4c] isaac T={isaac['T']}xE{isaac['E']}, mujoco T={mj['T']}xE{mj['E']}", flush=True)

    model = make_model(ckpt, dev)
    feat_i = build(ckpt, isaac, dev)
    feat_m = build(ckpt, mj, dev)

    regions_flag = bool(ckpt.get("regions", False))
    cthr = ckpt.get("contact_thresh", 0.05)
    cls_i, names, K, act_i = make_targets(isaac, regions_flag, cthr, dev)
    cls_m, _, _, act_m = make_targets(mj, regions_flag, cthr, dev)
    assert names == list(ckpt["body_names"]), "class set mismatch vs champion"

    # ---- indices ----
    # isaac: train on non-val envs (val envs 0..15 reserved for forgetting eval)
    from forcesense.common.data import build_index
    ti_a, ei_a, isval = build_index(isaac["T"], isaac["E"], W, 16, isaac["seg_bounds"])
    ti_a, ei_a, isval = ti_a.to(dev), ei_a.to(dev), isval.to(dev)
    t_tr_i, e_tr_i = ti_a[~isval], ei_a[~isval]
    t_va_i, e_va_i = ti_a[isval], ei_a[isval]
    # mujoco: last 10% of time held out
    t_cut = int(mj["T"] * (1 - args.mj_holdout))
    tm = torch.arange(W - 1, t_cut, device=dev)
    em = torch.arange(mj["E"], device=dev)
    tt, ee = torch.meshgrid(tm, em, indexing="ij")
    t_tr_m, e_tr_m = tt.reshape(-1), ee.reshape(-1)

    Ni, Nm = t_tr_i.shape[0], t_tr_m.shape[0]
    frac_m = (args.mj_oversample * Nm) / (args.mj_oversample * Nm + Ni)
    n_mj = max(int(round(bs * frac_m)), 1)
    n_is = bs - n_mj
    print(f"[v4c] batch mix: isaac {n_is} + mujoco {n_mj} "
          f"(effective mj weight x{args.mj_oversample})", flush=True)

    # optional rest-frame boost: draw a fixed fraction of the mj sub-batch
    # from pure-rest (phase 0) frames — the negatives the demo lives in.
    mj_rest_mask = (mj["phase"][t_tr_m, e_tr_m] == 0)
    mj_rest_idx = mj_rest_mask.nonzero(as_tuple=False).squeeze(-1)
    mj_act_idx = (~mj_rest_mask).nonzero(as_tuple=False).squeeze(-1)
    if args.mj_rest_frac is not None:
        n_mj_rest = int(round(n_mj * args.mj_rest_frac))
        print(f"[v4c] mj sub-batch rest boost: {n_mj_rest}/{n_mj} rest frames "
              f"(natural rest frac {float(mj_rest_mask.float().mean()):.2f})", flush=True)

    pos_mj, valid_mj = honest_labels(mj)
    hold_slices = [(t_cut, mj["T"])]

    def deploy_eval():
        return deploy_score(model, mj, feat_m, W, mj["E"], pos_mj, valid_mj,
                            slices=hold_slices, ths=(0.4, 0.5, 0.6, 0.7, 0.85),
                            bs=eval_bs)

    def gates():
        return mj_gate_metrics(model, mj, feat_m, W, t_cut, mj["T"], dev,
                               cls_names=list(ckpt["body_names"]),
                               regions_flag=regions_flag)

    def isaac_eval():
        m = evaluate(model, isaac, cls_i, act_i, K, W, t_va_i, e_va_i,
                     ckpt.get("det_thresh", 0.5), bs=eval_bs, featurize=feat_i)
        return {k: m[k] for k in ("active_acc", "det_f1", "force_cos", "mag_rel_err")}

    before = {"deploy": deploy_eval(), "gates": gates(), "isaac": isaac_eval()}
    print(f"[v4c] BEFORE: deployF1={before['deploy']['f1']:.3f} "
          f"rest_plateau={before['gates']['rest_det_plateau']:.3f} "
          f"push_det={before['gates']['push_det_at_th']:.3f} "
          f"trunk_det={before['gates']['trunk_det_at_th']:.3f} "
          f"isaac={before['isaac']}", flush=True)

    # ---- training ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pos_w = torch.tensor([args.pos_weight], device=dev)
    best = {"score": -1.0}
    best_state = None

    def sub_loss(grid, cls, act, feat, ti, ei):
        x = feat(ti, ei)
        det, loc, dirh, magh = model(x)
        a = act[ti, ei]
        det_loss = focal_bce(det.squeeze(-1), a.float(), 1.5, pos_w)
        if a.any():
            loc_loss = F.cross_entropy(loc[a], cls[ti, ei][a])
            t_unit = F.normalize(grid["force_b"][ti, ei][a], dim=-1)
            d_unit = F.normalize(dirh[a], dim=-1)
            dir_loss = (1 - F.cosine_similarity(d_unit, t_unit, dim=-1)).mean()
            mag_loss = F.smooth_l1_loss(magh.squeeze(-1)[a], grid["mag"][ti, ei][a])
        else:
            loc_loss = dir_loss = mag_loss = torch.zeros((), device=dev)
        return det_loss + loc_loss + dir_loss + mag_loss

    for ep in range(args.epochs):
        model.train()
        for _ in range(args.steps_per_epoch):
            bi = torch.randint(0, Ni, (n_is,), device=dev)
            if args.mj_rest_frac is not None:
                br = mj_rest_idx[torch.randint(0, mj_rest_idx.shape[0], (n_mj_rest,), device=dev)]
                ba = mj_act_idx[torch.randint(0, mj_act_idx.shape[0], (n_mj - n_mj_rest,), device=dev)]
                bm = torch.cat([br, ba])
            else:
                bm = torch.randint(0, Nm, (n_mj,), device=dev)
            li = sub_loss(isaac, cls_i, act_i, feat_i, t_tr_i[bi], e_tr_i[bi])
            lm = sub_loss(mj, cls_m, act_m, feat_m, t_tr_m[bm], e_tr_m[bm])
            loss = (li * n_is + lm * n_mj) / bs
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            ds = deploy_eval(); gt = gates(); ie = isaac_eval()
            print(f"[v4c] ep{ep:2d} deployF1={ds['f1']:.3f}@{ds['th_hi']:.2f} "
                  f"rest={gt['rest_det_plateau']:.3f} push={gt['push_det_at_th']:.3f} "
                  f"trunk={gt['trunk_det_at_th']:.3f} "
                  f"linkAcc={gt.get('link_acc_hold', float('nan')):.3f} "
                  f"regAcc={gt.get('region_acc_hold', float('nan')):.3f} "
                  f"| isaac act={ie['active_acc']:.3f} "
                  f"f1={ie['det_f1']:.3f}", flush=True)
            if ds["f1"] > best["score"]:
                best = {"score": ds["f1"], "epoch": ep, "deploy": ds, "gates": gt, "isaac": ie}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    after = best
    g = after["gates"]
    gate_pass = {
        "rest_plateau_lt_0.2": g["rest_det_plateau"] < 0.2,
        "push_det_ge_0.85": g["push_det_at_th"] >= 0.85,
        "trunk_det_recovered": g["trunk_det_at_th"] >= 0.85,
    }
    print(f"\n[v4c] GATES: {json.dumps(gate_pass)}", flush=True)
    print(f"[v4c] before -> after: rest {before['gates']['rest_det_plateau']:.3f}->"
          f"{g['rest_det_plateau']:.3f}  push {before['gates']['push_det_at_th']:.3f}->"
          f"{g['push_det_at_th']:.3f}  trunk {before['gates']['trunk_det_at_th']:.3f}->"
          f"{g['trunk_det_at_th']:.3f}  isaac act "
          f"{before['isaac']['active_acc']:.3f}->{after['isaac']['active_acc']:.3f} "
          f"f1 {before['isaac']['det_f1']:.3f}->{after['isaac']['det_f1']:.3f}", flush=True)

    out_ckpt = copy.deepcopy({k: v for k, v in ckpt.items() if k != "state_dict"})
    out_ckpt["state_dict"] = best_state
    out_ckpt["det_thresh"] = float(after["deploy"]["th_hi"])
    out_ckpt["finetune_v4c"] = {"before": before, "after": {k: after[k] for k in ("deploy", "gates", "isaac", "epoch")},
                                "gates_pass": gate_pass, "mj_data": args.mj_data,
                                "base_ckpt": ckpt_path, "lr": args.lr,
                                "mj_oversample": args.mj_oversample}
    out_ckpt["recommended_debounce"] = {"k": 3, "th_hi": float(after["deploy"]["th_hi"]),
                                        "th_lo": max(float(after["deploy"]["th_hi"]) - 0.15, 0.05)}
    torch.save(out_ckpt, args.out)
    print(f"[v4c] saved {args.out} (selected ep{after['epoch']})", flush=True)

    onnx_out = os.path.splitext(args.out)[0] + ".onnx"
    subprocess.run([sys.executable, "-u", os.path.join(REPO, "forcesense/export.py"),
                    "--ckpt", args.out, "--out", onnx_out], check=True, cwd=REPO)
    meta_path = os.path.splitext(onnx_out)[0] + ".meta.json"
    meta = json.load(open(meta_path))
    meta["domain_finetune"] = "mujoco"
    meta["recommended"] = out_ckpt["recommended_debounce"] | {
        "gates_pass": gate_pass,
        "mj_rest_plateau": g["rest_det_plateau"], "mj_push_det": g["push_det_at_th"],
        "mj_trunk_det": g["trunk_det_at_th"]}
    json.dump(meta, open(meta_path, "w"))
    print(f"[v4c] exported {onnx_out} + meta (recommended op point embedded)", flush=True)


if __name__ == "__main__":
    main()
