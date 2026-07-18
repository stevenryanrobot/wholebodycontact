"""Export a v4 force-sensor checkpoint (MLP or GRU) to ONNX + meta json.

Mirrors the v3 export that produced data/wbc/sweep_v3/force_sensor_v3_best.onnx:
  torch.onnx.export(model, zeros(input_shape), path, input_names=['x'],
                    output_names=['det_logit','loc_logits','dir','mag'], dynamo=False)

Input contract (documented in the meta json, key "input_layout"):
  - MLP ("v3" arch):     x = [1, window*base_dim]  — W normalized frames,
                         oldest..newest, flattened (same as v3 deployment).
  - GRU ("gru_v4" arch): x = [1, window, base_dim] — W normalized frames as a
                         sequence, oldest..newest.
Normalization (x_mean/x_std, per base-dim frame) is done OUTSIDE the ONNX by
the deployment, as in v3. If the checkpoint was trained with impedance-aware
torque normalization (meta "imp_norm": true), the deployment must first replace
the torque channels per frame:
  tau_i / (kp_i*|q_des_i - q_i| + kd_i*|qd_i| + eps)
using the meta's imp_kp/imp_kd/imp_qdef/imp_eps/ctrl_dt/imp_layout fields
(q = jp0 + qdef, qd = (jp0 - jp1)/ctrl_dt), BEFORE applying x_mean/x_std.

    python forcesense/export.py --ckpt data/wbc/sweep_v4/force_sensor_v4_best.pt
"""
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forcesense.models import ForceSensorV3, ContactGRUv4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default=None,
                   help="Output .onnx path (default: <ckpt without .pt>.onnx)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "v3")
    K = ckpt["num_bodies"]
    W = ckpt["window"]
    D = ckpt["base_dim"]

    if arch == "gru_v4":
        model = ContactGRUv4(D, K, ckpt["gru_hidden"], ckpt["gru_layers"])
        dummy = torch.zeros(1, W, D)
        input_layout = f"[1, {W}, {D}] normalized frames (oldest..newest)"
    else:
        model = ForceSensorV3(ckpt["in_dim"], K, ckpt["hidden"])
        dummy = torch.zeros(1, ckpt["in_dim"])
        input_layout = f"[1, {W}*{D}] normalized frames flattened (oldest..newest)"
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_path = args.out or (os.path.splitext(args.ckpt)[0] + ".onnx")
    torch.onnx.export(model, dummy, out_path, input_names=["x"],
                      output_names=["det_logit", "loc_logits", "dir", "mag"],
                      dynamo=False)

    meta = {
        "arch": arch,
        "input_layout": input_layout,
        "x_mean": ckpt["x_mean"].tolist(),
        "x_std": ckpt["x_std"].tolist(),
        "window": W,
        "base_dim": D,
        "in_dim": ckpt["in_dim"],
        "num_bodies": K,
        "body_names": ckpt["body_names"],
        "regions": ckpt["regions"],
        "det_thresh": ckpt["det_thresh"],
        "force_max": ckpt["force_max"],
        "contact_thresh": ckpt["contact_thresh"],
        "imp_norm": bool(ckpt.get("imp_norm", False)),
    }
    if arch == "gru_v4":
        meta.update({"gru_hidden": ckpt["gru_hidden"], "gru_layers": ckpt["gru_layers"]})
    if meta["imp_norm"]:
        meta.update({k: ckpt[k] for k in
                     ("imp_eps", "ctrl_dt", "imp_kp", "imp_kd", "imp_qdef",
                      "imp_layout", "isaac_joints")})
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # quick parity check
    import numpy as np
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        x = np.random.randn(*dummy.shape).astype(np.float32)
        ins = {"x": x}
        outs = sess.run(None, ins)
        with torch.no_grad():
            ref = model(torch.from_numpy(x))
        err = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref))
        print(f"[export] onnxruntime parity max_abs_err={err:.2e}")
    except ImportError:
        print("[export] onnxruntime not available; skipped parity check")

    print(f"[export] {args.ckpt} -> {out_path} + {meta_path} "
          f"(arch={arch}, K={K}, W={W}, imp_norm={meta['imp_norm']})")


if __name__ == "__main__":
    main()
