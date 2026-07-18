"""Cross-maze generalization: evaluate a sensor trained on maze A against maze B."""
from __future__ import annotations
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.maze.sensor_model import ContactGRU
from experiments.maze.train_sensor import load_grids, build_index, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="held-out maze h5(s)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev)
    model = ContactGRU(ck["in_dim"], ck["hidden"], ck["layers"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    H = ck["window"]
    mean, std = ck["x_mean"].to(dev), ck["x_std"].to(dev)

    grids = load_grids(args.data, dev)
    _, idx = build_index(grids, H, val_frac=1.0)      # every env is "validation"
    m = evaluate(model, grids, idx, H, mean, std)
    print(f"[cross-maze] windows={idx.shape[0]}  det P {m['prec']:.3f} R {m['rec']:.3f} "
          f"F1 {m['f1']:.3f} FA {m['fa_rate']:.3f} | az {m['az_deg']:.1f}° "
          f"bin12 {m['bin_acc']:.3f}")


if __name__ == "__main__":
    main()
