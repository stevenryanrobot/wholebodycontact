"""Whole-body contact sensor v4 (mjlab): GRU over proprioception history.

Outputs per step: contact logit | azimuth (sin,cos) robot frame | magnitude.
Deployment wrapper keeps a rolling [H,N,D] buffer and converts predictions to
the 12 world-frame azimuth bins the Pledge controller consumes.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class ContactGRU(nn.Module):
    def __init__(self, in_dim=96, hidden=128, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.det = nn.Linear(hidden, 1)
        self.az = nn.Linear(hidden, 2)      # sin, cos (robot frame)
        self.mag = nn.Linear(hidden, 1)

    def forward(self, x):                    # x [B,H,D] normalized
        h, _ = self.gru(x)
        z = h[:, -1]
        return self.det(z).squeeze(-1), self.az(z), self.mag(z).squeeze(-1)


class RollingEstimator:
    """Online wrapper for nav: proprio stream -> 12 world bins (+ debug fields)."""

    def __init__(self, ckpt_path, device="cuda:0", det_thresh=None):
        ck = torch.load(ckpt_path, map_location=device)
        self.model = ContactGRU(ck["in_dim"], ck["hidden"], ck["layers"]).to(device)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        self.mean = ck["x_mean"].to(device); self.std = ck["x_std"].to(device)
        self.H = ck["window"]
        self.det_thresh = det_thresh if det_thresh is not None else ck.get("det_thresh", 0.5)
        self.buf = None
        self.last = {}

    def reset(self):
        self.buf = None

    @torch.inference_mode()
    def world_bins(self, feats, yaw):
        """feats [N,D] raw proprio; yaw [N]. Returns [N,12] bool world bins."""
        x = (feats - self.mean) / self.std
        if self.buf is None:
            self.buf = x.unsqueeze(1).repeat(1, self.H, 1)      # [N,H,D]
        else:
            self.buf = torch.roll(self.buf, -1, dims=1)
            self.buf[:, -1] = x
        det, az, mag = self.model(self.buf)
        p = torch.sigmoid(det)
        az_r = torch.atan2(az[:, 0], az[:, 1])
        az_w = az_r + yaw
        deg = torch.rad2deg(az_w) % 360.0
        idx = ((deg + 15.0) % 360.0 // 30.0).long()             # [N]
        N = feats.shape[0]
        bins = torch.zeros(N, 12, dtype=torch.bool, device=feats.device)
        on = p > self.det_thresh
        bins[torch.arange(N, device=feats.device)[on], idx[on]] = True
        self.last = {"p": p, "az_r": az_r, "mag": mag}
        return bins
