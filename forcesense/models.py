"""Force-sensing model architectures.

ForceSensorV3: temporal-window MLP (flat W*D input); det/loc/dir/mag heads.
ContactGRUv4:  GRU over [B, W, D] normalized frames; same four heads off the
               final hidden state.

Split out of the v3/v4 trainer (see forcesense/train/core.py).
"""
import torch.nn as nn


class ForceSensorV3(nn.Module):
    """MLP on W stacked frames (flat W*D input); det/loc/dir/mag heads."""
    def __init__(self, in_dim, K, hidden, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for hsz in hidden:
            layers += [nn.Linear(d, hsz), nn.LayerNorm(hsz), nn.ELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hsz
        self.backbone = nn.Sequential(*layers)
        self.det_head = nn.Linear(d, 1)        # contact vs none (logit)
        self.loc_head = nn.Linear(d, K)        # which body/region (given contact)
        self.dir_head = nn.Linear(d, 3)        # force unit direction
        self.mag_head = nn.Linear(d, 1)        # force magnitude (normalized)

    def forward(self, x):
        z = self.backbone(x)
        return self.det_head(z), self.loc_head(z), self.dir_head(z), self.mag_head(z)


class ContactGRUv4(nn.Module):
    """GRU over [B, W, D] normalized frames; same four heads off the last
    hidden state (architecture per experiments/maze/sensor_model.py:ContactGRU,
    widened to the Isaac 320-dim frames and the v3 head set)."""
    def __init__(self, in_dim, K, hidden=384, layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.det_head = nn.Linear(hidden, 1)
        self.loc_head = nn.Linear(hidden, K)
        self.dir_head = nn.Linear(hidden, 3)
        self.mag_head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x [B,W,D]
        h, _ = self.gru(x)
        z = h[:, -1]
        return self.det_head(z), self.loc_head(z), self.dir_head(z), self.mag_head(z)
