"""Data + feature utilities for the force sensor (shared library).

Split out of the v3/v4 trainer. Contents:
  - impedance-aware torque normalization (SixthSense Eq. 10) + G1 constants
  - load_grid:   load h5 file(s) once into a [T, E, *] GPU grid
  - make_targets: build class ids / names from a loaded grid
  - gather_frames / gather_window / build_index: windowed-batch indexing

Frame layout of the 320-dim wbc_input_ vector (forcesense/cfg/collect*.yaml order):
  [0:29]    applied_torque                (Isaac joint order)
  [29:174]  joint_pos_history steps 0..4  (step0 = newest; q - q_default)
  [174:203] applied_action                (ABSOLUTE joint target q_des)
  [203:218] root_ang_vel_history, [218:233] projected_gravity_history,
  [233:320] prev_actions x3
"""
import re

import numpy as np
import torch
import h5py

from forcesense.common.regions import region_of

ORIG_K = 12          # legacy fallback: force-link onehot width of the v3 file
LABEL_PREFIX = 9     # point_b(3) | force_b(3) | force_w(3)
LABEL_SUFFIX = 6     # phase(4) | timer(1) | mag(1)


# --------------------------------------------------------------------------- #
# impedance-aware torque normalization (SixthSense Eq. 10)
# Constants duplicated from forcesense/sim2sim.py (importing it pulls mujoco).
# Verified empirically on wbc_train.h5: tau ~ kp*(q_des-q) - kd*qd, corr .95-.98.
# --------------------------------------------------------------------------- #
ISAAC_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
NJ = len(ISAAC_JOINTS)
_KP_MAP = {
    ".*_hip_yaw_joint": 40.17923847137318, ".*_hip_pitch_joint": 40.17923847137318,
    ".*_hip_roll_joint": 99.09842777666113, ".*_knee_joint": 99.09842777666113,
    "waist_yaw_joint": 40.17923847137318,
    "waist_roll_joint": 28.50124619574858, "waist_pitch_joint": 28.50124619574858,
    ".*_ankle_pitch_joint": 28.50124619574858, ".*_ankle_roll_joint": 28.50124619574858,
    ".*_shoulder_pitch_joint": 14.25062309787429, ".*_shoulder_roll_joint": 14.25062309787429,
    ".*_shoulder_yaw_joint": 14.25062309787429, ".*_elbow_joint": 14.25062309787429,
    ".*_wrist_roll_joint": 14.25062309787429,
    ".*_wrist_pitch_joint": 16.77832748089279, ".*_wrist_yaw_joint": 16.77832748089279,
}
_KD_MAP = {
    ".*_hip_yaw_joint": 2.5578897650279457, ".*_hip_pitch_joint": 2.5578897650279457,
    ".*_hip_roll_joint": 6.3088018534966395, ".*_knee_joint": 6.3088018534966395,
    "waist_yaw_joint": 2.5578897650279457,
    "waist_roll_joint": 1.814445686584846, "waist_pitch_joint": 1.814445686584846,
    ".*_ankle_pitch_joint": 1.814445686584846, ".*_ankle_roll_joint": 1.814445686584846,
    ".*_shoulder_pitch_joint": 0.907222843292423, ".*_shoulder_roll_joint": 0.907222843292423,
    ".*_shoulder_yaw_joint": 0.907222843292423, ".*_elbow_joint": 0.907222843292423,
    ".*_wrist_roll_joint": 0.907222843292423,
    ".*_wrist_pitch_joint": 1.06814150219, ".*_wrist_yaw_joint": 1.06814150219,
}
_QDEF_MAP = {
    ".*_hip_pitch_joint": -0.28, ".*_knee_joint": 0.5, ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16, "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16, "right_shoulder_pitch_joint": 0.35,
}


def _resolve_regex_map(regex_map):
    out = np.zeros(NJ, dtype=np.float32)
    for i, j in enumerate(ISAAC_JOINTS):
        for pat, v in regex_map.items():
            if re.fullmatch(pat, j):
                out[i] = v
                break
    return out


IMP_KP = _resolve_regex_map(_KP_MAP)
IMP_KD = _resolve_regex_map(_KD_MAP)
IMP_QDEF = _resolve_regex_map(_QDEF_MAP)
IMP_EPS = 0.1
CTRL_DT = 0.02
IMP_LAYOUT = {"torque": [0, 29], "jp0": [29, 58], "jp1": [58, 87], "q_des": [174, 203]}


def impedance_constants(device):
    return (torch.from_numpy(IMP_KP).to(device),
            torch.from_numpy(IMP_KD).to(device),
            torch.from_numpy(IMP_QDEF).to(device))


def impedance_normalize(x, kp, kd, qdef, eps=IMP_EPS):
    """x [..., D] raw frames -> copy with torque channels impedance-normalized."""
    tq, jp0, jp1, aa = (IMP_LAYOUT["torque"], IMP_LAYOUT["jp0"],
                        IMP_LAYOUT["jp1"], IMP_LAYOUT["q_des"])
    q_err = x[..., aa[0]:aa[1]] - qdef - x[..., jp0[0]:jp0[1]]     # q_des - q
    qd = (x[..., jp0[0]:jp0[1]] - x[..., jp1[0]:jp1[1]]) / CTRL_DT
    denom = kp * q_err.abs() + kd * qd.abs() + eps
    out = x.clone()
    out[..., tq[0]:tq[1]] = x[..., tq[0]:tq[1]] / denom
    return out


# --------------------------------------------------------------------------- #
# data: load h5 file(s) once, reshape/concat to [T, E, *], keep on GPU
# --------------------------------------------------------------------------- #
def load_grid(path, device, n_envs=128, x_key="X"):
    """path: str or list of str. Multiple files are concatenated along the
    TIME axis (grid['seg_bounds'] marks the file boundaries for build_index).

    x_key selects the input-feature dataset: "X" = raw 320-dim proprioception
    (default), "R" = the 29-dim policy-invariant residual channel (controller-
    agnostic external joint-torque estimate; see docs/residual_method_explained).
    Labels always come from "Y"; only the input source changes."""
    paths = [path] if isinstance(path, str) else list(path)
    Xs, Ys, seg_bounds = [], [], [0]
    body_names = force_max = K = None
    for p in paths:
        with h5py.File(p, "r") as h:
            X = h[x_key][:]                        # [N, D]
            Y = h["Y"][:]                          # [N, 9+K+6]
            bn = [b.decode() if isinstance(b, bytes) else str(b)
                  for b in h.attrs["body_names"]]
            fm = float(h.attrs["force_max"])
            k = int(h.attrs.get("num_bodies", len(bn) or ORIG_K))
        assert Y.shape[1] == LABEL_PREFIX + k + LABEL_SUFFIX, \
            f"{p}: label width {Y.shape[1]} != 9+{k}+6"
        if body_names is None:
            body_names, force_max, K = bn, fm, k
        else:
            assert bn == body_names and k == K and fm == force_max, \
                f"{p}: incompatible body set vs {paths[0]}"
        N, D = X.shape
        T = N // n_envs
        assert T * n_envs == N, f"{p}: N={N} not divisible by n_envs={n_envs}"
        Xs.append(X[:T * n_envs].reshape(T, n_envs, D))
        Ys.append(Y[:T * n_envs].reshape(T, n_envs, Y.shape[1]))
        seg_bounds.append(seg_bounds[-1] + T)
    X = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
    Y = np.concatenate(Ys, axis=0) if len(Ys) > 1 else Ys[0]
    del Xs, Ys
    T = X.shape[0]

    # ---- parse labels (per time,env) ----
    force_b = Y[:, :, 3:6].astype(np.float32)              # base-frame force / Fmax
    onehot = Y[:, :, LABEL_PREFIX:LABEL_PREFIX + K]
    phase = Y[:, :, LABEL_PREFIX + K:LABEL_PREFIX + K + 4].argmax(-1)  # 0 rest 1 up 2 hold 3 down
    mag = Y[:, :, -1].astype(np.float32)
    body_idx = onehot.argmax(-1).astype(np.int64)

    grid = {
        "X": torch.from_numpy(X).to(device),                       # [T,E,D] f32
        "force_b": torch.from_numpy(force_b).to(device),           # [T,E,3]
        "phase": torch.from_numpy(phase).to(device),               # [T,E]
        "mag": torch.from_numpy(mag).to(device),                   # [T,E]
        "body_idx": torch.from_numpy(body_idx).to(device),         # [T,E]
        "T": T, "E": n_envs, "D": X.shape[-1], "K": K,
        "body_names": body_names, "force_max": force_max,
        "seg_bounds": seg_bounds,                                  # [0, T1, T1+T2, ...]
    }
    return grid


def make_targets(grid, regions, contact_thresh, device):
    """Build class ids + names from the loaded grid (no recompute of X)."""
    body_names = grid["body_names"]
    body_idx = grid["body_idx"]                       # [T,E]
    if regions:
        names = sorted(set(region_of(n) for n in body_names))
        reg_id = {r: i for i, r in enumerate(names)}
        map_t = torch.tensor([reg_id[region_of(n)] for n in body_names],
                             device=device, dtype=torch.long)
        loc = map_t[body_idx]
    else:
        names = list(body_names)
        loc = body_idx
    K = len(names)
    active = grid["mag"] > contact_thresh             # [T,E] bool
    cls = torch.where(active, loc, torch.full_like(loc, K))   # class K == none
    return cls, names, K, active


# --------------------------------------------------------------------------- #
# windowed-batch gather: input at (t,e) = stack of frames [t-W+1 .. t]
# --------------------------------------------------------------------------- #
def gather_frames(Xgrid, t_idx, e_idx, W):
    """Xgrid [T,E,D]; t_idx,e_idx [B]; returns [B, W, D] (oldest..newest)."""
    if W == 1:
        return Xgrid[t_idx, e_idx].unsqueeze(1)
    steps = torch.arange(-(W - 1), 1, device=t_idx.device)
    tt = t_idx.unsqueeze(1) + steps.unsqueeze(0)                 # [B,W]
    return Xgrid[tt, e_idx.unsqueeze(1)]


def gather_window(Xgrid, t_idx, e_idx, W):
    """Legacy flat gather: [B, W*D] (oldest..newest)."""
    if W == 1:
        return Xgrid[t_idx, e_idx]
    return gather_frames(Xgrid, t_idx, e_idx, W).reshape(t_idx.shape[0], -1)


def build_index(T, E, W, val_envs, seg_bounds=None):
    """Valid (t,e) pairs whose W-frame window stays inside one file segment;
    split by held-out env."""
    if seg_bounds is None:
        seg_bounds = [0, T]
    ts = [torch.arange(s0 + W - 1, s1)
          for s0, s1 in zip(seg_bounds[:-1], seg_bounds[1:])]
    t = torch.cat(ts)
    e = torch.arange(E)
    tt, ee = torch.meshgrid(t, e, indexing="ij")
    t_flat, e_flat = tt.reshape(-1), ee.reshape(-1)
    is_val = e_flat < val_envs
    return t_flat, e_flat, is_val
