"""Whole-Body Contact (Plan A) — proprioceptive force-sensing dataset collector.

Drives an existing *low-level* G1 locomotion/tracking policy (which keeps the
robot upright) while random whole-body external forces are applied, and logs at
every step:

    X = wbc_input_   : proprioception only (joint torque, joint pos history,
                       commanded target, IMU history, prev actions)
    Y = wbc_label_   : net_pull_force_priv (sim ground truth: which body is being
                       pushed + the force vector + phase/magnitude)

The pairs are written to an HDF5 file consumed by scripts/train_force_sensor.py.
No RL training happens here — the robot policy is frozen.

Example:
    source start_gentle_local.sh
    python scripts/collect_force_data.py \
        -r luoxinyuan-duke-university/gentle_humanoid/<low_level_run_id> \
        -n 30000 --num_envs 64 -o data/wbc/wbc_train.h5

Pick a STIFF, non-compliant low-level policy for the strongest signal.
"""

import os
import sys
import argparse
import datetime
import itertools

import numpy as np
import torch
import wandb
import h5py
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FILE_PATH = os.path.dirname(__file__)
DEFAULT_OVERRIDES = os.path.join(FILE_PATH, "..", "cfg", "wbc", "collect.yaml")

# net_pull_force_priv layout (see motion_tracking.py:net_pull_force_priv).
# [ point_b(3) | force_b/Fmax(3) | force_w/Fmax(3) | body_onehot(K) | phase(4) | timer(1) | mag(1) ]
LABEL_PREFIX = 9          # point_b + force_b + force_w
LABEL_SUFFIX = 6          # phase(4) + timer(1) + mag(1)


def parse_args():
    p = argparse.ArgumentParser(description="Collect whole-body force-sensing data.")
    p.add_argument("-r", "--run_path", type=str, required=True,
                   help="wandb run path of the low-level driving policy.")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output .h5 path (default: data/wbc/wbc_<timestamp>.h5).")
    p.add_argument("-n", "--num_steps", type=int, default=20000,
                   help="Number of env steps to roll out (samples = num_steps * num_envs).")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--warmup", type=int, default=200,
                   help="Steps to discard at the start (let envs settle).")
    p.add_argument("--overrides", type=str, default=DEFAULT_OVERRIDES,
                   help="YAML with command/observation overrides to merge.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flush_every", type=int, default=500,
                   help="Flush buffered steps to disk every N steps.")
    return p.parse_args()


def load_run_cfg(run_path):
    """Download cfg + latest checkpoint for a wandb run (mirrors scripts/eval.py)."""
    api = wandb.Api()
    run = api.run(run_path)
    print(f"[collect] loading run {run.name}")
    root = os.path.join(FILE_PATH, "wandb", run.name)
    os.makedirs(root, exist_ok=True)

    checkpoints = []
    for f in run.files():
        if "checkpoint" in f.name:
            checkpoints.append(f)
        elif f.name in ("cfg.yaml", "files/cfg.yaml", "config.yaml"):
            f.download(root, replace=True)

    def sort_key(f):
        num = f.name[:-3].split("_")[-1]
        return 10**9 if num == "final" else int(num)

    checkpoints.sort(key=sort_key)
    ckpt = checkpoints[-1]
    print(f"[collect] downloading {ckpt.name}")
    ckpt.download(root, replace=True)

    try:
        cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    except FileNotFoundError:
        cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg["checkpoint_path"] = os.path.join(root, ckpt.name)
    return cfg


def unwrap_base_env(env):
    """Walk .base_env until we reach the SimpleEnv that owns command_manager."""
    e = env
    while not hasattr(e, "command_manager") and hasattr(e, "base_env"):
        e = e.base_env
    if not hasattr(e, "command_manager"):
        raise RuntimeError("Could not find command_manager on the env.")
    return e


def extract_force_body_names(base_env):
    cmd = base_env.command_manager
    asset = cmd.asset
    idx = cmd.net_pull_idx_asset.tolist()
    names = [asset.body_names[i] for i in idx]
    fmax = float(cmd.net_pull_force_range[1])
    return names, fmax


def main():
    args = parse_args()

    cfg = load_run_cfg(args.run_path)

    # Merge whole-body force + wbc observation-group overrides.
    overrides = OmegaConf.load(args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.task.command = OmegaConf.merge(cfg.task.command, overrides.command)
    if "observation" not in cfg.task or cfg.task.observation is None:
        cfg.task.observation = {}
    cfg.task.observation = OmegaConf.merge(cfg.task.observation, overrides.observation)

    cfg.task.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.app.headless = True
    if cfg.get("vecnorm", None) is not None:
        cfg.vecnorm = "eval"
    cfg.export_policy = False
    cfg.perf_test = False

    # Launch sim only after cfg is finalized (AppLauncher must precede env import).
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from scripts.utils.helpers import make_env_policy

    OmegaConf.resolve(cfg)
    env, policy, vecnorm, _ = make_env_policy(cfg)
    policy = policy.get_rollout_policy("eval")

    base_env = unwrap_base_env(env)
    body_names, force_max = extract_force_body_names(base_env)
    num_bodies = len(body_names)
    print(f"[collect] force bodies ({num_bodies}): {body_names}")
    print(f"[collect] force_max={force_max} N")

    out_path = args.output
    if out_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(FILE_PATH, "..", "data", "wbc", f"wbc_{ts}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[collect] writing to {out_path}")

    x_buf, y_buf = [], []
    total_written = 0
    h5 = None
    dset_x = dset_y = None

    def flush():
        nonlocal h5, dset_x, dset_y, total_written, x_buf, y_buf
        if not x_buf:
            return
        X = torch.cat(x_buf, dim=0).cpu().numpy().astype(np.float32)
        Y = torch.cat(y_buf, dim=0).cpu().numpy().astype(np.float32)
        x_buf, y_buf = [], []
        if h5 is None:
            h5 = h5py.File(out_path, "w")
            dset_x = h5.create_dataset("X", shape=(0, X.shape[1]), maxshape=(None, X.shape[1]),
                                       chunks=(4096, X.shape[1]), dtype="f4")
            dset_y = h5.create_dataset("Y", shape=(0, Y.shape[1]), maxshape=(None, Y.shape[1]),
                                       chunks=(4096, Y.shape[1]), dtype="f4")
            h5.attrs["num_bodies"] = num_bodies
            h5.attrs["force_max"] = force_max
            h5.attrs["body_names"] = np.array(body_names, dtype=h5py.special_dtype(vlen=str))
            h5.attrs["input_dim"] = X.shape[1]
            h5.attrs["label_dim"] = Y.shape[1]
            h5.attrs["label_prefix"] = LABEL_PREFIX
            h5.attrs["label_suffix"] = LABEL_SUFFIX
            h5.attrs["run_path"] = args.run_path
        n = X.shape[0]
        dset_x.resize(total_written + n, axis=0)
        dset_y.resize(total_written + n, axis=0)
        dset_x[total_written:total_written + n] = X
        dset_y[total_written:total_written + n] = Y
        total_written += n
        h5.flush()
        print(f"[collect] flushed, total samples = {total_written}")

    td_ = env.reset()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            td_ = policy(td_)
            if i >= args.warmup:
                x_buf.append(td_["wbc_input_"].detach().clone())
                y_buf.append(td_["wbc_label_"].detach().clone())
            td, td_ = env.step_and_maybe_reset(td_)

            if i >= args.warmup and (i - args.warmup) % args.flush_every == 0:
                flush()
            if i - args.warmup >= args.num_steps:
                break

    flush()
    if h5 is not None:
        h5.close()
    print(f"[collect] done. {total_written} samples -> {out_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
