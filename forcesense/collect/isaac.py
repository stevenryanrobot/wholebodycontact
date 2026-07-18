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
    python forcesense/collect/isaac.py \
        -r luoxinyuan-duke-university/gentle_humanoid/<low_level_run_id> \
        -n 30000 --num_envs 64 -o data/wbc/datasets/wbc_train.h5

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

FILE_PATH = os.path.dirname(__file__)
DEFAULT_OVERRIDES = os.path.join(FILE_PATH, "..", "..", "cfg", "wbc", "collect.yaml")

# net_pull_force_priv layout (see motion_tracking.py:net_pull_force_priv).
# [ point_b(3) | force_b/Fmax(3) | force_w/Fmax(3) | body_onehot(K) | phase(4) | timer(1) | mag(1) ]
LABEL_PREFIX = 9          # point_b + force_b + force_w
LABEL_SUFFIX = 6          # phase(4) + timer(1) + mag(1)

# Frozen standing wrist reference for --static_command, copied from
# forcesense/sim2sim.py:WRIST_REF_STILL (the web/MuJoCo demo's "still" mode):
# mean root_and_wrist_6d over ~47k standing-still frames of the training
# motion dataset. Layout: [l_pos_b(3), r_pos_b(3), l_axis_angle_b(3),
# r_axis_angle_b(3)] of the hand mimic bodies in the full root frame.
# Without this the 3-point policy keeps tracking the MOVING motion's wrist
# targets and the robot never stands still under a static root command.
WRIST_REF_STILL = [
    0.1395, 0.2587, -0.0435,     # left_hand_mimic pos_b
    0.1488, -0.2562, 0.0042,     # right_hand_mimic pos_b
    0.5543, 0.8831, -0.0440,     # left hand axis-angle (root-relative)
    -0.5530, 0.7476, -0.0262,    # right hand axis-angle
]


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
    p.add_argument("--static_command", action="store_true",
                   help="v4: collect with a STATIC standing root command instead of AMASS "
                        "motion tracking (matches the MuJoCo/web-demo deployment scenario). "
                        "Replaces the motion-tracking cum_error termination with a fall-only "
                        "check (a static robot ignores the reference motion, so cum_error "
                        "would fire constantly); falls still reset the episode, as in v3.")
    p.add_argument("--static_height", type=float, default=0.80,
                   help="Root height command in static mode (m).")
    # --- ProACT: active-probe (WIGGLE) mode -------------------------------- #
    p.add_argument("--probe", action="store_true",
                   help="ProACT active-sensing treatment: inject a scripted squat "
                        "(sinusoidal root-height oscillation) every step so the robot "
                        "MOVES during the sustained net_pull hold. The probe motion "
                        "sweeps the leg configuration under load -> changes the contact "
                        "Jacobian -> should make otherwise-unobservable leg/trunk "
                        "contacts observable. Only meaningful with --static_command; "
                        "compare against the passive --static_command baseline (no probe).")
    p.add_argument("--probe_amp", type=float, default=0.05,
                   help="Squat amplitude (m) for --probe (root-height +/- amp).")
    p.add_argument("--probe_period", type=int, default=60,
                   help="Squat period (env steps) for --probe (~1.2 s at 50 Hz).")
    p.add_argument("--probe_mode", type=str, default="squat",
                   choices=["squat", "shift", "both", "directed"],
                   help="Probe motion: 'squat' = vertical root-height bob; 'shift' = "
                        "open-loop lateral weight-shift oscillation (unloads legs "
                        "blindly); 'both' = squat+shift; 'directed' = ORACLE CLOSED-LOOP: "
                        "read the (ground-truth) contact body and steadily lean AWAY "
                        "from it so the contacted leg is unloaded for the whole hold -> "
                        "that leg's force stays observable. This is the closed-loop "
                        "upper bound (GT-directed); the real loop replaces GT with the "
                        "decoder's own estimate.")
    p.add_argument("--probe_lean_sign", type=float, default=1.0,
                   help="Sign of the directed lean (flip to +/-1 if legs get worse, "
                        "to match the body-y convention that unloads the contacted leg).")
    p.add_argument("--probe_vy", type=float, default=0.4,
                   help="Lateral velocity amplitude (m/s) for 'shift'/'both' modes.")
    return p.parse_args()


def load_run_cfg(run_path):
    """Download cfg + latest checkpoint for a wandb run (mirrors scripts/eval.py)."""
    api = wandb.Api()
    run = api.run(run_path)
    print(f"[collect] loading run {run.name}")
    root = os.path.join(FILE_PATH, "..", "..", "controllers", "ceer", "checkpoints", "wandb", run.name)
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

    # side lookup for 'directed' probe: +1 = left-side contact, -1 = right-side,
    # 0 = trunk/central. Indexed by net_pull_body_local_idx.
    side_lut = torch.tensor(
        [1.0 if n.startswith("left") else -1.0 if n.startswith("right") else 0.0
         for n in body_names], device=base_env.device)

    # ---- v4 static-standing mode ----------------------------------------- #
    # The command observation switches to a fixed (height, zero-vel, heading)
    # target via MotionTrackingCommand.set_static_root_command(). Because the
    # robot then ignores the reference motion, the run's cum_error termination
    # would fire every ~50 steps -> replace it with a fall-only check. Falls
    # still reset (consistent with v3 motion-mode collection); episode resets
    # re-init from a random motion frame, giving pose diversity before the
    # robot settles back into standing.
    cmd_mgr = base_env.command_manager
    if args.static_command:
        from active_adaptation.utils.math import quat_apply, yaw_quat, normalize
        asset = base_env.scene["robot"]
        _N = base_env.num_envs
        if hasattr(cmd_mgr, "disable_motion_finish"):
            cmd_mgr.disable_motion_finish = True   # no truncation when the (ignored) clip ends

        def _fall_only_termination(*a, **k):
            tilt = asset.data.projected_gravity_b[:, 2] > -0.5     # > ~60 deg from upright
            low = asset.data.root_pos_w[:, 2] < 0.45               # pelvis collapsed
            return (tilt | low).unsqueeze(1)

        base_env._compute_termination = _fall_only_termination
        _x_axis = torch.tensor([1.0, 0.0, 0.0], device=base_env.device)

        def _refresh_static_heading(mask: torch.Tensor):
            """Point the static heading target at the CURRENT yaw of the given
            envs (called after resets, which re-sample a random init yaw —
            otherwise the policy would spin the robot back to a stale heading)."""
            q = yaw_quat(asset.data.root_quat_w[mask])
            h = quat_apply(q, _x_axis.unsqueeze(0).expand(int(mask.sum().item()), 3))
            h[:, 2] = 0.0
            cmd_mgr.static_heading_w[mask] = normalize(h)

        print(f"[collect] STATIC mode: root height {args.static_height} m, fall-only termination")

    out_path = args.output
    if out_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(FILE_PATH, "..", "..", "data", "wbc", "datasets", f"wbc_{ts}.h5")
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
            h5.attrs["static_command"] = args.static_command
            h5.attrs["probe"] = args.probe
            h5.attrs["probe_mode"] = args.probe_mode
            h5.attrs["probe_amp"] = args.probe_amp
            h5.attrs["probe_period"] = args.probe_period
            h5.attrs["probe_vy"] = args.probe_vy
        n = X.shape[0]
        dset_x.resize(total_written + n, axis=0)
        dset_y.resize(total_written + n, axis=0)
        dset_x[total_written:total_written + n] = X
        dset_y[total_written:total_written + n] = Y
        total_written += n
        h5.flush()
        print(f"[collect] flushed, total samples = {total_written}")

    td_ = env.reset()
    prev_el = None
    if args.static_command:
        cmd_mgr.set_static_root_command(
            root_height=torch.full((base_env.num_envs, 1), args.static_height,
                                   device=base_env.device))
        # freeze the wrist targets too — this policy tracks root AND wrists
        if hasattr(cmd_mgr, "set_root_and_wrist_6d_command"):
            wrist_ref = torch.tensor(WRIST_REF_STILL, device=base_env.device)
            cmd_mgr.set_root_and_wrist_6d_command(
                wrist_ref.unsqueeze(0).expand(base_env.num_envs, 12))
            print("[collect] STATIC mode: wrist reference frozen (WRIST_REF_STILL)")
        # heading was captured from the post-reset yaw for all envs by
        # set_static_root_command's default; keep it fresh across future resets
        prev_el = base_env.episode_length_buf.clone()

    if args.probe and not args.static_command:
        print("[collect] WARNING: --probe has no effect without --static_command")
    if args.probe:
        print(f"[collect] PROBE (WIGGLE) mode: squat amp={args.probe_amp} m, "
              f"period={args.probe_period} steps")

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            # ProACT active probe: oscillate the commanded root height so the
            # frozen policy makes the robot squat during the sustained hold.
            # _static_root_command() re-reads static_root_height every step, so
            # writing it here is picked up by the next observation/step.
            if args.probe and args.static_command:
                ph = 2.0 * np.pi * i / args.probe_period
                if args.probe_mode in ("squat", "both"):
                    cmd_mgr.static_root_height[:] = float(
                        args.static_height + args.probe_amp * np.sin(ph))
                if args.probe_mode in ("shift", "both"):
                    # lateral weight shift: sway side-to-side so each leg is
                    # unloaded in turn (breaks double-support stance masking).
                    cmd_mgr.static_linvel_b[:, 0] = 0.0
                    cmd_mgr.static_linvel_b[:, 1] = float(args.probe_vy * np.sin(ph))
                if args.probe_mode == "directed":
                    # ORACLE closed-loop: lean AWAY from the contacted leg so it
                    # is unloaded for the whole hold. side=+1 (left contact) ->
                    # lean right (-y) to lift the left leg. Only during the hold
                    # phase (phase==2); neutral otherwise.
                    side = side_lut[cmd_mgr.net_pull_body_local_idx]          # [N]
                    holding = (cmd_mgr.net_pull_phase == 2).float()
                    cmd_mgr.static_linvel_b[:, 0] = 0.0
                    cmd_mgr.static_linvel_b[:, 1] = (
                        -args.probe_vy * args.probe_lean_sign * side * holding)
            td_ = policy(td_)
            if i >= args.warmup:
                x_buf.append(td_["wbc_input_"].detach().clone())
                y_buf.append(td_["wbc_label_"].detach().clone())
            td, td_ = env.step_and_maybe_reset(td_)

            if args.static_command:
                # episode-length watchdog: envs whose counter did not advance
                # by exactly 1 were reset this step -> re-aim their heading
                el = base_env.episode_length_buf
                reset_mask = el != (prev_el + 1)
                if reset_mask.any():
                    _refresh_static_heading(reset_mask)
                prev_el = el.clone()

            if i >= args.warmup and (i - args.warmup) % args.flush_every == 0:
                flush()
            if i - args.warmup >= args.num_steps:
                break

    flush()
    if h5 is not None:
        h5.close()
    print(f"[collect] done. {total_written} samples -> {out_path}")
    # Isaac's shutdown regularly hangs in env.close()/simulation_app.close()
    # (observed: process alive >5 min after `done.`, blocking pipeline stages).
    # The h5 is already closed above, so force-exit if graceful shutdown stalls.
    import threading
    threading.Timer(60.0, lambda: (print("[collect] shutdown hung, force-exit", flush=True),
                                   os._exit(0))).start()
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
