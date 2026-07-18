"""Whole-Body Contact (Plan A, **v3** dual-head) — live Isaac Sim viewer.

A frozen low-level policy keeps the robot upright while an external force is applied
(random by default, or KEYBOARD-driven with --interactive), and every step the v3 MLP
estimates the contact. Drawn in the Isaac Sim viewport:
    RED   arrow = applied external force (at the pushed link)
    GREEN       = MLP prediction: the whole predicted REGION lights up + force arrow
    a floating RED/GREEN curve = applied |F| vs predicted |F| over the last N steps

v3 specifics: rolling [W,N,D] window of wbc_input_ frames; dual head (detection gates
localization/dir/mag); region-level localization.

--interactive controls (keyboard, focus the Isaac Sim window):
    1/2/3/4/5 : push left_arm / right_arm / left_leg / right_leg / trunk
    W/S       : +x / -x   A/D : +y / -y   Q/E : +z / -z   (world-frame push dir)
    -/=       : decrease / increase force magnitude (5..40 N)
    hold a direction key to push; release to let the force ramp back to 0.

Run on a machine with a display (this box has X on :1):

    source scripts/start_gentle_local.sh
    DISPLAY=:1 python scripts/play_force_sensor_v3.py \
        -r luoxinyuan-duke-university/gentle_humanoid/gentle_finetune_3point_amass_limmt_full_stiff30 \
        --model data/wbc/sweep_v3/force_sensor_v3_best.pt --num_envs 1 --interactive --slow 0.02
"""
import os
import sys
import time
import argparse
import itertools
from collections import deque

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
FILE_PATH = os.path.dirname(__file__)
DEFAULT_OVERRIDES = os.path.join(FILE_PATH, "..", "cfg", "collect.yaml")

from forcesense.collect.isaac import load_run_cfg, unwrap_base_env, extract_force_body_names
from forcesense.common.regions import region_of
from forcesense.models import ForceSensorV3

RED = (0.95, 0.15, 0.15, 1.0)
GREEN = (0.15, 0.9, 0.2, 1.0)
AXIS = (0.8, 0.8, 0.8, 0.6)
ARROW_SCALE = 0.02  # metres per Newton
CURVE_LEN = 200     # steps shown on the live curve
RAMP_RATE = 3.0     # N per step the applied force ramps toward target (smooth, in-distribution)

# region -> a representative net_pull link to push when that number key is pressed
REGION_REP = {
    "left_arm": "left_wrist_roll_link",
    "right_arm": "right_wrist_roll_link",
    "left_leg": "left_ankle_roll_link",
    "right_leg": "right_ankle_roll_link",
    "trunk": "torso_link",
}
KEY_REGION_ORDER = ["left_arm", "right_arm", "left_leg", "right_leg", "trunk"]  # keys 1..5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--run_path", type=str, required=True)
    p.add_argument("--model", type=str, default="data/wbc/sweep_v3/force_sensor_v3_best.pt")
    p.add_argument("--overrides", type=str, default=DEFAULT_OVERRIDES)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--det_thresh", type=float, default=None,
                   help="override detection threshold (default: ckpt value, 0.5). "
                        "Raise (e.g. 0.7) to cut false-positive green arrows.")
    p.add_argument("--slow", type=float, default=0.0,
                   help="seconds to sleep per step so arrows linger (try 0.05). "
                        "Default 0 = full speed.")
    p.add_argument("--demo_force", action="store_true",
                   help="(random mode) long hold + short rest so contact events stay on "
                        "screen and happen often.")
    p.add_argument("--interactive", action="store_true",
                   help="drive the external force with the keyboard instead of random.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_run_cfg(args.run_path)
    overrides = OmegaConf.load(args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.task.command = OmegaConf.merge(cfg.task.command, overrides.command)
    if "observation" not in cfg.task or cfg.task.observation is None:
        cfg.task.observation = {}
    cfg.task.observation = OmegaConf.merge(cfg.task.observation, overrides.observation)

    if args.demo_force and not args.interactive:
        cfg.task.command.net_pull_ramp_up_range = [5, 10]
        cfg.task.command.net_pull_hold_range = [150, 300]
        cfg.task.command.net_pull_ramp_down_range = [5, 10]
        cfg.task.command.net_pull_rest_range = [15, 30]

    cfg.task.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.app.headless = False            # GUI window
    if cfg.get("vecnorm", None) is not None:
        cfg.vecnorm = "eval"
    cfg.export_policy = False
    cfg.perf_test = False

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from torchrl.envs.utils import set_exploration_type, ExplorationType
    sys.path.insert(0, os.path.join(FILE_PATH, "..", "..", "controllers", "ceer"))  # framework scripts.utils lives here now
    from scripts.utils.helpers import make_env_policy
    from active_adaptation.utils.math import quat_apply, quat_apply_inverse

    OmegaConf.resolve(cfg)
    env, policy, vecnorm, _ = make_env_policy(cfg)
    policy = policy.get_rollout_policy("eval")
    base_env = unwrap_base_env(env)
    cmd = base_env.command_manager
    body_names, force_max = extract_force_body_names(base_env)   # the M net_pull links
    device = base_env.device

    # ---- load v3 MLP ----
    ckpt = torch.load(args.model, map_location=device)
    assert ckpt.get("arch") == "v3", f"not a v3 checkpoint: arch={ckpt.get('arch')}"
    W = ckpt["window"]
    base_dim = ckpt["base_dim"]
    K = ckpt["num_bodies"]
    region_names = ckpt["body_names"]            # class -> region name
    det_thresh = args.det_thresh if args.det_thresh is not None else ckpt["det_thresh"]
    model = ForceSensorV3(ckpt["in_dim"], K, ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    x_mean = ckpt["x_mean"].to(device); x_std = ckpt["x_std"].to(device)   # [1, base_dim]
    x_mean_w = x_mean.repeat(1, W); x_std_w = x_std.repeat(1, W)

    # ---- map model class <-> net_pull candidate links ----
    # Works for BOTH a region model (class = body region, e.g. left_arm) and a
    # link model (class = a specific link, e.g. left_elbow_link -> finer granularity).
    is_regions = ckpt.get("regions", False)
    class_names = region_names                                     # ckpt["body_names"]
    name_to_local = {n: m for m, n in enumerate(body_names)}
    class_idx = {c: i for i, c in enumerate(class_names)}
    class_to_locals = [[] for _ in class_names]                   # class -> net_pull link idx(es) to light up
    local_to_class = []                                           # net_pull link -> model class (K = not a class)
    for m, n in enumerate(body_names):
        c = class_idx.get(region_of(n)) if is_regions else class_idx.get(n)
        local_to_class.append(c if c is not None else K)
        if c is not None:
            class_to_locals[c].append(m)
    force_idx_asset = cmd.net_pull_idx_asset                       # [M] body ids in asset
    arange = torch.arange(base_env.num_envs, device=device)

    # ---- interactive keyboard setup ----
    kb = None
    if args.interactive:
        kb = _setup_keyboard(body_names, name_to_local)
        # stop the random scheduler from overwriting our injected force each step
        cmd.net_pull_schedule = lambda *a, **k: None
        cmd.net_pull_update_target = lambda *a, **k: None
        # keep the sim running continuously: force done=False at the SOURCE so it
        # doesn't matter how the internal wiring computes it. done = terminated |
        # truncated, truncated = (ep_len>=max_ep) | command.finished (base.py:497-505).
        # The ONLY reset is the keyboard R key.
        _N = base_env.num_envs
        base_env.max_episode_length = 10 ** 9                       # kill the timeout truncation
        base_env._compute_termination = lambda *a, **k: torch.zeros((_N, 1), dtype=torch.bool, device=device)
        if hasattr(cmd, "disable_motion_finish"):
            cmd.disable_motion_finish = True
        if hasattr(cmd, "finished"):                                # belt-and-suspenders: zero `finished`
            _orig_bu = cmd.before_update                            # AFTER the command updates it each step
            def _bu_no_finish(*a, _o=_orig_bu, **k):
                _o(*a, **k)
                cmd.finished[:] = False
            cmd.before_update = _bu_no_finish
        print("[play-v3] INTERACTIVE: 1-5 jump to a region | [ ] cycle exact link | "
              "WASD/QE dir | -/= magnitude | R = reset. Focus the Isaac Sim window.")

    gran = "regions (5)" if is_regions else "links (12, finer)"
    print(f"[play-v3] W={W} granularity={gran} classes={class_names} det_thresh={det_thresh}")
    print(f"[play-v3] RED=applied  GREEN=pred. Floating curve: RED |F| vs GREEN |F|.")

    buf = None                          # rolling [W, N, base_dim] window
    applied_hist = deque(maxlen=CURVE_LEN)
    pred_hist = deque(maxlen=CURVE_LEN)
    cur_mag = 0.0                       # smoothly-ramped applied magnitude (interactive)
    prev_el = 0                         # episode-length watchdog to catch unexpected resets

    td_ = env.reset()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            # ---- interactive: keyboard-triggered manual reset (R) ----
            if args.interactive and kb["take_reset"]():
                td_ = env.reset()
                buf = None
                applied_hist.clear(); pred_hist.clear()
                cur_mag = 0.0
                print("[play-v3] manual reset (R)")
            td_ = policy(td_)

            # ---- interactive: inject keyboard force into net_pull state ----
            if args.interactive:
                dir_w, target_mag, body_local = kb["read"]()
                cur_mag += max(-RAMP_RATE, min(RAMP_RATE, target_mag - cur_mag))
                cur_mag = max(0.0, min(force_max, cur_mag))
                fw = torch.tensor(dir_w, device=device, dtype=torch.float32) * cur_mag
                root_quat0 = base_env.scene["robot"].data.root_quat_w
                cmd.net_pull_body_local_idx[:] = body_local
                cmd.net_pull_force_w[:] = fw
                cmd.net_pull_force_b[:] = quat_apply_inverse(root_quat0, fw.expand(base_env.num_envs, 3))

            # ---- v3 inference (rolling window) ----
            frame = td_["wbc_input_"]                              # [N, base_dim]
            assert frame.shape[-1] == base_dim, \
                f"wbc_input_ dim {frame.shape[-1]} != ckpt base_dim {base_dim}"
            if buf is None:
                buf = frame.unsqueeze(0).repeat(W, 1, 1)
            else:
                buf = torch.roll(buf, -1, 0); buf[-1] = frame     # newest at the end
            x = buf.permute(1, 0, 2).reshape(base_env.num_envs, W * base_dim)
            x = (x - x_mean_w) / x_std_w
            det, loc, dirh, magh = model(x)
            det_p = torch.sigmoid(det.squeeze(-1))                 # [N]
            pred_region = loc.argmax(-1)                           # [N]
            pred_active = det_p > det_thresh
            dir_unit = torch.nn.functional.normalize(dirh, dim=-1)
            pred_mag = magh.squeeze(-1).clamp_min(0) * force_max   # [N]
            pred_force_b = dir_unit * (pred_mag * pred_active.float()).unsqueeze(-1)
            root_quat = base_env.scene["robot"].data.root_quat_w
            pred_force_w = quat_apply(root_quat, pred_force_b)     # [N,3] world

            # ---- ground truth / applied ----
            true_force_w = cmd.net_pull_force_w                    # [N,3] (= our applied force in interactive)
            true_body_local = cmd.net_pull_body_local_idx          # [N]
            body_pos_w = base_env.scene["robot"].data.body_pos_w   # [N,B,3]
            true_body_id = force_idx_asset[true_body_local]        # [N]
            true_pos = body_pos_w[arange, true_body_id]            # [N,3]

            td, td_ = env.step_and_maybe_reset(td_)

            # ---- curve history (env 0) ----
            applied_hist.append(float(true_force_w[0].norm().item()))
            pred_hist.append(float((pred_mag[0] * pred_active[0].float()).item()))

            # ---- draw ----
            dd = getattr(base_env, "debug_draw", None)
            if dd is not None:
                dd.clear()
                true_active = true_force_w.norm(dim=-1) > 1.0
                if true_active.any():
                    xp = true_pos[true_active]
                    dd.vector(xp, true_force_w[true_active] * ARROW_SCALE, size=4.0, color=RED)
                    dd.point(xp, color=RED, size=12.0)
                if pred_active.any():
                    env_sel = arange[pred_active]
                    centroids, region_pts = _class_viz(
                        pred_region[pred_active], class_to_locals,
                        force_idx_asset, body_pos_w, env_sel, device)
                    if region_pts.numel():
                        dd.point(region_pts, color=GREEN, size=16.0)
                    dd.vector(centroids, pred_force_w[pred_active] * ARROW_SCALE,
                              size=3.0, color=GREEN)
                    dd.point(centroids, color=GREEN, size=9.0)
                # floating live curve anchored beside env-0 robot
                anchor = body_pos_w[0, 0].clone()                 # root link
                anchor[1] += 0.9; anchor[2] += 0.4                # 0.9m to the side, a bit up
                _draw_curve(dd, anchor, applied_hist, pred_hist, force_max, device)

            if args.slow > 0:
                time.sleep(args.slow)

            # watchdog: if the episode-length counter drops, an auto-reset slipped through
            if args.interactive:
                el = int(base_env.episode_length_buf[0].item())
                if el < prev_el:
                    tru = bool(base_env.command_manager.finished[0].item()) if hasattr(base_env.command_manager, "finished") else None
                    print(f"[play-v3] !! unexpected RESET (ep_len {prev_el}->{el}); "
                          f"finished={tru} max_ep={base_env.max_episode_length}. Tell me this line.")
                prev_el = el

            if i % 100 == 0:
                msg = (f"[play-v3] step {i}  applied|F|={applied_hist[-1]:.1f}N  "
                       f"pred|F|={pred_hist[-1]:.1f}N  det_rate={pred_active.float().mean().item():.2f}")
                if args.interactive:
                    pushed = body_names[int(true_body_local[0].item())]
                    pcls = class_names[int(pred_region[0].item())] if bool(pred_active[0].item()) else "<none>"
                    msg += f"  push={pushed} -> pred={pcls}  cur_mag={cur_mag:.0f}N"
                print(msg)

    if kb is not None:
        kb["close"]()
    env.close()
    simulation_app.close()


def _class_viz(pred_cls, class_to_locals, force_idx_asset, body_pos_w, env_sel, device):
    """Per active env: centroid (for the arrow) + all link positions of the predicted class.
    For a region model that's the whole region's links; for a link model it's the single link."""
    centroids = torch.zeros(len(env_sel), 3, device=device)
    region_pts = []
    for j, (e, c) in enumerate(zip(env_sel.tolist(), pred_cls.tolist())):
        locals_ = class_to_locals[c]
        if not locals_:
            centroids[j] = body_pos_w[e, 0]
            continue
        ids = force_idx_asset[torch.tensor(locals_, device=device)]
        pts = body_pos_w[e, ids]
        centroids[j] = pts.mean(0)
        region_pts.append(pts)
    region_pts = torch.cat(region_pts, dim=0) if region_pts else torch.zeros(0, 3, device=device)
    return centroids, region_pts


def _draw_curve(dd, anchor, applied_hist, pred_hist, force_max, device, width=1.6, height=0.8):
    """Draw a floating RED(applied) / GREEN(predicted) |F| time-series as 3D polylines.
    Billboard lives in the world Y-Z plane at `anchor`; time -> +Y, magnitude -> +Z."""
    n = len(applied_hist)
    ax = anchor.tolist()
    # L-shaped axes
    dd.plot(torch.tensor([[ax[0], ax[1], ax[2]], [ax[0], ax[1] + width, ax[2]]]), size=2.0, color=AXIS)
    dd.plot(torch.tensor([[ax[0], ax[1], ax[2]], [ax[0], ax[1], ax[2] + height]]), size=2.0, color=AXIS)
    if n < 2:
        return
    k = torch.arange(n, dtype=torch.float32)
    ys = ax[1] + (k / (n - 1)) * width
    xs = torch.full((n,), ax[0])

    def line(hist, color):
        t = torch.tensor(list(hist), dtype=torch.float32).clamp(0.0, force_max)
        zs = ax[2] + (t / force_max) * height
        dd.plot(torch.stack([xs, ys, zs], dim=-1), size=3.0, color=color)

    line(applied_hist, RED)
    line(pred_hist, GREEN)


def _setup_keyboard(body_names, name_to_local):
    """Subscribe to carb keyboard events; return {'read', 'take_reset', 'close'}.
    Push target is a net_pull LOCAL link index: keys 1-5 jump to a body region's
    representative link, and [ / ] cycle through ALL candidate links one by one."""
    import carb
    import omni.appwindow

    KI = carb.input.KeyboardInput
    ET = carb.input.KeyboardEventType
    # number keys 1..5 -> representative link (local idx) for each region
    key_to_local = {}
    for n, region in enumerate(KEY_REGION_ORDER):
        key_to_local[getattr(KI, f"KEY_{n + 1}")] = name_to_local.get(REGION_REP[region], 0)

    state = {"pressed": set(), "body_local": 0, "mag": 25.0, "reset_req": False}

    def announce():
        print(f"[kbd] push link -> {body_names[state['body_local']]}")

    def on_kbd(event, *a):
        t, ki = event.type, event.input
        if t in (ET.KEY_PRESS, ET.KEY_REPEAT):
            state["pressed"].add(ki)
            if ki in key_to_local:
                state["body_local"] = key_to_local[ki]; announce()
            elif ki == KI.RIGHT_BRACKET:
                state["body_local"] = (state["body_local"] + 1) % len(body_names); announce()
            elif ki == KI.LEFT_BRACKET:
                state["body_local"] = (state["body_local"] - 1) % len(body_names); announce()
            elif ki in (KI.EQUAL, KI.NUMPAD_ADD):
                state["mag"] = min(40.0, state["mag"] + 5.0); print(f"[kbd] mag={state['mag']:.0f}N")
            elif ki in (KI.MINUS, KI.NUMPAD_SUBTRACT):
                state["mag"] = max(5.0, state["mag"] - 5.0); print(f"[kbd] mag={state['mag']:.0f}N")
            elif ki == KI.R:
                state["reset_req"] = True
        elif t == ET.KEY_RELEASE:
            state["pressed"].discard(ki)
        return True

    appwin = omni.appwindow.get_default_app_window()
    kbd = appwin.get_keyboard()
    inp = carb.input.acquire_input_interface()
    sub = inp.subscribe_to_keyboard_events(kbd, on_kbd)

    DIRS = {KI.W: (1, 0, 0), KI.S: (-1, 0, 0), KI.A: (0, 1, 0),
            KI.D: (0, -1, 0), KI.Q: (0, 0, 1), KI.E: (0, 0, -1)}

    def read():
        d = [0.0, 0.0, 0.0]
        for ki, vec in DIRS.items():
            if ki in state["pressed"]:
                d[0] += vec[0]; d[1] += vec[1]; d[2] += vec[2]
        norm = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
        if norm < 1e-6:
            return (0.0, 0.0, 0.0), 0.0, state["body_local"]      # no direction -> force ramps to 0
        return (d[0] / norm, d[1] / norm, d[2] / norm), state["mag"], state["body_local"]

    def take_reset():
        if state["reset_req"]:
            state["reset_req"] = False
            return True
        return False

    def close():
        try:
            inp.unsubscribe_to_keyboard_events(kbd, sub)
        except Exception:
            pass

    return {"read": read, "close": close, "take_reset": take_reset}


if __name__ == "__main__":
    main()
