"""Whole-Body Contact (Plan A) — live Isaac Sim viewer for the force-sensing MLP.

Opens a GUI window: a frozen low-level policy keeps the robot upright while random
whole-body forces are applied, and every step the trained MLP estimates the contact.
Two arrows are drawn:
    RED   = ground-truth external force (at the truly-pushed link)
    GREEN = MLP-predicted force         (at the predicted link)
plus a small marker at each contact point. When they overlap, the sensor is right.

Run on a machine with a display (this repo's box has X on :1):

    source scripts/start_gentle_local.sh
    DISPLAY=:1 python scripts/play_force_sensor.py \
        -r luoxinyuan-duke-university/gentle_humanoid/gentle_finetune_3point_amass_limmt_full_stiff30 \
        --model data/wbc/force_sensor.pt --num_envs 4
"""
# ARCHIVED v1 — superseded by forcesense/ (kept for reference; not maintained).
import os
import sys
import argparse
import itertools

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)
DEFAULT_OVERRIDES = os.path.join(FILE_PATH, "..", "cfg", "wbc", "collect.yaml")

from scripts.collect_force_data import load_run_cfg, unwrap_base_env, extract_force_body_names
from scripts.train_force_sensor import ForceSensorMLP

RED = (0.95, 0.15, 0.15, 1.0)
GREEN = (0.15, 0.9, 0.2, 1.0)
ARROW_SCALE = 0.02  # metres per Newton


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--run_path", type=str, required=True)
    p.add_argument("--model", type=str, default="data/wbc/force_sensor.pt")
    p.add_argument("--overrides", type=str, default=DEFAULT_OVERRIDES)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
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
    from scripts.utils.helpers import make_env_policy
    from active_adaptation.utils.math import quat_apply

    OmegaConf.resolve(cfg)
    env, policy, vecnorm, _ = make_env_policy(cfg)
    policy = policy.get_rollout_policy("eval")
    base_env = unwrap_base_env(env)
    cmd = base_env.command_manager
    body_names, force_max = extract_force_body_names(base_env)
    device = base_env.device

    # load MLP
    ckpt = torch.load(args.model, map_location=device)
    model = ForceSensorMLP(ckpt["in_dim"], ckpt["num_classes"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    x_mean = ckpt["x_mean"].to(device); x_std = ckpt["x_std"].to(device)
    cls_names = ckpt["body_names"]      # class -> name (per-link model)
    none_cls = ckpt["num_classes"] - 1
    # map predicted class index -> index into net_pull bodies (for drawing position)
    name_to_force_idx = {n: i for i, n in enumerate(body_names)}
    pred_to_force_idx = [name_to_force_idx.get(cls_names[c], 0) for c in range(len(cls_names))]

    force_idx_asset = cmd.net_pull_idx_asset                       # [M] body ids in asset
    arange = torch.arange(base_env.num_envs, device=device)

    print(f"[play] model classes={cls_names}; drawing RED=true GREEN=pred. Close the window to stop.")

    td_ = env.reset()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            td_ = policy(td_)
            # ---- MLP inference ----
            x = (td_["wbc_input_"] - x_mean) / x_std
            logits, reg = model(x)
            pred_cls = logits.argmax(1)                             # [N]
            pred_force_b = reg * force_max                          # [N,3] base frame
            root_quat = base_env.scene["robot"].data.root_quat_w
            pred_force_w = quat_apply(root_quat, pred_force_b)      # [N,3] world

            # ---- ground truth ----
            true_body_local = cmd.net_pull_body_local_idx          # [N]
            true_force_w = cmd.net_pull_force_w                    # [N,3]
            body_pos_w = base_env.scene["robot"].data.body_pos_w   # [N,B,3]
            true_body_id = force_idx_asset[true_body_local]        # [N]
            true_pos = body_pos_w[arange, true_body_id]            # [N,3]

            td, td_ = env.step_and_maybe_reset(td_)

            # ---- draw (after step; persists until next step clears) ----
            dd = getattr(base_env, "debug_draw", None)
            if dd is not None:
                dd.clear()
                # true arrow (only where a real force is active)
                true_active = true_force_w.norm(dim=-1) > 1.0
                if true_active.any():
                    xp = true_pos[true_active]
                    dd.vector(xp, true_force_w[true_active] * ARROW_SCALE, size=4.0, color=RED)
                    dd.point(xp, color=RED, size=12.0)
                # predicted arrow (skip "no-contact" class)
                pred_active = pred_cls != none_cls
                if pred_active.any():
                    pidx = torch.tensor([pred_to_force_idx[c] for c in pred_cls[pred_active].tolist()],
                                        device=device)
                    env_sel = arange[pred_active]
                    pp = body_pos_w[env_sel, force_idx_asset[pidx]]
                    dd.vector(pp, pred_force_w[pred_active] * ARROW_SCALE, size=3.0, color=GREEN)
                    dd.point(pp, color=GREEN, size=9.0)

            if i % 100 == 0:
                acc = (pred_cls == _true_class(cmd, true_active, true_body_local, none_cls)).float().mean().item()
                print(f"[play] step {i}  live link-acc={acc:.2f}")

    env.close()
    simulation_app.close()


def _true_class(cmd, true_active, true_body_local, none_cls):
    # true class = pushed body when active, else the no-contact class
    return torch.where(true_active, true_body_local, torch.full_like(true_body_local, none_cls))


if __name__ == "__main__":
    main()
