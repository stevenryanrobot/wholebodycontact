"""
Evaluation script for manipulation tasks with teleoperation.

This script loads a trained policy and runs inference in the manipulation environment.
The policy and environment remain compatible - observation/action spaces are unchanged.

Usage (wandb):
    python scripts/eval_manipulation.py --run_path luoxinyuan-duke-university/gentle_humanoid/run_name -p

Usage (EE tracking eval):
    python scripts/eval_manipulation.py --run_path luoxinyuan-duke-university/gentle_humanoid/gentle_finetune_root_wrist_12 --objects cfg/objects/room_scene.yaml -p --full_collision --ee_tracking_eval --external_force off
    
Usage (local checkpoint):
    python scripts/eval_manipulation.py --checkpoint outputs/xxx/model.pt
"""

import torch
import wandb
import hydra
import argparse
import os
import sys
import json
import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type, ExplorationType
from scripts.utils.play import play
from scripts.utils.helpers import make_env_policy
from active_adaptation.utils.math import (
    clamp_norm,
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    normalize,
)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")
DEFAULT_EXTERNAL_FORCE_CFG = os.path.join(CONFIG_PATH, "eval", "external_force", "default_ee_net_pull.yaml")

EE_TRACKING_SEED = 0
EE_TRACKING_NUM_POINTS = 20
EE_TRACKING_RADIUS = 0.15
EE_TRACKING_HOLD_STEPS = 100
EE_TRACKING_WARMUP_STEPS = 50
EE_TRACKING_MAX_EPISODE_LENGTH = 1000000
EE_TRACKING_BODY_NAMES = "left_hand_mimic,right_hand_mimic"
EE_TRACKING_FEET_BODY_NAMES = "left_ankle_roll_link,right_ankle_roll_link"
EE_TRACKING_MIN_EE_CENTER_Z = 0.0
EE_TRACKING_DEFAULT_EE_CENTER_B = [
    [0.250, 0.180, 0.150],
    [0.250, -0.180, 0.150],
]


def _sample_uniform_ball(num_points: int, radius: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(num_points, 2, 3, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    magnitude = radius * torch.rand(num_points, 2, 1, generator=generator).pow(1.0 / 3.0)
    return direction * magnitude


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def _quat_to_rpy_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _body_pose_in_root_frame(asset, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1)
    body_pos_w = asset.data.body_pos_w[:, body_ids]
    body_quat_w = asset.data.body_quat_w[:, body_ids]
    root_quat_expanded = root_quat_w.expand(-1, len(body_ids), -1)
    pos_b = quat_apply_inverse(root_quat_expanded, body_pos_w - root_pos_w)
    quat_b = quat_mul(quat_conjugate(root_quat_expanded), body_quat_w)
    return pos_b, normalize(quat_b)


def _quat_angle_error_deg(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    actual = normalize(actual)
    target = normalize(target)
    dot = (actual * target).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def _summary(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "rmse": float(torch.sqrt((flat * flat).mean()).item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
        "std": float(flat.std(unbiased=False).item()),
    }


def _get_ee_compliance_params(cfg) -> dict:
    reward_cfg = OmegaConf.select(cfg, "task.reward.ee_compliance.ee_force_compliance_tracking")
    found = reward_cfg is not None
    reward_cfg = reward_cfg or {}
    return {
        "found_in_cfg": found,
        "stiffness": float(reward_cfg.get("stiffness", 60.0)),
        "max_offset": float(reward_cfg.get("max_offset", 0.25)),
        "force_deadband": float(reward_cfg.get("force_deadband", 2.0)),
    }


def _tensor_summary(value: torch.Tensor) -> dict:
    value = value.detach().float().reshape(-1).cpu()
    return {
        "mean": float(value.mean().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def _get_ee_compliance_eval_info(command_manager, params: dict) -> dict:
    if params["found_in_cfg"]:
        return {
            "target_mode": "explicit_force_over_stiffness",
            "actual_stiffness": params["stiffness"],
            "force_limit": None,
            "effective_stiffness": None,
        }

    if hasattr(command_manager, "force_keypoint_b"):
        force_limit = None
        effective_stiffness = None
        if hasattr(command_manager, "force_safe_limit_tl"):
            force_limit = _tensor_summary(command_manager.force_safe_limit_tl.current)
            effective_stiffness = {
                key: val / 0.05
                for key, val in force_limit.items()
            }
        return {
            "target_mode": "low_level_force_keypoint_b",
            "actual_stiffness": None,
            "force_limit": force_limit,
            "effective_stiffness": effective_stiffness,
        }

    return {
        "target_mode": "fallback_force_over_stiffness",
        "actual_stiffness": params["stiffness"],
        "force_limit": None,
        "effective_stiffness": None,
    }


def _compute_ee_compliance_target_b(command_manager, asset, body_ids: list[int], nominal_target_b: torch.Tensor, params: dict):
    force_w = torch.zeros_like(nominal_target_b)
    force_apply_idx = getattr(command_manager, "force_apply_idx_asset", None)
    force_applied_w = getattr(command_manager, "force_applied_w", None)

    if force_apply_idx is not None and force_applied_w is not None:
        force_apply_list = force_apply_idx.detach().cpu().tolist()
        for ee_i, body_i in enumerate(body_ids):
            if body_i in force_apply_list:
                force_i = force_apply_list.index(body_i)
                force_w[:, ee_i] = force_applied_w[:, force_i]

    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    force_b = quat_apply_inverse(root_quat, force_w)
    if not params["found_in_cfg"] and force_apply_idx is not None and hasattr(command_manager, "force_keypoint_b"):
        compliance_target_b = nominal_target_b.clone()
        force_apply_list = force_apply_idx.detach().cpu().tolist()
        active_force = force_b.norm(dim=-1, keepdim=True) > params["force_deadband"]
        for ee_i, body_i in enumerate(body_ids):
            if body_i in force_apply_list:
                force_i = force_apply_list.index(body_i)
                compliance_target_b[:, ee_i] = torch.where(
                    active_force[:, ee_i],
                    command_manager.force_keypoint_b[:, force_i],
                    nominal_target_b[:, ee_i],
                )
        target_offset_b = compliance_target_b - nominal_target_b
        return compliance_target_b, target_offset_b, force_b

    force_norm = force_b.norm(dim=-1, keepdim=True)
    active_force = torch.where(
        force_norm > params["force_deadband"],
        force_b,
        torch.zeros_like(force_b),
    )
    target_offset_b = clamp_norm(active_force / params["stiffness"], max=params["max_offset"])
    compliance_target_b = nominal_target_b + target_offset_b
    return compliance_target_b, target_offset_b, force_b


def _set_ee_command(command_manager, command: torch.Tensor):
    if hasattr(command_manager, "set_root_and_wrist_6d_command"):
        command_manager.set_root_and_wrist_6d_command(command)
        return
    raise RuntimeError(
        "EE tracking eval needs a command manager with set_root_and_wrist_6d_command(). "
        "Please use a high-level manipulation/root command task."
    )


def _is_hierarchical_action_manager(action_manager) -> bool:
    return hasattr(action_manager, "low_policy") and hasattr(action_manager, "_decode_ee_command")


def _set_ee_reference(command_manager, command: torch.Tensor):
    if hasattr(command_manager, "set_root_and_wrist_6d_reference_override"):
        command_manager.set_root_and_wrist_6d_reference_override(command)
        return
    _set_ee_command(command_manager, command)


def _set_ee_eval_target(command_manager, action_manager, command: torch.Tensor):
    if _is_hierarchical_action_manager(action_manager):
        _set_ee_reference(command_manager, command)
    else:
        _set_ee_command(command_manager, command)


def _set_external_force(command_manager, enabled: bool):
    if hasattr(command_manager, "set_external_force_enabled"):
        command_manager.set_external_force_enabled(enabled)
        return
    if not enabled:
        print(
            "[WARNING] --external_force off requested, but this command manager "
            "does not expose set_external_force_enabled().",
            flush=True,
        )


def _configure_external_force(cfg, enabled: bool):
    command_cfg = cfg["task"]["command"]
    command_target = command_cfg.get("_target_", "")
    if command_target.startswith("active_adaptation.envs.mdp.commands.motion_tracking.") and "impedance" in command_target:
        command_cfg["external_force_enabled"] = enabled
        print(f"  External force: {'on' if enabled else 'off'}")
        return True
    if not enabled:
        print(
            f"  External force: off requested, but command target does not support it: {command_target}"
        )
    return False


def _apply_external_force_mode(cfg, mode: str):
    if mode == "default":
        external_force_cfg = OmegaConf.load(DEFAULT_EXTERNAL_FORCE_CFG)
        command_overrides = external_force_cfg.get("command", {})
        cfg["task"]["command"].update(command_overrides)
        print(f"  External force config: default ({DEFAULT_EXTERNAL_FORCE_CFG})")

    enabled = mode != "off"
    return _configure_external_force(cfg, enabled)


def _uses_hierarchical_action_cfg(cfg) -> bool:
    action_cfg = cfg["task"].get("action", {})
    return action_cfg.get("_target_", "") == "active_adaptation.envs.mdp.action.HierarchicalRootCommand"


def _fix_low_level_force_limit_for_ee_eval(cfg):
    if _uses_hierarchical_action_cfg(cfg):
        return None
    if OmegaConf.select(cfg, "task.reward.ee_compliance.ee_force_compliance_tracking.stiffness") is not None:
        return None

    command_cfg = cfg["task"]["command"]
    if "force_safe_default" not in command_cfg:
        command_cfg["force_safe_default"] = 10.0
    force_limit = float(command_cfg["force_safe_default"])
    command_cfg["force_safe_bounds"] = [force_limit, force_limit]
    print(
        "  Low-level EE eval: fixed force_safe_limit "
        f"to default {force_limit:.2f} because no EE compliance stiffness config was found."
    )
    return force_limit


def _enable_root_passthrough_for_ee_only_hl(cfg):
    action_cfg = cfg["task"]["action"]
    action_target = action_cfg.get("_target_", "")
    if action_target != "active_adaptation.envs.mdp.action.HierarchicalRootCommand":
        return
    root_command_cfg = action_cfg.get("root_command", {})
    root_enabled = root_command_cfg.get("enabled", True)
    if root_enabled:
        return
    if "root_command" not in action_cfg or action_cfg["root_command"] is None:
        action_cfg["root_command"] = {}
    action_cfg["root_command"]["passthrough_reference"] = True
    print("  Root command: passthrough reference enabled for EE-only high-level teleop")


def _set_static_root_command(command_manager, asset):
    if hasattr(command_manager, "set_static_root_command"):
        command_manager.set_static_root_command(
            root_height=asset.data.root_pos_w[:, 2:3],
        )
        return
    if not hasattr(command_manager, "set_command_override"):
        return
    command = torch.zeros(command_manager.num_envs, 6, device=command_manager.device)
    command[:, 0] = asset.data.root_pos_w[:, 2]
    command[:, 3] = 1.0
    if hasattr(command_manager, "force_safe_limit_tl"):
        command[:, 5:6] = command_manager.force_safe_limit_tl.current
    command_manager.set_command_override(command)


def _set_default_feet_command(command_manager, asset):
    if not hasattr(command_manager, "set_feet_pos_b_command"):
        return False

    body_names = [name.strip() for name in EE_TRACKING_FEET_BODY_NAMES.split(",")]
    try:
        body_ids = [asset.body_names.index(name) for name in body_names]
    except ValueError:
        print(
            f"[WARNING] Cannot find feet body names {body_names}; skip feet command override.",
            flush=True,
        )
        return False

    feet_pos_b, _ = _body_pose_in_root_frame(asset, body_ids)
    command_manager.set_feet_pos_b_command(feet_pos_b.reshape(command_manager.num_envs, 6))
    return True


def _get_ee_sample_center(default_pos_b: torch.Tensor, device: torch.device) -> torch.Tensor:
    center = torch.tensor(EE_TRACKING_DEFAULT_EE_CENTER_B, dtype=torch.float32, device=device)
    center = center.unsqueeze(0).expand(default_pos_b.shape[0], -1, -1).clone()
    center[..., 2].clamp_min_(EE_TRACKING_MIN_EE_CENTER_Z)
    return center


def _disable_eval_timer_reset(base_env):
    base_env.episode_length_buf.zero_()
    if hasattr(base_env.command_manager, "finished"):
        base_env.command_manager.finished.zero_()


def _warn_if_done(tensordict, step_label: str):
    if "done" in tensordict.keys() and tensordict["done"].any():
        print(f"[WARNING] Env reset triggered during EE tracking eval at {step_label}.", flush=True)


def evaluate_ee_tracking(cfg, args):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app
    env = None

    try:
        env, policy, _vecnorm, _ = make_env_policy(cfg)
        rollout_policy = policy.get_rollout_policy("eval")
        base_env = env.base_env if hasattr(env, "base_env") else env
        asset = base_env.scene["robot"]
        command_manager = base_env.command_manager
        action_manager = base_env.action_manager
        _set_external_force(command_manager, args.external_force != "off")

        body_names = [name.strip() for name in EE_TRACKING_BODY_NAMES.split(",")]
        try:
            body_ids = [asset.body_names.index(name) for name in body_names]
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot find EE body name from {body_names}. Available bodies: {asset.body_names}"
            ) from exc
        if len(body_ids) != 2:
            raise RuntimeError(f"Expected exactly two EE body names, got {body_names}.")

        td_ = env.reset()
        _disable_eval_timer_reset(base_env)
        _set_static_root_command(command_manager, asset)
        print("Static root command enabled for EE tracking eval.", flush=True)
        if _set_default_feet_command(command_manager, asset):
            print("Default feet command enabled for EE tracking eval.", flush=True)
        default_pos_b, default_quat_b = _body_pose_in_root_frame(asset, body_ids)
        sample_center_b = _get_ee_sample_center(default_pos_b, base_env.device)
        compliance_params = _get_ee_compliance_params(cfg)
        compliance_eval_info = _get_ee_compliance_eval_info(command_manager, compliance_params)
        target_quat_b = torch.zeros_like(default_quat_b)
        target_quat_b[..., 0] = 1.0
        target_axis_angle_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)
        target_rpy_b = torch.zeros(base_env.num_envs, 2, 3, device=base_env.device)

        default_command = torch.cat(
            [sample_center_b.reshape(base_env.num_envs, 6), target_axis_angle_b.reshape(base_env.num_envs, 6)],
            dim=-1,
        )
        _set_ee_eval_target(command_manager, action_manager, default_command)
        if _is_hierarchical_action_manager(action_manager):
            print("EE tracking eval target is written as high-level EE reference override.", flush=True)
        else:
            print("EE tracking eval target is written as low-level EE command override.", flush=True)
        print(
            "EE sample center_b env0 "
            f"default={default_pos_b[0].detach().cpu().tolist()} "
            f"used={sample_center_b[0].detach().cpu().tolist()}",
            flush=True,
        )
        print("EE compliance target mode: " + compliance_eval_info["target_mode"], flush=True)
        if compliance_eval_info["actual_stiffness"] is not None:
            print(
                "EE compliance actual stiffness "
                f"{compliance_eval_info['actual_stiffness']:.2f}; "
                f"max_offset={compliance_params['max_offset']}, "
                f"force_deadband={compliance_params['force_deadband']} "
                f"(from_cfg={compliance_params['found_in_cfg']})",
                flush=True,
            )
        elif compliance_eval_info["effective_stiffness"] is not None:
            force_limit = compliance_eval_info["force_limit"]
            effective_stiffness = compliance_eval_info["effective_stiffness"]
            print(
                "EE compliance effective low-level stiffness "
                f"mean/min/max={effective_stiffness['mean']:.2f}/"
                f"{effective_stiffness['min']:.2f}/"
                f"{effective_stiffness['max']:.2f} N/m "
                f"from force_safe_limit mean/min/max={force_limit['mean']:.2f}/"
                f"{force_limit['min']:.2f}/"
                f"{force_limit['max']:.2f} N; "
                f"force_deadband={compliance_params['force_deadband']}",
                flush=True,
            )
        else:
            print(
                "EE compliance actual stiffness unavailable; "
                f"fallback stiffness={compliance_params['stiffness']}, "
                f"force_deadband={compliance_params['force_deadband']}",
                flush=True,
            )

        offsets = _sample_uniform_ball(
            EE_TRACKING_NUM_POINTS,
            EE_TRACKING_RADIUS,
            EE_TRACKING_SEED,
        ).to(base_env.device)
        target_pos_b = sample_center_b.unsqueeze(0) + offsets.unsqueeze(1)

        records = []
        print("Starting EE tracking rollout...", flush=True)
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for _ in range(EE_TRACKING_WARMUP_STEPS):
                td_ = rollout_policy(td_)
                td, td_ = env.step_and_maybe_reset(td_)
                _warn_if_done(td, "warmup")

            for point_idx in range(EE_TRACKING_NUM_POINTS):
                command = torch.cat(
                    [
                        target_pos_b[point_idx].reshape(base_env.num_envs, 6),
                        target_axis_angle_b.reshape(base_env.num_envs, 6),
                    ],
                    dim=-1,
                )
                _set_ee_eval_target(command_manager, action_manager, command)

                for _ in range(EE_TRACKING_HOLD_STEPS):
                    td_ = rollout_policy(td_)
                    td, td_ = env.step_and_maybe_reset(td_)
                    _warn_if_done(td, f"point {point_idx + 1}")

                actual_pos_b, actual_quat_b = _body_pose_in_root_frame(asset, body_ids)
                pos_error = (actual_pos_b - target_pos_b[point_idx]).norm(dim=-1)
                compliance_target_b, compliance_offset_b, force_b = _compute_ee_compliance_target_b(
                    command_manager,
                    asset,
                    body_ids,
                    target_pos_b[point_idx],
                    compliance_params,
                )
                compliance_pos_error = (actual_pos_b - compliance_target_b).norm(dim=-1)
                rpy_error = torch.rad2deg(_wrap_to_pi(_quat_to_rpy_wxyz(actual_quat_b) - target_rpy_b).abs())
                quat_error = _quat_angle_error_deg(actual_quat_b, target_quat_b)

                records.append({
                    "point_index": point_idx,
                    "target_pos_b": target_pos_b[point_idx].detach().cpu().tolist(),
                    "compliance_target_pos_b": compliance_target_b.detach().cpu().tolist(),
                    "compliance_offset_b": compliance_offset_b.detach().cpu().tolist(),
                    "ee_force_b": force_b.detach().cpu().tolist(),
                    "actual_pos_b": actual_pos_b.detach().cpu().tolist(),
                    "target_quat_b_wxyz": target_quat_b.detach().cpu().tolist(),
                    "target_axis_angle_b": target_axis_angle_b.detach().cpu().tolist(),
                    "target_rpy_b_deg": torch.rad2deg(target_rpy_b).detach().cpu().tolist(),
                    "actual_rpy_b_deg": torch.rad2deg(_quat_to_rpy_wxyz(actual_quat_b)).detach().cpu().tolist(),
                    "pos_error_m": pos_error.detach().cpu().tolist(),
                    "compliance_pos_error_m": compliance_pos_error.detach().cpu().tolist(),
                    "rpy_abs_error_deg": rpy_error.detach().cpu().tolist(),
                    "quat_angle_error_deg": quat_error.detach().cpu().tolist(),
                })

                mean_l = pos_error[:, 0].mean().item()
                mean_r = pos_error[:, 1].mean().item()
                compliance_mean_l = compliance_pos_error[:, 0].mean().item()
                compliance_mean_r = compliance_pos_error[:, 1].mean().item()
                print(
                    f"EE point {point_idx + 1:02d}/{EE_TRACKING_NUM_POINTS}: "
                    f"left_pos_err={mean_l:.4f} m, right_pos_err={mean_r:.4f} m, "
                    f"compliance_left_err={compliance_mean_l:.4f} m, "
                    f"compliance_right_err={compliance_mean_r:.4f} m"
                )

        pos_errors = torch.tensor([r["pos_error_m"] for r in records])
        compliance_pos_errors = torch.tensor([r["compliance_pos_error_m"] for r in records])
        compliance_offsets = torch.tensor([r["compliance_offset_b"] for r in records])
        ee_forces_b = torch.tensor([r["ee_force_b"] for r in records])
        rpy_errors = torch.tensor([r["rpy_abs_error_deg"] for r in records])
        quat_errors = torch.tensor([r["quat_angle_error_deg"] for r in records])

        report = {
            "checkpoint": args.checkpoint,
            "run_path": args.run_path,
            "task": args.task,
            "seed": EE_TRACKING_SEED,
            "num_envs": args.num_envs,
            "ee_body_names": body_names,
            "feet_body_names": EE_TRACKING_FEET_BODY_NAMES.split(","),
            "default_ee_center_b": default_pos_b.detach().cpu().tolist(),
            "sample_ee_center_b": sample_center_b.detach().cpu().tolist(),
            "configured_ee_center_b": EE_TRACKING_DEFAULT_EE_CENTER_B,
            "min_ee_center_z": EE_TRACKING_MIN_EE_CENTER_Z,
            "ee_radius_m": EE_TRACKING_RADIUS,
            "ee_points": EE_TRACKING_NUM_POINTS,
            "ee_hold_steps": EE_TRACKING_HOLD_STEPS,
            "ee_warmup_steps": EE_TRACKING_WARMUP_STEPS,
            "external_force": args.external_force,
            "external_force_default_cfg": DEFAULT_EXTERNAL_FORCE_CFG if args.external_force == "default" else None,
            "ee_compliance_target": compliance_params,
            "ee_compliance_eval_info": compliance_eval_info,
            "summary": {
                "position_error_m": {
                    "combined": _summary(pos_errors),
                    "left": _summary(pos_errors[:, :, 0]),
                    "right": _summary(pos_errors[:, :, 1]),
                },
                "compliance_position_error_m": {
                    "combined": _summary(compliance_pos_errors),
                    "left": _summary(compliance_pos_errors[:, :, 0]),
                    "right": _summary(compliance_pos_errors[:, :, 1]),
                },
                "compliance_offset_norm_m": {
                    "combined": _summary(compliance_offsets.norm(dim=-1)),
                    "left": _summary(compliance_offsets[:, :, 0].norm(dim=-1)),
                    "right": _summary(compliance_offsets[:, :, 1].norm(dim=-1)),
                },
                "ee_force_b_norm_n": {
                    "combined": _summary(ee_forces_b.norm(dim=-1)),
                    "left": _summary(ee_forces_b[:, :, 0].norm(dim=-1)),
                    "right": _summary(ee_forces_b[:, :, 1].norm(dim=-1)),
                },
                "rpy_abs_error_deg": {
                    "combined": _summary(rpy_errors),
                    "left": _summary(rpy_errors[:, :, 0]),
                    "right": _summary(rpy_errors[:, :, 1]),
                },
                "quat_angle_error_deg": {
                    "combined": _summary(quat_errors),
                    "left": _summary(quat_errors[:, :, 0]),
                    "right": _summary(quat_errors[:, :, 1]),
                },
            },
            "records": records,
        }

        if args.ee_output is None:
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.ee_output = os.path.join("outputs", f"ee_tracking_eval_{time_str}.json")
        os.makedirs(os.path.dirname(args.ee_output) or ".", exist_ok=True)
        with open(args.ee_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        pos = report["summary"]["position_error_m"]["combined"]
        compliance_pos = report["summary"]["compliance_position_error_m"]["combined"]
        compliance_offset = report["summary"]["compliance_offset_norm_m"]["combined"]
        ee_force = report["summary"]["ee_force_b_norm_n"]["combined"]
        quat = report["summary"]["quat_angle_error_deg"]["combined"]
        rpy = report["summary"]["rpy_abs_error_deg"]["combined"]
        print("\n" + "=" * 60)
        print("EE TRACKING EVAL")
        print("=" * 60)
        print(f"  Nominal position error mean/rmse/max: {pos['mean']:.4f} / {pos['rmse']:.4f} / {pos['max']:.4f} m")
        print(
            "  Compliance position error mean/rmse/max: "
            f"{compliance_pos['mean']:.4f} / {compliance_pos['rmse']:.4f} / {compliance_pos['max']:.4f} m"
        )
        print(
            "  Compliance offset norm mean/max: "
            f"{compliance_offset['mean']:.4f} / {compliance_offset['max']:.4f} m"
        )
        print(f"  EE force_b norm mean/max: {ee_force['mean']:.2f} / {ee_force['max']:.2f} N")
        print(f"  RPY abs error mean/rmse/max: {rpy['mean']:.2f} / {rpy['rmse']:.2f} / {rpy['max']:.2f} deg")
        print(f"  Quat angle error mean/rmse/max: {quat['mean']:.2f} / {quat['rmse']:.2f} / {quat['max']:.2f} deg")
        print(f"  Report: {args.ee_output}")
        print("=" * 60 + "\n")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Manipulation evaluation with teleoperation")
    parser.add_argument("-r", "--run_path", type=str, help="WandB run path")
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path (alternative to wandb)")
    parser.add_argument("--task", type=str, default=None, help="Override task config")
    parser.add_argument("-p", "--play", action="store_true", default=False, help="Play mode (visualize)")
    parser.add_argument("-i", "--iterations", type=int, default=None, help="Checkpoint iteration to load")
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("-e", "--export", action="store_true", default=False, help="Export policy")
    parser.add_argument("--objects", type=str, default=None, 
                        help="Objects config file (e.g., cfg/objects/test_scene.yaml)")
    parser.add_argument("--full_collision", action="store_true", default=False,
                        help="Use robot USD with full collision meshes (g1_col_full)")
    parser.add_argument("--obs_source", choices=["udp", "motion"], default="udp",
                        help="Observation source for command/root_and_wrist_6d in play mode")
    parser.add_argument("--ee_tracking_eval", "--ee-tracking-eval", action="store_true", default=False,
                        help="Evaluate EE tracking accuracy with scripted random EE commands")
    parser.add_argument("--external_force", choices=["on", "off", "default"], default="on",
                        help="External force mode: on uses run cfg, off disables it, default loads the shared eval force cfg")
    parser.add_argument("--ee_output", type=str, default=None, help="Path to write EE tracking JSON report")
    args = parser.parse_args()

    # Determine checkpoint source
    if args.run_path:
        # Load from wandb
        api = wandb.Api()
        run = api.run(args.run_path)
        print(f"Loading run: {run.name}")

        root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
        os.makedirs(root, exist_ok=True)

        # Download config and checkpoints
        checkpoints = []
        for file in run.files():
            print(file.name)
            if "checkpoint" in file.name:
                checkpoints.append(file)
            elif file.name == "cfg.yaml":
                file.download(root, replace=True)
            elif file.name == "files/cfg.yaml":
                file.download(root, replace=True)
            elif file.name == "config.yaml":
                file.download(root, replace=True)
        
        # Select checkpoint
        if args.iterations is None:
            def sort_by_time(file):
                number_str = file.name[:-3].split("_")[-1]
                if number_str == "final":
                    return 100000
                else:
                    return int(number_str)
            checkpoints.sort(key=sort_by_time)
            checkpoint = checkpoints[-1]
        else:
            for file in checkpoints:
                if file.name == f"checkpoint_{args.iterations}.pt":
                    checkpoint = file
                    break
        
        print(f"Downloading {checkpoint.name}")
        checkpoint.download(root, replace=True)

        # Load config
        try:
            cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
        except FileNotFoundError:
            cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))
        OmegaConf.set_struct(cfg, False)

        cfg["checkpoint_path"] = os.path.join(root, checkpoint.name)
        if cfg.get("vecnorm", None) is not None:
            cfg["vecnorm"] = "eval"

    elif args.checkpoint:
        # Load from local checkpoint with default config
        with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
            cfg = hydra.compose(config_name="eval", overrides=[])
        OmegaConf.set_struct(cfg, False)
        cfg["checkpoint_path"] = args.checkpoint
        cfg["vecnorm"] = "eval"
    else:
        print("Error: Must specify either --run_path or --checkpoint")
        sys.exit(1)

    # Note: UdpTeleopReceiver is already started by default in MotionTrackingCommand
    # The robot will listen on UDP port 15000 for teleoperation commands

    # Override task config if specified
    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval_manipulation", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags

    _apply_external_force_mode(cfg, args.external_force)

    if args.ee_tracking_eval:
        cfg["app"]["headless"] = not args.play
        cfg["task"]["num_envs"] = args.num_envs
        cfg["task"]["max_episode_length"] = EE_TRACKING_MAX_EPISODE_LENGTH
        cfg["task"]["termination"] = {}
        cfg["export_policy"] = False
        cfg["perf_test"] = False

        command_target = cfg["task"]["command"].get("_target_", "")
        if command_target == "active_adaptation.envs.mdp.commands.teleoperation.TeleopCommand":
            cfg["task"]["command"]["mode"] = "programmatic"
        elif command_target.startswith("active_adaptation.envs.mdp.commands.motion_tracking."):
            cfg["task"]["command"]["disable_motion_finish"] = True
            cfg["task"]["command"]["teleop"] = {"enabled": False, "obs_source": "motion"}

        fixed_low_level_force_limit = _fix_low_level_force_limit_for_ee_eval(cfg)

        if "init_noise" in cfg["task"]["command"]:
            cfg["task"]["command"]["init_noise"] = {
                "root_pos": 0.0,
                "root_ori": 0.0,
                "root_lin_vel": 0.0,
                "root_ang_vel": 0.0,
                "joint_pos": 0.0,
                "joint_vel": 0.0,
            }

        if args.full_collision:
            cfg["task"]["robot"]["name"] = "g1_col_full"
            print("  Robot: Using full collision USD (g1_col_full)")

        if args.objects:
            objects_cfg = OmegaConf.load(args.objects)
            cfg["task"]["objects"] = objects_cfg.get("objects", [])
            print(f"  Objects: Loaded {len(cfg['task']['objects'])} objects from {args.objects}")

        print("\n" + "="*60)
        print("MANIPULATION TASK (EE Tracking Eval)")
        print("="*60)
        print(f"  Run: {args.run_path or args.checkpoint}")
        print(f"  Num envs: {args.num_envs}")
        print(f"  Seed: {EE_TRACKING_SEED}")
        print(f"  EE points: {EE_TRACKING_NUM_POINTS}")
        print(f"  EE radius: {EE_TRACKING_RADIUS} m")
        print(f"  Hold steps: {EE_TRACKING_HOLD_STEPS}")
        print(f"  EE bodies: {EE_TRACKING_BODY_NAMES}")
        print(f"  External force: {args.external_force}")
        if fixed_low_level_force_limit is not None:
            print(f"  Low-level force limit: fixed at {fixed_low_level_force_limit:.2f}")
        print("="*60 + "\n")

        evaluate_ee_tracking(cfg, args)

    # Play mode settings
    elif args.play:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = args.num_envs
        cfg["task"]["max_episode_length"] = 1000000  # Very long episode for teleop (no auto-reset)
        # Disable all termination conditions for teleop mode
        cfg["task"]["termination"] = {}
        # Disable motion finish triggering reset only for live teleop mode.
        cfg["task"]["command"]["disable_motion_finish"] = args.obs_source == "udp"
        cfg["task"]["command"]["teleop"] = {
            "enabled": args.obs_source == "udp",
            "obs_source": args.obs_source,
        }
        _enable_root_passthrough_for_ee_only_hl(cfg)
        cfg["export_policy"] = args.export
        cfg["perf_test"] = False
        
        # Disable all init noise for consistent starting pose in teleop mode
        cfg["task"]["command"]["init_noise"] = {
            "root_pos": 0.0,
            "root_ori": 0.0,
            "root_lin_vel": 0.0,
            "root_ang_vel": 0.0,
            "joint_pos": 0.0,
            "joint_vel": 0.0,
        }
        
        # Use full collision robot USD if requested
        if args.full_collision:
            cfg["task"]["robot"]["name"] = "g1_col_full"
            print("  Robot: Using full collision USD (g1_col_full)")
        
        # Load objects configuration
        if args.objects:
            objects_cfg = OmegaConf.load(args.objects)
            cfg["task"]["objects"] = objects_cfg.get("objects", [])
            print(f"  Objects: Loaded {len(cfg['task']['objects'])} objects from {args.objects}")
        
        print("\n" + "="*60)
        print("MANIPULATION TASK (Teleoperation Mode)")
        print("="*60)
        print(f"  Run: {args.run_path or args.checkpoint}")
        print(f"  Num envs: {args.num_envs}")
        print(f"  Max episode length: 1000000 steps (~5.5 hours at 50Hz)")
        print(f"  External force: {args.external_force}")
        if args.obs_source == "udp":
            print(f"  Obs source: UDP teleop (waiting for input on port 15000)")
        else:
            print(f"  Obs source: motion dataset")
        if args.objects:
            print(f"  Objects: {args.objects}")
        print("="*60 + "\n")
        
        play(cfg)
    else:
        print("Error: Currently only play mode (-p) is supported for manipulation")
        sys.exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()
