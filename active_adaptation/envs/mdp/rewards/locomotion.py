from math import inf
import torch
import abc
from typing import TYPE_CHECKING, Callable, List, Tuple

import isaaclab.utils.string as string_utils
from isaaclab.utils.string import resolve_matching_names
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat, normalize, clamp_norm
from ..commands import *

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class Reward:
    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
    ):
        self.env: _Env = env
        self.weight = weight
        self.enabled = enabled

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass

    def post_step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum().item()
        return self.weight * rew, count 

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass


def reward_func(func):
    class RewFunc(Reward):
        def compute(self):
            return func(self.env)

    return RewFunc


def reward_wrapper(func: Callable[[], torch.Tensor]):
    class RewardWrapper(Reward):
        def compute(self):
            return func()
    return RewardWrapper


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)


class joint_torques_l2(Reward):
    def __init__(
        self, env, weight: float, enabled: bool = True, joint_names: str = ".*"
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.applied_torque[:, self.joint_ids]
            .square()
            .sum(1, keepdim=True)
        )


class impact_force_l2(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        )
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        contact_forces = self.contact_sensor.data.net_forces_w_history.norm(
            dim=-1
        ).mean(1)
        force = contact_forces[:, self.body_ids] / self.default_mass_total
        return -(force.square() * first_contact).sum(1, True).clamp_max(20.0)

class feet_slip(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        feet_vel = self.asset.data.body_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        return -slip

class feet_upright(Reward):
    def __init__(
        self, env, body_names: str, xy_sigma: float, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.body_ids_asset, _ = self.asset.find_bodies(body_names)

        down = torch.tensor([0.0, 0.0, -1.0], device=self.env.device)
        self.down = down.expand(self.num_envs, len(self.body_ids_asset), -1)
        self.xy_sigma = xy_sigma
        
    def compute(self):
        feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids_asset]
        feet_projected_down = quat_apply(feet_quat_w, self.down)
        feet_projected_down_xy = feet_projected_down[:, :, :2].norm(dim=-1)
        rew = (torch.exp(-feet_projected_down_xy / self.xy_sigma) - 1.0)
        return rew.float().mean(dim=1, keepdim=True)

class feet_air_time_ref(Reward):
    def __init__(
        self,
        env: "LocomotionEnv",
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

        self.reward_time = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)
        self.last_contact = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device, dtype=bool)
        self.h_low, self.h_high = 0.035, 0.12
        self.c_low, self.c_high = 0.5, 2.0
        self.exp_log_c_ratio = torch.log(torch.tensor(self.c_high / self.c_low, device=self.device))

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def compute(self):
        current_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        feet_height = self.asset.data.body_pos_w[:, self.articulation_body_ids, 2]

        t = (feet_height - self.h_low) / (self.h_high - self.h_low)
        t = torch.clamp(t, 0.0, 1.0)
        feet_height_coef = self.c_low * torch.exp(self.exp_log_c_ratio * t)

        if hasattr(self.env.command_manager, "skip_ref") and self.env.command_manager.skip_ref:
            self.reward_time = self.reward_time + self.env.step_dt * feet_height_coef
        else:
            contact_diff = self.env.command_manager.feet_standing ^ current_contact
            self.reward_time = self.reward_time + torch.where(contact_diff, -self.env.step_dt, self.env.step_dt * feet_height_coef)
        
        self.reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True
        )
        
        self.reward_time = self.reward_time * (~current_contact)
        return self.reward

class feet_contact_count(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().sum(1, True)

class joint_acc_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        # print(self.asset.data.joint_acc[:, self.joint_ids].max())
        r = - self.asset.data.joint_acc[:, self.joint_ids].clamp_max(100.0).square().sum(1, True)
        return r

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_factor = soft_factor
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(1, keepdim=True) / (1-self.soft_factor)


class root_position_hold(Reward):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.25,
        mode: str = "xy",
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        if mode not in ("xy", "xyz"):
            raise ValueError(f"Unsupported root_position_hold mode: {mode}")
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.mode = mode
        self.root_pos_ref = torch.zeros(self.num_envs, 3, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.root_pos_ref[env_ids] = self.asset.data.root_pos_w[env_ids]

    def compute(self) -> torch.Tensor:
        diff = self.asset.data.root_pos_w - self.root_pos_ref
        if self.mode == "xy":
            error = diff[:, :2].norm(dim=-1, keepdim=True)
        else:
            error = diff.norm(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class root_force_spring_tracking(Reward):
    def __init__(
        self,
        env,
        weight: float,
        stiffness: float = 400.0,
        sigma: float = 0.25,
        max_offset: float = 0.5,
        force_deadband: float = 5.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness = stiffness
        self.sigma = sigma
        self.max_offset = max_offset
        self.force_deadband = force_deadband
        self.root_pos_ref = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.root_pos_ref[env_ids] = self.asset.data.root_pos_w[env_ids, :2]

    def compute(self) -> torch.Tensor:
        force_buffer = getattr(self.env.command_manager, "force_apply_buffer", None)
        if force_buffer is None:
            target_offset_xy = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            force_xy = force_buffer.sum(dim=1)[:, :2]
            force_norm = force_xy.norm(dim=-1, keepdim=True)
            force_xy = torch.where(force_norm > self.force_deadband, force_xy, torch.zeros_like(force_xy))
            target_offset_xy = clamp_norm(force_xy / self.stiffness, max=self.max_offset)

        target_xy = self.root_pos_ref + target_offset_xy
        error = (self.asset.data.root_pos_w[:, :2] - target_xy).norm(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class root_height_hold(Reward):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.08,
        target_height: float | None = None,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.target_height = target_height
        self.root_height_ref = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        if self.target_height is None:
            self.root_height_ref[env_ids] = self.asset.data.root_pos_w[env_ids, 2:3]
        else:
            self.root_height_ref[env_ids] = self.target_height

    def compute(self) -> torch.Tensor:
        error = (self.asset.data.root_pos_w[:, 2:3] - self.root_height_ref).abs()
        return torch.exp(-error / self.sigma)


@reward_func
def high_action_l2(self):
    action_buf = getattr(self.action_manager, "high_action_buf", None)
    if action_buf is None:
        return torch.zeros(self.num_envs, 1, device=self.device)
    return -action_buf[:, 0].square().sum(dim=-1, keepdim=True)


@reward_func
def high_action_rate_l2(self):
    action_buf = getattr(self.action_manager, "high_action_buf", None)
    if action_buf is None or action_buf.shape[1] < 2:
        return torch.zeros(self.num_envs, 1, device=self.device)
    action_diff = action_buf[:, 0] - action_buf[:, 1]
    return -action_diff.square().sum(dim=-1, keepdim=True)


@reward_func
def action_rate_l2(self):
    action_diff = self.action_manager.action_buf[:, :, 0] - self.action_manager.action_buf[:, :, 1]
    return - action_diff.square().sum(dim=-1, keepdim=True)


@reward_func
def action_rate2_l2(self):
    action_diff = (
        self.action_manager.action_buf[:, :, 0] - 2 * self.action_manager.action_buf[:, :, 1] + self.action_manager.action_buf[:, :, 2]
    )
    return - action_diff.square().sum(dim=-1, keepdim=True)
