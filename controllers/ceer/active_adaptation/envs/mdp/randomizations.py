import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple

import active_adaptation

import isaaclab.utils.string as string_utils


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class Randomization:
    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass


RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]

class motor_params_implicit(Randomization):
    def __init__(
        self,
        env,
        stiffness_range,
        damping_range,
        armature_range,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        # 存下区间字典
        self.stiffness_range = dict(stiffness_range)
        self.damping_range   = dict(damping_range)
        self.armature_range  = dict(armature_range)
        # ------- stiffness -------
        ids, _, val = string_utils.resolve_matching_names_values(
            self.stiffness_range, self.asset.joint_names
        )
        self.stiffness_id  = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.stiffness_def = self.asset.data.joint_stiffness[0, self.stiffness_id]
        low, high          = (
            torch.tensor(val, device=self.device) * self.stiffness_def.unsqueeze(1)
        ).unbind(1)
        self.stiffness_low   = low
        self.stiffness_scale = high - low

        # ------- damping -------
        ids, _, val = string_utils.resolve_matching_names_values(
            self.damping_range, self.asset.joint_names
        )
        self.damping_id  = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.damping_def = self.asset.data.joint_damping[0, self.damping_id]
        low, high        = (
            torch.tensor(val, device=self.device) * self.damping_def.unsqueeze(1)
        ).unbind(1)
        self.damping_low   = low
        self.damping_scale = high - low

        # ------- armature (改为相对值) -------
        ids, _, val = string_utils.resolve_matching_names_values(
            self.armature_range, self.asset.joint_names
        )
        self.armature_id  = torch.tensor(ids, device=self.device, dtype=torch.long)
        self.armature_def = self.asset.data.joint_armature[0, self.armature_id]
        low, high         = (
            torch.tensor(val, device=self.device) * self.armature_def.unsqueeze(1)
        ).unbind(1)
        self.armature_low   = low
        self.armature_scale = high - low

    def _rand_u(self, n_env: int, k: int):
        return torch.rand(n_env, k, device=self.device)

    # ----------------------------------------------------------
    def reset(self, env_ids):
        n_env = len(env_ids)

        # stiffness
        stiff = self._rand_u(n_env, len(self.stiffness_id))
        stiff = stiff * self.stiffness_scale + self.stiffness_low
        self.asset.write_joint_stiffness_to_sim(stiff, self.stiffness_id, env_ids)

        # damping
        damp = self._rand_u(n_env, len(self.damping_id))
        damp = damp * self.damping_scale + self.damping_low
        self.asset.write_joint_damping_to_sim(damp, self.damping_id, env_ids)

        # armature
        arma = self._rand_u(n_env, len(self.armature_id))
        arma = arma * self.armature_scale + self.armature_low
        self.asset.write_joint_armature_to_sim(arma, self.armature_id, env_ids)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range=(0.6, 1.0),
        dynamic_friction_frac_range=(0.6, 1.0),
        restitution_range=(0.0, 0.2),
        homogeneous: bool = False,
        num_buckets: int = 16,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_frac_range = dynamic_friction_frac_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous
        self.num_buckets = num_buckets

        self.default_materials = (
            self.asset.root_physx_view.get_material_properties()
        )
        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0] + num_shapes_per_body)
        self.shape_ids = torch.cat(
            [torch.arange(cumsum[i], cumsum[i + 1]) for i in self.body_ids]
        )

    def startup(self):
        logging.info(f"Randomize body materials of {self.body_names} upon startup.")

        materials = self.default_materials.clone()
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))

        sf  = sample_uniform(shape, *self.static_friction_range)                      # static friction
        dff = sample_uniform(shape, *self.dynamic_friction_frac_range)                # dynamic-fraction
        res = sample_uniform(shape, *self.restitution_range)                          # restitution

        def _bucketize(x, lo, hi, n):
            step = (hi - lo) / (n - 1)
            idx  = ((x - lo) / step).round().clamp(0, n - 1)
            return lo + idx * step

        N = self.num_buckets                       # 16 ⇒ 4096 种组合上限
        sf  = _bucketize(sf,  *self.static_friction_range,           N)
        dff = _bucketize(dff, *self.dynamic_friction_frac_range,     N)
        res = _bucketize(res, *self.restitution_range,               N)
        # -------------------------------------------------------------

        materials[:, self.shape_ids, 0] = sf
        materials[:, self.shape_ids, 1] = dff * sf
        materials[:, self.shape_ids, 2] = res

        indices = torch.arange(self.asset.num_instances)

        self.asset.root_physx_view.set_material_properties(
            materials.flatten(), indices
        )
        self.asset.data.body_materials = materials.to(self.device)

class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        self.mass_ranges = torch.tensor(values)
        print(self.body_names)

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        masses = self.asset.data.default_mass.clone()
        inertias = self.asset.data.default_inertia.clone()
        print(f"Default masses: {masses[0]}")
        scale = uniform(
            self.mass_ranges[:, 0].expand_as(masses[:, self.body_ids]),
            self.mass_ranges[:, 1].expand_as(masses[:, self.body_ids])
        )
        masses[:, self.body_ids] *= scale
        inertias[:, self.body_ids] *= scale.unsqueeze(-1)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)
        self.asset.root_physx_view.set_inertias(inertias, indices)
        assert torch.allclose(self.asset.root_physx_view.get_masses(), masses)

class perturb_body_com(Randomization):
    def __init__(self, env, body_names = ".*", com_range=(-0.05, 0.05)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.com_range = com_range
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.ALL_INDICES = torch.arange(self.asset.num_instances)
    
    def startup(self):
        coms = self.asset.root_physx_view.get_coms()
        rand_offset = sample_uniform((self.asset.num_instances, len(self.body_ids), 3), *self.com_range)
        coms[:, self.body_ids, :3] += rand_offset
        self.asset.root_physx_view.set_coms(coms, indices=self.ALL_INDICES)

class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[:, 0], self.offset_range[:, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset

def clamp_norm(x: torch.Tensor, min: float = 0.0, max: float = torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight