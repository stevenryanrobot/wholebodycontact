import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
import hydra
import active_adaptation.utils.symmetry as symmetry_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
    
    def symmetry_transforms(self):
        raise NotImplementedError(
            "ActionManager subclasses must implement symmetry_transforms method."
            "This method should return a SymmetryTransform object that applies to the action space."
        )


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int | None = None,
        alpha: Tuple[float, float] = (0.9, 0.9),
        alpha_wide: Tuple[float, float] = (0.8, 1.0),
        boot_protect: bool = False,
        alpha_jit_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(env)

        # ------------------------------------------------------------------ cfg
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        # print(self.joint_ids, self.joint_names, self.action_scaling)
        # breakpoint()
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.max_delay = max_delay or 0  # physics steps

        self.alpha_range = alpha
        self.alpha_wide_range = alpha_wide

        # Boot‑protection ----------------------------------------------------
        self.boot_protect_enabled = boot_protect
        if self.boot_protect_enabled:
            self.boot_delay = torch.zeros(self.num_envs, 1, dtype=int, device=self.device)

        # α‑jitter -----------------------------------------------------------
        self.alpha_jit_scale = alpha_jit_scale
        if self.alpha_jit_scale is not None:
            self.alpha_jit = torch.zeros(self.num_envs, 1, device=self.device)

        # Persistent tensors -------------------------------------------------
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)

        with torch.device(self.device):
            hist = max((self.max_delay - 1) // self.env.decimation + 1, 3)
            self.action_buf = torch.zeros(self.num_envs, hist, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    # --------------------------------------------------------------------- util
    def resolve(self, spec):
        """Convenience helper for user APIs."""
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)
    
    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform
    # ------------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        # Delay selection ---------------------------------------------------
        if self.boot_protect_enabled:
            delay = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
            self.boot_delay[env_ids] = delay
            self.delay[env_ids] = delay
        else:
            self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)

        # α per environment --------------------------------------------------
        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    # ---------------------------------------------------------------- forward
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)

            ### debug symmetry
            # raw_action = self.symmetry_transforms().to(raw_action.device).forward(raw_action)

            # α with optional jitter -------------------------------------------
            if self.alpha_jit_scale is not None:
                self.alpha_jit.uniform_(-self.alpha_jit_scale, self.alpha_jit_scale)
                self.alpha.add_(self.alpha_jit).clamp_(*self.alpha_wide_range)

            self.action_buf = torch.roll(self.action_buf, shifts=1, dims=1)
            self.action_buf[:, 0, :] = raw_action

        # Communication delay ----------------------------------------------
        idx = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
        delayed_action = self.action_buf.take_along_dim(idx.unsqueeze(1), dim=1).squeeze(1)
        self.applied_action.lerp_(delayed_action, self.alpha)

        # Joint targets -----------------------------------------------------
        pos_tgt = self.default_joint_pos + self.offset
        pos_tgt[:, self.joint_ids] += self.applied_action * self.action_scaling

        # Optional boot‑protection -----------------------------------------
        if self.boot_protect_enabled:
            pos_tgt = torch.where(
                self.boot_delay > 0,
                self.env.command_manager.joint_pos_boot_protect,
                pos_tgt,
            )
            self.boot_delay.sub_(1).clamp_min_(0)

        # Write to simulator -----------------------------------------------
        self.asset.set_joint_position_target(pos_tgt)
        self.asset.write_data_to_sim()


class HierarchicalRootCommand(ActionManager):
    """High-level root-command action manager with a frozen low-level policy."""

    def __init__(
        self,
        env,
        low_action: Dict,
        low_policy: Dict,
        command_dim: int = 5,
        nominal_root_height: float = 0.79,
        command_scale: Tuple[float, float, float, float, float] = (0.25, 0.8, 0.8, 1.0, 1.0),
        low_policy_command_slice: Tuple[int, int] | None = (1, 7),
        low_policy_obs_key: str | None = "policy",
        **kwargs,
    ):
        super().__init__(env)
        if command_dim != 5:
            raise ValueError("HierarchicalRootCommand currently expects command_dim=5.")
        self.action_dim = command_dim
        self.nominal_root_height = nominal_root_height
        self.command_scale = torch.tensor(command_scale, device=self.device).reshape(1, command_dim)
        self.low_policy_command_slice = tuple(low_policy_command_slice) if low_policy_command_slice is not None else None
        self.low_policy_obs_key = low_policy_obs_key

        self.low_action_manager: JointPosition = hydra.utils.instantiate(low_action, env=env)
        from active_adaptation.learning.hierarchical.frozen_low_level import FrozenLowLevelPolicy
        self.low_policy = FrozenLowLevelPolicy(
            env=env,
            action_dim=self.low_action_manager.action_dim,
            **low_policy,
        )

        self.high_action_buf = torch.zeros(self.num_envs, 3, self.action_dim, device=self.device)
        self.root_command = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.low_action = torch.zeros(self.num_envs, self.low_action_manager.action_dim, device=self.device)
        self._reset_root_command(torch.arange(self.num_envs, device=self.device))

    @property
    def joint_ids(self):
        return self.low_action_manager.joint_ids

    @property
    def joint_names(self):
        return self.low_action_manager.joint_names

    @property
    def offset(self):
        return self.low_action_manager.offset

    @property
    def action_buf(self):
        return self.low_action_manager.action_buf

    @property
    def applied_action(self):
        return self.low_action_manager.applied_action

    def symmetry_transforms(self):
        return self.low_action_manager.symmetry_transforms()

    def _reset_root_command(self, env_ids: torch.Tensor):
        self.root_command[env_ids] = 0.0
        self.root_command[env_ids, 0] = self.nominal_root_height
        self.root_command[env_ids, 3] = 1.0

    def reset(self, env_ids: torch.Tensor):
        self.low_action_manager.reset(env_ids)
        self.high_action_buf[env_ids] = 0.0
        self.low_action[env_ids] = 0.0
        self._reset_root_command(env_ids)
        if hasattr(self.env.command_manager, "set_root_command"):
            self.env.command_manager.set_root_command(self.root_command)

    def debug_draw(self):
        self.low_action_manager.debug_draw()

    def _decode_root_command(self, raw_action: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(raw_action) * self.command_scale
        command = torch.zeros_like(action)
        command[:, 0] = self.nominal_root_height + action[:, 0]
        command[:, 1:3] = action[:, 1:3]

        heading_xy = torch.zeros(self.num_envs, 2, device=self.device)
        heading_xy[:, 0] = 1.0
        heading_xy = heading_xy + action[:, 3:5]
        heading_xy = heading_xy / heading_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        command[:, 3:5] = heading_xy
        return command

    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)
            self.high_action_buf = torch.roll(self.high_action_buf, shifts=1, dims=1)
            self.high_action_buf[:, 0, :] = raw_action

            self.root_command[:] = self._decode_root_command(raw_action)
            # self.root_command[:] = torch.tensor([self.nominal_root_height, 0.0, 0.0, 1.0, 0.0], device=self.device)
            if not hasattr(self.env.command_manager, "set_root_command"):
                raise RuntimeError(
                    "HierarchicalRootCommand requires a command manager with set_root_command()."
                )
            self.env.command_manager.set_root_command(self.root_command)
            low_td = tensordict.clone()
            if self.low_policy_obs_key is not None and self.low_policy_obs_key in low_td.keys():
                if self.low_policy_obs_key in self.env.observation_funcs:
                    low_td["policy"] = self.env.observation_funcs[self.low_policy_obs_key]._compute()
                else:
                    low_td["policy"] = low_td[self.low_policy_obs_key].clone()
            if self.low_policy_command_slice is not None:
                command = self.env.command_manager.command()
                start, stop = self.low_policy_command_slice
                low_td["policy"][:, start:stop] = command
                # low_td["policy"][:, stop:stop + 12] = torch.tensor([0.15, 0.1, 0.0, 0.15, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
            self.low_action[:] = self.low_policy.act(low_td)

        low_td = tensordict.clone()
        low_td["action"] = self.low_action
        self.low_action_manager(low_td, substep)
