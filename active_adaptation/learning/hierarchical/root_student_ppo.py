from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    REWARD_KEY,
    TERM_KEY,
    Actor,
    CatTensors,
    GAE,
    make_batch,
    make_mlp,
)


@dataclass
class RootStudentPPOConfig:
    _target_: str = "active_adaptation.learning.hierarchical.root_student_ppo.RootStudentPPOPolicy"
    name: str = "root_student_ppo"

    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8
    estimator_epochs: int = 2

    lr: float = 3e-4
    desired_kl: float = 0.01
    clip_param: float = 0.2
    entropy_coef_start: float = 0.005
    entropy_coef_end: float = 0.001
    init_noise_scale: float = 0.8
    load_noise_scale: float | None = None
    layer_norm: str | None = "before"
    latent_dim: int = 128
    reg_lambda: float = 0.2
    vecnorm: str | None = None
    phase: str = "train"  # train | adapt | finetune

    in_keys: List[str] = field(default_factory=lambda: ["hl_policy"])
    priv_in_keys: List[str] = field(default_factory=lambda: ["hl_priv"])
    critic_in_keys: List[str] = field(default_factory=lambda: ["hl_policy", "hl_priv"])


cs = ConfigStore.instance()
cs.store("root_student_ppo", node=RootStudentPPOConfig(), group="algo")


class RootStudentPPOPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: RootStudentPPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env=None,
    ) -> None:
        super().__init__()
        if cfg.phase not in {"train", "adapt", "finetune"}:
            raise ValueError(f"Unsupported root_student_ppo phase: {cfg.phase}")

        self.cfg = cfg
        self.device = torch.device(device)
        self.action_dim = action_spec.shape[-1]
        self.entropy_coef = cfg.entropy_coef_start
        self.clip_param = cfg.clip_param
        self.gae = GAE(0.99, 0.95)
        self.num_minibatches = cfg.num_minibatches
        self.progress = 0.0
        self.current_lr = cfg.lr
        self.num_updates = 0
        self.reg_lambda = 0.0

        actor_in_keys = list(cfg.in_keys)
        priv_in_keys = list(cfg.priv_in_keys)
        critic_in_keys = list(cfg.critic_in_keys)

        self.encoder_priv = Seq(
            CatTensors(priv_in_keys, "_priv_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(make_mlp([256], norm=cfg.layer_norm), nn.LazyLinear(cfg.latent_dim)),
                ["_priv_inp"],
                ["priv_feature"],
            ),
        ).to(self.device)

        self.adapt_module = Seq(
            CatTensors(actor_in_keys, "_adapt_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(make_mlp([512, 256], norm=cfg.layer_norm), nn.LazyLinear(cfg.latent_dim)),
                ["_adapt_inp"],
                ["priv_pred"],
            ),
        ).to(self.device)

        self.actor_teacher = self._build_actor(actor_in_keys + ["priv_feature"])
        self.actor_student = self._build_actor(actor_in_keys + ["priv_pred"])

        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(make_mlp([512, 256], norm=cfg.layer_norm), nn.LazyLinear(1)),
                ["_critic_inp"],
                ["state_value"],
            ),
        ).to(self.device)

        fake_td = observation_spec.zero().to(self.device)
        fake_td["is_init"] = torch.ones(fake_td.shape[0], 1, dtype=torch.bool, device=self.device)
        self.encoder_priv(fake_td)
        self.adapt_module(fake_td)
        self.actor_teacher(fake_td)
        self.actor_student(fake_td)
        self.critic(fake_td)

        self.world_size = 1
        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            ddp_kwargs = dict(
                device_ids=[aa.get_local_rank()],
                output_device=aa.get_local_rank(),
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
            self.encoder_priv = DDP(self.encoder_priv, **ddp_kwargs)
            self.adapt_module = DDP(self.adapt_module, **ddp_kwargs)
            self.actor_teacher = DDP(self.actor_teacher, **ddp_kwargs)
            self.actor_student = DDP(self.actor_student, **ddp_kwargs)
            self.critic = DDP(self.critic, **ddp_kwargs)

        self.opt_teacher = torch.optim.Adam(
            list(self.encoder_priv.parameters()) + list(self.actor_teacher.parameters()),
            lr=cfg.lr,
        )
        self.opt_student = torch.optim.Adam(
            list(self.adapt_module.parameters()) + list(self.actor_student.parameters()),
            lr=cfg.lr,
        )
        self.opt_estimator = torch.optim.Adam(self.adapt_module.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def _build_actor(self, in_keys: list[str]):
        return ProbabilisticActor(
            module=Seq(
                CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                Mod(make_mlp([512, 256], norm=self.cfg.layer_norm), ["_actor_inp"], ["_actor_feature"]),
                Mod(
                    Actor(
                        self.action_dim,
                        init_noise_scale=self.cfg.init_noise_scale,
                        load_noise_scale=self.cfg.load_noise_scale,
                    ),
                    ["_actor_feature"],
                    ["loc", "scale"],
                ),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "train"):
        if mode in {"eval", "deploy"}:
            return Seq(self.adapt_module, self.actor_student)
        if self.cfg.phase == "train":
            return Seq(self.encoder_priv, self.actor_teacher)
        return Seq(self.adapt_module, self.actor_student)

    def broadcast_parameters(self, extra_modules=[]):
        return None

    def step_schedule(self, progress: float, iter: int):
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        self.entropy_coef = start * (end / start) ** progress
        self.progress = progress
        self.reg_lambda = progress * self.cfg.reg_lambda

    def _do_lr_schedule(self, kl: float):
        if self.progress < 0.1:
            return
        new_lr = self.current_lr
        if kl > self.cfg.desired_kl * 2.0:
            new_lr = max(1e-5, new_lr / 1.1)
        elif 0.0 < kl < self.cfg.desired_kl / 2.0:
            new_lr = min(5e-3, new_lr * 1.1)
        self.current_lr = new_lr
        for opt in (self.opt_teacher, self.opt_student, self.opt_estimator, self.opt_critic):
            for group in opt.param_groups:
                group["lr"] = self.current_lr

    def train_op(self, td: TensorDict, vecnorm):
        if self.cfg.phase == "train":
            info = self._ppo_update(td, actor=self.actor_teacher, encoder=self.encoder_priv, opt_actor=self.opt_teacher)
            info.update(self._train_estimator(td))
        elif self.cfg.phase == "finetune":
            info = self._ppo_update(td, actor=self.actor_student, encoder=self.adapt_module, opt_actor=self.opt_student)
        else:
            info = self._train_estimator(td)
        self.num_updates += 1
        return info

    @torch.no_grad()
    def _compute_advantage(self, td: TensorDict):
        if "state_value" not in td.keys(True, True):
            self.critic(td.view(-1))
        if ("next", "state_value") not in td.keys(True, True):
            self.critic(td["next"].view(-1))

        rewards = td[REWARD_KEY].sum(dim=-1, keepdim=True)
        adv, ret = self.gae(
            rewards,
            td[TERM_KEY],
            td[DONE_KEY],
            td["state_value"],
            td["next", "state_value"],
        )
        td["adv"] = adv
        td["ret"] = ret

        valid = ~td["is_init"]
        mean = td["adv"][valid].mean()
        std = td["adv"][valid].std().clamp_min(1e-5)
        td["adv"][valid] = (td["adv"][valid] - mean) / std

    def _ppo_update(self, td: TensorDict, actor, encoder, opt_actor):
        infos = []
        self._compute_advantage(td)

        for _ in range(self.cfg.ppo_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update(mb, actor, encoder, opt_actor), []))

        info = {k: v.mean().item() for k, v in torch.stack(infos).items()}
        self._do_lr_schedule(info["actor/kl"])
        info["lr"] = self.current_lr
        return info

    def _update(self, mb: TensorDict, actor, encoder, opt_actor):
        loc_old = mb["loc"].clone()
        scale_old = mb["scale"].clone()
        action_old = mb["action"].clone()
        logp_old = mb["sample_log_prob"].clone()
        valid = ~mb["is_init"]

        mb = mb.exclude("next", "sample_log_prob", "action")
        encoder(mb)
        actor(mb)
        values = self.critic(mb)["state_value"]

        dist = IndependentNormal(mb["loc"], mb["scale"])
        logp = dist.log_prob(action_old)
        entropy = dist.entropy().mean()
        ratio = torch.exp(logp - logp_old).unsqueeze(-1)
        surr1 = mb["adv"] * ratio
        surr2 = mb["adv"] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2) * valid)
        entropy_loss = -self.entropy_coef * entropy
        value_loss = F.mse_loss(values, mb["ret"], reduction="none")
        value_loss = (value_loss * valid).mean()
        loss = policy_loss + entropy_loss + value_loss

        opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        encoder_grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        opt_actor.step()
        self.opt_critic.step()

        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            kl = torch.sum(
                torch.log(mb["scale"]) - torch.log(scale_old)
                + (scale_old.square() + (loc_old - mb["loc"]).square()) / (2.0 * mb["scale"].square())
                - 0.5,
                dim=-1,
            ).mean()

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/actor_grad_norm": actor_grad_norm.detach(),
            "actor/encoder_grad_norm": encoder_grad_norm.detach(),
            "actor/clamp_ratio": clipfrac.detach(),
            "actor/kl": kl.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/critic_grad_norm": critic_grad_norm.detach(),
        }

    def _train_estimator(self, td: TensorDict):
        infos = []
        for _ in range(self.cfg.estimator_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update_estimator(mb), []))
        return {k: v.mean().item() for k, v in torch.stack(infos).items()}

    def _update_estimator(self, mb: TensorDict):
        mb = mb.exclude("next")
        valid = ~mb["is_init"]

        with torch.no_grad():
            self.encoder_priv(mb)
        self.adapt_module(mb)

        estimator_loss = F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none")
        estimator_loss = torch.mean(estimator_loss * valid)

        self.opt_estimator.zero_grad()
        estimator_loss.backward()
        estimator_grad_norm = nn.utils.clip_grad_norm_(self.adapt_module.parameters(), 1.0)
        self.opt_estimator.step()

        return {
            "adapt/estimator_loss": estimator_loss.detach(),
            "adapt/estimator_grad_norm": estimator_grad_norm.detach(),
        }

    def state_dict(self):
        actor_teacher = self.actor_teacher.module if isinstance(self.actor_teacher, DDP) else self.actor_teacher
        actor_student = self.actor_student.module if isinstance(self.actor_student, DDP) else self.actor_student
        encoder_priv = self.encoder_priv.module if isinstance(self.encoder_priv, DDP) else self.encoder_priv
        adapt_module = self.adapt_module.module if isinstance(self.adapt_module, DDP) else self.adapt_module
        critic = self.critic.module if isinstance(self.critic, DDP) else self.critic
        return OrderedDict(
            actor_teacher=actor_teacher.state_dict(),
            actor_student=actor_student.state_dict(),
            encoder_priv=encoder_priv.state_dict(),
            adapt_module=adapt_module.state_dict(),
            critic=critic.state_dict(),
            _meta={
                "current_lr": self.current_lr,
                "entropy_coef": self.entropy_coef,
                "reg_lambda": self.reg_lambda,
                "progress": self.progress,
                "num_updates": self.num_updates,
            },
        )

    def load_state_dict(self, state_dict, strict=True):
        actor_teacher = self.actor_teacher.module if isinstance(self.actor_teacher, DDP) else self.actor_teacher
        actor_student = self.actor_student.module if isinstance(self.actor_student, DDP) else self.actor_student
        encoder_priv = self.encoder_priv.module if isinstance(self.encoder_priv, DDP) else self.encoder_priv
        adapt_module = self.adapt_module.module if isinstance(self.adapt_module, DDP) else self.adapt_module
        critic = self.critic.module if isinstance(self.critic, DDP) else self.critic

        actor_teacher.load_state_dict(state_dict.get("actor_teacher", {}), strict=strict)
        actor_student.load_state_dict(state_dict.get("actor_student", {}), strict=strict)
        encoder_priv.load_state_dict(state_dict.get("encoder_priv", {}), strict=strict)
        adapt_module.load_state_dict(state_dict.get("adapt_module", {}), strict=strict)
        critic.load_state_dict(state_dict.get("critic", {}), strict=strict)

        meta = state_dict.get("_meta", {})
        self.current_lr = meta.get("current_lr", self.current_lr)
        self.entropy_coef = meta.get("entropy_coef", self.entropy_coef)
        self.reg_lambda = meta.get("reg_lambda", self.reg_lambda)
        self.progress = meta.get("progress", self.progress)
        self.num_updates = meta.get("num_updates", self.num_updates)
