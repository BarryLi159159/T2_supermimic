"""Locomotion policy and critic.

Policy: a = MLP([o_hist (58 x H), g]) with the SuperMimic decoder's fixed per-term input scales. No goal encoder:
the goal g is the 7-d command, plus the gait clock (9-d) when the task uses it. Critic: V(canonical state 71, g, xi)
(privileged, imagination only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from supermimic.envs.sim_context import XI_DIM
from supermimic.world_model.dynamics import MLP
from supermimic.world_model.loco_heads import CMD_DIM, LOCO_OBS_TERMS, POLICY_ACT_DIM


@dataclass
class LocoPolicyCfg:
    hidden: tuple[int, ...] = (1024, 512, 256)
    history: int = 10
    # Fixed per-term observation scales (divide): base_ang_vel, projected_gravity, command, joint_pos, joint_vel, actions.
    obs_term_scale: tuple[float, ...] = (2.0, 1.0, 1.0, 1.0, 5.0, 1.0)
    last_gain: float = 0.01
    goal_dim: int = CMD_DIM  # set by the trainer from the task (9 with the gait clock)


@dataclass
class LocoCriticCfg:
    hidden: tuple[int, ...] = (256, 128)
    goal_dim: int = CMD_DIM  # the policy's goal, + 4 foot timers when the reward uses them


def loco_obs_scale_vector(cfg: LocoPolicyCfg) -> torch.Tensor:
    parts = [torch.full((cfg.history * d,), 1.0 / s) for (_, d), s in zip(LOCO_OBS_TERMS, cfg.obs_term_scale)]
    return torch.cat(parts)  # term-major layout matches the flattened history


class LocoPolicy(nn.Module):
    def __init__(self, cfg: LocoPolicyCfg):
        super().__init__()
        self.cfg = cfg
        obs_dim = cfg.history * sum(d for _, d in LOCO_OBS_TERMS)
        self.mlp = MLP(obs_dim + cfg.goal_dim, cfg.hidden, POLICY_ACT_DIM, last_gain=cfg.last_gain)
        self.register_buffer("scale", loco_obs_scale_vector(cfg))

    def forward(self, obs_hist: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([obs_hist * self.scale, cmd], dim=-1))


class LocoCritic(nn.Module):
    def __init__(self, cfg: LocoCriticCfg, feat_dim: int = 71, xi_dim: int = XI_DIM):
        super().__init__()
        self.mlp = MLP(feat_dim + cfg.goal_dim + xi_dim, cfg.hidden, 1)

    def forward(self, feat: torch.Tensor, cmd: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, cmd, xi_norm], dim=-1)).squeeze(-1)
