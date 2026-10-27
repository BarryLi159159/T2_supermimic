"""Lower-body joint-position action: the policy commands 15 joints, the environment holds the other 14.

The action term spans all 29 actuated joints (so every actuator gets a target each step) but exposes an action
dimension of 15, Booster_T2's lower-body policy joints in deployment order (``RobotSpec.policy_joints``). The arm
joints follow ``arm_action`` (zero = the default pose; an upper-body controller or a clip can write it). Execution
noise (plan D9) is added to the policy joints only; ``executed_raw`` / ``clean_raw`` give the full 29-d normalised
targets the world model is fit on, ``raw_action`` the policy's clean 15-d action (last-action observation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class LowerBodyAction(JointPositionAction):
    cfg: LowerBodyActionCfg

    def __init__(self, cfg: LowerBodyActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)  # 29 targets in model joint order; scale / offset are per joint
        assert self._target_names == list(cfg.joint_order), "the action term must span the model's joints in order"
        self.policy_ids = torch.tensor([cfg.joint_order.index(j) for j in cfg.policy_joints], dtype=torch.long, device=self.device)
        self.n_policy = len(cfg.policy_joints)
        self._action_dim = self.n_policy
        self._raw_actions = torch.zeros(self.num_envs, self.n_policy, device=self.device)
        self._clean_raw = torch.zeros(self.num_envs, self._num_targets, device=self.device)
        self._prev_clean_raw = torch.zeros_like(self._clean_raw)
        self._executed_raw = torch.zeros_like(self._clean_raw)
        self.arm_action = torch.zeros(self.num_envs, self._num_targets, device=self.device)  # non-policy joints' normalised targets
        self.noise_std = float(cfg.noise_std)

    @property
    def clean_raw(self) -> torch.Tensor:
        """(N, 29) normalised targets without execution noise."""
        return self._clean_raw

    @property
    def prev_clean_raw(self) -> torch.Tensor:
        """(N, 29) clean targets of the previous step (action-rate penalty)."""
        return self._prev_clean_raw

    @property
    def executed_raw(self) -> torch.Tensor:
        """(N, 29) normalised targets as executed (policy joints noisy)."""
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._prev_clean_raw = self._clean_raw
        full = self.arm_action.clone()
        full[:, self.policy_ids] = self._raw_actions
        self._clean_raw = full
        if self.noise_std > 0:
            noisy = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
            executed = full.clone()
            executed[:, self.policy_ids] = noisy
        else:
            executed = full.clone()
        self._executed_raw = executed
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._clean_raw[env_ids] = 0.0
        self._prev_clean_raw[env_ids] = 0.0
        self._executed_raw[env_ids] = 0.0


@dataclass(kw_only=True)
class LowerBodyActionCfg(JointPositionActionCfg):
    joint_order: tuple[str, ...]  # the model's joint order (= actuator_names resolved in order)
    policy_joints: tuple[str, ...]
    noise_std: float = 0.1

    def build(self, env: ManagerBasedRlEnv) -> LowerBodyAction:
        return LowerBodyAction(self, env)
