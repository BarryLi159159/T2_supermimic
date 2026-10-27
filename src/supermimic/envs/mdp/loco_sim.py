"""mjlab-side terms of the locomotion task, all evaluated through the analytic heads on the simulator state.

The reward term, the policy observation terms and the termination read ``LocoHeads`` (``world_model/loco_heads.py``)
on the state the simulator has just reached, so the MuJoCo reward is the very function the policy is optimised on
in imagination. The heads are built once per env (``env.loco_heads``) by ``make_loco_env``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from supermimic.envs.state_io import StateIO

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from supermimic.world_model.loco_heads import LocoHeads


_ROBOT = SceneEntityCfg("robot")


def heads_of(env: ManagerBasedRlEnv) -> LocoHeads:
    return env.loco_heads  # type: ignore[attr-defined]


class loco_reward:
    """The analytic reward on the simulator state: one term carrying the whole table (weight 1, dt applied inside)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.io = StateIO(env)
        self.prev_state = torch.zeros(env.num_envs, 71, device=env.device)
        self.command_name = cfg.params.get("command_name", "velocity")
        self.action_name = cfg.params.get("action_name", "joint_pos")
        self.last_terms: dict[str, torch.Tensor] = {}

    def reset(self, env_ids=None) -> None:
        # The reset state is written after this reward's reset; prev_state is refreshed on the next call.
        pass

    def __call__(self, env: ManagerBasedRlEnv, command_name: str = "velocity", action_name: str = "joint_pos") -> torch.Tensor:
        heads = heads_of(env)
        term = env.action_manager.get_term(action_name)
        cmd = env.command_manager.get_term(command_name).command
        s = self.io.read()
        # Envs reset since the previous call start a fresh episode: their previous state is unknown, use the current one.
        fresh = env.episode_length_buf <= 1
        s_prev = torch.where(fresh.unsqueeze(-1), s, self.prev_state)
        terminated = env.termination_manager.terminated.to(s.dtype)
        terms = heads.reward(s_prev, s, term.clean_raw, term.prev_clean_raw, cmd, terminated)
        self.prev_state = s
        self.last_terms = terms
        return terms["total"] / env.step_dt  # the manager multiplies by dt


def pelvis_height_below(env: ManagerBasedRlEnv, body_name: str, minimum_height: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    body_id = robot.body_names.index(body_name)
    return robot.data.body_link_pos_w[:, body_id, 2] < minimum_height


def reset_joints_by_scale(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, position_range: tuple[float, float], asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Isaac's reset_joints_by_scale: q = q_default * U(lo, hi), zero joint velocity, clipped to the soft limits."""
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    q0 = robot.data.default_joint_pos[env_ids]
    scale = torch.empty_like(q0).uniform_(*position_range)
    q = q0 * scale
    lim = robot.data.soft_joint_pos_limits[env_ids]
    q = q.clamp(lim[..., 0], lim[..., 1])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
