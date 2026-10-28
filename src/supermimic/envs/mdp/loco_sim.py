"""mjlab-side terms of the locomotion task, all evaluated through the analytic heads on the simulator state.

``reset_from_clip`` is the locomotion task's reference-state initialisation: a fraction of the resets start from a
random frame of a motion clip (the T2 walking clips), so the replay buffer and the world model see stepping
dynamics that a shuffling policy never produces (t2_loco_v1 / v2, 2026-10-27).

The reward term, the policy observation terms and the termination read ``LocoHeads`` (``world_model/loco_heads.py``)
on the state the simulator has just reached, so the MuJoCo reward is the very function the policy is optimised on
in imagination. The heads are built once per env (``env.loco_heads``) by ``make_loco_env``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from supermimic.envs.mdp.loco_rewards import advance_foot_timers, timer_contact
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
        self.foot_timers = torch.zeros(env.num_envs, 4, device=env.device)  # [air L, air R, contact L, contact R] of the current state (the replay recorder stores them)

    def reset(self, env_ids=None) -> None:
        # The reset state is written after this reward's reset; prev_state is refreshed on the next call.
        self.foot_timers[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, command_name: str = "velocity", action_name: str = "joint_pos", gait_clock: bool = False) -> torch.Tensor:
        heads = heads_of(env)
        term = env.action_manager.get_term(action_name)
        cmd_term = env.command_manager.get_term(command_name)
        # With the gait clock the reward sees the goal the policy sees (the reward runs before the command term
        # advances the phase, so this is the phase the policy acted on).
        cmd = torch.cat([cmd_term.command, cmd_term.clock], dim=-1) if gait_clock else cmd_term.command
        s = self.io.read()
        # Envs reset since the previous call start a fresh episode: their previous state is unknown, use the current one.
        fresh = env.episode_length_buf <= 1
        s_prev = torch.where(fresh.unsqueeze(-1), s, self.prev_state)
        terminated = env.termination_manager.terminated.to(s.dtype)
        b = heads.bodies(s)
        rcfg = heads.reward_cfg
        self.foot_timers[:] = advance_foot_timers(self.foot_timers, timer_contact(b.feet_pos_w, heads.spec.sole_offset, rcfg), rcfg.dt)
        timers = self.foot_timers if rcfg.foot_timers else None
        terms = heads.reward(s_prev, s, term.clean_raw, term.prev_clean_raw, cmd, terminated, bodies=b, foot_timers=timers)
        self.prev_state = s
        self.last_terms = terms
        return terms["total"] / env.step_dt  # the manager multiplies by dt


def gait_clock(env: ManagerBasedRlEnv, command_name: str = "velocity") -> torch.Tensor:
    """[sin, cos] of the command term's gait phase (the goal group's second term)."""
    return env.command_manager.get_term(command_name).clock


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


def reset_from_clip(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, npz_files: list[str], prob: float, robot_name: str, asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Start a ``prob`` fraction of the resetting envs from a random frame of the clips (pose and velocity)."""
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.robots import get_robot

    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    lib = getattr(env, "rsi_library", None)
    if lib is None:
        spec = get_robot(robot_name)
        lib = MotionLibrary(npz_files, spec.tracked_bodies, robot.data.default_joint_pos[0], device=env.device, compact=True, root_body_name=spec.root_body, fk_model=spec.load_model())
        env.rsi_library = lib
        print(f"[INFO] reset_from_clip: {lib.summary()}")
    pick = env_ids[torch.rand(env_ids.numel(), device=env.device) < prob]
    if pick.numel() == 0:
        return
    frame = torch.randint(0, lib.num_frames, (pick.numel(),), device=env.device)
    r = lib.ref(frame)
    lim = robot.data.soft_joint_pos_limits[pick]
    joint_pos = r.joint_pos.clamp(lim[..., 0], lim[..., 1])
    robot.write_joint_state_to_sim(joint_pos, r.joint_vel, env_ids=pick)
    root = torch.cat([r.root_pos_w + env.scene.env_origins[pick], r.root_quat_w, r.root_lin_vel_w, r.root_ang_vel_w], dim=-1)
    robot.write_root_state_to_sim(root, env_ids=pick)
