"""Analytic velocity-command locomotion reward (Booster_T2's Decoupled-WBC teacher table, state-only).

One differentiable implementation of IsaacLab-Decoupled-WBC's ``G1RewardCfg`` (with Booster_T2's T2 overrides)
shared by the simulator side (``envs/mdp/loco_sim.py``) and the imagination side (``world_model/loco_heads.py``),
so the policy is optimised in the world model on exactly the reward it collects in MuJoCo.

Every term is a function of the simulator state (through differentiable FK), the clean action, the command and
the actuator PD law. Three Isaac terms need contact sensors and have no state-only form, so they are left out:
``feet_air_time`` (contact timers), ``feet_force`` and ``feet_stumble`` (contact forces). Foot contact is a soft
indicator of the sole height (``sigmoid((h - sole_z) / w)``), which makes the fly / slide / standing-contact
terms differentiable.

When the command carries the gait clock (``cmd[..., 7:9]`` = [sin, cos] of the phase), ``feet_phase`` replaces
``feet_air_time``: an L1 penalty between each sole height and a swing reference (left foot swings in the first half
of the stride, right in the second, half-sine arcs of ``swing_height``). It is dense and its gradient does not
vanish at the ground, unlike every contact-gated or air-time form (v1-v6 stood or shuffled).

Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward manager.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from supermimic.kinematics.rotations import (
    projected_gravity,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

_EPS = 1.0e-12


def _norm(v: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last dim with a finite gradient at zero (BPTT passes through every term)."""
    return torch.sqrt((v * v).sum(-1) + _EPS)


def _roll_pitch_from_quat(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll and pitch of mjlab's euler_xyz_from_quat (XYZ extrinsic), with the asin argument kept inside (-1, 1) so the
    gradient stays finite when the trunk reaches +-90 deg pitch (a fallen robot in imagination)."""
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0 + 1.0e-6, 1.0 - 1.0e-6))
    return roll, pitch


@dataclass
class LocoRewardCfg:
    dt: float = 0.02
    # tracking kernels
    std_lin_vel: float = 0.5
    std_ang_vel: float = 0.5
    std_height: float = 0.1
    std_roll: float = 0.1
    std_pitch: float = 0.1
    std_yaw: float = 0.1
    w_track_lin_vel: float = 1.0
    w_track_ang_vel: float = 1.0
    w_track_height: float = 1.0
    w_track_roll: float = 1.0
    w_track_pitch: float = 1.5
    w_track_yaw: float = 1.0
    # posture
    w_cog: float = 0.5
    cog_sigma: float = 0.2
    w_knee_width: float = 0.5
    knee_ideal_width: float = 0.27 * 0.2010 / 0.2372  # G1's 0.27 m scaled by the T2 / G1 knee spacing
    knee_std: float = 0.1
    # Not in the Isaac table: feet_air_time (contact timers) has no state-only form, and without a gait term the first
    # world-model run (t2_loco_v1, 2026-10-27) walked at 0.28 m/s with both feet always on the ground. A moving robot is
    # rewarded for each foot's sole height as a linear ramp from ``swing_floor`` (below the ground: the world model
    # predicts a standing foot 1-2 cm under it, where a contact-gated term, v2-v4, had no gradient) to the swing
    # height, so the gradient is constant from the ground up and a real swing earns the full term (a wide Gaussian,
    # v5, paid 40 % of it for standing). The fly penalty keeps the robot from lifting both feet.
    w_feet_swing: float = 0.0
    swing_height: float = 0.08
    swing_floor: float = -0.015
    # Gait-clock swing reference (t2_loco_v7): -w * sum_feet |sole_z - z_ref(phase)| while moving.
    w_feet_phase: float = -10.0
    gait_period: float = 0.8  # s, must match the command term's
    w_feet_contact_standing: float = 0.5
    w_feet_flat_standing: float = 0.175
    feet_flat_dead_zone_deg: float = 3.0
    feet_flat_std_deg: float = 10.0
    feet_flat_roll_dead_zone: float = 0.5
    feet_flat_pitch_dead_zone: float = 1.1
    feet_flat_roll_scale: float = 0.35
    feet_flat_pitch_scale: float = 0.45
    # penalties
    w_lin_vel_z: float = -1.0
    w_ang_vel_xy: float = -0.1
    w_energy: float = -1e-3
    w_dof_acc: float = -2.5e-7
    w_action_rate: float = -0.01
    w_fly: float = -1.0
    w_flat_orientation: float = -2.0
    w_termination: float = -200.0
    w_feet_slide: float = -0.25
    w_feet_too_near: float = -2.0
    feet_too_near_threshold: float = 0.2 * 0.1928 / 0.2370  # G1's 0.2 m scaled by the ankle spacing
    w_dof_pos_limits: float = -2.0
    w_joint_deviation_hip: float = -0.5
    w_joint_deviation_waist: float = -0.2
    w_joint_deviation_legs: float = -0.02
    w_torque_limit_waist: float = -100.0
    soft_torque_waist: float = 0.9
    w_torque_limit_ankle: float = -100.0
    soft_torque_ankle: float = 0.99
    w_torque_limit_hip_pitch: float = -100.0
    soft_torque_hip_pitch: float = 0.99
    # Soft foot contact from the sole height: ~1 below 0.5 cm, ~0 above 2.5 cm. Until v6 it was (0.01, 0.01), which
    # gave a planted foot (sole -0.1 cm) only 0.75: standing paid fly 0.06 and lifting one foot quadrupled it.
    contact_height: float = 0.015
    contact_width: float = 0.004
    standing_cmd_threshold: float = 0.1

    @property
    def max_step_reward(self) -> float:
        """The positive terms' total weight x dt (normalised-return denominator)."""
        return self.dt * (
            self.w_track_lin_vel
            + self.w_track_ang_vel
            + self.w_track_height
            + self.w_track_roll
            + self.w_track_pitch
            + self.w_track_yaw
            + self.w_cog
            + self.w_knee_width
            + self.w_feet_contact_standing
            + self.w_feet_flat_standing
            + self.w_feet_swing
        )


@dataclass
class LocoJointIndex:
    """Joint subsets of the reward table, as index tensors in model joint order."""

    policy: torch.Tensor  # the 15 commanded joints (legs + waist): dof_acc, action rate
    hip_yaw_roll: torch.Tensor
    waist: torch.Tensor
    legs: torch.Tensor  # hip pitch, knee, ankle
    ankle: torch.Tensor
    hip_pitch: torch.Tensor

    @classmethod
    def from_names(cls, joint_names: tuple[str, ...], policy_joints: tuple[str, ...], device) -> LocoJointIndex:
        def ids(pred):
            return torch.tensor([i for i, n in enumerate(joint_names) if pred(n)], dtype=torch.long, device=device)

        return cls(
            policy=torch.tensor([joint_names.index(j) for j in policy_joints], dtype=torch.long, device=device),
            hip_yaw_roll=ids(lambda n: "hip_yaw" in n or "hip_roll" in n),
            waist=ids(lambda n: "waist" in n),
            legs=ids(lambda n: "hip_pitch" in n or "knee" in n or "ankle" in n),
            ankle=ids(lambda n: "ankle" in n),
            hip_pitch=ids(lambda n: "hip_pitch" in n),
        )


@dataclass
class LocoBodies:
    """World-frame link quantities the reward needs, from the simulator or from differentiable FK. Shapes (…, ·)."""

    pelvis_pos_w: torch.Tensor  # (…, 3)
    pelvis_quat_w: torch.Tensor  # (…, 4)
    pelvis_lin_vel_w: torch.Tensor  # (…, 3)
    pelvis_ang_vel_w: torch.Tensor  # (…, 3)
    trunk_quat_w: torch.Tensor  # (…, 4)
    height_z: torch.Tensor  # (…,) z of the height body
    feet_pos_w: torch.Tensor  # (…, 2, 3)
    feet_quat_w: torch.Tensor  # (…, 2, 4)
    feet_lin_vel_w: torch.Tensor  # (…, 2, 3)
    knee_pos_w: torch.Tensor  # (…, 2, 3)
    com_xy_w: torch.Tensor  # (…, 2) whole-body centre of mass


def soft_contact(feet_pos_w: torch.Tensor, sole_offset: float, cfg: LocoRewardCfg) -> torch.Tensor:
    """Soft foot-contact indicator (…, 2) from the sole height."""
    sole_z = feet_pos_w[..., 2] - sole_offset
    return torch.sigmoid((cfg.contact_height - sole_z) / cfg.contact_width)


def advance_clock(clock: torch.Tensor, dt: float, period: float) -> torch.Tensor:
    """The gait clock [sin, cos] (…, 2) one step later (a rotation by 2 pi dt / period)."""
    d = 2.0 * math.pi * dt / period
    c, s = math.cos(d), math.sin(d)
    sin, cos = clock[..., 0], clock[..., 1]
    return torch.stack([sin * c + cos * s, cos * c - sin * s], dim=-1)


def swing_reference(clock: torch.Tensor, height: float) -> torch.Tensor:
    """Sole-height reference (…, 2) [left, right]: half-sine arcs, left swings while sin > 0, right while sin < 0."""
    sin = clock[..., 0]
    return height * torch.stack([sin.clamp(min=0.0), (-sin).clamp(min=0.0)], dim=-1)


def _euler_rel_trunk(pelvis_quat_w: torch.Tensor, trunk_quat_w: torch.Tensor):
    """Roll / pitch of the trunk in the pelvis yaw frame and the trunk-vs-pelvis yaw difference (Isaac's
    track_body_{roll,pitch,yaw}_exp)."""
    pelvis_yaw = yaw_quat(pelvis_quat_w)
    q_rel = quat_mul(quat_conjugate(pelvis_yaw), trunk_quat_w)
    roll, pitch = _roll_pitch_from_quat(q_rel)
    roll = wrap_to_pi(roll)
    q_yaw = quat_mul(quat_conjugate(pelvis_yaw), yaw_quat(trunk_quat_w))
    yaw = 2.0 * torch.atan2(q_yaw[..., 3], q_yaw[..., 0])
    return roll, pitch, yaw


def loco_reward_terms(
    *,
    bodies: LocoBodies,
    joint_pos: torch.Tensor,  # (…, 29)
    joint_vel: torch.Tensor,  # (…, 29) of the state being rewarded
    prev_joint_vel: torch.Tensor,  # (…, 29) one control step earlier (joint acceleration)
    torque: torch.Tensor,  # (…, 29) PD torque behind the transition
    effort_limit: torch.Tensor,  # (29,)
    action: torch.Tensor,  # (…, 15) clean policy action
    prev_action: torch.Tensor,  # (…, 15)
    cmd: torch.Tensor,  # (…, 7) [vx, vy, wz, height, roll, pitch, yaw], or (…, 9) with the gait clock [sin, cos]
    term_prob: torch.Tensor,  # (…,) termination flag (sim) or probability (imagination)
    soft_limits: torch.Tensor,  # (29, 2)
    default_joint_pos: torch.Tensor,  # (29,)
    sole_offset: float,
    idx: LocoJointIndex,
    cfg: LocoRewardCfg,
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (…,) plus 'total'."""
    b = bodies
    dt = cfg.dt
    pelvis_yaw = yaw_quat(b.pelvis_quat_w)
    v_yaw = quat_apply_inverse(pelvis_yaw, b.pelvis_lin_vel_w)
    v_b = quat_apply_inverse(b.pelvis_quat_w, b.pelvis_lin_vel_w)
    w_b = quat_apply_inverse(b.pelvis_quat_w, b.pelvis_ang_vel_w)
    cmd_speed = _norm(cmd[..., :2]) + cmd[..., 2].abs()
    standing = (cmd_speed <= cfg.standing_cmd_threshold).to(joint_pos.dtype)

    # ---- tracking
    e_lin = ((cmd[..., :2] - v_yaw[..., :2]) ** 2).sum(-1)
    e_ang = (cmd[..., 2] - b.pelvis_ang_vel_w[..., 2]) ** 2
    e_height = (cmd[..., 3] - b.height_z) ** 2
    roll, pitch, yaw = _euler_rel_trunk(b.pelvis_quat_w, b.trunk_quat_w)
    e_roll = (roll - cmd[..., 4]) ** 2
    e_pitch = (cmd[..., 5] - pitch) ** 2
    e_yaw = (yaw - cmd[..., 6]) ** 2

    # ---- feet
    contact = soft_contact(b.feet_pos_w, sole_offset, cfg)  # (…, 2)
    both_down = contact[..., 0] * contact[..., 1]
    fly = (1.0 - contact[..., 0]) * (1.0 - contact[..., 1])
    feet_speed_xy = _norm(b.feet_lin_vel_w[..., :2])
    feet_slide = (feet_speed_xy * contact).sum(-1)
    feet_dist = _norm(b.feet_pos_w[..., 0, :] - b.feet_pos_w[..., 1, :])
    feet_too_near = (cfg.feet_too_near_threshold - feet_dist).clamp(min=0.0)
    g_feet = projected_gravity(b.feet_quat_w)  # (…, 2, 3)
    foot_tilt = torch.atan2(_norm(g_feet[..., :2]), -g_feet[..., 2])
    tilt_excess = (foot_tilt - math.radians(cfg.feet_flat_dead_zone_deg)).clamp(min=0.0)
    flatness = torch.exp(-((tilt_excess / math.radians(cfg.feet_flat_std_deg)) ** 2)).mean(-1)
    roll_excess = (cmd[..., 4].abs() - cfg.feet_flat_roll_dead_zone).clamp(min=0.0)
    pitch_excess = (cmd[..., 5].abs() - cfg.feet_flat_pitch_dead_zone).clamp(min=0.0)
    pose_gate = torch.exp(-((roll_excess / cfg.feet_flat_roll_scale) ** 2) - (pitch_excess / cfg.feet_flat_pitch_scale) ** 2)
    feet_flat_standing = standing * both_down * pose_gate * flatness
    feet_contact_standing = standing * both_down
    sole_z = b.feet_pos_w[..., 2] - sole_offset
    swing = ((sole_z - cfg.swing_floor) / (cfg.swing_height - cfg.swing_floor)).clamp(0.0, 1.0).sum(-1)
    feet_swing = (1.0 - standing) * swing
    if cmd.shape[-1] >= 9:
        feet_phase = (1.0 - standing) * (sole_z - swing_reference(cmd[..., 7:9], cfg.swing_height)).abs().sum(-1)
    else:
        feet_phase = torch.zeros_like(standing)

    # ---- posture
    feet_mid_xy = 0.5 * (b.feet_pos_w[..., 0, :2] + b.feet_pos_w[..., 1, :2])
    cog_dist_sq = ((b.com_xy_w - feet_mid_xy) ** 2).sum(-1)
    knee_width = _norm(b.knee_pos_w[..., 0, :2] - b.knee_pos_w[..., 1, :2])

    # ---- joints and actuators
    lo, hi = soft_limits[..., 0], soft_limits[..., 1]
    out_of_limits = ((lo - joint_pos).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)).sum(-1)
    dev = (joint_pos - default_joint_pos).abs()
    energy = _norm(torque * joint_vel)
    acc = (joint_vel[..., idx.policy] - prev_joint_vel[..., idx.policy]) / dt
    torque_ratio = torque.abs() / effort_limit

    def torque_excess(ids, soft):
        return (torque_ratio[..., ids] - soft).clamp(min=0.0).sum(-1)

    terms = {
        "track_lin_vel_xy_exp": dt * cfg.w_track_lin_vel * torch.exp(-e_lin / cfg.std_lin_vel**2),
        "track_ang_vel_z_exp": dt * cfg.w_track_ang_vel * torch.exp(-e_ang / cfg.std_ang_vel**2),
        "track_body_height_exp": dt * cfg.w_track_height * torch.exp(-e_height / cfg.std_height**2),
        "track_body_roll_exp": dt * cfg.w_track_roll * torch.exp(-e_roll / cfg.std_roll**2),
        "track_body_pitch_exp": dt * cfg.w_track_pitch * torch.exp(-e_pitch / cfg.std_pitch**2),
        "track_body_yaw_exp": dt * cfg.w_track_yaw * torch.exp(-e_yaw / cfg.std_yaw**2),
        "center_of_gravity_tracking": dt * cfg.w_cog * torch.exp(-cog_dist_sq / cfg.cog_sigma**2),
        "knee_stance_width": dt * cfg.w_knee_width * torch.exp(-((knee_width - cfg.knee_ideal_width) ** 2) / cfg.knee_std**2),
        "feet_contact_standing_stable": dt * cfg.w_feet_contact_standing * feet_contact_standing,
        "feet_flat_standing": dt * cfg.w_feet_flat_standing * feet_flat_standing,
        "feet_swing": dt * cfg.w_feet_swing * feet_swing,
        "feet_phase": dt * cfg.w_feet_phase * feet_phase,
        "lin_vel_z_l2": dt * cfg.w_lin_vel_z * v_b[..., 2] ** 2,
        "ang_vel_xy_l2": dt * cfg.w_ang_vel_xy * (w_b[..., :2] ** 2).sum(-1),
        "energy": dt * cfg.w_energy * energy,
        "dof_acc_l2": dt * cfg.w_dof_acc * (acc**2).sum(-1),
        "action_rate_l2": dt * cfg.w_action_rate * ((action - prev_action) ** 2).sum(-1),
        "fly": dt * cfg.w_fly * fly,
        "flat_orientation_l2": dt * cfg.w_flat_orientation * (projected_gravity(b.pelvis_quat_w)[..., :2] ** 2).sum(-1),
        "termination_penalty": dt * cfg.w_termination * term_prob,
        "feet_slide": dt * cfg.w_feet_slide * feet_slide,
        "feet_too_near": dt * cfg.w_feet_too_near * feet_too_near,
        "dof_pos_limits": dt * cfg.w_dof_pos_limits * out_of_limits,
        "joint_deviation_hip": dt * cfg.w_joint_deviation_hip * dev[..., idx.hip_yaw_roll].sum(-1),
        "joint_deviation_waists": dt * cfg.w_joint_deviation_waist * dev[..., idx.waist].sum(-1),
        "joint_deviation_legs": dt * cfg.w_joint_deviation_legs * dev[..., idx.legs].sum(-1),
        "joint_torque_limits_waists": dt * cfg.w_torque_limit_waist * torque_excess(idx.waist, cfg.soft_torque_waist),
        "joint_torque_limits_ankles": dt * cfg.w_torque_limit_ankle * torque_excess(idx.ankle, cfg.soft_torque_ankle),
        "joint_torque_limits_hip_pitch": dt * cfg.w_torque_limit_hip_pitch * torque_excess(idx.hip_pitch, cfg.soft_torque_hip_pitch),
    }
    total = None
    for v in terms.values():
        total = v if total is None else total + v
    terms["total"] = total
    return terms


LOCO_REWARD_TERM_NAMES = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "track_body_height_exp",
    "track_body_roll_exp",
    "track_body_pitch_exp",
    "track_body_yaw_exp",
    "center_of_gravity_tracking",
    "knee_stance_width",
    "feet_contact_standing_stable",
    "feet_flat_standing",
    "feet_swing",
    "feet_phase",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "energy",
    "dof_acc_l2",
    "action_rate_l2",
    "fly",
    "flat_orientation_l2",
    "termination_penalty",
    "feet_slide",
    "feet_too_near",
    "dof_pos_limits",
    "joint_deviation_hip",
    "joint_deviation_waists",
    "joint_deviation_legs",
    "joint_torque_limits_waists",
    "joint_torque_limits_ankles",
    "joint_torque_limits_hip_pitch",
)
