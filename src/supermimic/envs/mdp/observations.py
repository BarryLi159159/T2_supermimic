from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def base_ang_vel_biased(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Root angular velocity in the body frame plus the per-episode gyro bias (events.sensor_offsets)."""
    w = env.scene[entity_name].data.root_link_ang_vel_b
    bias = getattr(env, "gyro_bias", None)
    return w if bias is None else w + bias


def projected_gravity_tilted(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Gravity in the body frame as an IMU mounted with a small roll / pitch error measures it (events.sensor_offsets):
    g_meas = Ry(-pitch) Rx(-roll) g_b."""
    g = env.scene[entity_name].data.projected_gravity_b
    tilt = getattr(env, "imu_tilt", None)
    if tilt is None:
        return g
    cr, sr = torch.cos(tilt[:, 0]), torch.sin(tilt[:, 0])
    cp, sp = torch.cos(tilt[:, 1]), torch.sin(tilt[:, 1])
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    # Rx(-roll)
    gy, gz = cr * gy + sr * gz, -sr * gy + cr * gz
    # Ry(-pitch)
    gx, gz = cp * gx - sp * gz, sp * gx + cp * gz
    return torch.stack([gx, gy, gz], dim=-1)
