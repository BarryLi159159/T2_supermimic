"""Velocity + body-pose command of the Decoupled-WBC lower-body task (7-d, piecewise constant).

``[vx, vy, yaw rate, body height, torso roll, torso pitch, torso yaw]``: linear velocity in the pelvis yaw frame,
yaw rate about world z, height of the robot's height body, torso angles in the pelvis yaw frame. Sampled uniformly
in the per-axis ranges every ``resampling_time_range`` seconds; a fraction of the envs stand (zero velocity) and
half of the moving envs get an exact zero yaw rate (IsaacLab-Decoupled-WBC's ``UniformVelocityBodyCommand`` without
the heading-feedback mode, which has no constant-goal form).

The torso axes widen from ``*_start`` to the full ranges with ``progress`` in [0, 1] (the Isaac per-axis curriculum;
the trainer sets it from the iteration). Evaluation pins a command with ``set_fixed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

CMD_DIM = 7
AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "body_height", "body_roll", "body_pitch", "body_yaw")


class VelocityBodyCommand(CommandTerm):
    cfg: VelocityBodyCommandCfg

    def __init__(self, cfg: VelocityBodyCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.vel_command_b = torch.zeros(n, CMD_DIM, device=self.device)
        self.is_standing_env = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fixed_command = torch.zeros(n, CMD_DIM, device=self.device)
        self.progress = 0.0
        self.metrics["error_vel_xy"] = torch.zeros(n, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    # ------------------------------------------------------------------ curriculum / evaluation hooks

    def set_progress(self, progress: float) -> None:
        self.progress = float(min(max(progress, 0.0), 1.0))

    def set_fixed(self, env_ids: torch.Tensor, command: torch.Tensor) -> None:
        """Pin ``command`` (n, 7) on ``env_ids`` from now on (resamples keep it)."""
        self.fixed[env_ids] = True
        self.fixed_command[env_ids] = command
        self.vel_command_b[env_ids] = command

    def _range(self, axis: str) -> tuple[float, float]:
        end = getattr(self.cfg, axis)
        start = getattr(self.cfg, f"{axis}_start", None)
        if start is None:
            return end
        p = self.progress
        return (start[0] + p * (end[0] - start[0]), start[1] + p * (end[1] - start[1]))

    # ------------------------------------------------------------------ CommandTerm

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        for k, axis in enumerate(AXES):
            self.vel_command_b[env_ids, k] = r.uniform_(*self._range(axis))
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs
        zero_yaw = (r.uniform_(0.0, 1.0) < self.cfg.rel_zero_yaw_envs) & ~self.is_standing_env[env_ids]
        self.vel_command_b[env_ids, 2] = torch.where(zero_yaw, torch.zeros_like(r), self.vel_command_b[env_ids, 2])
        standing = self.is_standing_env[env_ids].unsqueeze(-1)
        self.vel_command_b[env_ids, :3] = torch.where(standing, torch.zeros_like(self.vel_command_b[env_ids, :3]), self.vel_command_b[env_ids, :3])
        fixed = self.fixed[env_ids]
        if fixed.any():
            self.vel_command_b[env_ids] = torch.where(fixed.unsqueeze(-1), self.fixed_command[env_ids], self.vel_command_b[env_ids])

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass  # piecewise constant between resamples

    def _update_metrics(self) -> None:
        robot = self._env.scene[self.cfg.entity_name]
        max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += torch.norm(self.vel_command_b[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1) / max_steps
        self.metrics["error_vel_yaw"] += (self.vel_command_b[:, 2] - robot.data.root_link_ang_vel_w[:, 2]).abs() / max_steps


@dataclass(kw_only=True)
class VelocityBodyCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (3.0, 6.0)
    rel_standing_envs: float = 0.4
    rel_zero_yaw_envs: float = 0.5  # of the moving envs
    # Full ranges (Booster_T2 config/t2_flat_config.py: G1's table, height scaled, roll capped at T2's 0.50).
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z: tuple[float, float] = (-1.57, 1.57)
    body_height: tuple[float, float] = (0.55, 0.88)
    body_roll: tuple[float, float] = (-0.5, 0.5)
    body_pitch: tuple[float, float] = (-0.52, 1.57)
    body_yaw: tuple[float, float] = (-1.57, 1.57)
    # Curriculum start ranges (progress 0); None = no curriculum on that axis.
    body_height_start: tuple[float, float] | None = (0.567, 0.88)
    body_roll_start: tuple[float, float] | None = (-0.4, 0.4)
    body_pitch_start: tuple[float, float] | None = (-0.3, 1.0)
    body_yaw_start: tuple[float, float] | None = (-0.65, 0.65)
    lin_vel_x_start: tuple[float, float] | None = None
    lin_vel_y_start: tuple[float, float] | None = None
    ang_vel_z_start: tuple[float, float] | None = None

    def build(self, env: ManagerBasedRlEnv) -> VelocityBodyCommand:
        return VelocityBodyCommand(self, env)
