"""A calendar ramp for the robot-side randomisation (Booster_T2 v4): observation noise, actuator gains and IMU
offsets start at their easy values and widen to full over the first iterations, so the gait forms before the hard
world arrives (a from-scratch teacher with everything at full from the start never learned to walk, t2_wbc_v3h).

One ``RampState`` per env (``env.dr_ramp``): the events read it through the env, the noise configs hold a reference
that ``make_env`` points at it after construction (mjlab's managers deep-copy their configs). The trainer sets
``progress`` from the iteration; nothing sets it in evaluation, where it stays at 1.0 (full)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.utils.noise import UniformNoiseCfg


@dataclass
class RampState:
    progress: float = 1.0  # 0 = easy, 1 = full


def ramp_progress(env) -> float:
    """The env's ramp progress (``env.dr_ramp``, set by ``make_env``); 1.0 = full when there is none."""
    ramp = getattr(env, "dr_ramp", None)
    return 1.0 if ramp is None else ramp.progress


@dataclass
class RampedUniformNoiseCfg(UniformNoiseCfg):
    """Additive uniform noise whose bounds grow from (n_min, n_max) to ``scale_to`` x that with the ramp."""

    scale_to: float = 2.0
    ramp: RampState = field(default_factory=RampState)

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        s = 1.0 + (self.scale_to - 1.0) * self.ramp.progress
        n_min = self._get_cached_tensor("n_min", self.n_min, data.device) * s
        n_max = self._get_cached_tensor("n_max", self.n_max, data.device) * s
        return data + torch.rand_like(data) * (n_max - n_min) + n_min
