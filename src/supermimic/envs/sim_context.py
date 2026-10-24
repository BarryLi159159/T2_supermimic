"""Per-environment simulation context xi: foot friction 1 + encoder bias 29 + torso COM offset 3 + command delay 1 = 34.

The delay (physics steps, drawn once per episode by the actuator model) is a simulator parameter like the others:
the world model and the critic see it, the policy never does."""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 34


class SimContext:
    def __init__(
        self,
        env,
        foot_geom_regex: str,
        com_body_name: str = "torso_link",
        entity_name: str = "robot",
    ):
        self.env = env
        self.robot = env.scene[entity_name]
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(
            self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item()
        )
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]),
            dtype=torch.float32,
            device=env.device,
        )
        # Fused builtin actuators share one delay buffer; any actuator with a buffer reads the per-env lag.
        self.delay_buffer = next(
            (a._delay_buffer for a in self.robot.actuators if a._delay_buffer is not None), None
        )
        self.zero_lag = torch.zeros(env.num_envs, 1, device=env.device)

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        lag = (
            self.delay_buffer.current_lags.reshape(n, 1).float()
            if self.delay_buffer is not None
            else self.zero_lag
        )
        xi = torch.cat([friction.reshape(n, 1), bias, com, lag], dim=-1)
        assert xi.shape == (n, XI_DIM)
        return xi.clone()
