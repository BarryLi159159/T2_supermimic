"""Per-environment simulation context xi: foot friction 1 + encoder bias 29 + torso COM offset 3 + command delay 1
+ wrist payload 4 (left mass, left offset, right mass, right offset; zero without the event)
+ joint friction factor 29 + stiffness factor 29 + damping factor 29 + torso mass added [kg] 1 = 126.

The last four groups are Booster_T2 v4's robot-side randomisation (``--env.dr-v4``): the factors are read back from
the simulator (frictionloss / its default, the actuators' gains / their defaults, body mass - its default), so they
are exactly 1 / 0 when the events are off.

The delay (physics steps, drawn once per episode by the actuator model) is a simulator parameter like the others:
the world model and the critic see it, the policy never does."""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 126


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
        self.mass_nominal = float(env.sim.mj_model.body_mass[self.com_body_global])
        # Fused builtin actuators share one delay buffer; any actuator with a buffer reads the per-env lag.
        self.delay_buffer = next(
            (a._delay_buffer for a in self.robot.actuators if a._delay_buffer is not None), None
        )
        n, nj = env.num_envs, self.robot.num_joints
        self.zero_lag = torch.zeros(n, 1, device=env.device)
        self.zero_payload = torch.zeros(n, 4, device=env.device)
        # joint friction: the hinge dofs in joint order and their compiled (nominal) frictionloss
        self.dof_adr = self.robot.indexing.joint_v_adr.reshape(-1)
        self.friction_nominal = env.sim.get_default_field("dof_frictionloss")[self.dof_adr].to(env.device).float()
        self.ones_joints = torch.ones(n, nj, device=env.device)
        # actuator gains: each actuator's (num_envs, targets) stiffness / damping, scattered into joint order
        self.gain_actuators = [
            a
            for a in self.robot.actuators
            if hasattr(a, "default_stiffness") and a.default_stiffness is not None
        ]
        self.kp_factor = torch.ones(n, nj, device=env.device)
        self.kd_factor = torch.ones(n, nj, device=env.device)

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
        payload = getattr(self.env, "wrist_payload", None)
        if payload is None:
            payload = self.zero_payload
        fl = model.dof_frictionloss
        if fl.dim() == 2:  # expanded per world; a joint the model gives no friction reads 1
            nominal = self.friction_nominal
            joint_friction = torch.where(nominal > 0, fl[:, self.dof_adr] / nominal, self.ones_joints)
        else:
            joint_friction = self.ones_joints
        for a in self.gain_actuators:
            self.kp_factor[:, a.target_ids] = a.stiffness / a.default_stiffness
            self.kd_factor[:, a.target_ids] = a.damping / a.default_damping
        mass = model.body_mass
        if mass.dim() == 2:
            mass_add = (mass[:, self.com_body_global] - self.mass_nominal).reshape(n, 1)
        else:
            mass_add = self.zero_lag
        xi = torch.cat(
            [
                friction.reshape(n, 1),
                bias,
                com,
                lag,
                payload,
                joint_friction,
                self.kp_factor,
                self.kd_factor,
                mass_add,
            ],
            dim=-1,
        )
        assert xi.shape == (n, XI_DIM)
        return xi.clone()
