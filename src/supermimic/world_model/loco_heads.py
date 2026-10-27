"""Analytic heads of the locomotion world model: observation, reward, termination features.

Everything is a differentiable function of the 71-d simulator state (through the batched FK), the clean action,
the command and the sim context xi, so one implementation serves the MuJoCo side (``envs/mdp/loco_sim.py``) and the
imagined rollouts (``imagination/loco_rollout.py``).

The policy's observation is Booster_T2's deployable student layout (58): trunk angular velocity 3, trunk projected
gravity 3, command 7, the 15 policy joints' position offset and velocity, last action 15. The termination features
are the pelvis height above the Isaac threshold and the pelvis tilt (0 upright); only the height terminates in the
simulator, the learned head sees both.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from supermimic.envs.mdp.loco_rewards import (
    LocoBodies,
    LocoJointIndex,
    LocoRewardCfg,
    loco_reward_terms,
)
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import projected_gravity, quat_apply
from supermimic.robots import RobotSpec
from supermimic.world_model.heads import split_state
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

CMD_DIM = 7
POLICY_ACT_DIM = 15
LOCO_OBS_DIM = 3 + 3 + CMD_DIM + 3 * POLICY_ACT_DIM  # 58
LOCO_OBS_TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("command", CMD_DIM), ("joint_pos", POLICY_ACT_DIM), ("joint_vel", POLICY_ACT_DIM), ("actions", POLICY_ACT_DIM))


@dataclass
class LocoTerminationCfg:
    min_pelvis_height: float = 0.3 * 0.7754 / 0.7429  # Isaac's 0.3 m scaled by the T2 / G1 leg length


class LocoHeads:
    """FK-based bodies, PD torque, reward, observation and termination features from the predicted state."""

    def __init__(
        self,
        fk: G1Kinematics,
        spec: RobotSpec,
        model: mujoco.MjModel,
        soft_limits: torch.Tensor,  # (29, 2)
        default_joint_pos: torch.Tensor,  # (29,)
        action_scale: torch.Tensor,  # (29,) rad per unit action, model joint order
        reward_cfg: LocoRewardCfg | None = None,
        term_cfg: LocoTerminationCfg | None = None,
    ):
        assert spec.pelvis_body and spec.height_body and spec.foot_bodies and spec.knee_bodies and spec.policy_joints and spec.pd_gains
        dev = fk.device
        self.fk = fk
        self.spec = spec
        self.reward_cfg = reward_cfg or LocoRewardCfg()
        self.term_cfg = term_cfg or LocoTerminationCfg()
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.action_scale = action_scale
        self.pelvis = fk.body_index(spec.pelvis_body)
        self.trunk = fk.body_index(spec.root_body)
        self.height = fk.body_index(spec.height_body)
        self.feet = fk.body_indices(spec.foot_bodies)
        self.knees = fk.body_indices(spec.knee_bodies)
        self.policy_ids = torch.tensor(spec.policy_joint_ids(), dtype=torch.long, device=dev)
        self.idx = LocoJointIndex.from_names(spec.joint_names, spec.policy_joints, dev)
        gains = spec.pd_gains()
        self.kp = torch.tensor(gains.kp, dtype=torch.float32, device=dev)
        self.kd = torch.tensor(gains.kd, dtype=torch.float32, device=dev)
        self.effort = torch.tensor(gains.effort, dtype=torch.float32, device=dev)
        # Nominal masses and COM offsets of every body (world body excluded), for the whole-body COM.
        self.body_mass = torch.tensor(np.array(model.body_mass[1:]), dtype=torch.float32, device=dev)
        self.body_ipos = torch.tensor(np.array(model.body_ipos[1:]), dtype=torch.float32, device=dev)
        self.total_mass = self.body_mass.sum()

    # ------------------------------------------------------------------ kinematics

    def bodies(self, s: torch.Tensor) -> LocoBodies:
        out = self.fk(s[..., :36], s[..., 36:])
        pos, quat, lin, ang = out["pos"], out["quat"], out["lin_vel"], out["ang_vel"]
        com = pos + quat_apply(quat, self.body_ipos.expand(pos.shape))
        com_xy = (self.body_mass[:, None] * com[..., :2]).sum(-2) / self.total_mass
        return LocoBodies(
            pelvis_pos_w=pos[..., self.pelvis, :],
            pelvis_quat_w=quat[..., self.pelvis, :],
            pelvis_lin_vel_w=lin[..., self.pelvis, :],
            pelvis_ang_vel_w=ang[..., self.pelvis, :],
            trunk_quat_w=quat[..., self.trunk, :],
            height_z=pos[..., self.height, 2],
            feet_pos_w=pos[..., self.feet, :],
            feet_quat_w=quat[..., self.feet, :],
            feet_lin_vel_w=lin[..., self.feet, :],
            knee_pos_w=pos[..., self.knees, :],
            com_xy_w=com_xy,
        )

    def torque(self, s: torch.Tensor, action29: torch.Tensor) -> torch.Tensor:
        """PD torque of the actuators for the executed (29-d, normalised) action, clipped at the effort limit."""
        _, _, q, _, _, qd = split_state(s)
        target = self.default_joint_pos + action29 * self.action_scale
        tau = self.kp * (target - q) - self.kd * qd
        return torch.maximum(torch.minimum(tau, self.effort), -self.effort)

    # ------------------------------------------------------------------ heads

    def reward(self, s_prev: torch.Tensor, s_next: torch.Tensor, action29: torch.Tensor, prev_action29: torch.Tensor, cmd: torch.Tensor, term_prob: torch.Tensor, bodies: LocoBodies | None = None) -> dict[str, torch.Tensor]:
        """Reward r_{t+1} of reaching s_next from s_prev under action29 (the clean 29-d action, arms included)."""
        b = bodies if bodies is not None else self.bodies(s_next)
        _, _, q, _, _, qd = split_state(s_next)
        qd_prev = split_state(s_prev)[5]
        return loco_reward_terms(
            bodies=b,
            joint_pos=q,
            joint_vel=qd,
            prev_joint_vel=qd_prev,
            torque=self.torque(s_prev, action29),
            effort_limit=self.effort,
            action=action29[..., self.policy_ids],
            prev_action=prev_action29[..., self.policy_ids],
            cmd=cmd,
            term_prob=term_prob,
            soft_limits=self.soft_limits,
            default_joint_pos=self.default_joint_pos,
            sole_offset=self.spec.sole_offset,
            idx=self.idx,
            cfg=self.reward_cfg,
        )

    def observation(self, s: torch.Tensor, prev_action15: torch.Tensor, cmd: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Noise-free 58-d policy observation. xi[..., 1:30] is the encoder bias (observed q = q + bias). ``cmd`` may
        carry the gait clock after the 7 command numbers; the observation holds the command only."""
        _, root_quat, q, _, ang_b, qd = split_state(s)
        bias = xi[..., 1:30]
        ids = self.policy_ids
        q_obs = (q + bias - self.default_joint_pos)[..., ids]
        return torch.cat([ang_b, projected_gravity(root_quat), cmd[..., :CMD_DIM], q_obs, qd[..., ids], prev_action15], dim=-1)

    def termination_features(self, bodies: LocoBodies) -> torch.Tensor:
        """[pelvis height above the threshold, pelvis tilt (1 + g_z, 0 upright)] (…, 2)."""
        height = bodies.pelvis_pos_w[..., 2] - self.term_cfg.min_pelvis_height
        tilt = 1.0 + projected_gravity(bodies.pelvis_quat_w)[..., 2]
        return torch.stack([height, tilt], dim=-1)

    def terminated(self, bodies: LocoBodies) -> torch.Tensor:
        return bodies.pelvis_pos_w[..., 2] < self.term_cfg.min_pelvis_height


class LocoWorldModel(WorldModel):
    """The world model with the locomotion termination features (pelvis height / tilt via FK, no reference)."""

    heads: LocoHeads

    def __init__(self, cfg: WorldModelCfg, heads: LocoHeads, device):
        super().__init__(cfg, heads, device)

    def termination_logits(self, s_next: torch.Tensor, bodies: LocoBodies | None = None) -> torch.Tensor:  # type: ignore[override]
        b = bodies if bodies is not None else self.heads.bodies(s_next)
        return self.term_head(self.heads.termination_features(b))

    def _loss_step(self, s, a, xi, s_true, done):  # type: ignore[override]
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        logits = self.termination_logits(s_next)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor):  # type: ignore[override]
        """k-step autoregressive loss (Eq. 1): states (k+1, n, 71), a_exec (k, n, 29), xi (n, X), done (k, n)."""
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j])
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        metrics = {
            "state_loss": (total_state / k).detach(),
            "term_bce": (total_term / k).detach(),
            "nse_1step": err_first.detach(),
            "nse_kstep": err_last.detach(),
            "term_acc": (acc_num / k).detach(),
        }
        return loss, metrics
