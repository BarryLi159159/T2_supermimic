"""Adversarial motion prior for the locomotion task (AMP, Peng et al. 2021), differentiable edition (t2_loco_v11).

A discriminator of consecutive-state features is trained to tell the reference walking clips (the RSI clips) from the
policy's simulator transitions in the replay buffer (least-squares GAN, gradient penalty on the reference samples). In
imagination the style reward ``max(0, 1 - 0.25 (D - 1)^2)`` is added to the task reward while the command moves, and
its gradient reaches the policy through the imagined state (BPTT, not a score-function estimator). Unlike the contact
timers of the teacher table it has a gradient at a standing robot, which is what t2_loco_v9 / v10 lacked: their
imagined feet_air_time stayed 0 and no foot ever lifted.

Features (46): the 15 policy joints' offset from the default pose and velocity, root height, projected gravity, root
linear velocity in the heading frame, root angular velocity in the body frame, both feet relative to the root in the
heading frame. A transition is the features of two consecutive states (92).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.loco_rewards import LocoBodies, standing_indicator
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.kinematics.rotations import heading_quat, projected_gravity, quat_apply_inverse
from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import split_state
from supermimic.world_model.loco_heads import LocoHeads

AMP_FEAT_DIM = 46
_REF_CHUNK = 8192


@dataclass
class LocoAmpCfg:
    weight: float = 0.0  # style reward weight (x dt, like the reward table); 0 = no discriminator
    hidden: tuple[int, ...] = (1024, 512)
    lr: float = 1e-4
    grad_penalty: float = 10.0  # on the reference samples (AMP's default)
    batch_size: int = 4096  # policy transitions (and as many reference ones) per discriminator step
    moving_only: bool = (
        True  # reward and policy samples only under a moving command (the clips walk)
    )


def amp_features(s: torch.Tensor, bodies: LocoBodies, heads: LocoHeads) -> torch.Tensor:
    """(…, 46) heading-invariant features of one state."""
    root_pos, root_quat, q, v_w, w_b, qd = split_state(s)
    hq = heading_quat(root_quat)
    ids = heads.policy_ids
    feet_rel = quat_apply_inverse(
        hq[..., None, :].expand(*bodies.feet_pos_w.shape[:-1], 4),
        bodies.feet_pos_w - root_pos[..., None, :],
    )
    return torch.cat(
        [
            (q - heads.default_joint_pos)[..., ids],
            qd[..., ids],
            root_pos[..., 2:3],
            projected_gravity(root_quat),
            quat_apply_inverse(hq, v_w),
            w_b,
            feet_rel.flatten(-2),
        ],
        dim=-1,
    )


class LocoDiscriminator(nn.Module):
    """D(transition) on the reference-normalised 92-d transition features; +1 = reference, -1 = policy."""

    def __init__(self, cfg: LocoAmpCfg):
        super().__init__()
        self.mlp = MLP(2 * AMP_FEAT_DIM, cfg.hidden, 1)
        self.register_buffer("mean", torch.zeros(2 * AMP_FEAT_DIM))
        self.register_buffer("std", torch.ones(2 * AMP_FEAT_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp((x - self.mean) / self.std).squeeze(-1)

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        """AMP's style reward in [0, 1]: 1 where D says reference, 0 beyond D = -1."""
        return (1.0 - 0.25 * (self(x) - 1.0) ** 2).clamp(min=0.0)


class LocoAmp:
    """The reference transition set, the discriminator and its training step."""

    def __init__(self, cfg: LocoAmpCfg, heads: LocoHeads, npz_files: list[str], device):
        self.cfg, self.heads = cfg, heads
        self.disc = LocoDiscriminator(cfg).to(device)
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=cfg.lr)
        self.ref_x = self._reference_transitions(npz_files, device)
        self.disc.mean[:] = self.ref_x.mean(0)
        self.disc.std[:] = self.ref_x.std(0).clamp(min=1.0e-3)

    @torch.no_grad()
    def _reference_transitions(self, npz_files: list[str], device) -> torch.Tensor:
        spec = self.heads.spec
        lib = MotionLibrary(
            npz_files,
            spec.tracked_bodies,
            self.heads.default_joint_pos,
            device=device,
            compact=True,
            store_dtype=torch.float32,
            root_body_name=spec.root_body,
            fk_model=spec.load_model(),
        )
        feats = []
        for lo in range(0, lib.num_frames, _REF_CHUNK):
            r = lib.ref(torch.arange(lo, min(lo + _REF_CHUNK, lib.num_frames), device=device))
            # the clips store the root angular velocity in the world frame, the simulator state in the body frame
            s = torch.cat(
                [
                    r.root_pos_w,
                    r.root_quat_w,
                    r.joint_pos,
                    r.root_lin_vel_w,
                    quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w),
                    r.joint_vel,
                ],
                dim=-1,
            )
            feats.append(amp_features(s, self.heads.bodies(s), self.heads))
        feat = torch.cat(feats)
        frame = torch.arange(lib.num_frames, device=device)
        first = frame[
            frame != lib.last_frame_of(frame)
        ]  # a transition never crosses a clip boundary
        return torch.cat([feat[first], feat[first + 1]], dim=-1)

    def policy_transitions(self, buffer: ReplayBuffer) -> torch.Tensor:
        """(m, 92) features of replay transitions, the moving-command ones when ``moving_only``."""
        w = buffer.sample_windows(self.cfg.batch_size, 1)
        with torch.no_grad():
            x = torch.cat(
                [
                    amp_features(w.states[i], self.heads.bodies(w.states[i]), self.heads)
                    for i in range(2)
                ],
                dim=-1,
            )
        if self.cfg.moving_only:
            x = x[standing_indicator(w.goal[0], self.heads.reward_cfg) < 0.5]
        return x

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        x_pol = self.policy_transitions(buffer)
        if x_pol.shape[0] == 0:
            return {}
        x_ref = self.ref_x[
            torch.randint(0, self.ref_x.shape[0], (x_pol.shape[0],), device=x_pol.device)
        ].requires_grad_(True)
        d_ref, d_pol = self.disc(x_ref), self.disc(x_pol)
        (grad,) = torch.autograd.grad(d_ref.sum(), x_ref, create_graph=True)
        gp = (grad * grad).sum(-1).mean()
        loss = (
            0.5 * ((d_ref - 1.0) ** 2).mean()
            + 0.5 * ((d_pol + 1.0) ** 2).mean()
            + self.cfg.grad_penalty * gp
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {
            "amp_disc_loss": float(loss.detach()),
            "amp_disc_ref": float(d_ref.mean().detach()),
            "amp_disc_pol": float(d_pol.mean().detach()),
            "amp_disc_acc": float(
                ((d_ref > 0).float().mean() + (d_pol < 0).float().mean()).detach() / 2
            ),
            "amp_grad_penalty": float(gp.detach()),
        }

    def state_dict(self) -> dict:
        return {"amp_disc": self.disc.state_dict(), "amp_opt": self.opt.state_dict()}

    def load_state_dict(self, ck: dict) -> None:
        self.disc.load_state_dict(ck["amp_disc"])
        self.opt.load_state_dict(ck["amp_opt"])
