"""Extra domain-randomisation events for the T2 Sharpa version.

``wrist_payload``: a point mass on each hand, drawn per episode (TWIST2 v3's randomisation: 0-1.5 kg at 5-20 cm from
the wrist roll origin along the forearm axis, +y on the left hand and -y on the right), merged into the wrist body's
mass, COM and inertia with the parallel-axis theorem so the dynamics stay consistent. The sampled (mass, offset) per
hand is kept on ``env.wrist_payload`` for the simulation context xi (world model and critic see it, the policy never).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from supermimic.envs.mdp.ramp import ramp_progress

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

PAYLOAD_DIM = 4  # left mass, left offset, right mass, right offset
_ROBOT = SceneEntityCfg("robot")


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion of a batch of rotation matrices (Shepperd's method, branch-free)."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    t = 1 + m00 + m11 + m22
    w = torch.sqrt(torch.clamp(t, min=1e-12)) / 2
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)
    q = torch.stack([w, x, y, z], -1)
    return q / q.norm(dim=-1, keepdim=True)


def _point_shift(mass: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Inertia of a point mass at d about the origin: m (|d|^2 I - d d^T)."""
    eye = torch.eye(3, device=d.device, dtype=d.dtype)
    return mass[..., None, None] * (
        (d * d).sum(-1)[..., None, None] * eye - d[..., :, None] * d[..., None, :]
    )


@requires_model_fields(
    "body_mass", "body_ipos", "body_inertia", "body_iquat", recompute=RecomputeLevel.set_const
)
def wrist_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float],
    offset_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _ROBOT,
    body_names: tuple[str, str] = ("left_wrist_roll_link", "right_wrist_roll_link"),
    axis_signs: tuple[float, float] = (1.0, -1.0),
) -> None:
    robot = env.scene[asset_cfg.name]
    model = env.sim.model
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if not hasattr(env, "wrist_payload"):
        env.wrist_payload = torch.zeros(env.num_envs, PAYLOAD_DIM, device=env.device)
    n = env_ids.numel()
    for k, (name, sign) in enumerate(zip(body_names, axis_signs)):
        b = int(robot.indexing.body_ids[robot.body_names.index(name)].item())
        m0 = env.sim.get_default_field("body_mass")[b].expand(n)
        c0 = env.sim.get_default_field("body_ipos")[b].expand(n, 3)
        i0_principal = env.sim.get_default_field("body_inertia")[b].expand(n, 3)
        r0 = _quat_to_mat(env.sim.get_default_field("body_iquat")[b].expand(n, 4))
        i0 = (
            r0 @ torch.diag_embed(i0_principal) @ r0.transpose(-1, -2)
        )  # full tensor about the nominal COM, body frame

        m_add = torch.empty(n, device=env.device).uniform_(*mass_range)
        dist = torch.empty(n, device=env.device).uniform_(*offset_range)
        d = torch.zeros(n, 3, device=env.device)
        d[:, 1] = sign * dist
        m1 = m0 + m_add
        c1 = (m0[:, None] * c0 + m_add[:, None] * d) / m1[:, None]
        i1 = i0 + _point_shift(m0, c0 - c1) + _point_shift(m_add, d - c1)
        principal, axes = torch.linalg.eigh(i1)
        flip = torch.det(axes) < 0
        axes[flip, :, 0] *= -1

        model.body_mass[env_ids, b] = m1
        model.body_ipos[env_ids, b] = c1
        model.body_inertia[env_ids, b] = principal
        model.body_iquat[env_ids, b] = _mat_to_quat(axes)
        env.wrist_payload[env_ids, 2 * k] = m_add
        env.wrist_payload[env_ids, 2 * k + 1] = dist


def sensor_offsets(
    env,
    env_ids: torch.Tensor | None,
    gyro_bias_rad_s: float,
    imu_tilt_rad: float,
) -> None:
    """Per-episode IMU offsets (Booster_T2 v4): a gyro bias on each axis and a tilt of the measured gravity direction
    (roll, pitch; a mounting error), both uniform within +-full x the ramp (``env.dr_ramp``). Read by
    ``observations.base_ang_vel_biased`` / ``projected_gravity_tilted``; the policy sees them, xi does not (sensor, not dynamics)."""
    n = env.num_envs
    if not hasattr(env, "gyro_bias"):
        env.gyro_bias = torch.zeros(n, 3, device=env.device)
        env.imu_tilt = torch.zeros(n, 2, device=env.device)
    ids = slice(None) if env_ids is None else env_ids
    k = n if env_ids is None else len(env_ids)
    p = ramp_progress(env)
    env.gyro_bias[ids] = torch.empty(k, 3, device=env.device).uniform_(-1.0, 1.0) * (
        gyro_bias_rad_s * p
    )
    env.imu_tilt[ids] = torch.empty(k, 2, device=env.device).uniform_(-1.0, 1.0) * (
        imu_tilt_rad * p
    )


def pd_gains_ramped(
    env,
    env_ids: torch.Tensor | None,
    kp_scale: float,
    kd_scale: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Per-episode actuator gains x U(1 -+ scale x ramp) per joint (Booster_T2 v4: +-10 %, the sysID bound; the ramp is
    ``env.dr_ramp``). The factors are read back into xi from the actuators' stiffness / damping tensors."""
    p = ramp_progress(env)
    dr.pd_gains(
        env,
        env_ids,
        kp_range=(1.0 - kp_scale * p, 1.0 + kp_scale * p),
        kd_range=(1.0 - kd_scale * p, 1.0 + kd_scale * p),
        asset_cfg=asset_cfg if asset_cfg is not None else SceneEntityCfg("robot"),
        operation="scale",
    )
