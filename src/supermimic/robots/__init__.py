"""Robot specifications: everything the tracking task needs to know about a humanoid, in one place.

The task code (env cfg, trainer, motion library, evaluation) never names a body or a joint itself; it asks the
``RobotSpec`` selected by ``--env.robot`` (``g1`` = the paper's Unitree G1, ``t2`` = Booster T2).

The locomotion task (velocity-command walking, ``envs/loco_env_cfg.py``) needs a few more names: the pelvis and the
height body of the Decoupled-WBC command convention, the feet and knees, the policy's joint subset and the PD table
of the actuators. They are optional (``None`` on robots that only track motions).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class PDGains:
    """Per-joint PD gains and effort limits in model joint order (the analytic torque of the locomotion rewards)."""

    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort: tuple[float, ...]


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path: Path  # standalone CPU model: differentiable FK, data conversion, ghost
    entity_cfg: Callable[[], EntityCfg]  # mjlab entity: actuators, gains, keyframe, collisions
    action_scale: dict[str, float]  # rad per unit action, per joint expression (0.25 * effort / kp)
    joint_names: tuple[str, ...]  # model joint order = policy / npz order
    root_body: str  # the free-joint body: paper terminations, world-model root state
    anchor_body: str  # BeyondMimic anchor: global pos / ori rewards, relative body errors
    tracked_bodies: tuple[str, ...]  # bodies in the reward and the reference library
    foot_geom_regex: str  # friction domain randomisation and xi
    com_body: str  # COM offset domain randomisation and xi
    # ---- locomotion task (optional)
    pelvis_body: str | None = None  # velocity command frame and flat-orientation body (G1 pelvis convention)
    height_body: str | None = None  # the body whose z the height command refers to
    foot_bodies: tuple[str, str] | None = None  # (left, right) foot links
    knee_bodies: tuple[str, str] | None = None  # (left, right) knee links
    sole_offset: float = 0.0  # sole plane below the foot link origin [m]
    policy_joints: tuple[str, ...] | None = None  # the joints the locomotion policy commands (legs + waist)
    pd_gains: Callable[[], PDGains] | None = None  # kp / kd / effort per joint in model order

    def load_model(self) -> mujoco.MjModel:
        """CPU MjModel of ``xml_path``; asserts the free joint comes first and the hinge order is ``joint_names``."""
        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
        assert tuple(names[1:]) == self.joint_names, f"joint order mismatch: {names[1:]}"
        assert model.nq == 7 + len(self.joint_names) and model.nv == 6 + len(self.joint_names)
        return model

    def policy_joint_ids(self) -> list[int]:
        """Indices of ``policy_joints`` in model joint order."""
        assert self.policy_joints is not None, f"{self.name} has no locomotion joint subset"
        return [self.joint_names.index(j) for j in self.policy_joints]


def get_robot(name: str) -> RobotSpec:
    if name == "g1":
        from supermimic.robots.g1 import G1

        return G1
    if name == "t2":
        from supermimic.robots.t2 import T2

        return T2
    if name == "t2_nodelay":
        from supermimic.robots.t2 import T2_NODELAY

        return T2_NODELAY
    if name == "t2_sharpa":
        from supermimic.robots.t2 import T2_SHARPA

        return T2_SHARPA
    if name == "t2_v4":
        from supermimic.robots.t2 import T2_V4

        return T2_V4
    if name == "t2_v4_sharpa":
        from supermimic.robots.t2 import T2_V4_SHARPA

        return T2_V4_SHARPA
    raise KeyError(f"unknown robot {name!r}; known: g1, t2, t2_nodelay, t2_sharpa, t2_v4, t2_v4_sharpa")
