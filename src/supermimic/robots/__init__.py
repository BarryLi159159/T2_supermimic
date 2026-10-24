"""Robot specifications: everything the tracking task needs to know about a humanoid, in one place.

The task code (env cfg, trainer, motion library, evaluation) never names a body or a joint itself; it asks the
``RobotSpec`` selected by ``--env.robot`` (``g1`` = the paper's Unitree G1, ``t2`` = Booster T2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg


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

    def load_model(self) -> mujoco.MjModel:
        """CPU MjModel of ``xml_path``; asserts the free joint comes first and the hinge order is ``joint_names``."""
        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
        assert tuple(names[1:]) == self.joint_names, f"joint order mismatch: {names[1:]}"
        assert model.nq == 7 + len(self.joint_names) and model.nv == 6 + len(self.joint_names)
        return model


def get_robot(name: str) -> RobotSpec:
    if name == "g1":
        from supermimic.robots.g1 import G1

        return G1
    if name == "t2":
        from supermimic.robots.t2 import T2

        return T2
    raise KeyError(f"unknown robot {name!r}; known: g1, t2")
