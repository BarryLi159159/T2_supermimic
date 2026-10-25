"""Booster T2 (29 actuated joints, head welded) under Booster's deployment convention ("v3", Booster_T2 2026-10-22).

Model: ``assets/t2/t2_29dof_trunk.xml`` = Booster's official T2 MJCF (booster_deploy / t2_dance) with the two head
joints welded at zero. The free body is the trunk, which carries the IMU, so the policy's proprioception
(``base_ang_vel``, ``projected_gravity`` of the root link) is the real IMU's signal with no frame change on deployment.

Actuators are Booster's model from booster_train: a PD law whose target arrives with a lag of 0..8 physics steps
(5 ms each, drawn per episode), clipped on a torque-speed curve (full torque up to the knee speed, zero at the
no-load speed), rotor armature per motor; the parallel ankle and waist joints scale a base motor by the linkage
ratios. mjlab's ``DcMotorActuatorCfg`` draws the same curve when its stall torque is ``effort * v_max / (v_max -
v_knee)``: min(force_limit, that line) is Booster's piecewise curve exactly (above the no-load speed mjlab brakes
where Booster outputs zero, which no tracked motion reaches). Gains are the "hard" t2_walk table that has run on the
real T2 (natural frequency 10 Hz), wrists 18.21 / 1.449 for the hand's own weight. Action scale is Booster's
0.25 * effort / kp per joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from supermimic.robots import RobotSpec

T2_XML = Path(__file__).resolve().parents[3] / "assets" / "t2" / "t2_29dof_trunk.xml"
T2_SHARPA_XML = T2_XML.with_name(
    "t2_29dof_trunk_sharpa.xml"
)  # wrist links carry the Sharpa Wave hands (1.64 kg)

PHYSICS_DT = 0.005
DELAY_MAX_STEPS = 8  # 0-40 ms command latency, booster_train's range

RPM = 2.0 * math.pi / 60.0


@dataclass(frozen=True)
class Motor:
    effort: float  # N m, continuous = peak in Booster's model
    no_load_speed: float  # rad/s
    knee_speed: float  # rad/s, full torque below this speed
    armature: float  # kg m^2 reflected rotor inertia

    def parallel(self, effort_ratio: float, speed_ratio: float, armature_ratio: float) -> Motor:
        """A linkage-driven joint: the motor's numbers scaled by the linkage ratios (booster_train wrappers)."""
        return Motor(
            self.effort * effort_ratio,
            self.no_load_speed * speed_ratio,
            self.knee_speed * speed_ratio,
            self.armature * armature_ratio,
        )


# Booster motors (booster_train assets/robots/actuator.py).
B7629x22 = Motor(135.0, 200.0 * RPM, 59.0 * RPM, 0.0281)
B6030x24 = Motor(74.0, 224.0 * RPM, 110.0 * RPM, 0.017)
B6025x24 = Motor(60.0, 227.0 * RPM, 123.0 * RPM, 0.0137)
B4522x35 = Motor(22.0, 177.0 * RPM, 122.0 * RPM, 0.007209)
# Parallel linkages: ankle (pitch, roll) and waist (pitch, roll) ratios, effort / speed / armature.
ANKLE_PITCH = B6025x24.parallel(2.2, 0.7, 2.620008)
ANKLE_ROLL = B6025x24.parallel(1.2, 1.0, 0.9734019)
WAIST_PITCH = B6025x24.parallel(2.3, 0.5, 2.884254)
WAIST_ROLL = B6025x24.parallel(2.4, 0.4, 3.825669)


def booster_actuator(
    names: tuple[str, ...], kp: float, kd: float, motor: Motor, delay_steps: int = DELAY_MAX_STEPS
) -> DcMotorActuatorCfg:
    v_max, v_knee = motor.no_load_speed, motor.knee_speed
    return DcMotorActuatorCfg(
        target_names_expr=names,
        stiffness=kp,
        damping=kd,
        effort_limit=motor.effort,
        saturation_effort=motor.effort * v_max / (v_max - v_knee),
        velocity_limit=v_max,
        armature=motor.armature,
        delay_min_lag=0,
        delay_max_lag=delay_steps,
        delay_hold_prob=1.0,  # one lag per episode: sampled at reset, held until the next reset
    )


def t2_actuators(delay_steps: int = DELAY_MAX_STEPS) -> tuple[DcMotorActuatorCfg, ...]:
    """Gains: Booster_T2 config/t2_robot_config.py (= booster_deploy t2_walk), wrists 18.21 / 1.449."""
    legs = (
        ".*_hip_yaw_joint",
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*_knee_pitch_joint",
        "waist_yaw_joint",
    )
    arms = (".*_shoulder_roll_joint", ".*_elbow_pitch_joint", ".*_elbow_yaw_joint")
    wrists = (".*_wrist_pitch_joint", ".*_wrist_yaw_joint", ".*_wrist_roll_joint")
    return (
        booster_actuator(legs, 110.9343, 3.5311, B7629x22, delay_steps),
        booster_actuator(("waist_roll_joint",), 206.9129, 1.7215, WAIST_ROLL, delay_steps),
        booster_actuator(("waist_pitch_joint",), 155.9961, 1.7215, WAIST_PITCH, delay_steps),
        booster_actuator((".*_ankle_pitch_joint",), 141.7042, 1.7215, ANKLE_PITCH, delay_steps),
        booster_actuator((".*_ankle_roll_joint",), 52.6468, 1.7215, ANKLE_ROLL, delay_steps),
        booster_actuator((".*_shoulder_pitch_joint",), 67.1133, 2.1362, B6030x24, delay_steps),
        booster_actuator(arms, 54.0854, 1.7215, B6025x24, delay_steps),
        booster_actuator(wrists, 18.214394, 1.449455, B4522x35, delay_steps),
    )


T2_ACTUATORS = t2_actuators()

# Booster's standing pose. Trunk height 0.994 m puts the soles 2 cm above the ground in this pose
# (pelvis at 0.828 m, the same stance Teleopit's pelvis-rooted T2 stands in at 0.83 m).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.994),
    joint_pos={
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_pitch_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_shoulder_pitch_joint": 0.35,
        # zero shoulder roll is a T-pose on the T2; -+1.37 brings the arms to the sides
        "left_shoulder_roll_joint": -1.3665,
        "right_shoulder_roll_joint": 1.3665,
        "left_elbow_yaw_joint": -0.7008,
        "right_elbow_yaw_joint": 0.7008,
    },
    joint_vel={".*": 0.0},
)

FOOT_GEOM_REGEX = r"^(left|right)_ankle_collision$"

# Every ``*_collision`` geom of the MJCF; the feet are the two ankle boxes (same as Teleopit's T2).
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    condim={FOOT_GEOM_REGEX: 3, ".*_collision": 1},
    priority={FOOT_GEOM_REGEX: 1, ".*": 0},
    friction={FOOT_GEOM_REGEX: (0.6,)},
)


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(T2_XML))


def get_sharpa_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(T2_SHARPA_XML))


def get_t2_robot_cfg(delay_steps: int = DELAY_MAX_STEPS, sharpa: bool = False) -> EntityCfg:
    articulation = EntityArticulationInfoCfg(
        actuators=t2_actuators(delay_steps), soft_joint_pos_limit_factor=0.9
    )
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_sharpa_spec if sharpa else get_spec,
        articulation=articulation,
    )


T2_ACTION_SCALE: dict[str, float] = {}
for _act in T2_ACTUATORS:
    for _name in _act.target_names_expr:
        T2_ACTION_SCALE[_name] = 0.25 * _act.effort_limit / _act.stiffness

# Model joint order (arms, waist, legs), which is also the policy and npz order.
_ARM = (
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_pitch",
    "elbow_yaw",
    "wrist_pitch",
    "wrist_yaw",
    "wrist_roll",
)
_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
T2_JOINT_NAMES: tuple[str, ...] = (
    *(f"left_{j}_joint" for j in _ARM),
    *(f"right_{j}_joint" for j in _ARM),
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    *(f"left_{j}_joint" for j in _LEG),
    *(f"right_{j}_joint" for j in _LEG),
)

T2 = RobotSpec(
    name="t2",
    xml_path=T2_XML,
    entity_cfg=get_t2_robot_cfg,
    action_scale=T2_ACTION_SCALE,
    joint_names=T2_JOINT_NAMES,
    root_body="trunk",
    anchor_body="trunk",
    # The G1 list body by body: pelvis -> waist_yaw_link, knee_link -> knee_pitch_link, torso_link -> trunk,
    # elbow_link (forearm) -> elbow_yaw_link, wrist_yaw_link (hand) -> wrist_roll_link.
    tracked_bodies=(
        "waist_yaw_link",
        "left_hip_roll_link",
        "left_knee_pitch_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_pitch_link",
        "right_ankle_roll_link",
        "trunk",
        "left_shoulder_roll_link",
        "left_elbow_yaw_link",
        "left_wrist_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_yaw_link",
        "right_wrist_roll_link",
    ),
    foot_geom_regex=FOOT_GEOM_REGEX,
    com_body="trunk",
)

# Diagnostic variant: the same robot without the command delay (everything else equal), to price the delay model.
T2_NODELAY = replace(T2, name="t2_nodelay", entity_cfg=lambda: get_t2_robot_cfg(delay_steps=0))

# Sharpa Wave hands: the wrist links carry the hand + flange (TWIST2 v3's URDF); train with the wrist payload event.
T2_SHARPA = replace(
    T2, name="t2_sharpa", xml_path=T2_SHARPA_XML, entity_cfg=lambda: get_t2_robot_cfg(sharpa=True)
)
