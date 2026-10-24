"""T2 SuperMimic task configurations: one task per motion npz in robots/t2/motions/ (t2_supermimic_<name>).

Gains, default pose and joint order are in Booster's real-robot (serial) order. The PD gains are the v3 deploy contract
(Booster's t2_walk table, wrists 18.21 / 1.449, head 20 / 30 and 0.5 / 1.0), exactly the gains the policy was trained
with in SuperMimic-Reproduced/src/supermimic/robots/t2.py.
"""
import glob
import os

from booster_deploy.controllers.controller_cfg import BoosterRobotControllerCfg, ControllerCfg, MujocoControllerCfg, PrepareStateCfg
from booster_deploy.robots.t2 import T2_31DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...tracker import SuperMimicPolicyCfg

CHECKPOINT = "robots/t2/models/t2_supermimic.onnx"

# v3 default pose (SuperMimic T2 HOME_KEYFRAME), head at zero
DEFAULT_JOINT_POS = [
    0.0, 0.0,  # head yaw, pitch
    0.35, -1.3665, 0.0, -0.7008, 0.0, 0.0, 0.0,  # left arm: shoulder pitch, roll, elbow pitch, yaw, wrist pitch, yaw, roll
    0.35, 1.3665, 0.0, 0.7008, 0.0, 0.0, 0.0,  # right arm
    0.0, 0.0, 0.0,  # waist pitch, roll, yaw
    -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,  # left leg: hip pitch, roll, yaw, knee, ankle pitch, roll
    -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,  # right leg
]
JOINT_STIFFNESS = [
    20.0, 30.0,
    67.1133, 54.0854, 54.0854, 54.0854, 18.214394, 18.214394, 18.214394,
    67.1133, 54.0854, 54.0854, 54.0854, 18.214394, 18.214394, 18.214394,
    155.9961, 206.9129, 110.9343,
    110.9343, 110.9343, 110.9343, 110.9343, 141.7042, 52.6468,
    110.9343, 110.9343, 110.9343, 110.9343, 141.7042, 52.6468,
]
JOINT_DAMPING = [
    0.5, 1.0,
    2.1362, 1.7215, 1.7215, 1.7215, 1.449455, 1.449455, 1.449455,
    2.1362, 1.7215, 1.7215, 1.7215, 1.449455, 1.449455, 1.449455,
    1.7215, 1.7215, 3.5311,
    3.5311, 3.5311, 3.5311, 3.5311, 1.7215, 1.7215,
    3.5311, 3.5311, 3.5311, 3.5311, 1.7215, 1.7215,
]

T2_SUPERMIMIC_CFG = T2_31DOF_CFG.replace(
    prepare_mode="standing",  # X: hold, switch to Custom, blend 1 s into the default pose; A: start the clip
    default_joint_pos=DEFAULT_JOINT_POS,
    joint_stiffness=JOINT_STIFFNESS,
    joint_damping=JOINT_DAMPING,
    prepare_state=PrepareStateCfg(
        stiffness=T2_31DOF_CFG.prepare_state.stiffness,
        damping=T2_31DOF_CFG.prepare_state.damping,
        joint_pos=DEFAULT_JOINT_POS,
    ),
)


@configclass
class T2SuperMimicControllerCfg(ControllerCfg):
    policy_dt: float = 0.02
    robot = T2_SUPERMIMIC_CFG
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg(exit_mode="damping")
    policy: SuperMimicPolicyCfg = SuperMimicPolicyCfg(checkpoint_path=CHECKPOINT, motion_path="")
    mujoco: MujocoControllerCfg = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.98],  # trunk height: soles touch the ground at 0.974 in the default pose
        decimation=20,  # 1 kHz physics, the official xml timestep
        visualize_reference_ghost=True,
    )


MOTION_DIR = os.path.join(os.path.dirname(__file__), "motions")
for path in sorted(glob.glob(os.path.join(MOTION_DIR, "*.npz"))):
    name = os.path.splitext(os.path.basename(path))[0]
    cfg = T2SuperMimicControllerCfg()
    cfg.policy.motion_path = os.path.join("robots", "t2", "motions", os.path.basename(path))
    register_task(f"t2_supermimic_{name}", cfg)
