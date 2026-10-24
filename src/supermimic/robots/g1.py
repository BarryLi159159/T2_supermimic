"""Unitree G1 as in the paper: mjlab's BeyondMimic robot layer, pelvis root, torso anchor."""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg

from supermimic.data.motion_convert import G1_JOINT_NAMES, g1_xml_path
from supermimic.robots import RobotSpec

G1 = RobotSpec(
    name="g1",
    xml_path=g1_xml_path(),
    entity_cfg=get_g1_robot_cfg,
    action_scale=G1_ACTION_SCALE,
    joint_names=G1_JOINT_NAMES,
    root_body="pelvis",
    anchor_body="torso_link",
    tracked_bodies=(
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    com_body="torso_link",
)
