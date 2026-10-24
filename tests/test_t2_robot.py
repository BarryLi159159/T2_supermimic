"""Booster T2 robot layer: Booster's deployment convention (v3) as the task sees it."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.robots import get_robot
from supermimic.robots.t2 import DELAY_MAX_STEPS, HOME_KEYFRAME, T2_ACTUATORS

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/t2_lafan1"


def test_action_scale_is_quarter_effort_over_kp():
    scale = get_robot("t2").action_scale
    # Booster_T2 docs/twist2_v3_config_brief.md table, rad per unit action
    expected = {
        ".*_hip_pitch_joint": 0.3042,
        "waist_roll_joint": 0.1740,
        "waist_pitch_joint": 0.2212,
        ".*_ankle_pitch_joint": 0.2329,
        ".*_ankle_roll_joint": 0.3419,
        ".*_shoulder_pitch_joint": 0.2757,
        ".*_elbow_yaw_joint": 0.2773,
        ".*_wrist_roll_joint": 0.3020,
    }
    for name, value in expected.items():
        assert abs(scale[name] - value) < 5e-5, (name, scale[name])


def test_torque_speed_curve_matches_booster():
    """mjlab's line through (0, saturation) and (v_max, 0), capped at effort, equals Booster's knee curve."""
    knee = next(a for a in T2_ACTUATORS if ".*_knee_pitch_joint" in a.target_names_expr)
    v_knee = 59.0 * 2 * np.pi / 60
    v_max = 200.0 * 2 * np.pi / 60
    assert (
        abs(knee.velocity_limit - v_max) < 1e-9
        and knee.effort_limit == 135.0
        and knee.armature == 0.0281
    )
    for v in np.linspace(0.0, v_max, 41):
        mjlab_curve = min(knee.effort_limit, knee.saturation_effort * (1 - v / v_max))
        booster_curve = (
            knee.effort_limit if v <= v_knee else knee.effort_limit * (v_max - v) / (v_max - v_knee)
        )
        assert abs(mjlab_curve - booster_curve) < 1e-9, v
    assert (
        knee.delay_min_lag == 0
        and knee.delay_max_lag == DELAY_MAX_STEPS == 8
        and knee.delay_hold_prob == 1.0
    )
    waist_roll = next(a for a in T2_ACTUATORS if "waist_roll_joint" in a.target_names_expr)
    assert (
        abs(waist_roll.effort_limit - 144.0) < 1e-9 and abs(waist_roll.armature - 0.052412) < 1e-6
    )


def test_home_pose_soles_two_centimetres_above_ground():
    spec = get_robot("t2")
    model = spec.load_model()
    data = mujoco.MjData(model)
    data.qpos[:3] = HOME_KEYFRAME.pos
    data.qpos[3:7] = HOME_KEYFRAME.rot
    import re

    for j in range(1, model.njnt):
        name = model.joint(j).name
        for pattern, value in HOME_KEYFRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_kinematics(model, data)
    lowest = np.inf
    for g in range(model.ngeom):
        if re.fullmatch(spec.foot_geom_regex, model.geom(g).name or ""):
            rot = data.geom_xmat[g].reshape(3, 3)
            half = model.geom_size[g]
            corners = np.array(
                [
                    [sx, sy, sz]
                    for sx in (-half[0], half[0])
                    for sy in (-half[1], half[1])
                    for sz in (-half[2], half[2])
                ]
            )
            lowest = min(lowest, (data.geom_xpos[g] + corners @ rot.T)[:, 2].min())
    assert abs(lowest - 0.02) < 1e-3
    assert model.body(model.jnt_bodyid[0]).name == spec.root_body == spec.anchor_body == "trunk"
    assert model.site("imu").bodyid == model.body("trunk").id and np.allclose(
        model.site_pos[model.site("imu").id], 0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not NPZ.exists(), reason="T2 LAFAN1 npz not built")
def test_env_builds_with_paper_dimensions():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = [str(f) for f in discover_npz(NPZ)[:2]]
    env = make_env(
        SuperMimicEnvCfgArgs(npz_files=files, robot="t2", num_envs=4, replay_capacity=16, seed=1),
        device="cuda:0",
    )
    try:
        robot = env.scene["robot"]
        spec = get_robot("t2")
        assert tuple(robot.joint_names) == spec.joint_names
        assert robot.body_names[0] == "trunk"
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, 930) and obs["goal"].shape == (4, 680)
        action_term = env.action_manager.get_term("joint_pos")
        hip = robot.joint_names.index("left_hip_pitch_joint")
        assert abs(float(action_term._scale[0, hip]) - 0.3042) < 5e-5
        for _ in range(5):
            obs, rew, term, _trunc, _ = env.step(torch.zeros(4, 29, device=env.device))
        assert torch.isfinite(rew).all() and not term.any()
    finally:
        env.close()
