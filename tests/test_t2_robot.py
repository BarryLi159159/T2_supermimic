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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not NPZ.exists(), reason="T2 LAFAN1 npz not built")
def test_sharpa_wrist_payload_matches_parallel_axis():
    """The per-episode hand payload must land in xi and compose mass / COM / inertia exactly like a point mass."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.sim_context import XI_DIM

    files = [str(f) for f in discover_npz(NPZ)[:2]]
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            robot="t2_sharpa",
            wrist_payload=True,
            num_envs=8,
            replay_capacity=8,
            seed=0,
        ),
        device="cuda:0",
    )
    try:
        robot, model, mj = env.scene["robot"], env.sim.model, env.sim.mj_model
        env.reset()
        xi = env.recorder_manager.get_term("replay").ctx.read()
        assert xi.shape == (8, XI_DIM) and XI_DIM == 126
        payload = xi[:, 34:38].cpu().numpy()
        assert (payload[:, [0, 2]] >= 0).all() and (payload[:, [0, 2]] <= 1.5).all()
        assert (payload[:, [1, 3]] >= 0.05).all() and (payload[:, [1, 3]] <= 0.20).all()
        assert len(np.unique(np.round(payload[:, 0], 4))) > 1  # drawn per env
        for k, (name, sign) in enumerate(
            (("left_wrist_roll_link", 1.0), ("right_wrist_roll_link", -1.0))
        ):
            b = int(robot.indexing.body_ids[robot.body_names.index(name)].item())
            assert abs(mj.body_mass[b] - 1.64) < 0.01  # the Sharpa model, not the bare wrist
            m0, c0 = float(mj.body_mass[b]), mj.body_ipos[b].copy()
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, mj.body_iquat[b])
            rot = rot.reshape(3, 3)
            i0 = rot @ np.diag(mj.body_inertia[b]) @ rot.T
            for e in range(8):
                m_add, dist = float(payload[e, 2 * k]), float(payload[e, 2 * k + 1])
                d = np.array([0.0, sign * dist, 0.0])
                m1 = m0 + m_add
                c1 = (m0 * c0 + m_add * d) / m1
                shift = lambda m, v: m * (v @ v * np.eye(3) - np.outer(v, v))
                i1 = i0 + shift(m0, c0 - c1) + shift(m_add, d - c1)
                q = model.body_iquat[e, b].cpu().numpy()
                r1 = np.zeros(9)
                mujoco.mju_quat2Mat(r1, q)
                r1 = r1.reshape(3, 3)
                i1_sim = r1 @ np.diag(model.body_inertia[e, b].cpu().numpy()) @ r1.T
                assert abs(float(model.body_mass[e, b]) - m1) < 1e-5
                np.testing.assert_allclose(model.body_ipos[e, b].cpu().numpy(), c1, atol=1e-6)
                np.testing.assert_allclose(i1_sim, i1, atol=1e-7)
        masses = model.body_mass[:, b].cpu().numpy().copy()
        env.reset()
        xi2 = env.recorder_manager.get_term("replay").ctx.read()[:, 34:38].cpu().numpy()
        assert np.allclose(
            model.body_mass[:, b].cpu().numpy(), m0 + xi2[:, 2], atol=1e-5
        )  # re-drawn from the nominal, not accumulated
        assert not np.allclose(masses, model.body_mass[:, b].cpu().numpy())
    finally:
        env.close()


def test_v4_actuator_is_the_measured_one():
    """v4: delay 0-20 ms, the four parallel axes damped by the sent Kd x armature ratio x 0.69, everything else as t2."""
    from supermimic.robots.t2 import DELAY_MAX_STEPS_V4, T2, T2_V4, t2_actuators

    assert DELAY_MAX_STEPS_V4 == 4
    acts = t2_actuators(DELAY_MAX_STEPS_V4, measured=True)
    kd = {e: a.damping for a in acts for e in a.target_names_expr}
    kp = {e: a.stiffness for a in acts for e in a.target_names_expr}
    ref = {e: a.damping for a in T2_ACTUATORS for e in a.target_names_expr}
    expected = {"waist_pitch_joint": 3.4260, "waist_roll_joint": 4.5443, ".*_ankle_pitch_joint": 3.1121, ".*_ankle_roll_joint": 1.1562}
    for e, v in expected.items():
        assert abs(kd[e] - v) < 1e-3, (e, kd[e])
        assert kd[e] != 1.7215  # changed from the sent value
    for e, value in ref.items():
        if e not in expected:
            assert kd[e] == value, e
    assert kp == {e: a.stiffness for a in T2_ACTUATORS for e in a.target_names_expr}
    assert all(a.delay_max_lag == 4 for a in acts)
    assert T2_V4.action_scale == T2.action_scale
    assert T2_V4.pd_gains().kd[T2_V4.joint_names.index("waist_pitch_joint")] == kd["waist_pitch_joint"]


def test_v4_model_has_the_measured_joint_friction():
    """The compiled v4 model carries the sysID friction table as frictionloss on every hinge, none on the free joint; t2 has none."""
    from supermimic.robots.t2 import JOINT_FRICTION_V4, joint_friction

    m_v4 = get_robot("t2_v4").entity_cfg().spec_fn().compile()
    m_t2 = get_robot("t2").entity_cfg().spec_fn().compile()
    assert m_t2.dof_frictionloss.sum() == 0.0
    assert m_v4.dof_frictionloss[:6].sum() == 0.0
    for name in get_robot("t2_v4").joint_names:
        got = m_v4.dof_frictionloss[m_v4.joint(name).dofadr[0]]
        assert abs(got - joint_friction(name)) < 1e-9, (name, got)
    assert joint_friction("left_knee_pitch_joint") == 0.72 and joint_friction("waist_roll_joint") == 3.3
    assert len(JOINT_FRICTION_V4) == 14
    sharpa = get_robot("t2_v4_sharpa").entity_cfg().spec_fn().compile()
    assert abs(sharpa.body("left_wrist_roll_link").mass[0] - 1.64194) < 1e-4 and sharpa.dof_frictionloss.sum() > 0


def test_ramped_noise_widens_with_progress():
    from supermimic.envs.mdp.ramp import RampedUniformNoiseCfg, RampState

    ramp = RampState(progress=0.0)
    cfg = RampedUniformNoiseCfg(n_min=-0.1, n_max=0.1, scale_to=2.0, ramp=ramp)
    x = torch.zeros(20000)
    e0 = (cfg.apply(x) - x).abs().max().item()
    ramp.progress = 1.0
    e1 = (cfg.apply(x) - x).abs().max().item()
    assert 0.095 < e0 <= 0.1 + 1e-6 and 0.19 < e1 <= 0.2 + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not NPZ.exists(), reason="T2 LAFAN1 npz not built")
def test_v4_randomisation_lands_in_xi():
    """dr_v4: joint friction x0.5-1.5 per joint, gains +-10 %, torso +-5 kg read back into xi; IMU offsets drawn."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.sim_context import XI_DIM

    files = [str(f) for f in discover_npz(NPZ)[:2]]
    env = make_env(
        SuperMimicEnvCfgArgs(npz_files=files, robot="t2_v4", dr_v4=True, num_envs=8, replay_capacity=8, seed=0),
        device="cuda:0",
    )
    try:
        env.dr_ramp.progress = 1.0
        env.reset()
        xi = env.recorder_manager.get_term("replay").ctx.read()
        assert xi.shape == (8, XI_DIM)
        fric, kp, kd, mass = xi[:, 38:67], xi[:, 67:96], xi[:, 96:125], xi[:, 125]
        assert (fric >= 0.5 - 1e-5).all() and (fric <= 1.5 + 1e-5).all() and fric.std(dim=1).min() > 0.05
        assert (kp >= 0.9 - 1e-5).all() and (kp <= 1.1 + 1e-5).all() and kp.std(dim=1).min() > 0.01
        assert (kd >= 0.9 - 1e-5).all() and (kd <= 1.1 + 1e-5).all()
        assert (mass.abs() <= 5.0 + 1e-4).all() and mass.std() > 0.5
        ground = xi[:, 0]
        assert (ground >= 0.3 - 1e-5).all() and (ground <= 1.6 + 1e-5).all()
        assert env.gyro_bias.abs().max() <= 0.02 + 1e-6 and env.imu_tilt.abs().max() <= 0.035 + 1e-6
        assert env.gyro_bias.abs().max() > 0 and env.imu_tilt.abs().max() > 0
        # the model's frictionloss itself moved (not only the read-back)
        fl = env.sim.model.dof_frictionloss
        knee = env.sim.mj_model.joint("robot/left_knee_pitch_joint").dofadr[0]
        assert fl.dim() == 2 and fl[:, knee].min() >= 0.36 - 1e-5 and fl[:, knee].max() <= 1.08 + 1e-5 and fl[:, knee].std() > 0.05
        # easy end of the ramp: gains and offsets at nominal
        env.dr_ramp.progress = 0.0
        env.reset()
        xi0 = env.recorder_manager.get_term("replay").ctx.read()
        assert torch.allclose(xi0[:, 67:125], torch.ones_like(xi0[:, 67:125]), atol=1e-6)
        assert env.gyro_bias.abs().max() == 0 and env.imu_tilt.abs().max() == 0
    finally:
        env.close()
