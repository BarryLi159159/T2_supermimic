"""Locomotion task: the analytic heads must agree with mjlab's own link data, and the loop's pieces must run."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.fixture(scope="module")
def env():
    from supermimic.envs.loco_env_cfg import LocoEnvCfgArgs, make_loco_env

    e = make_loco_env(LocoEnvCfgArgs(num_envs=16, replay_capacity=128, obs_noise=False, seed=3, gait_clock=True), device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    e.reset()
    for _ in range(30):
        e.step(0.3 * torch.randn(16, 15, device=e.device))
    yield e
    e.close()


def test_spaces(env):
    obs = env.observation_manager.compute()
    assert obs["policy"].shape == (16, 580)
    assert obs["goal"].shape == (16, 9)  # command 7 + gait clock 2
    assert env.action_manager.total_action_dim == 15
    term = env.action_manager.get_term("joint_pos")
    assert term.clean_raw.shape == (16, 29) and term.executed_raw.shape == (16, 29)
    # The arms are held: their normalised targets stay zero, the policy joints carry the action.
    assert torch.all(term.clean_raw[:, :14] == 0)


def test_fk_bodies_match_mjlab(env):
    """FK from the 71-d state reproduces mjlab's link poses / velocities and the whole-body COM."""
    from supermimic.envs.state_io import StateIO

    heads = env.loco_heads
    robot = env.scene["robot"]
    s = StateIO(env).read()
    b = heads.bodies(s)
    origins = env.scene.env_origins
    names = robot.body_names
    pelvis = names.index("waist_yaw_link")
    feet = [names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    np.testing.assert_allclose((b.pelvis_pos_w + origins).cpu().numpy(), robot.data.body_link_pos_w[:, pelvis].cpu().numpy(), atol=1e-4)
    np.testing.assert_allclose(b.pelvis_lin_vel_w.cpu().numpy(), robot.data.body_link_lin_vel_w[:, pelvis].cpu().numpy(), atol=2e-3)
    np.testing.assert_allclose(b.pelvis_ang_vel_w.cpu().numpy(), robot.data.body_link_ang_vel_w[:, pelvis].cpu().numpy(), atol=2e-3)
    np.testing.assert_allclose((b.feet_pos_w + origins[:, None]).cpu().numpy(), robot.data.body_link_pos_w[:, feet].cpu().numpy(), atol=1e-4)
    np.testing.assert_allclose(b.height_z.cpu().numpy(), robot.data.body_link_pos_w[:, names.index("waist_pitch_link"), 2].cpu().numpy(), atol=1e-4)
    # Mass-weighted COM from mjlab's link poses and the nominal body COM offsets (the heads ignore the trunk's DR
    # COM offset, so the per-env randomised body_com_pos_w is not the reference here).
    from supermimic.kinematics.rotations import quat_apply

    body_ids = robot.indexing.body_ids.cpu().numpy()
    model = env.sim.mj_model
    mass = torch.as_tensor(np.array(model.body_mass)[body_ids], dtype=torch.float32, device=env.device)
    ipos = torch.as_tensor(np.array(model.body_ipos)[body_ids], dtype=torch.float32, device=env.device)
    com_b = robot.data.body_link_pos_w + quat_apply(robot.data.body_link_quat_w, ipos.expand(16, -1, -1))
    com = (mass[:, None] * com_b[:, :, :2]).sum(1) / mass.sum()
    np.testing.assert_allclose((b.com_xy_w + origins[:, :2]).cpu().numpy(), com.cpu().numpy(), atol=1e-4)


def test_reward_matches_env(env):
    """The reward the env paid equals the analytic table on the state it reached, dt-scaled."""
    from supermimic.envs.state_io import StateIO

    heads = env.loco_heads
    term = env.action_manager.get_term("joint_pos")
    cmd_term = env.command_manager.get_term("velocity")
    cmd = torch.cat([cmd_term.command, cmd_term.clock], dim=-1)
    io = StateIO(env)
    s_prev = io.read()
    _, rew, *_ = env.step(0.2 * torch.randn(16, 15, device=env.device))
    s = io.read()
    terminated = env.termination_manager.terminated.float()
    ours = heads.reward(s_prev, s, term.clean_raw, term.prev_clean_raw, cmd, terminated)["total"]
    fresh = env.episode_length_buf <= 1  # a reset env's previous state is unknown to the env-side term
    np.testing.assert_allclose(ours[~fresh].cpu().numpy(), rew[~fresh].cpu().numpy(), atol=1e-5)
    assert rew.max() <= heads.reward_cfg.max_step_reward + 1e-6


def test_imagination_step_and_update(env):
    """One world-model update and one policy update run on the live buffer and leave finite parameters."""
    from supermimic.config import TrainConfig
    from supermimic.imagination.loco_policy_trainer import LocoPolicyTrainer
    from supermimic.policy.loco_nets import LocoCritic, LocoPolicy
    from supermimic.world_model.loco_heads import LocoWorldModel
    from supermimic.world_model.wm_trainer import LocoWorldModelTrainer

    cfg = TrainConfig()
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.compile = None
    cfg.wm_trainer.batch_size = 64
    cfg.wm_trainer.batches_per_update = 1
    cfg.policy_trainer.batch_size = 32
    cfg.policy_trainer.horizon = 4
    cfg.world_model.hidden = (64, 64)
    cfg.loco.policy.goal_dim = cfg.loco.critic.goal_dim = 9
    wm = LocoWorldModel(cfg.world_model, env.loco_heads, env.device)
    policy = LocoPolicy(cfg.loco.policy).to(env.device)
    critic = LocoCritic(cfg.loco.critic).to(env.device)
    buf = env.recorder_manager.get_term("replay").buffer
    m_wm = LocoWorldModelTrainer(wm, cfg.wm_trainer).update(buf)
    assert np.isfinite(m_wm["loss"])
    m_pi = LocoPolicyTrainer(wm, policy, critic, cfg.policy_trainer).update(buf)
    assert np.isfinite(m_pi["imag_objective"]) and m_pi["policy_grad_norm"] > 0
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    # Imagined observations have the policy's layout and the command is the start row's.
    start = buf.sample_starts(4, 10)
    assert start.goal.shape == (4, 9)
    obs = env.loco_heads.observation(start.state, start.prev_action[:, env.loco_heads.policy_ids], start.goal, start.xi)
    assert obs.shape == (4, 58)


def test_gait_clock(env):
    """The clock advances one control step per env step in the replay rows and in imagination, and the swing
    reference lifts the left foot in the first half of the stride and the right foot in the second."""
    from supermimic.config import TrainConfig
    from supermimic.envs.mdp.loco_rewards import advance_clock, swing_reference
    from supermimic.imagination.loco_rollout import loco_imagine
    from supermimic.policy.loco_nets import LocoPolicy
    from supermimic.world_model.loco_heads import LocoWorldModel

    rcfg = env.loco_heads.reward_cfg
    clock0 = torch.tensor([[0.0, 1.0]], device=env.device)
    c = clock0
    for _ in range(round(rcfg.gait_period / rcfg.dt)):
        c = advance_clock(c, rcfg.dt, rcfg.gait_period)
    np.testing.assert_allclose(c.cpu().numpy(), clock0.cpu().numpy(), atol=1e-5)
    ref = swing_reference(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]), 0.08)
    np.testing.assert_allclose(ref.numpy(), [[0.08, 0.0], [0.0, 0.08], [0.0, 0.0]], atol=1e-7)

    buf = env.recorder_manager.get_term("replay").buffer
    w = buf.sample_windows(64, 4)
    for j in range(3):
        np.testing.assert_allclose(advance_clock(w.goal[j, :, 7:], rcfg.dt, rcfg.gait_period).cpu().numpy(), w.goal[j + 1, :, 7:].cpu().numpy(), atol=1e-5)

    # Episode starts: the first row holds the phase the policy saw (an auto-reset env does not advance that step).
    term = env.command_manager.get_term("velocity")
    ids = torch.tensor([0, 1], device=env.device)
    env.episode_length_buf[ids] = env.max_episode_length  # force a time-out reset on the next step
    env.step(torch.zeros(16, 15, device=env.device))
    assert torch.all(env.episode_length_buf[ids] == 0)
    rec = env.recorder_manager.get_term("replay")
    np.testing.assert_allclose(rec.prev_cmd[ids, 7:].cpu().numpy(), term.clock[ids].cpu().numpy(), atol=1e-6)
    obs = env.observation_manager.compute()
    np.testing.assert_allclose(obs["goal"][ids, 7:].cpu().numpy(), term.clock[ids].cpu().numpy(), atol=1e-6)
    phase_after_reset = term.phase[ids].clone()
    env.step(torch.zeros(16, 15, device=env.device))
    np.testing.assert_allclose(((term.phase[ids] - phase_after_reset) % 1.0).cpu().numpy(), rcfg.dt / rcfg.gait_period, atol=1e-5)

    cfg = TrainConfig()
    cfg.world_model.hidden = (64, 64)
    cfg.loco.policy.goal_dim = 9
    wm = LocoWorldModel(cfg.world_model, env.loco_heads, env.device)
    ro = loco_imagine(wm, LocoPolicy(cfg.loco.policy).to(env.device), buf.sample_starts(8, 10), horizon=3, obs_noise=False)
    assert ro.cmd.shape == (4, 8, 9)
    for j in range(3):
        np.testing.assert_allclose(advance_clock(ro.cmd[j, :, 7:], rcfg.dt, rcfg.gait_period).detach().cpu().numpy(), ro.cmd[j + 1, :, 7:].detach().cpu().numpy(), atol=1e-5)
