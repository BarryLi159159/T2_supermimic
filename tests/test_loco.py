"""Locomotion task: the analytic heads must agree with mjlab's own link data, and the loop's pieces must run."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.fixture(scope="module")
def env():
    from supermimic.envs.loco_env_cfg import LocoEnvCfgArgs, make_loco_env

    e = make_loco_env(LocoEnvCfgArgs(num_envs=16, replay_capacity=128, obs_noise=False, seed=3), device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    e.reset()
    for _ in range(30):
        e.step(0.3 * torch.randn(16, 15, device=e.device))
    yield e
    e.close()


def test_spaces(env):
    obs = env.observation_manager.compute()
    assert obs["policy"].shape == (16, 580)
    assert obs["goal"].shape == (16, 7)
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
    cmd = env.command_manager.get_term("velocity").command
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
    assert start.goal.shape == (4, 7)
    obs = env.loco_heads.observation(start.state, start.prev_action[:, env.loco_heads.policy_ids], start.goal, start.xi)
    assert obs.shape == (4, 58)
