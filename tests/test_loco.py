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



def test_foot_timers_and_switches():
    """Isaac's contact-timer terms on carried foot timers (straight-through contact), the non-foot geoms' lowest points
    against MuJoCo, the sim / replay / imagination timers agreeing, imagined command switches and action noise."""
    import mujoco

    from supermimic.config import TrainConfig
    from supermimic.envs.loco_env_cfg import LocoEnvCfgArgs, make_loco_env
    from supermimic.envs.mdp.loco_rewards import LocoRewardCfg, advance_foot_timers, timer_contact
    from supermimic.envs.state_io import StateIO
    from supermimic.imagination.loco_policy_trainer import LocoPolicyTrainer
    from supermimic.imagination.loco_rollout import loco_imagine
    from supermimic.policy.loco_nets import LocoCritic, LocoPolicy
    from supermimic.world_model.loco_heads import LocoWorldModel

    rcfg = LocoRewardCfg(foot_timers=True, w_feet_air_time=0.15, w_undesired_contacts=-1.0)
    # A planted foot 1.2 cm under the ground (where the world model puts it) is hard contact with a lift gradient.
    sole = torch.tensor([[[0.0, 0.0, -0.012], [0.0, 0.0, 0.05]]], requires_grad=True)
    c = timer_contact(sole, 0.0, rcfg)
    np.testing.assert_allclose(c.detach().numpy(), [[1.0, 0.0]])
    c[..., 0].sum().backward()
    z = (rcfg.contact_height + 0.012) / rcfg.timer_grad_width
    sig = 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(sole.grad[0, 0, 2].item(), -sig * (1 - sig) / rcfg.timer_grad_width, rtol=1e-4)
    timers = advance_foot_timers(torch.tensor([[0.1, 0.3, 0.5, 0.0]]), torch.tensor([[1.0, 0.0]]), 0.02)
    np.testing.assert_allclose(timers.numpy(), [[0.0, 0.32, 0.52, 0.0]], atol=1e-7)

    env = make_loco_env(LocoEnvCfgArgs(num_envs=16, replay_capacity=128, obs_noise=False, seed=4, gait_clock=False, foot_timers=True), device="cuda:0", reward_cfg=rcfg)
    try:
        env.reset()
        heads = env.loco_heads
        io = StateIO(env)
        dt = rcfg.dt

        # Isaac's forms on a standing state: both feet held 0.7 / 0.8 s under a standing command; then the left knee
        # bent (left foot in the air 0.3 s, right held 0.6 s) under a walking command pays the shorter mode time.
        s = io.read()[:1].clone()
        s[0, 2] -= (heads.bodies(s).feet_pos_w[0, :, 2] - heads.spec.sole_offset).min()  # lower sole on the ground
        cmd_stand = torch.zeros(1, 7, device=env.device)
        cmd_stand[0, 3] = 0.85
        zero = torch.zeros(1, 29, device=env.device)
        held = torch.tensor([[0.0, 0.0, 0.7, 0.8]], device=env.device)
        terms = heads.reward(s, s, zero, zero, cmd_stand, torch.zeros(1, device=env.device), foot_timers=held)
        np.testing.assert_allclose(terms["feet_contact_standing_stable"].item(), dt * 0.5 * np.tanh(0.7 / 0.5), rtol=1e-5)
        assert terms["feet_air_time"].item() == 0.0
        knee = 7 + heads.spec.joint_names.index(next(n for n in heads.spec.joint_names if n.startswith("left_knee")))
        s_lift = s.clone()
        s_lift[0, knee] += 1.0
        assert timer_contact(heads.bodies(s_lift).feet_pos_w, heads.spec.sole_offset, rcfg).tolist() == [[0.0, 1.0]]
        cmd_walk = cmd_stand.clone()
        cmd_walk[0, 0] = 0.5
        swing = torch.tensor([[0.3, 0.0, 0.0, 0.6]], device=env.device)
        terms = heads.reward(s, s_lift, zero, zero, cmd_walk, torch.zeros(1, device=env.device), foot_timers=swing)
        np.testing.assert_allclose(terms["feet_air_time"].item(), dt * 0.15 * 0.3, rtol=1e-5)

        # Non-foot collision geoms: lowest bounding-box point from FK = MuJoCo's, none near the ground when standing,
        # and the trunk / knees count as touching when the robot lies 5 cm above the floor.
        m = heads.spec.load_model()
        d = mujoco.MjData(m)
        d.qpos[:] = s_lift[0, :36].cpu().numpy()
        mujoco.mj_kinematics(m, d)
        geoms = [g for g in range(m.ngeom) if m.geom(g).name.endswith("_collision") and "ankle" not in m.geom(g).name]
        low = []
        for g in geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            centre = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            low.append(centre[2] - np.abs(R[2]) @ m.geom_aabb[g, 3:])
        np.testing.assert_allclose(heads.bodies(s_lift).geom_low_z[0].cpu().numpy(), low, atol=1e-4)
        assert heads.bodies(s).geom_low_z.min() > 0.1
        lying = s.clone()
        lying[0, :3] = torch.tensor([0.0, 0.0, 0.05], device=env.device)
        lying[0, 3:7] = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=env.device)  # pitched 90 deg, face down
        terms = heads.reward(lying, lying, zero, zero, cmd_stand, torch.zeros(1, device=env.device), foot_timers=held)
        assert terms["undesired_contacts"].item() < -dt * 3.0

        # Sim: the reward term's timers follow the contact of each reached state, and the env pays the table on them.
        timer_term = env.reward_manager.get_term_cfg("loco").func
        lifted = 0.0
        for _ in range(40):
            env.step(0.5 * torch.randn(16, 15, device=env.device))
            lifted = max(lifted, float(timer_term.foot_timers[:, :2].max()))
        assert lifted > 0.0
        term = env.action_manager.get_term("joint_pos")
        cmd = env.command_manager.get_term("velocity").command.clone()
        s_prev = io.read()
        timers_prev = timer_term.foot_timers.clone()
        _, rew, *_ = env.step(0.5 * torch.randn(16, 15, device=env.device))
        s_now = io.read()
        timers_now = advance_foot_timers(timers_prev, timer_contact(heads.bodies(s_now).feet_pos_w, heads.spec.sole_offset, rcfg), dt)
        fresh = env.episode_length_buf <= 1
        np.testing.assert_allclose(timers_now[~fresh].cpu().numpy(), timer_term.foot_timers[~fresh].cpu().numpy(), atol=1e-6)
        ours = heads.reward(s_prev, s_now, term.clean_raw, term.prev_clean_raw, cmd, env.termination_manager.terminated.float(), foot_timers=timers_now)["total"]
        np.testing.assert_allclose(ours[~fresh].cpu().numpy(), rew[~fresh].cpu().numpy(), atol=1e-5)

        # Replay rows: goal = command 7 + timers 4, row j+1's timers follow from row j and state j+1.
        buf = env.recorder_manager.get_term("replay").buffer
        w = buf.sample_windows(64, 4)
        assert w.goal.shape[-1] == 11
        for j in range(3):
            feet = heads.bodies(w.states[j + 1]).feet_pos_w
            expect = advance_foot_timers(w.goal[j, :, 7:], timer_contact(feet, heads.spec.sole_offset, rcfg), dt)
            np.testing.assert_allclose(expect.cpu().numpy(), w.goal[j + 1, :, 7:].cpu().numpy(), atol=1e-6)

        # Imagination: the policy sees the command, the rollout goal carries the advancing timers; with switches every
        # rollout changes its command once, after the first step, to another row's command.
        cfg = TrainConfig()
        cfg.policy_trainer.compile = None
        cfg.policy_trainer.batch_size = 32
        cfg.policy_trainer.horizon = 6
        cfg.world_model.hidden = (64, 64)
        cfg.loco.policy.goal_dim = 7
        cfg.loco.critic.goal_dim = 11
        wm = LocoWorldModel(cfg.world_model, heads, env.device)
        policy = LocoPolicy(cfg.loco.policy).to(env.device)
        start = buf.sample_starts(16, 10)
        ro = loco_imagine(wm, policy, start, horizon=6, obs_noise=False, switch_prob=1.0)
        assert ro.cmd.shape == (7, 16, 11)
        for j in range(6):
            feet = heads.bodies(ro.states[j + 1]).feet_pos_w
            expect = advance_foot_timers(ro.cmd[j, :, 7:], timer_contact(feet, heads.spec.sole_offset, rcfg), dt)
            np.testing.assert_allclose(expect.detach().cpu().numpy(), ro.cmd[j + 1, :, 7:].detach().cpu().numpy(), atol=1e-6)
        used = ro.cmd[1:, :, :7]  # the command each imagined step acted on
        torch.testing.assert_close(used[0], start.goal[:, :7])
        changes = (used[1:] != used[:-1]).any(-1).sum(0)
        assert int(changes.max()) <= 1 and int(changes.sum()) > 8  # one switch at most, most rows got a different command
        # Executed-action noise: without it the rollout is deterministic, with it the dynamics differ and the gradient
        # still reaches the policy through the noise (the fresh dynamics' zero last layer would ignore the action).
        with torch.no_grad():
            torch.nn.init.normal_([m for m in wm.dynamics.modules() if isinstance(m, torch.nn.Linear)][-1].weight, std=1e-2)
        plain = [loco_imagine(wm, policy, start, horizon=6, obs_noise=False).states for _ in range(2)]
        torch.testing.assert_close(plain[0], plain[1])
        noisy = loco_imagine(wm, policy, start, horizon=6, obs_noise=False, action_noise=0.3)
        assert (noisy.states[1:] - plain[0][1:]).abs().max() > 1e-4
        grads = torch.autograd.grad(noisy.rewards.sum(), list(policy.parameters()))
        assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
        trainer = LocoPolicyTrainer(wm, policy, LocoCritic(cfg.loco.critic).to(env.device), cfg.policy_trainer, switch_prob=0.5)
        trainer.action_noise = 0.3
        m_pi = trainer.update(buf)
        assert np.isfinite(m_pi["imag_objective"]) and np.isfinite(m_pi["critic_loss"]) and m_pi["policy_grad_norm"] > 0
    finally:
        env.close()
