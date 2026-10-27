"""The three supermimic_vibed additions to the tracking line: orientation-only anchor goal, soft termination,
world-model-only warm-up gate."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built"),
]


def test_soft_termination_logits_match_hard_thresholds():
    from supermimic.world_model.heads import soft_termination_logits

    feats = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 1.5], [-0.5, 0.0], [0.25, 0.8]])
    d = torch.sigmoid(soft_termination_logits(feats, 0.25, 0.8))
    np.testing.assert_allclose(d[:4].numpy(), [0.0, 1.0, 1.0, 1.0], atol=1e-4)
    assert 0.6 < d[4] < 0.8  # exactly on both thresholds: 1 - 0.5 * 0.5
    feats.requires_grad_(True)
    soft_termination_logits(feats, 0.25, 0.8).sum().backward()
    assert torch.isfinite(feats.grad).all()


def test_anchor_ori_only_goal_dim_env_and_imagination():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.imagination.compiled import ImagStep
    from supermimic.imagination.rollout import imagine
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.robots import get_robot
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, anchor_obs_dim
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    files = discover_npz(NPZ, names=["walk1_subject1"])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=8, replay_capacity=64, anchor_obs=True, anchor_ori_only=True, seed=2), device="cuda:0")
    try:
        obs, _ = env.reset()
        assert anchor_obs_dim(True, True) == 6
        assert obs["goal"].shape == (8, 680 + 6)
        for _ in range(30):
            env.step(0.3 * torch.randn(8, 29, device=env.device))
        buf = env.recorder_manager.get_term("replay").buffer
        lib = env.command_manager.get_term("motion").library
        spec = get_robot("g1")
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(spec.load_model(), device=env.device), spec.tracked_bodies, spec.anchor_body, spec.root_body, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(WorldModelCfg(hidden=(64, 64)), heads, env.device)
        policy = Policy(PolicyCfg(goal_dim=686)).to(env.device)
        step = ImagStep(wm, lib.default_joint_pos, soft_termination=True)
        ro = imagine(wm, policy, lib, buf.sample_starts(4, 10), horizon=3, step_fn=step, anchor_obs=True, anchor_ori_only=True)
        assert ro.goals.shape == (4, 4, 686)
        assert torch.isfinite(ro.dhat).all() and torch.isfinite(ro.rewards).all()
    finally:
        env.close()
