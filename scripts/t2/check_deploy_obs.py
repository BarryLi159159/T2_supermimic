"""Cross-check the booster_deploy observation layout (deploy/booster_deploy_supermimic/obs_layout.py) against the
mjlab T2 environment: same state in, same 930-d policy input and 680-d goal out, over a short rollout.

  uv run python scripts/t2/check_deploy_obs.py --clip data/npz/t2_lafan1/walk1_subject1.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.append(str(Path(__file__).resolve().parents[2] / "deploy" / "booster_deploy_supermimic"))
from obs_layout import History, ReferenceClip, proprio_frame

from supermimic.envs import SuperMimicEnvCfgArgs, make_env


def main(clip: Path, steps: int = 30, device: str = "cuda:0", seed: int = 0) -> None:
    args = SuperMimicEnvCfgArgs(npz_files=[str(clip)], robot="t2", num_envs=1, action_noise_std=0.0, obs_noise=False, reset_noise=False,
                                domain_randomization=False, sampling_mode="start", record_transitions=False, seed=seed)
    env = make_env(args, device=device)
    robot = env.scene["robot"]
    default = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float64)
    ref = ReferenceClip(str(clip), "trunk", default)
    assert ref.joint_names == list(robot.joint_names)
    history = History(10)
    rng = np.random.default_rng(seed)

    def deploy_obs(last_action, frame):
        d = robot.data
        q = d.joint_pos[0].cpu().numpy().astype(np.float64)
        qd = d.joint_vel[0].cpu().numpy().astype(np.float64)
        quat = d.root_link_quat_w[0].cpu().numpy().astype(np.float64)
        ang = d.root_link_ang_vel_b[0].cpu().numpy().astype(np.float64)
        return history.push(proprio_frame(q, qd, ang, quat, last_action, default)), ref.goal_window(frame, 10)

    obs, _ = env.reset()
    cmd = env.command_manager.get_term("motion")
    frame0 = int(cmd.frame[0].item()) - int(cmd.library.clip_start[0].item())
    last_action = np.zeros(29)
    worst_obs = worst_goal = 0.0
    for t in range(steps):
        o, g = deploy_obs(last_action, frame0 + t)
        worst_obs = max(worst_obs, float(np.abs(o - obs["policy"][0].cpu().numpy()).max()))
        worst_goal = max(worst_goal, float(np.abs(g - obs["goal"][0].cpu().numpy()).max()))
        action = rng.normal(size=29) * 0.5
        obs, *_ = env.step(torch.as_tensor(action, dtype=torch.float32, device=device)[None])
        last_action = action
    env.close()
    print(f"{clip.name}: start frame {frame0}, {steps} steps | policy obs max |diff| {worst_obs:.2e} | goal max |diff| {worst_goal:.2e}")
    assert worst_obs < 1e-4 and worst_goal < 1e-4, "deploy observation layout differs from the training environment"
    print("LAYOUT OK")


if __name__ == "__main__":
    tyro.cli(main)
