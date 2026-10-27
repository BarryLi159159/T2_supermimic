"""Headless video of a locomotion checkpoint on the walking-protocol scenarios (one panel per scenario).

Runs eval_loco's deterministic env (no noise / DR) for the chosen scenarios, records the state per step and replays
it in a CPU MuJoCo model of the robot with an offscreen renderer (MUJOCO_GL=egl), piping frames to ffmpeg.

  MUJOCO_GL=egl uv run python scripts/t2/render_loco.py --ckpt logs/<run>/ckpt_latest.pt --scenarios "walk 0.5" "kick 1.2 side" --out /tmp/loco.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import sys

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_loco import DURATION_S, KICK_S, SCENARIOS, load_policy  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["walk 0.5", "walk 0.3 turn 0.8", "kick 1.2 side"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--size", type=int, nargs=2, default=(480, 480))
    args = ap.parse_args()

    from supermimic.envs.loco_env_cfg import LocoEnvCfgArgs, make_loco_env
    from supermimic.envs.state_io import StateIO
    from supermimic.robots import get_robot

    policy, cfg, iteration = load_policy(args.ckpt, args.device)
    spec = get_robot(cfg["env"]["robot"])
    n = len(args.scenarios)
    env = make_loco_env(
        LocoEnvCfgArgs(robot=spec.name, num_envs=n, episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, record_transitions=False, seed=0),
        device=args.device,
    )
    robot = env.scene["robot"]
    io = StateIO(env)
    command = torch.tensor([SCENARIOS[s][0] for s in args.scenarios], device=env.device)
    kick = torch.tensor([SCENARIOS[s][1] for s in args.scenarios], device=env.device)
    obs, _ = env.reset()
    env.command_manager.get_term("velocity").set_fixed(torch.arange(n, device=env.device), command)
    obs = env.observation_manager.compute()
    steps = int(DURATION_S / env.step_dt)
    kick_step = int(KICK_S / env.step_dt)
    qpos = np.zeros((steps, n, 36), dtype=np.float64)
    fell = np.zeros((steps, n), dtype=bool)
    for step in range(steps):
        act = policy(obs["policy"], obs["goal"])
        if step == kick_step:
            state = torch.cat([robot.data.root_link_pose_w, robot.data.root_link_vel_w], dim=-1).clone()
            state[:, 8] += kick
            robot.write_root_state_to_sim(state)
        obs, _, term, _, _ = env.step(act)
        qpos[step] = io.read()[:, :36].cpu().numpy()
        fell[step] = term.cpu().numpy()
    env.close()

    m = mujoco.MjModel.from_xml_path(str(spec.xml_path))
    d = mujoco.MjData(m)
    w, h = args.size
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.6, 135.0, -15.0
    opt = mujoco.MjvOption()
    skip = max(1, round(1.0 / (env.step_dt * args.fps)))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{n * w}x{h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE,
    )
    for step in range(0, steps, skip):
        panels = []
        for k in range(n):
            d.qpos[:] = qpos[step, k]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.qpos[:3] + np.array([0.0, 0.0, -0.3])
            renderer.update_scene(d, camera=cam, scene_option=opt)
            panels.append(renderer.render())
        ff.stdin.write(np.concatenate(panels, axis=1).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"[render] {args.out}: {args.scenarios} from {args.ckpt} (iteration {iteration}); fell = {fell.any(0).tolist()}")


if __name__ == "__main__":
    main()
