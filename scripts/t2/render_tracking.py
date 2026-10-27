"""Headless video of a tracking checkpoint: one column per clip, reference on top, policy below.

Deterministic evaluation (no noise / DR), every env starts its clip at the first frame; a fall resets the env to the
clip start (the run prints which clips fell). States are replayed in a CPU MuJoCo model with an offscreen renderer.

  MUJOCO_GL=egl uv run python scripts/t2/render_tracking.py --ckpt logs/<run>/ckpt_latest.pt --clips walk1_subject1 dance1_subject1 --out /tmp/track.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz-dir", default="data/npz/t2_lafan1")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--size", type=int, default=360)
    args = ap.parse_args()

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots import get_robot

    files = [str(f) for f in discover_npz(args.npz_dir, names=args.clips)]
    policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)
    spec = get_robot(cfg["env"].get("robot", "g1"))
    n = len(files)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            robot=spec.name,
            num_envs=n,
            future_frames=cfg["env"]["future_frames"],
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            anchor_obs=cfg["env"].get("anchor_obs", False),
            anchor_ori_only=cfg["env"].get("anchor_ori_only", False),
        ),
        device=args.device,
    )
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    names = [os.path.basename(f)[:-4] for f in files]
    cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=env.device)]
    obs, _ = env.reset()
    io = StateIO(env)
    steps = int(args.seconds / env.step_dt)
    q_pol = np.zeros((steps, n, 36))
    q_ref = np.zeros((steps, n, 36))
    fell = np.zeros(n, dtype=bool)
    origins = env.scene.env_origins
    for t in range(steps):
        obs, _, term, _, _ = env.step(policy(obs["policy"], obs["goal"]))
        fell |= term.cpu().numpy()
        s = io.read()[:, :36].clone()
        s[:, :3] -= origins
        q_pol[t] = s.cpu().numpy()
        r = lib.ref(cmd.frame)
        q_ref[t] = torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], dim=-1).cpu().numpy()
    env.close()

    m = mujoco.MjModel.from_xml_path(str(spec.xml_path))
    d = mujoco.MjData(m)
    w = h = args.size
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135.0, -12.0
    skip = max(1, round(1.0 / (env.step_dt * args.fps)))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{n * w}x{2 * h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE,
    )
    for t in range(0, steps, skip):
        rows = []
        for q in (q_ref, q_pol):
            panels = []
            for k in range(n):
                d.qpos[:] = q[t, k]
                mujoco.mj_forward(m, d)
                cam.lookat[:] = d.qpos[:3] + np.array([0.0, 0.0, -0.2])
                renderer.update_scene(d, camera=cam)
                panels.append(renderer.render())
            rows.append(np.concatenate(panels, axis=1))
        ff.stdin.write(np.concatenate(rows, axis=0).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"[render] {args.out}: top = reference, bottom = policy; clips {names}; fell {dict(zip(names, fell.tolist()))}")


if __name__ == "__main__":
    main()
