"""Render a SuperMimic T2 policy in booster_deploy's MuJoCo, headless: reference pose (left) next to the policy (right).

  conda activate teleopit && MUJOCO_GL=egl python scripts/t2/render_booster_deploy.py --task t2_supermimic_walk1_subject1 --seconds 10 --out /tmp/x.mp4
"""
from __future__ import annotations

import argparse
import os
import pkgutil
import subprocess
import sys

import mujoco
import numpy as np


def load_tasks(deploy_dir: str) -> None:
    sys.path.insert(0, deploy_dir)
    sys.path.insert(0, os.path.expanduser("~/Booster_T2/asset/booster_assets/src"))
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, nargs=2, default=(640, 480))
    p.add_argument("--deploy_dir", default=os.path.expanduser("~/booster_deploy"))
    a = p.parse_args()
    load_tasks(a.deploy_dir)
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    cfg = get_task(a.task)
    cfg.policy.stop_at_motion_end = False
    cfg.policy.enable_safety_fallback = False
    cfg.policy.start_frame = a.start_frame
    ctrl = MujocoController(cfg)
    pol, clip = ctrl.policy, ctrl.policy.clip
    m, d = ctrl.mj_model, ctrl.mj_data
    qadr = [m.joint(n).qposadr[0] for n in pol.policy_joints]
    vadr = [m.joint(n).dofadr[0] for n in pol.policy_joints]
    f0 = a.start_frame
    mujoco.mj_resetData(m, d)
    d.qpos[qadr], d.qvel[vadr] = clip.joint_pos[f0], clip.joint_vel[f0]
    d.qpos[:3], d.qpos[3:7] = clip.root_pos_w[f0], clip.root_quat_w[f0]
    mujoco.mj_forward(m, d)
    ctrl.update_state()
    ctrl.start()

    d_ref = mujoco.MjData(m)
    w, h = a.size
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # floor
    opt.geomgroup[1] = 1  # visual meshes (the official xml keeps its collision boxes in group 3)
    every = max(1, round(1.0 / (ctrl.cfg.policy_dt * a.fps)))
    steps = min(int(a.seconds / ctrl.cfg.policy_dt), clip.num_frames - f0)
    out_dir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}", "-r", str(a.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
    err, zmin = [], 1e9
    for t in range(steps):
        ctrl.update_state()
        targets = ctrl.policy_step()
        ctrl.ctrl_step(targets)
        k = min(f0 + t, clip.num_frames - 1)
        err.append(np.abs(d.qpos[qadr] - clip.joint_pos[k]).mean())
        zmin = min(zmin, d.qpos[2])
        if t % every:
            continue
        d_ref.qpos[:] = 0.0
        d_ref.qpos[qadr] = clip.joint_pos[k]
        d_ref.qpos[:3], d_ref.qpos[3:7] = clip.root_pos_w[k], clip.root_quat_w[k]
        mujoco.mj_forward(m, d_ref)
        frames = []
        for data in (d_ref, d):
            cam.lookat[:] = data.qpos[:3] * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, 0.8])
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(renderer.render())
        ff.stdin.write(np.concatenate(frames, axis=1).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{a.out}: {steps} steps ({steps * ctrl.cfg.policy_dt:.1f} s), mean |dq| {np.mean(err):.3f} rad, trunk z min {zmin:.2f} m, "
          f"end {d.qpos[2]:.2f} m, {'fell' if zmin < 0.5 else 'stayed up'}")


if __name__ == "__main__":
    main()
