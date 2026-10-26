"""Three-panel video in booster_deploy's MuJoCo: reference | TWIST2 tracker | SuperMimic, same clip, same simulator.

The TWIST2 motion (export_twist2_motion.py) starts with a 1 s lead-in from the default pose; the TWIST2 controller
runs through it unrecorded, then both robots start from the clip's first frame together.

  conda activate teleopit && MUJOCO_GL=egl python scripts/t2/render_compare_twist2.py --twist2 t2_twist2_walk --supermimic t2_supermimic_sm_walk --out walk.mp4
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


def quat_rotate_inverse(q, v):
    w, u = q[0], q[1:]
    return v * (2 * w * w - 1) - np.cross(u, v) * 2 * w + u * np.dot(u, v) * 2


SHARPA_XML = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "booster_deploy_supermimic", "robots", "t2", "T2_31dof_sharpa.xml")


def delayed_ctrl_step(c, targets, hist: list, delay_steps: int) -> None:
    """Booster's ctrl_step with a command delay: physics step m sees the newest target issued at or before m - delay.
    hist holds one target per policy step (numpy, all joints)."""
    import mujoco
    targets = targets.cpu().numpy()
    c.log_states(targets)
    hist.append(targets)
    n, D = len(hist) - 1, c.decimation
    dof_pos = c.mj_data.qpos.astype(np.float32)[7:]
    dof_vel = c.mj_data.qvel.astype(np.float32)[6:]
    kp, kd = c.robot.joint_stiffness.numpy(), c.robot.joint_damping.numpy()
    lim = c.robot.effort_limit.numpy()
    for i in range(D):
        j = max(0, n + (i - delay_steps) // D)  # floor division: the command that was already out at this physics step
        tgt = hist[j]
        c.mj_data.ctrl = np.clip(kp * (tgt - dof_pos) - kd * dof_vel, -lim, lim)
        mujoco.mj_step(c.mj_model, c.mj_data)
        dof_pos = c.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = c.mj_data.qvel.astype(np.float32)[6:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--twist2", required=True)
    p.add_argument("--supermimic", required=True)
    p.add_argument("--twist2_ckpt", default=None, help="override the TWIST2 task checkpoint (absolute .onnx path)")
    p.add_argument("--supermimic_ckpt", default=None, help="override the SuperMimic task checkpoint (absolute .onnx path)")
    p.add_argument("--delay_ms", type=float, default=0.0, help="command latency added in the MuJoCo PD loop (both robots); Booster's controller itself has none")
    p.add_argument("--sharpa", action="store_true", help="both robots carry the Sharpa wrist inertia (T2_31dof_sharpa.xml); use with the Sharpa-trained policies only")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, nargs=2, default=(480, 480))
    p.add_argument("--deploy_dir", default=os.path.expanduser("~/booster_deploy"))
    a = p.parse_args()
    load_tasks(a.deploy_dir)
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    # --- SuperMimic: trunk-rooted clip, start on its first frame
    cfg_s = get_task(a.supermimic)
    cfg_s.policy.stop_at_motion_end = False
    cfg_s.policy.enable_safety_fallback = False
    if a.supermimic_ckpt:
        cfg_s.policy.checkpoint_path = os.path.abspath(a.supermimic_ckpt)
    if a.sharpa:
        cfg_s.robot = cfg_s.robot.replace(mjcf_path=SHARPA_XML)
    cs = MujocoController(cfg_s)
    ps, clip = cs.policy, cs.policy.clip
    ms, ds = cs.mj_model, cs.mj_data
    wrist_mass = ms.body("left_wrist_roll_link").mass[0]
    if (ps.trained_robot == "t2_sharpa") != a.sharpa or (wrist_mass > 1.0) != a.sharpa:
        raise ValueError(f"hand mismatch: policy trained on {ps.trained_robot}, MuJoCo wrist link {wrist_mass:.3f} kg, --sharpa={a.sharpa}")
    qadr_s = [ms.joint(n).qposadr[0] for n in ps.policy_joints]
    vadr_s = [ms.joint(n).dofadr[0] for n in ps.policy_joints]

    # --- TWIST2: pelvis-rooted policy, motion with lead-in; start from its frame 0 (default pose) like the real deployment
    cfg_t = get_task(a.twist2)
    cfg_t.policy.stop_at_motion_end = False
    cfg_t.policy.enable_safety_fallback = False
    if a.twist2_ckpt:
        cfg_t.policy.checkpoint_path = os.path.abspath(a.twist2_ckpt)
    if a.sharpa:
        cfg_t.robot = cfg_t.robot.replace(mjcf_path=SHARPA_XML)
    if ("sharpa" in cfg_t.policy.checkpoint_path) != a.sharpa:  # TWIST2 ONNX has no contract: go by the run / file name
        raise ValueError(f"hand mismatch: TWIST2 checkpoint {cfg_t.policy.checkpoint_path} with --sharpa={a.sharpa}")
    ct = MujocoController(cfg_t)
    pt = ct.policy
    mt, dt = ct.mj_model, ct.mj_data
    motion = np.load(os.path.join(pt.task_path, cfg_t.policy.motion_path), allow_pickle=True)
    lead = int(motion["lead_in_frames"])
    tw_joints = [str(n) for n in motion["joint_names"]]
    qadr_t = [mt.joint(n).qposadr[0] for n in tw_joints]
    vadr_t = [mt.joint(n).dofadr[0] for n in tw_joints]
    assert abs(len(clip.joint_pos) + lead - len(motion["dof_pos"])) <= 5, "the two motions are not the same clip"
    # SuperMimic joint order -> TWIST2 order for the shared reference
    perm = [ps.policy_joints.index(n) for n in tw_joints]

    for m, d, qadr, vadr, pos, quat, q, qd in (
        (ms, ds, qadr_s, vadr_s, clip.root_pos_w[0], clip.root_quat_w[0], clip.joint_pos[0], clip.joint_vel[0]),
        (mt, dt, qadr_t, vadr_t, motion["trunk_pos"][0], motion["trunk_quat_wxyz"][0], motion["dof_pos"][0], np.zeros(len(tw_joints))),
    ):
        mujoco.mj_resetData(m, d)
        d.qpos[qadr], d.qvel[vadr] = q, qd
        d.qpos[:3], d.qpos[3:7] = pos, quat
        mujoco.mj_forward(m, d)
    for c in (cs, ct):
        c.update_state()
        c.start()
    delay_steps = int(round(a.delay_ms / 1000.0 / ms.opt.timestep))
    hist = {id(cs): [], id(ct): []}
    step = (lambda c: delayed_ctrl_step(c, c.policy_step(), hist[id(c)], delay_steps)) if delay_steps > 0 else (lambda c: c.ctrl_step(c.policy_step()))
    print(f"command delay {a.delay_ms:.0f} ms = {delay_steps} physics steps of {ms.opt.timestep * 1000:.0f} ms")
    for _ in range(lead):  # TWIST2 walks its lead-in from the default pose to the clip's first frame
        ct.update_state()
        step(ct)

    d_ref = mujoco.MjData(ms)
    w, h = a.size
    renderer = mujoco.Renderer(ms, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    every = max(1, round(1.0 / (cs.cfg.policy_dt * a.fps)))
    steps = min(int(a.seconds / cs.cfg.policy_dt), clip.num_frames, len(motion["dof_pos"]) - lead)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * w}x{h}", "-r", str(a.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
    err = {"twist2": [], "supermimic": []}
    zmin = {"twist2": 1e9, "supermimic": 1e9}
    for t in range(steps):
        for c in (cs, ct):
            c.update_state()
            step(c)
        k = min(t, clip.num_frames - 1)
        ref_s = clip.joint_pos[k]
        err["supermimic"].append(np.abs(ds.qpos[qadr_s] - ref_s).mean())
        err["twist2"].append(np.abs(dt.qpos[qadr_t] - ref_s[perm]).mean())
        zmin["supermimic"] = min(zmin["supermimic"], ds.qpos[2])
        zmin["twist2"] = min(zmin["twist2"], dt.qpos[2])
        if t % every:
            continue
        d_ref.qpos[:] = 0.0
        d_ref.qpos[qadr_s] = ref_s
        d_ref.qpos[:3], d_ref.qpos[3:7] = clip.root_pos_w[k], clip.root_quat_w[k]
        mujoco.mj_forward(ms, d_ref)
        frames = []
        for data in (d_ref, dt, ds):  # the two robot models are the same T2_31dof xml, so one renderer serves all three
            cam.lookat[:] = data.qpos[:3] * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, 0.8])
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(renderer.render())
        ff.stdin.write(np.concatenate(frames, axis=1).tobytes())
    ff.stdin.close()
    ff.wait()
    for k in ("twist2", "supermimic"):
        print(f"{os.path.basename(a.out)} {k:10s}: {steps * cs.cfg.policy_dt:.1f} s, mean |dq| {np.mean(err[k]):.3f} rad, trunk z min {zmin[k]:.2f} m, {'fell' if zmin[k] < 0.5 else 'stayed up'}")


if __name__ == "__main__":
    main()
