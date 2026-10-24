"""Roll a SuperMimic T2 policy through booster_deploy's MuJoCo controller, headless (teleopit env; booster_deploy and
booster_assets/src are put on sys.path). Starts the robot on the clip's first frame like the training reset, then
reports the IMU route (task state vs MuJoCo trunk body), the joint tracking error against the reference and the
trunk height at the end.

  conda activate teleopit && python scripts/t2/check_booster_deploy.py --task t2_supermimic_walk1_subject1 --steps 400
"""
from __future__ import annotations

import argparse
import os
import pkgutil
import sys

import mujoco
import numpy as np


def load_tasks(deploy_dir: str) -> None:
    """booster_deploy's own registry walk (scripts/deploy.py): import every tasks.* module so it registers itself."""
    sys.path.insert(0, deploy_dir)
    sys.path.insert(0, os.path.expanduser("~/Booster_T2/asset/booster_assets/src"))
    import tasks as tasks_pkg

    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--deploy_dir", default=os.path.expanduser("~/booster_deploy"))
    a = p.parse_args()
    load_tasks(a.deploy_dir)
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from booster_deploy.utils.registry import get_task

    cfg = get_task(a.task)
    cfg.policy.stop_at_motion_end = False
    cfg.policy.enable_safety_fallback = False
    ctrl = MujocoController(cfg)
    pol = ctrl.policy
    clip = pol.clip
    m, d = ctrl.mj_model, ctrl.mj_data
    qadr = [m.joint(n).qposadr[0] for n in pol.policy_joints]
    vadr = [m.joint(n).dofadr[0] for n in pol.policy_joints]
    trunk = m.body("trunk").id
    assert m.body(m.jnt_bodyid[0]).name == "trunk", "booster_deploy's T2 xml must be trunk-rooted"

    mujoco.mj_resetData(m, d)
    d.qpos[qadr] = clip.joint_pos[0]
    d.qvel[vadr] = clip.joint_vel[0]
    d.qpos[:3], d.qpos[3:7] = clip.root_pos_w[0], clip.root_quat_w[0]
    mujoco.mj_forward(m, d)
    ctrl.update_state()
    ctrl.start()

    steps = min(a.steps, clip.num_frames)
    dq_max = dw_max = 0.0
    dof, z = [], []
    for _ in range(steps):
        ctrl.update_state()
        quat = ctrl.robot.data.root_quat_w.cpu().numpy()
        ang = ctrl.robot.data.root_ang_vel_b.cpu().numpy()
        mujoco.mj_forward(m, d)  # xquat lags mj_step by one substep otherwise
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, trunk, vel, 1)
        dq_max = max(dq_max, min(np.abs(quat - d.xquat[trunk]).max(), np.abs(quat + d.xquat[trunk]).max()))
        dw_max = max(dw_max, np.abs(ang - vel[:3]).max())
        targets = ctrl.policy_step()
        ctrl.ctrl_step(targets)
        dof.append(d.qpos[qadr].copy())
        z.append(d.qpos[2])
    dof = np.array(dof)
    err = np.abs(dof - clip.joint_pos[:steps]).mean(axis=1)
    print(f"IMU route: task root state vs MuJoCo trunk over {steps} steps: max |dquat| {dq_max:.1e}, max |dang_vel| {dw_max:.1e} rad/s")
    print(f"tracking: mean |dq| {err.mean():.3f} rad (first 50 steps {err[:50].mean():.3f}, last 50 {err[-50:].mean():.3f}); "
          f"trunk z start {z[0]:.2f} end {z[-1]:.2f} min {min(z):.2f} m")


if __name__ == "__main__":
    main()
