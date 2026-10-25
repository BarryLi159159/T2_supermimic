"""Booster T2 MJCF for SuperMimic: the official T2_31dof.xml with the trunk as the free body (= IMU body, the v3 deploy
convention), the two head joints welded at zero and all actuators removed (mjlab adds its own), so the policy sees 29 hinges like the G1.
Usage: python make_t2_mjcf.py --src <booster_assets>/robots/T2/T2_31dof/T2_31dof.xml --out assets/t2/t2_29dof_trunk.xml
"""
import argparse
import os

import mujoco
import numpy as np

HEAD_JOINTS = ("aa_head_yaw_joint", "head_pitch_joint")


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a rotation matrix (MuJoCo convention)."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, r.reshape(-1))
    return q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sharpa", action="store_true", help="wrist links carry a Sharpa Wave hand with its flange (TWIST2 booster_t2_29dof_sharpa.urdf inertia)")
    a = p.parse_args()
    spec = mujoco.MjSpec.from_file(a.src)
    def delete(elem):  # mujoco < 3.9 deletes through the spec, later versions through the element
        spec.delete(elem) if hasattr(spec, "delete") else elem.delete()

    # No actuators in the asset: mjlab adds its own (gains, limits, delay) from the robot config, named by joint.
    for act in list(spec.actuators):
        delete(act)
    for j in list(spec.joints):
        if j.name in HEAD_JOINTS:
            delete(j)
    if a.sharpa:
        # bare wrist + Sharpa Wave with flange, from TWIST2's booster_t2_29dof_sharpa.urdf (mass, com, full inertia in the link frame)
        for name, mass, com, full in (
            ("left_wrist_roll_link", 1.64194, (0.006973, 0.130999, 0.008526), (0.00882431, 0.00250928, 0.00903107, 2.77141e-05, 0.000300885, 0.00100726)),
            ("right_wrist_roll_link", 1.63994, (0.006767, -0.125802, 0.008540), (0.00783809, 0.00248532, 0.00802636, 0.000280105, 0.000292174, -0.000927109)),
        ):
            ixx, iyy, izz, ixy, ixz, iyz = full
            tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            principal, axes = np.linalg.eigh(tensor)  # MjSpec takes principal moments + the inertial frame rotation
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            body = spec.body(name)
            body.mass = mass
            body.ipos = com
            body.inertia = principal
            body.iquat = quat_from_matrix(axes)
            body.explicitinertial = True
    spec.modelname = "booster_t2_29dof_trunk" + ("_sharpa" if a.sharpa else "")
    spec.meshdir = os.path.join(os.path.dirname(os.path.abspath(a.src)), "meshes")  # keep the official meshes
    m = spec.compile()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(spec.to_xml())
    names = [m.joint(i).name for i in range(m.njnt)]
    print(f"{a.out}: nq {m.nq} nv {m.nv} nu {m.nu} nbody {m.nbody} mass {m.body_subtreemass[1]:.3f} kg, wrist_roll_link {m.body('left_wrist_roll_link').mass[0]:.3f} kg")
    print("free joint:", names[0], "root body:", m.body(m.jnt_bodyid[0]).name, "| hinges:", m.njnt - 1)
    assert m.nu == 0 and m.njnt == 30 and m.body(m.jnt_bodyid[0]).name == "trunk"


if __name__ == "__main__":
    main()
