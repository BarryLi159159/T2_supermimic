"""Booster T2 MJCF for SuperMimic: the official T2_31dof.xml with the trunk as the free body (= IMU body, the v3 deploy
convention), the two head joints welded at zero and all actuators removed (mjlab adds its own), so the policy sees 29 hinges like the G1.
Usage: python make_t2_mjcf.py --src <booster_assets>/robots/T2/T2_31dof/T2_31dof.xml --out assets/t2/t2_29dof_trunk.xml
"""
import argparse
import os

import mujoco

HEAD_JOINTS = ("aa_head_yaw_joint", "head_pitch_joint")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
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
    spec.modelname = "booster_t2_29dof_trunk"
    spec.meshdir = os.path.join(os.path.dirname(os.path.abspath(a.src)), "meshes")  # keep the official meshes
    m = spec.compile()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(spec.to_xml())
    names = [m.joint(i).name for i in range(m.njnt)]
    print(f"{a.out}: nq {m.nq} nv {m.nv} nu {m.nu} nbody {m.nbody} mass {m.body_subtreemass[1]:.3f} kg")
    print("free joint:", names[0], "root body:", m.body(m.jnt_bodyid[0]).name, "| hinges:", m.njnt - 1)
    assert m.nu == 0 and m.njnt == 30 and m.body(m.jnt_bodyid[0]).name == "trunk"


if __name__ == "__main__":
    main()
