"""Write Booster's official T2_31dof.xml with the Sharpa-hand wrist inertia, for booster_deploy's MuJoCo controller.

The wrist values are the ones in assets/t2/t2_29dof_trunk_sharpa.xml (TWIST2 booster_t2_29dof_sharpa.urdf); everything
else (actuators, head joints, meshes) stays Booster's, so the Sharpa policies are judged in the same simulator as the
bare ones, only with the hand mass they were trained with.
"""
import argparse
import os
import re

OFFICIAL = os.path.expanduser("~/Booster_T2/asset/booster_assets/robots/T2/T2_31dof/T2_31dof.xml")
SHARPA = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "t2", "t2_29dof_trunk_sharpa.xml")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "booster_deploy_supermimic", "robots", "t2", "T2_31dof_sharpa.xml")


def wrist_inertial(xml: str, side: str) -> str:
    m = re.search(rf'<body name="{side}_wrist_roll_link">\s*\n\s*(<inertial [^\n]*/>)', xml)
    assert m, side
    return m.group(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.abspath(OUT))
    a = p.parse_args()
    official = open(OFFICIAL).read()
    sharpa = open(SHARPA).read()
    out = official
    for side in ("left", "right"):
        old, new = wrist_inertial(official, side), wrist_inertial(sharpa, side)
        assert out.count(old) == 1, side
        out = out.replace(old, new)
    meshdir = os.path.join(os.path.dirname(OFFICIAL), "meshes")
    out = out.replace('meshdir="meshes/"', f'meshdir="{meshdir}/"', 1)
    assert meshdir in out
    with open(a.out, "w") as f:
        f.write(out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
