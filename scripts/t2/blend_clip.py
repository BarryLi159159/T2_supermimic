"""Ease a deploy clip in from (and out to) the standing pose the robot holds after X, so a clip cut mid-task does not
snap the arms at A. The ramps are smoothstep in joint space, the trunk upright at the standing height under the clip's
first (last) frame's xy and yaw; body states are recomputed by FK and the extra per-frame arrays hold their end value.
  python scripts/t2/blend_clip.py IN.npz OUT.npz [--out]      (--out: also ease back to standing at the end)
"""

import argparse
import math

import mujoco
import numpy as np

from supermimic.data.motion_convert import ResampledMotion, _so3_derivative, forward_kinematics
from supermimic.robots.t2 import T2_SHARPA_XML

# booster_deploy_supermimic DEFAULT_JOINT_POS without the head: the pose X blends the robot into
STAND_POS = {
    "shoulder_pitch": 0.35, "shoulder_roll": 1.3665, "elbow_yaw": 0.7008,
    "hip_pitch": -0.20, "knee_pitch": 0.42, "ankle_pitch": -0.23,
}
STAND_TRUNK_Z = 0.974  # [m] soles on the ground in the standing pose (booster_deploy_supermimic init comment)
RAMP_RAD_PER_S = 1.0  # ramp length = the largest joint move / this, clamped to RAMP_S
RAMP_S = (1.0, 2.5)

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--out", action="store_true")
args = parser.parse_args()

d = np.load(args.src)
fps = float(np.asarray(d["fps"]).reshape(-1)[0])
model = mujoco.MjModel.from_xml_path(str(T2_SHARPA_XML))
joints = [model.joint(i).name for i in range(1, model.njnt)]
assert joints == [str(n) for n in d["joint_names"]], "clip joint order must be the MJCF's"
bodies = [str(n) for n in d["body_names"]]
trunk = bodies.index("trunk")
stand = np.zeros(len(joints))
for i, name in enumerate(joints):
    for key, value in STAND_POS.items():
        if key in name:
            stand[i] = -value if name.startswith("left_") and key in ("shoulder_roll", "elbow_yaw") else value


def yaw_quat(q):  # wxyz -> the yaw part only
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def slerp(q0, q1, s):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    theta = math.acos(min(1.0, abs(float(np.dot(q0, q1)))))
    if theta < 1e-6:
        return q0
    return (math.sin((1 - s) * theta) * q0 + math.sin(s * theta) * q1) / math.sin(theta)


def ramp(q_from, q_to, p_from, p_to, r_from, r_to):
    move = np.abs(q_to - q_from).max()
    n = int(round(fps * min(max(move / RAMP_RAD_PER_S, RAMP_S[0]), RAMP_S[1])))
    s = np.arange(n) / n
    s = 3 * s**2 - 2 * s**3
    return (q_from + s[:, None] * (q_to - q_from), p_from + s[:, None] * (p_to - p_from),
            np.array([slerp(r_from, r_to, x) for x in s]))


q, p, r = d["joint_pos"], d["body_pos_w"][:, trunk], d["body_quat_w"][:, trunk]
stand_p0 = np.r_[p[0, :2], STAND_TRUNK_Z]
parts = [ramp(stand, q[0], stand_p0, p[0], yaw_quat(r[0]), r[0]), (q, p, r)]
if args.out:
    stand_p1 = np.r_[p[-1, :2], STAND_TRUNK_Z]
    q_out, p_out, r_out = ramp(q[-1], stand, p[-1], stand_p1, r[-1], yaw_quat(r[-1]))
    parts.append((np.vstack([q_out[1:], stand]), np.vstack([p_out[1:], stand_p1]), np.vstack([r_out[1:], yaw_quat(r[-1])])))
joint_pos = np.vstack([x[0] for x in parts])
root_pos = np.vstack([x[1] for x in parts])
root_quat = np.vstack([x[2] for x in parts])
root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True)
dt = 1.0 / fps
motion = ResampledMotion(fps=fps, root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos,
                         root_lin_vel_w=np.gradient(root_pos, dt, axis=0), root_ang_vel_w=_so3_derivative(root_quat, dt),
                         joint_vel=np.gradient(joint_pos, dt, axis=0))
states = forward_kinematics(model, motion)
assert states.body_names == bodies
n_in, n_clip, n = len(parts[0][0]), len(q), len(joint_pos)
out = {}
for key in d.files:
    v = d[key]
    if v.ndim > 0 and len(v) == n_clip:  # extra per-frame arrays (fingers, objects): hold their end values
        out[key] = np.concatenate([np.repeat(v[:1], n_in, 0), v, np.repeat(v[-1:], n - n_in - n_clip, 0)])
    else:
        out[key] = v
out.update(joint_pos=joint_pos, joint_vel=motion.joint_vel, body_pos_w=states.pos_w, body_quat_w=states.quat_w,
           body_lin_vel_w=states.lin_vel_w, body_ang_vel_w=states.ang_vel_w)
out["source"] = np.array(f"{d['source']} + ease in {n_in / fps:.2f} s" + (f", out {(n - n_in - n_clip) / fps:.2f} s" if args.out else ""))
np.savez(args.dst, **out)
print(f"{args.dst}: {n_in / fps:.2f} s in + {n_clip / fps:.2f} s clip + {(n - n_in - n_clip) / fps:.2f} s out")
