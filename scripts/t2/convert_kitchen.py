"""t2_kitchen_reference (30 fps, pelvis-rooted TWIST2-style T2) -> SuperMimic tracking npz (50 Hz, trunk-rooted T2 Sharpa).

Joint map from joint_map.json (fitted: q_sm = sign * q_data + offset), pelvis -> trunk with the repository converter,
the kitchen floor (z = -0.105) lifted to z = 0. Extras kept for the kitchen render: fingers, objects, piece index.
    python scripts/t2/convert_kitchen.py <kitchen_dir> scripts/t2/kitchen_joint_map.json <out_dir>   (make_kitchen_clips.sh)
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, "/home/lch/SuperMimic-Reproduced/src")
sys.path.insert(0, "/home/lch/SuperMimic-Reproduced/scripts/t2")
from pkl_to_npz_t2 import pelvis_to_trunk  # noqa: E402
from supermimic.data.motion_convert import forward_kinematics, resample_motion  # noqa: E402
from supermimic.robots.t2 import T2_SHARPA_XML  # noqa: E402

K, MAP, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
FLOOR_Z = -0.105
IN_FPS, OUT_FPS = 30.0, 50.0
OUT.mkdir(parents=True, exist_ok=True)
model = mujoco.MjModel.from_xml_path(str(T2_SHARPA_XML))
sm_joints = [model.joint(i).name for i in range(1, model.njnt)]
jmap = json.load(open(MAP))
s = np.load(K / "sequence/full_sequence_30fps.npz")
names = [str(n) for n in s["robot_names"]]
Q, O = s["robot_qpos"], s["object_qpos"]
pieces = json.load(open(K / "pieces.json"))["pieces"]
finger_cols = [i for i, n in enumerate(names) if any(k in n for k in ("thumb", "index", "middle", "ring", "pinky"))]


def grid(n_in):
    """resample_motion's output grid: output frame -> (idx0, idx1, blend)."""
    duration = (n_in - 1) / IN_FPS
    n_out = math.floor(duration * OUT_FPS + 1e-6) + 1
    phase = np.arange(n_out) / OUT_FPS / duration
    i0 = np.clip(np.floor(phase * (n_in - 1)).astype(int), 0, n_in - 1)
    return i0, np.minimum(i0 + 1, n_in - 1), (phase * (n_in - 1) - i0)[:, None]


def lerp(x, i0, i1, b):
    return x[i0] * (1 - b) + x[i1] * b


def convert(a, b, name):
    q = Q[a : b + 1]
    joint_pos = np.stack([jmap[j]["sign"] * q[:, names.index(jmap[j]["src"])] + jmap[j]["offset"] for j in sm_joints], 1)
    p_pelvis = q[:, :3] - np.array([0, 0, FLOOR_Z])
    p_trunk, q_trunk = pelvis_to_trunk(model, joint_pos, p_pelvis, q[:, 3:7])
    raw = np.concatenate([p_trunk, q_trunk[:, [1, 2, 3, 0]], joint_pos], axis=1)
    motion = resample_motion(raw, IN_FPS, OUT_FPS)
    bodies = forward_kinematics(model, motion)
    i0, i1, bl = grid(len(q))
    assert len(i0) == motion.num_frames
    obj = lerp(O[a : b + 1], i0, i1, bl)
    for c in (3, 10):  # bottle quaternions: renormalise the lerp
        obj[:, c : c + 4] /= np.linalg.norm(obj[:, c : c + 4], axis=1, keepdims=True)
    np.savez(
        OUT / f"{name}.npz",
        fps=np.array([OUT_FPS], np.float32),
        input_fps=np.array([IN_FPS], np.float32),
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        joint_names=np.array(sm_joints),
        body_names=np.array(bodies.body_names),
        source=np.array(f"t2_kitchen_reference frames {a}-{b}"),
        finger_q=lerp(Q[a : b + 1][:, finger_cols], i0, i1, bl).astype(np.float32),
        finger_names=np.array([names[i] for i in finger_cols]),
        object_q=obj.astype(np.float32),
        object_names=s["object_names"],
        piece=s["piece"][a : b + 1][i0],
        seq_frame=i0 + a,
        floor_z=np.array([FLOOR_Z]),
        joint_map=np.array(json.dumps(jmap)),
    )
    return motion.num_frames


rows = {}
for p in pieces:
    a, b = p["frames"]
    nm = f"p{p['index']:02d}_{p['label'].split(' r')[0].replace(' ', '_')}"
    rows[nm] = convert(a, b, nm)
rows["full"] = convert(0, len(Q) - 1, "full")
print(json.dumps(rows))
