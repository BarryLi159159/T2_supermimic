"""TWIST2 T2 motion pkls -> SuperMimic / mjlab motion npz for the trunk-rooted T2 model.

A TWIST2 pkl holds the pelvis (waist_yaw_link) pose, root_rot xyzw, and 29 joint angles in legged_gym order at the
clip's fps. SuperMimic's T2 model (assets/t2/t2_29dof_trunk.xml) is rooted at the trunk, so the root pose is moved from
the pelvis to the trunk with MuJoCo kinematics of the waist chain; joints are reordered to the model's joint order; the
rest (resampling to 50 Hz, finite-difference velocities, world-frame body states) is the repository's own converter.
Usage: python pkl_to_npz_t2.py --xml assets/t2/t2_29dof_trunk.xml --out data/npz/t2_demo <clip.pkl> [...]
       python pkl_to_npz_t2.py --xml ... --out data/npz/t2_twist2 --list clips.txt --workers 16
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from supermimic.data.motion_convert import forward_kinematics, quat_conjugate, quat_mul, resample_motion  # noqa: E402

TWIST2_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")] \
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] \
    + [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw", "wrist_roll")]
PELVIS = "waist_yaw_link"

_MODEL: mujoco.MjModel | None = None
_XML = ""


def _init(xml: str) -> None:
    global _MODEL, _XML
    _XML = xml
    _MODEL = mujoco.MjModel.from_xml_path(xml)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:  # wxyz, batched
    w, u = q[:, :1], q[:, 1:]
    return v * (2 * w * w - 1) + np.cross(u, v) * 2 * w + u * np.sum(u * v, axis=-1, keepdims=True) * 2


def pelvis_to_trunk(model: mujoco.MjModel, joint_pos: np.ndarray, p_pelvis: np.ndarray, q_pelvis: np.ndarray):
    """Trunk pose (world, wxyz) from the pelvis pose and the joint angles (model order), frame by frame."""
    data = mujoco.MjData(model)
    pelvis = model.body(PELVIS).id
    p_rel = np.empty_like(p_pelvis)
    q_rel = np.empty_like(q_pelvis)
    for t in range(len(joint_pos)):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[7:] = joint_pos[t]
        mujoco.mj_kinematics(model, data)
        p_rel[t], q_rel[t] = data.xpos[pelvis], data.xquat[pelvis]
    q_trunk = quat_mul(q_pelvis, quat_conjugate(q_rel))
    return p_pelvis - quat_rotate(q_trunk, p_rel), q_trunk


def convert_one(args: tuple[str, str, float]) -> dict:
    pkl, out, output_fps = args
    model = _MODEL
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    model_joints = [model.joint(i).name for i in range(1, model.njnt)]
    order = [TWIST2_JOINTS.index(n) for n in model_joints]
    joint_pos = np.asarray(d["dof_pos"], dtype=np.float64)[:, order]
    p_pelvis = np.asarray(d["root_pos"], dtype=np.float64)
    q_xyzw = np.asarray(d["root_rot"], dtype=np.float64)
    q_pelvis = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    p_trunk, q_trunk = pelvis_to_trunk(model, joint_pos, p_pelvis, q_pelvis)
    raw = np.concatenate([p_trunk, np.concatenate([q_trunk[:, 1:], q_trunk[:, :1]], axis=1), joint_pos], axis=1)  # the converter's layout: quat xyzw
    motion = resample_motion(raw, float(d["fps"]), output_fps)
    bodies = forward_kinematics(model, motion)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, fps=np.array([output_fps], dtype=np.float32),
             joint_pos=motion.joint_pos.astype(np.float32), joint_vel=motion.joint_vel.astype(np.float32),
             body_pos_w=bodies.pos_w.astype(np.float32), body_quat_w=bodies.quat_w.astype(np.float32),
             body_lin_vel_w=bodies.lin_vel_w.astype(np.float32), body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
             joint_names=np.array(model_joints), body_names=np.array(bodies.body_names), source=np.array(str(pkl)),
             input_fps=np.array([float(d["fps"])], dtype=np.float32))
    return {"name": Path(out).stem, "frames": motion.num_frames, "fps": output_fps, "duration_s": (motion.num_frames - 1) / output_fps, "input_fps": float(d["fps"])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pkls", nargs="*")
    p.add_argument("--list", help="text file with one pkl path per line")
    p.add_argument("--xml", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--output-fps", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    pkls = list(a.pkls)
    if a.list:
        pkls += [l.strip() for l in open(a.list) if l.strip()]
    jobs = [(pk, os.path.join(a.out, Path(pk).stem.replace("_stageii", "") + ".npz"), a.output_fps) for pk in pkls]
    with Pool(a.workers, initializer=_init, initargs=(a.xml,)) as pool:
        rows = pool.map(convert_one, jobs, chunksize=4)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "frames", "fps", "duration_s", "input_fps"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips, {sum(r['duration_s'] for r in rows) / 3600:.2f} h -> {a.out}")


if __name__ == "__main__":
    main()
