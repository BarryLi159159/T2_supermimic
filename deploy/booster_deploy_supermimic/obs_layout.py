"""SuperMimic observation layout as numpy, shared by the booster_deploy task and the mjlab cross-check.

Policy input (930) = five proprioception terms, each as its 10-frame history oldest -> newest (mjlab term-major
flattening; a fresh history is back-filled with the first frame):
  joint_pos - default (29) | joint_vel (29) | IMU angular velocity, body frame (3) | projected gravity (3) | last action (29)
Goal input (K*68) = reference frames f .. f+K-1 (clamped at the clip end), each
  joint_pos - default (29) | joint_vel (29) | gravity in the reference root frame (3) | reference root angular
  velocity in the root frame (3) | reference root linear velocity in the root's heading (yaw) frame (3) | root height (1)
Quaternions are wxyz. Joints are in the training model's order (the npz joint_names).
"""
from __future__ import annotations

import numpy as np

OBS_TERM_DIMS = (29, 29, 3, 3, 29)
GOAL_FRAME_DIM = 68
GRAVITY_W = np.array([0.0, 0.0, -1.0])


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express v (parent frame) in the frame q; batched over leading dims."""
    w, u = q[..., :1], q[..., 1:]
    return v * (2 * w * w - 1) - np.cross(u, v) * 2 * w + u * np.sum(u * v, axis=-1, keepdims=True) * 2


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = np.zeros_like(q)
    out[..., 0] = np.cos(yaw / 2)
    out[..., 3] = np.sin(yaw / 2)
    return out


def proprio_frame(joint_pos: np.ndarray, joint_vel: np.ndarray, ang_vel_b: np.ndarray, root_quat_w: np.ndarray,
                  last_action: np.ndarray, default_joint_pos: np.ndarray) -> np.ndarray:
    """One 93-d proprioception frame in term order."""
    gravity_b = quat_rotate_inverse(np.asarray(root_quat_w, dtype=np.float64), GRAVITY_W)
    return np.concatenate([joint_pos - default_joint_pos, joint_vel, ang_vel_b, gravity_b, last_action]).astype(np.float64)


class History:
    """Per-term circular history flattened term-major, oldest first; back-filled with the first frame after reset."""

    def __init__(self, length: int = 10):
        self.length = length
        self.frames = np.zeros((length, sum(OBS_TERM_DIMS)))
        self.empty = True

    def reset(self) -> None:
        self.empty = True

    def push(self, frame: np.ndarray) -> np.ndarray:
        if self.empty:
            self.frames[:] = frame
            self.empty = False
        else:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = frame
        chunks = []
        start = 0
        for d in OBS_TERM_DIMS:
            chunks.append(self.frames[:, start:start + d].reshape(-1))
            start += d
        return np.concatenate(chunks)


class ReferenceClip:
    """A mjlab-format motion npz (50 Hz) read for the goal window and the ghost."""

    def __init__(self, path: str, root_body: str, default_joint_pos: np.ndarray):
        d = np.load(path)
        self.joint_names = [str(n) for n in d["joint_names"]]
        names = [str(n) for n in d["body_names"]]
        r = names.index(root_body)
        self.fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        self.joint_pos = d["joint_pos"].astype(np.float64)
        self.joint_vel = d["joint_vel"].astype(np.float64)
        self.root_pos_w = d["body_pos_w"][:, r].astype(np.float64)
        self.root_quat_w = d["body_quat_w"][:, r].astype(np.float64)
        root_lin_vel_w = d["body_lin_vel_w"][:, r].astype(np.float64)
        root_ang_vel_w = d["body_ang_vel_w"][:, r].astype(np.float64)
        self.num_frames = len(self.joint_pos)
        q = self.root_quat_w
        self.goal_feat = np.concatenate([
            self.joint_pos - default_joint_pos,
            self.joint_vel,
            quat_rotate_inverse(q, np.broadcast_to(GRAVITY_W, q[:, :3].shape)),
            quat_rotate_inverse(q, root_ang_vel_w),
            quat_rotate_inverse(yaw_quat(q), root_lin_vel_w),
            self.root_pos_w[:, 2:3],
        ], axis=1)  # (T, 68)
        assert self.goal_feat.shape[1] == GOAL_FRAME_DIM

    def goal_window(self, frame: int, k: int) -> np.ndarray:
        idx = np.minimum(frame + np.arange(k), self.num_frames - 1)
        return self.goal_feat[idx].reshape(-1)
