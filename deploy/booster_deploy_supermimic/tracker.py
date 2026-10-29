"""SuperMimic whole-body tracking policy (T2, 29 joints, trunk = policy root = IMU body) for booster_deploy.

The policy sees only proprioception (obs_layout.py) and the next K reference frames; the IMU quantities from
robot.data (trunk orientation, trunk gyro) enter the observation as they are. Targets = default + per-joint
action_scale * action, the scale and the default pose coming from the <model>.json that scripts/export_onnx.py
writes next to the ONNX graph. The two head joints are not in the policy and hold zero.
"""
from __future__ import annotations

from dataclasses import MISSING
import json
import os

import numpy as np
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab.configclass import configclass

from .obs_layout import History, ReferenceClip, proprio_frame, quat_rotate_inverse, GRAVITY_W

ROOT_BODY = "trunk"
HEAD_JOINTS = ["aa_head_yaw_joint", "head_pitch_joint"]


class OnnxPolicy:
    """Plain onnxruntime session on the fixed-batch graph (see export_onnx.py on booster_deploy's CpuOnnxRunner)."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.session.get_inputs()]

    def __call__(self, obs_hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        feeds = {self.names[0]: obs_hist.astype(np.float32)[None], self.names[1]: goal.astype(np.float32)[None]}
        return self.session.run(None, feeds)[0][0]


class SuperMimicPolicy(Policy):
    def __init__(self, cfg: SuperMimicPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        model_path = cfg.checkpoint_path if os.path.isabs(cfg.checkpoint_path) else os.path.join(self.task_path, cfg.checkpoint_path)
        self._model = OnnxPolicy(model_path)
        with open(os.path.splitext(model_path)[0] + ".json") as f:
            meta = json.load(f)
        if not meta["robot"].startswith("t2"):  # t2, t2_sharpa, t2_v4, t2_v4_sharpa: one deployment table
            raise ValueError(f"{model_path} was trained on {meta['robot']}, this task is for the T2")
        self.trained_robot = meta["robot"]  # t2 = bare 0.425 kg wrist link, t2_sharpa = Sharpa hand 1.64 kg; the MuJoCo xml must match
        self.policy_joints = list(meta["joint_names"])
        self.action_scale = np.asarray(meta["action_scale"], dtype=np.float64)
        self.default = np.asarray(meta["default_joint_pos"], dtype=np.float64)
        self.history_len = int(meta["history"])
        self.future_frames = int(meta["future_frames"])
        if meta.get("anchor_obs", False):
            raise ValueError("anchor observations cannot be built on the real robot; export a pure-proprio policy")

        names = list(self.robot.cfg.joint_names)  # real-robot (serial) order
        self.policy_idx = np.array([names.index(n) for n in self.policy_joints])
        self.head_idx = np.array([names.index(n) for n in HEAD_JOINTS])
        self.default_all = self.robot.default_joint_pos.cpu().numpy().astype(np.float64)
        if np.abs(self.default_all[self.policy_idx] - self.default).max() > 1e-6:
            raise ValueError("robot default_joint_pos differs from the default pose the policy was trained with")

        motion_path = cfg.motion_path if os.path.isabs(cfg.motion_path) else os.path.join(self.task_path, cfg.motion_path)
        self.clip = ReferenceClip(motion_path, ROOT_BODY, self.default)
        if self.clip.joint_names != self.policy_joints:
            raise ValueError(f"{motion_path} joint order differs from the policy's")
        if abs(self.clip.fps * controller.cfg.policy_dt - 1.0) > 1e-6:
            raise ValueError(f"{motion_path} is {self.clip.fps:.1f} Hz, the policy runs at {1 / controller.cfg.policy_dt:.1f} Hz")

        self.num_actions = len(self.policy_joints)
        self.history = History(self.history_len)
        self.last_action = np.zeros(self.num_actions)
        self.frame = cfg.start_frame

    def reset(self) -> None:
        self.frame = self.cfg.start_frame
        self.last_action[:] = 0.0
        self.history.reset()

    def compute_observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.robot.data
        q_all = d.joint_pos.cpu().numpy().astype(np.float64)
        qd_all = d.joint_vel.cpu().numpy().astype(np.float64)
        root_quat = d.root_quat_w.cpu().numpy().astype(np.float64)
        ang_vel_b = d.root_ang_vel_b.cpu().numpy().astype(np.float64)
        frame = proprio_frame(q_all[self.policy_idx], qd_all[self.policy_idx], ang_vel_b, root_quat, self.last_action, self.default)
        obs_hist = self.history.push(frame)
        goal = self.clip.goal_window(self.frame, self.future_frames)
        return obs_hist, goal, root_quat

    def inference(self) -> torch.Tensor:
        obs_hist, goal, root_quat = self.compute_observation()
        action = np.asarray(self._model(obs_hist, goal), dtype=np.float64).reshape(-1)
        if action.shape[0] != self.num_actions:
            raise RuntimeError(f"policy returned {action.shape[0]} actions, expected {self.num_actions}")
        self.last_action[:] = action

        targets = self.default_all.copy()
        targets[self.policy_idx] = self.default + self.action_scale * action
        targets[self.head_idx] = 0.0

        k = min(self.frame, self.clip.num_frames - 1)
        if hasattr(self.controller, "set_reference_qpos"):  # MuJoCo ghost: trunk pose + all joints in the real order
            joints = self.default_all.copy()
            joints[self.policy_idx] = self.clip.joint_pos[k]
            self.controller.set_reference_qpos(np.concatenate([self.clip.root_pos_w[k], self.clip.root_quat_w[k], joints]))

        if self.cfg.enable_safety_fallback:
            if np.dot(quat_rotate_inverse(root_quat, GRAVITY_W), quat_rotate_inverse(self.clip.root_quat_w[k], GRAVITY_W)) < self.cfg.safety_gravity_dot:
                print("\nLarge root tracking error is detected, stopping policy for safety. "
                      f"Disable it with {self.cfg.__class__.__name__}.enable_safety_fallback = False.")
                self.controller.stop()

        self.frame += 1
        if self.cfg.stop_at_motion_end and self.frame >= self.clip.num_frames:
            self.controller.stop()
        return torch.from_numpy(targets.astype(np.float32))


@configclass
class SuperMimicPolicyCfg(PolicyCfg):
    constructor = SuperMimicPolicy
    checkpoint_path: str = MISSING  # type: ignore  # .onnx with its .json contract next to it
    motion_path: str = MISSING  # type: ignore  # mjlab-format npz, 50 Hz, trunk-rooted T2
    start_frame: int = 0
    stop_at_motion_end: bool = True
    # stop when the trunk gravity direction differs from the reference by more than 60 degrees (Booster's BeyondMimic rule)
    safety_gravity_dot: float = 0.5
