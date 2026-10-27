"""ReplayRecorder of the locomotion task: rows carry the 7-d command instead of a reference frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.managers.recorder_manager import RecorderTermCfg

from supermimic.data.recorder import ReplayRecorder
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.loco_command import CLOCK_DIM, CMD_DIM
from supermimic.envs.sim_context import XI_DIM
from supermimic.envs.state_io import STATE_DIM


class LocoReplayRecorder(ReplayRecorder):
    def __init__(self, cfg: LocoReplayRecorderCfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        self.gait_clock = bool(cfg.params.get("gait_clock", False))
        goal_dim = CMD_DIM + (CLOCK_DIM if self.gait_clock else 0)
        self.buffer = ReplayBuffer(n, int(cfg.params["capacity"]), STATE_DIM, 29, XI_DIM, env.device, goal_dim=goal_dim)
        self.prev_cmd = torch.zeros(n, goal_dim, device=env.device)

    def _remember(self, ids: torch.Tensor) -> None:
        self.prev_s[ids] = self.state_io.read()[ids]
        self.prev_xi[ids] = self.ctx.read()[ids]
        cmd = self.command
        self.prev_cmd[ids] = torch.cat([cmd.command[ids], cmd.clock[ids]], dim=-1) if self.gait_clock else cmd.command[ids]

    def _write_rows(self, ids: torch.Tensor, done_next: torch.Tensor) -> None:
        env = self._env
        term = self.action_term
        self.buffer.add(
            ids,
            s=self.prev_s[ids],
            a=term.clean_raw[ids],
            a_exec=term.executed_raw[ids],
            xi=self.prev_xi[ids],
            r=env.reward_buf[ids],
            done=done_next,
            frame=self.prev_frame[ids],
            clip=self.prev_clip[ids],
            terminal=torch.zeros_like(done_next, dtype=torch.bool),
            ep_start=self.ep_start[ids],
            goal=self.prev_cmd[ids],
        )
        self.ep_start[ids] = False

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled or env_ids.numel() == 0:
            return
        env = self._env
        terminated = env.reset_terminated[env_ids].float()
        self._write_rows(env_ids, terminated)
        s_next = self.state_io.read()[env_ids]
        zeros_a = torch.zeros(env_ids.numel(), 29, device=env.device)
        self.buffer.add(
            env_ids,
            s=s_next,
            a=zeros_a,
            a_exec=zeros_a,
            xi=self.prev_xi[env_ids],
            r=torch.zeros(env_ids.numel(), device=env.device),
            done=terminated,
            frame=self.prev_frame[env_ids],
            clip=self.prev_clip[env_ids],
            terminal=torch.ones(env_ids.numel(), dtype=torch.bool, device=env.device),
            ep_start=torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device),
            goal=self.prev_cmd[env_ids],
        )
        self.buffer.end_episode(env_ids)


@dataclass
class LocoReplayRecorderCfg(RecorderTermCfg):
    func: type = LocoReplayRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000, "foot_geom_regex": r"^(left|right)_ankle_collision$", "command_name": "velocity"})
