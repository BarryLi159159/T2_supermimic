"""Training loop of the locomotion task: the tracking loop with the locomotion env, heads, nets and trainers."""

from __future__ import annotations

from pathlib import Path

import torch

from supermimic.envs.loco_env_cfg import LocoEnvCfgArgs, make_loco_env
from supermimic.imagination.loco_policy_trainer import LocoPolicyTrainer
from supermimic.policy.loco_nets import LocoCritic, LocoPolicy
from supermimic.training.distributed import broadcast_params
from supermimic.training.loop import Trainer
from supermimic.world_model.loco_heads import LocoWorldModel
from supermimic.world_model.wm_trainer import LocoWorldModelTrainer


class LocoTrainer(Trainer):
    def _build(self) -> None:
        cfg = self.cfg
        self.env = make_loco_env(
            LocoEnvCfgArgs(
                robot=cfg.env.robot,
                num_envs=cfg.env.num_envs,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                replay_capacity=cfg.env.replay_capacity,
                seed=cfg.seed + 1000 * self.rank,
                resampling_time_range=cfg.loco.resampling_time_range,
                rel_standing_envs=cfg.loco.rel_standing_envs,
            ),
            device=self.device,
            reward_cfg=cfg.loco.reward,
        )
        env = self.env
        self.command = env.command_manager.get_term("velocity")
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        heads = env.loco_heads
        self.wm = LocoWorldModel(cfg.world_model, heads, self.device)
        self.policy = LocoPolicy(cfg.loco.policy).to(self.device)
        self.critic = LocoCritic(cfg.loco.critic).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = LocoWorldModelTrainer(self.wm, cfg.wm_trainer)
        self.policy_trainer = LocoPolicyTrainer(self.wm, self.policy, self.critic, cfg.policy_trainer)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self._print_params()

    # ------------------------------------------------------------------ hooks

    def _sampler_state(self) -> dict:
        return {}

    def _load_sampler_state(self, ck: dict) -> None:
        pass

    def _extra_scalars(self) -> dict[str, float]:
        return {
            "command/progress": self.command.progress,
            "command/error_vel_xy": float(self.command.metrics["error_vel_xy"].mean()),
            "command/error_vel_yaw": float(self.command.metrics["error_vel_yaw"].mean()),
        }

    def _on_log(self, iteration: int) -> None:
        lo, hi = self.cfg.loco.curriculum_start, self.cfg.loco.curriculum_end
        self.command.set_progress((iteration - lo) / max(hi - lo, 1))

    def load_weights(self, path: Path) -> None:
        """Fine-tune: weights and Adam moments from the checkpoint, iteration and schedule fresh."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        print(f"[INFO] fine-tune from {path} (iteration {ck['iteration']}): weights and Adam moments loaded, schedule fresh")
