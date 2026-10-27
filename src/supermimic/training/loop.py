"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robots import get_robot
from supermimic.training.distributed import (
    allreduce_mean,
    broadcast_params,
    finalize,
    init_distributed,
)
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads, anchor_obs_dim
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        cfg.policy.goal_dim = cfg.env.future_frames * 68 + anchor_obs_dim(cfg.env.anchor_obs, cfg.env.anchor_ori_only)
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.anchor_ori_only = cfg.env.anchor_ori_only
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(
                json.dumps(asdict(cfg), indent=2, default=str)
            )
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None
        self._build()

        # Episode statistics of the training env (paper's "training return").
        env = self.env
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)
        elif cfg.finetune is not None:
            self.load_weights(cfg.finetune)

    # ------------------------------------------------------------------ task-specific construction and hooks

    def _build(self) -> None:
        """Env, world model, policy, critic and their trainers (the motion-tracking task)."""
        cfg = self.cfg
        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(
                cfg.data.npz_dir,
                limit=cfg.data.limit,
                names=cfg.data.clips,
                manifest=cfg.data.manifest,
            )
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot=cfg.env.robot,
                wrist_payload=cfg.env.wrist_payload,
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                anchor_ori_only=cfg.env.anchor_ori_only,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        spec = get_robot(cfg.env.robot)
        fk = G1Kinematics(spec.load_model(), device=self.device)
        heads = AnalyticHeads(
            fk,
            spec.tracked_bodies,
            spec.anchor_body,
            spec.root_body,
            robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(
            self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer
        )
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self._print_params()

    def _print_params(self) -> None:
        n_params = lambda m: sum(p.numel() for p in m.parameters())
        if self.is_main:
            print(
                f"[INFO] params: world model {n_params(self.wm) / 1e6:.2f}M, policy {n_params(self.policy) / 1e6:.2f}M, critic {n_params(self.critic) / 1e6:.3f}M"
            )

    def _sampler_state(self) -> dict:
        """Task state saved with a checkpoint (the return-based sampler's bins)."""
        return {"bin_return": self.command.bin_return, "bin_memory": self.command.bin_memory}

    def _load_sampler_state(self, ck: dict) -> None:
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])

    def _extra_scalars(self) -> dict[str, float]:
        """Task scalars logged at every log interval."""
        return {
            "sampler/entropy": float(self.command.metrics["sampling_entropy"][0]),
            "sampler/bin_return_mean": float(self.command.bin_return.mean()),
        }

    def _on_log(self, iteration: int) -> None:
        """Per-log-interval task hook (curricula)."""

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                **self._sampler_state(),
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def load_weights(self, path: Path) -> None:
        """Fine-tune on a changed robot: weights, Adam moments and the return-based sampler state come from the
        checkpoint, iteration and learning-rate schedule start fresh. A wider simulation context (new xi entries
        appended at the end of xi) gets zero input columns, and zero Adam moments, in the first layer of the dynamics
        network and the critic."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        wm_sd, critic_sd = ck["wm"], ck["critic"]
        new_xi = self.wm.dynamics.mlp.net[0].in_features - (71 + 29)
        old_xi = wm_sd["dynamics.mlp.net.0.weight"].shape[1] - (71 + 29)
        goal_dim = self.cfg.policy.goal_dim
        widen = lambda w, start: torch.cat(
            [
                w[:, : start + old_xi],
                w.new_zeros(w.shape[0], new_xi - old_xi),
                w[:, start + old_xi :],
            ],
            dim=1,
        )
        if old_xi != new_xi:
            wm_sd["dynamics.mlp.net.0.weight"] = widen(wm_sd["dynamics.mlp.net.0.weight"], 71 + 29)
            critic_sd["mlp.net.0.weight"] = widen(critic_sd["mlp.net.0.weight"], 71 + goal_dim)
            # Adam keeps one exp_avg / exp_avg_sq per parameter, indexed in parameters() order
            for opt_sd, params, first, start in (
                (ck["wm_opt"], list(self.wm.parameters()), self.wm.dynamics.mlp.net[0].weight, 71 + 29),
                (ck["critic_opt"], list(self.critic.parameters()), self.critic.mlp.net[0].weight, 71 + goal_dim),
            ):
                idx = next(i for i, q in enumerate(params) if q is first)
                st = opt_sd["state"][idx]
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = widen(st[k], start)
            print(
                f"[INFO] fine-tune: xi widened {old_xi} -> {new_xi} (zero columns and zero Adam moments for the new entries)"
            )
        self.wm.load_state_dict(wm_sd)
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(critic_sd)
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self._load_sampler_state(ck)
        print(
            f"[INFO] fine-tune from {path} (iteration {ck['iteration']}): weights, Adam moments and sampler state loaded, schedule fresh"
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self._load_sampler_state(ck)
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(
                        self.iteration,
                        sch.total_iterations,
                        cfg.wm_trainer.lr,
                        sch.lr_final_ratio,
                        sch.lr_decay_fraction,
                    )
                    if sch.lr_warmup_iterations > 0:
                        lr *= min(1.0, self.iteration / sch.lr_warmup_iterations)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    m_pi = {}
                    for _ in range(sch.policy_updates_per_wm_update if self.iteration >= sch.wm_only_iterations else 0):
                        m_pi = self.policy_trainer.update(
                            self.buffer, train_actor=self.iteration >= sch.policy_freeze_iterations
                        )
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                self._on_log(it)
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar(
                        "train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it
                    )
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    for k, v in self._extra_scalars().items():
                        self.writer.add_scalar(k, v, it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed / 3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan')) / max(agg_n, 1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan')) / max(agg_n, 1):.3f} surv {agg.get('policy/imag_survival_H', float('nan')) / max(agg_n, 1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(
                        stop
                    )  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
