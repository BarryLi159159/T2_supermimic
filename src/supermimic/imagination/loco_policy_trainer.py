"""Short-horizon actor-critic of the locomotion policy through the frozen world model (paper §3.3)."""

from __future__ import annotations

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.imagination.loco_rollout import LocoImagStep, LocoRollout, loco_imagine
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.loco_nets import LocoCritic, LocoPolicy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.loco_heads import LocoWorldModel


class LocoPolicyTrainer:
    def __init__(self, wm: LocoWorldModel, policy: LocoPolicy, critic: LocoCritic, cfg: PolicyTrainerCfg, switch_prob: float = 0.0):
        self.wm, self.policy, self.critic, self.cfg = wm, policy, critic, cfg
        self.switch_prob = switch_prob
        self.action_noise = 0.0  # set by the training loop's schedule
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        self.step_fn = LocoImagStep(wm, compile_mode=cfg.compile)
        self.skipped = 0

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_feat(self, ro: LocoRollout) -> torch.Tensor:
        H1, n, _ = ro.states.shape
        return self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)

    def update(self, buffer: ReplayBuffer, train_actor: bool = True) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        with torch.set_grad_enabled(train_actor):
            ro = loco_imagine(self.wm, self.policy, start, cfg.horizon, obs_noise=cfg.obs_noise, step_fn=self.step_fn, switch_prob=self.switch_prob, action_noise=self.action_noise)
            feat = self._critic_feat(ro)
            v_H = self.critic(feat[-1], ro.cmd[-1], xi_norm) if cfg.use_critic else None
            J = policy_objective(ro, v_H, cfg.gamma)
        if train_actor:
            loss = -J / cfg.horizon
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.policy)
            gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
            if torch.isfinite(gn):
                torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
                self.opt.step()
            else:
                self.skipped += 1  # a non-finite gradient (one bad rollout) must not poison the weights
        else:
            gn = torch.zeros(())
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "skipped_updates": float(self.skipped),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v

        # ---- critic: fit to GAE targets of this rollout (targets fixed)
        if cfg.use_critic:
            with torch.no_grad():
                feat_d = feat.detach()
                H1, n, _ = feat_d.shape
                cmd_rep = ro.cmd
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(feat_d.reshape(H1 * n, -1), cmd_rep.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1)).reshape(H1, n)
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_cmd = cmd_rep[:-1].reshape(-1, cmd_rep.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_cmd[idx], X_xi[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics
