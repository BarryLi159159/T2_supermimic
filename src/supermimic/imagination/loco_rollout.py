"""Imagined rollouts of the locomotion policy through the world model (paper §3.3, velocity-command edition).

The command of the start row is held for the whole horizon (it is piecewise constant over seconds; the horizon is
0.64 s); a gait clock after it (9-d goal) advances one control step per imagined step, as in the simulator. The
replay goal may end with the feet air timers of the start state (``feet_air_time``): they advance on each imagined
state's contact and stay in the rollout's goal (the critic sees them, the policy does not). The arm joints keep the
start row's normalised targets; the policy's 15 outputs are scattered into the 29-d action the dynamics network
consumes. The reward uses the termination probability as the termination penalty's expectation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.loco_rewards import (
    LOCO_REWARD_TERM_NAMES,
    advance_air_time,
    advance_clock,
    timer_contact,
)
from supermimic.policy.loco_nets import LocoPolicy
from supermimic.world_model.loco_heads import LOCO_OBS_TERMS, LocoWorldModel

# Isaac's sensor noise (uniform half-widths) in the 58-d observation order; commands and actions are exact.
LOCO_OBS_NOISE_HALF = torch.tensor([0.2] * 3 + [0.05] * 3 + [0.0] * 7 + [0.01] * 15 + [1.5] * 15 + [0.0] * 15)


@dataclass
class LocoRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 15)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    cmd: torch.Tensor  # (H+1, n, G) goal at each state (7, + 2 with the gait clock, + 2 with the air timers)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def flatten_term_major(hist: torch.Tensor, terms=LOCO_OBS_TERMS) -> torch.Tensor:
    """(H, N, D) oldest→newest  ->  (N, H*D) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for _, d in terms:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


class LocoImagStep:
    """One imagined step (dynamics, FK, reward, termination logits, next observation) as one compiled function."""

    def __init__(self, wm: LocoWorldModel, compile_mode: str | None = None):
        self.wm = wm
        self.ids = wm.heads.policy_ids
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a15, a15_prev, arm29, xi, cmd, air):
        a29 = arm29.index_copy(-1, self.ids, a15)
        a29_prev = arm29.index_copy(-1, self.ids, a15_prev)
        heads = self.wm.heads
        s_next = self.wm.step(s, a29, xi)
        bodies = heads.bodies(s_next)
        logits = self.wm.termination_logits(s_next, bodies)
        d = torch.sigmoid(logits)
        if air is not None:
            rcfg = heads.reward_cfg
            air = advance_air_time(air, timer_contact(bodies.feet_pos_w, heads.spec.sole_offset, rcfg), rcfg.dt)
        terms = heads.reward(s, s_next, a29, a29_prev, cmd, d, bodies, air)
        r = terms["total"]
        term_vec = torch.stack([terms[k] for k in LOCO_REWARD_TERM_NAMES], dim=0)
        obs_clean = heads.observation(s_next, a15, cmd, xi)
        return s_next, r, term_vec, logits, obs_clean, air

    def __call__(self, s, a15, a15_prev, arm29, xi, cmd, air=None):
        return self._fn(s, a15, a15_prev, arm29, xi, cmd, air)


def loco_imagine(wm: LocoWorldModel, policy: LocoPolicy, start: StartBatch, horizon: int, obs_noise: bool = True, step_fn: LocoImagStep | None = None) -> LocoRollout:
    device = start.state.device
    n = start.state.shape[0]
    heads = wm.heads
    ids = heads.policy_ids
    step_fn = step_fn or LocoImagStep(wm)
    noise_half = LOCO_OBS_NOISE_HALF.to(device)
    g_pi = policy.cfg.goal_dim
    cmd = start.goal[:, :g_pi]
    air = start.goal[:, g_pi:] if start.goal.shape[-1] > g_pi else None
    has_clock = g_pi > 7
    rcfg = heads.reward_cfg

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    hist = [add_noise(heads.observation(start.hist_states[i], start.hist_actions[i][:, ids], cmd, start.xi)) for i in range(start.hist_states.shape[0])]
    s = start.state
    arm29 = start.prev_action  # the arms' normalised targets, held over the horizon
    a_prev = start.prev_action[:, ids]
    c = torch.ones(n, device=device)
    states = [s]
    actions, rewards = [], []
    dhats, survivals = [torch.zeros(n, device=device)], [c]
    term_sum = torch.zeros(len(LOCO_REWARD_TERM_NAMES), device=device)
    cmds = [start.goal]

    for _ in range(horizon):
        obs_flat = flatten_term_major(torch.stack(hist, 0))
        a = policy(obs_flat, cmd)
        s_next, r, term_vec, logits, obs_clean, air = step_fn(s, a, a_prev, arm29, start.xi, cmd, air)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)
        hist = hist[1:] + [add_noise(obs_clean)]
        if has_clock:
            cmd = torch.cat([cmd[..., :7], advance_clock(cmd[..., 7:], rcfg.dt, rcfg.gait_period)], dim=-1)
        cmds.append(cmd if air is None else torch.cat([cmd, air], dim=-1))
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)

    return LocoRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        cmd=torch.stack(cmds),
        reward_terms={k: float(v / horizon) for k, v in zip(LOCO_REWARD_TERM_NAMES, term_sum.tolist())},
    )
