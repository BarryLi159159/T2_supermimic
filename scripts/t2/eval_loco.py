"""Walking protocol of Booster_T2's script/eval_student.py for a locomotion checkpoint, in mjlab.

Six scenarios, each held for 6 s on ``--reps`` envs from the default standing pose (no noise, no DR, friction 1.0):
stand, walk 0.5, walk 0.3 + turn 0.8, torso pitch 1.0, height 0.6, and a 1.2 m/s sideways root kick at 2 s while
standing. Reports, per scenario: the fall rate, the mean xy velocity error after 2 s and the mean forward speed.
Gate of the plan's step 1 (docs/supermimic_locoforce_brief.md in Booster_T2): walk 0.5 at >= 0.45 m/s,
walk 0.3 + turn at >= 0.28 m/s, no falls.

  uv run python scripts/t2/eval_loco.py --ckpt logs/<run>/ckpt_latest.pt --reps 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# name: (vx, vy, yaw rate, height, roll, pitch, yaw), kick speed m/s at KICK_S
SCENARIOS = {
    "stand": ((0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0), 0.0),
    "walk 0.5": ((0.5, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0), 0.0),
    "walk 0.3 turn 0.8": ((0.3, 0.0, 0.8, 0.85, 0.0, 0.0, 0.0), 0.0),
    "pitch 1.0": ((0.0, 0.0, 0.0, 0.85, 0.0, 1.0, 0.0), 0.0),
    "height 0.6": ((0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0), 0.0),
    "kick 1.2 side": ((0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0), 1.2),
}
KICK_S = 2.0
DURATION_S = 6.0


def load_policy(path: str, device: str):
    from supermimic.policy.loco_nets import LocoPolicy, LocoPolicyCfg

    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["loco"]["policy"]
    policy = LocoPolicy(LocoPolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"], ck["iteration"]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="json summary path (default: next to the checkpoint)")
    args = ap.parse_args()

    from supermimic.envs.loco_env_cfg import LocoEnvCfgArgs, make_loco_env

    policy, cfg, iteration = load_policy(args.ckpt, args.device)
    names = list(SCENARIOS)
    n = len(names) * args.reps
    env = make_loco_env(
        LocoEnvCfgArgs(
            robot=cfg["env"]["robot"],
            num_envs=n,
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            record_transitions=False,
            seed=0,
        ),
        device=args.device,
    )
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("velocity")
    command = torch.tensor([SCENARIOS[s][0] for s in names for _ in range(args.reps)], device=env.device)
    kick = torch.tensor([SCENARIOS[s][1] for s in names for _ in range(args.reps)], device=env.device)
    obs, _ = env.reset()
    cmd_term.set_fixed(torch.arange(n, device=env.device), command)
    obs = env.observation_manager.compute()
    pelvis = robot.body_names.index(env.loco_heads.spec.pelvis_body)

    dt = env.step_dt
    steps = int(DURATION_S / dt)
    kick_step = int(KICK_S / dt)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    vel_err = torch.zeros(n, device=env.device)
    speed = torch.zeros(n, device=env.device)
    air = torch.zeros(n, device=env.device)  # fraction of feet off the ground (stepping vs shuffling)
    from supermimic.envs.mdp.loco_rewards import soft_contact
    from supermimic.envs.state_io import StateIO

    io = StateIO(env)
    heads = env.loco_heads
    for step in range(steps):
        act = policy(obs["policy"], obs["goal"])
        if step == kick_step:
            state = torch.cat([robot.data.root_link_pose_w, robot.data.root_link_vel_w], dim=-1).clone()
            state[:, 8] += kick  # world y velocity: a sideways shove (the robot faces +x)
            robot.write_root_state_to_sim(state)
        obs, _, term, _, _ = env.step(act)
        fell |= term
        if step >= kick_step:
            v_w = robot.data.body_link_lin_vel_w[:, pelvis]
            q = robot.data.body_link_quat_w[:, pelvis]
            from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

            v_yaw = quat_apply_inverse(yaw_quat(q), v_w)
            vel_err += torch.linalg.vector_norm(v_yaw[:, :2] - command[:, :2], dim=-1)
            speed += v_yaw[:, 0]
            contact = soft_contact(heads.bodies(io.read()).feet_pos_w, heads.spec.sole_offset, heads.reward_cfg)
            air += (contact < 0.5).float().mean(-1)
    vel_err /= steps - kick_step
    speed /= steps - kick_step
    air /= steps - kick_step

    summary = {"ckpt": args.ckpt, "iteration": iteration, "reps": args.reps, "scenarios": {}}
    print(f"[loco eval] {args.ckpt} (iteration {iteration}), {args.reps} envs per scenario")
    print(f"{'scenario':<20} {'fell':>6} {'vel_err':>8} {'v_fwd':>7} {'air':>6}")
    for k, name in enumerate(names):
        rows = slice(k * args.reps, (k + 1) * args.reps)
        f, e, v, a = float(fell[rows].float().mean()), float(vel_err[rows].mean()), float(speed[rows].mean()), float(air[rows].mean())
        summary["scenarios"][name] = {"fell": f, "vel_err": e, "v_fwd": v, "air": a}
        print(f"{name:<20} {f:>6.2f} {e:>8.3f} {v:>7.3f} {a:>6.2f}")
    out = Path(args.out) if args.out else Path(args.ckpt).with_name(f"eval_loco_{iteration}.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"[loco eval] written {out}")
    env.close()


if __name__ == "__main__":
    main()
