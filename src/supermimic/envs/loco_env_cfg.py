"""Velocity-command locomotion environment on mjlab (Booster_T2's lower-body task, world-model edition).

What the policy sees and earns follows IsaacLab-Decoupled-WBC's T2 flat teacher (Booster_T2 config/t2_flat_config.py)
as far as a state-only world model allows:
  * observations: the deployable 58-d layout (trunk IMU, command 7, 15 policy joints, last action), 10-frame history,
    Isaac's uniform sensor noise; a separate ``goal`` group carries the command;
  * action: ``LowerBodyAction`` (15 policy joints, arms held, execution noise on the policy joints);
  * command: ``VelocityBodyCommand`` (7-d, 3-6 s, 40 % standing);
  * rewards: the analytic table of ``envs/mdp/loco_rewards.py`` evaluated on the simulator state (one term);
  * terminations: pelvis height (Isaac's 0.3 m scaled to the T2), 20 s time-out;
  * events: the SuperMimic DR set (COM at startup, encoder bias and foot friction per episode) plus Isaac's reset
    noise (root pose / velocity, joint scale 0.5-1.5);
  * a ``LocoReplayRecorder`` streams transitions (with the command) into the GPU replay buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.loco_recorder import LocoReplayRecorderCfg
from supermimic.envs.loco_actions import LowerBodyActionCfg
from supermimic.envs.mdp import loco_sim
from supermimic.envs.mdp.loco_command import VelocityBodyCommandCfg
from supermimic.envs.mdp.loco_rewards import LocoRewardCfg
from supermimic.envs.tracking_env_cfg import QuietSimulationCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots import get_robot
from supermimic.world_model.loco_heads import LocoHeads, LocoTerminationCfg


@dataclass
class LocoEnvCfgArgs:
    robot: str = "t2"
    num_envs: int = 4096
    history_length: int = 10
    episode_length_s: float = 20.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    resampling_time_range: tuple[float, float] = (3.0, 6.0)
    rel_standing_envs: float = 0.4
    min_pelvis_height: float = LocoTerminationCfg().min_pelvis_height
    rsi_npz_files: list[str] | None = None  # reference-state initialisation clips (None = off)
    rsi_prob: float = 0.5
    gait_clock: bool = False  # the goal (policy input, replay) carries the gait clock: 9-d instead of 7-d
    gait_period: float = 0.8
    foot_timers: bool = False  # replay rows carry the foot air / contact timers (the reward's Isaac timer forms)


def make_loco_env_cfg(a: LocoEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    robot = get_robot(a.robot)
    assert robot.policy_joints is not None and robot.pelvis_body is not None
    policy_cfg = SceneEntityCfg("robot", joint_names=robot.policy_joints, preserve_order=True)
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    # Isaac's layout and noise scales (base_env.py / base_env_config.py).
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "command": ObservationTermCfg(func=base_mdp.generated_commands, params={"command_name": "velocity"}),
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True, "asset_cfg": policy_cfg}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, params={"asset_cfg": policy_cfg}, noise=noise(-1.5, 1.5)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "command": ObservationTermCfg(func=base_mdp.generated_commands, params={"command_name": "velocity"}),
                **({"gait_clock": ObservationTermCfg(func=loco_sim.gait_clock, params={"command_name": "velocity"})} if a.gait_clock else {}),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": LowerBodyActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=robot.action_scale,
            use_default_offset=True,
            joint_order=robot.joint_names,
            policy_joints=robot.policy_joints,
            noise_std=a.action_noise_std,
        )
    }

    commands = {
        "velocity": VelocityBodyCommandCfg(
            resampling_time_range=a.resampling_time_range,
            rel_standing_envs=a.rel_standing_envs,
            gait_period=a.gait_period,
        )
    }

    # ---------------------------------------------------------------- events
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(robot.com_body,)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=robot.foot_geom_regex),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
    else:
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=robot.foot_geom_regex),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )
    # Isaac's reset: root pose xy +-0.5, yaw +-pi, root velocity +-0.5 on every axis, joints scaled 0.5-1.5.
    pose_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)} if a.reset_noise else {}
    vel_range = {k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")} if a.reset_noise else {}
    events["reset_base"] = EventTermCfg(
        mode="reset",
        func=base_mdp.reset_root_state_uniform,
        params={"pose_range": pose_range, "velocity_range": vel_range, "asset_cfg": SceneEntityCfg("robot")},
    )
    events["reset_joints"] = EventTermCfg(
        mode="reset",
        func=loco_sim.reset_joints_by_scale,
        params={"position_range": (0.5, 1.5) if a.reset_noise else (1.0, 1.0), "asset_cfg": SceneEntityCfg("robot")},
    )

    if a.rsi_npz_files:
        events["reset_from_clip"] = EventTermCfg(
            mode="reset",
            func=loco_sim.reset_from_clip,
            params={"npz_files": list(a.rsi_npz_files), "prob": a.rsi_prob, "robot_name": a.robot, "asset_cfg": SceneEntityCfg("robot")},
        )

    rewards = {
        "loco": RewardTermCfg(func=loco_sim.loco_reward, weight=1.0, params={"command_name": "velocity", "action_name": "joint_pos", "gait_clock": a.gait_clock}),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "pelvis_height": TerminationTermCfg(
            func=loco_sim.pelvis_height_below,
            params={"body_name": robot.pelvis_body, "minimum_height": a.min_pelvis_height},
        ),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = LocoReplayRecorderCfg(
            params={
                "capacity": a.replay_capacity,
                "foot_geom_regex": robot.foot_geom_regex,
                "com_body_name": robot.com_body,
                "command_name": "velocity",
                "action_name": "joint_pos",
                "gait_clock": a.gait_clock,
                "foot_timers": a.foot_timers,
            }
        )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=a.num_envs,
            entities={"robot": robot.entity_cfg()},
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=robot.root_body,
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=QuietSimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )


def build_loco_heads(env: ManagerBasedRlEnv, robot_name: str, reward_cfg: LocoRewardCfg | None = None, min_pelvis_height: float | None = None) -> LocoHeads:
    """The analytic heads on the env's robot (soft limits, default pose, per-joint action scale)."""
    spec = get_robot(robot_name)
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    scale = term.scale
    action_scale = scale[0].clone() if isinstance(scale, torch.Tensor) else torch.full((len(spec.joint_names),), float(scale), device=env.device)
    fk = G1Kinematics(spec.load_model(), device=env.device)
    term_cfg = LocoTerminationCfg() if min_pelvis_height is None else LocoTerminationCfg(min_pelvis_height=min_pelvis_height)
    return LocoHeads(
        fk,
        spec,
        spec.load_model(),
        robot.data.soft_joint_pos_limits[0].clone(),
        robot.data.default_joint_pos[0].clone(),
        action_scale,
        reward_cfg=reward_cfg,
        term_cfg=term_cfg,
    )


def make_loco_env(a: LocoEnvCfgArgs, device: str = "cuda:0", reward_cfg: LocoRewardCfg | None = None) -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env = ManagerBasedRlEnv(make_loco_env_cfg(a), device=device)
    env.loco_heads = build_loco_heads(env, a.robot, reward_cfg, a.min_pelvis_height)  # read by envs/mdp/loco_sim.py
    return env
