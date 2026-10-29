"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import events as sm_events
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.envs.mdp.ramp import RampedUniformNoiseCfg, RampState
from supermimic.robots import get_robot
from supermimic.robots.g1 import G1


class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


# Booster_T2 v4's robot-side randomisation (config/t2_flat_v4_config.py, 2026-10-29): Booster's official set on the
# sysID actuator. The ramped items (noise x1 -> x2, gains, IMU offsets) reach full at the trainer's dr_ramp_iterations.
V4_JOINT_FRICTION_SCALE = (0.5, 1.5)  # x the measured table, a factor per joint and robot
V4_GAIN_SCALE = 0.1  # stiffness and damping +-10 %: the sysID bound (serial gains within 9 %, parallel factor 0.64-0.78)
V4_GROUND_FRICTION = (0.3, 1.6)
V4_LINK_MASS_SCALE = (0.95, 1.05)  # every link but the torso (its +-5 kg) and the wrists (their payload)
V4_TORSO_MASS_ADD_KG = (-5.0, 5.0)
V4_GYRO_BIAS_RAD_S = 0.02
V4_IMU_TILT_RAD = 0.035  # 2 deg of mounting error on the gravity direction
V4_NOISE_SCALE_TO = 2.0

# The paper's robot (tests and benchmarks of the G1 reproduction import these names).
TRACKED_BODIES = G1.tracked_bodies
ANCHOR_BODY = G1.anchor_body
ROOT_BODY = G1.root_body
FOOT_GEOM_REGEX = G1.foot_geom_regex


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    robot: str = "g1"  # supermimic.robots.get_robot
    wrist_payload: bool = (
        False  # per-episode point mass on each hand (Sharpa version), see mdp/events.py
    )
    wrist_payload_mass: tuple[float, float] = (0.0, 1.5)  # kg
    wrist_payload_offset: tuple[float, float] = (0.05, 0.20)  # m along the forearm
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = (
        False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    )
    anchor_ori_only: bool = False  # with anchor_obs: the reference root orientation only (6), no position (supermimic_vibed)
    dr_v4: bool = False  # Booster_T2 v4 robot-side randomisation (V4_* above); with domain_randomization
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs, ramp: RampState | None = None) -> ManagerBasedRlEnvCfg:
    robot = get_robot(a.robot)
    ramp = ramp if ramp is not None else RampState()
    # ---------------------------------------------------------------- observations
    if not a.obs_noise:
        noise = lambda lo, hi: None
    elif a.dr_v4:
        noise = lambda lo, hi: RampedUniformNoiseCfg(n_min=lo, n_max=hi, scale_to=V4_NOISE_SCALE_TO, ramp=ramp)
    else:
        noise = lambda lo, hi: Unoise(n_min=lo, n_max=hi)
    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)
        ),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(
            func=sm_obs.base_ang_vel_biased if a.dr_v4 else base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)
        ),
        "projected_gravity": ObservationTermCfg(
            func=sm_obs.projected_gravity_tilted if a.dr_v4 else base_mdp.projected_gravity,
            noise=noise(-0.05, 0.05),
        ),
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
                "future_frames": ObservationTermCfg(
                    func=sm_obs.goal_future_frames, params={"command_name": "motion"}
                ),
                **(
                    {
                        **(
                            {}
                            if a.anchor_ori_only
                            else {
                                "anchor_pos": ObservationTermCfg(
                                    func=track_mdp.motion_anchor_pos_b,
                                    params={"command_name": "motion"},
                                    noise=noise(-0.25, 0.25),
                                )
                            }
                        ),
                        "anchor_ori": ObservationTermCfg(
                            func=track_mdp.motion_anchor_ori_b,
                            params={"command_name": "motion"},
                            noise=noise(-0.05, 0.05),
                        ),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=robot.action_scale,
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    }
    vel_range = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
    }
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            robot=a.robot,
            anchor_body_name=robot.anchor_body,
            root_body_name=robot.root_body,
            body_names=robot.tracked_bodies,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
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
    if a.domain_randomization and a.dr_v4:
        events["foot_friction"].params["ranges"] = V4_GROUND_FRICTION
        events["joint_friction"] = EventTermCfg(
            mode="reset",
            func=dr.joint_friction,
            params={"asset_cfg": SceneEntityCfg("robot"), "operation": "scale", "ranges": V4_JOINT_FRICTION_SCALE},
        )
        events["actuator_gains"] = EventTermCfg(
            mode="reset",
            func=sm_events.pd_gains_ramped,
            params={"kp_scale": V4_GAIN_SCALE, "kd_scale": V4_GAIN_SCALE},
        )
        events["link_mass"] = EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(f"(?!{robot.com_body}$|.*_wrist_roll_link$).*",)),
                "operation": "scale",
                "ranges": V4_LINK_MASS_SCALE,
            },
        )
        events["torso_mass"] = EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(robot.com_body,)),
                "operation": "add",
                "ranges": V4_TORSO_MASS_ADD_KG,
            },
        )
        events["sensor_offsets"] = EventTermCfg(
            mode="reset",
            func=sm_events.sensor_offsets,
            params={"gyro_bias_rad_s": V4_GYRO_BIAS_RAD_S, "imu_tilt_rad": V4_IMU_TILT_RAD},
        )
    if a.wrist_payload:
        events["wrist_payload"] = EventTermCfg(
            mode="reset",
            func=sm_events.wrist_payload,
            params={
                "mass_range": tuple(a.wrist_payload_mass),
                "offset_range": tuple(a.wrist_payload_offset),
            },
        )
    if not a.domain_randomization:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
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

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(
            func=track_mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_global_root_ori": RewardTermCfg(
            func=track_mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_pos": RewardTermCfg(
            func=track_mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=track_mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=track_mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=track_mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.14},
        ),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(
            func=base_mdp.joint_pos_limits,
            weight=a.joint_limit_weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(
            func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True
        ),
        "root_height": TerminationTermCfg(
            func=sm_term.bad_root_height,
            params={"command_name": "motion", "threshold": a.root_height_threshold},
        ),
        "root_tilt": TerminationTermCfg(
            func=sm_term.bad_root_tilt,
            params={"command_name": "motion", "threshold": a.root_tilt_threshold},
        ),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(
            params={
                "capacity": a.replay_capacity,
                "foot_geom_regex": robot.foot_geom_regex,
                "com_body_name": robot.com_body,
            }
        )

    cfg = ManagerBasedRlEnvCfg(
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
            body_name=robot.anchor_body,
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=QuietSimulationCfg(
            nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)
        ),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    ramp = RampState()
    env = ManagerBasedRlEnv(make_env_cfg(a, ramp), device=device)
    env.dr_ramp = ramp  # the trainer drives it from the iteration; evaluation leaves it at full
    # the managers deep-copied their configs: point every ramped noise term at the env's ramp
    for group, names in env.observation_manager.active_terms.items():
        for name in names:
            noise_cfg = env.observation_manager.get_term_cfg(group, name).noise
            if isinstance(noise_cfg, RampedUniformNoiseCfg):
                noise_cfg.ramp = ramp
    return env
