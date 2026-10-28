"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.envs.mdp.loco_rewards import LocoRewardCfg
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.loco_nets import LocoCriticCfg, LocoPolicyCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = (
        None  # packed compact library (.pt from build_library.py); overrides npz_dir
    )


@dataclass
class EnvCfg:
    robot: str = "g1"  # g1 | t2 | t2_sharpa (supermimic.robots)
    wrist_payload: bool = False  # Sharpa version: per-episode hand payload 0-1.5 kg at 5-20 cm
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False
    anchor_ori_only: bool = False  # with anchor_obs: reference root orientation only (6-d, IMU-observable), no position


@dataclass
class LocoCfg:
    """The velocity-command locomotion task (``--task loco``); env size / noise / DR come from ``env``."""

    resampling_time_range: tuple[float, float] = (3.0, 6.0)
    rel_standing_envs: float = 0.4
    curriculum_start: int = 50_000  # iterations over which the torso command ranges widen (Isaac: 20k-60k PPO it)
    curriculum_end: int = 250_000
    rsi_dir: Path | None = None  # reference-state initialisation: start rsi_prob of the episodes from these clips
    rsi_clips: list[str] | None = None  # clip names (None = every npz in rsi_dir)
    rsi_prob: float = 0.5
    gait_clock: bool = True  # goal = command + gait clock; the swing reference (reward.w_feet_phase) needs it
    imag_switch_prob: float = 0.0  # share of imagined rollouts whose command switches once mid-horizon (t2_loco_v9)
    imag_action_noise: float = 0.0  # std of the executed-action noise in imagination (t2_loco_v10)
    imag_action_noise_decay: tuple[int, int] = (0, 0)  # iterations over which it fades linearly to 0 ((0, 0): constant)
    reward: LocoRewardCfg = field(default_factory=LocoRewardCfg)
    policy: LocoPolicyCfg = field(default_factory=LocoPolicyCfg)
    critic: LocoCriticCfg = field(default_factory=LocoCriticCfg)


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    lr_warmup_iterations: int = 0  # linear ramp of the learning rate from 0; for fine-tuning a converged checkpoint
    wm_only_iterations: int = 0  # neither actor nor critic is updated before this iteration (world-model warm-up)
    policy_freeze_iterations: int = 0  # actor not updated before this iteration (world model / critic adapt first when fine-tuning on changed dynamics)
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    task: str = "tracking"  # tracking (motion imitation, the paper) | loco (velocity-command walking, Booster_T2)
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    loco: LocoCfg = field(default_factory=LocoCfg)
    resume: Path | None = None  # continue a run: weights, optimizers, iteration, sampling state
    finetune: Path | None = (
        None  # start a new run from a checkpoint's weights, Adam moments and sampler state (fresh schedule); xi may be wider
    )
