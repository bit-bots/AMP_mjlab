"""RL configuration for the Pi Plus AMP locomotion task (mirrors the G1 rl_cfg)."""

import dataclasses
import os
from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from src.tasks.amp_loco.config.g1.rl_cfg import RslRlAmpRunnerCfg
from src.tasks.amp_loco.config.piplus.env_cfgs import AMP_BODY_NAMES, ANCHOR_NAME

_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
  "src", "assets", "motions", "piplus", "amp",
)

_NUM_JOINTS = 20


def piplus_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
    ),
    experiment_name="piplus_amp_locomotion",
    logger="wandb",
    wandb_project="piplus_amp",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=30_000,
    amp_reward_coef=0.1,
    amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
    amp_num_preload_transitions=200000,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * _NUM_JOINTS,
    amp_body_names=AMP_BODY_NAMES,
    amp_anchor_name=ANCHOR_NAME,
  )


def piplus_amp_kick_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Kick task variant: AMP discriminator trains on the Kick clips + one run
  trajectory (``KickAndRun/``, symlinks into ``Kick/`` and ``WalkandRun/`` --
  see src/assets/motions/piplus/amp/KickAndRun/).

  RslRlMotionLoader (rsl_rl/utils/motion_loader.py) walks ``amp_motion_files``
  recursively, so pointing it at the shared ``amp/`` root would pull in the walk
  clip too and dilute things further than intended -- KickAndRun/ is a
  deliberate, minimal mix: the 5 kick clips plus piplus_run1_subject2.npz, added
  because the kick clips' own approach-phase locomotion looked a bit off in
  practice ("style is a bit bad") -- a real running reference should firm up the
  gait between kicks without diluting the kick motion itself.  Also splits the
  experiment/wandb namespace so kick runs don't land in the locomotion
  project's dashboard.

  amp_reward_coef / amp_task_reward_lerp history (``Discriminator.
  predict_amp_reward`` combines the two additively -- ``(1-lerp)*amp_reward +
  lerp*task_reward`` -- with NO renormalization, while the AMP term alone is
  hard-clamped to ``[0, amp_reward_coef]``): 0.1/0.75 (locomotion defaults)
  left style numerically irrelevant (0.025/step vs task terms reaching
  several units) and the policy learned a task-optimal but styleless
  toe-poke/dribble. 1.0/0.3 (70% style) overshot the other way: standing
  still scored well against the Kick clips' calmer frames while dodging
  every task penalty, so the policy stopped kicking altogether. 0.3/0.5 (50%
  style) was tried twice -- both times the policy learned style (AMP-like
  motion) without learning the task (approach/kick), so it was dropped to
  0.25/0.7 (~30% style), raised slightly to 0.25/0.6 (~40% style),
  amp_reward_coef lowered a bit further to 0.2/0.6 (~32% style), lerp raised
  back to 0.2/0.75 (~20% style), amp_reward_coef lowered again to 0.15/0.75
  (~15% style), 0.2/0.9 (~8% style), 0.2/0.85 (~12% style), 0.2/0.8
  (~16% style), 0.2/0.7 (~24% style), 0.2/0.6 (~32% style), amp_reward_coef
  lowered to 0.13/0.6 (~22% style), lerp lowered to 0.13/0.45 (~40% style),
  lerp lowered further to 0.13/0.35 (~48% style). 0.2/0.25 (stricter, ~2.5x the
  style-to-task ratio) learned the style but NOT the kick (kick_impact 0.12 vs 4.8
  at iteration 4000), so the middle ground 0.16/0.30 (~1.5x) is used now -- applied from the
  start of training (not a curriculum), unlike the torque/action-rate
  penalty bump below.
  """
  base = piplus_amp_ppo_runner_cfg()
  # init_std=2.0 (was 1.0): a wider initial action-noise distribution for
  # more exploration early in training.
  kick_actor = dataclasses.replace(
    base.actor,
    distribution_cfg={**base.actor.distribution_cfg, "init_std": 2.0},
  )
  return dataclasses.replace(
    base,
    actor=kick_actor,
    experiment_name="piplus_amp_kick",
    wandb_project="piplus_amp_kick",
    amp_motion_files=os.path.normpath(os.path.join(_MOTION_DATA_DIR, "KickAndRun")),
    amp_reward_coef=0.16,
    amp_task_reward_lerp=0.30,
    save_interval=500,
    max_iterations=30_000,  # was 100001; the kick policy converges (or breaks) well before
  )


@dataclass
class RslRlSymmetryCfg:
  """Symmetry settings that rsl_rl PPO/AMPPPO read from ``symmetry_cfg``.

  The mirror loss weight is 0.0 by default. Set it explicitly.
  """

  use_data_augmentation: bool = False
  use_mirror_loss: bool = False
  mirror_loss_coeff: float = 0.0
  data_augmentation_func: str = ""
  # None: the mirror loss is the MSE. A float: the mirror loss is the Huber
  # loss with this delta on each element (bounded gradient).
  mirror_loss_huber_delta: float | None = None
  # None: the mirror loss stays on. An int: the runner switches it off from this
  # iteration on. The data augmentation (mirrored samples) stays on.
  mirror_loss_off_iteration: int | None = None


@dataclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO algorithm cfg with the rsl_rl symmetry option (mjlab has no field)."""

  symmetry_cfg: RslRlSymmetryCfg | None = None
  # None: no limit. A float: AMPPPO skips a mini-batch if its total loss,
  # value loss or mirror loss is larger than this value or not finite.
  skip_batch_loss_threshold: float | None = None


@dataclass
class RslRlAmpSymmetricRunnerCfg(RslRlAmpRunnerCfg):
  algorithm: RslRlAmpPpoAlgorithmCfg = field(default_factory=RslRlAmpPpoAlgorithmCfg)


def piplus_amp_kick_ppo_symmetric_runner_cfg() -> RslRlAmpSymmetricRunnerCfg:
  """Kick runner with left/right symmetry in PPO (as the piplus_amp_mjlab
  "Flat-Amp-DA" task): mirrored samples go into each PPO mini-batch, and the
  mirror loss (weight 1.0) pulls actor(mirror(obs)) to mirror(actor(obs)).
  All other settings are the same as ``piplus_amp_kick_ppo_runner_cfg``. The
  AMP reference data (KickAndRun/) already has a mirrored copy of each clip.
  """
  base = piplus_amp_kick_ppo_runner_cfg()
  algorithm = RslRlAmpPpoAlgorithmCfg(
    **dataclasses.asdict(base.algorithm),
    symmetry_cfg=RslRlSymmetryCfg(
      use_data_augmentation=True,
      use_mirror_loss=True,
      mirror_loss_coeff=1.0,
      mirror_loss_off_iteration=10_000,
      data_augmentation_func=(
        "src.tasks.amp_loco.config.piplus.kick_symmetry:augment_symmetries_piplus_kick"
      ),
    ),
  )
  fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
  # Rollout diagnostics are on for this task only, to find the source of the
  # numerical blow-ups (see rsl_rl/utils/obs_diagnostics.py).
  fields.update(
    algorithm=algorithm,
    experiment_name="piplus_amp_kick_symmetric",
    obs_diagnostics=True,
  )
  return RslRlAmpSymmetricRunnerCfg(**fields)
