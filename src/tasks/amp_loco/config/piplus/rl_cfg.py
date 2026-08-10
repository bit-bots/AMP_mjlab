"""RL configuration for the Pi Plus AMP locomotion task (mirrors the G1 rl_cfg)."""

import dataclasses
import os

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
    max_iterations=100001,
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

  amp_reward_coef / amp_task_reward_lerp bumped again (was 0.1/0.75, the
  locomotion defaults): ``Discriminator.predict_amp_reward`` combines the two
  additively -- ``(1-lerp)*amp_reward + lerp*task_reward`` -- with NO
  renormalization, while the AMP term alone is hard-clamped to
  ``[0, amp_reward_coef]``, so 0.1/0.75 left style numerically irrelevant
  (0.025/step vs task terms reaching several units) and the policy learned a
  task-optimal but styleless toe-poke/dribble. An earlier, more aggressive bump
  (1.0/0.3, 70% style) overshot into a different failure: standing still scored
  well against the Kick clips' calmer frames while dodging every task penalty,
  so the policy stopped kicking altogether. Trying the in-between setting
  (0.3/0.5, 50% style) that was flagged but never actually run last time --
  now combined with the tightened task rewards (one-shot contact reward, style-
  gated kick_impact, bumped move_to_ball) that should make idling far less
  attractive regardless of how much style contributes.
  """
  base = piplus_amp_ppo_runner_cfg()
  return dataclasses.replace(
    base,
    experiment_name="piplus_amp_kick",
    wandb_project="piplus_amp_kick",
    amp_motion_files=os.path.normpath(os.path.join(_MOTION_DATA_DIR, "KickAndRun")),
    amp_reward_coef=0.3,
    amp_task_reward_lerp=0.5,
  )
