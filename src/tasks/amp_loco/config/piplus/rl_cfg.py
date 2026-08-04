"""RL configuration for the Pi Plus AMP locomotion task (mirrors the G1 rl_cfg)."""

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
