from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .ball_stop_env_cfg import piplus_amp_ball_stop_env_cfg
from .env_cfgs import piplus_amp_flat_env_cfg, piplus_amp_rough_env_cfg
from .kick_env_cfg import piplus_amp_kick_env_cfg
from .rl_cfg import (
  piplus_amp_ball_stop_ppo_runner_cfg,
  piplus_amp_kick_ppo_runner_cfg,
  piplus_amp_kick_ppo_symmetric_runner_cfg,
  piplus_amp_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="PiPlus-AMP-Rough",
  env_cfg=piplus_amp_rough_env_cfg(),
  play_env_cfg=piplus_amp_rough_env_cfg(play=True),
  rl_cfg=piplus_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="PiPlus-AMP-Flat",
  env_cfg=piplus_amp_flat_env_cfg(),
  play_env_cfg=piplus_amp_flat_env_cfg(play=True),
  rl_cfg=piplus_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="PiPlus-AMP-Kick",
  env_cfg=piplus_amp_kick_env_cfg(),
  play_env_cfg=piplus_amp_kick_env_cfg(play=True),
  rl_cfg=piplus_amp_kick_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# Same env as PiPlus-AMP-Kick. PPO adds left/right mirrored samples and a
# mirror loss (see kick_symmetry.py), so that the policy kicks with both feet.
register_mjlab_task(
  task_id="PiPlus-AMP-Kick-DA",
  env_cfg=piplus_amp_kick_env_cfg(),
  play_env_cfg=piplus_amp_kick_env_cfg(play=True),
  rl_cfg=piplus_amp_kick_ppo_symmetric_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# The ball is rolled at the robot; the robot should stop it (AMP style reward
# + task rewards, see ball_stop_env_cfg.py).
register_mjlab_task(
  task_id="PiPlus-AMP-BallStop",
  env_cfg=piplus_amp_ball_stop_env_cfg(),
  play_env_cfg=piplus_amp_ball_stop_env_cfg(play=True),
  rl_cfg=piplus_amp_ball_stop_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
