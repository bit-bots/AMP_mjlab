"""Per-environment randomization of the Pi Plus leg actuator model.

The nominal model is in ``assets/robots/piplus/piplus_constants.py``. The
ranges are from the system identification of the real robot (handoff
SIM_RL_HANDOFF.md, section "Domain randomisation magnitudes"). The armature and
the friction loss are randomized by the ``dr.joint_armature`` and
``dr.joint_friction`` events of env_cfgs.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator.delayed_actuator import DelayedActuator
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.piplus import piplus_constants as pc
from src.assets.robots.piplus.leg_motor import LegMotorActuator

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _uniform(low: float, high: float, shape, device) -> torch.Tensor:
  return low + (high - low) * torch.rand(shape, device=device)


def randomize_leg_actuators(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  kp_scale: tuple[float, float] = (0.8, 1.05),
  kd_scale: tuple[float, float] = (0.7, 1.1),
  torque_scale: tuple[float, float] = (0.9, 1.1),
  torque_joint_scale: tuple[float, float] = (0.97, 1.03),
  speed_scale: tuple[float, float] = (0.9, 1.1),
  lag_range: tuple[int, int] = (pc.LEG_DELAY_LAG_MIN, pc.LEG_DELAY_LAG_MAX),
) -> None:
  """Sample the leg actuator model once per environment (startup event).

  * kp and kd: one scale per joint. The scale acts on the simulated motor only.
    The policy commands the same targets.
  * Torque limit (the peak and the braking limit, and the stall torque with
    them, so the curve keeps its shape): one scale per environment times a small
    scale per joint.
  * No-load speed (the knee speed follows it): one scale per environment, as the
    battery voltage is the same for all joints.
  * Command delay: one lag in physics steps per environment for all leg joints.
    The delay buffer sets its lag to 0 at every reset, so ``apply_leg_lags``
    (reset event) sets the stored lag again.
  """
  robot = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  n = env_ids.shape[0]
  dev = env.device
  lags = torch.randint(lag_range[0], lag_range[1] + 1, (n,), device=dev, dtype=torch.long)
  if not hasattr(env, "leg_lags"):
    env.leg_lags = torch.zeros(env.num_envs, device=dev, dtype=torch.long)
  env.leg_lags[env_ids] = lags
  for actuator in robot.actuators:
    base = actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
    if not isinstance(base, LegMotorActuator):
      continue
    joints = base.stiffness.shape[1]
    base.set_gains(
      env_ids,
      kp=pc.LEG_KP * _uniform(*kp_scale, (n, joints), dev),
      kd=pc.LEG_KD * _uniform(*kd_scale, (n, joints), dev),
    )
    torque = _uniform(*torque_scale, (n, 1), dev) * _uniform(
      *torque_joint_scale, (n, joints), dev
    )
    base.force_limit[env_ids] = pc.LEG_EFFORT_LIMIT * torque
    base.saturation_effort[env_ids] = pc.LEG_STALL_TORQUE * torque
    base.velocity_limit_motor[env_ids] = pc.LEG_NO_LOAD_SPEED * _uniform(
      *speed_scale, (n, 1), dev
    )
    if isinstance(actuator, DelayedActuator):
      actuator.set_lags(lags, env_ids)


def apply_leg_lags(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Set the stored command delay of the legs again after a reset (reset event).

  The delay buffer sets its lag to 0 at every reset. ``randomize_leg_actuators``
  draws the lag once per environment.
  """
  assert hasattr(env, "leg_lags"), "randomize_leg_actuators must run first (startup event)"
  robot = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  for actuator in robot.actuators:
    base = actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
    if isinstance(base, LegMotorActuator) and isinstance(actuator, DelayedActuator):
      actuator.set_lags(env.leg_lags[env_ids], env_ids)
