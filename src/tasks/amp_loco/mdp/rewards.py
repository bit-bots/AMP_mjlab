from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse, 
  yaw_quat, 
  quat_apply
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_delay_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Get delaying env mask from DelayedTerminationManager if installed."""
  tm = env.termination_manager
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  delay_counters = getattr(tm, "_delay_counters", None)
  if isinstance(delay_env_mask, torch.Tensor) and isinstance(delay_counters, torch.Tensor):
    return delay_env_mask & (delay_counters > 0)
  return None


def _apply_delay_env_reward_scaling(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return reward

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return reward

  scaled_reward = reward * delay_env_rew_ratio
  return torch.where(delay_env_mask, scaled_reward, reward)


def _apply_delay_env_reward_mask_only(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return torch.zeros_like(reward)

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return torch.zeros_like(reward)

  scaled_reward = reward * delay_env_rew_ratio
  masked_reward = torch.where(delay_env_mask, scaled_reward, torch.zeros_like(reward))
  return masked_reward

def track_anchor_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  std_scale: float | None = None,
  std_min: float = 0.15,
) -> torch.Tensor:
  """Reward for tracking the commanded anchor linear velocity.

  The commanded z velocity is assumed to be zero.

  A flat ``std`` (exp(-error^2/std^2)) saturates near 1.0 once tracking is
  "good enough" relative to std, leaving little gradient to keep improving --
  and gives the SAME absolute tolerance regardless of how fast the command
  is, so a slow/near-zero command is trivially "satisfied" without precise
  tracking. If ``std_scale`` is set, this switches to an effective std that
  scales with the commanded speed instead: ``eff_std = max(std_scale *
  |command_xy|, std_min)``, demanding proportionally tighter tracking at low
  commanded speeds than at high ones, while ``std_min`` floors it so a
  literally-zero command doesn't require unachievable exact stillness.
  ``std_scale`` is meant to be annealed down over training (see
  ``anneal_reward_param_linear``) to keep restoring gradient as the flat-std
  reward would otherwise saturate. When ``std_scale`` is None, behavior is
  unchanged (flat ``std``), so existing configs (e.g. G1) are unaffected.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  command_xyz_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
  command_xyz_w = quat_apply(
    yaw_quat(asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]),
    command_xyz_b,
  )
  lin_vel_error = torch.sum(torch.square(command_xyz_w[:,:3] - asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]), dim=1)
  if std_scale is not None:
    command_speed = torch.norm(command[:, :2], dim=-1)
    eff_std = torch.clamp(std_scale * command_speed, min=std_min)
  else:
    eff_std = std
  reward = torch.exp(-lin_vel_error / eff_std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  std_scale: float | None = None,
  std_min: float = 0.15,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.

  Same command-magnitude-scaled tolerance as ``track_anchor_linear_velocity``
  (see its docstring): if ``std_scale`` is set, ``eff_std = max(std_scale *
  |command_yaw_rate|, std_min)`` instead of a flat ``std``, so low-yaw-rate
  commands demand proportionally tighter tracking than high ones, and the
  reward keeps gradient instead of saturating once "good enough" at a fixed
  tolerance. ``std_scale`` is meant to be annealed down over training. When
  ``std_scale`` is None, behavior is unchanged (flat ``std``).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  anchor_ang_vel_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
  anchor_ang_z_vel_w = anchor_ang_vel_w[:, 2]
  command_ang_vel_w = command[:, 2]
  ang_vel_z_error = torch.square(command_ang_vel_w - anchor_ang_z_vel_w)

  anchor_ang_vel_b =  quat_apply_inverse(
    asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]],
    anchor_ang_vel_w,
  )
  ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)

  total_error = ang_vel_z_error + ang_vel_xy_error

  if std_scale is not None:
    eff_std = torch.clamp(std_scale * torch.abs(command_ang_vel_w), min=std_min)
  else:
    eff_std = std
  reward = torch.exp(-total_error / eff_std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)

def body_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[body_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids[0]]
  body_ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, body_cfg.body_ids[0]],
    body_ang_vel_w,
  )
  body_ang_vel_xy_b = body_ang_vel_b[:, :2]
  ang_vel_xy_error = torch.sum(torch.square(body_ang_vel_xy_b), dim=-1)

  reward = torch.exp(-ang_vel_xy_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)

def track_root_height(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded anchor height."""
  asset: Entity = env.scene[asset_cfg.name]

  desired_height = asset.data.default_root_state[:, 2]
  cur_root_height = asset.data.body_link_pos_w[:, 0, 2]
  height_error = torch.square(desired_height - cur_root_height)
  reward = torch.exp(-height_error / std**2)
  return _apply_delay_env_reward_mask_only(env, reward, mask_delay, delay_env_rew_ratio)

def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost

def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost

def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def accel_toward_command_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  close_speed: float = 0.15,
  close_bonus: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward accelerating TOWARD the commanded velocity, to fix a dead zone in
  ``track_anchor_linear_velocity``: its exp(-error^2/std^2) shape has near-zero
  gradient both very close to AND very far from the target -- so right after
  an aggressive command switch, the robot lands in the far-zero zone where
  "stand still" and "half-heartedly try" score almost the same, and standing
  is lower-risk (avoids the large is_terminated penalty), so that's what gets
  learned.

  SIGNED and symmetric: accelerating toward the command gives positive reward,
  accelerating away gives a matching PENALTY (not just zero) -- computed as
  the dot product of this step's velocity change with the unit vector from
  current velocity to the command, so it has real gradient everywhere, unlike
  the tracking reward's exp shape.

  Anti-farming: that raw dot product is only well-defined, and only worth
  paying out, while genuinely far from the target -- near it, the direction
  vector gets noisy and a policy could farm small aligned jitters for
  repeated reward instead of settling. Below ``close_speed`` error, this
  returns a flat ``close_bonus`` (deliberately >= the typical far-zone peak)
  regardless of the actual acceleration, so holding still and accurate is
  already at least as good as any jitter -- removing the incentive to keep
  moving once close.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  command_xyz_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
  command_xyz_w = quat_apply(
    yaw_quat(asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]),
    command_xyz_b,
  )
  cur_vel = asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]

  prev_vel = getattr(env, "_accel_reward_prev_vel", None)
  if prev_vel is None or prev_vel.shape != cur_vel.shape:
    prev_vel = cur_vel.clone()
  # First step of a fresh episode: last step's velocity belonged to the
  # PREVIOUS episode (or env construction) -- treat as no acceleration yet
  # rather than measuring a bogus reset-induced spike.
  just_reset = env.episode_length_buf == 0
  prev_vel = torch.where(just_reset.unsqueeze(-1), cur_vel, prev_vel)

  delta_v = cur_vel - prev_vel
  env._accel_reward_prev_vel = cur_vel.detach().clone()

  error_vec = command_xyz_w - cur_vel
  error_mag = torch.norm(error_vec, dim=-1)
  error_dir = error_vec / torch.clamp(error_mag, min=1e-6).unsqueeze(-1)

  far_reward = torch.sum(delta_v * error_dir, dim=-1)
  is_close = error_mag < close_speed
  return torch.where(is_close, torch.full_like(far_reward, close_bonus), far_reward)


def contact_sensor_touched(sensor: ContactSensor, force_threshold: float = 0.1) -> torch.Tensor:
  """Whether a contact registered at ANY substep within the current control
  step, not just the final one ``found`` samples -- a fast strike-and-
  separate can happen entirely within one step's decimation loop and be
  invisible to ``found`` alone (confirmed empirically on the kick task: ball
  visibly moves, ``found`` never shows touching that step).

  Uses ``force_history`` (populated when the sensor's ``fields`` includes
  "force" and ``history_length`` is set to at least the sim's decimation --
  see ``ContactSensorCfg``'s docstring) if available, falling back to plain
  ``found`` otherwise. Same fix ``self_collision_cost`` above already applies
  for its own force-magnitude count; this is the boolean-touch equivalent for
  any other contact check that currently only reads ``found`` (e.g.
  ``feet_slip``, ``soft_landing``) -- not wired into either of those here,
  since that would change this task's already-tuned reward behavior without
  being asked, but the fix is a drop-in swap if the same failure mode shows
  up in training.
  """
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=(1, 2))
  assert data.found is not None
  return (data.found > 0).any(dim=-1)


