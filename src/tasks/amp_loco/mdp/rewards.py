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
  quat_apply,
  subtract_frame_transforms,
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from src.assets.objects import BALL_RADIUS

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
) -> torch.Tensor:
  """Reward for tracking the commanded anchor linear velocity.

  The commanded z velocity is assumed to be zero.
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
  reward = torch.exp(-lin_vel_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
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

  reward = torch.exp(-total_error / std**2)
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

def _offset_anchor_pos_w(
  robot: Entity, anchor_cfg: SceneEntityCfg, offset_b: tuple[float, float, float]
) -> torch.Tensor:
  """Anchor body position, shifted by a local-frame offset (e.g. off-center
  toward the kicking foot instead of the body origin)."""
  pos_w = robot.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
  if offset_b == (0.0, 0.0, 0.0):
    return pos_w
  quat_w = robot.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
  offset = torch.tensor(offset_b, device=pos_w.device, dtype=pos_w.dtype)
  return pos_w + quat_apply(quat_w, offset.expand(pos_w.shape[0], 3))


def move_toward_object(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  offset_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
  alignment_power: float = 1.0,
) -> torch.Tensor:
  """World-frame, heading-independent reward for the anchor's speed toward a
  free-body object.

  reward = ``speed * sign(cos(angle)) * |cos(angle)| ** alignment_power``,
  where ``angle`` is between the anchor's own linear velocity and the unit
  direction from the (locally-offset) anchor position to the object.
  ``alignment_power`` > 1 narrows the reward around the correct heading: a
  bare cosine (power=1) gives most of its max credit even fairly off-angle
  (e.g. cos(30deg)=0.87), while cubing it (power=3) drops that same
  30deg-off motion to ~0.65 and 60deg-off to ~0.13, so only genuinely
  well-aimed motion is rewarded much. Moving AWAY from the object
  (angle > 90deg) is penalized, not just zero -- the sign is preserved
  through the power regardless of whether ``alignment_power`` is odd or
  even. No distance scaling (would vanish right at the object -- see git
  history) and no upper cap on speed.

  Deliberately uses the anchor body's own linear velocity, not the offset
  point's true (``omega x r``-corrected) velocity: pure rotation in place
  (zero body translation) then scores zero regardless of how far the local
  offset sweeps through space, so pivoting/circling without actually
  translating the body isn't rewarded (confirmed empirically: circling
  scored a high reward when the offset-corrected velocity was used instead).

  ``offset_b``: local-frame offset from the anchor body's origin (e.g. toward
  the kicking foot rather than the torso center), used for the tracked
  position only.
  """
  robot: Entity = env.scene[anchor_cfg.name]
  obj: Entity = env.scene[object_cfg.name]

  anchor_pos = _offset_anchor_pos_w(robot, anchor_cfg, offset_b)[:, :2]
  anchor_vel = robot.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :2]
  to_obj = obj.data.root_link_pos_w[:, :2] - anchor_pos
  direction = to_obj / (torch.norm(to_obj, dim=-1, keepdim=True) + 1e-6)
  speed = torch.norm(anchor_vel, dim=-1)
  cos_align = torch.clamp(torch.sum(anchor_vel * direction, dim=-1) / (speed + 1e-6), -1.0, 1.0)
  signed_align = torch.sign(cos_align) * torch.abs(cos_align) ** alignment_power
  return speed * signed_align


def torso_orient_to_object(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  offset_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
  """Reward the anchor body's forward direction pointing at a free-body object.

  reward = exp(-|angle to object in the anchor's local frame|), 1.0 when the
  object is dead ahead, decaying smoothly as it swings to the side/behind.
  Mirrors mjxperiment kick.py's ``_reward_orient_to_ball``.

  ``offset_b``: see ``move_toward_object`` -- shifts which point's-eye-view the
  angle is computed from, without changing the anchor's own orientation axes.
  """
  robot: Entity = env.scene[anchor_cfg.name]
  obj: Entity = env.scene[object_cfg.name]

  anchor_pos_w = _offset_anchor_pos_w(robot, anchor_cfg, offset_b)
  anchor_quat_w = robot.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w, anchor_quat_w, obj.data.root_link_pos_w, obj.data.root_link_quat_w
  )
  angle = torch.atan2(pos_b[:, 1], pos_b[:, 0])
  return torch.exp(-torch.abs(angle))


def foot_slip_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in ground contact).

  Unconditional version of ``mdp.feet_slip`` (mjlab.tasks.velocity): that one
  gates the penalty on a velocity command's magnitude, but the kick task has no
  velocity command, so this just penalizes slip whenever a tracked foot site is
  in contact, regardless of intent.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
  return torch.sum(vel_xy_norm_sq * in_contact, dim=1)


def foot_short_air_time_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_air_time: float = 0.25,
) -> torch.Tensor:
  """Penalize a foot for landing after too short a swing (a shuffling gait),
  in place of ``mjlab.tasks.velocity.mdp.feet_air_time``'s medium-range
  reward -- this only fires on landing, and only for the deficit below
  ``min_air_time``, not the full [min, max] band that reward used.

  penalty = ``sum_over_feet(clamp(min_air_time - last_air_time, min=0))``,
  evaluated the instant each foot lands (``ContactSensor.
  compute_first_contact``) -- i.e. checked against the JUST-COMPLETED swing
  duration, not the still-growing ``current_air_time`` mid-swing (which
  would otherwise wrongly penalize the first ``min_air_time`` seconds of
  every swing, including ones that turn out fine).

  Requires the sensor to have ``track_air_time=True``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.last_air_time is not None
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  shortfall = torch.clamp(min_air_time - data.last_air_time, min=0.0)
  return torch.sum(shortfall * first_contact.float(), dim=1)


def object_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Binary reward while the named contact sensor registers a contact."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).any(dim=-1).float()


def first_object_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  state_attr: str = "contact_reward_claimed",
  ready_attr: str | None = None,
) -> torch.Tensor:
  """One-shot reward on the first step the sensor registers contact, not again
  until the tracked object (ball) is repositioned.

  ``object_contact_reward`` pays out every step contact holds, which let the
  kick-task policy farm reward by just resting the right foot on the ball
  instead of kicking it. This claims the reward once per object "life": the
  claim flag (``state_attr`` on ``env``) is cleared by ``_place_ball_near_robot``
  (mdp/events.py) whenever the ball's position is reset -- either the episode
  reset or the post-kick ``kick_contact_cycle`` respawn -- so a fresh touch
  after the next reset earns the reward again.

  ``ready_attr`` is an optional extra gate: if set to an attribute name that
  exists on ``env`` and is truthy for a given env, contact only counts while
  that flag is set. Unused by the kick task (no such gating attribute is
  set there anymore); kept for other tasks that might want it.
  """
  claimed = getattr(env, state_attr, None)
  if claimed is None:
    claimed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  touching = (sensor.data.found > 0).any(dim=-1)
  if ready_attr is not None:
    ready = getattr(env, ready_attr, None)
    if ready is not None:
      touching = touching & ready
  new_touch = touching & ~claimed
  setattr(env, state_attr, claimed | touching)
  return new_touch.float()


def first_ball_kick_reward(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  speed_threshold: float = 0.2,
  state_attr: str = "contact_reward_claimed",
  ready_attr: str | None = None,
) -> torch.Tensor:
  """One-shot reward the first step the ball's speed exceeds
  ``speed_threshold`` (i.e. it was struck), not again until the ball is
  repositioned.

  Uses ball velocity instead of a foot/ball contact sensor: a fast strike
  can separate within a single RL step's physics substeps, which a contact
  sensor (sampled once per RL step) can silently miss even though the kick
  was genuine. Ball velocity persists past the moment of contact, so a real
  kick is never missed here regardless of how quickly it separates.

  Same one-shot-per-life semantics as ``first_object_contact_reward``: the
  claim flag (``state_attr`` on ``env``) is cleared by
  ``_place_ball_near_robot`` (mdp/events.py) whenever the ball's position is
  reset, so a fresh kick after the next reset earns the reward again.

  ``ready_attr`` is an optional extra gate, see ``first_object_contact_reward``;
  unused by the kick task.
  """
  claimed = getattr(env, state_attr, None)
  if claimed is None:
    claimed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  ball: Entity = env.scene[ball_cfg.name]
  ball_speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)
  kicked = ball_speed > speed_threshold
  if ready_attr is not None:
    ready = getattr(env, ready_attr, None)
    if ready is not None:
      kicked = kicked & ready
  new_kick = kicked & ~claimed
  setattr(env, state_attr, claimed | kicked)
  return new_kick.float()


def standing_still_penalty(
  env: ManagerBasedRlEnv,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  vel_threshold: float = 0.15,
  state_attr: str = "contact_reward_claimed",
) -> torch.Tensor:
  """Flat penalty for near-zero root speed before the ball has been touched
  (in the current ball "life" -- see ``first_object_contact_reward``'s
  ``contact_reward_claimed``, cleared on every ball reset, so this re-applies
  after every kick too, not just at episode start).

  Flat (not proportional to how slow) so it can't be cheaply "paid off" by
  drifting just above zero -- either the robot is genuinely moving toward the
  ball or it eats the full penalty. Counters the standing-still local optimum
  (avoiding fall/wrong-foot penalties by simply not doing anything) that kept
  showing up even after several rounds of reward-scale fixes.
  """
  robot: Entity = env.scene[anchor_cfg.name]
  speed = torch.norm(robot.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :2], dim=-1)
  claimed = getattr(env, state_attr, None)
  not_touched_yet = torch.ones_like(speed, dtype=torch.bool) if claimed is None else ~claimed
  standing_still = speed < vel_threshold
  return (standing_still & not_touched_yet).float()


def foot_medial_alignment(
  foot_pos_xy: torch.Tensor,
  foot_quat: torch.Tensor,
  ball_pos_xy: torch.Tensor,
  medial_sign: float,
) -> torch.Tensor:
  """Cosine alignment in [-1, 1] between a foot's medial (inside) axis and the
  foot->ball direction: +1 = inside of foot points straight at the ball (ideal
  side-foot kick), 0 = front/back of foot, -1 = outside/lateral edge.

  ``medial_sign``: whether the body's local +Y axis is the medial (inside)
  direction for this foot. Empirically, piplus.xml's l_/r_ankle_roll_link share
  local +Y == world +Y at zero joint angles, but the left foot sits on the +Y
  side and the right on the -Y side -- so +Y is medial for the right foot
  (medial_sign=+1) and lateral for the left (medial_sign=-1).

  Shared by ``foot_medial_alignment_penalty`` (this module) and the
  ``kick_contact_cycle`` event (mdp/events.py), which freezes this value at the
  moment of contact as ``kick_impact_reward``'s style multiplier.
  """
  to_ball = ball_pos_xy - foot_pos_xy
  direction = to_ball / (torch.norm(to_ball, dim=-1, keepdim=True) + 1e-6)
  local_y = torch.tensor(
    [0.0, medial_sign, 0.0], device=foot_quat.device
  ).expand(foot_quat.shape[0], 3)
  medial_w = quat_apply(foot_quat, local_y)
  medial_xy = medial_w[:, :2]
  medial_xy = medial_xy / (torch.norm(medial_xy, dim=-1, keepdim=True) + 1e-6)
  return torch.sum(medial_xy * direction, dim=-1)


def _kick_dir_factor(
  ball_vel_xy: torch.Tensor, kick_dir: torch.Tensor, sigma: float
) -> torch.Tensor:
  """Signed direction quality in [-1, 1]: +1 exact match with the target
  direction, 0 at the angular-tolerance boundary, -1 opposite. Mirrors
  mjxperiment kick.py's ``_kick_dir_factor`` exactly: a Gaussian centered on
  the target direction, remapped from [0,1] to [-1,1] so kicks worse than the
  tolerance actively score negative rather than merely "less positive".
  """
  speed = torch.norm(ball_vel_xy, dim=-1)
  direction = ball_vel_xy / (speed.unsqueeze(-1) + 1e-6)
  cos_angle = torch.clamp(torch.sum(direction * kick_dir, dim=-1), -1.0, 1.0)
  angle_err = torch.acos(cos_angle)
  f_dir = torch.exp(-0.5 * torch.square(angle_err / sigma))
  return (f_dir - 0.5) * 2.0


def kick_impact_reward(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  vel_weight: float = 1.0,
  height_weight: float = 1.0,
  saturation_scale: float = 5.0,
  dir_sigma: float = 0.5,
) -> torch.Tensor:
  """Reward ball speed + height while a post-contact reward window is open,
  scaled by how well the foot was aligned with the ball at the moment of
  contact (so a proper side-foot kick earns much more than a toe-poke/front
  kick for the same ball speed/height) AND by how well the ball's resulting
  direction matches the target (``env.kick_dir_world``) -- a kick outside the
  direction tolerance doesn't just earn less, it goes negative (see
  ``_kick_dir_factor``), so direction can't be ignored in favor of raw power.

  Raw ``vel_weight*speed + height_weight*height`` is unbounded and was
  swamping the AMP style term (hard-capped to ``[0, amp_reward_coef]``) in the
  overall PPO reward -- the discriminator's contribution became numerically
  irrelevant regardless of its lerp weight. ``tanh(raw / saturation_scale)``
  smoothly bounds this term to [0, 1) (near-linear for small raw, saturating
  as it grows), so with the reward term's outer ``weight`` set to 2.0 the
  hardest possible kick still only reaches ~2 -- the same order of magnitude as
  the other reward terms instead of dwarfing them.

  ``dir_sigma`` is meant to be narrowed over training via a curriculum
  (``anneal_reward_param_linear``, mdp/curriculums.py) alongside
  ``kick_direction_reward``'s own sigma -- generous early (a wide "doesn't
  count against you" cone) so the robot can discover kicking at all, then
  tightening to demand real directional accuracy for full credit.

  The window (``env.kick_timer``, steps remaining) and the frozen style
  multiplier (``env.kick_style`` in [0, 1], see ``foot_medial_alignment``) are
  owned and set by the ``kick_contact_cycle`` event (mdp/events.py) -- opened on
  right-foot/ball contact, held for a few seconds so the robot can "farm" reward
  from the kick's aftermath, then the event resets the ball. This function just
  reads that shared per-env state; it holds no state of its own.
  """
  ball: Entity = env.scene[ball_cfg.name]
  timer = getattr(env, "kick_timer", None)
  ball_vel_xy = ball.data.root_link_lin_vel_w[:, :2]
  ball_speed = torch.norm(ball_vel_xy, dim=-1)
  if timer is None:
    return torch.zeros_like(ball_speed)
  ball_height = torch.clamp(ball.data.root_link_pos_w[:, 2] - BALL_RADIUS, min=0.0)
  active = timer > 0
  style = getattr(env, "kick_style", None)
  style = torch.ones_like(ball_speed) if style is None else style
  raw = vel_weight * ball_speed + height_weight * ball_height
  bounded = torch.tanh(raw / saturation_scale)

  kick_dir = getattr(env, "kick_dir_world", None)
  dir_factor = (
    torch.ones_like(ball_speed)
    if kick_dir is None
    else _kick_dir_factor(ball_vel_xy, kick_dir, dir_sigma)
  )
  return torch.where(active, style * bounded * dir_factor, torch.zeros_like(ball_speed))


def kick_direction_reward(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  sigma: float = 0.5,
  min_ball_speed: float = 0.1,
) -> torch.Tensor:
  """Reward/penalize the ball's post-kick velocity direction against a
  randomly sampled per-env target (``env.kick_dir_world``, a world-frame unit
  vector resampled every time the ball is repositioned -- see
  ``_sample_kick_dir``/``_place_ball_near_robot`` in mdp/events.py).

  Signed via ``_kick_dir_factor``: +1 for an exact match, 0 at the tolerance
  boundary, down to -1 for the opposite direction -- a kick outside the
  tolerance is an active penalty, not just a missed reward. Gated to the same
  post-contact window as ``kick_impact_reward`` and to the ball actually
  moving (a stationary ball has no meaningful "direction"). ``sigma`` (the
  angular tolerance, radians) is meant to be narrowed over training via a
  curriculum (``anneal_reward_param_linear``, mdp/curriculums.py).
  """
  ball: Entity = env.scene[ball_cfg.name]
  timer = getattr(env, "kick_timer", None)
  kick_dir = getattr(env, "kick_dir_world", None)
  ball_vel_xy = ball.data.root_link_lin_vel_w[:, :2]
  ball_speed = torch.norm(ball_vel_xy, dim=-1)
  if timer is None or kick_dir is None:
    return torch.zeros_like(ball_speed)
  dir_factor = _kick_dir_factor(ball_vel_xy, kick_dir, sigma)
  active = (timer > 0) & (ball_speed > min_ball_speed)
  return torch.where(active, dir_factor, torch.zeros_like(dir_factor))


def foot_medial_alignment_penalty(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  foot_cfg: SceneEntityCfg,
  medial_sign: float,
  close_dist: float = 0.5,
  ready_attr: str | None = None,
) -> torch.Tensor:
  """Penalize the foot's inside (medial) edge not facing the ball when close.

  Targets the "kicked with the front of the foot instead of the side" failure
  mode: only active within ``close_dist`` of the ball, penalty = 1 - cos(angle
  between the foot's medial axis and the foot->ball direction), so 0 when the
  inside of the foot points straight at the ball and up to 2 when it faces
  directly away.

  ``ready_attr`` is an optional extra gate: if set to an attribute name that
  exists on ``env`` and is truthy for a given env, the penalty only applies
  while that flag is set. Unused by the kick task (no such gating attribute
  is set there anymore, so this fires whenever close to the ball regardless
  of what the robot is doing).
  """
  robot: Entity = env.scene[foot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]

  foot_pos = robot.data.body_link_pos_w[:, foot_cfg.body_ids[0], :2]
  foot_quat = robot.data.body_link_quat_w[:, foot_cfg.body_ids[0]]
  dist = torch.norm(ball.data.root_link_pos_w[:, :2] - foot_pos, dim=-1)

  align = foot_medial_alignment(foot_pos, foot_quat, ball.data.root_link_pos_w[:, :2], medial_sign)
  misalign = 1.0 - torch.clamp(align, -1.0, 1.0)
  active = dist < close_dist
  if ready_attr is not None:
    ready = getattr(env, ready_attr, None)
    if ready is not None:
      active = active & ready
  return misalign * active.float()


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


