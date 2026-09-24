from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
  quat_apply_inverse,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _apply_imu_bias(env: "ManagerBasedRlEnv", vec: torch.Tensor) -> torch.Tensor:
  """Left-multiply a (num_envs, 3) body-frame vector by the per-env IMU mounting
  rotation set by ``randomize_imu_mounting_bias`` (identity if unset)."""
  R = getattr(env, "imu_bias_rot", None)
  if R is None:
    return vec
  return torch.bmm(R, vec.unsqueeze(-1)).squeeze(-1)


def imu_gyro_biased(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """Gyro (base angular velocity) sensor reading with the IMU mounting bias applied.

  Actor-only variant of ``builtin_sensor``; the clean critic keeps the unbiased
  reading. Additive gyro noise is applied on top by the observation manager.
  """
  sensor = env.scene[sensor_name]
  return _apply_imu_bias(env, sensor.data)


def imu_projected_gravity_biased(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Projected-gravity obs with the IMU mounting bias applied (actor-only)."""
  asset: Entity = env.scene[asset_cfg.name]
  return _apply_imu_bias(env, asset.data.projected_gravity_b)


def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)

def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_body_lin_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_lin_vel_w.shape[1]

    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)

def object_pos_b(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Free-body object's root position relative to the anchor body (e.g. ball in
    torso frame), used in place of a velocity command for goal-directed tasks."""
    anchor: Entity = env.scene[anchor_cfg.name]
    obj: Entity = env.scene[object_cfg.name]

    anchor_pos_w = anchor.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = anchor.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]

    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w, anchor_quat_w, obj.data.root_link_pos_w, obj.data.root_link_quat_w,
    )
    return pos_b


def object_pos_b_noisy(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    std: float = 0.07,
    std_post_kick: float = 1.0,
    timer_attr: str = "kick_timer",
    range_noise_std: float = 0.2,
    bearing_noise_std_deg: float = 5.0,
) -> torch.Tensor:
    """Actor-only noisy variant of ``object_pos_b``: much noisier during the
    post-kick phase (``timer_attr`` on ``env`` > 0, see ``kick_state_obs``)
    than during the normal approach -- models the real robot's vision losing
    good track of the ball right after a fast kick (motion blur / high ball
    speed), not just a flat perception-noise floor throughout.

    On top of that isotropic Cartesian noise, also applies range/bearing
    noise in the anchor's local xy-plane: the ball's true (x, y) there is
    treated as polar coordinates (range ``r = norm(x, y)``, bearing
    ``theta = atan2(y, x)`` -- the direction FROM the robot TO the ball, not
    ``env.kick_dir_world``, which is the commanded direction to kick it in
    and is unrelated to this noise). ``range_noise_std`` is relative (e.g.
    0.2 = 20% of the current range, so error grows with distance, like a real
    range sensor); ``bearing_noise_std_deg`` is an absolute angular error in
    degrees (like a real bearing/angle-of-view estimate, roughly
    distance-independent). This model is applied to the true position BEFORE
    the isotropic noise below, so the two stack as independent error sources
    rather than one replacing the other.

    Noise is applied here (per-env-conditional std), not via the generic
    ``ObservationTermCfg.noise`` pipeline (a single fixed ``GaussianNoiseCfg``
    can't vary by state); ``ObservationTermCfg.delay_max_lag`` still applies
    downstream as normal.
    """
    pos_b = object_pos_b(env, object_cfg, anchor_cfg)

    xy = pos_b[:, :2]
    r = torch.norm(xy, dim=-1)
    theta = torch.atan2(xy[:, 1], xy[:, 0])
    r_noisy = r * (1.0 + torch.randn_like(r) * range_noise_std)
    theta_noisy = theta + torch.randn_like(theta) * math.radians(bearing_noise_std_deg)
    xy_noisy = torch.stack(
        [r_noisy * torch.cos(theta_noisy), r_noisy * torch.sin(theta_noisy)], dim=-1
    )
    pos_b = torch.cat([xy_noisy, pos_b[:, 2:]], dim=-1)

    timer = getattr(env, timer_attr, None)
    if timer is None:
        post_kick = torch.zeros(pos_b.shape[0], dtype=torch.bool, device=pos_b.device)
    else:
        post_kick = timer > 0
    sigma = torch.where(
        post_kick,
        torch.full_like(pos_b[:, 0], std_post_kick),
        torch.full_like(pos_b[:, 0], std),
    )
    return pos_b + torch.randn_like(pos_b) * sigma.unsqueeze(-1)


def object_lin_vel_b(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Free-body object's linear velocity expressed in the anchor body's frame."""
    anchor: Entity = env.scene[anchor_cfg.name]
    obj: Entity = env.scene[object_cfg.name]

    anchor_quat_w = anchor.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    return quat_apply_inverse(anchor_quat_w, obj.data.root_link_lin_vel_w)


def kick_dir_heading_b(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target kick direction (``env.kick_dir_world``, a world-frame unit
    vector in the ground plane -- see ``kick_contact_cycle``, mdp/events.py)
    rotated into the robot base link's yaw frame: a 2D vector ``(x, y)``
    where +x is the base's current forward heading and +y is 90deg to its
    left. Yaw-only (not the base's full 3D orientation), since both the
    target direction and this representation live entirely in the ground
    plane.

    Without this, the policy has no way to know which direction is
    currently being asked for -- ``kick_direction_reward``/
    ``kick_impact_reward`` score against ``kick_dir_world``, but nothing
    exposed it as an observation.
    """
    robot: Entity = env.scene[robot_cfg.name]
    kick_dir = getattr(env, "kick_dir_world", None)
    if kick_dir is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    kick_dir = kick_dir / (torch.norm(kick_dir, dim=-1, keepdim=True) + 1e-6)
    yaw = robot.data.heading_w
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    x_b = cos_yaw * kick_dir[:, 0] + sin_yaw * kick_dir[:, 1]
    y_b = -sin_yaw * kick_dir[:, 0] + cos_yaw * kick_dir[:, 1]
    return torch.stack([x_b, y_b], dim=-1)


def kick_dir_heading_b_noisy(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std_rad: float = 0.0873,  # ~5deg
) -> torch.Tensor:
    """Actor-only noisy variant of ``kick_dir_heading_b``: Gaussian noise
    added directly to the underlying yaw ANGLE, not the raw (x, y)
    components -- component-wise additive noise on a unit vector (mjlab's
    generic ``GaussianNoiseCfg`` pipeline, used for e.g. ``ball_pos_b``)
    would corrupt the unit-norm representation, whereas perturbing the angle
    and re-encoding as ``(cos, sin)`` models a noisy heading sensor while
    keeping the output a proper unit vector regardless of noise magnitude.

    Resampled every step (like mjlab's other per-step observation noise),
    not a fixed per-episode bias.
    """
    robot: Entity = env.scene[robot_cfg.name]
    kick_dir = getattr(env, "kick_dir_world", None)
    if kick_dir is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    angle_w = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
    angle_b = angle_w - robot.data.heading_w
    angle_b = angle_b + torch.randn_like(angle_b) * std_rad
    return torch.stack([torch.cos(angle_b), torch.sin(angle_b)], dim=-1)


def kick_state_obs(
    env: ManagerBasedRlEnv, timer_attr: str = "kick_timer"
) -> torch.Tensor:
    """One-hot encoding of the kick_contact_cycle two-phase state
    (mdp/events.py) -- ``[kicking, post_kick]``:

    - kicking: the ball hasn't been struck yet in its current "life"
      (``kick_timer`` == 0); the robot may strike it at any time.
    - post_kick: reward-farming window after a kick, purely timer-driven
      (``kick_timer`` > 0).

    Lets the policy directly condition behavior (e.g. approach/strike vs
    hold-through) on which phase it's in, instead of having to infer it
    implicitly from the ball's position/its own state.
    """
    timer = getattr(env, timer_attr, None)
    if timer is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    post_kick = timer > 0
    kicking = ~post_kick
    return torch.stack([kicking, post_kick], dim=-1).float()


def kick_intensity_obs(
    env: ManagerBasedRlEnv, intensity_attr: str = "kick_intensity"
) -> torch.Tensor:
    """Commanded kick intensity for the current ball "life" (``env.kick_intensity``,
    set by ``kick_contact_cycle``/``_place_ball_near_robot``, mdp/events.py):
    a continuous value in [0, 1], nonlinearly mapped to a target ball speed
    by ``kick_intensity_to_target_speed`` (mdp/rewards.py) -- 0.0 -> 1 m/s,
    0.5 -> 4 m/s, 1.0 -> unbounded ("as hard as possible").

    This is a task command, not a sensed quantity -- the actor must see the
    exact value it is being asked to act on. It goes to both the actor and
    critic groups with no noise, the same as a real commanded input would be.
    ``kick_speed_cap_penalty`` (mdp/rewards.py) reads the same attribute to
    penalize a trial where the ball ends up faster than its target speed.
    """
    intensity = getattr(env, intensity_attr, None)
    if intensity is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return intensity.float().unsqueeze(-1)


def robot_body_ang_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_ang_vel_w.shape[1]

    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)