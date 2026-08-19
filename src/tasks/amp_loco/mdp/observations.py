from __future__ import annotations

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