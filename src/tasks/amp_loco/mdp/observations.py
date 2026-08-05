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