"""Left/right mirror symmetry for the Pi Plus kick task (PPO data augmentation
and mirror loss).

Ported from piplus_amp_mjlab (booster_mjlab/tasks/velocity/mdp/observations.py,
``augment_symmetries_piplus``). The mirror is a reflection across the robot's
sagittal (XZ) plane: y -> -y. Each term is in a frame that has its xz-plane on
the robot's sagittal plane (torso, base or base-yaw frame), so the same rule
applies to all terms:
  - polar vectors (positions, linear velocities, gravity, directions):
    (x, y, z) -> (x, -y, z),
  - axial vectors (angular velocities): (x, y, z) -> (-x, y, -z),
  - rotation matrices: R -> M R M, M = diag(1, -1, 1),
  - joints: swap each l_X <-> r_X joint and negate ALL joints. The Pi Plus XML
    mirrors each joint axis (see scripts/mirror_amp_motion.py, which uses the
    same rule for the AMP motion data),
  - left/right bodies: swap the pair, then apply the vector rule,
  - scalars without a side (kick intensity, kick phase): no change.

The actor and the critic use a time-major history (``history_ordering="time"``,
4 frames), so the mirror applies to each frame. ``_check_kick_obs_layout``
compares the layout below with the env and raises an error on a mismatch.

The rsl_rl copy in this repo calls the function as
``func(obs=tensor, actions=tensor, env=env, obs_type="policy" | "critic")``.
"""

from __future__ import annotations

from typing import Callable

import torch

from src.tasks.amp_loco.config.piplus.env_cfgs import AMP_BODY_NAMES

# Joint order of the env (robot entity) and of the AMP motion data.
PIPLUS_JOINT_NAMES: tuple[str, ...] = (
  "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint",
  "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint",
  "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint",
  "r_ankle_pitch_joint", "r_ankle_roll_joint",
  "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
  "l_ankle_pitch_joint", "l_ankle_roll_joint",
)
JOINT_DIM = len(PIPLUS_JOINT_NAMES)
NUM_BODIES = len(AMP_BODY_NAMES)
HISTORY_LENGTH = 4


def _lr_swap_index(names: tuple[str, ...]) -> tuple[int, ...]:
  """Index map that swaps each l_X name with its r_X name (central names stay)."""
  index = []
  for name in names:
    if name.startswith("l_"):
      index.append(names.index("r_" + name[2:]))
    elif name.startswith("r_"):
      index.append(names.index("l_" + name[2:]))
    else:
      index.append(names.index(name))
  return tuple(index)


JOINT_MIRROR_INDEX = _lr_swap_index(PIPLUS_JOINT_NAMES)
# All 20 joints change sign (mirrored joint axes in piplus.xml).
JOINT_MIRROR_SIGN = (-1.0,) * JOINT_DIM
BODY_MIRROR_INDEX = _lr_swap_index(tuple(AMP_BODY_NAMES))

_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)
# robot_body_ori_b keeps the first two columns of R, row-major:
# [R00, R01, R10, R11, R20, R21]. M R M negates R_ij when exactly one of i, j
# is the y axis (index 1).
_ROT_2COL = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)

# Expected (term name, dim per frame) of each group.
KICK_ACTOR_TERMS: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", JOINT_DIM),
  ("joint_vel", JOINT_DIM),
  ("actions", JOINT_DIM),
  ("ball_pos_b", 3),
  ("kick_dir_b", 2),
  ("kick_intensity", 1),
)
KICK_CRITIC_TERMS: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", JOINT_DIM),
  ("joint_vel", JOINT_DIM),
  ("actions", JOINT_DIM),
  ("base_lin_vel", 3),
  ("body_pos_b", 3 * NUM_BODIES),
  ("body_ori_b", 6 * NUM_BODIES),
  ("ball_pos_b", 3),
  ("ball_lin_vel_b", 3),
  ("kick_state", 2),
  ("kick_dir_b", 2),
  ("kick_intensity", 1),
)


def _sign(x: torch.Tensor, signs: tuple[float, ...]) -> torch.Tensor:
  return x * x.new_tensor(signs)


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
  """Mirror a [N, 20] joint tensor (positions, velocities or actions)."""
  index = torch.as_tensor(JOINT_MIRROR_INDEX, device=x.device)
  return _sign(x[:, index], JOINT_MIRROR_SIGN)


def _mirror_bodies(x: torch.Tensor, per_body_signs: tuple[float, ...]) -> torch.Tensor:
  """Swap the left/right bodies of a [N, B * k] tensor and apply the signs."""
  k = len(per_body_signs)
  index = torch.as_tensor(BODY_MIRROR_INDEX, device=x.device)
  x = x.reshape(x.shape[0], NUM_BODIES, k)[:, index]
  return _sign(x, per_body_signs).reshape(x.shape[0], NUM_BODIES * k)


def _identity(x: torch.Tensor) -> torch.Tensor:
  return x


# Mirror rule per term. Frames (see mdp/observations.py and env_cfgs.py):
_TERM_MIRROR: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
  # IMU gyro in the base frame (axial vector).
  "base_ang_vel": lambda x: _sign(x, _AXIAL),
  # Gravity in the base frame.
  "projected_gravity": lambda x: _sign(x, _POLAR),
  # Joint position/velocity relative to the default pose, and the last raw
  # action (20 joint position targets, no other action term).
  "joint_pos": mirror_joints,
  "joint_vel": mirror_joints,
  "actions": mirror_joints,
  # IMU velocimeter in the base frame.
  "base_lin_vel": lambda x: _sign(x, _POLAR),
  # AMP key bodies, position and orientation in the torso frame.
  "body_pos_b": lambda x: _mirror_bodies(x, _POLAR),
  "body_ori_b": lambda x: _mirror_bodies(x, _ROT_2COL),
  # Ball position and velocity in the torso frame.
  "ball_pos_b": lambda x: _sign(x, _POLAR),
  "ball_lin_vel_b": lambda x: _sign(x, _POLAR),
  # One-hot [kicking, post_kick]: no side.
  "kick_state": _identity,
  # Target kick direction (x, y) in the base yaw frame.
  "kick_dir_b": lambda x: _sign(x, (1.0, -1.0)),
  # Kick intensity command in [0, 1]: no side.
  "kick_intensity": _identity,
}


def _mirror_group(obs: torch.Tensor, terms: tuple[tuple[str, int], ...]) -> torch.Tensor:
  frame_dim = sum(d for _, d in terms)
  if obs.shape[-1] != HISTORY_LENGTH * frame_dim:
    raise ValueError(
      f"Kick observation dim mismatch: got {obs.shape[-1]}, expected "
      f"{HISTORY_LENGTH} x {frame_dim} = {HISTORY_LENGTH * frame_dim}."
    )
  n = obs.shape[0]
  # Time-major history: [frame_0 terms, frame_1 terms, ...].
  frames = obs.reshape(n * HISTORY_LENGTH, frame_dim)
  out = torch.empty_like(frames)
  start = 0
  for name, dim in terms:
    out[:, start : start + dim] = _TERM_MIRROR[name](frames[:, start : start + dim])
    start += dim
  return out.reshape(n, HISTORY_LENGTH * frame_dim)


def mirror_kick_actor_obs(obs: torch.Tensor) -> torch.Tensor:
  return _mirror_group(obs, KICK_ACTOR_TERMS)


def mirror_kick_critic_obs(obs: torch.Tensor) -> torch.Tensor:
  return _mirror_group(obs, KICK_CRITIC_TERMS)


def mirror_kick_actions(actions: torch.Tensor) -> torch.Tensor:
  if actions.shape[-1] != JOINT_DIM:
    raise ValueError(
      f"Kick action dim mismatch: got {actions.shape[-1]}, expected {JOINT_DIM}."
    )
  return mirror_joints(actions)


_layout_checked = False


def _check_kick_obs_layout(env) -> None:
  """Compare the env observation and action layout with the constants above.

  The check runs one time. It does nothing if ``env`` is None (unit tests).
  """
  global _layout_checked
  if _layout_checked or env is None:
    return
  unwrapped = getattr(env, "unwrapped", env)
  om = unwrapped.observation_manager
  am = unwrapped.action_manager
  for group, terms in (("actor", KICK_ACTOR_TERMS), ("critic", KICK_CRITIC_TERMS)):
    got = [
      (name, tuple(int(d) for d in dims))
      for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group])
    ]
    expected = [(name, (HISTORY_LENGTH, dim)) for name, dim in terms]
    if got != expected:
      raise ValueError(
        f"Pi Plus kick '{group}' observation layout mismatch: got {got}, "
        f"expected {expected} (history {HISTORY_LENGTH})."
      )
    if om.cfg[group].history_ordering != "time":
      raise ValueError(
        f"Pi Plus kick '{group}' history ordering must be 'time', got "
        f"'{om.cfg[group].history_ordering}'."
      )
    # The body mirror needs the AMP body order.
    for term_name, term_cfg in zip(om.active_terms[group], om._group_obs_term_cfgs[group]):
      body_cfg = term_cfg.params.get("body_cfg")
      if body_cfg is not None and (
        tuple(body_cfg.body_names) != tuple(AMP_BODY_NAMES) or not body_cfg.preserve_order
      ):
        raise ValueError(
          f"Pi Plus kick '{group}/{term_name}' body order mismatch: got "
          f"{body_cfg.body_names} (preserve_order={body_cfg.preserve_order}), "
          f"expected {AMP_BODY_NAMES} in this order."
        )
  if list(am.active_terms) != ["joint_pos"]:
    raise ValueError(
      f"Pi Plus kick action terms must be ['joint_pos'], got {list(am.active_terms)}."
    )
  joint_names = tuple(am.get_term("joint_pos").target_names)
  if joint_names != PIPLUS_JOINT_NAMES:
    raise ValueError(
      f"Pi Plus kick action joint order mismatch: got {joint_names}, "
      f"expected {PIPLUS_JOINT_NAMES}."
    )
  robot_joints = tuple(unwrapped.scene["robot"].joint_names)
  if robot_joints != PIPLUS_JOINT_NAMES:
    raise ValueError(
      f"Pi Plus kick robot joint order mismatch: got {robot_joints}, "
      f"expected {PIPLUS_JOINT_NAMES}."
    )
  _layout_checked = True


def augment_symmetries_piplus_kick(
  env=None,
  obs: torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
  obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Apply the left/right mirror to a PPO mini-batch.

  The output has the original samples first and the mirrored samples second.
  ``obs_type`` "policy" selects the actor layout, "critic" the critic layout.
  """
  _check_kick_obs_layout(env)
  if obs is not None:
    if obs_type == "policy":
      mirrored = mirror_kick_actor_obs(obs)
    elif obs_type == "critic":
      mirrored = mirror_kick_critic_obs(obs)
    else:
      raise ValueError(f"Unsupported obs_type '{obs_type}'.")
    obs = torch.cat((obs, mirrored), dim=0)
  if actions is not None:
    actions = torch.cat((actions, mirror_kick_actions(actions)), dim=0)
  return obs, actions
