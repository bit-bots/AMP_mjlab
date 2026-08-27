"""Reward-weight curricula: fade some terms out, ramp others in over training.

``env.common_step_counter`` is a raw env-step count (incremented once per
``env.step()`` call), not a PPO iteration count. With ``num_steps_per_env=24``
(rl_cfg.py), iteration N corresponds to ``common_step_counter == N * 24`` --
callers pass iteration-scaled step thresholds, matching the existing
``commands_vel`` curriculum's convention (src/tasks/velocity/mdp/curriculums.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _base_weight(term_cfg) -> float:
  """The term's original configured weight, cached on first curriculum call so
  repeated invocations (every reset) scale from the same reference instead of
  compounding."""
  if not hasattr(term_cfg, "_curriculum_base_weight"):
    term_cfg._curriculum_base_weight = term_cfg.weight
  return term_cfg._curriculum_base_weight


def fade_reward_weights_linear(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_names: list[str],
  start_step: int,
  end_step: int,
  end_scale: float = 0.1,
) -> torch.Tensor:
  """Linearly fade the given reward terms' weights from 1x their original
  value (at/before ``start_step``) down to ``end_scale``x (at/after
  ``end_step``), holding flat outside that range.

  Used to fade out the kick task's dense approach-shaping rewards
  (move_to_ball, torso_orient_to_ball, right_foot_ball_contact) once the
  policy has learned to reliably reach and touch the ball, so it stops
  leaning on them and lets kick_impact / the AMP style term drive the rest.
  """
  del env_ids
  step = env.common_step_counter
  span = max(end_step - start_step, 1)
  frac = min(max((step - start_step) / span, 0.0), 1.0)
  scale = 1.0 - (1.0 - end_scale) * frac
  for name in reward_names:
    term_cfg = env.reward_manager.get_term_cfg(name)
    term_cfg.weight = _base_weight(term_cfg) * scale
  return torch.tensor(scale)


def anneal_reward_param_linear(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_name: str,
  start_step: int,
  end_step: int,
  start_value: float,
  end_value: float,
) -> torch.Tensor:
  """Linearly anneal one of a reward term's own params (not its weight) from
  ``start_value`` to ``end_value`` over ``[start_step, end_step]``, holding
  flat outside that range.

  Used to narrow ``kick_direction_reward``'s ``sigma`` (the angular tolerance
  for "close enough" direction) over training: wide/forgiving at first so the
  robot can discover the reward at all, tightening later to demand real
  directional accuracy.
  """
  del env_ids
  step = env.common_step_counter
  span = max(end_step - start_step, 1)
  frac = min(max((step - start_step) / span, 0.0), 1.0)
  value = start_value + (end_value - start_value) * frac
  term_cfg = env.reward_manager.get_term_cfg(reward_name)
  term_cfg.params[param_name] = value
  return torch.tensor(value)


def disable_termination_at_step(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  term_name: str,
  step: int,
) -> torch.Tensor:
  """Hard step-change for a TERMINATION term (not a reward): once
  ``env.common_step_counter`` passes ``step``, set the term's own
  ``enabled=False`` param so it never fires again -- requires the
  termination function to accept and honor an ``enabled`` kwarg (see
  ``object_contact``, mdp/terminations.py). Mirrors
  ``bump_reward_weight_at_step``, but TerminationManager (not RewardManager)
  re-reads ``term_cfg.params`` fresh every ``compute()`` call, so mutating
  the params dict here works the same way.
  """
  del env_ids
  term_cfg = env.termination_manager.get_term_cfg(term_name)
  if env.common_step_counter >= step:
    term_cfg.params["enabled"] = False
  return torch.tensor(float(term_cfg.params.get("enabled", True)))


def terrain_flat_fraction_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  flat_cols: tuple[int, ...],
  nonflat_cols: tuple[int, ...],
  start_nonflat_frac: float,
  end_nonflat_frac: float,
  end_step: int,
) -> torch.Tensor:
  """On every reset, reassign each resetting env's terrain COLUMN (not row --
  see ``TerrainGeneratorCfg``/``TerrainEntity`` in mjlab.terrains): the
  probability of landing on a non-flat column ramps linearly from
  ``start_nonflat_frac`` (at step 0) to ``end_nonflat_frac`` (at
  ``end_step``), held flat thereafter. Only resetting envs are touched, so
  the actual population fraction converges to the scheduled target as
  episodes churn, not instantaneously.

  Must run BEFORE the reset-mode events that place the robot -- it does,
  since ``ManagerBasedRlEnv._reset_idx`` calls ``curriculum_manager.compute()``
  ahead of ``event_manager.apply(mode="reset", ...)``, and
  ``reset_from_motion_data`` reads ``env.scene.env_origins`` (which this
  mutates via ``terrain.env_origins``) to place the robot.

  No-op if the scene has no procedural terrain generator (e.g. a plain
  flat-plane terrain, which has no columns to choose between).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return torch.tensor(0.0)

  step = env.common_step_counter
  frac = min(max(step / max(end_step, 1), 0.0), 1.0)
  nonflat_frac = start_nonflat_frac + (end_nonflat_frac - start_nonflat_frac) * frac

  n = env_ids.shape[0]
  is_nonflat = torch.rand(n, device=env.device) < nonflat_frac
  flat_cols_t = torch.tensor(flat_cols, device=env.device)
  nonflat_cols_t = torch.tensor(nonflat_cols, device=env.device)
  flat_choice = flat_cols_t[torch.randint(0, len(flat_cols), (n,), device=env.device)]
  nonflat_choice = nonflat_cols_t[
    torch.randint(0, len(nonflat_cols), (n,), device=env.device)
  ]
  chosen_cols = torch.where(is_nonflat, nonflat_choice, flat_choice)

  terrain.terrain_types[env_ids] = chosen_cols
  # Row (difficulty level) is left as whatever each env already has -- rows
  # aren't used as a difficulty axis here, just extra random variety within
  # a chosen column.
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
  ]
  return torch.tensor(nonflat_frac)


def bump_reward_weight_at_step(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_names: list[str],
  step: int,
  scale: float,
) -> torch.Tensor:
  """Hard step-change: once ``env.common_step_counter`` passes ``step``,
  multiply the given reward terms' weights by ``scale`` (relative to their
  original configured value); holds their original weight before that.

  Used to tighten the kick task's foot-slip/torque penalties once the policy
  has already learned to kick, rather than fighting exploration with a strict
  penalty from the start.
  """
  del env_ids
  past = env.common_step_counter >= step
  for name in reward_names:
    term_cfg = env.reward_manager.get_term_cfg(name)
    term_cfg.weight = _base_weight(term_cfg) * (scale if past else 1.0)
  return torch.tensor(scale if past else 1.0)
