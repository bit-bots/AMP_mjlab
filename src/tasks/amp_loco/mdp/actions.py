"""Joint position action that clips the policy action to the joint limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LimitedJointPositionAction(JointPositionAction):
  """``JointPositionAction`` with the raw action clipped so that the joint
  position target (offset + scale * action) stays inside the joint's soft
  position limits.

  The clipped value is the one stored as the raw action. So the "last action"
  observation and the action-rate reward see a bounded action too. Without
  the clip, one policy output of 1e3 or more (seen in the kick task) went into
  the observation history, grew by feedback, and collapsed training.
  """

  def process_actions(self, actions: torch.Tensor):
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]  # (N, J, 2)
    lo = (limits[..., 0] - self._offset) / self._scale
    hi = (limits[..., 1] - self._offset) / self._scale
    actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    super().process_actions(torch.minimum(torch.maximum(actions, lo), hi))


@dataclass(kw_only=True)
class LimitedJointPositionActionCfg(JointPositionActionCfg):
  """Config for ``LimitedJointPositionAction``."""

  def build(self, env: ManagerBasedRlEnv) -> LimitedJointPositionAction:
    return LimitedJointPositionAction(self, env)
