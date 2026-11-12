"""Leg motor of the Pi Plus: DC motor with a driving-side torque-speed limit."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg
from mjlab.entity import Entity


def drive_limited_clip(
  effort: torch.Tensor,
  saturation_effort: torch.Tensor,
  velocity_limit: torch.Tensor,
  force_limit: torch.Tensor,
  vel: torch.Tensor,
) -> torch.Tensor:
  """Clip the motor torque to the torque-speed limit of the leg motors.

  The limit falls linearly from ``saturation_effort`` at zero speed to zero at
  ``velocity_limit`` and acts on the driving side only (torque and joint speed
  with the same sign). It never becomes negative: above the no-load speed the
  motor gives no driving torque, but it does not brake by itself (the DC motor
  of mjlab would). A braking torque is limited to ``force_limit`` at every
  speed. ``force_limit`` bounds both sides.
  """
  drive = (saturation_effort * (1.0 - vel / velocity_limit)).clamp(min=0.0)
  brake = (saturation_effort * (1.0 + vel / velocity_limit)).clamp(min=0.0)
  max_effort = torch.minimum(drive, force_limit)
  min_effort = -torch.minimum(brake, force_limit)
  return torch.clamp(effort, min=min_effort, max=max_effort)


class LegMotorActuator(DcMotorActuator):
  """``DcMotorActuator`` with ``drive_limited_clip`` as its torque limit."""

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self.saturation_effort is not None
    assert self.velocity_limit_motor is not None
    assert self.force_limit is not None
    assert self._joint_vel_clipped is not None
    return drive_limited_clip(
      effort,
      self.saturation_effort,
      self.velocity_limit_motor,
      self.force_limit,
      self._joint_vel_clipped,
    )


@dataclass(kw_only=True)
class LegMotorActuatorCfg(DcMotorActuatorCfg):
  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> LegMotorActuator:
    return LegMotorActuator(self, entity, target_ids, target_names)
