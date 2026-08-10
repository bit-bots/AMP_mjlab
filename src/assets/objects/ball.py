"""Soccer ball object for the kick task.

Physical parameters (radius, mass, friction, condim) are copied verbatim from the
mjxperiment kick task's ball_geom (mujoco_playground/_src/locomotion/piplus/xmls/
scene_mjx_kick_flat_terrain.xml), so the retargeted-kick policy sees the same ball
dynamics as the reference deployment env.
"""

import mujoco

from mjlab.entity import EntityCfg

BALL_BODY_NAME = "ball"
BALL_GEOM_NAME = "ball_geom"

BALL_RADIUS = 0.07  # m
BALL_MASS = 0.205  # kg
# (sliding, torsional, rolling) -- condim=6 so torsional/rolling friction apply.
BALL_FRICTION = (0.6, 0.1, 0.1)


def _get_ball_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=BALL_BODY_NAME)
  body.add_freejoint()
  geom = body.add_geom(
    name=BALL_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS, 0.0, 0.0],
    mass=BALL_MASS,
    rgba=[0.85, 0.15, 0.1, 1.0],
  )
  geom.friction = list(BALL_FRICTION)
  geom.condim = 6
  geom.contype = 1
  geom.conaffinity = 1
  return spec


def get_ball_cfg(
  spawn_pos: tuple[float, float, float] = (0.7, 0.0, BALL_RADIUS),
) -> EntityCfg:
  """Fresh ball EntityCfg (avoids shared-mutation issues).

  ``spawn_pos`` is only the fallback pose used when nothing else positions the
  ball (e.g. before the first reset event runs); the kick task's reset/resample
  events place it randomly relative to the robot every episode.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=spawn_pos),
    spec_fn=_get_ball_spec,
  )
