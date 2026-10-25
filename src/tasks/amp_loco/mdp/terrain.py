from __future__ import annotations

from typing import TYPE_CHECKING

import mjlab.terrains as terrain_gen
from mjlab.envs.mdp.dr._core import Ranges, _randomize_model_field
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.spec_config import CollisionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


RANDOM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.4),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.6,
      noise_range=(0.02, 0.05),
      noise_step=0.02,
      border_width=0.25,
    ),
    # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
    #   proportion=0.1,
    #   amplitude_range=(0.0, 0.2),
    #   num_waves=4,
    #   border_width=0.25,
    # ),
  },
  add_lights=True,
)

def terrain_collisions() -> tuple[CollisionCfg, ...]:
  """Terrain contact policy: priority 1, above the robot (0) and below the ball
  (2).

  MuJoCo uses the friction, condim and solver parameters of the geom with the
  higher priority alone. The floor friction (randomized by an event) and the
  turf compliance (``randomize_terrain_contact``) therefore decide every
  robot-floor contact. The ball keeps its own parameters.
  """
  return (
    CollisionCfg(
      geom_names_expr=(".*",),
      contype=1,
      conaffinity=1,
      condim=3,
      priority=1,
      friction=(1.0,),
    ),
  )


@requires_model_fields("geom_solref", "geom_solimp")
def randomize_terrain_contact(
  env: ManagerBasedRlEnv,
  env_ids,
  asset_cfg: SceneEntityCfg,
  solref_ranges: Ranges,
  solimp_ranges: Ranges,
  shared_random: bool = True,
) -> None:
  """Randomize short-turf-like contact compliance on terrain geoms (from
  booster_mjlab).

  ``solref`` randomizes MuJoCo's contact time constant and damping ratio.
  ``solimp`` randomizes the contact impedance transition and keeps its
  midpoint and power parameters. The draw is once per env at startup, so each
  env has one consistent turf realization.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=solref_ranges,
    operation="abs",
    asset_cfg=asset_cfg,
    shared_random=shared_random,
    valid_axes=[0, 1],
  )
  _randomize_model_field(
    env,
    env_ids,
    "geom_solimp",
    entity_type="geom",
    ranges=solimp_ranges,
    operation="abs",
    asset_cfg=asset_cfg,
    shared_random=shared_random,
    valid_axes=[0, 1, 2, 3, 4],
  )
