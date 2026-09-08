from dataclasses import dataclass

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import (
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp, darken_rgba

_MUJOCO_GREEN = (0.25, 0.80, 0.45)


@dataclass(kw_only=True)
class GroundedTiltedGridTerrainCfg(terrain_gen.BoxTiltedGridTerrainCfg):
  """``BoxTiltedGridTerrainCfg`` with the platform's absolute height exposed
  as a config field. Upstream hardcodes it to 0.2m above the patch's own
  origin, which -- combined with the default ``floor_depth=2.0`` pit below
  it -- makes the whole patch read as a raised island floating over a deep
  canyon instead of sitting flush with neighboring flat/other patches.
  Identical to the parent's ``function`` except ``base_h`` is a field
  instead of a literal, defaulted to 0.0 so it lines up with flat ground.

  Ported from the kick task (config/piplus/kick_env_cfg.py's KICK_TERRAIN_CFG)
  as a general-purpose fix, not currently wired into any terrain config here.
  """

  base_h: float = 0.0
  # The upstream design carves out a smooth flat square (of size
  # ``platform_width``) in the middle of the tile grid for spawning on.
  # Its boundary generally doesn't land on a tile edge (platform_width is
  # rarely an exact multiple of grid_width, and even when it is, floating
  # point can leave a sliver gap), which exposes the dark floor underneath
  # as a visible seam/"canyon" ring right at the platform's border. Set to
  # False for uniform coverage instead, removing both the seam and the
  # platform look entirely.
  use_platform: bool = True

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    max_tilt = np.deg2rad(self.tilt_range_deg * difficulty)
    actual_range = self.height_range * difficulty

    num_boxes_x = int((self.size[0] - 2 * self.border_width) / self.grid_width)
    num_boxes_y = int((self.size[1] - 2 * self.border_width) / self.grid_width)

    border_actual = self.size[0] - num_boxes_x * self.grid_width
    half_border = border_actual / 2

    border_rgba = darken_rgba(brand_ramp(_MUJOCO_GREEN, 0.0), 0.85)
    base_h = self.base_h
    z_center = (base_h - self.floor_depth) / 2
    half_height = (base_h + self.floor_depth) / 2

    if border_actual > 0:
      border_center = (self.size[0] / 2, self.size[1] / 2, z_center)
      border_boxes = make_border(
        body,
        self.size,
        (num_boxes_x * self.grid_width, num_boxes_y * self.grid_width),
        base_h + self.floor_depth,
        border_center,
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.1, 0.1, 0.1, 1.0)))

    platform_min = platform_max = 0.0
    if self.use_platform:
      platform_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.platform_width / 2, self.platform_width / 2, half_height),
        pos=(self.size[0] / 2, self.size[1] / 2, z_center),
      )
      geometries.append(
        TerrainGeometry(geom=platform_geom, color=brand_ramp(_MUJOCO_GREEN, 0.5))
      )
      platform_half = self.platform_width / 2
      terrain_center = self.size[0] / 2
      platform_min = terrain_center - platform_half
      platform_max = terrain_center + platform_half

    for i in range(num_boxes_x):
      bx_center = half_border + (i + 0.5) * self.grid_width
      for j in range(num_boxes_y):
        by_center = half_border + (j + 0.5) * self.grid_width

        if (
          self.use_platform
          and (platform_min <= bx_center <= platform_max)
          and (platform_min <= by_center <= platform_max)
        ):
          continue

        h_noise = rng.uniform(-actual_range / 2, actual_range / 2)
        total_h = base_h + h_noise

        tilt_x = rng.uniform(-max_tilt, max_tilt)
        tilt_y = rng.uniform(-max_tilt, max_tilt)

        x_min, x_max = bx_center - self.grid_width / 2, bx_center + self.grid_width / 2
        y_min, y_max = by_center - self.grid_width / 2, by_center + self.grid_width / 2

        verts = []
        for vx in [x_min, x_max]:
          for vy in [y_min, y_max]:
            verts.append([vx, vy, -self.floor_depth])
        for vx in [x_min, x_max]:
          for vy in [y_min, y_max]:
            vz = total_h + tilt_x * (vx - bx_center) + tilt_y * (vy - by_center)
            verts.append([vx, vy, vz])

        faces = [
          4, 6, 7, 4, 7, 5,  # Top (+z)
          0, 1, 3, 0, 3, 2,  # Bottom (-z)
          0, 2, 6, 0, 6, 4,  # Front (-y)
          1, 5, 7, 1, 7, 3,  # Back (+y)
          0, 4, 5, 0, 5, 1,  # Left (-x)
          2, 3, 7, 2, 7, 6,  # Right (+x)
        ]  # fmt: skip

        m_name = f"tile_{i}_{j}_{rng.integers(int(1e9))}"
        mesh = spec.add_mesh(
          name=m_name,
          uservert=np.array(verts).flatten().tolist(),
          userface=np.array(faces).flatten().tolist(),
        )

        rgba = brand_ramp(_MUJOCO_GREEN, rng.uniform(0.3, 0.7))
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_MESH,
          meshname=mesh.name,
          pos=(0, 0, 0),
        )
        geometries.append(TerrainGeometry(geom=geom, color=rgba))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, base_h])
    return TerrainOutput(origin=origin, geometries=geometries)


# Ported from the kick task's KICK_TERRAIN_CFG: flat / tilted-grid / Perlin-noise
# mix, driven by terrain_flat_fraction_curriculum (mdp/curriculums.py) instead of
# mjlab's closed-loop terrain_levels_vel. Each non-flat type has a mild and a rough
# variant so half of all non-flat ground is noticeably bumpier, not just the same
# mild texture everywhere. num_cols is a multiple of 5 with 5 equal-proportion
# sub-terrains for an exact, predictable column split -- see
# WALK_TERRAIN_FLAT_COLS/WALK_TERRAIN_NONFLAT_COLS below.
WALK_TERRAIN_FLAT_COLS = (0, 1, 2)
WALK_TERRAIN_NONFLAT_COLS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
WALK_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(6.0, 6.0),
  border_width=10.0,
  num_rows=2,
  num_cols=15,
  curriculum=True,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
    "tilted_mild": GroundedTiltedGridTerrainCfg(
      proportion=1.0,
      grid_width=0.3,
      tilt_range_deg=3.0,
      height_range=0.02,
      border_width=0.25,
      base_h=0.0,
      floor_depth=0.15,
      use_platform=False,
    ),
    "tilted_rough": GroundedTiltedGridTerrainCfg(
      proportion=1.0,
      grid_width=0.3,
      tilt_range_deg=4.5,
      height_range=0.032,
      border_width=0.25,
      base_h=0.0,
      floor_depth=0.15,
      use_platform=False,
    ),
    "perlin_mild": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(0.02, 0.04),
      scale=12.0,
      octaves=3,
      resolution=0.1,
      border_width=0.25,
    ),
    "perlin_rough": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(0.035, 0.055),
      scale=12.0,
      octaves=3,
      resolution=0.1,
      border_width=0.25,
    ),
  },
  add_lights=True,
)


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
