import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

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

# Kick task: flat / tilted-grid / Perlin-noise mix, deliberately mild --
# the Pi Plus is small and the target scenario is playing on roughly even
# grass, not rugged off-road terrain. curriculum=True with 3 equal-
# proportion sub-terrains and num_cols a multiple of 3 gives an exact,
# predictable column split (verified against TerrainGenerator's cumulative-
# proportion column assignment): cols [0:4)=flat, [4:8)=tilted_grid,
# [8:12)=perlin -- see KICK_TERRAIN_FLAT_COLS/KICK_TERRAIN_NONFLAT_COLS
# below, which terrain_flat_fraction_curriculum (mdp/curriculums.py) uses
# to reassign envs between them over training.
KICK_TERRAIN_FLAT_COLS = (0, 1, 2, 3)
KICK_TERRAIN_NONFLAT_COLS = (4, 5, 6, 7, 8, 9, 10, 11)
KICK_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(6.0, 6.0),
  border_width=10.0,
  num_rows=2,
  num_cols=12,
  curriculum=True,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
    # Grid of individually tilted tiles -- small tiles (grid_width) and a
    # low tilt/height cap so it reads as uneven ground, not stairs/rubble.
    "tilted_grid": terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=1.0,
      grid_width=0.3,
      tilt_range_deg=6.0,
      height_range=0.04,
      platform_width=1.2,
      border_width=0.25,
    ),
    # Smooth fractal noise -- a continuous collidable surface, the natural
    # choice for a ball that needs to roll (unlike the tiled/box terrains,
    # which have small sharp edges between tiles). scale is the feature
    # size in meters; kept small (near foot-sized) to read as grassy
    # unevenness rather than rolling hills.
    "perlin": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(0.01, 0.03),
      scale=0.6,
      octaves=3,
      # Coarser than the 0.05 default -- matches horizontal_scale, which
      # quarters the triangle density. The default combination triggered
      # MuJoCo's "height field collision overflow" warning (>=50
      # simultaneous candidate triangles under a single geom); this stays
      # well clear of that while still resolving the requested foot-sized
      # (scale=0.6) bumps just fine.
      resolution=0.1,
      border_width=0.25,
    ),
  },
  add_lights=True,
)
