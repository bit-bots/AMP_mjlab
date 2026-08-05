"""HighTorque Pi Plus AMP locomotion environment configs (mirrors the G1 config)."""

import copy
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import events as event_fns
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg as Unoise

from src.assets.robots import PIPLUS_ACTION_SCALE, get_piplus_robot_cfg
from src.tasks.amp_loco import mdp as amp_mdp
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg

# --- Pi Plus name mapping ---------------------------------------------------
ANCHOR_NAME = "torso_link"
ROOT_NAME = "base_link"
SITE_NAMES = ("l_foot", "r_foot")
FOOT_GEOMS = tuple(
  f"{side}_ankle_roll_link_collision{i}" for side in ("l", "r") for i in range(5)
)
# AMP / critic key bodies: root + (hip_roll, knee=calf, ankle_roll) x2 + (shoulder_roll,
# elbow, wrist) x2 = 13 bodies (matches the G1 count).
AMP_BODY_NAMES = (
  "base_link",
  "l_hip_roll_link",
  "l_calf_link",
  "l_ankle_roll_link",
  "r_hip_roll_link",
  "r_calf_link",
  "r_ankle_roll_link",
  "l_shoulder_roll_link",
  "l_elbow_link",
  "l_wrist_link",
  "r_shoulder_roll_link",
  "r_elbow_link",
  "r_wrist_link",
)


def piplus_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_amp_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_piplus_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = ROOT_NAME

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=ROOT_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=ROOT_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PIPLUS_ACTION_SCALE

  cfg.viewer.body_name = ANCHOR_NAME

  # Pi Plus is much shorter than G1: standing base ~0.34 m (ref motion dips to
  # ~0.20 m), so G1's 0.5 m "fallen" threshold terminates every episode instantly.
  cfg.terminations["bad_base_height"].params["minimum_height"] = 0.15

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.5

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_NAME,)

  # mjlab_piplus-style random impulse on the base, on top of AMP's friction/COM/
  # encoder DR (AMP has no impulse term). Matches the deployment repo: +/-80 N for
  # 0.1-0.2 s, 1-5 s cooldown, applied to base_link.
  cfg.events["impulse"] = EventTermCfg(
    func=event_fns.apply_body_impulse,
    mode="step",
    params={
      "force_range": (-80.0, 80.0),
      "torque_range": (0.0, 0.0),
      "duration_s": (0.1, 0.2),
      "cooldown_s": (1.0, 5.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_NAME,)),
    },
  )

  # Joint (encoder calibration) bias: mjxperiment resamples a per-EPISODE Gaussian
  # bias of std 2deg (0.0349 rad) added to the motor reference. AMP defaults to a
  # STARTUP-only uniform +/-0.015 (std ~0.009), ~4x smaller and fixed for all of
  # training. encoder_bias only supports uniform sampling, so match the std with a
  # +/-0.06 range (uniform half-width 0.0349*sqrt(3)) and resample per reset.
  cfg.events["encoder_bias"].mode = "reset"
  cfg.events["encoder_bias"].params["bias_range"] = (-0.06, 0.06)

  # No recovery clips yet: disable delayed-reset recovery for the first walk run.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0

  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "piplus", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _motion_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (ANCHOR_NAME,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (ANCHOR_NAME,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = SITE_NAMES
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (ROOT_NAME,)

  for grp in ("critic", "amp"):
    for term in cfg.observations[grp].terms:
      p = cfg.observations[grp].terms[term].params
      if "anchor_cfg" in p:
        p["anchor_cfg"].body_names = (ANCHOR_NAME,)
        p["anchor_cfg"].preserve_order = True
      if "body_cfg" in p:
        p["body_cfg"].body_names = AMP_BODY_NAMES
        # AMPLoader indexes bodies in AMP_BODY_NAMES order; SceneEntityCfg defaults to
        # sorted-by-model-index, which paired the wrong bodies in the discriminator.
        p["body_cfg"].preserve_order = True

  # G1 IMU sensor names -> Pi Plus equivalents (same physical sensors).
  _sensor_remap = {"robot/imu_ang_vel": "robot/gyro", "robot/imu_lin_vel": "robot/local_linvel"}
  for grp in cfg.observations.values():
    for term in grp.terms.values():
      p = getattr(term, "params", None)
      if p and p.get("sensor_name") in _sensor_remap:
        p["sensor_name"] = _sensor_remap[p["sensor_name"]]

  # Match the deployed mjxperiment walk's IMU noise: same distribution (Gaussian)
  # and magnitude. mjxperiment uses gyro ~ N(0, 0.05 rad/s) and projected-gravity
  # ~ N(0, 0.03). AMP defaults are uniform +/-0.2 (gyro) and +/-0.05 (gravity),
  # both noisier and the wrong shape.
  cfg.observations["actor"].terms["base_ang_vel"].noise = GaussianNoiseCfg(mean=0.0, std=0.05)
  cfg.observations["actor"].terms["projected_gravity"].noise = GaussianNoiseCfg(mean=0.0, std=0.03)
  # joint_pos: mjxperiment uses uniform +/-0.05 rad (same distribution as AMP's, but
  # AMP defaulted to a quieter +/-0.01). Bump to match.
  cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.05, n_max=0.05)

  # IMU mounting bias (mjxperiment): a per-episode roll/pitch misalignment (Gaussian
  # std 0.01 rad) rotates the gyro + projected-gravity readings. Applied to the ACTOR
  # only -- the critic keeps enable_corruption=False and the true (unbiased) readings,
  # so we de-share those two actor terms (critic_terms = {**actor_terms} makes them
  # the same objects) before swapping their obs functions to the biased variants.
  _biased = {
    "base_ang_vel": amp_mdp.imu_gyro_biased,
    "projected_gravity": amp_mdp.imu_projected_gravity_biased,
  }
  for _name, _fn in _biased.items():
    _term = copy.copy(cfg.observations["actor"].terms[_name])
    _term.func = _fn
    cfg.observations["actor"].terms[_name] = _term
  cfg.events["imu_mounting_bias"] = EventTermCfg(
    func=amp_mdp.randomize_imu_mounting_bias,
    mode="reset",
    params={"std_rad": 0.01},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def piplus_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg
