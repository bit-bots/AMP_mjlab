"""HighTorque Pi Plus constants (AMP port).

The robot model is kept byte-identical to the smp/BeyondMimic Pi Plus port
(``piplus.xml`` + meshes copied verbatim). For real-robot deployment the actuators
use two groups. The arms are position servos (kp 6, kd 0.6) with a randomized
0-5 physics-step latency. The legs use the identified actuator model of the real
robot (see "Leg actuator model" below).
"""

import re
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH
from src.assets.robots.piplus.leg_motor import LegMotorActuatorCfg

# Actuator delay range in PHYSICS timesteps (mjlab DelayBuffer quantizes to physics
# steps, not control steps). At 5 ms physics (decimation 4 -> 50 Hz control):
# 5 physics steps = 25 ms = 1.25 control steps.
ACTUATOR_LAG_MIN = 0
ACTUATOR_LAG_MAX = 5

##
# MJCF and assets.
##

PIPLUS_XML: Path = SRC_PATH / "assets" / "robots" / "piplus" / "xmls" / "piplus.xml"
assert PIPLUS_XML.exists()


# Diagnostic: zero the MJCF's passive joint damping (0.25 arms / 0.75 legs Nm*s/rad, on
# top of the actuator kd), as in the booster_mjlab walk run. Friction loss and armature
# are unchanged. Set False for the real robot's model.
REMOVE_PASSIVE_DAMPING = True


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PIPLUS_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  # Normalize meshdir so embedded-asset keys match MuJoCo's lookup ("meshes/<f>").
  spec.meshdir = "meshes"
  spec.assets = get_assets("meshes")
  # Drop the XML's built-in <position> actuators; the BuiltinPositionActuatorCfg
  # (wrapped in DelayedActuatorCfg) re-adds them, so keeping the XML ones would
  # double actuation (nu=40) with the XML set idle at ctrl=0 fighting the policy.
  for act in list(spec.actuators):
    spec.delete(act)
  if REMOVE_PASSIVE_DAMPING:
    for joint in spec.joints:
      if joint.type != mujoco.mjtJoint.mjJNT_FREE:
        joint.damping = np.zeros(3)
  # Leg joint force cap above the torque limit of the model (see LEG_FORCE_CAP).
  for joint in spec.joints:
    if re.fullmatch("|".join(LEG_JOINT_NAMES), joint.name):
      joint.actfrcrange = [-LEG_FORCE_CAP, LEG_FORCE_CAP]
      joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
  return spec


##
# Actuators: gains based on the bitbots mjlab_playground pi_plus, leg/hip_roll kp
# bumped 35->50 for stiffer tracking
# (arm kp=6 kv=0.6 effort=10; leg kp=50 kv=1.1; hip_roll kp=50 kv=1.4; effort=20;
# armature from bitbots_main), built-in position servos wrapped in
# DelayedActuatorCfg for a randomized 0-5 physics-step latency (= 0-25 ms,
# ~1.25 control steps @ 50 Hz; mjlab 1.2.0's BuiltinPositionActuatorCfg has no
# delay field, so the delay is a wrapper).
##

PIPLUS_ACTUATOR_ARM = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_upper_arm_joint",
      ".*_elbow_joint",
    ),
    stiffness=6.0,
    damping=0.6,
    effort_limit=10.0,
    armature=0.01317,
    frictionloss=0.2,
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)

##
# Leg actuator model (system identification of the real robot, handoff
# SIM_RL_HANDOFF.md). All 12 leg joints share one model:
#
# * PD gains kp 50, kd 2.0 (hip roll too).
# * Armature 0.02, Coulomb friction loss 0.3 Nm, no passive viscous damping
#   (REMOVE_PASSIVE_DAMPING above).
# * Torque-speed limit on the driving side only (stall torque 32 Nm, no-load
#   speed 8 rad/s, limit 20 Nm). The limit is 20 Nm up to 3 rad/s and 0 Nm from
#   8 rad/s on (no braking by the motor itself). A braking torque (torque against
#   the joint velocity) is limited to 20 Nm at every speed (leg_motor.py).
# * The PD law runs explicitly in every physics step (LegMotorActuator) because
#   the torque limit depends on the joint velocity of that step.
# * Command delay of 2 to 4 physics steps (10 to 20 ms): the whole round trip of
#   the real actuator, one value per environment for all leg joints
#   (``randomize_leg_actuators``, mdp/actuator_model.py).
#
# These are the nominal values. ``randomize_leg_actuators`` and the
# armature/friction events of env_cfgs.py randomize them per environment.
##

LEG_KP = 50.0
LEG_KD = 2.0
LEG_ARMATURE = 0.02
LEG_FRICTIONLOSS = 0.3
LEG_STALL_TORQUE = 32.0  # Nm; with LEG_EFFORT_LIMIT it gives the knee at 3 rad/s
LEG_EFFORT_LIMIT = 20.0  # Nm; TAU_PEAK and TAU_BRAKE of the handoff
LEG_NO_LOAD_SPEED = 8.0  # rad/s; W0 of the handoff
# MuJoCo cap of the leg motors and of the leg joints (actuatorfrcrange). It is
# above LEG_EFFORT_LIMIT so that the torque randomization (up to x1.1 x1.03)
# is not cut by it. The torque limit of the model is LEG_EFFORT_LIMIT.
LEG_FORCE_CAP = 25.0
# Round trip of the real actuator (command to torque in the joint state): 10 to
# 20 ms. The model has this as the command delay only, with no extra delay of
# the joint state in the observation.
LEG_DELAY_LAG_MIN = 2  # physics steps (10 ms)
LEG_DELAY_LAG_MAX = 4  # physics steps (20 ms)
LEG_JOINT_NAMES = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_thigh_joint",
  ".*_calf_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
)

PIPLUS_ACTUATOR_LEG = DelayedActuatorCfg(
  base_cfg=LegMotorActuatorCfg(
    target_names_expr=LEG_JOINT_NAMES,
    stiffness=LEG_KP,
    damping=LEG_KD,
    effort_limit=LEG_FORCE_CAP,
    saturation_effort=LEG_STALL_TORQUE,
    velocity_limit=LEG_NO_LOAD_SPEED,
    armature=LEG_ARMATURE,
    frictionloss=LEG_FRICTIONLOSS,
  ),
  delay_min_lag=LEG_DELAY_LAG_MIN,
  delay_max_lag=LEG_DELAY_LAG_MAX,
  # The lag is set once per environment by randomize_leg_actuators and again after every reset by apply_leg_lags. Hold it.
  delay_hold_prob=1.0,
)

##
# Keyframe (dataset-mean pose over WalkandRun -- 10 CMU walk clips + the 2
# LAFAN1 run/walk clips, plus a mirror of every one of those 12, 69458 frames
# total). Every clip now has a mirror counterpart (the 2 LAFAN1 clips didn't
# used to), so the mean is exactly L/R antisymmetric: r_X == -l_X for every
# joint. Not directly comparable to bitbots_main's static walkready pose --
# this is the mean of dynamic running/walking gaits, not a rest pose.
#
# Synced from AMP_mjlab main's piplus_constants.py -- this worktree's own
# copy had been left on the pre-retargeting-fix values (wrong-signed arm
# joints) since this file diverged from main on the amp-soccer-kick branch,
# same class of miss as the run-motion reset-sampling weight (see
# mdp/events.py git history): a fix landed on main and never got ported over
# here. Confirmed by diff against main: this keyframe block was the only
# unintended difference (the actuator-delay range above is a real,
# intentional kick-specific divergence and was left untouched).
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.3532),
  joint_pos={
    "r_shoulder_pitch_joint": -1.3329,
    "r_shoulder_roll_joint": -1.0420,
    "r_upper_arm_joint": -0.5610,
    "r_elbow_joint": -1.0056,
    "l_shoulder_pitch_joint": 1.3329,
    "l_shoulder_roll_joint": 1.0420,
    "l_upper_arm_joint": 0.5610,
    "l_elbow_joint": 1.0055,
    "r_hip_pitch_joint": 0.5311,
    "r_hip_roll_joint": -0.0981,
    "r_thigh_joint": 0.1370,
    "r_calf_joint": 0.6787,
    "r_ankle_pitch_joint": 0.1294,
    "r_ankle_roll_joint": 0.0000,
    "l_hip_pitch_joint": -0.5311,
    "l_hip_roll_joint": 0.0981,
    "l_thigh_joint": -0.1370,
    "l_calf_joint": -0.6787,
    "l_ankle_pitch_joint": -0.1294,
    "l_ankle_roll_joint": 0.0000,
  },
  joint_vel={".*": 0.0},
)

##
# Collision (all named *_collision[0-9] geoms; feet get friction for contact).
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision[0-9]$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

PIPLUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(PIPLUS_ACTUATOR_ARM, PIPLUS_ACTUATOR_LEG),
  soft_joint_pos_limit_factor=0.9,
)


def get_piplus_robot_cfg() -> EntityCfg:
  """Fresh Pi Plus EntityCfg instance (avoids shared-mutation issues)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PIPLUS_ARTICULATION,
  )


# Action scale for JointPositionActionCfg (smp used a flat 0.1 for Pi Plus).
PIPLUS_ACTION_SCALE: float = 0.1
