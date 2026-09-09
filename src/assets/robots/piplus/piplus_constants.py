"""HighTorque Pi Plus constants (AMP port).

The robot model is kept byte-identical to the smp/BeyondMimic Pi Plus port
(``piplus.xml`` + meshes copied verbatim). For real-robot deployment the actuators
mirror the bitbots ``mjlab_piplus`` model: ``XmlPositionActuatorCfg`` reads the XML's
``<position>`` gains (kp=30/50, kv=0.6/1.1, forcerange), wrapped in
``DelayedActuatorCfg`` for a randomized 0-3 control-step (0-60 ms @ 50 Hz) actuator
latency. (mjlab 1.2.0 splits into Xml + Delayed cfgs what mjlab_piplus's 1.3.0
``XmlActuatorCfg`` combines.)
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

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
  return spec


##
# Actuators: gains based on the bitbots mjlab_playground pi_plus, leg/hip_roll kp
# bumped 35->50 for stiffer tracking
# (arm kp=6 kv=0.6 effort=10; leg kp=50 kv=1.1; hip_roll kp=50 kv=1.4; effort=20;
# armature from bitbots_main), built-in position servos wrapped in
# DelayedActuatorCfg for a randomized 0-12 physics-step latency (= 0-60 ms / 0-3
# control steps @ 50 Hz; mjlab 1.2.0's BuiltinPositionActuatorCfg has no delay
# field, so the delay is a wrapper).
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

# Hip-roll gets stiffer derivative gain (kv=1.4) for lateral stability.
PIPLUS_ACTUATOR_HIP_ROLL = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    stiffness=50.0,
    damping=1.4,
    effort_limit=20.0,
    armature=0.01316,
    frictionloss=0.2,
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)

PIPLUS_ACTUATOR_LEG = DelayedActuatorCfg(
  base_cfg=BuiltinPositionActuatorCfg(
    target_names_expr=(
      ".*_hip_pitch_joint",
      ".*_thigh_joint",
      ".*_calf_joint",
      ".*_ankle_pitch_joint",
      ".*_ankle_roll_joint",
    ),
    stiffness=50.0,
    damping=1.1,
    effort_limit=20.0,
    armature=0.01316,
    frictionloss=0.2,
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)

##
# Keyframe (dataset-mean pose over WalkandRun -- 10 CMU walk clips + their
# mirrors + the 2 unpaired LAFAN1 run/walk clips, 44503 frames total;
# per-joint, not L/R symmetric). Regenerated after fixing the GMR retargeting
# pipeline's left/right arm offset_quat bugs (see general_motion_retargeting
# git history on piplus-gmr-fixes): the right arm used the same offset_quat
# as the left (wrong -- arms aren't mirror-symmetric at rest like legs are),
# and the left arm's own offset had an unfixed ~180 deg axial twist that a
# T-pose can't observe (so it silently inherited a mismatched convention).
# Not directly comparable to bitbots_main's static walkready pose -- the two
# unpaired LAFAN1 clips are long, dynamic running/walking gaits
# (run1_subject2 alone is 7135 of the 44503 frames), so their swinging-arm
# mean doesn't sit at a static rest pose the way the paired CMU clips' mean
# roughly does.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.3559),
  joint_pos={
    "r_shoulder_pitch_joint": -1.3439,
    "r_shoulder_roll_joint": -1.0447,
    "r_upper_arm_joint": -0.4919,
    "r_elbow_joint": -0.9727,
    "l_shoulder_pitch_joint": 1.3886,
    "l_shoulder_roll_joint": 1.1183,
    "l_upper_arm_joint": 0.4600,
    "l_elbow_joint": 0.8501,
    "r_hip_pitch_joint": 0.4942,
    "r_hip_roll_joint": -0.1108,
    "r_thigh_joint": 0.1501,
    "r_calf_joint": 0.6299,
    "r_ankle_pitch_joint": 0.1735,
    "r_ankle_roll_joint": 0.0000,
    "l_hip_pitch_joint": -0.5046,
    "l_hip_roll_joint": 0.0822,
    "l_thigh_joint": -0.0952,
    "l_calf_joint": -0.6319,
    "l_ankle_pitch_joint": -0.1357,
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
  actuators=(PIPLUS_ACTUATOR_ARM, PIPLUS_ACTUATOR_HIP_ROLL, PIPLUS_ACTUATOR_LEG),
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
