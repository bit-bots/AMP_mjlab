"""HighTorque Pi Plus constants (AMP port).

The robot model is kept byte-identical to the smp/BeyondMimic Pi Plus port
(``piplus.xml`` + meshes copied verbatim). Only the mjlab wiring is adapted to the
AMP repo's mjlab 1.2.0 API: because 1.2.0 lacks ``XmlActuatorCfg``, the actuators
are re-declared with ``BuiltinPositionActuatorCfg`` using the exact gains from the
XML defaults (arm: kp=30 kv=0.6 force=10 armature=0.01317; leg: kp=50 kv=1.1
force=20 armature=0.044277; frictionloss=0.2).
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

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
  # Drop the XML's built-in <position> actuators: mjlab re-adds them from
  # BuiltinPositionActuatorCfg, so keeping the XML ones doubles actuation (nu=40).
  # The XML set would sit at ctrl=0 fighting the policy-driven set -> wrong pose.
  for act in list(spec.actuators):
    spec.delete(act)
  return spec


##
# Actuators (exact gains from piplus.xml <default> classes).
##

PIPLUS_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_upper_arm_joint",
    ".*_elbow_joint",
  ),
  stiffness=30.0,
  damping=0.6,
  effort_limit=10.0,
  armature=0.01317,
  frictionloss=0.2,
)

PIPLUS_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=50.0,
  damping=1.1,
  effort_limit=20.0,
  armature=0.044277,
  frictionloss=0.2,
)

##
# Keyframe (dataset-mean pose from the smp port; per-joint, not L/R symmetric).
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.3413),
  joint_pos={
    "r_shoulder_pitch_joint": 1.2902,
    "r_shoulder_roll_joint": 1.1479,
    "r_upper_arm_joint": -0.1694,
    "r_elbow_joint": 1.8082,
    "l_shoulder_pitch_joint": -1.3218,
    "l_shoulder_roll_joint": -1.0850,
    "l_upper_arm_joint": 0.1584,
    "l_elbow_joint": -1.9507,
    "r_hip_pitch_joint": 0.6691,
    "r_hip_roll_joint": -0.1417,
    "r_thigh_joint": 0.1645,
    "r_calf_joint": 1.0644,
    "r_ankle_pitch_joint": 0.1619,
    "r_ankle_roll_joint": 0.0000,
    "l_hip_pitch_joint": -0.6540,
    "l_hip_roll_joint": 0.1116,
    "l_thigh_joint": -0.1363,
    "l_calf_joint": -1.0866,
    "l_ankle_pitch_joint": -0.1045,
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
