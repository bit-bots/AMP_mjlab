"""HighTorque Pi Plus AMP soccer-kick task configuration.

Ball model and spawn/respawn geometry are copied from mjxperiment's kick task
(mujoco_playground/_src/locomotion/piplus/kick.py, read but not edited -- see
src/assets/objects/ball.py and the event functions in
src/tasks/amp_loco/mdp/events.py for exact values/logic mirrored from there).

The reward design is deliberately simpler than mjxperiment's: no commanded kick
direction/speed, no kick-ready/style shaping. The AMP discriminator (trained on
the retargeted CMU soccer-kick clips, src/assets/motions/piplus/amp/Kick/) already
supplies a natural kicking-motion prior; the task reward only needs to steer WHERE
and WHEN that motion fires:
  - move toward the ball instead of tracking a velocity command,
  - reward for the right foot touching the ball, scaled up by the resulting ball
    speed/height for a few seconds after contact (kick_impact_reward, windowed by
    the kick_contact_cycle event), after which the ball resets nearby WITHOUT
    ending the episode -- the whole ball lifecycle is contact-driven, not
    distance-driven (no penalty/reset for the ball just being far away),
  - a penalty when close to the ball if the right foot's inside (medial) edge
    isn't facing it, to discourage toe-poke/front-of-foot contact,
  - the left foot touching the ball ends the episode.
"""

import copy
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import GaussianNoiseCfg

from src.assets.objects import get_ball_cfg
from src.tasks.amp_loco import mdp as amp_mdp
from src.tasks.amp_loco.config.piplus.env_cfgs import (
  ANCHOR_NAME,
  SITE_NAMES,
  piplus_amp_flat_env_cfg,
)

BALL_NAME = "ball"
BALL_ENTITY_BODY_NAME = "ball"  # body name inside the ball's own MJCF (see ball.py)
RIGHT_FOOT_SUBTREE = "r_ankle_roll_link"
LEFT_FOOT_SUBTREE = "l_ankle_roll_link"

# Ball spawn/respawn geometry (mjxperiment kick.py: ball_distance, ball_spawn_cone).
BALL_DIST_RANGE = (0.4, 1.2)
BALL_SPAWN_CONE_DEG = 90.0
# Post-contact reward-farming window before the ball resets in place (episode
# keeps running). Halved from 2.0s.
KICK_WINDOW_S = 1.0
# Bumped from 0.5 so the foot starts turning to line up the medial edge earlier
# in the approach, not just in the last half-meter.
FOOT_ALIGNMENT_CLOSE_DIST = 0.9
# move_to_ball/torso_orient_to_ball measure from a point 20cm to the robot's
# right of the anchor (torso) instead of the anchor's own origin -- lines the
# approach up around the kicking (right) foot's side rather than the body
# center. -Y is the robot's right in the torso's local frame (matches
# foot_medial_alignment's convention: r_ankle_roll_link sits on the -Y side).
KICK_APPROACH_OFFSET_B = (0.0, -0.20, 0.0)


def piplus_amp_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_amp_flat_env_cfg(play=play)

  # --- Ball entity --------------------------------------------------------
  cfg.scene.entities[BALL_NAME] = get_ball_cfg()

  # --- Per-foot ball contact sensors ---------------------------------------
  r_foot_ball_cfg = ContactSensorCfg(
    name="r_foot_ball_contact",
    primary=ContactMatch(mode="subtree", pattern=RIGHT_FOOT_SUBTREE, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=BALL_ENTITY_BODY_NAME, entity=BALL_NAME),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  l_foot_ball_cfg = ContactSensorCfg(
    name="l_foot_ball_contact",
    primary=ContactMatch(mode="subtree", pattern=LEFT_FOOT_SUBTREE, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=BALL_ENTITY_BODY_NAME, entity=BALL_NAME),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (r_foot_ball_cfg, l_foot_ball_cfg)

  # --- Drop the velocity command; add ball-relative observations ----------
  cfg.commands.pop("twist", None)
  cfg.curriculum.pop("command_vel", None)
  ball_obs_terms = {
    "ball_pos_b": ObservationTermCfg(
      func=amp_mdp.object_pos_b,
      params={
        "object_cfg": SceneEntityCfg(BALL_NAME),
        "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      },
    ),
    "ball_lin_vel_b": ObservationTermCfg(
      func=amp_mdp.object_lin_vel_b,
      params={
        "object_cfg": SceneEntityCfg(BALL_NAME),
        "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      },
    ),
  }
  for grp_name in ("actor", "critic"):
    grp = cfg.observations[grp_name]
    grp.terms.pop("command", None)
    # Copy (not share) the term objects -- ball_obs_terms's dict values would
    # otherwise be the SAME ObservationTermCfg instances in both groups, so
    # adding actor-only noise below would leak into the critic's ground truth.
    for name, term in ball_obs_terms.items():
      grp.terms[name] = copy.copy(term)

  # Actor-only ball position noise (critic keeps ground truth): models a noisy
  # ball-tracking perception (e.g. vision) rather than perfect state.
  cfg.observations["actor"].terms["ball_pos_b"].noise = GaussianNoiseCfg(mean=0.0, std=0.07)

  # --- Drop velocity-tracking rewards; add ball-approach + kick rewards ----
  for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity", "foot_slip"):
    cfg.rewards.pop(name, None)
  # Bumped 5x (was 1.0): unlike the sparse contact/impact rewards this one fires
  # every step (dense, bounded [-0.5, 1.0]), so it doesn't need a 40x-style jump
  # to matter -- just enough to make closing the distance clearly worth more
  # than the safe standing-still local optimum.
  cfg.rewards["move_to_ball"] = RewardTermCfg(
    func=amp_mdp.move_toward_object,
    weight=5.0,
    params={
      "object_cfg": SceneEntityCfg(BALL_NAME),
      "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      "max_speed": 0.5,
      "offset_b": KICK_APPROACH_OFFSET_B,
    },
  )
  # Small dense reward for facing the ball (mirrors mjxperiment's
  # _reward_orient_to_ball) -- helps line up a proper kick approach rather than
  # arriving side-on or backing into it.
  cfg.rewards["torso_orient_to_ball"] = RewardTermCfg(
    func=amp_mdp.torso_orient_to_object,
    weight=0.5,
    params={
      "object_cfg": SceneEntityCfg(BALL_NAME),
      "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      "offset_b": KICK_APPROACH_OFFSET_B,
    },
  )
  # Small penalty for foot sliding while grounded. mdp.feet_slip (mjlab base,
  # used by the locomotion task) gates this on a velocity command's magnitude;
  # the kick task has none, so this is the unconditional variant.
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=amp_mdp.foot_slip_penalty,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES),
    },
  )
  # Tiny torque penalty (not present anywhere in the kick task's reward set
  # before this): mdp.joint_torques_l2 is mjlab's built-in L2 actuator-force
  # cost, small enough to just discourage needlessly forceful motion without
  # fighting the kick_impact incentive.
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=amp_mdp.joint_torques_l2,
    weight=-1.0e-5,
  )
  # right_foot_ball_contact/kick_impact bumped ~40x (2.0->80.0, 1.0->40.0): even
  # with AMP back at its original weight, the policy still converged to standing
  # still -- track_root_height/body_ang_vel_xy_l2 already reward a stable stance,
  # and is_terminated=-200 makes any fall-risking approach-and-kick attempt look
  # very unattractive in expectation, so the ball-contact/impact incentives need
  # to be large enough to actually outweigh that safe local optimum.
  #
  # first_object_contact_reward (not object_contact_reward): the continuous
  # per-step version paid out for every step of contact, so the policy learned
  # to just rest the foot on the ball instead of kicking it. One-shot per ball
  # "life" removes that exploit while keeping the same weight.
  cfg.rewards["right_foot_ball_contact"] = RewardTermCfg(
    func=amp_mdp.first_object_contact_reward,
    weight=80.0,
    params={"sensor_name": "r_foot_ball_contact"},
  )
  # weight=10.0 (bumped 5x from 2.0): kick_impact's raw speed+height term is
  # tanh-saturated to [0, 1) (see rewards.py), so this weight IS the max
  # achievable reward (~10.0 for a maxed-out kick) instead of an unbounded
  # multiplier -- still bounded/predictable, just a stronger incentive to
  # actually connect well rather than settle for a token tap.
  cfg.rewards["kick_impact"] = RewardTermCfg(
    func=amp_mdp.kick_impact_reward,
    weight=10.0,
    params={
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "vel_weight": 2.0,
      "height_weight": 5.0,
      "saturation_scale": 5.0,
    },
  )
  # Bumped (was -0.2): a proper side-foot kick now pays off twice over --
  # avoiding this penalty AND collecting the full kick_impact style multiplier
  # above -- while front/back or outside contact are hit by both this penalty
  # AND a heavily discounted (or zero) kick_impact.
  cfg.rewards["foot_ball_alignment"] = RewardTermCfg(
    func=amp_mdp.foot_medial_alignment_penalty,
    weight=-0.5,
    params={
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "foot_cfg": SceneEntityCfg("robot", body_names=(RIGHT_FOOT_SUBTREE,)),
      "medial_sign": 1.0,  # +Y is medial for the right foot (see rewards.py docstring).
      "close_dist": FOOT_ALIGNMENT_CLOSE_DIST,
    },
  )

  # --- Left-foot/ball contact ends the episode -----------------------------
  cfg.terminations["left_foot_touched_ball"] = TerminationTermCfg(
    func=amp_mdp.object_contact,
    params={"sensor_name": "l_foot_ball_contact"},
  )

  # --- Ball spawn (reset) + contact-driven reward window/reset (every step) --
  cfg.events["reset_ball"] = EventTermCfg(
    func=amp_mdp.reset_ball_near_robot,
    mode="reset",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "dist_range": BALL_DIST_RANGE,
      "cone_deg": BALL_SPAWN_CONE_DEG,
    },
  )
  cfg.events["kick_contact_cycle"] = EventTermCfg(
    func=amp_mdp.kick_contact_cycle,
    mode="step",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "contact_sensor_name": "r_foot_ball_contact",
      "foot_body_name": RIGHT_FOOT_SUBTREE,
      "medial_sign": 1.0,  # +Y is medial for the right foot (see rewards.py docstring).
      "window_s": KICK_WINDOW_S,
      "dist_range": BALL_DIST_RANGE,
      "cone_deg": BALL_SPAWN_CONE_DEG,
    },
  )

  # --- RSI + AMP reference clips: kicks + the run trajectory ---------------
  # KickAndRun/ (symlinks into Kick/ + WalkandRun/piplus_run1_subject2.npz, see
  # rl_cfg.py) is also used for RSI now: episodes sometimes start mid-run
  # instead of only mid-kick-sequence, matching the same broadened reference
  # the AMP discriminator (amp_motion_files) trains against.
  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "piplus", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "KickAndRun"))
  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _motion_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  return cfg
