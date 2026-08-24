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
  - reward for EITHER foot touching the ball (foot-agnostic -- ball-velocity-
    based, not tied to a specific foot), scaled up by the resulting ball
    speed/height for a few seconds after contact (kick_impact_reward, windowed by
    the kick_contact_cycle event, which also picks whichever foot was actually
    closer to the ball to score contact quality), after which the ball resets
    nearby WITHOUT ending the episode -- the whole ball lifecycle is
    contact-driven, not distance-driven (no penalty/reset for the ball just
    being far away),
  - a penalty when close to the ball if either foot's inside (medial) edge
    isn't facing it, to discourage toe-poke/front-of-foot contact with either
    foot,
  - no termination for kicking with the "wrong" foot -- there isn't one.
"""

import copy
import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp.rewards import feet_air_time
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

# Ball spawn/respawn geometry: forward distance from the robot + a
# distance-scaled lateral offset (see _place_ball_near_robot, mdp/events.py)
# -- always somewhere in front of the robot, narrower close-in and wider
# further out. Replaces an earlier design that walked the robot to a
# structured "approach point" behind the ball before allowing contact; that
# extra state machine was dropped as unnecessary -- the ordinary
# move_to_ball/torso_orient_to_ball rewards already handle getting the robot
# to the ball without it.
BALL_DIST_RANGE = (0.3, 1.0)
BALL_LATERAL_RANGE = (0.2, 0.7)  # (near, far) lateral half-width, meters.
# Target kick direction sampled within the robot's heading at spawn time +/-
# this cone -- a direction the robot has a plausible chance of actually
# hitting from its current orientation, rather than a fully arbitrary one.
KICK_DIR_CONE_DEG = 80.0
# Post-contact reward-farming window before the ball resets in place (episode
# keeps running). Halved from 2.0s.
KICK_WINDOW_S = 1.0
# Bumped from 0.5 so the foot starts turning to line up the medial edge earlier
# in the approach, not just in the last half-meter.
FOOT_ALIGNMENT_CLOSE_DIST = 0.9
# move_to_ball/torso_orient_to_ball measure from a point 20cm to the side of
# the anchor (torso) instead of the anchor's own origin -- lines the approach
# up around whichever foot select_kick_foot_is_left currently picks, rather
# than the body center. -Y is the robot's right (matches
# foot_medial_alignment's convention: r_ankle_roll_link sits on the -Y side),
# so a right-foot selection gets a negative offset, a left-foot selection a
# positive one -- recomputed every step from the CURRENT ball position and
# target kick angle, not a fixed side.
KICK_APPROACH_OFFSET_MAGNITUDE = 0.20
# select_kick_foot_is_left's hard decision boundary (mdp/rewards.py): within
# this many degrees of straight ahead, pick the foot on the ball's own side;
# beyond it, pick by kick angle alone (a left-angled kick needs the right
# foot to swing across the body, a right-angled kick needs the left foot).
# Interactively tuned/validated as a hard rule, not a blend -- see the kick
# foot selector debug tool. Shared by the approach offset above,
# kick_contact_cycle's foot-correctness gating, and ball_kick_contact's
# reward/penalty, so all three agree on which foot is "correct" at any
# given moment.
KICK_FOOT_SELECT_ANGLE_THRESHOLD_DEG = 20.0
# kick_direction_reward/kick_impact_reward's angular tolerance (Gaussian
# sigma, radians), narrowed over training via a curriculum -- see
# anneal_reward_param_linear below. Was 45deg -> 5deg over iterations
# 2000-7000, but both rewards are SIGNED (_kick_dir_factor ranges -1 to +1),
# and the log showed kick_impact/kick_direction flipping from clearly
# positive to negative right at iteration 7000 -- exactly when the tolerance
# finished tightening to a very demanding 5deg, before the policy could
# reliably hit it, making kicking net-negative in expectation and collapsing
# the incentive to commit to a full kick. Now a longer runway (2000-14000)
# and a less extreme final tolerance (15deg).
KICK_DIR_SIGMA_START = math.radians(45.0)
KICK_DIR_SIGMA_END = math.radians(15.0)


def piplus_amp_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_amp_flat_env_cfg(play=play)

  # --- Ball entity --------------------------------------------------------
  cfg.scene.entities[BALL_NAME] = get_ball_cfg()

  # --- Per-foot ball contact sensors ---------------------------------------
  # Used only by cross_foot_touch_penalty below (which foot touched the ball
  # first each life) -- NOT for a foot-restriction termination, there isn't
  # one anymore.
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
    # One-hot [kicking, post_kick] -- see kick_state_obs docstring. No
    # noise: this is task-internal state, not a perception.
    "kick_state": ObservationTermCfg(func=amp_mdp.kick_state_obs),
    # Target kick direction (env.kick_dir_world), rotated into the robot
    # base's yaw frame -- see kick_dir_heading_b docstring. Without this the
    # policy has no way to know which direction is currently being asked
    # for (kick_direction/kick_impact score against it, but nothing exposed
    # it as an observation before). Critic gets this clean copy; actor's is
    # overridden below with a noisy variant.
    "kick_dir_b": ObservationTermCfg(
      func=amp_mdp.kick_dir_heading_b,
      params={"robot_cfg": SceneEntityCfg("robot")},
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
  # Actor-only random observation delay, up to 0.2s (step_dt=0.02s -> 10
  # steps), re-sampled per env every step (mjlab's built-in delay pipeline,
  # see ObservationTermCfg.delay_min_lag/delay_max_lag) -- models the ball
  # position lagging behind a real perception pipeline. Critic keeps
  # instantaneous ground truth.
  cfg.observations["actor"].terms["ball_pos_b"].delay_max_lag = 10
  # Actor-only noisy kick-direction observation: Gaussian noise on the
  # underlying yaw angle (not the raw vector -- see kick_dir_heading_b_noisy
  # docstring for why), std ~5deg. Overrides the clean copy from the loop
  # above (can't use the generic per-component GaussianNoiseCfg here without
  # corrupting the unit-norm representation).
  cfg.observations["actor"].terms["kick_dir_b"] = ObservationTermCfg(
    func=amp_mdp.kick_dir_heading_b_noisy,
    params={"robot_cfg": SceneEntityCfg("robot"), "std_rad": math.radians(5.0)},
    # Same random up-to-0.2s delay as ball_pos_b above -- delaying a buffered
    # sequence of already-unit-norm vectors just returns an older valid one,
    # so this doesn't reintroduce the norm-corruption issue noise did.
    delay_max_lag=10,
  )

  # --- Drop velocity-tracking rewards; add ball-approach + kick rewards ----
  for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity", "foot_slip"):
    cfg.rewards.pop(name, None)
  # History: 1.0 -> 5.0 -> 12.0 -> 6.0 -> 12.0 -> 6.0 -> 4.0 -> 6.0 -> 4.0 --
  # simplified back to a direct move-toward-the-ball reward (no more
  # structured "approach point" state, see BALL_DIST_RANGE/BALL_LATERAL_RANGE
  # above) now that kick_contact_cycle no longer gates contact on a
  # position/orientation "ready" state -- the robot may kick as soon as it
  # reaches the ball, so there's nothing left for an approach point to set up.
  cfg.rewards["move_to_ball"] = RewardTermCfg(
    func=amp_mdp.move_toward_object,
    weight=4.0,
    params={
      "object_cfg": SceneEntityCfg(BALL_NAME),
      "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      "adaptive_foot_offset": KICK_APPROACH_OFFSET_MAGNITUDE,
      "foot_select_angle_threshold_deg": KICK_FOOT_SELECT_ANGLE_THRESHOLD_DEG,
      "alignment_power": 3.0,
    },
  )
  # Significant flat penalty for standing still before the ball's been
  # touched (per ball "life", so this re-arms after every kick too) -- the
  # standing-still local optimum kept surviving previous reward-scale fixes,
  # so this directly targets it rather than relying on move_to_ball's
  # incentive alone.
  cfg.rewards["standing_still"] = RewardTermCfg(
    func=amp_mdp.standing_still_penalty,
    weight=-5.0,
    params={
      "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      "vel_threshold": 0.15,
    },
  )
  # Small dense reward for facing the ball (mirrors mjxperiment's
  # _reward_orient_to_ball) -- helps line up a proper kick approach rather
  # than arriving side-on or backing into it. Plain "face the ball" the whole
  # time now (no more switching to face the target kick direction once
  # "ready" -- that state no longer exists, see move_to_ball above).
  cfg.rewards["torso_orient_to_ball"] = RewardTermCfg(
    func=amp_mdp.torso_orient_to_object,
    weight=0.5,
    params={
      "object_cfg": SceneEntityCfg(BALL_NAME),
      "anchor_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,)),
      "adaptive_foot_offset": KICK_APPROACH_OFFSET_MAGNITUDE,
      "foot_select_angle_threshold_deg": KICK_FOOT_SELECT_ANGLE_THRESHOLD_DEG,
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
  # Back to mjlab's standard velocity-task gait reward (was briefly a
  # short-air-time-only penalty): +1 per foot whose current swing (air time)
  # falls in a MEDIUM range (0.05s-0.5s, mjlab's own defaults), no
  # command_name (the kick task has none, matching foot_slip above). Bumped
  # from 1.0.
  cfg.rewards["foot_air_time"] = RewardTermCfg(
    func=feet_air_time,
    weight=2.5,
    params={"sensor_name": "feet_ground_contact"},
  )
  # Tiny torque penalty (not present anywhere in the kick task's reward set
  # before this): mdp.joint_torques_l2 is mjlab's built-in L2 actuator-force
  # cost, small enough to just discourage needlessly forceful motion without
  # fighting the kick_impact incentive.
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=amp_mdp.joint_torques_l2,
    weight=-1.0e-5,
  )
  # ball_kick_contact/kick_impact bumped ~40x (2.0->80.0, 1.0->40.0): even
  # with AMP back at its original weight, the policy still converged to standing
  # still -- track_root_height/body_ang_vel_xy_l2 already reward a stable stance,
  # and is_terminated=-200 makes any fall-risking approach-and-kick attempt look
  # very unattractive in expectation, so the ball-contact/impact incentives need
  # to be large enough to actually outweigh that safe local optimum.
  #
  # ball_kick_contact_reward (not object_contact_reward, not
  # first_object_contact_reward): the continuous per-step version paid out
  # for every step of contact, so the policy learned to just rest the foot on
  # the ball instead of kicking it -- one-shot per ball "life" removes that
  # exploit. Sourced from kick_contact_cycle's ball-velocity-based contact
  # detection (not a foot/ball contact sensor: a fast kick can make contact
  # and separate again within a single RL step's physics substeps, which a
  # sensor sampled once per RL step can silently miss even though the kick
  # was genuine -- confirmed empirically, see kick_contact_cycle's
  # docstring). Foot-gated: full reward only if the foot that actually struck
  # the ball was the one select_kick_foot_is_left says should have been used
  # (given the ball's position/target angle at that moment); a 50% penalty
  # (wrong_foot_penalty) if the other foot struck it instead.
  cfg.rewards["ball_kick_contact"] = RewardTermCfg(
    func=amp_mdp.ball_kick_contact_reward,
    weight=80.0,
    params={"wrong_foot_penalty": 0.5},
  )
  # weight=10.0 (bumped 5x from 2.0): kick_impact's raw speed+height term is
  # tanh-saturated to [0, 1) (see rewards.py), so this weight IS the max
  # achievable reward (~10.0 for a maxed-out kick) instead of an unbounded
  # multiplier -- still bounded/predictable, just a stronger incentive to
  # actually connect well rather than settle for a token tap. Also gated by
  # dir_sigma (see _kick_dir_factor, rewards.py): a wrong-direction kick isn't
  # just under-rewarded, it goes negative once outside the tolerance cone.
  cfg.rewards["kick_impact"] = RewardTermCfg(
    func=amp_mdp.kick_impact_reward,
    weight=10.0,
    params={
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "vel_weight": 2.0,
      "height_weight": 5.0,
      "saturation_scale": 5.0,
      "dir_sigma": KICK_DIR_SIGMA_START,
    },
  )
  # Rewards/penalizes the ball's post-kick direction against a randomly
  # sampled target (env.kick_dir_world, resampled every ball reset -- see
  # events.py). sigma starts very broad (direction essentially doesn't matter)
  # and narrows over training via the narrow_kick_direction curriculum below.
  cfg.rewards["kick_direction"] = RewardTermCfg(
    func=amp_mdp.kick_direction_reward,
    weight=2.0,
    params={
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "sigma": KICK_DIR_SIGMA_START,
      "min_ball_speed": 0.1,
    },
  )
  # Bumped (was -0.2): a proper side-foot kick now pays off twice over --
  # avoiding this penalty AND collecting the full kick_impact style multiplier
  # above -- while front/back or outside contact are hit by both this penalty
  # AND a heavily discounted (or zero) kick_impact. Extended to both feet
  # independently (was right-foot only) -- kicking is no longer restricted
  # to one foot, so misalignment should be checked/penalized for whichever
  # foot is actually close to the ball, not just the right one.
  cfg.rewards["foot_ball_alignment_right"] = RewardTermCfg(
    func=amp_mdp.foot_medial_alignment_penalty,
    weight=-0.5,
    params={
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "foot_cfg": SceneEntityCfg("robot", body_names=(RIGHT_FOOT_SUBTREE,)),
      "medial_sign": 1.0,  # +Y is medial for the right foot (see rewards.py docstring).
      "close_dist": FOOT_ALIGNMENT_CLOSE_DIST,
    },
  )
  cfg.rewards["foot_ball_alignment_left"] = RewardTermCfg(
    func=amp_mdp.foot_medial_alignment_penalty,
    weight=-0.5,
    params={
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "foot_cfg": SceneEntityCfg("robot", body_names=(LEFT_FOOT_SUBTREE,)),
      "medial_sign": -1.0,  # -Y is medial for the left foot (see rewards.py docstring).
      "close_dist": FOOT_ALIGNMENT_CLOSE_DIST,
    },
  )
  # Discourage double-touching the ball with both feet: once one foot has
  # touched it first (this ball "life"), the other foot subsequently also
  # touching is penalized -- kicking should be a single clean strike with
  # one foot, not both feet fumbling at it.
  cfg.rewards["cross_foot_touch"] = RewardTermCfg(
    func=amp_mdp.cross_foot_touch_penalty,
    weight=-30.0,
    params={
      "right_sensor_name": "r_foot_ball_contact",
      "left_sensor_name": "l_foot_ball_contact",
    },
  )

  # --- Ball spawn (reset) + contact-driven reward window/reset (every step) --
  cfg.events["reset_ball"] = EventTermCfg(
    func=amp_mdp.reset_ball_near_robot,
    mode="reset",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      "dist_range": BALL_DIST_RANGE,
      "lateral_range": BALL_LATERAL_RANGE,
      "kick_dir_cone_deg": KICK_DIR_CONE_DEG,
    },
  )
  cfg.events["kick_contact_cycle"] = EventTermCfg(
    func=amp_mdp.kick_contact_cycle,
    mode="step",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "ball_cfg": SceneEntityCfg(BALL_NAME),
      # Both feet, foot-agnostic: at the moment of a kick, style (contact
      # quality) is scored using whichever of these two is actually closer
      # to the ball, not always the right foot -- see kick_contact_cycle's
      # docstring (mdp/events.py).
      "foot_body_names": (RIGHT_FOOT_SUBTREE, LEFT_FOOT_SUBTREE),
      "medial_signs": (1.0, -1.0),  # +Y medial for right, -Y medial for left (see rewards.py docstring).
      "window_s": KICK_WINDOW_S,
      "dist_range": BALL_DIST_RANGE,
      "lateral_range": BALL_LATERAL_RANGE,
      "kick_dir_cone_deg": KICK_DIR_CONE_DEG,
      "anchor_body_name": ANCHOR_NAME,
      "angle_threshold_deg": KICK_FOOT_SELECT_ANGLE_THRESHOLD_DEG,
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

  # --- Curricula -------------------------------------------------------------
  # Tighten the foot-slip penalty at iteration 10000, once kicking is
  # established, to clean up the gait rather than fight exploration early on.
  # (joint_torques_l2 used to also bump here -- moved to its own, much
  # earlier and stronger bump below, alongside action_rate_l2.)
  cfg.curriculum["bump_penalties"] = CurriculumTermCfg(
    func=amp_mdp.bump_reward_weight_at_step,
    params={
      "reward_names": ["foot_slip"],
      "step": 10000 * 24,
      "scale": 3.0,
    },
  )
  # Tighten the torque/action-rate penalties at iteration 3000 -- much
  # earlier than the foot-slip bump above, to clamp down on jerky/forceful
  # motion well before the kick-direction curriculum below starts demanding
  # accuracy. scale=8.0 turned out too strong; brought down to a middle
  # ground between that and no bump at all.
  cfg.curriculum["bump_control_penalties"] = CurriculumTermCfg(
    func=amp_mdp.bump_reward_weight_at_step,
    params={
      "reward_names": ["joint_torques_l2", "action_rate_l2"],
      "step": 3000 * 24,
      "scale": 4.0,
    },
  )
  # Completely drop ball_kick_contact's one-shot reward at iteration 2000 --
  # it exists purely to bootstrap discovering ball contact at all; once
  # that's established this early, kick_impact/kick_direction (which score
  # contact QUALITY, not just its occurrence) should drive the rest without
  # a flat +80 one-off still on offer.
  cfg.curriculum["drop_ball_kick_contact"] = CurriculumTermCfg(
    func=amp_mdp.bump_reward_weight_at_step,
    params={
      "reward_names": ["ball_kick_contact"],
      "step": 2000 * 24,
      "scale": 0.0,
    },
  )
  # Narrow kick_direction_reward's tolerance (45deg -> 15deg) over iterations
  # 2000-14000 -- was 2000-7000 -> 5deg, but that let the tolerance finish
  # tightening well before the policy could reliably hit it, flipping
  # kick_impact/kick_direction (both SIGNED, -1 to +1) from net-positive to
  # net-negative right at iteration 7000 and collapsing the incentive to
  # commit to a full kick (see git history/commit message for the log
  # evidence). Longer runway + less extreme final tolerance this time.
  # kick_impact's own dir_sigma is narrowed on the exact same schedule so the
  # "wrong direction gets penalized" gate in both terms tightens together.
  cfg.curriculum["narrow_kick_direction"] = CurriculumTermCfg(
    func=amp_mdp.anneal_reward_param_linear,
    params={
      "reward_name": "kick_direction",
      "param_name": "sigma",
      "start_step": 2000 * 24,
      "end_step": 14000 * 24,
      "start_value": KICK_DIR_SIGMA_START,
      "end_value": KICK_DIR_SIGMA_END,
    },
  )
  cfg.curriculum["narrow_kick_impact_dir_gate"] = CurriculumTermCfg(
    func=amp_mdp.anneal_reward_param_linear,
    params={
      "reward_name": "kick_impact",
      "param_name": "dir_sigma",
      "start_step": 2000 * 24,
      "end_step": 14000 * 24,
      "start_value": KICK_DIR_SIGMA_START,
      "end_value": KICK_DIR_SIGMA_END,
    },
  )

  return cfg
