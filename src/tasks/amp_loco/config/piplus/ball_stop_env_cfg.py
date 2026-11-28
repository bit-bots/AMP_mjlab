"""HighTorque Pi Plus AMP ball-stopping task configuration.

One ball per episode. It lies a few meters in front of the robot while the
robot stands still (idle), is then rolled at the robot with a Gaussian lateral
aim offset, and the robot should stop it. After contact the robot should hold
still; after a 1-2 s hold the episode ends with a success bonus. A ball that
passes, stops short or rolls away ends the episode without a penalty. Phase
logic: ball_stop_cycle / reset_ball_stop (mdp/events.py).

Built on the flat AMP locomotion env, so the AMP discriminator reward (added
by the AMP runner, see rl_cfg.py) and the base regularization rewards stay in.

Shared with the kick task (kick_env_cfg.py): action clipping to joint limits,
flat turf terrain contact model, the ball entity and its physics randomization,
per-foot ball contact sensors and the noisy, delayed actor ball observation
(here with a std that grows with the ball distance).

Removed from the walk task: the velocity command, its observation, the
velocity-tracking rewards and the command-gated foot slip penalty.
"""

import copy
import dataclasses
import os

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.objects import BALL_GEOM_NAME, get_ball_cfg, get_ball_ghost_cfg
from src.tasks.amp_loco import mdp as amp_mdp
from src.tasks.amp_loco.config.piplus.env_cfgs import (
  ANCHOR_NAME,
  ROOT_NAME,
  SITE_NAMES,
  piplus_amp_flat_env_cfg,
)
from src.tasks.amp_loco.config.piplus.kick_env_cfg import (
  _BALL_SIZE_MASS_DIST,
  BALL_ENTITY_BODY_NAME,
  BALL_NAME,
  BALL_OBS_MAX_LAG_STEPS,
  BALL_ROLLING_FRICTION_RANGE,
  BALL_SIZE_MASS_SCALE_RANGE,
  BALL_SLIDING_FRICTION_RANGE,
  BALL_TORSIONAL_FRICTION_RANGE,
  LEFT_FOOT_SUBTREE,
  RIGHT_FOOT_SUBTREE,
)
from src.tasks.amp_loco.mdp.terrain import (
  randomize_terrain_contact,
  terrain_collisions,
)

# --- Ball (see reset_ball_stop / ball_stop_cycle, mdp/events.py) ----------
# Spawn distance from the robot base (m).
BALL_SPAWN_DIST_RANGE = (2.0, 4.0)
# Bearing of the spawn point from the robot's heading, +/- deg (0 = straight
# in front). Set by the ball_bearing curriculum below from these stages:
# (iteration, bearing_deg). Only "from the front" for now; add stages later,
# e.g. [(0, 0.0), (5000, 30.0), (10000, 90.0)].
BALL_BEARING_STAGES = [(0, 0.0)]
# Robot stands still this long (s) before the ball is launched. Was
# (1.0, 3.0); shortened so the ball does not just lie there for long.
BALL_IDLE_TIME_RANGE_S = (0.2, 0.6)
# After the first contact the robot holds this long (s), then success.
BALL_HOLD_TIME_RANGE_S = (1.0, 2.0)
# Ball speed when it reaches the robot (m/s, approximate).
BALL_ARRIVAL_SPEED_RANGE = (0.5, 2.0)
# Lateral aim offset at the robot base: N(0, BALL_AIM_STD) meters, not
# clipped (BALL_AIM_CLIP = None; a float clips to +/- that value).
BALL_AIM_CLIP = None
# Aim sigma curriculum (m): next stage once at least BALL_AIM_PROMOTE_RATE of
# the ended episodes were a success with the ball NOT behind the robot's
# start position at the end (not past it along the launch direction;
# distance and speed do not matter). The
# threshold 0.65. Checked every
# BALL_AIM_EVAL_EPISODES ended episodes (~45 iterations at 4096 envs).
# Stage 3 keeps sigma 0.5 and adds harder balls (BALL_STAGE_PARAMS).
BALL_AIM_STD_STAGES = [0.25, 0.375, 0.5, 0.5]
BALL_AIM_PROMOTE_RATE = 0.65
BALL_AIM_EVAL_EPISODES = 20_000
BALL_AIM_STD = BALL_AIM_STD_STAGES[0]
# Stage 3, sampled per ball (the other balls stay as in stage 2):
# - BALL_FAR_PROB of the balls spawn BALL_FAR_DIST_RANGE m away and arrive
#   with BALL_FAR_ARRIVAL_SPEED_RANGE m/s (faster than normal).
# - BALL_LOFT_PROB of the other balls are launched upward at
#   BALL_LOFT_ANGLE_DEG_RANGE deg with BALL_LOFT_SPEED_SCALE_RANGE times the
#   normal launch speed; they bounce before they roll.
BALL_FAR_PROB = 0.2
BALL_FAR_DIST_RANGE = (4.0, 6.0)
BALL_FAR_ARRIVAL_SPEED_RANGE = (1.5, 3.0)
BALL_LOFT_PROB = 0.15
BALL_LOFT_ANGLE_DEG_RANGE = (10.0, 25.0)
BALL_LOFT_SPEED_SCALE_RANGE = (1.2, 1.5)
BALL_STAGE_PARAMS = {
  3: {
    "reset_ball": {"far_prob": BALL_FAR_PROB},
    "ball_stop_cycle": {"loft_prob": BALL_LOFT_PROB},
  },
}
# Ball friction: kick-env ranges (log-uniform), but a share of the balls
# (BALL_LOW_FRICTION_PROB) gets a range BALL_LOW_FRICTION_FACTOR times lower
# instead (log-uniform in [lo / factor, lo]): slides more on the foot and
# rolls further. Applied to sliding and rolling friction. Resampled at
# every reset (was once at startup).
BALL_LOW_FRICTION_PROB = 0.3
BALL_LOW_FRICTION_FACTOR = 3.0
# Actor ball position noise: std = base + per_m * distance (m).
BALL_OBS_STD_BASE = 0.0175  # = kick env std
BALL_OBS_STD_PER_M = 0.02

# Reference clips for RSI (reset_from_motion), subfolder of
# src/assets/motions/piplus/amp/. Keep it the same as the AMP discriminator's
# clips (rl_cfg.py, BALL_STOP_AMP_MOTION_SUBDIR), so episodes start from poses
# the discriminator knows.
BALL_STOP_MOTION_SUBDIR = "BallStop2"


def _low_tail_log_uniform_sample(lower, upper, shape, device):
  """Log-uniform in [lower, upper]; with prob BALL_LOW_FRICTION_PROB
  log-uniform in [lower / BALL_LOW_FRICTION_FACTOR, lower] instead."""
  lower = torch.as_tensor(lower, device=device, dtype=torch.float32)
  upper = torch.as_tensor(upper, device=device, dtype=torch.float32)
  u = torch.rand(shape, device=device)
  normal = torch.exp(torch.log(lower) + u * (torch.log(upper) - torch.log(lower)))
  low_lo = lower / BALL_LOW_FRICTION_FACTOR
  low = torch.exp(torch.log(low_lo) + u * (torch.log(lower) - torch.log(low_lo)))
  pick_low = torch.rand(shape, device=device) < BALL_LOW_FRICTION_PROB
  return torch.where(pick_low, low, normal)


_LOW_TAIL_LOG_UNIFORM = dr.Distribution(
  name="low_tail_log_uniform", sample=_low_tail_log_uniform_sample
)


def piplus_amp_ball_stop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_amp_flat_env_cfg(play=play)

  # Clip the policy action to the joint limits (see kick_env_cfg.py).
  _a = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = amp_mdp.LimitedJointPositionActionCfg(
    entity_name=_a.entity_name,
    actuator_names=_a.actuator_names,
    scale=_a.scale,
    use_default_offset=_a.use_default_offset,
  )

  # --- Terrain: flat turf, same contact model as the kick task -------------
  cfg.scene.terrain.collisions = terrain_collisions()
  _robot = cfg.scene.entities["robot"]
  _robot.collisions = tuple(
    dataclasses.replace(_c, priority=0) for _c in _robot.collisions
  )
  cfg.events["terrain_contact"] = EventTermCfg(
    func=randomize_terrain_contact,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("terrain"),
      "solref_ranges": {0: (0.006, 0.03), 1: (0.95, 1.05)},
      "solimp_ranges": {0: (0.88, 0.92), 1: (0.94, 0.99), 2: (0.003, 0.01)},
      "shared_random": True,
    },
  )
  del cfg.events["foot_friction"]
  cfg.events["floor_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)),
      "operation": "abs",
      "ranges": (0.75, 1.25),
      "shared_random": True,
    },
  )

  # --- Ball entity + physics randomization (same as kick) ------------------
  cfg.scene.entities[BALL_NAME] = get_ball_cfg()
  _ball_geom = SceneEntityCfg(BALL_NAME, geom_names=(BALL_GEOM_NAME,))
  cfg.events["ball_size"] = EventTermCfg(
    mode="startup",
    func=dr.geom_size,
    params={
      "asset_cfg": copy.deepcopy(_ball_geom),
      "operation": "scale",
      "distribution": _BALL_SIZE_MASS_DIST,
      "ranges": BALL_SIZE_MASS_SCALE_RANGE,
      "axes": [0],
    },
  )
  cfg.events["ball_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg(BALL_NAME, body_names=(BALL_ENTITY_BODY_NAME,)),
      "operation": "scale",
      "distribution": _BALL_SIZE_MASS_DIST,
      "ranges": BALL_SIZE_MASS_SCALE_RANGE,
    },
  )
  for _name, _range, _axis, _dist in (
    ("ball_sliding_friction", BALL_SLIDING_FRICTION_RANGE, 0, _LOW_TAIL_LOG_UNIFORM),
    ("ball_torsional_friction", BALL_TORSIONAL_FRICTION_RANGE, 1, "log_uniform"),
    ("ball_rolling_friction", BALL_ROLLING_FRICTION_RANGE, 2, _LOW_TAIL_LOG_UNIFORM),
  ):
    cfg.events[_name] = EventTermCfg(
      mode="reset",
      func=dr.geom_friction,
      params={
        "asset_cfg": copy.deepcopy(_ball_geom),
        "operation": "abs",
        "distribution": _dist,
        "ranges": _range,
        "axes": [_axis],
      },
    )

  # --- Per-foot ball contact sensors (for task rewards) --------------------
  # history_length=4 (= decimation) so a short touch within one control step
  # is not missed (see contact_sensor_touched, mdp/rewards.py).
  def _foot_ball_sensor(name: str, foot: str) -> ContactSensorCfg:
    return ContactSensorCfg(
      name=name,
      primary=ContactMatch(mode="subtree", pattern=foot, entity="robot"),
      secondary=ContactMatch(mode="body", pattern=BALL_ENTITY_BODY_NAME, entity=BALL_NAME),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    _foot_ball_sensor("r_foot_ball_contact", RIGHT_FOOT_SUBTREE),
    _foot_ball_sensor("l_foot_ball_contact", LEFT_FOOT_SUBTREE),
    # Any robot body touching the ball: drives the contact phase.
    _foot_ball_sensor("robot_ball_contact", ROOT_NAME),
  )

  # --- Observations: drop the velocity command, add the ball ---------------
  cfg.commands.pop("twist", None)
  cfg.curriculum.pop("command_vel", None)
  _anchor = SceneEntityCfg("robot", body_names=(ANCHOR_NAME,))
  for grp_name in ("actor", "critic"):
    grp = cfg.observations[grp_name]
    grp.terms.pop("command", None)
    grp.terms["ball_pos_b"] = ObservationTermCfg(
      func=amp_mdp.object_pos_b,
      params={"object_cfg": SceneEntityCfg(BALL_NAME), "anchor_cfg": copy.deepcopy(_anchor)},
    )
  # Ball velocity: critic only (privileged). The actor infers it from its
  # ball position history (history_length=4).
  cfg.observations["critic"].terms["ball_lin_vel_b"] = ObservationTermCfg(
    func=amp_mdp.object_lin_vel_b,
    params={"object_cfg": SceneEntityCfg(BALL_NAME), "anchor_cfg": copy.deepcopy(_anchor)},
  )
  # Ball-stopping phase + progress: critic only (privileged, the real robot
  # has no phase/timer). See ball_stop_state_obs (mdp/observations.py).
  cfg.observations["critic"].terms["ball_stop_state"] = ObservationTermCfg(
    func=amp_mdp.ball_stop_state_obs,
  )
  # Actor ball position: noisy and delayed in the world frame, like the kick
  # task, but the isotropic std grows with the ball distance
  # (BALL_OBS_STD_BASE + BALL_OBS_STD_PER_M * r: ~2 cm close like the kick
  # env, ~10 cm at 4 m).
  # The range noise (relative, 1 %) also grows with the distance. No
  # post-kick phase here (no kick_timer), so std_post_kick never applies.
  cfg.observations["actor"].terms["ball_pos_b"] = ObservationTermCfg(
    func=amp_mdp.object_pos_b_noisy_delayed_world,
    params={
      "object_cfg": SceneEntityCfg(BALL_NAME),
      "anchor_cfg": copy.deepcopy(_anchor),
      "std": BALL_OBS_STD_BASE,
      "std_post_kick": BALL_OBS_STD_BASE,
      "std_per_m": BALL_OBS_STD_PER_M,
      "range_noise_std": 0.01,
      "bearing_noise_std_deg": 2.0,
      "max_lag": BALL_OBS_MAX_LAG_STEPS,
    },
  )

  # --- Rewards ------------------------------------------------------------
  # Velocity tracking needs the twist command, which is gone. foot_slip is
  # swapped for the command-free variant.
  for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity", "foot_slip"):
    cfg.rewards.pop(name, None)
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=amp_mdp.foot_slip_penalty,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES),
    },
  )
  # Kept from the AMP base env: track_root_height, body_ang_vel_xy_l2,
  # joint_acc_l2, joint_pos_limits, action_rate_l2, self_collisions. The AMP
  # style reward is added by the runner.
  #
  # is_terminated (-200) would also hit the success and missed-ball
  # terminations; only falling is penalized now.
  cfg.rewards["is_terminated"] = RewardTermCfg(
    func=amp_mdp.terminated_by,
    weight=-200.0,
    params={"term_names": ["bad_orientation", "bad_base_height"]},
  )

  # --- Task rewards ----------------------------------------------------------
  # mjlab scales every reward by dt (0.02 s): a per-step term with weight w
  # gives up to w per second, a one-shot term gives w * 0.02 once.
  _ball = SceneEntityCfg(BALL_NAME)
  # Idle (ball not launched yet): stand still, exp(-(|v_xy|^2 + w_z^2)/std^2).
  cfg.rewards["bs_idle_still"] = RewardTermCfg(
    func=amp_mdp.ball_stop_idle_still,
    weight=1.0,
    params={"std": 0.2},
  )
  # Idle: penalty per meter the base moved away from its start point beyond
  # 0.25 m (RSI drift tolerance) -- the robot must wait for the ball, not run
  # to it. -10 per meter and second, e.g. 1 m too far for 2 s costs
  # 0.75 * 10 * 2 = 15, three times the contact bonus.
  cfg.rewards["bs_idle_displacement"] = RewardTermCfg(
    func=amp_mdp.ball_stop_idle_displacement,
    weight=-10.0,
    params={"tolerance": 0.25},
  )
  # Ball incoming: move toward the closest point on the ball's path (original
  # intercept, lateral_only=False). Normalized to [0, 1] (1 at max_speed, or
  # within on_line_tol of the path). lateral_only=True: only sideways steps
  # toward where the path crosses the start line count.
  cfg.rewards["bs_intercept"] = RewardTermCfg(
    func=amp_mdp.ball_stop_intercept,
    weight=2.0,
    params={
      "max_speed": 0.5,
      "on_line_tol": 0.05,
      "start_line_tol": 0.15,
      # False: original intercept (toward the closest point on the path).
      "lateral_only": False,
    },
  )
  # First robot-ball contact: one-shot, 250 * 0.02 = 5.
  cfg.rewards["bs_contact"] = RewardTermCfg(
    func=amp_mdp.ball_stop_contact,
    weight=250.0,
  )
  # After contact: ball still rolling (|v_xy|, m/s), x2 while the ball is
  # behind the robot. -2 per m/s and second: a ball still rolling at 1 m/s
  # behind the robot for 1 s costs 4.
  cfg.rewards["bs_post_contact_ball_speed"] = RewardTermCfg(
    func=amp_mdp.ball_stop_post_contact_ball_speed,
    weight=-2.0,
    params={"ball_cfg": copy.deepcopy(_ball), "behind_scale": 2.0},
  )
  # After contact: robot base still moving (|v_xy|^2 + w_z^2).
  cfg.rewards["bs_post_contact_robot_motion"] = RewardTermCfg(
    func=amp_mdp.ball_stop_post_contact_robot_motion,
    weight=-2.0,
  )
  # Hold time after contact passed: one-shot success bonus, 250 * 0.02 = 5.
  cfg.rewards["bs_success"] = RewardTermCfg(
    func=amp_mdp.ball_stop_success,
    weight=250.0,
  )

  # --- Terminations ------------------------------------------------------------
  # Both are real terminations (no value bootstrapping), not time-outs.
  cfg.terminations["ball_stop_success"] = TerminationTermCfg(
    func=amp_mdp.ball_stop_done_success,
  )
  cfg.terminations["ball_missed"] = TerminationTermCfg(
    func=amp_mdp.ball_stop_done_missed,
  )
  # Idle (<= 3 s) + roll (~2-4 s) + hold (<= 2 s) fits well within this.
  if not play:
    cfg.episode_length_s = 12.0

  # --- Ball: one per episode -------------------------------------------------
  # Must come after reset_from_motion (dict order = event order).
  cfg.events["reset_ball"] = EventTermCfg(
    func=amp_mdp.reset_ball_stop,
    mode="reset",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "ball_cfg": copy.deepcopy(_ball),
      "spawn_dist_range": BALL_SPAWN_DIST_RANGE,
      "spawn_bearing_deg": BALL_BEARING_STAGES[0][1],
      "idle_time_range_s": BALL_IDLE_TIME_RANGE_S,
      "hold_time_range_s": BALL_HOLD_TIME_RANGE_S,
      "far_prob": 0.0,  # set by the ball_aim_std curriculum (stage 3)
      "far_dist_range": BALL_FAR_DIST_RANGE,
    },
  )
  cfg.events["ball_stop_cycle"] = EventTermCfg(
    func=amp_mdp.ball_stop_cycle,
    mode="step",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
      "ball_cfg": copy.deepcopy(_ball),
      "contact_sensor_name": "robot_ball_contact",
      "aim_std": BALL_AIM_STD,
      "aim_clip": BALL_AIM_CLIP,
      "arrival_speed_range": BALL_ARRIVAL_SPEED_RANGE,
      "far_arrival_speed_range": BALL_FAR_ARRIVAL_SPEED_RANGE,
      "loft_prob": 0.0,  # set by the ball_aim_std curriculum (stage 3)
      "loft_angle_deg_range": BALL_LOFT_ANGLE_DEG_RANGE,
      "loft_speed_scale_range": BALL_LOFT_SPEED_SCALE_RANGE,
      # Ball farther than this from the robot while incoming = miss; above
      # the far spawn distance.
      "miss_max_dist": 8.0,
    },
  )

  # --- Curriculum: direction the ball comes from ------------------------------
  cfg.curriculum["ball_bearing"] = CurriculumTermCfg(
    func=amp_mdp.stage_event_param,
    params={
      "event_name": "reset_ball",
      "param_name": "spawn_bearing_deg",
      "stages": [(it * 24, deg) for it, deg in BALL_BEARING_STAGES],
    },
  )

  # --- Curriculum: aim sigma, promoted by "ball stopped in front" rate -------
  cfg.curriculum["ball_aim_std"] = CurriculumTermCfg(
    func=amp_mdp.ball_stop_aim_curriculum,
    params={
      "event_name": "ball_stop_cycle",
      "stages": BALL_AIM_STD_STAGES,
      "promote_rate": BALL_AIM_PROMOTE_RATE,
      "eval_episodes": BALL_AIM_EVAL_EPISODES,
      "ball_name": BALL_NAME,
      "start_stage": 0,
      "stage_params": BALL_STAGE_PARAMS,
    },
  )

  # --- RSI reference clips ---------------------------------------------------
  _motion_dir = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__), "..", "..", "..", "..",
      "assets", "motions", "piplus", "amp", BALL_STOP_MOTION_SUBDIR,
    )
  )
  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _motion_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  if play:
    # PLAY_AIM_STAGE=<n> starts the aim sigma curriculum at stage n (the
    # stage is not saved in the checkpoint, so play always starts at 0).
    _stage = min(int(os.environ.get("PLAY_AIM_STAGE", "0")), len(BALL_AIM_STD_STAGES) - 1)
    cfg.curriculum["ball_aim_std"].params["start_stage"] = _stage
    cfg.events["ball_stop_cycle"].params["aim_std"] = BALL_AIM_STD_STAGES[_stage]
    # Ghost ball at the position the actor observes (noise + delay).
    cfg.scene.entities["ball_ghost"] = get_ball_ghost_cfg()
    cfg.events["ball_obs_ghost"] = EventTermCfg(
      func=amp_mdp.update_ball_obs_ghost,
      mode="step",
      params={
        "ghost_cfg": SceneEntityCfg("ball_ghost"),
        "anchor_cfg": copy.deepcopy(_anchor),
      },
    )
    # Resample startup DR on every reset (few envs in play).
    for _ev in cfg.events.values():
      if _ev.mode == "startup" and (
        "mdp.dr" in getattr(_ev.func, "__module__", "")
        or _ev.func is randomize_terrain_contact
      ):
        _ev.mode = "reset"

  return cfg
