from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    sample_uniform,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from src.assets.objects import BALL_RADIUS, BALL_STRONG_RGBA, BALL_WEAK_RGBA
from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.rewards import (
    contact_sensor_touched,
    foot_medial_alignment,
    local_kick_dir_xy,
    select_kick_foot_is_left,
)
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# Extra height above the estimated local ground level to spawn the ball at
# -- see _place_ball_near_robot -- so small terrain-height estimation error
# never buries it. Sized to clear a bump of a few centimeters on non-flat terrain;
# the kick task trains on a flat plane, where it only adds a short drop.
_BALL_SPAWN_HEIGHT_MARGIN = 0.08


class MotionResetManager:
    """Manages motion frame data and delayed-reset logic for AMP environments."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        env: ManagerBasedRlEnv,
        motion_dir: str,
        recovery_dir: str | None = None,
    ) -> None:
        if motion_dir in self.walk_run_frames:
            return

        loader = MotionLoader(
            motion_dir=motion_dir,
            tgt_body_indexes=[],
            tgt_anchor_indexes=0,
            feet_indexes=0,
            device=str(env.device),
            recovery_dir=recovery_dir,
        )

        self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
        motion_count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] Loaded {len(loader.motion_data)} clips, {motion_count} frames from {motion_dir}")

        if loader.motion_data_recovery:
            self.recovery_frames[motion_dir] = self._concat_frames(loader.motion_data_recovery)
            recovery_count = self.recovery_frames[motion_dir]["root_pos"].shape[0]
            print(f"[MotionResetManager] Loaded {len(loader.motion_data_recovery)} recovery clips, {recovery_count} frames from {recovery_dir}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        # Split into delay envs and normal envs.
        delay_mask = self._get_delay_env_mask(env)
        if delay_mask is not None:
            is_delay = delay_mask[env_ids]
            delay_ids = env_ids[is_delay]
            normal_ids = env_ids[~is_delay]
        else:
            delay_ids = env_ids[:0]  # empty
            normal_ids = env_ids

        # Reset normal envs with walk/run data.
        if len(normal_ids) > 0:
            self._write_reset_state(env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg)

        # Reset delay envs with recovery data (fallback to walk/run if unavailable).
        if len(delay_ids) > 0:
            recovery = self.recovery_frames.get(motion_dir)
            frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
            self._write_reset_state(env, delay_ids, frames, asset_cfg)

    def _get_delay_env_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        """Get delay env mask from DelayedTerminationManager if installed."""
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            return tm._delay_env_mask
        return None

    def _write_reset_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        total_frames = frames["root_pos"].shape[0]
        num_reset = len(env_ids)
        idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset: Entity = env.scene[asset_cfg.name]

        # --- Root pose ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()

        # --- Key Fix for terrain ---
        terrain_z = positions[:, 2].clone()
        positions[:, 2] = terrain_z + root_pos[:, 2]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # --- Root velocity ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # --- Joint state ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
        root_pos_list = []
        root_quat_list = []
        root_lin_vel_list = []
        root_ang_vel_list = []
        joint_pos_list = []
        joint_vel_list = []
        for motion in motions:
            root_pos_list.append(motion["body_pos_w"][:, 0, :])
            root_quat_list.append(motion["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
            joint_pos_list.append(motion["dof_pos"])
            joint_vel_list.append(motion["dof_vel"])
        return {
            "root_pos": torch.cat(root_pos_list, dim=0),
            "root_quat": torch.cat(root_quat_list, dim=0),
            "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
            "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
            "joint_pos": torch.cat(joint_pos_list, dim=0),
            "joint_vel": torch.cat(joint_vel_list, dim=0),
        }


# ------------------------------------------------------------------
# Event callback wrappers (thin delegates to singleton)
# ------------------------------------------------------------------

def init_motion_loader(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
) -> None:
    """Startup event: load motion data and optionally install delayed termination."""
    MotionResetManager.get().init(
        env=env,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
    )

    # Install DelayedTerminationManager if requested.
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay > 0 and max_delay_steps > 0:
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
        delay_mask[delay_indices] = True
        env.termination_manager = DelayedTerminationManager(
            base=env.termination_manager,
            delay_env_mask=delay_mask,
            max_delay_steps=max_delay_steps,
        )
        print(
            "[init_motion_loader] DelayedTerminationManager installed: "
            f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
        )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random motion frames, with delay support."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        asset_cfg=asset_cfg,
    )


def randomize_imu_mounting_bias(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    std_rad: float = 0.01,
) -> None:
    """Per-episode IMU mounting misalignment (roll/pitch).

    Mirrors the mjxperiment deployment model: sample roll,pitch ~ N(0, std_rad),
    build ``R = Ry(pitch) @ Rx(roll)`` and stash it on the env as ``imu_bias_rot``.
    The biased gyro / projected-gravity actor obs functions left-multiply their
    (body-frame) vector by this rotation, so the policy sees the IMU as if it were
    mounted with a small fixed tilt for the whole episode.
    """
    if not hasattr(env, "imu_bias_rot"):
        env.imu_bias_rot = (
            torch.eye(3, device=env.device).unsqueeze(0).repeat(env.num_envs, 1, 1)
        )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    n = env_ids.shape[0]
    rp = torch.randn(n, 2, device=env.device) * std_rad
    roll, pitch = rp[:, 0], rp[:, 1]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    z, o = torch.zeros_like(cr), torch.ones_like(cr)
    # fmt: off
    Rx = torch.stack([o, z, z,  z, cr, -sr,  z, sr, cr], dim=-1).reshape(n, 3, 3)
    Ry = torch.stack([cp, z, sp,  z, o, z,  -sp, z, cp], dim=-1).reshape(n, 3, 3)
    # fmt: on
    env.imu_bias_rot[env_ids] = torch.bmm(Ry, Rx)


# ------------------------------------------------------------------
# Kick task: ball spawn / respawn
#
# Placement is a forward distance + a lateral (left/right) offset relative
# to the robot's current heading, not a full cone/circle around it -- the
# ball always sits somewhere in front of the robot, closer to dead-ahead
# when near and increasingly spread side-to-side the farther out it is.
# Placement itself is otherwise convention-free (world-frame xy + fixed
# z=radius, identity orientation).
# ------------------------------------------------------------------


def _sample_kick_dir(
    n: int,
    device: str,
    heading: torch.Tensor | None = None,
    cone_deg: float | None = None,
) -> torch.Tensor:
    """Target kick direction, unit vector (n, 2) in the world xy plane.

    If ``heading``/``cone_deg`` are given, samples within ``heading +/-
    cone_deg`` (a target direction the robot has a plausible chance of
    actually hitting from its current orientation); otherwise fully
    uniform over [-pi, pi] (used only for the initial value before any
    reset has run)."""
    if heading is None or cone_deg is None:
        angle = sample_uniform(-math.pi, math.pi, (n,), device=device)
    else:
        cone = math.radians(cone_deg)
        angle = heading + sample_uniform(-cone, cone, (n,), device=device)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def _place_ball_near_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    robot: Entity,
    ball: Entity,
    dist_range: tuple[float, float],
    angle_range_deg: float,
    kick_dir_cone_deg: float,
) -> None:
    n = env_ids.shape[0]
    robot_xy = robot.data.root_link_pos_w[env_ids, :2]
    robot_yaw = robot.data.heading_w[env_ids]

    dist = sample_uniform(dist_range[0], dist_range[1], (n,), device=env.device)
    # Bearing from the robot's heading, uniform in +/- angle_range_deg (180 =
    # anywhere around the robot, including behind it).
    half = math.radians(angle_range_deg)
    bearing = robot_yaw + sample_uniform(-half, half, (n,), device=env.device)
    offset = torch.stack([torch.cos(bearing), torch.sin(bearing)], dim=-1) * dist.unsqueeze(-1)

    # Ground level is BALL_RADIUS above the robot's own terrain patch
    # baseline (env_origins' z), not a fixed world z=0 -- a non-flat terrain
    # patch (e.g. a tilted-grid platform) can sit well
    # above/below world z=0, and spawning at a fixed world height would
    # bury or float the ball. env_origins reflects the patch the robot was
    # last placed on, which is a good local approximation (not exact -- the
    # ball's own xy offset can be on slightly different local terrain
    # height, e.g. a Perlin bump/dip) since the ball always spawns close to
    # the robot. A small extra margin (_BALL_SPAWN_HEIGHT_MARGIN) on top
    # absorbs that residual error on the safe side: a ball that spawns
    # slightly too high just drops the last centimeters under gravity,
    # while one that spawns embedded in the ground can get shoved out by a
    # sudden contact-resolution impulse -- which, combined with a foot
    # happening to be nearby, could even register as a false "kick".
    ground_z = env.scene.env_origins[env_ids, 2]
    pose = torch.zeros((n, 7), device=env.device)
    pose[:, 0:2] = robot_xy + offset
    pose[:, 2] = ground_z + BALL_RADIUS + _BALL_SPAWN_HEIGHT_MARGIN
    pose[:, 3] = 1.0  # identity quat (w,x,y,z)
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)

    # Every ball reposition (episode reset or post-kick respawn) starts a fresh
    # "life" for one-shot rewards keyed off the ball, e.g.
    # first_object_contact_reward's contact_reward_claimed (mdp/rewards.py).
    if hasattr(env, "contact_reward_claimed"):
        env.contact_reward_claimed[env_ids] = False
    # ...and cross_foot_touch_penalty's first-toucher claim (mdp/rewards.py):
    # a fresh ball life means neither foot has touched it yet.
    if hasattr(env, "ball_first_touch_foot"):
        env.ball_first_touch_foot[env_ids] = -1
    # ...and a fresh target kick direction (kick_direction_reward, mdp/rewards.py),
    # sampled relative to the robot's heading at THIS reposition (not its
    # heading at some later point), so it stays a plausible target.
    if hasattr(env, "kick_dir_world"):
        env.kick_dir_world[env_ids] = _sample_kick_dir(
            n, env.device, heading=robot_yaw, cone_deg=kick_dir_cone_deg
        )
    # ...and a fresh commanded kick intensity for this life (kick_intensity_obs,
    # mdp/observations.py; kick_speed_cap_penalty/kick_impact_reward, mdp/rewards.py).
    if hasattr(env, "kick_intensity"):
        env.kick_intensity[env_ids] = sample_uniform(0.0, 1.0, (n,), device=env.device)
        # Color the ball by the new intent (linearly interpolated between
        # BALL_WEAK_RGBA at intensity=0 and BALL_STRONG_RGBA at intensity=1,
        # src/assets/objects/ball.py) -- a direct visual cue in the viewer,
        # not only in the observation. Written straight to the model field
        # (like the DR functions do for radius/friction) since this is a
        # per-env value, not a one-time asset default.
        ball_geom_id = ball.indexing.geom_ids[0]
        strong_rgba = torch.tensor(BALL_STRONG_RGBA, device=env.device)
        weak_rgba = torch.tensor(BALL_WEAK_RGBA, device=env.device)
        t = env.kick_intensity[env_ids].unsqueeze(-1)
        env.sim.model.geom_rgba[env_ids, ball_geom_id] = weak_rgba + t * (strong_rgba - weak_rgba)


def reset_ball_near_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    dist_range: tuple[float, float] = (0.3, 1.0),
    angle_range_deg: float = 180.0,
    kick_dir_cone_deg: float = 180.0,
) -> None:
    """Reset event: spawn the ball around the robot (distance from ``dist_range``,
    bearing within +/- ``angle_range_deg`` of the heading, see
    ``_place_ball_near_robot``).

    Must run AFTER any reset event that repositions the robot (e.g.
    ``reset_from_motion_data``) -- ``EntityData`` documents that write_* calls
    only land in ``qpos``/``qvel``, and read properties like
    ``root_link_pos_w``/``heading_w`` (``xpos``/``xquat``-derived) need a
    ``sim.forward()`` in between to pick them up. ``ManagerBasedRlEnv.reset()``
    defers that forward() until *all* reset events have run, so this event
    forces one early to read the robot's just-written, final reset pose.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.scene.write_data_to_sim()
    env.sim.forward()
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    _place_ball_near_robot(env, env_ids, robot, ball, dist_range, angle_range_deg, kick_dir_cone_deg)
    if hasattr(env, "kick_timer"):
        env.kick_timer[env_ids] = 0
    if hasattr(env, "kick_style"):
        env.kick_style[env_ids] = 0.0


# Measured in the training sim (mujoco_warp, ball radius, mass, friction and
# torsional friction randomization on): the ball rolls this many times farther
# than the energy model in ``random_ball_push`` says, per rolling friction
# value. The launch speed is scaled down by the square root of the ratio.
_ROLL_FRICTION_POINTS = (0.01, 0.03, 0.1)
_ROLL_DISTANCE_RATIO = (1.7, 3.0, 3.2)


def random_ball_push(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg,
    distance_range: tuple[float, float] = (0.05, 2.0),
    ramp_iterations: tuple[int, int] | None = None,
    steps_per_iteration: int = 24,
) -> None:
    """Interval event: give the ball a random horizontal velocity so that it
    rolls about ``distance`` meters, with ``distance`` log-uniform in
    ``distance_range`` and a uniform random heading.

    The launch speed comes from the energy of a ball that rolls without slip
    (kinetic energy 0.7 m v^2) and loses it to the rolling friction torque
    (mu_r m g per radian), with the env's own ball radius and rolling friction
    draw. The ball also gets the spin of rolling without slip, so sliding
    friction does not eat part of the energy. The distance is approximate.

    Skipped while the post-kick window runs (``env.kick_timer`` > 0), so a push
    never disturbs the reward window of a kick.

    Curriculum: with ``ramp_iterations=(start, end)`` no push happens before
    iteration ``start``. From there the chance that a due push really happens
    rises linearly to 1 at iteration ``end``. The iteration is
    ``env.common_step_counter / steps_per_iteration``.
    """
    if hasattr(env, "kick_timer"):
        env_ids = env_ids[env.kick_timer[env_ids] == 0]
    if ramp_iterations is not None:
        start, end = ramp_iterations
        iteration = env.common_step_counter / steps_per_iteration
        scale = min(max((iteration - start) / (end - start), 0.0), 1.0)
        env_ids = env_ids[torch.rand(env_ids.shape[0], device=env.device) < scale]
    n = env_ids.shape[0]
    if n == 0:
        return
    ball: Entity = env.scene[ball_cfg.name]
    k, radius = _ball_roll_coeff(env, env_ids, ball)

    lo, hi = distance_range
    u = torch.rand(n, device=env.device)
    distance = lo * (hi / lo) ** u
    speed = torch.sqrt(distance * k)

    heading = torch.rand(n, device=env.device) * (2.0 * math.pi)
    direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    ball.write_root_link_velocity_to_sim(
        _rolling_ball_velocity(direction, speed, radius), env_ids=env_ids
    )


def _ball_roll_coeff(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball: Entity
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env rolling-loss coefficient ``k`` (m/s^2 * 1/m) and ball radius.

    A ball rolling without slip from speed ``v`` covers ``v^2 / k`` meters
    before it stops (energy model of ``random_ball_push``, corrected by the
    measured ``_ROLL_DISTANCE_RATIO``), with the env's own radius and rolling
    friction draw.
    """
    geom_id = ball.indexing.geom_ids[0]
    model = env.sim.model
    world = env_ids % model.geom_friction.shape[0]
    mu_r = model.geom_friction[world, geom_id, 2].clamp(min=1e-3)
    radius = model.geom_size[env_ids % model.geom_size.shape[0], geom_id, 0]
    points = torch.tensor(_ROLL_FRICTION_POINTS, device=env.device)
    ratios = torch.tensor(_ROLL_DISTANCE_RATIO, device=env.device)
    # Piecewise-linear interpolation of the measured ratio (clamped at the ends).
    mu_c = mu_r.clamp(points[0], points[-1])
    idx = torch.clamp(torch.searchsorted(points, mu_c) - 1, 0, len(points) - 2)
    frac = (mu_c - points[idx]) / (points[idx + 1] - points[idx])
    ratio = ratios[idx] + frac * (ratios[idx + 1] - ratios[idx])
    k = mu_r * 9.81 / (0.7 * radius * ratio)
    return k, radius


def _rolling_ball_velocity(
    direction: torch.Tensor, speed: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor:
    """(n, 6) root velocity for a ball rolling without slip along the unit xy
    ``direction`` at ``speed``: omega = (z_hat x v) / r, so sliding friction
    does not eat part of the launch energy."""
    vel = torch.zeros((direction.shape[0], 6), device=direction.device)
    vel[:, 0:2] = direction * speed.unsqueeze(-1)
    vel[:, 3] = -direction[:, 1] * speed / radius
    vel[:, 4] = direction[:, 0] * speed / radius
    return vel


# ------------------------------------------------------------------
# Ball-stopping task: one ball per episode, rolled at the robot
#
# Phases (env.bs_phase), updated by the ``ball_stop_cycle`` step event:
#   BS_IDLE     ball rests at its spawn point, robot should stand still
#   BS_INCOMING ball was launched toward the robot (after a random idle time)
#   BS_CONTACT  the robot touched the ball; after a random hold time the
#               episode ends successfully (env.bs_success)
# A ball that passes the robot, stops short or rolls away without contact
# sets env.bs_missed. Both flags end the episode (terminations.py).
#
# The step event runs after the rewards/terminations of a step (mjlab step
# order: physics -> terminations -> rewards -> reset -> step events -> obs),
# so rewards and terminations see the phase and flags of the previous step.
# ------------------------------------------------------------------

BS_IDLE = 0
BS_INCOMING = 1
BS_CONTACT = 2

# (c, sliding, rolling, radius) exponents, see _ball_launch_speed.
_LAUNCH_ROLL_CORRECTION = (-2.341, -0.47, 0.165, -0.934)


def _ball_stop_state(env: ManagerBasedRlEnv) -> None:
    """Create the per-env ball-stopping state on first use."""
    if hasattr(env, "bs_phase"):
        return
    n, dev = env.num_envs, env.device
    env.bs_phase = torch.zeros(n, dtype=torch.long, device=dev)
    env.bs_launch_step = torch.zeros(n, dtype=torch.long, device=dev)
    env.bs_hold_steps = torch.zeros(n, dtype=torch.long, device=dev)
    env.bs_phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
    env.bs_launch_origin = torch.zeros((n, 2), device=dev)
    env.bs_launch_dir = torch.zeros((n, 2), device=dev)
    env.bs_launch_dir[:, 0] = 1.0
    env.bs_start_xy = torch.zeros((n, 2), device=dev)
    env.bs_new_contact = torch.zeros(n, dtype=torch.bool, device=dev)
    env.bs_success = torch.zeros(n, dtype=torch.bool, device=dev)
    env.bs_missed = torch.zeros(n, dtype=torch.bool, device=dev)


def _ball_launch_speed(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball: Entity,
    path_len: torch.Tensor,
    arrival_speed: torch.Tensor,
    max_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch speed so that the ball still has ``arrival_speed`` after
    ``path_len`` meters, and the ball radius.

    ``_ball_roll_coeff`` with an empirical correction (fit over 511 launches
    with the kick task's ball DR, flat plane, no robot in the way):
    real/predicted roll distance = exp(c) * slide^a * roll^b * radius^d.
    Sliding friction is the main factor (higher -> shorter roll). Cuts the
    spread of real/predicted from ~0.49 to ~0.21 (std of the log). Measured:
    a requested arrival speed of 1.0 m/s gives 0.7-2.0 m/s (10-90 %), and
    about 20 % of the balls stop short.
    """
    k, radius = _ball_roll_coeff(env, env_ids, ball)
    geom_id = ball.indexing.geom_ids[0]
    model = env.sim.model
    fr = model.geom_friction[env_ids % model.geom_friction.shape[0], geom_id]
    c, a, b, d = _LAUNCH_ROLL_CORRECTION
    corr = math.exp(c) * fr[:, 0].pow(a) * fr[:, 2].clamp(min=1e-3).pow(b) * radius.pow(d)
    k = k / corr
    speed = torch.sqrt(arrival_speed.square() + path_len * k).clamp(max=max_speed)
    return speed, radius


def reset_ball_stop(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    spawn_dist_range: tuple[float, float] = (2.0, 4.0),
    spawn_bearing_deg: float = 0.0,
    idle_time_range_s: tuple[float, float] = (1.0, 3.0),
    hold_time_range_s: tuple[float, float] = (1.0, 2.0),
    spawn_height_margin: float = 0.002,
) -> None:
    """Reset event: put the ball at rest ``spawn_dist_range`` meters from the
    robot, at a bearing within +/- ``spawn_bearing_deg`` of the robot's
    heading (0 = straight in front; set by a curriculum), and start the idle
    phase. Draws this episode's idle time before the launch and hold time
    after contact.

    Must run after ``reset_from_motion`` (event dict order). Forces a
    ``sim.forward()`` to read the robot pose just written there (see
    ``reset_ball_near_robot``).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    n = env_ids.shape[0]
    if n == 0:
        return
    _ball_stop_state(env)
    env.scene.write_data_to_sim()
    env.sim.forward()
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    dev = env.device

    robot_xy = robot.data.root_link_pos_w[env_ids, :2]
    robot_yaw = robot.data.heading_w[env_ids]
    dist = sample_uniform(spawn_dist_range[0], spawn_dist_range[1], (n,), device=dev)
    half = math.radians(spawn_bearing_deg)
    bearing = robot_yaw + sample_uniform(-half, half, (n,), device=dev)
    to_spawn = torch.stack([torch.cos(bearing), torch.sin(bearing)], dim=-1)

    radius = env.sim.model.geom_size[
        env_ids % env.sim.model.geom_size.shape[0], ball.indexing.geom_ids[0], 0
    ]
    pose = torch.zeros((n, 7), device=dev)
    pose[:, 0:2] = robot_xy + to_spawn * dist.unsqueeze(-1)
    # On the ground (flat plane): a ball dropped from higher loses speed and
    # spin sync on landing after the launch.
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + radius + spawn_height_margin
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=dev), env_ids=env_ids)

    dt = env.step_dt
    idle = sample_uniform(idle_time_range_s[0], idle_time_range_s[1], (n,), device=dev)
    hold = sample_uniform(hold_time_range_s[0], hold_time_range_s[1], (n,), device=dev)
    env.bs_phase[env_ids] = BS_IDLE
    # Where the robot stands at the episode start (after RSI): the idle
    # phase penalizes moving away from here (ball_stop_idle_displacement).
    env.bs_start_xy[env_ids] = robot_xy
    env.bs_launch_step[env_ids] = torch.round(idle / dt).long()
    env.bs_hold_steps[env_ids] = torch.round(hold / dt).long().clamp(min=1)
    env.bs_phase_steps[env_ids] = 0
    env.bs_launch_origin[env_ids] = pose[:, 0:2]
    env.bs_launch_dir[env_ids] = -to_spawn
    env.bs_new_contact[env_ids] = False
    env.bs_success[env_ids] = False
    env.bs_missed[env_ids] = False


def ball_stop_cycle(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    contact_sensor_name: str,
    aim_std: float = 0.075,
    aim_clip: float | None = None,
    arrival_speed_range: tuple[float, float] = (0.5, 2.0),
    max_launch_speed: float = 8.0,
    miss_pass_dist: float = 0.5,
    miss_stop_speed: float = 0.05,
    miss_min_time_s: float = 0.5,
    miss_max_dist: float = 6.0,
) -> None:
    """Step event: advance the ball-stopping phases (see the section comment).

    - Idle -> incoming at the episode's launch step: the ball is rolled from
      where it lies toward the robot base, with a lateral aim offset
      ~ N(0, ``aim_std``), clipped to +/- ``aim_clip`` meters unless None (perpendicular to
      the path), so that it arrives with ``arrival_speed_range`` m/s
      (approximate, see ``_ball_launch_speed``).
    - Incoming -> contact on the first robot-ball contact
      (``contact_sensor_name``, any substep of the step). Sets the one-step
      flag ``env.bs_new_contact``.
    - Contact -> ``env.bs_success`` after the episode's hold time.
    - Incoming -> ``env.bs_missed`` when the ball is ``miss_pass_dist`` past
      the robot along its path, slower than ``miss_stop_speed`` after
      ``miss_min_time_s``, or farther than ``miss_max_dist`` from the robot.
    """
    del env_ids
    _ball_stop_state(env)
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    dev = env.device

    # Steps of this episode so far; 0 for envs reset in this step, whose
    # contact sensor data still comes from the previous episode.
    fresh = env.episode_length_buf == 0
    env.bs_new_contact[:] = False
    env.bs_phase_steps += 1

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]
    ball_speed = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)

    # --- contact (incoming only) -------------------------------------------
    touched = contact_sensor_touched(env.scene[contact_sensor_name]) & ~fresh
    new_contact = touched & (env.bs_phase == BS_INCOMING)
    env.bs_new_contact[:] = new_contact
    env.bs_phase[new_contact] = BS_CONTACT
    env.bs_phase_steps[new_contact] = 0

    # --- success after the hold time ------------------------------------------
    env.bs_success |= (env.bs_phase == BS_CONTACT) & (env.bs_phase_steps >= env.bs_hold_steps)

    # --- miss -----------------------------------------------------------------
    incoming = env.bs_phase == BS_INCOMING
    past = ((ball_xy - robot_xy) * env.bs_launch_dir).sum(-1) > miss_pass_dist
    stopped = (ball_speed < miss_stop_speed) & (
        env.bs_phase_steps * env.step_dt > miss_min_time_s
    )
    away = torch.norm(ball_xy - robot_xy, dim=-1) > miss_max_dist
    env.bs_missed |= incoming & (past | stopped | away)

    # --- launch ---------------------------------------------------------------
    launch = (env.bs_phase == BS_IDLE) & (env.episode_length_buf >= env.bs_launch_step)
    ids = launch.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
        return
    n = ids.shape[0]
    to_robot = robot_xy[ids] - ball_xy[ids]
    dist = torch.norm(to_robot, dim=-1).clamp(min=1e-6)
    u = to_robot / dist.unsqueeze(-1)
    lateral = torch.stack([-u[:, 1], u[:, 0]], dim=-1)
    offset = torch.randn(n, device=dev) * aim_std
    if aim_clip is not None:
        offset = offset.clamp(-aim_clip, aim_clip)
    path = to_robot + lateral * offset.unsqueeze(-1)
    path_len = torch.norm(path, dim=-1).clamp(min=1e-6)
    direction = path / path_len.unsqueeze(-1)
    v_arr = sample_uniform(arrival_speed_range[0], arrival_speed_range[1], (n,), device=dev)
    speed, radius = _ball_launch_speed(env, ids, ball, path_len, v_arr, max_launch_speed)
    ball.write_root_link_velocity_to_sim(
        _rolling_ball_velocity(direction, speed, radius), env_ids=ids
    )
    env.bs_phase[ids] = BS_INCOMING
    env.bs_phase_steps[ids] = 0
    env.bs_launch_origin[ids] = ball_xy[ids]
    env.bs_launch_dir[ids] = direction


@requires_model_fields("geom_rgba")
class kick_contact_cycle:
    """Contact-driven ball cycle: kicking -> post-kick -> kicking.

    **kicking** (``env.kick_timer`` == 0). The robot may strike the ball at
    any time, with EITHER foot -- there's no separate "positioned/aligned"
    gate before contact counts, and no foot restriction. Getting the robot
    to actually walk up and line up a good strike is left entirely to the
    ordinary rewards (``move_toward_object``, ``torso_orient_to_object``,
    ``foot_medial_alignment_penalty``), not a state machine here.

    **post-kick** (``env.kick_timer`` > 0). Entered once the ball's speed
    exceeds ``impact_speed_threshold`` AND an actual foot/ball contact is
    registered (see ``contact_sensor_touched``, mdp/rewards.py) while
    kicking; holds for ``window_s`` seconds so the robot can "farm" reward
    from the kick's aftermath, then the ball resets (new random position in
    front of the robot + zero velocity, fresh target kick direction)
    WITHOUT ending the episode, and the cycle returns to kicking.

    Speed alone isn't enough to detect a kick: on non-flat terrain
    a resting ball can pick up real velocity from
    gravity alone (rolling down a tilted tile, settling off a bump) with no
    foot anywhere near it. Contact alone isn't enough either: the
    continuous per-step version paid out for every step of contact,
    letting the policy learn to just rest a foot on the ball. Combining
    both -- a real velocity change AND a real touch -- rules out both
    failure modes.

    ``env.kick_timer`` (steps remaining in the post-kick window) and
    ``env.kick_style`` (the CLOSER foot's ball alignment quality at the
    moment of contact, in [0, 1], frozen for the whole window -- foot-
    agnostic: whichever of the configured feet is nearer the ball when the
    kick is detected is used, not always a fixed one) are shared with
    ``kick_impact_reward`` (mdp/rewards.py), which reads them to gate/scale
    the ball speed+height reward; this event owns writing both.

    Also owns ``env.kick_intensity``: a per-life commanded intensity in
    [0, 1], resampled uniformly by ``_place_ball_near_robot`` every time the
    ball is repositioned, and mapped to a target ball speed by
    ``kick_intensity_to_target_speed`` (mdp/rewards.py) -- continuously from
    a soft 1 m/s tap at 0.0 through a 4 m/s kick at 0.5 to an unbounded,
    as-hard-as-possible kick at 1.0. Every life is penalized for exceeding
    its own target speed (``kick_speed_cap_penalty``, mdp/rewards.py) --
    inert at intensity=1 since the target speed there is astronomically
    high. The policy sees the current value via ``kick_intensity_obs``
    (mdp/observations.py).

    Also computes, at the same moment: ``env.kick_new_contact`` (a
    transient, this-event-round-only flag -- overwritten every call, not
    held) and ``env.kick_foot_correct`` (frozen for the window like
    ``kick_style``) -- whether the foot that actually struck the ball (the
    same "closest configured foot" used for ``kick_style``) matches the
    foot ``select_kick_foot_is_left`` (mdp/rewards.py) says should have been
    used, given the ball's position and target kick angle AT THE MOMENT of
    contact. ``ball_kick_contact_reward`` and ``kick_impact_reward``
    (mdp/rewards.py) read these to reward/penalize by foot correctness. Use
    with ``mode="step"``.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        self._env = env
        self._robot: Entity = env.scene[cfg.params["robot_cfg"].name]
        self._ball: Entity = env.scene[cfg.params["ball_cfg"].name]
        foot_body_names = cfg.params["foot_body_names"]
        self._foot_body_ids = [self._robot.find_bodies(n)[0][0] for n in foot_body_names]
        self._foot_is_left = torch.tensor(
            [n.startswith("l_") for n in foot_body_names], device=env.device
        )
        self._medial_signs = list(cfg.params.get("medial_signs", (1.0,) * len(foot_body_names)))
        self._impact_speed_threshold = cfg.params.get("impact_speed_threshold", 0.2)
        window_s = cfg.params.get("window_s", 2.0)
        self._window_steps = max(1, round(window_s / env.step_dt))
        self._dist_range = cfg.params.get("dist_range", (0.3, 1.0))
        self._angle_range_deg = cfg.params.get("angle_range_deg", 180.0)
        self._kick_dir_cone_deg = cfg.params.get("kick_dir_cone_deg", 80.0)
        anchor_body_name = cfg.params.get("anchor_body_name", "torso_link")
        self._anchor_body_id = self._robot.find_bodies(anchor_body_name)[0][0]
        self._angle_threshold_deg = cfg.params.get("angle_threshold_deg", 20.0)
        right_sensor_name = cfg.params.get("right_ball_sensor_name", "r_foot_ball_contact")
        left_sensor_name = cfg.params.get("left_ball_sensor_name", "l_foot_ball_contact")
        self._right_ball_sensor = env.scene[right_sensor_name]
        self._left_ball_sensor = env.scene[left_sensor_name]
        env.kick_timer = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.kick_style = torch.zeros(env.num_envs, device=env.device)
        env.kick_new_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.kick_foot_correct = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.kick_dir_world = _sample_kick_dir(env.num_envs, env.device)
        env.kick_intensity = sample_uniform(0.0, 1.0, (env.num_envs,), device=env.device)
        # Ball xy at the moment of the last detected kick; the ball respawns
        # at the end of the window only if it moved far enough from here.
        env.kick_start_ball_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self._min_ball_travel = cfg.params.get("min_ball_travel", 0.0)

    def reset(self, env_ids) -> None:
        """No-op: this event has no instance state of its own that needs
        clearing (kick_timer/kick_style/kick_dir_world/kick_intensity all
        live on ``env`` and are cleared/reset by ``_place_ball_near_robot``
        whenever the ball is repositioned). Required anyway -- EventManager
        only adds a
        class-based term to its debug_vis registry when it also defines
        ``reset`` (see ``EventManager._prepare_terms``); without this, our
        arrow never rendered even though everything downstream was correct.
        """
        del env_ids

    def debug_vis(self, visualizer) -> None:
        """Draw an arrow at the ball pointing along the current target kick
        direction. Auto-invoked by EventManager.debug_vis for any
        class-based event term that defines this method -- no extra cfg flag
        needed.

        Colored by ``env.kick_intensity`` (linearly interpolated between
        BALL_WEAK_RGBA at 0 and BALL_STRONG_RGBA at 1). This is the visual
        intensity cue instead of recoloring the ball geom itself: mjlab's
        viser viewer bakes a dynamic mesh's color in at construction and has
        no runtime color update for it (see mjlab/viewer/viser/scene.py's
        add_batched_meshes_trimesh docstring), but debug-vis primitives like
        this arrow go through a batched-mesh path that DOES support a
        per-frame color update, so this is what actually shows up live.
        """
        env = self._env
        ball_pos = self._ball.data.root_link_pos_w
        kick_dir = env.kick_dir_world
        length = 0.5
        for env_idx in visualizer.get_env_indices(env.num_envs):
            start = ball_pos[env_idx]
            end = start.clone()
            end[0] = start[0] + kick_dir[env_idx, 0] * length
            end[1] = start[1] + kick_dir[env_idx, 1] * length
            t = float(env.kick_intensity[env_idx])
            color = tuple(
                w + t * (s - w) for w, s in zip(BALL_WEAK_RGBA, BALL_STRONG_RGBA)
            )
            visualizer.add_arrow(start, end, color=color, width=0.015, label="kick_dir")

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        robot_cfg: SceneEntityCfg,
        ball_cfg: SceneEntityCfg,
        foot_body_names: tuple[str, ...],
        medial_signs: tuple[float, ...] = (1.0,),
        impact_speed_threshold: float = 0.2,
        window_s: float = 2.0,
        dist_range: tuple[float, float] = (0.3, 1.0),
        angle_range_deg: float = 180.0,
        kick_dir_cone_deg: float = 180.0,
        anchor_body_name: str = "torso_link",
        angle_threshold_deg: float = 20.0,
        right_ball_sensor_name: str = "r_foot_ball_contact",
        left_ball_sensor_name: str = "l_foot_ball_contact",
        min_ball_travel: float = 0.0,
    ) -> None:
        del (
            env_ids, robot_cfg, ball_cfg, foot_body_names, medial_signs,
            impact_speed_threshold, window_s, dist_range, angle_range_deg, kick_dir_cone_deg,
            anchor_body_name, angle_threshold_deg, right_ball_sensor_name, left_ball_sensor_name,
            min_ball_travel,
        )
        # All params unused here; resolved once in __init__.

        # Ball speed exceeding the threshold alone isn't enough: on
        # non-flat terrain a resting ball can pick up
        # real velocity from gravity alone (rolling down a tilted tile,
        # settling off a bump) with no foot anywhere near it. Require an
        # ACTUAL foot/ball contact too -- contact_sensor_touched
        # (mdp/rewards.py) checks every physics substep within the control
        # step (via the sensors' force_history), not just the final one, so
        # a fast strike-and-separate within one step's decimation loop
        # still counts even though it wouldn't show up in a plain
        # end-of-step ``found`` check.
        ball_speed = torch.norm(self._ball.data.root_link_lin_vel_w[:, :2], dim=-1)
        foot_touching = contact_sensor_touched(self._right_ball_sensor) | contact_sensor_touched(
            self._left_ball_sensor
        )
        kicked = (ball_speed > self._impact_speed_threshold) & foot_touching

        idle = env.kick_timer == 0
        new_contact = kicked & idle
        env.kick_new_contact = new_contact
        env.kick_timer = torch.where(
            new_contact, torch.full_like(env.kick_timer, self._window_steps), env.kick_timer
        )

        if new_contact.any():
            # Foot-agnostic: score contact quality using whichever configured
            # foot is actually CLOSER to the ball right now, not a fixed one
            # -- so a kick with either foot gets its own alignment scored
            # correctly instead of always being judged against one foot's
            # (possibly irrelevant) position/orientation.
            ball_pos = self._ball.data.root_link_pos_w[:, :2]
            dists, aligns = [], []
            for foot_id, sign in zip(self._foot_body_ids, self._medial_signs):
                foot_pos = self._robot.data.body_link_pos_w[:, foot_id, :2]
                foot_quat = self._robot.data.body_link_quat_w[:, foot_id]
                dists.append(torch.norm(ball_pos - foot_pos, dim=-1))
                aligns.append(foot_medial_alignment(foot_pos, foot_quat, ball_pos, sign))
            closest = torch.argmin(torch.stack(dists, dim=-1), dim=-1, keepdim=True)
            align = torch.gather(torch.stack(aligns, dim=-1), 1, closest).squeeze(-1)
            # Squared clip-to-[0,1], mirroring mjxperiment's kick_style: only
            # alignment close to perfectly medial earns close to full credit, so
            # a proper side-foot kick pays much more than a glancing one.
            style = torch.clamp(align, 0.0, 1.0) ** 2
            env.kick_style = torch.where(new_contact, style, env.kick_style)
            env.kick_start_ball_xy = torch.where(
                new_contact.unsqueeze(-1), ball_pos, env.kick_start_ball_xy
            )

            # Foot correctness: which foot ACTUALLY struck the ball (the same
            # "closest configured foot" used for style, above) vs. which foot
            # select_kick_foot_is_left says SHOULD have, given the ball's
            # position and target kick angle in the robot's local frame at
            # this exact moment -- not a static/pre-approach decision.
            actual_is_left = self._foot_is_left[closest.squeeze(-1)]
            anchor_pos_w = self._robot.data.body_link_pos_w[:, self._anchor_body_id]
            anchor_quat_w = self._robot.data.body_link_quat_w[:, self._anchor_body_id]
            ball_pos_b, _ = subtract_frame_transforms(
                anchor_pos_w, anchor_quat_w, self._ball.data.root_link_pos_w
            )
            kick_dir_b = local_kick_dir_xy(env, self._robot)
            desired_is_left = select_kick_foot_is_left(
                ball_pos_b[:, :2], kick_dir_b, self._angle_threshold_deg
            )
            env.kick_foot_correct = torch.where(
                new_contact, actual_is_left == desired_is_left, env.kick_foot_correct
            )

        expiring = env.kick_timer == 1  # last active step of the window
        env.kick_timer = torch.clamp(env.kick_timer - 1, min=0)

        if not expiring.any():
            return
        # The ball was touched (a kick needs a foot contact). It respawns only if
        # it also moved at least ``min_ball_travel`` from where it was kicked.
        # A weak tap ends the window without a respawn: the robot has to kick
        # the same ball again.
        travel = torch.norm(
            self._ball.data.root_link_pos_w[:, :2] - env.kick_start_ball_xy, dim=-1
        )
        expiring = expiring & (travel >= self._min_ball_travel)
        if not expiring.any():
            return
        ids = expiring.nonzero(as_tuple=False).squeeze(-1)
        _place_ball_near_robot(
            env, ids, self._robot, self._ball,
            self._dist_range, self._angle_range_deg, self._kick_dir_cone_deg,
        )


def update_ball_obs_ghost(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ghost_cfg: SceneEntityCfg,
    anchor_cfg: SceneEntityCfg,
) -> None:
    """Step event (viewer only): move the mocap ghost ball to the ball position
    that the actor observes (noise and delay included), in the world frame.

    ``env.ball_obs_delayed_w`` is set by ``object_pos_b_noisy_delayed_world``.
    It is one step old, as the observation manager runs after the events.
    """
    del env_ids, anchor_cfg
    pos_w = getattr(env, "ball_obs_delayed_w", None)
    if pos_w is None:
        return
    pose = torch.zeros((pos_w.shape[0], 7), device=pos_w.device)
    pose[:, 0:3] = pos_w
    pose[:, 3] = 1.0
    env.scene[ghost_cfg.name].write_mocap_pose_to_sim(
        pose, env_ids=torch.arange(pos_w.shape[0], device=pos_w.device)
    )
