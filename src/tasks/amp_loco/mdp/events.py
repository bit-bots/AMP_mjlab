from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from src.assets.objects import BALL_RADIUS
from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.rewards import foot_medial_alignment
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
# Spawn geometry (distance + heading cone around the robot) mirrors
# mjxperiment's kick.py ``_sample_ball_offset``; placement is otherwise
# convention-free (world-frame xy + fixed z=radius, identity orientation).
# ------------------------------------------------------------------


def _place_ball_near_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    robot: Entity,
    ball: Entity,
    dist_range: tuple[float, float],
    cone_deg: float,
) -> None:
    n = env_ids.shape[0]
    robot_xy = robot.data.root_link_pos_w[env_ids, :2]
    robot_yaw = robot.data.heading_w[env_ids]

    dist = sample_uniform(dist_range[0], dist_range[1], (n,), device=env.device)
    cone = math.radians(cone_deg)
    angle = robot_yaw + sample_uniform(-cone, cone, (n,), device=env.device)
    offset = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) * dist.unsqueeze(-1)

    pose = torch.zeros((n, 7), device=env.device)
    pose[:, 0:2] = robot_xy + offset
    pose[:, 2] = BALL_RADIUS
    pose[:, 3] = 1.0  # identity quat (w,x,y,z)
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)

    # Every ball reposition (episode reset or post-kick respawn) starts a fresh
    # "life" for one-shot rewards keyed off the ball, e.g.
    # first_object_contact_reward's contact_reward_claimed (mdp/rewards.py).
    if hasattr(env, "contact_reward_claimed"):
        env.contact_reward_claimed[env_ids] = False


def reset_ball_near_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    dist_range: tuple[float, float] = (0.4, 1.2),
    cone_deg: float = 90.0,
) -> None:
    """Reset event: spawn the ball at a random distance/heading around the robot.

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
    _place_ball_near_robot(env, env_ids, robot, ball, dist_range, cone_deg)
    if hasattr(env, "kick_timer"):
        env.kick_timer[env_ids] = 0
    if hasattr(env, "kick_style"):
        env.kick_style[env_ids] = 0.0


class kick_contact_cycle:
    """Contact-driven ball cycle: open a reward-farming window on right-foot/ball
    contact, hold it for ``window_s`` seconds, then reset the ball (new random
    position near the robot + zero velocity) WITHOUT ending the episode.

    No more distance-based "ball rolled too far" respawn/penalty -- the ball only
    ever moves once the robot actually kicks it, making the whole ball-reset
    lifecycle purely contact-driven. ``env.kick_timer`` (steps remaining in the
    window) and ``env.kick_style`` (the foot/ball alignment quality at the moment
    of contact, in [0, 1], frozen for the whole window) are shared with
    ``kick_impact_reward`` (mdp/rewards.py), which reads them to gate/scale the
    ball speed+height reward; this event owns writing both. Use with
    ``mode="step"``.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        self._robot: Entity = env.scene[cfg.params["robot_cfg"].name]
        self._ball: Entity = env.scene[cfg.params["ball_cfg"].name]
        self._foot_body_id = self._robot.find_bodies(cfg.params["foot_body_name"])[0][0]
        self._medial_sign = cfg.params.get("medial_sign", 1.0)
        window_s = cfg.params.get("window_s", 2.0)
        self._window_steps = max(1, round(window_s / env.step_dt))
        env.kick_timer = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.kick_style = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        robot_cfg: SceneEntityCfg,
        ball_cfg: SceneEntityCfg,
        contact_sensor_name: str,
        foot_body_name: str,
        medial_sign: float = 1.0,
        window_s: float = 2.0,
        dist_range: tuple[float, float] = (0.4, 1.2),
        cone_deg: float = 90.0,
    ) -> None:
        del env_ids, robot_cfg, ball_cfg, foot_body_name, medial_sign, window_s
        # Unused; resolved once in __init__.
        sensor = env.scene[contact_sensor_name]
        assert sensor.data.found is not None
        touching = (sensor.data.found > 0).any(dim=-1)

        idle = env.kick_timer == 0
        new_contact = touching & idle
        env.kick_timer = torch.where(
            new_contact, torch.full_like(env.kick_timer, self._window_steps), env.kick_timer
        )

        if new_contact.any():
            foot_pos = self._robot.data.body_link_pos_w[:, self._foot_body_id, :2]
            foot_quat = self._robot.data.body_link_quat_w[:, self._foot_body_id]
            ball_pos = self._ball.data.root_link_pos_w[:, :2]
            align = foot_medial_alignment(foot_pos, foot_quat, ball_pos, self._medial_sign)
            # Squared clip-to-[0,1], mirroring mjxperiment's kick_style: only
            # alignment close to perfectly medial earns close to full credit, so
            # a proper side-foot kick pays much more than a glancing one.
            style = torch.clamp(align, 0.0, 1.0) ** 2
            env.kick_style = torch.where(new_contact, style, env.kick_style)

        expiring = env.kick_timer == 1  # last active step of the window
        env.kick_timer = torch.clamp(env.kick_timer - 1, min=0)

        if not expiring.any():
            return
        ids = expiring.nonzero(as_tuple=False).squeeze(-1)
        _place_ball_near_robot(env, ids, self._robot, self._ball, dist_range, cone_deg)
