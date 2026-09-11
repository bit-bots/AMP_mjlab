from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Reset-frame oversampling weight for clips whose motion_name marks them as a
# run (currently just "piplus_run1_subject2" in WalkandRun/). Uniform
# per-frame sampling gives running only its raw frame-count share of the
# pool (~16%), so the policy barely sees it and ends up walking almost
# everywhere -- boost it here instead of only reweighting via new data.
RUN_MOTION_NAME_MARKER = "run"
RUN_MOTION_SAMPLE_WEIGHT = 2.0
# At weight=2.0, running goes from its raw ~27% frame share (11890 of 44503
# frames in WalkandRun/) to ~42% of resets -- a real but not overwhelming
# boost. Raise this if the policy still favors walking; watch for walking
# quality degrading if pushed much higher, since it would still be the
# majority of clips (21 of 22) but a minority of reset frames.


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
        num_reset = len(env_ids)
        sample_weight = frames.get("sample_weight")
        if sample_weight is not None:
            idx = torch.multinomial(
                sample_weight.to(env.device), num_reset, replacement=True
            )
        else:
            total_frames = frames["root_pos"].shape[0]
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
        sample_weight_list = []
        for motion in motions:
            num_frames = motion["body_pos_w"].shape[0]
            root_pos_list.append(motion["body_pos_w"][:, 0, :])
            root_quat_list.append(motion["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
            joint_pos_list.append(motion["dof_pos"])
            joint_vel_list.append(motion["dof_vel"])
            motion_name = motion.get("motion_name", "")
            weight = (
                RUN_MOTION_SAMPLE_WEIGHT
                if RUN_MOTION_NAME_MARKER in motion_name.lower()
                else 1.0
            )
            sample_weight_list.append(
                torch.full((num_frames,), weight, dtype=torch.float32)
            )
        return {
            "root_pos": torch.cat(root_pos_list, dim=0),
            "root_quat": torch.cat(root_quat_list, dim=0),
            "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
            "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
            "joint_pos": torch.cat(joint_pos_list, dim=0),
            "joint_vel": torch.cat(joint_vel_list, dim=0),
            "sample_weight": torch.cat(sample_weight_list, dim=0),
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
