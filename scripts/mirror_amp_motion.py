"""Left-right mirror an AMP motion capture npz (see ampmotion_loader.py for the
format), so a right-foot-kick clip becomes a left-foot-kick clip.

Rule (derived from piplus.xml's per-joint axis/ref attributes and verified by
an end-to-end forward-kinematics check -- mirroring is a reflection across the
robot's XZ (sagittal) plane, i.e. negate world Y; -Y is the robot's right,
+Y its left, matching the convention used throughout src/tasks/amp_loco):
  - joint_pos/joint_vel: swap each r_X_joint <-> l_X_joint column, negate ALL
    20 values (every joint on this robot flips sign under an L/R swap, not
    just roll/yaw ones -- confirmed by FK, see conversation/verify_mirror.py).
  - body_pos_w / body_lin_vel_w: swap r_*_link <-> l_*_link rows, negate the Y
    component.
  - body_quat_w (w,x,y,z): swap rows, negate x and z, keep w and y (mirrors
    the rotation via R' = M R M with M = diag(1,-1,1)).
  - body_ang_vel_w: swap rows, negate x and z, keep y (angular velocity is a
    pseudovector, so it picks up an extra sign relative to a plain position/
    linear-velocity vector).
Unpaired midline bodies (base_link, torso_link, head_yaw_link,
head_pitch_link, camera_link) keep their row but still get the same
per-component sign flips.

Usage:
  .venv/bin/python scripts/mirror_amp_motion.py \\
      --dir src/assets/motions/piplus/amp/Kick \\
      --out src/assets/motions/piplus/amp/Kick_mirrored
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
]
BODY_NAMES = [
    "base_link", "torso_link",
    "r_shoulder_pitch_link", "r_shoulder_roll_link", "r_upper_arm_link", "r_elbow_link", "r_wrist_link",
    "l_shoulder_pitch_link", "l_shoulder_roll_link", "l_upper_arm_link", "l_elbow_link", "l_wrist_link",
    "head_yaw_link", "head_pitch_link", "camera_link",
    "r_hip_pitch_link", "r_hip_roll_link", "r_thigh_link", "r_calf_link",
    "r_ankle_pitch_link", "r_ankle_roll_link",
    "l_hip_pitch_link", "l_hip_roll_link", "l_thigh_link", "l_calf_link",
    "l_ankle_pitch_link", "l_ankle_roll_link",
]


def _swap_index(names: list[str]) -> list[int]:
    idx = list(range(len(names)))
    for i, n in enumerate(names):
        if n.startswith("r_"):
            idx[i] = names.index("l_" + n[2:])
        elif n.startswith("l_"):
            idx[i] = names.index("r_" + n[2:])
    return idx


_J_SWAP = _swap_index(JOINT_NAMES)
_B_SWAP = _swap_index(BODY_NAMES)


def mirror_motion(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    joint_pos = d["joint_pos"]
    joint_vel = d["joint_vel"]
    body_pos_w = d["body_pos_w"]
    body_quat_w = d["body_quat_w"]
    body_lin_vel_w = d["body_lin_vel_w"]
    body_ang_vel_w = d["body_ang_vel_w"]

    assert joint_pos.shape[-1] == len(JOINT_NAMES), joint_pos.shape
    assert body_pos_w.shape[-2] == len(BODY_NAMES), body_pos_w.shape

    m_joint_pos = -joint_pos[:, _J_SWAP]
    m_joint_vel = -joint_vel[:, _J_SWAP]

    m_body_pos = body_pos_w[:, _B_SWAP, :].copy()
    m_body_pos[..., 1] *= -1

    m_body_quat = body_quat_w[:, _B_SWAP, :].copy()
    m_body_quat[..., 1] *= -1
    m_body_quat[..., 3] *= -1

    m_lin_vel = body_lin_vel_w[:, _B_SWAP, :].copy()
    m_lin_vel[..., 1] *= -1

    m_ang_vel = body_ang_vel_w[:, _B_SWAP, :].copy()
    m_ang_vel[..., 0] *= -1
    m_ang_vel[..., 2] *= -1

    return {
        "fps": d["fps"],
        "joint_pos": m_joint_pos.astype(np.float32),
        "joint_vel": m_joint_vel.astype(np.float32),
        "body_pos_w": m_body_pos.astype(np.float32),
        "body_quat_w": m_body_quat.astype(np.float32),
        "body_lin_vel_w": m_lin_vel.astype(np.float32),
        "body_ang_vel_w": m_ang_vel.astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of source AMP motion npz files")
    ap.add_argument("--out", required=True, help="output directory for mirrored npz files")
    ap.add_argument(
        "--suffix", default="_mirrored",
        help="appended to each output filename stem (default: _mirrored)",
    )
    args = ap.parse_args()

    src_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files found in {src_dir}")

    for f in files:
        d = np.load(f)
        mirrored = mirror_motion({k: d[k] for k in d.files})
        out_path = out_dir / f"{f.stem}{args.suffix}.npz"
        np.savez(out_path, **mirrored)
        print(f"{f.name} ({mirrored['joint_pos'].shape[0]} frames) -> {out_path}")


if __name__ == "__main__":
    main()
