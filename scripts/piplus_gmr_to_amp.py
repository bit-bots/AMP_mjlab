"""Convert the GMR-retargeted Pi Plus motion (qpos npz) to the AMP motion format.

Drives the qpos trajectory through a Pi Plus mjlab Entity and logs full-body world
kinematics (body_pos/quat/lin_vel/ang_vel) exactly as csv_to_npz does for G1, so the
body ordering + velocity conventions match what the AMP loader/discriminator use at
RL time. Input qpos is [root_pos(3), root_quat_wxyz(4), joints(20)] @ input fps.

Usage:
  .venv/bin/python scripts/piplus_gmr_to_amp.py \
    --gmr-npz /homes/17vahl/smp/smp/datasets/gmr_retarget_rlmodel.npz \
    --out src/assets/motions/piplus/amp/WalkandRun/piplus_walk.npz --out-fps 50
"""

from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation as R

from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from src.assets.robots.piplus import get_piplus_robot_cfg


def _resample(arr: np.ndarray, in_t: np.ndarray, out_t: np.ndarray) -> np.ndarray:
  return np.stack([np.interp(out_t, in_t, arr[:, c]) for c in range(arr.shape[1])], axis=1)


def _slerp_quat_wxyz(quats: np.ndarray, in_t: np.ndarray, out_t: np.ndarray) -> np.ndarray:
  from scipy.spatial.transform import Slerp

  rot = R.from_quat(quats[:, [1, 2, 3, 0]])  # wxyz -> xyzw
  out = Slerp(in_t, rot)(np.clip(out_t, in_t[0], in_t[-1])).as_quat()  # xyzw
  return out[:, [3, 0, 1, 2]]  # -> wxyz


def _ang_vel_world(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
  """World-frame angular velocity via central-difference of consecutive rotations."""
  r = R.from_quat(quats_wxyz[:, [1, 2, 3, 0]])
  n = len(r)
  out = np.zeros((n, 3))
  for i in range(n):
    i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
    out[i] = (r[i1] * r[i0].inv()).as_rotvec() / ((i1 - i0) * dt)
  return out


def main(
  gmr_npz: str,
  out: str,
  out_fps: int = 50,
  device: str = "cuda:0",
) -> None:
  data = np.load(gmr_npz)
  qpos = data["qpos"].astype(np.float64)  # (T, 27)
  in_fps = int(data["fps"])
  T = qpos.shape[0]
  print(f"loaded {gmr_npz}: {T} frames @ {in_fps}fps")

  root_pos, root_quat, dof = qpos[:, 0:3], qpos[:, 3:7], qpos[:, 7:]
  in_t = np.arange(T) / in_fps
  out_t = np.arange(0.0, (T - 1) / in_fps, 1.0 / out_fps)
  root_pos_i = _resample(root_pos, in_t, out_t)
  dof_i = _resample(dof, in_t, out_t)
  quat_i = _slerp_quat_wxyz(root_quat, in_t, out_t)
  dt = 1.0 / out_fps
  lin_vel = np.gradient(root_pos_i, dt, axis=0)
  dof_vel = np.gradient(dof_i, dt, axis=0)
  ang_vel = _ang_vel_world(quat_i, dt)
  n = len(out_t)
  print(f"resampled -> {n} frames @ {out_fps}fps")

  # Build a minimal Pi Plus scene + sim (flat plane).
  scene = Scene(
    SceneCfg(
      num_envs=1,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_piplus_robot_cfg()},
    ),
    device=device,
  )
  model = scene.compile()
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / out_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]

  # Joint names in the qpos column order (skip the free joint).
  joint_names = [
    mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i).split("/")[-1]
    for i in range(sim.mj_model.njnt)
    if sim.mj_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
  ]
  assert len(joint_names) == dof.shape[1], (len(joint_names), dof.shape[1])
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                          "body_lin_vel_w", "body_ang_vel_w")}
  scene.reset()
  dev = sim.device
  for f in range(n):
    rs = robot.data.default_root_state.clone()
    rs[:, 0:3] = torch.tensor(root_pos_i[f], device=dev, dtype=rs.dtype)
    rs[:, :2] += scene.env_origins[:, :2]
    rs[:, 3:7] = torch.tensor(quat_i[f], device=dev, dtype=rs.dtype)
    rs[:, 7:10] = torch.tensor(lin_vel[f], device=dev, dtype=rs.dtype)
    rs[:, 10:] = torch.tensor(ang_vel[f], device=dev, dtype=rs.dtype)
    robot.write_root_state_to_sim(rs)
    jp = robot.data.default_joint_pos.clone()
    jv = robot.data.default_joint_vel.clone()
    jp[:, robot_joint_indexes] = torch.tensor(dof_i[f], device=dev, dtype=jp.dtype)
    jv[:, robot_joint_indexes] = torch.tensor(dof_vel[f], device=dev, dtype=jv.dtype)
    robot.write_joint_state_to_sim(jp, jv)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  out_data = {"fps": np.array([out_fps], dtype=np.float64)}
  for k in log:
    out_data[k] = np.stack(log[k], axis=0).astype(np.float32)
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **out_data)
  print(f"\nsaved {out}")
  for k, v in out_data.items():
    print(f"  {k}: {v.shape}")


if __name__ == "__main__":
  tyro.cli(main)
