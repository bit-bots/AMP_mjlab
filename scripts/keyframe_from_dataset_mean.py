"""Set the Pi Plus home keyframe to the reference dataset's mean pose.

The action term uses ``JointPositionActionCfg(use_default_offset=True)``, so the
joint target is ``default_joint_pos + scale * action`` -- i.e. zero action holds
the keyframe pose. If the keyframe is far from the motion the policy must
produce, the policy can never bring those joints onto the manifold.
Re-centering the keyframe on the dataset mean puts zero action *on* the motion
manifold so the policy only learns the residual oscillation.

Computes, from every AMP-format npz in a directory (recursively, following
symlinks -- so pointing this at WalkandRun/ picks up both the real files and
the CMU_Locomotion*/ clips symlinked into it):
  - per-joint mean angle (in the compiled sim joint order), and
  - the mean pelvis height (body_pos_w[:, 0, 2], body index 0 = base_link),
then rewrites the ``HOME_KEYFRAME`` block in ``piplus_constants.py`` in place.

Usage:
  .venv/bin/python scripts/keyframe_from_dataset_mean.py                     # rewrite the file
  .venv/bin/python scripts/keyframe_from_dataset_mean.py --no-write          # preview only
  .venv/bin/python scripts/keyframe_from_dataset_mean.py \
    --motion-dir src/assets/motions/piplus/amp/WalkandRun
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from src.assets.robots.piplus import get_piplus_robot_cfg


@dataclass
class Cfg:
  motion_dir: str = "src/assets/motions/piplus/amp/WalkandRun"
  """Directory of AMP-format motion npz files (has joint_pos/body_pos_w) to
  average over -- every *.npz found here (symlinks followed) counts once."""
  constants_file: str = "src/assets/robots/piplus/piplus_constants.py"
  """File whose HOME_KEYFRAME block is rewritten."""
  write: bool = True
  """If false, only print the computed keyframe (no file edit)."""


def _render_block(joint_names: list[str], means: np.ndarray, root_z: float) -> str:
  lines = [
    "HOME_KEYFRAME = EntityCfg.InitialStateCfg(",
    f"  pos=(0, 0, {root_z:.4f}),",
    "  joint_pos={",
  ]
  for name, val in zip(joint_names, means):
    lines.append(f'    "{name}": {val:.4f},')
  lines.append("  },")
  lines.append('  joint_vel={".*": 0.0},')
  lines.append(")")
  return "\n".join(lines)


def main(cfg: Cfg) -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Authoritative sim joint order.
  scene = Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_piplus_robot_cfg()},
      num_envs=1,
      extent=2.0,
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  joint_names = list(robot.joint_names)

  files = sorted(glob.glob(os.path.join(cfg.motion_dir, "*.npz")))
  if not files:
    raise SystemExit(f"no .npz files found in {cfg.motion_dir}")

  # AMP npz joint_pos columns are already in this same compiled sim joint
  # order (piplus_gmr_to_amp.py writes them via the same robot.joint_names
  # lookup), so no reordering needed.
  all_jp, all_root_z = [], []
  for f in files:
    d = np.load(f)
    assert d["joint_pos"].shape[-1] == len(joint_names), (
      f"{f}: {d['joint_pos'].shape[-1]} joints != {len(joint_names)} expected"
    )
    all_jp.append(d["joint_pos"])
    all_root_z.append(d["body_pos_w"][:, 0, 2])
  jp = np.concatenate(all_jp, axis=0)
  root_z_mean = float(np.concatenate(all_root_z, axis=0).mean())
  joint_mean = jp.mean(axis=0)

  # Current defaults for the before/after diff.
  cur_default = robot.data.default_joint_pos[0].detach().cpu().numpy()

  print(f"{len(files)} files, {jp.shape[0]} total frames, from {cfg.motion_dir}")
  print(f"mean pelvis height: {root_z_mean:.4f} m\n")
  print(f"{'joint':24s} {'current':>9s} {'new(mean)':>9s} {'diff':>8s}")
  for n, c, m in zip(joint_names, cur_default, joint_mean):
    print(f"{n:24s} {c:9.4f} {m:9.4f} {m - c:8.4f}")

  block = _render_block(joint_names, joint_mean, root_z_mean)
  print("\n--- new HOME_KEYFRAME ---")
  print(block)

  if not cfg.write:
    print("\n[--no-write] file left unchanged.")
    return

  path = Path(cfg.constants_file)
  text = path.read_text()
  pattern = re.compile(
    r"HOME_KEYFRAME = EntityCfg\.InitialStateCfg\(.*?\n\)", re.DOTALL
  )
  new_text, n = pattern.subn(block, text)
  if n != 1:
    raise RuntimeError(
      f"expected exactly one HOME_KEYFRAME block, matched {n}. Aborting."
    )
  path.write_text(new_text)
  print(f"\nWrote HOME_KEYFRAME to {path} ({n} block replaced).")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
