"""Localize the expert-vs-policy AMP-obs mismatch:
 (A) transform consistency: AMPLoader obs vs the mdp transform applied to the SAME npz frame.
 (B) sim reproduction: does setting the sim to a frame reproduce the npz world body poses?
"""

import numpy as np
import torch

import src.tasks.amp_loco.config.piplus  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)
from rsl_rl.utils import AMPLoader
from src.tasks.amp_loco.config.piplus.env_cfgs import AMP_BODY_NAMES, ANCHOR_NAME

NPZ = "src/assets/motions/piplus/amp/WalkandRun/piplus_walk.npz"
DEV = "cuda:0"

env_cfg = load_env_cfg("PiPlus-AMP-Flat")
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(env_cfg, device=DEV)
env.reset()
robot = env.scene["robot"]
bn = list(robot.body_names)
body_ids = [bn.index(n) for n in AMP_BODY_NAMES]
anchor_id = bn.index(ANCHOR_NAME)

loader = AMPLoader(motion_file=NPZ, body_names=list(AMP_BODY_NAMES), anchor_name=ANCHOR_NAME,
                   all_body_names=bn, device=DEV)
d = np.load(NPZ)
t = lambda a: torch.tensor(a, device=DEV, dtype=torch.float32)  # noqa: E731

for T in (100, 1500):
  # --- (A) transform consistency: apply the mdp math directly to the npz frame ---
  bpw = t(d["body_pos_w"][T])[None]      # (1, nb_all, 3)
  bqw = t(d["body_quat_w"][T])[None]
  blw = t(d["body_lin_vel_w"][T])[None]
  baw = t(d["body_ang_vel_w"][T])[None]
  a_pos = bpw[:, anchor_id][:, None].expand(-1, len(body_ids), -1)
  a_quat = bqw[:, anchor_id][:, None].expand(-1, len(body_ids), -1)
  b_pos, b_quat = subtract_frame_transforms(a_pos, a_quat, bpw[:, body_ids], bqw[:, body_ids])
  mdp_pos = b_pos.reshape(-1)
  mdp_ori = matrix_from_quat(b_quat)[..., :2].reshape(-1)
  mdp_lin = quat_apply_inverse(bqw[:, body_ids].reshape(-1, 4), blw[:, body_ids].reshape(-1, 3)).reshape(-1)
  exp_pos = loader._body_pos_b[T].reshape(-1)
  exp_ori = loader._body_ori_b[T].reshape(-1)
  exp_lin = loader._body_lin_vel_b[T].reshape(-1)
  print(f"\n=== frame {T} (A) mdp-transform-on-npz  vs  AMPLoader-expert ===")
  print("  pos Δ:", f"{(mdp_pos-exp_pos).abs().max().item():.4f}",
        "| ori Δ:", f"{(mdp_ori-exp_ori).abs().max().item():.4f}",
        "| lin Δ:", f"{(mdp_lin-exp_lin).abs().max().item():.4f}")

  # --- (B) sim reproduction: set sim to frame T, read world body poses ---
  rs = robot.data.default_root_state.clone()
  rs[:, 0:3] = t(d["body_pos_w"][T, 0]); rs[:, :2] += env.scene.env_origins[:, :2]
  rs[:, 3:7] = t(d["body_quat_w"][T, 0])
  rs[:, 7:10] = t(d["body_lin_vel_w"][T, 0]); rs[:, 10:] = t(d["body_ang_vel_w"][T, 0])
  robot.write_root_state_to_sim(rs)
  jp = robot.data.default_joint_pos.clone(); jv = robot.data.default_joint_vel.clone()
  jp[:] = t(d["joint_pos"][T]); jv[:] = t(d["joint_vel"][T])
  robot.write_joint_state_to_sim(jp, jv)
  env.sim.forward(); env.scene.update(env.physics_dt)
  sim_bpw = robot.data.body_link_pos_w[0].detach()
  origin = env.scene.env_origins[0]
  diff = (sim_bpw[body_ids] - origin - t(d["body_pos_w"][T])[body_ids]).abs()
  print(f"    (B) sim-vs-npz world body_pos max Δ: {diff.max().item():.4f} (should ~0)")

  # --- (C) env 'amp' obs group vs the raw mdp transform on the SAME sim state ---
  env_amp = env.observation_manager.compute()["amp"][0].detach()
  n = len(body_ids)
  segs = {"pos": (0, 3 * n), "ori": (3 * n, 9 * n), "lin": (9 * n, 12 * n), "ang": (12 * n, 15 * n)}
  # raw mdp transform on sim state:
  sbpw = robot.data.body_link_pos_w[0][None]; sbqw = robot.data.body_link_quat_w[0][None]
  sblw = robot.data.body_link_lin_vel_w[0][None]; sbaw = robot.data.body_link_ang_vel_w[0][None]
  ap = sbpw[:, anchor_id][:, None].expand(-1, n, -1); aq = sbqw[:, anchor_id][:, None].expand(-1, n, -1)
  pb, qb = subtract_frame_transforms(ap, aq, sbpw[:, body_ids], sbqw[:, body_ids])
  raw = torch.cat([pb.reshape(-1), matrix_from_quat(qb)[..., :2].reshape(-1),
                   quat_apply_inverse(sbqw[:, body_ids].reshape(-1, 4), sblw[:, body_ids].reshape(-1, 3)).reshape(-1),
                   quat_apply_inverse(sbqw[:, body_ids].reshape(-1, 4), sbaw[:, body_ids].reshape(-1, 3)).reshape(-1)])
  print("    (C) env-amp-group vs raw-mdp-on-sim:", {k: round((env_amp[a:b] - raw[a:b]).abs().max().item(), 4) for k, (a, b) in segs.items()})
  if T == 100:
    amp_grp = env.observation_manager.cfg["amp"] if isinstance(env.observation_manager.cfg, dict) else None
    print("    amp group corruption:", getattr(env_cfg.observations["amp"], "enable_corruption", "?"),
          "| term order:", list(env_cfg.observations["amp"].terms.keys()))
