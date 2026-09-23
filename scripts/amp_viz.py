"""Viser preview of AMP-format Pi Plus motion npz files (post piplus_gmr_to_amp.py).

Unlike gmr_viz.py (which drives raw qpos through MuJoCo FK), the AMP format
already stores per-body world kinematics (body_pos_w/body_quat_w) logged
straight from the mjlab Entity used at RL time -- there's no qpos to re-derive
FK from, and no guarantee the AMP-format body order matches a fresh qpos
vector. So this script poses each mesh geom directly from the logged body
pose (body_pos_w/quat_w) plus that geom's fixed local offset within its body,
read once from the training MJCF.

Runs in the GMR venv (mujoco + viser), not this repo's own .venv.

Usage (GMR venv), single clip:
  ../gmrvenv/bin/python scripts/amp_viz.py \
    --npz /path/to/amp_format.npz \
    --xml src/assets/robots/piplus/xmls/piplus.xml \
    --port 8080

Usage, browse every clip in one or more directories (adds a clip picker,
grouped by directory -- a button per clip rather than a dropdown, since
viser's dropdown is unusable on mobile):
  ../gmrvenv/bin/python scripts/amp_viz.py \
    --dir /homes/17vahl/smp/AMP_mjlab-kick-noballvel/src/assets/motions/piplus/amp/Kick \
    --dir src/assets/motions/piplus/amp/WalkandRun \
    --port 8080
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import mujoco as mj
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R


def _geom_mesh(model, gid: int):
  did = int(model.geom_dataid[gid])
  va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
  fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
  verts = np.asarray(model.mesh_vert[va : va + vn]).reshape(-1, 3).astype(np.float32)
  faces = np.asarray(model.mesh_face[fa : fa + fn]).reshape(-1, 3).astype(np.int32)
  return verts, faces


def _load_clip(path: str) -> tuple[np.ndarray, np.ndarray, float]:
  data = np.load(path, allow_pickle=True)
  body_pos_w = data["body_pos_w"]
  body_quat_w = data["body_quat_w"]
  fps = float(data["fps"][0]) if np.ndim(data["fps"]) else float(data["fps"])
  return body_pos_w, body_quat_w, fps


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", default=None, help="a single AMP-format motion npz (has body_pos_w/body_quat_w)")
  ap.add_argument(
    "--dir", default=None, action="append",
    help="directory of AMP-format npz files to browse (repeatable; each becomes its own group)",
  )
  ap.add_argument(
    "--xml", default="/homes/17vahl/smp/AMP_mjlab/src/assets/robots/piplus/xmls/piplus.xml",
    help="training MJCF -- only used for mesh geometry + each geom's local offset in its body",
  )
  ap.add_argument("--port", type=int, default=8080)
  args = ap.parse_args()

  if (args.npz is None) == (not args.dir):
    raise SystemExit("pass exactly one of --npz or one-or-more --dir")

  # (group_label, path) pairs, in the order clips should appear.
  clip_entries: list[tuple[str, str]] = []
  if args.dir:
    for d in args.dir:
      paths = sorted(glob.glob(os.path.join(d, "*.npz")))
      if not paths:
        raise SystemExit(f"no .npz files found in {d}")
      group = os.path.basename(os.path.normpath(d))
      clip_entries.extend((group, p) for p in paths)
  else:
    clip_entries = [("clip", args.npz)]

  clip_paths = [p for _, p in clip_entries]
  clip_groups = [g for g, _ in clip_entries]
  clip_names = [os.path.splitext(os.path.basename(p))[0] for p in clip_paths]

  clips = [_load_clip(p) for p in clip_paths]
  print(f"loaded {len(clips)} clip(s): {clip_names}")
  body_pos_w, body_quat_w, fps = clips[0]
  print(f"body_pos_w {body_pos_w.shape}, body_quat_w {body_quat_w.shape}, fps {fps}")

  model = mj.MjModel.from_xml_path(args.xml)
  # mjlab's per-entity body_link_* logs exclude the implicit "world" body (id 0),
  # so array index i corresponds to model body id (i + 1). Sanity-check the count
  # instead of assuming it silently.
  n_named_bodies = model.nbody - 1
  for name, (bp, _, _) in zip(clip_names, clips):
    assert bp.shape[1] == n_named_bodies, (
      f"{name}: AMP npz has {bp.shape[1]} bodies but {args.xml} has {n_named_bodies} "
      f"non-world bodies -- body-order assumption (array index i == mj body id i+1) "
      f"doesn't hold, fix the offset before trusting this viewer."
    )
  print("body order (npz index -> mj body name):")
  for i in range(min(5, n_named_bodies)):
    print(f"  {i} -> {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i + 1)}")
  print("  ...")

  # Each mesh geom's fixed pose within its own body (local frame), read once.
  geoms = []
  for gid in range(model.ngeom):
    if int(model.geom_type[gid]) != int(mj.mjtGeom.mjGEOM_MESH):
      continue
    rgba = np.asarray(model.geom_rgba[gid])
    if rgba[3] == 0:
      continue
    bid = int(model.geom_bodyid[gid])
    local_pos = np.asarray(model.geom_pos[gid])
    local_quat_wxyz = np.asarray(model.geom_quat[gid])  # wxyz
    geoms.append((gid, bid, local_pos, local_quat_wxyz, rgba))

  server = viser.ViserServer(port=args.port)
  server.scene.add_grid("/grid", width=4.0, height=4.0)

  handles = []
  for gid, bid, local_pos, local_quat_wxyz, rgba in geoms:
    verts, faces = _geom_mesh(model, gid)
    h = server.scene.add_mesh_simple(
      f"/geom_{gid}",
      vertices=verts,
      faces=faces,
      color=tuple((rgba[:3] * 255).astype(np.uint8).tolist()),
      opacity=float(rgba[3]),
    )
    handles.append((gid, bid, local_pos, local_quat_wxyz, h))
  print(f"rendering {len(handles)} mesh geoms")

  state = {"clip_idx": 0, "playing": True}

  current_label = server.gui.add_markdown(f"**Playing:** {clip_names[0]}")

  def select_clip(idx: int) -> None:
    state["clip_idx"] = idx
    new_T = clips[idx][0].shape[0]
    frame_slider.max = new_T - 1
    frame_slider.value = 0
    current_label.content = f"**Playing:** {clip_names[idx]}"

  # Button-per-clip picker, grouped by source directory. A dropdown is
  # unusable on mobile (small tap target, scroll-vs-open conflicts), so use
  # one button per clip instead, nested under a folder per group.
  seen_groups: set[str] = set()
  for group in clip_groups:
    if group in seen_groups:
      continue
    seen_groups.add(group)
    with server.gui.add_folder(group):
      for idx, (g, name) in enumerate(zip(clip_groups, clip_names)):
        if g != group:
          continue
        btn = server.gui.add_button(name)

        def _make_handler(i: int):
          def _(_e) -> None:
            select_clip(i)
          return _

        btn.on_click(_make_handler(idx))

  frame_slider = server.gui.add_slider(
    "Frame", min=0, max=clips[0][0].shape[0] - 1, step=1, initial_value=0
  )
  speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
  play = server.gui.add_button("Play / Pause")

  @play.on_click
  def _(_e) -> None:
    state["playing"] = not state["playing"]

  local_rot_cache = {}

  def render(clip_idx: int, frame: int) -> None:
    body_pos_w, body_quat_w, _ = clips[clip_idx]
    for gid, bid, local_pos, local_quat_wxyz, h in handles:
      npz_idx = bid - 1  # world body excluded, see assertion above
      body_p = body_pos_w[frame, npz_idx]
      body_q_wxyz = body_quat_w[frame, npz_idx]
      body_rot = R.from_quat(body_q_wxyz, scalar_first=True)
      if gid not in local_rot_cache:
        local_rot_cache[gid] = R.from_quat(local_quat_wxyz, scalar_first=True)
      local_rot = local_rot_cache[gid]
      world_rot = body_rot * local_rot
      world_pos = body_p + body_rot.apply(local_pos)
      quat_xyzw = world_rot.as_quat()
      h.position = (float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
      h.wxyz = (
        float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]),
      )

  print(f"Viser on http://localhost:{args.port}")
  while True:
    clip_idx = state["clip_idx"]
    _, _, clip_fps = clips[clip_idx]
    T = clips[clip_idx][0].shape[0]
    render(clip_idx, int(frame_slider.value))
    if state["playing"]:
      frame_slider.value = (int(frame_slider.value) + 1) % T
    time.sleep(1.0 / (clip_fps * speed.value))


if __name__ == "__main__":
  main()
