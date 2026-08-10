"""Viser playback of AMP motion npz(s) (body_pos_w/body_quat_w) on the current Pi
Plus model meshes. Plays the recorded body world poses directly (no qpos), so it
shows exactly what the AMP loader will see.

Single file:
  .venv/bin/python scripts/amp_npz_viz.py --npz src/assets/motions/piplus/amp/Kick/kick_10_01.npz --port 8080

Whole directory, selectable via a dropdown (defaults to the Kick clips):
  .venv/bin/python scripts/amp_npz_viz.py --dir src/assets/motions/piplus/amp/Kick --port 8080
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

from src.assets.robots.piplus.piplus_constants import PIPLUS_XML, get_assets


def _geom_mesh(model, gid):
  did = int(model.geom_dataid[gid])
  va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
  fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
  v = np.asarray(model.mesh_vert[va:va + vn]).reshape(-1, 3).astype(np.float32)
  f = np.asarray(model.mesh_face[fa:fa + fn]).reshape(-1, 3).astype(np.int32)
  return v, f


def _load(npz_path: Path):
  d = np.load(npz_path)
  bpw = d["body_pos_w"]   # (T, nb, 3)
  bqw = d["body_quat_w"]  # (T, nb, 4) wxyz
  fps = float(np.ravel(d["fps"])[0])
  return bpw, bqw, fps


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", type=str, default=None, help="single AMP motion npz")
  ap.add_argument("--dir", type=str, default=None,
                   help="directory of AMP motion npz files, selectable via dropdown")
  ap.add_argument("--port", type=int, default=8080)
  args = ap.parse_args()
  if not args.npz and not args.dir:
    ap.error("pass --npz or --dir")

  motions: dict[str, Path] = {}
  if args.dir:
    for p in sorted(Path(args.dir).glob("*.npz")):
      motions[p.stem] = p
  if args.npz:
    p = Path(args.npz)
    motions[p.stem] = p
  names = list(motions.keys())
  print(f"loaded {len(names)} motion(s): {names}")

  cache: dict[str, tuple] = {}

  def get(name: str):
    if name not in cache:
      cache[name] = _load(motions[name])
      bpw, _, fps = cache[name]
      print(f"  {name}: {bpw.shape[0]} frames, {bpw.shape[1]} bodies @ {fps}fps")
    return cache[name]

  spec = mj.MjSpec.from_file(str(PIPLUS_XML))
  spec.meshdir = "meshes"
  spec.assets = get_assets("meshes")
  model = spec.compile()
  # npz body order == mjlab entity bodies (world excluded) -> npz idx = geom_bodyid - 1
  geoms = []
  for gid in range(model.ngeom):
    if int(model.geom_type[gid]) != int(mj.mjtGeom.mjGEOM_MESH):
      continue
    bid = int(model.geom_bodyid[gid])
    if bid == 0:
      continue
    v, f = _geom_mesh(model, gid)
    gpos = np.asarray(model.geom_pos[gid])
    gquat = np.asarray(model.geom_quat[gid])  # wxyz
    geoms.append((gid, bid - 1, gpos, gquat, v, f))

  server = viser.ViserServer(port=args.port)
  server.scene.add_grid("/grid", width=4.0, height=4.0)
  handles = []
  for gid, npz_idx, gpos, gquat, v, f in geoms:
    h = server.scene.add_mesh_simple(f"/g{gid}", vertices=v, faces=f,
                                     color=(160, 160, 170), opacity=0.9)
    handles.append((npz_idx, gpos, gquat, h))
  print(f"rendering {len(handles)} geoms; Viser on http://localhost:{args.port}")

  speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
  playing = {"v": True}
  btn = server.gui.add_button("Play/Pause")

  @btn.on_click
  def _(_e):
    playing["v"] = not playing["v"]

  state = {"name": names[0], "slider": None}

  def make_slider(name: str):
    if state["slider"] is not None:
      state["slider"].remove()
    bpw, _, _ = get(name)
    state["slider"] = server.gui.add_slider(
      "Frame", min=0, max=bpw.shape[0] - 1, step=1, initial_value=0
    )

  make_slider(state["name"])

  # One standalone button per motion (not a dropdown/text combobox, so tapping
  # it doesn't pop the on-screen keyboard on touch devices; and not a button
  # group, which wraps all options onto one row -- separate buttons stack one
  # per line in the GUI panel).
  for name in names:
    def _select(_e, name=name):
      state["name"] = name
      make_slider(name)
    server.gui.add_button(name).on_click(_select)

  def render(name: str, fr: int):
    bpw, bqw, _ = get(name)
    fr = min(fr, bpw.shape[0] - 1)
    for npz_idx, gpos, gquat, h in handles:
      bp = bpw[fr, npz_idx]
      Rb = R.from_quat(bqw[fr, npz_idx][[1, 2, 3, 0]])
      wpos = bp + Rb.apply(gpos)
      wq = (Rb * R.from_quat(gquat[[1, 2, 3, 0]])).as_quat()  # xyzw
      h.position = (float(wpos[0]), float(wpos[1]), float(wpos[2]))
      h.wxyz = (float(wq[3]), float(wq[0]), float(wq[1]), float(wq[2]))

  while True:
    name = state["name"]
    _, _, fps = get(name)
    try:
      fslider = state["slider"]
      render(name, int(fslider.value))
      if playing["v"]:
        fslider.value = (int(fslider.value) + 1) % (get(name)[0].shape[0])
    except RuntimeError:
      # The Motion selector swapped the slider out from under us (removed
      # GuiSliderHandle) between the read above and this write; state["slider"]
      # already points at the new one, so just retry next tick.
      pass
    time.sleep(1.0 / (fps * speed.value))


if __name__ == "__main__":
  main()
