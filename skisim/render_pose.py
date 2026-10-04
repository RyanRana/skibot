"""Side by side: race frames with detected skeletons, and the G1 holding the measured neutral and turn poses."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from skisim.scene import g1_model, place
from skisim.ski import SkiParams
from skisim.sonic import DEFAULT_ANGLES, JOINTS
from skisim.terrain import slope

ROOT = Path(__file__).resolve().parents[1]


def main():
  poses = json.loads((ROOT / "data/pose/ski_pose.json").read_text())
  grid = slope(12.0, length=60, width=30, cell=0.5)
  p = SkiParams()
  model = g1_model(grid, p)
  data = mujoco.MjData(model)
  r = mujoco.Renderer(model, 420, 360)
  tiles = []
  for name in ("left_turn", "neutral", "right_turn"):
    q = np.zeros(model.nq)
    qj = DEFAULT_ANGLES.copy()
    for j, v in poses[name]["g1"].items():
      qj[JOINTS.index(j)] = v
    q[7:] = qj
    place(model, data, grid, (20.0, 0.0), 0.0, joint_qpos=q, params=p)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.xpos[1] - np.array([0, 0, 0.15])
    cam.distance, cam.azimuth, cam.elevation = 2.6, 180.0, -8.0  # from downhill, facing the skier like a broadcast camera
    r.update_scene(data, cam)
    im = Image.fromarray(r.render())
    ImageDraw.Draw(im).text((10, 8), f"G1 {name.replace('_', ' ')}", fill=(20, 30, 45))
    tiles.append(im)
  overlays = sorted((ROOT / "out/pose/overlays").glob("*.jpg"))[:3]
  row2 = [Image.open(o).resize((360, 203)) for o in overlays]
  W = 360 * 3
  canvas = Image.new("RGB", (W, 420 + 203), (245, 247, 250))
  for i, t in enumerate(tiles):
    canvas.paste(t, (360 * i, 0))
  for i, t in enumerate(row2):
    canvas.paste(t, (360 * i, 420))
  out = ROOT / "out/pose/compare.jpg"
  canvas.save(out, quality=85)
  print(out)


if __name__ == "__main__":
  main()
