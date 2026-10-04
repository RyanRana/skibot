"""Renders a recorded hike: chase-cam MP4, a mid-run still and a 3 x 4 sheet of screengrabs along the walk.

    .venv/bin/python -m hikesim.render out/hike/runs/bird-hills-nature-area/smoke40
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hikesim.trees import camera_position, course_trees, draw_trees
from hikesim.rough import roughen
from hikesim.walk import crop, fine, hiking_model, load

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")


def model_for(run: Path, satellite: bool = True):
  info = json.loads((run / "run.json").read_text())
  f = np.load(run / "frames.npz")
  grid, route = load(info["course"])
  rxy = np.stack([route["x"], route["y"]], 1)
  seg = rxy[int(f["start"]):int(f["end"]) + 1]
  g = crop(grid, seg)
  origin = None
  if info.get("bumps", 0) > 0:  # the same bumpy surface the run walked
    origin = json.loads((Path(__file__).resolve().parents[1] / "out/hike/courses" / info["course"] / "grid.json").read_text())["utm_origin"]
    g = roughen(g, origin=origin, amp=info["bumps"])
  elif info.get("ckpt", "missing") != "missing":  # course-env runs walked the 0.5 m surface
    g = fine(g)
  return hiking_model(g, seg, satellite=info["course"] if satellite else None, relief=info.get("bumps", 0), origin=origin), f["qpos"], info, route


def _font(size: int):
  for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
    if Path(p).exists():
      return ImageFont.truetype(p, size)
  return ImageFont.load_default()


def render(run: Path, fps: int = 25, size=(960, 540), distance=5.0, elevation=-11.0, side=-40.0, sheet=(3, 4),
           trees: bool = True, satellite: bool = True):
  model, frames, info, route = model_for(run, satellite)
  forest = course_trees(info["course"]) if trees else None   # real trees from the canopy map, drawn 8-bit
  drawn = []
  model.vis.global_.offwidth, model.vis.global_.offheight = max(size[0], 1920), max(size[1], 1080)
  data = mujoco.MjData(model)
  r = mujoco.Renderer(model, size[1], size[0], max_geom=20000)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  tmp = run / "frames"
  shutil.rmtree(tmp, ignore_errors=True)
  tmp.mkdir()
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  look, head = None, None
  for i, q in enumerate(frames):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    p = data.xpos[pelvis].copy()
    look = p if look is None else 0.85 * look + 0.15 * p
    vel = frames[min(i + 12, len(frames) - 1)][0:2] - frames[max(i - 12, 0)][0:2]
    h = np.degrees(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel) > 0.05 else (head if head is not None else 0.0)
    head = h if head is None else head + 0.08 * ((h - head + 180) % 360 - 180)
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = distance, head + side, elevation
    r.update_scene(data, cam)
    if forest is not None:
      drawn.append(draw_trees(r.scene, forest, look, camera_position(cam)))
    Image.fromarray(r.render()).save(tmp / f"{i:04d}.png")
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(run / "chase.mp4")], check=True)
  (run / "render.json").write_text(json.dumps({
    "video": "chase.mp4", "frames": len(frames), "fps": fps, "trees": forest is not None,
    "trees_in_view_mean": round(float(np.mean(drawn)), 1) if drawn else 0}, indent=1))
  Image.open(tmp / f"{len(frames) // 2:04d}.png").save(run / "still.jpg", quality=88)

  # Screengrab sheet: evenly spaced moments along the walk, labelled with time and trail grade.
  rows, cols = sheet
  n = rows * cols
  pick = np.linspace(0, len(frames) - 1, n + 2)[1:-1].astype(int)
  tw, th = size[0] // 2, size[1] // 2
  canvas = Image.new("RGB", (cols * tw, rows * th + 40), "white")
  draw = ImageDraw.Draw(canvas)
  f_small, f_title = _font(16), _font(20)
  prog = np.load(run / "frames.npz")["progress"]
  grade = np.array(route["grade_deg"])
  for k, i in enumerate(pick):
    im = Image.open(tmp / f"{i:04d}.png").resize((tw, th), Image.LANCZOS)
    x, y = (k % cols) * tw, 40 + (k // cols) * th
    canvas.paste(im, (x, y))
    gi = int(info["start_m"]) + int(prog[min(i, len(prog) - 1)])
    label = f"{i / fps:4.1f} s   {gi - info['start_m']:.0f} m   grade {grade[min(gi, len(grade) - 1)]:+.0f} deg"
    bx = draw.textbbox((x + 8, y + 6), label, font=f_small)
    draw.rectangle((bx[0] - 4, bx[1] - 3, bx[2] + 4, bx[3] + 3), fill=(15, 18, 16))
    draw.text((x + 8, y + 6), label, fill=(240, 240, 235), font=f_small)
  draw.text((10, 9), f"{info['area']}  |  {info['result']}, {info['walked_m']} m at {info['mean_speed_mps']} m/s  |  "
                     f"steepest {info['max_grade_deg']} deg", fill=(0, 0, 0), font=f_title)
  canvas.save(run / "sheet.jpg", quality=88)
  return run


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+", type=Path)
  a = ap.parse_args()
  for run in a.runs:
    render(run)
    print(f"rendered {run}")


if __name__ == "__main__":
  main()
