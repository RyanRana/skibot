"""Chase-cam MP4/GIF of a single-robot run recorded by skisim.tests.test_sonic.run(record=...)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")


def render(rec: dict, out_stem: Path, fps: int = 25, size=(960, 540), distance=3.6, elevation=-12.0, side=-35.0):
  model, frames = rec["model"], rec["frames"]
  data = mujoco.MjData(model)
  r = mujoco.Renderer(model, size[1], size[0])
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  tmp = out_stem.parent / (out_stem.name + "_frames")
  tmp.mkdir(parents=True, exist_ok=True)
  for f in tmp.glob("*.png"):
    f.unlink()
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  look = None
  for i, q in enumerate(frames):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    p = data.xpos[pelvis].copy()
    look = p if look is None else 0.7 * look + 0.3 * p
    vel = (frames[min(i + 1, len(frames) - 1)][0:2] - frames[max(i - 1, 0)][0:2])
    head = np.degrees(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel) > 1e-3 else 0.0
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = distance, head + side, elevation
    r.update_scene(data, cam)
    Image.fromarray(r.render()).save(tmp / f"{i:04d}.png")
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_stem.with_suffix(".mp4"))], check=True)
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                  "-vf", "fps=15,scale=480:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                  str(out_stem.with_suffix(".gif"))], check=True)
  mid = len(frames) // 2
  Image.open(tmp / f"{mid:04d}.png").save(out_stem.parent / (out_stem.name + "_still.png"))
  return out_stem
