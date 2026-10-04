"""USGS NAIP aerial photo (public domain) draped on the MuJoCo ground, so renders show the real place from above.

The course photo is fetched once at ~0.3 m and cached as out/hike/courses/<slug>/naip.jpg; each render crops it to its
terrain window and writes it into the ground material's texture (MuJoCo maps a 2D texture with texrepeat 1 1 once
across the whole hfield, north-up like the photo).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from hikesim.export_web import naip
from skisim.terrain import HeightGrid

ROOT = Path(__file__).resolve().parents[1]
COURSES = ROOT / "out" / "hike" / "courses"


def course_naip(slug: str, px: float = 0.3) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
  """The course's aerial photo and its bounds in course-local metres (x0, y0, x1, y1)."""
  d = COURSES / slug
  g = json.loads((d / "grid.json").read_text())
  ox, oy = g["utm_origin"]
  w, h = g["ncol"] * g["cell"], g["nrow"] * g["cell"]
  path = d / "naip.jpg"
  epsg = int(str(g.get("crs", "EPSG:26917")).split(":")[-1])
  if not naip(ox, oy, ox + w, oy + h, px, path, epsg=epsg):
    return None
  return np.asarray(Image.open(path).convert("RGB")), (0.0, 0.0, w, h)


DIRT = np.array([112, 90, 62], np.float32)


def drape(model: mujoco.MjModel, grid: HeightGrid, slug: str, route: np.ndarray | None = None, material: str = "snow",
          texture: str = "snowtex", brightness: float = 1.0, trail_width: float = 0.9, relief: float = 1.0,
          origin=None) -> bool:
  """Write the aerial photo under `grid`'s window into the ground texture, with the trail painted in as packed dirt
  and fine leaf-litter grain on top (the photo is 0.3 m per pixel, the texture ~5 cm). Call before creating a Renderer."""
  got = course_naip(slug)
  if got is None:
    return False
  img, (bx0, by0, bx1, by1) = got
  H, W = img.shape[:2]
  x0, x1, y0, y1 = grid.bounds()
  # photo pixel columns run west to east from bx0, rows north to south from by1
  c0, c1 = (x0 - bx0) / (bx1 - bx0) * W, (x1 - bx0) / (bx1 - bx0) * W
  r0, r1 = (by1 - y1) / (by1 - by0) * H, (by1 - y0) / (by1 - by0) * H
  t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture)
  m = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material)
  tw, th, nc = int(model.tex_width[t]), int(model.tex_height[t]), int(model.tex_nchannel[t])
  crop = Image.fromarray(img).crop((int(c0), int(r0), int(np.ceil(c1)), int(np.ceil(r1)))).resize((tw, th), Image.LANCZOS)
  a = np.asarray(crop, np.float32) * brightness
  from hikesim.rough import _value_noise
  # Texel centres in local metres, for grain that stays put as the camera moves.
  X, Y = np.meshgrid(x0 + (np.arange(tw) + 0.5) / tw * (x1 - x0), y1 - (np.arange(th) + 0.5) / th * (y1 - y0))
  grain = 0.6 * _value_noise(X, Y, 0.12, 11) + 0.4 * _value_noise(X, Y, 0.35, 12)
  a *= (1 + 0.28 * grain)[..., None]
  # Relief: shade every texel from the grid's slope plus the exact micro-relief (hikesim.rough at texel resolution),
  # so roots and rocks read even where the 0.25 m mesh smooths them; tint rock crowns grey and root crests dark.
  if relief:
    from hikesim.rough import bumps as micro
    o = origin or (0.0, 0.0)
    bz = micro(X + o[0], Y + o[1], amp=relief) if relief > 0 else np.zeros_like(X)
    hz, _ = grid.height_normal(X.ravel(), Y.ravel())
    zt = hz.reshape(X.shape) + bz * 0  # mesh already carries the 0.25 m bumps; add only the finer part below
    px = (x1 - x0) / tw
    gy_, gx_ = np.gradient(zt + bz, px)
    lx, ly, lz = 0.62, 0.38, 0.55                               # same direction as the lowered sun
    nrm = np.sqrt(gx_ ** 2 + gy_ ** 2 + 1)
    shade = np.clip((-gx_ * lx + gy_ * ly + lz) / nrm / lz, 0.35, 1.45)  # rows run north to south, so +gy flips
    a *= shade[..., None] ** 1.2
    rock = np.clip((bz - 0.05) / 0.04, 0, 1)[..., None]
    a = a * (1 - 0.7 * rock) + np.array([128, 124, 116], np.float32) * (1 + 0.15 * grain[..., None]) * 0.7 * rock
  if route is not None and len(route) > 1:
    from PIL import ImageDraw, ImageFilter
    mask = Image.new("L", (tw, th), 0)
    pts = [((x - x0) / (x1 - x0) * tw, (y1 - y) / (y1 - y0) * th) for x, y in route]
    ImageDraw.Draw(mask).line(pts, fill=255, width=max(2, int(trail_width / (x1 - x0) * tw)), joint="curve")
    mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, 0.12 / (x1 - x0) * tw)))
    w = (np.asarray(mask, np.float32) / 255)[..., None] * 0.9
    dirt = DIRT * (1 + 0.22 * grain)[..., None]
    a = a * (1 - w) + dirt * w
  a = np.clip(a, 0, 255).astype(np.uint8)
  if nc == 4:
    a = np.concatenate([a, np.full((th, tw, 1), 255, np.uint8)], -1)
  adr = int(model.tex_adr[t])
  model.tex_data[adr:adr + tw * th * nc] = a.reshape(-1)
  model.mat_texrepeat[m] = (1, 1)
  model.mat_texuniform[m] = 0
  model.mat_rgba[m] = (1, 1, 1, 1)
  model.mat_specular[m] = 0.02
  model.mat_shininess[m] = 0.02
  return True
