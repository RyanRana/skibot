"""Real trees for the hiking courses, drawn 8-bit: every tree stands where Meta's 1 m canopy height map finds a crown.

    .venv/bin/python -m hikesim.trees bluffs-nature-area      # one course
    .venv/bin/python -m hikesim.trees --all                   # every built course

Source: Meta and WRI High Resolution Canopy Height Maps (Tolan et al. 2024, CC BY 4.0), 1 m tree heights read straight
from the public cloud tile over the course (about 25 s, only the rows we need). Tree tops are local maxima of the
smoothed canopy, thinned so crowns do not overlap, and canopy they leave uncovered is filled at its measured height.
All trees are broadleaf in October Ann Arbor colours: the map cannot tell species, so the colours are a look, not data.
Bushes: where the map shows 1.5 to 4 m vegetation they are real; under closed canopy the map cannot see the understory,
so sparse saplings there are a look too.

Writes out/hike/courses/<slug>/:
  canopy.npy   canopy height (m) on the course grid (1 m, rows south to north like elevation.npy)
  trees.npy    rows (x, y, crown_radius, height, kind, ground_z) in course metres; kind 0 broadleaf, 1 pine, 2 bush
  trees.json   source and counts
Drawing (draw_trees) adds them to a MuJoCo scene as visual-only boxes, in the same voxel style as the ski pines.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from hikesim.course import OUT

CHM = "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/{tile}.tif"
CHM_TILES = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/tiles.geojson"
MIN_TREE_M = 4.0        # canopy peaks lower than this are shrubs
TRAIL_CLEAR_M = 2.0     # trunks closer than this to the walk get nudged off it (crowns may still overhang)
BUSH_CLEAR_M = 1.0      # bushes keep this far from the walk

TRUNK = np.array([0.36, 0.24, 0.14, 1.0], np.float32)
# Three shades each, dark to sunlit. Early October in Ann Arbor: mostly green, turning in places.
LEAVES = {
  "green": np.array([[0.13, 0.32, 0.11, 1], [0.18, 0.41, 0.14, 1], [0.25, 0.50, 0.18, 1]], np.float32),
  "yellow": np.array([[0.60, 0.50, 0.10, 1], [0.76, 0.63, 0.14, 1], [0.87, 0.75, 0.22, 1]], np.float32),
  "orange": np.array([[0.60, 0.29, 0.08, 1], [0.78, 0.41, 0.12, 1], [0.89, 0.54, 0.18, 1]], np.float32),
  "red": np.array([[0.46, 0.12, 0.08, 1], [0.60, 0.17, 0.10, 1], [0.72, 0.25, 0.13, 1]], np.float32),
}
MIX = [("green", 0.58), ("yellow", 0.18), ("orange", 0.14), ("red", 0.10)]
NEEDLES = np.array([[0.06, 0.21, 0.12, 1], [0.09, 0.29, 0.15, 1], [0.13, 0.37, 0.18, 1]], np.float32)


def chm_tile(lat: float, lon: float) -> str:
  import urllib.request
  cache = OUT.parent / "chm_tiles.geojson"
  if not cache.exists():
    cache.write_bytes(urllib.request.urlopen(CHM_TILES, timeout=120).read())
  for f in json.loads(cache.read_text())["features"]:
    xs, ys = zip(*f["geometry"]["coordinates"][0])
    if min(xs) <= lon <= max(xs) and min(ys) <= lat <= max(ys):
      return f["properties"]["tile"]
  raise LookupError(f"no canopy tile covers {lat}, {lon}")


def canopy_on_grid(cdir: Path) -> np.ndarray:
  """Canopy height (m) resampled onto the course's 1 m grid, rows south to north."""
  import rasterio
  from rasterio.warp import Resampling, reproject, transform, transform_bounds
  from rasterio.windows import from_bounds
  from rasterio.transform import from_origin
  g = json.loads((cdir / "grid.json").read_text())
  ox, oy = g["utm_origin"]
  x0, y0 = ox + g["x0"] - g["cell"] / 2, oy + g["y0"] - g["cell"] / 2
  x1, y1 = x0 + g["ncol"] * g["cell"], y0 + g["nrow"] * g["cell"]
  lon, lat = transform("EPSG:26917", "EPSG:4326", [(x0 + x1) / 2], [(y0 + y1) / 2])
  with rasterio.open(CHM.format(tile=chm_tile(lat[0], lon[0]))) as src:
    b = transform_bounds("EPSG:26917", src.crs, x0 - 20, y0 - 20, x1 + 20, y1 + 20)
    win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
    a = src.read(1, window=win).astype(np.float32)
    st = src.window_transform(win); crs = src.crs
  out = np.zeros((g["nrow"], g["ncol"]), np.float32)
  reproject(a, out, src_transform=st, src_crs=crs, dst_transform=from_origin(x0, y1, g["cell"], g["cell"]),
            dst_crs="EPSG:26917", resampling=Resampling.bilinear)
  return out[::-1].copy()


def find_trees(canopy: np.ndarray, grid: dict, ground: np.ndarray, route_xy: np.ndarray) -> np.ndarray:
  c = ndimage.gaussian_filter(canopy, 1.0)
  peak = (c == ndimage.maximum_filter(c, size=5)) & (c >= MIN_TREE_M)
  r, k = np.nonzero(peak)
  h = c[r, k]
  order = np.argsort(-h)
  r, k, h = r[order], k[order], h[order]
  cell = grid["cell"]
  x, y = grid["x0"] + k * cell, grid["y0"] + r * cell
  # Greedy thinning, tallest first: a top is kept unless a taller kept tree's crown zone already covers it.
  taken = np.zeros(canopy.shape, bool)
  keep = []
  for i in range(len(h)):
    if taken[r[i], k[i]]:
      continue
    keep.append(i)
    s = int(np.ceil(max(2.5, 0.18 * h[i]) / cell))
    ra, rb, ka, kb = max(0, r[i] - s), min(canopy.shape[0], r[i] + s + 1), max(0, k[i] - s), min(canopy.shape[1], k[i] + s + 1)
    yy, xx = np.ogrid[ra - r[i]:rb - r[i], ka - k[i]:kb - k[i]]
    taken[ra:rb, ka:kb] |= yy * yy + xx * xx <= s * s
  x, y, h = x[keep], y[keep], h[keep]
  # The tops above are the dominant trees only; under a closed canopy the map shows one smooth surface. Fill the
  # canopy they do not cover with trees at the local measured height, about one per 25 m2 (a few hundred per ha).
  rng = np.random.default_rng(0)
  sp = 5
  gr, gk = np.mgrid[sp // 2:canopy.shape[0]:sp, sp // 2:canopy.shape[1]:sp]
  gr = np.clip(gr + rng.integers(-2, 3, gr.shape), 0, canopy.shape[0] - 1).ravel()
  gk = np.clip(gk + rng.integers(-2, 3, gk.shape), 0, canopy.shape[1] - 1).ravel()
  top3 = ndimage.maximum_filter(c, size=3)
  fx, fy, fh = [], [], []
  for i in rng.permutation(len(gr)):
    a, b = gr[i], gk[i]
    if c[a, b] < MIN_TREE_M or taken[a, b]:
      continue
    hh = float(top3[a, b]); fx.append(grid["x0"] + b * cell); fy.append(grid["y0"] + a * cell); fh.append(hh)
    s = int(np.ceil(max(2.5, 0.12 * hh) / cell))
    ra, rb, ka, kb = max(0, a - s), min(canopy.shape[0], a + s + 1), max(0, b - s), min(canopy.shape[1], b + s + 1)
    yy, xx = np.ogrid[ra - a:rb - a, ka - b:kb - b]
    taken[ra:rb, ka:kb] |= yy * yy + xx * xx <= s * s
  x, y, h = np.concatenate([x, fx]), np.concatenate([y, fy]), np.concatenate([h, fh])
  crown = np.clip(0.22 * h + 0.8, 1.5, 7.0)
  # Species: the canopy map is a smooth surface (crown shape does not separate pines here) and Ann Arbor's parks are
  # mostly broadleaf, so everything is broadleaf. kind = 1 (pine) is kept for an evergreen map later.
  kind = np.zeros_like(h)
  # Shrub layer: real where the canopy map shows 1.5 to 4 m vegetation, plus sparse saplings under the canopy.
  bx, by, bh = [], [], []
  for spc, lo, hi, p_keep in ((3, 1.5, MIN_TREE_M, 1.0), (6, MIN_TREE_M, 99.0, 0.35)):
    gr, gk = np.mgrid[spc // 2:canopy.shape[0]:spc, spc // 2:canopy.shape[1]:spc]
    gr = np.clip(gr + rng.integers(-1, 2, gr.shape), 0, canopy.shape[0] - 1).ravel()
    gk = np.clip(gk + rng.integers(-1, 2, gk.shape), 0, canopy.shape[1] - 1).ravel()
    m = (c[gr, gk] >= lo) & (c[gr, gk] < hi) & (rng.random(len(gr)) < p_keep)
    bx += list(grid["x0"] + gk[m] * cell); by += list(grid["y0"] + gr[m] * cell)
    bh += list(np.minimum(c[gr[m], gk[m]], 2.5) if lo < MIN_TREE_M else rng.uniform(0.8, 2.4, m.sum()))
  bh = np.asarray(bh, float)
  x, y, h = np.concatenate([x, bx]), np.concatenate([y, by]), np.concatenate([h, bh])
  crown = np.concatenate([crown, np.clip(0.45 * bh + 0.3, 0.5, 1.4)])
  kind = np.concatenate([kind, np.full(len(bh), 2.0)])
  # Trunks stand beside the trail, not on it.
  rt = cKDTree(route_xy)
  d, j = rt.query(np.stack([x, y], 1))
  clear = np.where(kind == 2, BUSH_CLEAR_M, TRAIL_CLEAR_M)
  near = d < clear
  if near.any():
    away = np.stack([x[near] - route_xy[j[near], 0], y[near] - route_xy[j[near], 1]], 1)
    n = np.linalg.norm(away, axis=1, keepdims=True)
    away = np.where(n > 1e-6, away / np.maximum(n, 1e-6), np.array([[1.0, 0.0]]))
    x[near] = route_xy[j[near], 0] + away[:, 0] * clear[near]
    y[near] = route_xy[j[near], 1] + away[:, 1] * clear[near]
  rr = np.clip((y - grid["y0"]) / cell, 0, ground.shape[0] - 1).astype(int)
  kk = np.clip((x - grid["x0"]) / cell, 0, ground.shape[1] - 1).astype(int)
  return np.stack([x, y, crown, h, kind, ground[rr, kk]], 1).astype(np.float32)


def build(slug: str) -> dict:
  cdir = OUT / slug
  grid = json.loads((cdir / "grid.json").read_text())
  route = json.loads((cdir / "route.json").read_text())
  ground = np.load(cdir / "elevation.npy")
  canopy = canopy_on_grid(cdir)
  np.save(cdir / "canopy.npy", canopy)
  trees = find_trees(canopy, grid, ground, np.stack([route["x"], route["y"]], 1))
  np.save(cdir / "trees.npy", trees)
  info = {"source": "Meta/WRI High Resolution Canopy Height Maps v1, 1 m (CC BY 4.0)", "trees": int((trees[:, 4] != 2).sum()),
          "bushes": int((trees[:, 4] == 2).sum()), "tallest_m": round(float(trees[:, 3].max()), 1) if len(trees) else 0,
          "canopy_share_over_5m": round(float((canopy > 5).mean()), 3)}
  (cdir / "trees.json").write_text(json.dumps(info, indent=1))
  return info


def course_trees(slug: str) -> np.ndarray | None:
  """The course's trees, building them on first use. None if the canopy map cannot be reached."""
  p = OUT / slug / "trees.npy"
  if not p.exists():
    try:
      build(slug)
    except Exception as e:
      print(f"  no trees for {slug}: {type(e).__name__}: {e}")
      return None
  return np.load(p)


def _yaw(x, y):
  return float((x * 12.9898 + y * 78.233) % (np.pi / 2))


@functools.lru_cache(maxsize=20000)
def tree_boxes(x: float, y: float, z: float, r: float, h: float, kind: int):
  """One 8-bit tree as boxes: (half sizes, centre, rgba). Stepped layers like pixels, lighter toward the top."""
  rng = np.random.default_rng(int(abs(x * 73.856 + y * 19.349) * 1000) % 2**31)
  boxes = []
  tw = float(np.clip(0.011 * h + 0.1, 0.12, 0.32))   # half width: 0.25 to 0.6 m trunks
  if kind == 2:  # bush: a wide low block with a smaller one on top
    pal = LEAVES["green"] if rng.random() < 0.75 else LEAVES[("yellow", "red")[int(rng.integers(0, 2))]]
    boxes.append(((r, r * float(rng.uniform(0.7, 1.0)), 0.3 * h), (x, y, z + 0.3 * h - 0.05), pal[int(rng.integers(0, 2))]))
    boxes.append(((0.6 * r, 0.6 * r, 0.22 * h), (x, y, z + 0.78 * h), pal[2]))
  elif kind == 1:  # pine: trunk, then stepped green layers narrowing upward, then a tip
    trunk_h = 0.2 * h; n = int(np.clip(round(h / 3.0), 3, 7)); t = (h - trunk_h - 0.08 * h) / n
    w0 = 1.1 * r; step = 0.72 * w0 / max(n - 1, 1)
    boxes.append(((tw, tw, trunk_h / 2 + 0.15), (x, y, z + trunk_h / 2 - 0.1), TRUNK))
    for k in range(n):
      g = NEEDLES[min(2, k * 3 // n)]
      boxes.append(((w0 - k * step, w0 - k * step, t / 2), (x, y, z + trunk_h + k * t + t / 2), g))
    tip = max(0.15, 0.3 * (w0 - (n - 1) * step))
    boxes.append(((tip, tip, 0.04 * h), (x, y, z + h - 0.04 * h), NEEDLES[2]))
  else:  # broadleaf: tall trunk, a blocky rounded crown in one colour family, and a plus-shaped top
    u = rng.random(); acc = 0.0; fam = "green"
    for name, p in MIX:
      acc += p
      if u <= acc: fam = name; break
    pal = LEAVES[fam]
    c0 = 0.36 * h; ct = h - c0; n = 4 if h > 12 else 3; t = ct / n
    widths = [0.78, 1.0, 0.92, 0.6][:n] if n == 4 else [0.85, 1.0, 0.65]
    boxes.append(((tw, tw, c0 / 2 + 0.3), (x, y, z + c0 / 2 - 0.15), TRUNK))
    for k, wf in enumerate(widths):
      w = wf * r
      boxes.append(((w, w * float(rng.uniform(0.85, 1.0)), t / 2), (x, y, z + c0 + k * t + t / 2), pal[min(2, k * 3 // n)]))
    top = 0.38 * r
    boxes.append(((top, 0.16 * r, 0.06 * h), (x, y, z + h - 0.03 * h), pal[2]))
    boxes.append(((0.16 * r, top, 0.06 * h), (x, y, z + h - 0.03 * h), pal[2]))
  c, s = np.cos(_yaw(x, y)), np.sin(_yaw(x, y))
  mat = np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
  return tuple((np.array(a, float), np.array(b, float), col) for a, b, col in boxes), mat


def draw_trees(scene, trees: np.ndarray, center, cam_pos=None, radius: float = 90.0, max_trees: int = 1500) -> int:
  """Adds the trees near `center` to a MuJoCo scene (after update_scene). Trees whose trunk would block the view from
  the camera to `center`, or that the camera would sit inside, are left out of that frame. Returns how many were drawn."""
  if trees is None or not len(trees):
    return 0
  cx, cy = float(center[0]), float(center[1])
  d = np.hypot(trees[:, 0] - cx, trees[:, 1] - cy)
  d = np.where(trees[:, 4] == 2, d * radius / 40.0, d)       # bushes only within 40 m
  near = np.argsort(d)[:max_trees]
  near = near[d[near] < radius]
  if cam_pos is not None:
    px, py, pz = (float(v) for v in cam_pos)
    vx, vy = cx - px, cy - py; L2 = vx * vx + vy * vy + 1e-9
    tx, ty = trees[near, 0], trees[near, 1]
    t = np.clip(((tx - px) * vx + (ty - py) * vy) / L2, 0, 1)
    seg = np.hypot(tx - (px + t * vx), ty - (py + t * vy))
    trunk = np.where(trees[near, 4] == 2, trees[near, 2], np.clip(0.011 * trees[near, 3] + 0.1, 0.12, 0.32))
    base = np.where(trees[near, 4] == 2, 0.0, 0.36 * trees[near, 3])
    low = trees[near, 5] + base < pz + 0.6      # crown (or bush) low enough to wrap the lens or hide the robot
    cam_d = np.hypot(tx - px, ty - py)
    blocks = np.where(trees[near, 4] == 2, (seg < trunk + 0.3) & (trees[near, 3] > 1.0), seg <= trunk + 0.5)
    ok = ~blocks & ~(low & (cam_d < trees[near, 2] + 0.5)) & ~((trees[near, 4] == 2) & (cam_d < 3.0))   # no bush in the lens
    near = near[ok]
  box = mujoco.mjtGeom.mjGEOM_BOX
  drawn = 0
  for i in near:
    x, y, r, h, kind, z = (float(v) for v in trees[i])
    geoms, mat = tree_boxes(round(x, 2), round(y, 2), round(z, 2), round(r, 2), round(h, 2), int(kind))
    if scene.ngeom + len(geoms) > scene.maxgeom:
      break
    for size, pos, rgba in geoms:
      g = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(g, box, size, pos, mat, rgba)
      g.specular, g.shininess = 0.06, 0.2
      scene.ngeom += 1
    drawn += 1
  return drawn


def camera_position(cam) -> np.ndarray:
  """World position of a MuJoCo free camera (lookat, distance, azimuth, elevation in degrees)."""
  az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
  fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
  return np.asarray(cam.lookat) - cam.distance * fwd


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("course", nargs="?")
  ap.add_argument("--all", action="store_true")
  a = ap.parse_args()
  slugs = sorted(p.name for p in OUT.iterdir() if (p / "route.json").exists()) if a.all else [a.course]
  for s in slugs:
    info = build(s)
    print(f"{s}: {info['trees']} trees and {info['bushes']} bushes, tallest {info['tallest_m']} m, "
          f"{info['canopy_share_over_5m'] * 100:.0f}% under canopy over 5 m")


if __name__ == "__main__":
  main()
