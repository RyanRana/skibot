"""Training terrain from every Ann Arbor trail: straight-ish 40 m stretches cut out of the 1 m lidar, turned so the
trail runs along +x, in both walking directions, plus steepened copies for a curriculum. Packed side by side in y
into one mosaic, the same layout the ski env uses (mosaic.json + mosaic_elevation.npy).

    .venv/bin/python -m hikesim.tiles --n 192 --out out/hike/tiles
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from hikesim.course import DEM, TRAILS
from hikesim.geo import Dem, fill_nan, utm17
from hikesim.rough import bumps

ROOT = Path(__file__).resolve().parents[1]
CHORD = 40.0      # metres of trail per tile
LENGTH = 48.0     # tile length along x (2 m lead-in, 6 m run-out)
WIDTH = 12.0      # tile width in y
CELL = 0.25       # lidar resampled to 0.25 m (bilinear), fine enough for the roots and rocks added on top
LEAD = 2.0


def stretches(trails: Path, min_len: float = CHORD) -> list[dict]:
  """Every CHORD-metre stretch of trail that stays within 2 m of its straight chord."""
  out = []
  for f in json.loads(Path(trails).read_text())["features"]:
    p = f["properties"]
    if (p.get("statistics") or {}).get("length_m", 0) < min_len:
      continue
    xy = np.array([utm17(la, lo) for lo, la in f["geometry"]["coordinates"]])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], 1.0)
    pts = np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], 1)
    for a in range(0, len(pts) - int(CHORD), 20):
      b = a + int(CHORD)
      d = pts[b] - pts[a]
      L = np.linalg.norm(d)
      if L < CHORD * 0.9:
        continue
      u = d / L
      off = np.abs((pts[a:b + 1] - pts[a]) @ np.array([-u[1], u[0]]))
      if off.max() > 2.0:
        continue
      out.append({"id": p["id"], "name": p.get("name"), "area": (p.get("area") or {}).get("name"),
                  "surface": p.get("surface"), "start": pts[a].tolist(), "dir": u.tolist()})
  return out


def cut(dem: Dem, st: dict) -> np.ndarray:
  """(rows, cols) heights in the trail frame: x along the stretch from LEAD m behind its start, y across it."""
  x0, y0 = st["start"]
  ux, uy = st["dir"]
  xs = np.arange(0, LENGTH + 1e-6, CELL) - LEAD
  ys = np.arange(-WIDTH / 2, WIDTH / 2 + 1e-6, CELL)
  X, Y = np.meshgrid(xs, ys)
  wx = x0 + X * ux - Y * uy
  wy = y0 + X * uy + Y * ux
  bx, by = math.floor(wx.min()) - 2, math.floor(wy.min()) - 2
  nx, ny = int(math.ceil(wx.max())) - bx + 3, int(math.ceil(wy.max())) - by + 3
  win = fill_nan(dem.window(bx, by, nx, ny))
  # window cell centres sit at bx + c + 0.5, by + r + 0.5
  return ndimage.map_coordinates(win, [wy - by - 0.5, wx - bx - 0.5], order=1, mode="nearest")


def corridor_slopes(zt: np.ndarray, half: float = 4.5) -> np.ndarray:
  """Cell slopes (deg) in the strip a robot can walk before it counts as off the trail."""
  yc = zt.shape[0] // 2
  k = int(half / CELL)
  sub = zt[yc - k:yc + k + 1]
  gy, gx = np.gradient(sub, CELL)
  return np.degrees(np.arctan(np.hypot(gx, gy)))


def build(n: int, trails: Path, dem_dir: Path, out: Path, seed: int = 0, scales=(1.0, 1.25, 1.5),
          max_cell_deg: float = 45.0, max_share_over_35: float = 0.05, bump_range=(0.3, 1.5)) -> None:
  rng = np.random.default_rng(seed)
  sts = stretches(trails)
  print(f"{len(sts)} straight {CHORD:.0f} m stretches on the trails")
  dem = Dem(dem_dir)
  cand = []
  for st in sts:
    z = cut(dem, st)
    mid = z.shape[0] // 2
    line = z[mid]
    i0, i1 = int(LEAD / CELL), int((LEAD + CHORD) / CELL)
    grade = math.degrees(math.atan((line[i1] - line[i0]) / CHORD))
    steep = float(np.degrees(np.arctan(np.abs(np.diff(ndimage.uniform_filter1d(line, 10))) / CELL)).max())
    rough = float(np.std(line[i0:i1] - np.linspace(line[i0], line[i1], i1 - i0)))
    cand.append((st, z, grade, steep, rough))
  # Keep the most varied set: all the steep and rough ones, then fill with a random sample of the rest.
  order = sorted(range(len(cand)), key=lambda k: -(abs(cand[k][2]) + cand[k][3] * 0.5 + cand[k][4] * 5))
  base_n = max(1, int(n * 1.6) // (2 * len(scales)))  # extra candidates: the slope filter drops some
  pick = order[: base_n // 2] + list(rng.choice(order[base_n // 2:], size=min(base_n - base_n // 2, len(order) - base_n // 2), replace=False))
  tiles, zs, rejected = [], [], 0
  for k in pick:
    st, z, grade, steep, rough = cand[k]
    for direction in (1, -1):
      zz = z if direction == 1 else z[::-1, ::-1]
      if direction == -1:  # walking the other way: the lead-in is now at the far end, so shift it back
        shift = int((LENGTH - CHORD - 2 * LEAD) / CELL)
        zz = np.concatenate([zz[:, shift:], np.repeat(zz[:, -1:], shift, 1)], 1)
      for s in scales:
        # Steepen: scale the along-trail trend, keep the small-scale lidar texture as it is.
        trend = ndimage.uniform_filter(zz, size=int(8 / CELL), mode="nearest")
        zt = trend * s + (zz - trend)
        zt = zt - zt[zt.shape[0] // 2, int(LEAD / CELL)]
        sl = corridor_slopes(zt)
        # Steepened hillside trails turn into 50-60 deg side slopes nobody walks, and MuJoCo Warp explodes on them.
        if sl.max() > max_cell_deg or np.mean(sl > 35) > max_share_over_35:
          rejected += 1
          continue
        line = zt[zt.shape[0] // 2]
        i0, i1 = int(LEAD / CELL), int((LEAD + CHORD) / CELL)
        g = math.degrees(math.atan((line[i1] - line[i0]) / CHORD))
        st10 = float(np.degrees(np.arctan(np.abs(np.diff(ndimage.uniform_filter1d(line, 20))) / CELL)).max())
        # Trail micro-relief the lidar cannot see (hikesim.rough), at a random strength per tile.
        amp = float(rng.uniform(*bump_range))
        X, Y = np.meshgrid(np.arange(zt.shape[1]) * CELL - LEAD, np.arange(zt.shape[0]) * CELL - WIDTH / 2)
        off = rng.uniform(0, 1e4, 2)
        zt = zt + bumps(X + off[0], Y + off[1], seed=int(rng.integers(1 << 20)), amp=amp)
        tiles.append({"trail": st["id"], "name": st["name"], "area": st["area"], "direction": direction, "scale": s,
                      "grade_deg": round(g, 1), "steepest_deg": round(st10, 1), "rough_m": round(rough * s, 3),
                      "bumps": round(amp, 2)})
        zs.append(zt)
  # Difficulty: climbing is harder than descending at the same grade for SONIC, steep spots and roughness add on.
  diff = [abs(t["grade_deg"]) * (1.2 if t["grade_deg"] > 0 else 1.0) + 0.5 * t["steepest_deg"] + 10 * t["rough_m"]
          + 6 * t.get("bumps", 0) for t in tiles]
  order = np.argsort(diff)
  tiles = [tiles[i] for i in order]
  zs = [zs[i] for i in order]
  print(f"rejected {rejected} too-steep variants")
  rows = zs[0].shape[0]
  gap = 2  # rows of flat seam between tiles
  mosaic = np.zeros(((rows + gap) * len(zs), zs[0].shape[1]), np.float32)
  for i, z in enumerate(zs):
    r0 = i * (rows + gap)
    mosaic[r0:r0 + rows] = z
    mosaic[r0 + rows:r0 + rows + gap] = z[-1]
    tiles[i]["y_center_m"] = round((r0 + (rows - 1) / 2) * CELL, 3)
    tiles[i]["difficulty"] = round(float(diff[order[i]]), 2)
  out.mkdir(parents=True, exist_ok=True)
  np.save(out / "mosaic_elevation.npy", mosaic)
  meta = {"x0": -LEAD, "y0": 0.0, "cell": CELL, "tile_length_m": LENGTH, "tile_width_m": WIDTH, "chord_m": CHORD,
          "lead_m": LEAD, "tiles": tiles, "source": "OpenStreetMap trails (ODbL), USGS 3DEP 1 m lidar (public domain), "
          "plus hikesim.rough micro-relief"}
  (out / "mosaic.json").write_text(json.dumps(meta, indent=1))
  g = np.array([t["grade_deg"] for t in tiles])
  print(f"wrote {len(tiles)} tiles to {out}: grade {g.min():.0f} to {g.max():.0f} deg, mosaic {mosaic.shape[1]} x {mosaic.shape[0]} cells")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=192)
  ap.add_argument("--out", type=Path, default=ROOT / "out" / "hike" / "tiles")
  ap.add_argument("--trails", type=Path, default=TRAILS)
  ap.add_argument("--dem", type=Path, default=DEM)
  a = ap.parse_args()
  build(a.n, a.trails, a.dem, a.out)


if __name__ == "__main__":
  main()
