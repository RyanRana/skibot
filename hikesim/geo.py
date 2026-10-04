"""UTM 17N and the 1 m USGS 3DEP lidar tiles cached by ground.py (data/cache/3dep/utm17n).

Tiles are 2 km squares named by their south-west corner (<x>_<y>.tif, or dem_1m_utm17n_<x>_<y>.tif), row 0 north,
pixel (r, c) centred at x = x_sw + c + 0.5, y = y_sw + 2000 - r - 0.5.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

TILE_M = 2000


def utm17(lat: float, lon: float) -> tuple[float, float]:
  """WGS84 to UTM zone 17N (Ann Arbor), same as ground.py."""
  return utm(lat, lon, 17)


def utm(lat: float, lon: float, zone: int) -> tuple[float, float]:
  """WGS84 to UTM zone `zone` north (Krueger series)."""
  a = 6378137.0; f = 1 / 298.257222101; e2 = f * (2 - f); k0 = 0.9996; lon0 = math.radians(zone * 6 - 183)
  phi = math.radians(lat); lam = math.radians(lon)
  N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2); T = math.tan(phi) ** 2; ep2 = e2 / (1 - e2)
  C = ep2 * math.cos(phi) ** 2; A = math.cos(phi) * (lam - lon0)
  M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
           - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
           + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi) - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
  x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
  y = k0 * (M + N * math.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                     + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
  return x, y


def bridges_from_trails(trails: Path, step: float = 0.5, zone: int = 17) -> list[tuple[np.ndarray, np.ndarray]]:
  """Bridge and boardwalk decks from ground.py's trails: UTM polyline every `step` m with the deck height, taken from
  the way's elevation profile (ground.py draws those straight between the ends, since bare earth lidar sees the creek)."""
  import json
  out = []
  for f in json.loads(Path(trails).read_text())["features"]:
    p = f["properties"]
    if p.get("bridge") in (None, "", "no") and p.get("elevation_method") != "interpolated":
      continue
    prof = p.get("elevationProfile") or {}
    hs = prof.get("heights")
    if not hs:
      continue
    xy = np.array([utm(la, lo, zone) for lo, la in f["geometry"]["coordinates"]])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1] + 1e-6, step)
    pts = np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], 1)
    sp = np.arange(len(hs)) * float(prof.get("resolution_m", 5.0))
    z = np.interp(t, sp, np.array(hs, float))
    out.append((pts, z))
  return out


def burn_bridges(z: np.ndarray, xi0: int, yi0: int, bridges, half_width: float = 1.2) -> np.ndarray:
  """Raise cells within half_width of a deck to the deck height (cell centres at xi0 + c + 0.5, yi0 + r + 0.5)."""
  ny, nx = z.shape
  for pts, dz in bridges:
    if pts[:, 0].max() < xi0 - 2 or pts[:, 0].min() > xi0 + nx + 2 or pts[:, 1].max() < yi0 - 2 or pts[:, 1].min() > yi0 + ny + 2:
      continue
    k = int(np.ceil(half_width))
    for (x, y), h in zip(pts, dz):
      c, r = int(np.floor(x - xi0)), int(np.floor(y - yi0))
      for rr in range(max(0, r - k), min(ny, r + k + 1)):
        for cc in range(max(0, c - k), min(nx, c + k + 1)):
          if (xi0 + cc + 0.5 - x) ** 2 + (yi0 + rr + 0.5 - y) ** 2 <= half_width ** 2:
            z[rr, cc] = max(z[rr, cc], h) if np.isfinite(z[rr, cc]) else h
  return z


class Dem:
  """Reads any UTM window out of the cached 2 km tiles, north-up arrays flipped to the HeightGrid convention.
  With bridges (bridges_from_trails), their decks are burned in, so a trail crosses a creek on the deck, not the bed."""

  def __init__(self, dem_dir: Path, bridges=None):
    self.bridges = bridges or []
    self.dir = Path(dem_dir)
    self.tiles = {}
    for p in self.dir.glob("*.tif"):
      m = re.fullmatch(r"(?:dem_1m_utm17n_)?(\d+)_(\d+)\.tif", p.name)
      if m:
        self.tiles[(int(m[1]), int(m[2]))] = p
    self._cache: dict[tuple[int, int], np.ndarray] = {}

  def _tile(self, xs: int, ys: int) -> np.ndarray | None:
    if (xs, ys) not in self._cache:
      p = self.tiles.get((xs, ys))
      z = None
      if p is not None:
        z = np.array(Image.open(p), dtype=np.float32)
        z[z < -1000] = np.nan
        if z.shape != (TILE_M, TILE_M):  # 10 m 3DEP where no lidar was flown
          from scipy import ndimage
          z = ndimage.zoom(z, (TILE_M / z.shape[0], TILE_M / z.shape[1]), order=1)
      self._cache[(xs, ys)] = z
    return self._cache[(xs, ys)]

  def window(self, x0: float, y0: float, nx: int, ny: int) -> np.ndarray:
    """(ny, nx) elevations, row 0 south, cell centres at floor(x0) + c + 0.5, floor(y0) + r + 0.5. NaN where no lidar."""
    xi0, yi0 = int(math.floor(x0)), int(math.floor(y0))
    out = np.full((ny, nx), np.nan, np.float32)
    for xs in range(xi0 // TILE_M * TILE_M, xi0 + nx, TILE_M):
      for ys in range(yi0 // TILE_M * TILE_M, yi0 + ny, TILE_M):
        z = self._tile(xs, ys)
        if z is None:
          continue
        # overlap in integer metres: [ax, bx) x [ay, by)
        ax, bx = max(xi0, xs), min(xi0 + nx, xs + TILE_M)
        ay, by = max(yi0, ys), min(yi0 + ny, ys + TILE_M)
        if ax >= bx or ay >= by:
          continue
        rows = TILE_M - 1 - (np.arange(ay, by) - ys)          # north-up tile rows for these y
        cols = np.arange(ax, bx) - xs
        out[ay - yi0:by - yi0, ax - xi0:bx - xi0] = z[np.ix_(rows, cols)]
    if self.bridges:
      out = burn_bridges(out, xi0, yi0, self.bridges)
    return out


def fill_nan(z: np.ndarray) -> np.ndarray:
  """Fill lidar holes (water, buildings removed) with the nearest valid height."""
  if not np.isnan(z).any():
    return z
  from scipy import ndimage
  idx = ndimage.distance_transform_edt(np.isnan(z), return_distances=False, return_indices=True)
  return z[tuple(idx)]
