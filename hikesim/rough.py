"""Trail micro-relief the 1 m lidar cannot see: fractal roughness, roots and rocks, a few cm to ~15 cm tall.

Deterministic in absolute position (hashed value noise on fixed lattices), so the physics, the renders and any crop of
the same course get exactly the same bumps. Heights are added on a 0.25 m grid; `amp` scales everything (0 = smooth).
"""

from __future__ import annotations

import numpy as np

from skisim.terrain import HeightGrid

CELL = 0.25


def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
  """Uniform [0, 1) per integer lattice point."""
  h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (seed * 83492791)
  h = (h ^ (h >> 13)) * 1274126177
  h = h ^ (h >> 16)
  return (h & 0xFFFFFF).astype(np.float64) / float(1 << 24)


def _value_noise(x: np.ndarray, y: np.ndarray, spacing: float, seed: int) -> np.ndarray:
  """Smooth value noise in [-1, 1] with lattice spacing `spacing` metres."""
  fx, fy = x / spacing, y / spacing
  ix, iy = np.floor(fx), np.floor(fy)
  tx, ty = fx - ix, fy - iy
  tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
  v00, v10 = _hash(ix, iy, seed), _hash(ix + 1, iy, seed)
  v01, v11 = _hash(ix, iy + 1, seed), _hash(ix + 1, iy + 1, seed)
  v = v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty
  return 2 * v - 1


def bumps(x: np.ndarray, y: np.ndarray, seed: int = 0, amp: float = 1.0) -> np.ndarray:
  """Micro-relief height (m) at absolute positions x, y (any shape)."""
  if amp <= 0:
    return np.zeros(np.broadcast(x, y).shape)
  x, y = np.asarray(x, float), np.asarray(y, float)
  # Fractal roughness: ~2 cm RMS from 2 m down to 0.3 m wavelengths.
  h = (0.033 * _value_noise(x, y, 2.0, seed) + 0.018 * _value_noise(x, y, 0.9, seed + 1)
       + 0.010 * _value_noise(x, y, 0.45, seed + 2))
  # Roots: ridged noise (1 - |n|) sharpened into narrow crests, up to 8 cm, mostly where a second field says so.
  ridge = 1 - np.abs(_value_noise(x * 0.8 + y * 0.6, y * 0.8 - x * 0.6, 1.6, seed + 3))
  where = np.clip(_value_noise(x, y, 6.0, seed + 4) * 1.5 + 0.2, 0, 1)
  h += 0.08 * where * np.clip((ridge - 0.82) / 0.18, 0, 1) ** 2
  # Rocks: sparse rounded blobs, 6 to 15 cm, about one per 6 m^2.
  L = 2.5
  ix, iy = np.floor(x / L), np.floor(y / L)
  rock = np.zeros_like(h)
  for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
      cx, cy = ix + dx, iy + dy
      p = _hash(cx, cy, seed + 5)
      ox, oy = _hash(cx, cy, seed + 6) * L, _hash(cx, cy, seed + 7) * L
      rad = 0.15 + 0.2 * _hash(cx, cy, seed + 8)
      ht = 0.06 + 0.09 * _hash(cx, cy, seed + 9)
      d2 = ((x - cx * L - ox) ** 2 + (y - cy * L - oy) ** 2) / rad ** 2
      rock = np.maximum(rock, np.where(p > 0.6, ht * np.clip(1 - d2, 0, 1) ** 0.7, 0))
  return amp * (h + rock)


def roughen(grid: HeightGrid, origin=(0.0, 0.0), seed: int = 0, amp: float = 1.0, cell: float = CELL) -> HeightGrid:
  """Resample a grid to `cell` (bilinear) and add bumps. `origin` is added to local x, y before hashing (pass the
  course's UTM origin so every crop of a course sees the same rocks)."""
  from scipy import ndimage
  k = grid.cell / cell
  z = ndimage.zoom(grid.z, k, order=1)
  ny, nx = z.shape
  cx = (grid.ncol - 1) * grid.cell / (nx - 1)
  cy = (grid.nrow - 1) * grid.cell / (ny - 1)
  X, Y = np.meshgrid(grid.x0 + np.arange(nx) * cx, grid.y0 + np.arange(ny) * cy)
  z = z + bumps(X + origin[0], Y + origin[1], seed, amp)
  return HeightGrid(z=z, x0=grid.x0, y0=grid.y0, cell=cx)
