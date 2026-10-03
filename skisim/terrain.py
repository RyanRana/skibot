"""Height grids shared by the MuJoCo hfield (for rendering and body falls) and the ski contact.

Convention: row r is y = y0 + r * cell (north), column c is x = x0 + c * cell (east), z is up.
MuJoCo stores hfield data row-major with row 0 at -y, so the grid maps onto it without flips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeightGrid:
  z: np.ndarray  # (nrow, ncol) elevation in meters
  x0: float
  y0: float
  cell: float

  @property
  def nrow(self) -> int:
    return self.z.shape[0]

  @property
  def ncol(self) -> int:
    return self.z.shape[1]

  def bounds(self) -> tuple[float, float, float, float]:
    return (
      self.x0,
      self.x0 + (self.ncol - 1) * self.cell,
      self.y0,
      self.y0 + (self.nrow - 1) * self.cell,
    )

  def height_normal(self, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear height and its unit normal at world (x, y). Points off the grid clamp to the edge."""
    fx = (np.asarray(x, dtype=float) - self.x0) / self.cell
    fy = (np.asarray(y, dtype=float) - self.y0) / self.cell
    c = np.clip(np.floor(fx).astype(int), 0, self.ncol - 2)
    r = np.clip(np.floor(fy).astype(int), 0, self.nrow - 2)
    tx = np.clip(fx - c, 0.0, 1.0)
    ty = np.clip(fy - r, 0.0, 1.0)
    z00 = self.z[r, c]
    z01 = self.z[r, c + 1]
    z10 = self.z[r + 1, c]
    z11 = self.z[r + 1, c + 1]
    h = z00 * (1 - tx) * (1 - ty) + z01 * tx * (1 - ty) + z10 * (1 - tx) * ty + z11 * tx * ty
    dzdx = ((z01 - z00) * (1 - ty) + (z11 - z10) * ty) / self.cell
    dzdy = ((z10 - z00) * (1 - tx) + (z11 - z01) * tx) / self.cell
    n = np.stack([-dzdx, -dzdy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return h, n

  def hfield(self, base: float = 1.0) -> dict:
    """MuJoCo hfield size, geom position and normalized data for this grid."""
    zmin = float(self.z.min())
    zrange = max(float(self.z.max()) - zmin, 1e-3)
    rx = (self.ncol - 1) * self.cell / 2
    ry = (self.nrow - 1) * self.cell / 2
    return {
      "nrow": self.nrow,
      "ncol": self.ncol,
      "size": (rx, ry, zrange, base),
      "pos": (self.x0 + rx, self.y0 + ry, zmin),
      "data": ((self.z - zmin) / zrange).astype(np.float32),
    }


def slope(
  theta_deg: float,
  length: float = 300.0,
  width: float = 120.0,
  cell: float = 1.0,
  bump_amp: float = 0.0,
  bump_wavelength: float = 30.0,
  seed: int = 0,
) -> HeightGrid:
  """Planar slope falling toward +x, with optional smooth rolling bumps."""
  xs = np.arange(0.0, length + cell / 2, cell)
  ys = np.arange(-width / 2, width / 2 + cell / 2, cell)
  X, Y = np.meshgrid(xs, ys)
  z = -np.tan(np.radians(theta_deg)) * X
  if bump_amp > 0:
    rng = np.random.default_rng(seed)
    for _ in range(6):
      k = 2 * np.pi / (bump_wavelength * rng.uniform(0.6, 1.6))
      a = rng.uniform(0, 2 * np.pi)
      z += bump_amp / 6 * np.sin(k * (np.cos(a) * X + np.sin(a) * Y) + rng.uniform(0, 2 * np.pi))
  return HeightGrid(z=z, x0=float(xs[0]), y0=float(ys[0]), cell=cell)


def from_course(course_dir) -> HeightGrid:
  """Load a real run built by the resort pipeline (out/courses/<slug>: elevation.npy + grid.json)."""
  import json
  from pathlib import Path

  course_dir = Path(course_dir)
  meta = json.loads((course_dir / "grid.json").read_text())
  z = np.load(course_dir / "elevation.npy").astype(float)
  return HeightGrid(z=z, x0=meta["x0"], y0=meta["y0"], cell=meta["cell"])
