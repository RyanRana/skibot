"""Trees for terrain mosaics: tree lines along each tile's edges plus scattered trees inside the run.

Run: .venv/bin/python -m skisim.trees out/terrains_v3
Writes <mosaic>/trees.npy with rows (x, y, radius, height, tile). Radius is the solid trunk-and-lower-branches
radius the skier must clear; height is only for drawing. Sizes are at G1 scale (the robot is 0.73 of a person).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_trees(mosaic_dir: Path, seed: int = 0, half_width: float = 25.0, length: float = 160.0):
  meta = json.loads((mosaic_dir / "mosaic.json").read_text())
  tiles = meta["tiles"]
  rng = np.random.default_rng(seed)
  rows = []
  n = len(tiles)
  for k, t in enumerate(tiles):
    yc = t["y_center_m"]
    level = k / max(n - 1, 1)
    # Tree lines along both edges of the run, like a groomer cut through the forest.
    for side in (-1, 1):
      x = rng.uniform(0, 6)
      while x < length:
        y = yc + side * rng.uniform(17.5, 24.0)
        rows.append((x, y, rng.uniform(0.5, 0.9), rng.uniform(3.5, 7.0), k))
        x += rng.uniform(3.0, 7.0)
    # Trees inside the run: more of them on harder tiles, none near the start, some in small clusters.
    for _ in range(int(2 + 10 * level + rng.integers(0, 4))):
      cx, cy = rng.uniform(22, length - 12), yc + rng.uniform(-14, 14)
      for _ in range(1 if rng.random() < 0.7 else int(rng.integers(2, 4))):
        rows.append((cx + rng.normal(0, 1.5), cy + rng.normal(0, 1.5), rng.uniform(0.4, 0.8), rng.uniform(2.5, 5.5), k))
  return np.array(rows, dtype=np.float32)


def course_trees(course_line: np.ndarray, seed: int = 0, in_run: int = 14):
  """Trees for a real course: lines 16 to 24 m either side of the centerline, plus a few inside the run."""
  rng = np.random.default_rng(seed)
  rows = []
  s = course_line[:, 3]
  def at(sv):
    i = int(np.clip(np.searchsorted(s, sv), 1, len(s) - 1))
    p0, p1 = course_line[i - 1, :2], course_line[i, :2]
    d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
    return np.array([np.interp(sv, s, course_line[:, 0]), np.interp(sv, s, course_line[:, 1])]), np.array([-d[1], d[0]])
  sv = 0.0
  while sv < s[-1]:
    c, nrm = at(sv)
    for side in (-1, 1):
      p = c + side * rng.uniform(16, 24) * nrm
      rows.append((p[0], p[1], rng.uniform(0.5, 0.9), rng.uniform(3.5, 7.0), 0))
    sv += rng.uniform(3.0, 6.0)
  for sv in rng.uniform(40, s[-1] - 20, in_run):
    c, nrm = at(sv)
    p = c + rng.uniform(-9, 9) * nrm
    rows.append((p[0], p[1], rng.uniform(0.4, 0.8), rng.uniform(2.5, 5.5), 0))
  return np.array(rows, dtype=np.float32)


def main():
  mosaic = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "out/terrains_v3")
  trees = make_trees(mosaic)
  np.save(mosaic / "trees.npy", trees)
  per = np.bincount(trees[:, 4].astype(int))
  print(f"{len(trees)} trees on {len(per)} tiles: {per.min()} to {per.max()} per tile -> {mosaic / 'trees.npy'}")


if __name__ == "__main__":
  main()
