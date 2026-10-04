"""Mirror a training mosaic: every tile followed by its left-right mirror image, trees included.

Run: .venv/bin/python -m skisim.mirror_mosaic out/terrains_v3 out/terrains_v4
The real runs behind the tiles lean one way more than the other (45 tiles fall to the right, 32 to the left), and
the policy picked up the bias: it rode its left edges 60 to 98% of the time on real courses and failed right turns
twice as often as left ones. With each tile and its mirror next to each other, both directions get the same practice
at every curriculum level (level k of the old mosaic is level 2k here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from skisim.perturb import SNOW


def main():
  src, dst = Path(sys.argv[1]), Path(sys.argv[2])
  dst.mkdir(parents=True, exist_ok=True)
  meta = json.loads((src / "mosaic.json").read_text())
  Z = np.load(src / "mosaic_elevation.npy")
  trees = np.load(src / "trees.npy") if (src / "trees.npy").exists() else np.zeros((0, 5), np.float32)
  cell, tiles = meta["cell"], meta["tiles"]
  n = len(tiles)
  gap = int(round(8.0 / cell))
  ny = (Z.shape[0] - (n - 1) * gap) // n
  half = ny // 2
  rows, info, out_trees = [], [], []
  y = 0
  for k, t in enumerate(tiles):
    r0 = int(round(t["y_center_m"] / cell)) - half
    band = Z[r0 : r0 + ny]
    tt = trees[trees[:, 4] == k] if len(trees) else trees
    for m, z in enumerate((band, band[::-1])):
      j = 2 * k + m
      if j:
        prev = rows[-1][-1]
        w = np.linspace(0, 1, gap + 2)[1:-1, None]
        rows.append(prev[None, :] * (1 - w) + z[0][None, :] * w)
        y += gap
      rows.append(z)
      yc_new = (y + half) * cell
      info.append({**t, "index": j, "y_center_m": yc_new, "mirrored": bool(m), "snow_params": t.get("snow_params", SNOW[t.get("snow", "groomed")])})
      if len(tt):
        d = tt[:, 1] - t["y_center_m"]
        moved = tt.copy()
        moved[:, 1] = yc_new + (-d if m else d)
        moved[:, 4] = j
        out_trees.append(moved)
      y += ny
  Zm = np.concatenate(rows, axis=0).astype(np.float32)
  np.save(dst / "mosaic_elevation.npy", Zm)
  out = {k: v for k, v in meta.items() if k != "tiles"}
  out.update({"nrow": Zm.shape[0], "ncol": Zm.shape[1], "tiles": info, "mirrored_from": str(src)})
  (dst / "mosaic.json").write_text(json.dumps(out))
  if out_trees:
    np.save(dst / "trees.npy", np.concatenate(out_trees).astype(np.float32))
  print(f"{n} tiles -> {len(info)} (each followed by its mirror), grid {Zm.shape}, trees {sum(len(a) for a in out_trees)} -> {dst}")


if __name__ == "__main__":
  main()
