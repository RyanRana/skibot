"""The real mountains around a course, for the horizon: a coarse DEM in the course frame.

Run: .venv/bin/python tools/export_far.py kitzbuhel-streif [--half-km 10] [--cell 60] [--zoom 11]
Writes game/public/courses/<slug>/far.bin (little-endian float32 row-major, row 0 = south, z relative to
the course datum) and far.json (x0, y0, cell, nrow, ncol, z range, source).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from course import LocalFrame, Terrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("slug", nargs="?", default="kitzbuhel-streif")
ap.add_argument("--half-km", type=float, default=10.0)
ap.add_argument("--cell", type=float, default=60.0)
ap.add_argument("--zoom", type=int, default=11)
a = ap.parse_args()

dst = ROOT / "game/public/courses" / a.slug
meta = json.loads((dst / "course.json").read_text())
fr = LocalFrame(meta["lat0"], meta["lon0"])
terrain = Terrain(a.zoom)
half = a.half_km * 1000
n = int(round(2 * half / a.cell)) + 1
xs = -half + a.cell * np.arange(n)
xx, yy = np.meshgrid(xs, xs)
lat, lon = fr.to_latlon(xx, yy)
z = terrain.sample(lat, lon).astype(np.float64) - meta["z_datum_msl"]
z = z.astype("<f4")
(dst / "far.bin").write_bytes(z.tobytes())
(dst / "far.json").write_text(json.dumps({
  "x0": float(xs[0]), "y0": float(xs[0]), "cell": a.cell, "nrow": n, "ncol": n,
  "z_min": float(z.min()), "z_max": float(z.max()), "z_datum_msl": meta["z_datum_msl"],
  "source": f"AWS Terrain Tiles terrarium z{a.zoom}",
}))
print(f"far terrain {n}x{n} at {a.cell} m, z {z.min():.0f}..{z.max():.0f} m rel. datum ({meta['z_datum_msl'] + z.max():.0f} m MSL peak) -> {dst / 'far.bin'} ({(dst / 'far.bin').stat().st_size / 1e6:.1f} MB)")
