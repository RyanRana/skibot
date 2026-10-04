"""A built course (out/courses/<slug>) -> game/public/courses/<slug>/ for the browser.

Run: .venv/bin/python tools/export_course.py kitzbuhel-streif [--cell 2]
Writes heights.bin (little-endian float32 row-major, row 0 = south), course.json (grid origin, size,
centerline, gates, start pose, stats, attribution).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
slug = sys.argv[1] if len(sys.argv) > 1 else "kitzbuhel-streif"
src = ROOT / "out/courses" / slug
dst = ROOT / "game/public/courses" / slug
dst.mkdir(parents=True, exist_ok=True)

grid = np.load(src / "elevation.npy").astype("<f4")
gm = json.loads((src / "grid.json").read_text())
run = json.loads((src / "run.json").read_text())
gates = json.loads((src / "gates.json").read_text())
start = json.loads((src / "start_pose.json").read_text())
stats = json.loads((src / "stats.json").read_text())

(dst / "heights.bin").write_bytes(grid.tobytes())
course = {
  "slug": slug,
  "resort": stats["resort_query"],
  "run": stats["run"]["osm_names"][0] if stats["run"]["osm_names"] else stats["run"]["name"],
  "difficulty": stats["run"]["difficulty"],
  "lat0": gm["lat0"], "lon0": gm["lon0"], "z_datum_msl": gm["z_datum_msl"],
  "x0": gm["x0"], "y0": gm["y0"], "cell": gm["cell"], "nrow": gm["nrow"], "ncol": gm["ncol"],
  "z_min": gm["z_min"], "z_max": gm["z_max"],
  "centerline": [[round(p[0], 2), round(p[1], 2), round(p[2], 2), round(p[3], 1)] for p in run["course"]],
  "gates": {**gates, "gates": [g for g in gates["gates"] if g["s"] >= 30]},
  "start": start,
  "stats": {"length_m": stats["course"]["length_2d_m"], "drop_m": stats["course"]["vertical_drop_m"],
            "mean_slope_deg": stats["course"]["mean_slope_deg"], "max_slope_deg": stats["course"]["max_slope_10m_deg"],
            "start_elevation_msl_m": stats["course"]["start_elevation_msl_m"], "gates": stats["course"]["gates"]},
  "sources": stats["sources"],
}
(dst / "course.json").write_text(json.dumps(course))
print(f"{slug}: {grid.shape} grid, {len(course['centerline'])} centerline pts, {len(gates.get('gates', []))} gates -> {dst}")
print("gates.json keys:", list(gates.keys()), "first gate:", json.dumps(gates["gates"][0])[:400] if gates.get("gates") else None)
print("start:", json.dumps(start)[:300])
