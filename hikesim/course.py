"""A real trail as a MuJoCo hiking course on the 1 m USGS lidar around it: the longest walk through one park's OSM
trail network, or the trail route between two points anywhere ground.py has data.

    .venv/bin/python -m hikesim.course "Bird Hills Nature Area"
    .venv/bin/python -m hikesim.course --all            # the default Ann Arbor course list
    .venv/bin/python -m hikesim.course --ground zion-angels --zone 12 --from 37.2707,-112.9533 --to 37.2694,-112.9478 \
        --name "Angels Landing"                          # Walter's Wiggles to the summit

Writes out/hike/courses/<slug>/:
  elevation.npy, grid.json   HeightGrid (same format as ski courses, loads with skisim.terrain.from_course)
  route.json                 the walk at 1 m spacing in local metres: x, y, z, grade, plus UTM origin and trail names
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from hikesim.geo import Dem, bridges_from_trails, fill_nan, utm

ROOT = Path(__file__).resolve().parents[1]
TRAILS = ROOT / "data" / "ground" / "annarbor" / "trails.geojson"   # ground.py --place "Ann Arbor, Michigan"
DEM = ROOT / "data" / "cache" / "3dep" / "utm17n"
OUT = ROOT / "out" / "hike" / "courses"
COURSES = ["Bird Hills Nature Area", "Nichols Arboretum", "Bluffs Nature Area", "Kuebler Langford Nature Area",
           "Barton Nature Area", "Nelson Meade County Farm Park"]


def slug(name: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def trail_graph(feats: list[dict], zone: int = 17):
  """Undirected graph of trail vertices (snapped to 0.5 m) with edge lengths, plus the way name on each edge."""
  key = lambda p: (round(p[0] * 2), round(p[1] * 2))
  pos, adj = {}, defaultdict(dict)
  for f in feats:
    xy = [utm(la, lo, zone) for lo, la in f["geometry"]["coordinates"]]
    name = f["properties"].get("name") or ""
    for a, b in zip(xy, xy[1:]):
      ka, kb = key(a), key(b)
      if ka == kb:
        continue
      pos[ka], pos[kb] = a, b
      d = math.dist(a, b)
      adj[ka][kb] = adj[kb][ka] = (d, name)
  return pos, adj


def dijkstra(adj, src):
  dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
  while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]:
      continue
    for v, (w, _) in adj[u].items():
      nd = d + w
      if nd < dist.get(v, math.inf):
        dist[v], prev[v] = nd, u
        heapq.heappush(pq, (nd, v))
  return dist, prev


def longest_walk(adj) -> list:
  """Approximate graph diameter (double sweep) of the largest connected piece: the longest walk with no detours."""
  seen, best = set(), []
  for s in adj:
    if s in seen:
      continue
    dist, _ = dijkstra(adj, s)
    seen |= dist.keys()
    if len(dist) > len(best):
      best = list(dist)
  dist, _ = dijkstra(adj, best[0])
  a = max(dist, key=dist.get)
  dist, prev = dijkstra(adj, a)
  b = max(dist, key=dist.get)
  path = [b]
  while path[-1] != a:
    path.append(prev[path[-1]])
  return path[::-1]


def resample(xy: np.ndarray, step: float = 1.0) -> np.ndarray:
  seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
  s = np.concatenate([[0], np.cumsum(seg)])
  t = np.arange(0, s[-1], step)
  return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], 1)


def route_between(pos, adj, a_xy, b_xy) -> list:
  """Shortest trail path between the graph vertices nearest two points."""
  keys = list(pos)
  P = np.array([pos[k] for k in keys])
  a = keys[int(np.argmin(np.linalg.norm(P - a_xy, axis=1)))]
  b = keys[int(np.argmin(np.linalg.norm(P - b_xy, axis=1)))]
  dist, prev = dijkstra(adj, a)
  if b not in dist:
    raise SystemExit("the two points are not connected by trails")
  path = [b]
  while path[-1] != a:
    path.append(prev[path[-1]])
  return path[::-1]


def build(area: str | None, max_len: float = 2000.0, margin: float = 25.0, trails: Path = TRAILS, dem: Path = DEM,
          out_root: Path = OUT, zone: int = 17, ends=None, name: str | None = None) -> Path:
  allf = json.loads(Path(trails).read_text())["features"]
  if ends:
    feats = allf
    pos, adj = trail_graph(feats, zone)
    path = route_between(pos, adj, np.array(utm(*ends[0], zone)), np.array(utm(*ends[1], zone)))
    area = name or "route"
  else:
    feats = [f for f in allf if (f["properties"].get("area") or {}).get("name") == area]
    if not feats:
      raise SystemExit(f"no trails in {area!r}")
    pos, adj = trail_graph(feats, zone)
    path = longest_walk(adj)
  names = []
  for u, v in zip(path, path[1:]):
    n = adj[u][v][1]
    if n and (not names or names[-1] != n):
      names.append(n)
  xy = resample(np.array([pos[k] for k in path]))[: int(max_len) + 1]

  x0, y0 = math.floor(xy[:, 0].min() - margin), math.floor(xy[:, 1].min() - margin)
  nx, ny = int(math.ceil(xy[:, 0].max() + margin)) - x0, int(math.ceil(xy[:, 1].max() + margin)) - y0
  raw = Dem(dem, bridges=bridges_from_trails(trails, zone=zone)).window(x0, y0, nx, ny)
  holes = float(np.isnan(raw).mean())
  z = fill_nan(raw).astype(np.float64)
  local = xy - (x0, y0)
  c = np.clip(local - 0.5, 0, [nx - 1.001, ny - 1.001])
  ci, ri = c[:, 0].astype(int), c[:, 1].astype(int)
  fx, fy = c[:, 0] - ci, c[:, 1] - ri
  zr = (z[ri, ci] * (1 - fx) * (1 - fy) + z[ri, ci + 1] * fx * (1 - fy)
        + z[ri + 1, ci] * (1 - fx) * fy + z[ri + 1, ci + 1] * fx * fy)
  win = 5  # metres of smoothing, edge-padded so the ends don't read as cliffs
  zs = np.convolve(np.pad(zr, win // 2, mode="edge"), np.ones(win) / win, mode="valid")
  grade = np.degrees(np.arctan(np.gradient(zs)))

  out = out_root / slug(name or area)
  out.mkdir(parents=True, exist_ok=True)
  np.save(out / "elevation.npy", z.astype(np.float32))
  (out / "grid.json").write_text(json.dumps({"x0": 0.5, "y0": 0.5, "cell": 1.0, "nrow": ny, "ncol": nx,
                                             "utm_origin": [x0, y0], "crs": f"EPSG:{26900 + zone}"}))
  route = {"area": area, "trails": names, "utm_origin": [x0, y0], "length_m": round(len(xy) - 1.0, 1),
           "climb_m": round(float(np.clip(np.diff(zr), 0, None).sum()), 1),
           "vertical_m": round(float(zr.max() - zr.min()), 1),
           "max_grade_deg": round(float(np.abs(grade).max()), 1),
           "share_steeper_than_10deg": round(float(np.mean(np.abs(grade) > 10)), 3),
           "lidar_holes": round(holes, 4),
           "x": np.round(local[:, 0], 2).tolist(), "y": np.round(local[:, 1], 2).tolist(),
           "z": np.round(zr, 2).tolist(), "grade_deg": np.round(grade, 1).tolist()}
  (out / "route.json").write_text(json.dumps(route))
  print(f"{area}: {route['length_m']:.0f} m walk, climb {route['climb_m']:.0f} m, vertical {route['vertical_m']:.0f} m, "
        f"max grade {route['max_grade_deg']:.0f} deg, {route['share_steeper_than_10deg']:.0%} over 10 deg, "
        f"grid {nx} x {ny} m -> {out.relative_to(ROOT)}")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("area", nargs="?")
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--max-len", type=float, default=2000.0)
  ap.add_argument("--trails", type=Path, default=TRAILS)
  ap.add_argument("--dem", type=Path, default=DEM)
  ap.add_argument("--ground", help="a ground.py dataset name (data/ground/<name>/trails.geojson)")
  ap.add_argument("--zone", type=int, default=17, help="UTM zone of the lidar tiles")
  ap.add_argument("--from", dest="src", help="lat,lon: route from the trail point nearest this")
  ap.add_argument("--to", dest="dst", help="lat,lon: to the trail point nearest this")
  ap.add_argument("--name", help="course name for --from/--to")
  a = ap.parse_args()
  trails = ROOT / "data" / "ground" / a.ground / "trails.geojson" if a.ground else a.trails
  dem = ROOT / "data" / "cache" / "3dep" / f"utm{a.zone}n" if a.ground else a.dem
  if a.src and a.dst:
    ends = [tuple(map(float, a.src.split(","))), tuple(map(float, a.dst.split(",")))]
    build(None, a.max_len, trails=trails, dem=dem, zone=a.zone, ends=ends, name=a.name)
    return
  for area in (COURSES if a.all or not a.area else [a.area]):
    build(area, a.max_len, trails=trails, dem=dem, zone=a.zone)


if __name__ == "__main__":
  main()
