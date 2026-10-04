"""Pipeline status for the viewer: every park in the ground.py dataset, what sets its trails apart, and how far each
one has got through the stages.

    .venv/bin/python -m hikesim.pipeline        # -> out/hike/web/pipeline.json (+ pipeline.js), then reload :8770

Stages, in order:
  data    ground.py: OpenStreetMap trails with 1 m lidar profiles (data/ground/annarbor, shared with mhacks-ski)
  course  hikesim.course: the park's longest walk as a MuJoCo heightfield course
  trees   hikesim.trees: the real trees along it from the 1 m canopy height map, drawn 8-bit
  tiles   hikesim.tiles: 40 m training stretches cut from the park's trails
  walk    hikesim.walk: G1 runs on the course
  video   hikesim.render: chase-cam videos of those runs, with the 8-bit trees (copied to web/videos for the drawer)
  viewer  hikesim.export_web: course and runs exported to the 3D viewer
A stage is "stale" when something it was built from changed after it was built. For courses that is checked on
content: the course's route has to still lie on the current trail lines (ground.py rewrites its files on every run,
so file times alone would flag courses whose trails did not move).
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hikesim.course import OUT, TRAILS, slug
from hikesim.geo import utm17

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "out" / "hike" / "web"
RUNS = ROOT / "out" / "hike" / "runs"
TILES = ROOT / "out" / "hike" / "tiles" / "mosaic.json"
MIN_KM = 0.5          # parks with less trail than this are listed but not ranked
DRIFT_M = 2.0         # a course is stale if its route sits farther than this from today's trail lines


def mtime(p: Path) -> float | None:
  return p.stat().st_mtime if p.exists() else None


def hhmm(t: float | None) -> str | None:
  return datetime.fromtimestamp(t).strftime("%b %d %H:%M") if t else None


def ordinal(n: int) -> str:
  return {1: "", 2: "2nd ", 3: "3rd "}.get(n, f"{n}th ")


def trail_tree(feats: list[dict]) -> cKDTree:
  """Every trail in the dataset, resampled at 1 m in UTM 17N."""
  pts = []
  for f in feats:
    xy = np.array([utm17(la, lo) for lo, la in f["geometry"]["coordinates"]])
    for a, b in zip(xy, xy[1:]):
      n = max(1, int(np.hypot(*(b - a))))
      pts.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
  return cKDTree(np.vstack(pts))


def route_drift(route: dict, tree: cKDTree) -> float:
  """How far the course's route sits from the current trail lines (99th percentile, metres)."""
  ox, oy = route["utm_origin"]
  d, _ = tree.query(np.stack([np.asarray(route["x"]) + ox, np.asarray(route["y"]) + oy], 1))
  return float(np.percentile(d, 99))


def signature(feats: list[dict]) -> dict:
  L = sum(f["properties"]["statistics"]["length_m"] for f in feats)
  st = [f["properties"]["statistics"] for f in feats]
  surf = defaultdict(float)
  for f in feats:
    surf[f["properties"].get("surface") or "unmapped"] += f["properties"]["statistics"]["length_m"]
  steep = sum(s["length_m"] * s.get("share_steeper_than_15deg", 0) for s in st) / L if L else 0
  lo = [s["min_elevation_m"] for s in st if "min_elevation_m" in s]; hi = [s["max_elevation_m"] for s in st if "max_elevation_m" in s]
  src = {f["properties"].get("elevation_source") for f in feats} - {None}
  return {"ways": len(feats), "km": round(L / 1000, 2), "climb_m": round(sum(s.get("climb_m", 0) for s in st)),
          "vertical_m": round(max(hi) - min(lo), 1) if lo else None,
          "steepest_deg": max((s.get("max_grade_deg_10m", 0) for s in st), default=0),
          "steep_share": round(steep, 3),
          "surfaces": [[k, round(v / L, 2)] for k, v in sorted(surf.items(), key=lambda kv: -kv[1])[:3]] if L else [],
          "bridges": sum(f["properties"].get("elevation_method") == "interpolated" for f in feats),
          "elevation": "1 m lidar" if src == {"3dep"} else "global 30 m tiles" if src == {"terrarium"} else "mixed" if src else None}


def differences(rows: list[dict]) -> None:
  """A few words on what sets each park apart, ranked against the parks with real trail networks."""
  big = [r for r in rows if r["sig"]["km"] >= MIN_KM]
  ranks = {}
  for key, label, rev in (("steepest_deg", "steepest", True), ("km", "most trail", True),
                          ("climb_per_km", "hilliest", True), ("steep_share", "most ground over 15 deg", True)):
    for i, r in enumerate(sorted(big, key=lambda r: r["sig"].get(key) or 0, reverse=rev)):
      ranks.setdefault(r["name"], []).append((i + 1, label))
  for r in rows:
    s = r["sig"]; tags = []
    for rank, label in sorted(ranks.get(r["name"], [])):
      if rank <= 3 and len(tags) < 2:
        tags.append(f"{ordinal(rank)}{label}")
    if s["km"] >= MIN_KM and s["steepest_deg"] < 8:
      tags.append("flat, nothing over 8 deg")
    top = s["surfaces"][0] if s["surfaces"] else None
    if top and top[0] == "wood" and top[1] >= 0.2:
      tags.append("boardwalk heavy")
    elif top and top[0] == "unmapped" and top[1] >= 0.6:
      tags.append("surface mostly unmapped")
    elif top and top[1] >= 0.6:
      tags.append(f"mostly {top[0].replace('_', ' ')}")
    if s["bridges"]:
      tags.append(f"{s['bridges']} bridges or boardwalks")
    if s["elevation"] and s["elevation"] != "1 m lidar":
      tags.append(f"elevation from {s['elevation']}")
    if s["km"] < MIN_KM:
      tags.append("too little trail to rank")
    r["different"] = tags[:4]


def main():
  feats = json.loads(TRAILS.read_text())["features"]
  areas = json.loads((TRAILS.parent / "areas.geojson").read_text())["features"]
  data_t = mtime(TRAILS)
  tree = trail_tree(feats)
  by_area = defaultdict(list)
  for f in feats:
    if f["properties"].get("area"):
      by_area[f["properties"]["area"]["name"]].append(f)
  tiles = defaultdict(int)
  if TILES.exists():
    for t in json.loads(TILES.read_text())["tiles"]:
      tiles[t.get("area")] += 1
  tiles_t = mtime(TILES)
  man = json.loads((WEB / "manifest.json").read_text()) if (WEB / "manifest.json").exists() else {"courses": []}
  man_t = mtime(WEB / "manifest.json")
  in_viewer = {c["slug"]: c for c in man["courses"]}

  rows = []
  for a in sorted({a["properties"]["name"] for a in areas}):
    s, f = slug(a), by_area.get(a, [])
    if not f:
      continue
    sig = signature(f)
    sig["climb_per_km"] = round(sig["climb_m"] / sig["km"], 1) if sig["km"] else 0
    route = OUT / s / "route.json"; route_t = mtime(route)
    runs = [json.loads(p.read_text()) | {"_t": mtime(p)} for p in sorted((RUNS / s).glob("*/run.json"))]
    run_t = max((r["_t"] for r in runs), default=None)
    st = {"data": {"state": "done", "at": hhmm(data_t), "detail": f"{sig['ways']} ways, {sig['km']} km on {sig['elevation']}"}}
    if route_t:
      r = json.loads(route.read_text())
      drift = route_drift(r, tree)
      stale = drift > DRIFT_M
      st["course"] = {"state": "stale" if stale else "done", "at": hhmm(route_t),
                      "detail": f"{r['length_m'] / 1000:.2f} km walk, steepest {r['max_grade_deg']} deg; "
                                + (f"trails moved up to {drift:.0f} m since it was built" if stale
                                   else "route still on today's trail lines")}
      if stale:
        st["course"]["cmd"] = f'.venv/bin/python -m hikesim.course "{a}"'
    else:
      st["course"] = {"state": "todo", "cmd": f'.venv/bin/python -m hikesim.course "{a}"'}
    if tiles[a]:
      st["tiles"] = {"state": "stale" if data_t and tiles_t and data_t > tiles_t else "done", "at": hhmm(tiles_t),
                     "detail": f"{tiles[a]} of {sum(tiles.values())} training stretches"}
    else:
      st["tiles"] = {"state": "todo" if sig["km"] >= MIN_KM else "skip",
                     "detail": "no straight 40 m stretch picked yet" if sig["km"] >= MIN_KM else "too little trail"}
    if runs:
      fin = sum(r.get("result") == "finished" for r in runs)
      best = max((r.get("walked_m") or 0) for r in runs)
      # Modal benchmark runs carry "ckpt": null for SONIC alone, a checkpoint name for a trained policy.
      base = [r.get("walked_m") or 0 for r in runs if "ckpt" in r and r["ckpt"] is None]
      trained = [r.get("walked_m") or 0 for r in runs if r.get("ckpt")]
      vs = (f"; SONIC alone best {max(base):.0f} m" if base else "") + (f", trained best {max(trained):.0f} m" if trained else "")
      stale = route_t and run_t and route_t > run_t
      st["walk"] = {"state": "stale" if stale else "done", "at": hhmm(run_t),
                    "detail": f"{len(runs)} runs, {fin} finished, best {best:.0f} m{vs}" + (", course rebuilt since" if stale else "")}
    else:
      st["walk"] = {"state": "todo" if route_t else "blocked",
                    **({"cmd": f".venv/bin/python -m hikesim.walk {s} --length 120"} if route_t else {})}
    if s in in_viewer:
      newer = max(t for t in (route_t, run_t) if t) if (route_t or run_t) else None
      stale = newer and man_t and newer > man_t
      st["viewer"] = {"state": "stale" if stale else "done", "at": hhmm(man_t),
                      "detail": f"{len(in_viewer[s].get('runs', []))} runs in the viewer"
                                + (", newer course or runs not exported" if stale else ""),
                      **({"cmd": ".venv/bin/python -m hikesim.export_web --skip-area"} if stale else {})}
    else:
      st["viewer"] = {"state": "todo" if route_t else "blocked",
                      **({"cmd": ".venv/bin/python -m hikesim.export_web --skip-area"} if route_t else {})}
    # 8-bit trees: built per course from the canopy map; stale if the course was rebuilt after them.
    tj = OUT / s / "trees.json"; trees_t = mtime(OUT / s / "trees.npy")
    if trees_t and tj.exists():
      t = json.loads(tj.read_text())
      stale = route_t and route_t > trees_t
      st["trees"] = {"state": "stale" if stale else "done", "at": hhmm(trees_t),
                     "detail": f"{t['trees']} real trees, {t.get('bushes', 0)} bushes, tallest {t['tallest_m']} m"
                               + (", course rebuilt since" if stale else "")}
      if stale:
        st["trees"]["cmd"] = f".venv/bin/python -m hikesim.trees {s}"
    else:
      st["trees"] = {"state": "todo" if route_t else "blocked",
                     **({"cmd": f".venv/bin/python -m hikesim.trees {s}"} if route_t else {})}
    # Hike videos: a run's chase.mp4. Stale if the run or the trees changed after it was rendered.
    vids = []
    for rd in sorted((RUNS / s).glob("*/chase.mp4")):
      rd = rd.parent; v_t = mtime(rd / "chase.mp4")
      meta = json.loads((rd / "render.json").read_text()) if (rd / "render.json").exists() else {}
      info = json.loads((rd / "run.json").read_text())
      stale_v = (mtime(rd / "frames.npz") or 0) > v_t or (trees_t or 0) > v_t
      dst = WEB / "videos" / s / f"{rd.name}.mp4"
      if not dst.exists() or dst.stat().st_mtime < v_t:
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(rd / "chase.mp4", dst)
      if (rd / "still.jpg").exists() and not (dst.with_suffix(".jpg")).exists():
        shutil.copy(rd / "still.jpg", dst.with_suffix(".jpg"))
      vids.append({"run": rd.name, "file": f"videos/{s}/{rd.name}.mp4", "poster": f"videos/{s}/{rd.name}.jpg",
                   "trees": bool(meta.get("trees")), "stale": bool(stale_v), "result": info.get("result"),
                   "walked_m": info.get("walked_m"), "at": hhmm(v_t)})
    if vids:
      bad = [v for v in vids if v["stale"] or not v["trees"]]
      st["video"] = {"state": "stale" if bad else "done", "at": max(v["at"] for v in vids),
                     "detail": f"{len(vids)} videos, {sum(v['trees'] for v in vids)} with trees"
                               + (f"; {len(bad)} need a re-render" if bad else "")}
      if bad:
        st["video"]["cmd"] = f".venv/bin/python -m hikesim.render out/hike/runs/{s}/{bad[0]['run']}"
    elif runs:
      best = max(sorted((RUNS / s).glob("*/run.json")), key=lambda p: json.loads(p.read_text()).get("walked_m") or 0).parent
      st["video"] = {"state": "todo", "cmd": f".venv/bin/python -m hikesim.render out/hike/runs/{s}/{best.name}"}
    else:
      st["video"] = {"state": "blocked"}
    order = ["data", "course", "trees", "tiles", "walk", "video", "viewer"]
    nxt = next((k for k in order if st[k]["state"] in ("todo", "stale") and st[k].get("cmd")), None)
    rows.append({"name": a, "slug": s, "sig": sig, "stages": st, "next": nxt, "videos": vids,
                 "built": st["course"]["state"] in ("done", "stale"), "in_viewer": s in in_viewer})
  differences(rows)
  rows.sort(key=lambda r: (not r["in_viewer"], not r["built"], -r["sig"]["km"]))

  count = lambda k: sum(r["stages"][k]["state"] in ("done", "stale") for r in rows)
  out = {"generated": hhmm(datetime.now().timestamp()), "dataset": str(TRAILS.relative_to(ROOT)),
         "stages": [{"key": k, "label": lbl, "done": count(k), "stale": sum(r["stages"][k]["state"] == "stale" for r in rows)}
                    for k, lbl in (("data", "Terrain data"), ("course", "Course"), ("trees", "8-bit trees"),
                                   ("tiles", "Training tiles"), ("walk", "G1 walks"), ("video", "Hike videos"),
                                   ("viewer", "In viewer"))],
         "parks": len(rows), "areas": rows}
  WEB.mkdir(parents=True, exist_ok=True)
  (WEB / "pipeline.json").write_text(json.dumps(out))
  shutil.copy(Path(__file__).parent / "web" / "pipeline.js", WEB / "pipeline.js")
  print(f"{len(rows)} parks with trails: " + ", ".join(f"{s['label']} {s['done']}" + (f" ({s['stale']} stale)" if s["stale"] else "")
                                                     for s in out["stages"]))
  for r in rows[:12]:
    flags = "".join({"done": "#", "stale": "~", "todo": ".", "blocked": " ", "skip": "-"}[r["stages"][k]["state"]]
                    for k in ("data", "course", "trees", "tiles", "walk", "video", "viewer"))
    print(f"  [{flags}] {r['name'][:34]:<34} {r['sig']['km']:5.2f} km  {'; '.join(r['different'])}")
  print(f"wrote {WEB / 'pipeline.json'}")


if __name__ == "__main__":
  main()
