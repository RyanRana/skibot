"""Load everything Hazard Intelligence has that is not code into the team's SpacetimeDB (spacetime/spacetimedb).

    python tools/stdb_load.py --server local                  # every dataset into a local `spacetime start`
    python tools/stdb_load.py --server maincloud courses video  # just these
    python tools/stdb_load.py --server local courses --only kitzbuhel-streif

Datasets (each loader standardises one kind of file into the module's tables):
    courses   47 ski courses: resort, centreline, gates, start pose, MuJoCo scene, 2 m heightfield
    far       the game's far-field terrain around the Streif (60 m cells, +-10 km)
    trails    ground.py networks: every way with its elevation profile, and the parks they run through
    hike      the hike courses: route, 1 m lidar elevation, canopy height, trees and bushes
    tiles     training mosaics: the ski tile mosaics (v1 to v4, with trees) and the hike tiles
    video     joints from YouTube GoPro follow-cam ski videos (links only, never the video)
    gopro     GoPro telemetry at 100 Hz from two Marmolada ski days, split into descents
    clips     the walking clips the hike policy imitates
    poses     the ski reference poses measured from World Cup giant slalom video
    policies  the policy registry (the ONNX runtime stays on Modal)
    evals     benchmark runs, course rides and hike stretches, and the training runs behind them
    sets      the training sets: which terrains, poses, courses and clips each policy family trains on

It only reads --root (out/ and data/ of the checkout that holds the data) and --hike-root, and writes under the public
tenant as the identity the spacetime CLI is logged in as. Run `spacetime call <db> claim_platform_admin` once first.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
SERVERS = {"local": "http://127.0.0.1:3000", "maincloud": "https://maincloud.spacetimedb.com"}
TENANT = "hazard-intelligence"
API = "https://ryanrana04--hazard-intelligence-api-api.modal.run"
MAX_VALUES = 40_000  # f32 values per reducer call, about 400 KB of JSON
OSM = "OpenStreetMap contributors, ODbL 1.0"
YOUTUBE_NOTE = "Joint angles measured from a public YouTube video. The video itself is not stored, only its link."

# MuJoCo joint order of the Unitree G1 29-dof (skisim/sonic.py), used for every joint list written here.
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# The GoPro follow-cam videos the pose pipeline ran on (skisim/pose_video.py). Titles as YouTube lists them.
VIDEOS = {
    "fXVIpXY3rI4": ("Ted Ligety Carving Follow Cam | Deer Valley, UT (January 2022)", "SHRED."),
    "qN2L2YVj45o": ("Follow Cam Carving - Up close like you are there", "Tom Gellie - Big Picture Skiing"),
    "QNkNHEG8qC8": ("Ted Ligety GS Turns and Angles | Catch me if you can! | Training Follow Cam (2018)", "SHRED."),
}


class Db:
    def __init__(self, base: str, name: str, token: str):
        self.base, self.name = base.rstrip("/"), name
        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        self.calls = 0

    def call(self, reducer: str, args: dict):
        body = json.dumps(args, separators=(",", ":"), allow_nan=False)
        for attempt in range(4):
            try:
                r = self.http.post(f"{self.base}/v1/database/{self.name}/call/{reducer}", data=body, timeout=120)
            except requests.RequestException as e:
                if attempt == 3:
                    raise RuntimeError(f"{reducer}: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                self.calls += 1
                return
            if r.status_code in (502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{reducer}: HTTP {r.status_code} {r.text.strip()[:400]}")

    def sql(self, query: str) -> list[list]:
        r = self.http.post(f"{self.base}/v1/database/{self.name}/sql", data=query, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"sql: HTTP {r.status_code} {r.text.strip()[:300]}")
        return [row for stmt in r.json() for row in stmt["rows"]]


def cli_token() -> str:
    cfg = (Path.home() / ".config" / "spacetime" / "cli.toml").read_text()
    m = re.search(r'spacetimedb_token\s*=\s*"([^"]+)"', cfg)
    if not m:
        sys.exit("no spacetime CLI login; run `spacetime login` first")
    return m.group(1)


# ---------------------------------------------------------------------------------------------------- helpers

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def ts(seconds: float) -> dict:
    return {"__timestamp_micros_since_unix_epoch__": int(seconds * 1e6)}


def finite(a: np.ndarray) -> tuple[np.ndarray, int]:
    """JSON has no NaN: non-finite values become 0, and the caller records how many there were."""
    bad = ~np.isfinite(a)
    return np.where(bad, 0.0, a), int(bad.sum())


def rnd(a, nd: int) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def num(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def text(x) -> str:
    return "" if x is None or x == "None" else str(x)


def parallel(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(fn, items):
            pass


def put_grid(db: Db, tid: str, course: str, layer: str, z: np.ndarray, cell: float, x0: float, y0: float,
             source: str, license_: str, meta: dict, nd: int = 3):
    """A 2-D grid as a terrain row plus chunks of whole rows."""
    z = np.asarray(z, dtype=np.float32)
    z, holes = finite(z)
    ny, nx = z.shape
    if holes:
        meta = {**meta, "non_finite_cells_set_to_0": holes}
    db.call("put_terrain", {"id": tid, "tenant": TENANT, "courseId": course, "layer": layer, "nx": nx, "ny": ny,
                            "cellM": float(cell), "x0": float(x0), "y0": float(y0), "zMin": float(z.min()),
                            "zMax": float(z.max()), "source": source, "license": license_, "meta": json.dumps(meta)})
    rows = max(1, MAX_VALUES // nx)
    parallel(lambda r0: db.call("push_terrain_rows", {"terrainId": tid, "row0": r0, "heights": rnd(z[r0:r0 + rows].ravel(), nd)}),
             range(0, ny, rows))


def put_points(db: Db, sid: str, course: str, kind: str, columns: list[str], units: list[str], pts: np.ndarray,
               source: str, license_: str, meta: dict, nd: int = 3):
    pts, holes = finite(np.asarray(pts, dtype=np.float32))
    if holes:
        meta = {**meta, "non_finite_values_set_to_0": holes}
    db.call("put_point_set", {"id": sid, "tenant": TENANT, "courseId": course, "kind": kind, "columns": columns,
                              "units": units, "source": source, "license": license_, "meta": json.dumps(meta)})
    per = max(1, MAX_VALUES // len(columns))
    parallel(lambda i: db.call("push_points", {"setId": sid, "seq": i // per, "data": rnd(pts[i:i + per].ravel(), nd)}),
             range(0, len(pts), per))


def put_recording(db: Db, key: str, source: str, activity: str, channels: list[str], units: list[str],
                  frames: np.ndarray, rate_hz: float, title: str, license_: str, meta: dict, ref: str = "",
                  uri: str = "", t_ms: np.ndarray | None = None, nd: int = 4):
    """frames is [n, channels]. Regular streams pass rate_hz; irregular ones pass rate_hz 0 and t_ms per frame."""
    frames, holes = finite(np.asarray(frames, dtype=np.float32))
    if holes:
        meta = {**meta, "non_finite_values_set_to_0": holes}
    db.call("put_recording", {"key": key, "tenant": TENANT, "source": source, "activity": activity, "ref": ref,
                              "uri": uri, "channels": channels, "units": units, "rateHz": float(rate_hz),
                              "title": title, "license": license_, "meta": json.dumps(meta)})
    per = max(1, MAX_VALUES // len(channels))

    def push(i):
        chunk = {"recordingKey": key, "seq": i // per, "data": rnd(frames[i:i + per].ravel(), nd)}
        if rate_hz > 0:
            chunk.update(t0Ms=int(round(i * 1000 / rate_hz)), tMs=[])
        else:
            base = int(t_ms[i])
            chunk.update(t0Ms=base, tMs=[int(v) - base for v in t_ms[i:i + per]])
        db.call("push_recording_chunk", chunk)

    parallel(push, range(0, len(frames), per))
    db.call("close_recording", {"key": key})


def segments(db: Db, key: str, segs: list[dict]):
    for i in range(0, len(segs), 200):
        db.call("push_segments", {"recordingKey": key, "segments": segs[i:i + 200]})


def regular(t_s: np.ndarray) -> float:
    """The sample rate if the timestamps are evenly spaced to within a millisecond, else 0."""
    d = np.diff(t_s)
    if len(d) == 0:
        return 0.0
    dt = float(np.median(d))
    return round(1.0 / dt, 4) if dt > 0 and np.abs(d - dt).max() < 1e-3 else 0.0


def utm_to_latlon(e: float, n: float, zone: int) -> tuple[float, float]:
    """Inverse transverse Mercator on GRS80 (NAD83 / UTM north, e.g. EPSG:26917), Snyder's series; sub-millimetre."""
    a, f, k0 = 6378137.0, 1 / 298.257222101, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    x, y = e - 500000.0, n
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
         + (151 * e1 ** 3 / 96) * math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    c1, t1 = ep2 * math.cos(p) ** 2, math.tan(p) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(p) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = p - (n1 * math.tan(p) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
                                         + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(p)
    return math.degrees(lat), (zone - 1) * 6 - 180 + 3 + math.degrees(lon)


# ---------------------------------------------------------------------------------------------------- loaders

def load_courses(db: Db, root: Path, only: set[str]):
    resorts = set()
    dirs = sorted(d for d in (root / "out" / "courses").iterdir() if (d / "stats.json").exists())
    for d in dirs:
        if only and d.name not in only:
            continue
        stats = json.loads((d / "stats.json").read_text())
        grid = json.loads((d / "grid.json").read_text())
        run = json.loads((d / "run.json").read_text())
        gates = json.loads((d / "gates.json").read_text())
        geo = stats.get("geocode") or {}
        rid = slug(stats.get("resort_query") or d.name.split("-")[0])
        if rid not in resorts:
            resorts.add(rid)
            db.call("upsert_resort", {"id": rid, "tenant": TENANT, "name": stats.get("resort_query") or rid,
                                      "country": (geo.get("display_name") or "").split(",")[-1].strip(),
                                      "lat": num(geo.get("lat")), "lon": num(geo.get("lon")), "osm": text(geo.get("osm")),
                                      "source": "Nominatim (OpenStreetMap)", "meta": json.dumps(geo)})
        c, full = stats.get("course") or {}, stats.get("full_run") or {}
        pts = lambda a: [{"x": round(p[0], 3), "y": round(p[1], 3), "z": round(p[2], 3), "s": round(p[3], 3)} for p in a]
        gate_rows = [{"s": round(g["s"], 3), "side": g.get("side", ""), "color": g.get("color", ""), "turn": g.get("turn_direction", ""),
                      "turn_x": g["turn_pole"][0], "turn_y": g["turn_pole"][1], "turn_z": g["turn_pole"][2],
                      "outer_x": g["outer_pole"][0], "outer_y": g["outer_pole"][1], "outer_z": g["outer_pole"][2]}
                     for g in gates.get("gates", [])]
        scene = (d / "scene_g1.xml").read_text() if (d / "scene_g1.xml").exists() else ""
        start = json.loads((d / "start_pose.json").read_text()) if (d / "start_pose.json").exists() else {}
        db.call("upsert_course", {
            "id": d.name, "tenant": TENANT, "resortId": rid, "name": (stats.get("run") or {}).get("name") or d.name,
            "activity": "ski", "difficulty": text((stats.get("run") or {}).get("difficulty")),
            "lengthM": num(c.get("length_3d_m")), "dropM": num(c.get("vertical_drop_m")),
            "meanSlopeDeg": num(c.get("mean_slope_deg")), "maxSlopeDeg": num(c.get("max_slope_10m_deg")),
            "originLat": num(grid.get("lat0")), "originLon": num(grid.get("lon0")), "zDatumMsl": num(grid.get("z_datum_msl")),
            "line": pts(run.get("course", [])), "fullRun": pts(run.get("full_run", [])), "gates": gate_rows,
            "startPose": json.dumps(start), "scene": scene,
            "stats": json.dumps({**stats, "gate_params": gates.get("params"), "finish": gates.get("finish"), "gate_rules": gates.get("rules")}),
            "source": "course.py: OpenStreetMap pistes, AWS Terrain Tiles elevation",
            "license": f"{OSM}; AWS Terrain Tiles (Mapzen/Tilezen), attribution required"})
        put_grid(db, d.name, d.name, "elevation", np.load(d / "elevation.npy"), grid["cell"], grid["x0"], grid["y0"],
                 text(grid.get("dem")), "AWS Terrain Tiles (Mapzen/Tilezen), attribution required", grid)
        print(f"  course {d.name}: {len(run.get('course', []))} points, {len(gate_rows)} gates, grid {grid['nrow']}x{grid['ncol']}", flush=True)


def load_far(db: Db, repo: Path, only: set[str]):
    d = repo / "game" / "public" / "courses" / "kitzbuhel-streif"
    meta = json.loads((d / "far.json").read_text())
    z = np.fromfile(d / "far.bin", dtype="<f4").reshape(meta["nrow"], meta["ncol"])
    put_grid(db, "kitzbuhel-streif/far", "kitzbuhel-streif", "far", z, meta["cell"], meta["x0"], meta["y0"],
             text(meta.get("source")), "AWS Terrain Tiles (Mapzen/Tilezen), attribution required", meta, nd=2)
    print(f"  far field {meta['nrow']}x{meta['ncol']} at {meta['cell']} m", flush=True)


def load_trails(db: Db, root: Path, only: set[str]):
    for d in sorted((root / "data" / "ground").iterdir()):
        if not (d / "meta.json").exists() or (only and d.name not in only):
            continue
        meta = json.loads((d / "meta.json").read_text())
        kind = meta.get("kind", "trails")
        ways = json.loads((d / f"{kind}.geojson").read_text())["features"]
        areas = json.loads((d / "areas.geojson").read_text())["features"] if (d / "areas.geojson").exists() else []
        center = meta.get("center") or [0, 0]
        db.call("upsert_trail_network", {
            "id": d.name, "tenant": TENANT, "name": meta.get("name") or d.name, "place": text(meta.get("place")), "kind": kind,
            "centerLat": num(center[0]), "centerLon": num(center[1]), "radiusM": num(meta.get("radius_m")),
            "ways": len(ways), "lengthKm": num(meta.get("length_km")), "areas": len(areas),
            "sources": json.dumps(meta.get("sources") or {}), "meta": json.dumps(meta)})
        rows = []
        for f in ways:
            p = f["properties"]
            st = p.get("statistics") or {}
            prof = p.get("elevationProfile") or {}
            heights = [num(h) for h in prof.get("heights") or []]
            coords = f["geometry"]["coordinates"] if f["geometry"]["type"] == "LineString" else [c for part in f["geometry"]["coordinates"] for c in part]
            known = {"id", "name", "highway", "surface", "sac_scale", "statistics", "elevationProfile", "area"}
            rows.append({
                "osm": text(p.get("id")), "name": text(p.get("name")), "highway": text(p.get("highway")),
                "surface": text(p.get("surface")), "sac_scale": text(p.get("sac_scale")),
                "area": text((p.get("area") or {}).get("name")), "length_m": num(st.get("length_m")),
                "climb_m": num(st.get("climb_m")), "descent_m": num(st.get("descent_m")),
                "max_grade_deg": num(st.get("max_grade_deg_10m", st.get("max_grade_deg"))),
                "path": [{"lat": round(c[1], 7), "lon": round(c[0], 7)} for c in coords],
                "profile_step_m": num(prof.get("resolution", prof.get("step_m", 5.0)), 5.0),
                "profile": [round(h, 2) for h in heights],
                "tags": json.dumps({k: v for k, v in p.items() if k not in known and v not in (None, "None")} | {"statistics": st})})
        batches, cur, size = [], [], 0
        for r in rows:
            n = len(json.dumps(r))
            if cur and size + n > 400_000:
                batches.append(cur); cur, size = [], 0
            cur.append(r); size += n
        if cur:
            batches.append(cur)
        parallel(lambda b: db.call("push_trails", {"networkId": d.name, "trails": b}), batches)
        arows = [{"osm": text(f["properties"].get("id")), "name": text(f["properties"].get("name")),
                  "kind": text(f["properties"].get("kind")), "ways": int(num(f["properties"].get("ways"))),
                  "length_m": num(f["properties"].get("length_m")), "geometry": json.dumps(f["geometry"]),
                  "stats": json.dumps(f["properties"])} for f in areas]
        for i in range(0, len(arows), 20):
            db.call("push_areas", {"networkId": d.name, "areas": arows[i:i + 20]})
        print(f"  trails {d.name}: {len(rows)} ways, {len(arows)} areas", flush=True)


def load_hike(db: Db, hike_root: Path, only: set[str]):
    lic = f"USGS 3DEP lidar (public domain); Meta/WRI canopy height (CC BY 4.0); {OSM}"
    for d in sorted((hike_root / "out" / "hike" / "courses").iterdir()):
        if not (d / "route.json").exists() or (only and d.name not in only):
            continue
        grid = json.loads((d / "grid.json").read_text())
        route = json.loads((d / "route.json").read_text())
        e0, n0 = route.get("utm_origin") or grid["utm_origin"]
        zone = int(str(grid.get("crs", "EPSG:26917")).split(":")[-1]) - 26900
        lat0, lon0 = utm_to_latlon(e0, n0, zone)
        x, y, z = (np.asarray(route[k], dtype=np.float64) for k in ("x", "y", "z"))
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        grade = np.asarray(route.get("grade_deg", []), dtype=np.float64)
        stats = {k: v for k, v in route.items() if k not in ("x", "y", "z", "grade_deg")}
        db.call("upsert_course", {
            "id": d.name, "tenant": TENANT, "resortId": "", "name": route.get("area") or d.name, "activity": "hike",
            "difficulty": "", "lengthM": num(route.get("length_m")), "dropM": num(route.get("vertical_m")),
            "meanSlopeDeg": float(np.nanmean(np.abs(grade))) if len(grade) else 0.0, "maxSlopeDeg": num(route.get("max_grade_deg")),
            "originLat": round(lat0, 7), "originLon": round(lon0, 7), "zDatumMsl": 0.0,
            "line": [{"x": round(a, 3), "y": round(b, 3), "z": round(c, 3), "s": round(t, 3)} for a, b, c, t in zip(x, y, z, s)],
            "fullRun": [], "gates": [], "startPose": "{}", "scene": "",
            "stats": json.dumps({**stats, "crs": grid.get("crs"), "utm_origin": [e0, n0], "frame": "x east, y north from utm_origin, z metres MSL"}),
            "source": "hikesim/course.py: OpenStreetMap trails on USGS 3DEP 1 m lidar", "license": lic})
        meta = {**grid, "frame": "x east, y north metres from utm_origin; values are metres"}
        put_grid(db, d.name, d.name, "elevation", np.load(d / "elevation.npy"), grid["cell"], grid["x0"], grid["y0"],
                 "USGS 3DEP 1 m lidar", "Public domain", meta)
        if (d / "canopy.npy").exists():
            put_grid(db, f"{d.name}/canopy", d.name, "canopy", np.load(d / "canopy.npy"), grid["cell"], grid["x0"], grid["y0"],
                     "Meta/WRI global canopy height", "CC BY 4.0", meta, nd=2)
        if (d / "trees.npy").exists():
            tmeta = json.loads((d / "trees.json").read_text()) if (d / "trees.json").exists() else {}
            put_points(db, f"{d.name}/trees", d.name, "trees", ["x", "y", "crown_radius", "height", "kind", "ground_z"],
                       ["m", "m", "m", "m", "0 tree, 1 bush", "m"], np.load(d / "trees.npy"),
                       "hikesim/trees.py from the Meta/WRI canopy height map", "CC BY 4.0", tmeta, nd=2)
        print(f"  hike {d.name}: {len(x)} route points, grid {grid['nrow']}x{grid['ncol']}", flush=True)


def load_tiles(db: Db, root: Path, hike_root: Path, only: set[str]):
    mosaics = [(f"ski-tiles{'' if v == '' else '-' + v}", root / "out" / f"terrains{'' if v == '' else '_' + v}")
               for v in ("", "v2", "v3", "v4")] + [("hike-tiles", hike_root / "out" / "hike" / "tiles")]
    for tid, d in mosaics:
        if not (d / "mosaic.json").exists() or (only and tid not in only):
            continue
        meta = json.loads((d / "mosaic.json").read_text())
        z = np.load(d / "mosaic_elevation.npy")
        src = "skisim/perturb.py tile mosaic" if tid.startswith("ski") else "hikesim/tiles.py tile mosaic"
        put_grid(db, tid, "", "tiles", z, meta["cell"], meta.get("x0", 0.0), meta.get("y0", 0.0), src,
                 f"Derived from {OSM} and USGS 3DEP / AWS Terrain Tiles", meta)
        if (d / "trees.npy").exists():
            trees = np.load(d / "trees.npy")
            put_points(db, f"{tid}/trees", "", "trees", ["x", "y", "radius", "height", "tile"][:trees.shape[1]],
                       ["m", "m", "m", "m", "index"][:trees.shape[1]], trees, src, "", {"terrain": tid})
        print(f"  tiles {tid}: {z.shape[0]}x{z.shape[1]} at {meta['cell']} m", flush=True)


def load_video(db: Db, root: Path, only: set[str]):
    for vid, (title, author) in VIDEOS.items():
        f = root / "out" / "pose" / f"pose_data_{vid}.json"
        if not f.exists() or (only and vid not in only):
            continue
        rec = json.loads(f.read_text())
        t = np.array([r["t"] for r in rec], dtype=np.float64)
        human = [("kL", "knee_l", "deg"), ("kR", "knee_r", "deg"), ("lean", "lean", "deg"), ("ang", "feet_out", "% of leg"),
                 ("gyro", "torso_gyro", "deg/s"), ("torso", "torso", "deg"), ("measured", "measured", "bool"),
                 ("robot_heading_deg", "robot_heading", "deg"), ("robot_turn_rate_dps", "robot_turn_rate", "deg/s")]
        human = [h for h in human if all(h[0] in r for r in rec)]
        joints = [j for j in G1_JOINTS if any(j in r["g1"] for r in rec)]
        absent = sum(1 for r in rec for j in joints if j not in r["g1"])
        cols = [num(r[k]) if k != "measured" else float(bool(r[k])) for r in rec for k, _, _ in human]
        frames = np.array(cols, dtype=np.float64).reshape(len(rec), len(human))
        g1 = np.array([[num(r["g1"].get(j)) for j in joints] for r in rec], dtype=np.float64)
        rate = regular(t)
        key = f"video/{vid}"
        meta = {"author": author, "pipeline": "skisim/pose_video.py", "pose_model": "MediaPipe Pose (complexity 2), 3D world landmarks",
                "robot": "unitree_g1_29dof", "start_s": float(t[0]), "frames": len(rec),
                "g1_joints_absent_on_a_frame_set_to_0": absent, "note": YOUTUBE_NOTE}
        put_recording(db, key, "video", "ski", [h[1] for h in human] + joints, [h[2] for h in human] + ["rad"] * len(joints),
                      np.concatenate([frames, g1], 1), rate, title, YOUTUBE_NOTE, meta, ref=vid,
                      uri=f"https://www.youtube.com/watch?v={vid}", t_ms=np.round((t - t[0]) * 1000).astype(np.int64))
        segs, start = [], 0
        for i in range(1, len(rec) + 1):
            if i == len(rec) or rec[i]["phase"] != rec[start]["phase"]:
                segs.append({"key": f"phase-{len(segs)}", "kind": "phase", "t_0_ms": int(round((t[start] - t[0]) * 1000)),
                             "t_1_ms": int(round((t[i - 1] - t[0]) * 1000)), "label": rec[start]["phase"], "meta": "{}"})
                start = i
        segments(db, key, segs)
        print(f"  video {vid}: {len(rec)} frames at {rate or 'irregular'} Hz, {len(human) + len(joints)} channels, {len(segs)} phases", flush=True)


def load_gopro(db: Db, root: Path, only: set[str]):
    desc = json.loads((root / "out" / "gopro" / "descents.json").read_text())
    for f in sorted((root / "data" / "zenodo").glob("*.csv")):
        if only and f.stem not in only:
            continue
        with open(f) as fh:
            header = next(csv.reader(fh))
        use = [i for i in range(len(header)) if i != 1]  # column 1 is the ISO timestamp
        a = np.loadtxt(f, delimiter=",", skiprows=1, usecols=use)
        col = {header[i]: a[:, k] for k, i in enumerate(use)}
        t_ms = col["video-time (ms)"]
        lat, lon = col["latitude (deg)"], col["longitude (deg)"]
        fix = np.flatnonzero((lat != 0) | (lon != 0))
        lat0, lon0 = float(lat[fix[0]]), float(lon[fix[0]])
        east = np.radians(lon - lon0) * 6371008.8 * math.cos(math.radians(lat0))
        north = np.radians(lat - lat0) * 6371008.8
        chans = [("acc_x", "accelerometer-x (m/s²)", "m/s2", 1), ("acc_y", "accelerometer-y (m/s²)", "m/s2", 1),
                 ("acc_z", "accelerometer-z (m/s²)", "m/s2", 1), ("grav_x", "gravity-vector-x", "unit", 1),
                 ("grav_y", "gravity-vector-y", "unit", 1), ("grav_z", "gravity-vector-z", "unit", 1),
                 ("quat_w", "orientation-quaternion-w", "unit", 1), ("quat_x", "orientation-quaternion-x", "unit", 1),
                 ("quat_y", "orientation-quaternion-y", "unit", 1), ("quat_z", "orientation-quaternion-z", "unit", 1),
                 ("speed_2d", "speed [2d gps] (km/h)", "m/s", 1 / 3.6), ("speed_3d", "speed [3d gps] (km/h)", "m/s", 1 / 3.6),
                 ("alt_msl", "altitude [gps msl] (m)", "m", 1)]
        frames = np.stack([col[c] * s for _, c, _, s in chans] + [east, north], 1)
        rate = regular(t_ms / 1000.0)
        with open(f) as fh:
            next(fh)
            start_utc = next(fh).split(",")[1]
        key = f"gopro/{f.stem}"
        meta = {"origin_lat": lat0, "origin_lon": lon0, "start_utc": start_utc, "camera": "GoPro (GPMF telemetry)",
                "source_columns": header, "east_north": "metres from the first GPS fix (equirectangular)",
                "doi": "10.5281/zenodo.21777004", "author": "Prochazka (2026)", "parser": "gopro.py"}
        put_recording(db, key, "gopro", "ski", [c[0] for c in chans] + ["east", "north"], [c[2] for c in chans] + ["m", "m"],
                      frames, rate, f"Marmolada ski day, GoPro {f.stem}", "CC BY 4.0, Prochazka 2026, doi:10.5281/zenodo.21777004",
                      meta, ref="marmolada", uri="https://doi.org/10.5281/zenodo.21777004", t_ms=t_ms.astype(np.int64))
        segs = []
        for dsc in (desc.get("files", {}).get(f.name) or {}).get("descents", []):
            top = (dsc.get("piste_match") or {}).get("top") or []
            label = ", ".join(f"{name} {share:.0%}" for name, share in top)[:200]  # OSM pistes the descent followed, by share
            segs.append({"key": f"descent-{dsc['id']}", "kind": "descent", "t_0_ms": int(dsc["t_start_s"] * 1000),
                         "t_1_ms": int(dsc["t_end_s"] * 1000), "label": label, "meta": json.dumps(dsc)})
        segments(db, key, segs)
        print(f"  gopro {f.stem}: {len(frames)} frames at {rate or 'irregular'} Hz, {len(segs)} descents", flush=True)


def load_clips(db: Db, hike_root: Path, only: set[str]):
    c = np.load(hike_root / "out" / "hike" / "clips.npz")
    for i, name in enumerate(c["names"]):
        name = str(name)
        if only and name not in only:
            continue
        frames = np.concatenate([c["q"][i], c["quat"][i]], 1)
        put_recording(db, f"sim/hike-clip/{name}", "sim", "hike", G1_JOINTS + ["root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"],
                      ["rad"] * 29 + ["unit"] * 4, frames, 50.0, f"Walking clip {name}", "",
                      {"speed_mps": float(c["speed"][i]), "generator": "hikesim/clips.py (SONIC kinematic planner)",
                       "heading": "removed, so the env can point the reference anywhere"}, ref="g1-hike")
        print(f"  clip {name}: {len(frames)} frames at 50 Hz", flush=True)


def load_poses(db: Db, root: Path, only: set[str]):
    d = json.loads((root / "data" / "pose" / "ski_pose.json").read_text())
    m = d.get("_meta", {})
    sources = {**m, "videos": {k: {"frames": v, "url": f"https://www.youtube.com/watch?v={k}"} for k, v in (m.get("videos") or {}).items()},
               "pipeline": "skisim/pose_from_video.py", "used_by": "train/ski_env.py athletic_poses()"}
    for phase, p in d.items():
        if phase.startswith("_"):
            continue
        joints = [j for j in G1_JOINTS if j in p["g1"]]
        db.call("upsert_reference_pose", {"id": f"ski/{phase}", "tenant": TENANT, "activity": "ski", "phase": phase,
                                          "robot": "unitree_g1_29dof", "joints": joints, "values": [round(p["g1"][j], 5) for j in joints],
                                          "frames": int(p.get("frames", 0) if not isinstance(p.get("frames"), list) else len(p["frames"])),
                                          "human": json.dumps({"human_deg": p.get("human_deg"), "g1_raw": p.get("g1_raw")}),
                                          "sources": json.dumps(sources)})
    print(f"  poses: {[k for k in d if not k.startswith('_')]}", flush=True)


def load_policies(db: Db, repo: Path, only: set[str]):
    for d in sorted((repo / "policy_api" / "policies").iterdir()):
        if not (d / "meta.json").exists() or (only and d.name not in only):
            continue
        m = json.loads((d / "meta.json").read_text())
        db.call("upsert_policy", {"id": m["id"], "tenant": TENANT, "name": m.get("name", m["id"]), "robot": m.get("robot", ""),
                                  "skill": m.get("skill", ""), "obsDim": int(m["obs_dim"]), "actionDim": int(m["action_dim"]),
                                  "controlHz": float(m.get("control_hz", 0)), "runtime": "modal",
                                  "endpoint": f"{API}/v1/policies/{m['id']}", "trainingSet": "g1-ski",
                                  "benchmark": json.dumps(m.get("benchmark")), "spec": json.dumps(m)})
        print(f"  policy {m['id']}: {m['obs_dim']} -> {m['action_dim']}", flush=True)


def _ckpt(path: str) -> str:
    """Checkpoint paths relative to the checkout (runs_live/...) or the Modal volume (/runs/...), never a home folder."""
    return re.sub(r"^.*?(runs_live/|runs_smoke/|/runs/)", r"\1", path or "")


def _run_of(ckpt: str) -> tuple[str, int]:
    m = re.search(r"(\d{8}-\d{6})/model_(\d+)\.pt", ckpt or "")
    return (m.group(1), int(m.group(2))) if m else ("", 0)


def load_evals(db: Db, root: Path, hike_root: Path, repo: Path, only: set[str]):
    published = {}
    for d in (repo / "policy_api" / "policies").iterdir():
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            published[_run_of((m.get("provenance") or {}).get("checkpoint", ""))] = m["id"]
    runs: dict[str, dict] = {}

    def seen(family: str, ckpt: str, mtime: float):
        run, it = _run_of(ckpt)
        if not run:
            return run, it
        r = runs.setdefault(f"{family}/{run}", {"family": family, "run": run, "iterations": 0, "checkpoint": "", "mtime": mtime})
        if it >= r["iterations"]:
            r.update(iterations=it, checkpoint=ckpt)
        r["mtime"] = max(r["mtime"], mtime)
        return run, it

    items = []
    for f in sorted((root / "out" / "play").glob("*/eval.json")):
        e = json.loads(f.read_text())
        e["checkpoint"] = _ckpt(e.get("checkpoint", ""))
        run, it = seen("g1-ski", e.get("checkpoint", ""), f.stat().st_mtime)
        summary = {k: v for k, v in e.items() if not isinstance(v, (list, dict))}
        items.append({"id": f"benchmark/{f.parent.name}", "tenant": TENANT, "policyId": published.get((run, it), ""),
                      "runId": f"g1-ski/{run}" if run else "", "checkpoint": e.get("checkpoint", ""), "kind": "benchmark",
                      "courseId": "", "summary": json.dumps(summary), "detail": json.dumps({"per_tile": e.get("per_tile")}),
                      "at": ts(f.stat().st_mtime)})
    for f in sorted((root / "out").glob("*/*/*.json")):
        if f.parts[-3] in ("courses", "play", "site", "pose", "climb", "gopro", "diag"):
            continue
        try:
            e = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(e, dict) or not {"course", "checkpoint", "gates_reached"} <= e.keys():
            continue
        e["checkpoint"] = _ckpt(e["checkpoint"])
        run, it = seen("g1-ski", e["checkpoint"], f.stat().st_mtime)
        items.append({"id": f"ride/{f.parts[-3]}/{f.parent.name}/{e['course']}", "tenant": TENANT,
                      "policyId": published.get((run, it), ""), "runId": f"g1-ski/{run}" if run else "",
                      "checkpoint": e["checkpoint"], "kind": "ride", "courseId": e["course"],
                      "summary": json.dumps({k: v for k, v in e.items() if not isinstance(v, (list, dict))}),
                      "detail": json.dumps({k: v for k, v in e.items() if isinstance(v, (list, dict))}), "at": ts(f.stat().st_mtime)})
    for f in sorted((hike_root / "out" / "hike" / "bench").glob("*.json")):
        e = json.loads(f.read_text())
        ckpt = _ckpt((e.get("checkpoint") or e.get("ckpt") or "") if isinstance(e, dict) else "")
        run, it = seen("g1-hike", ckpt, f.stat().st_mtime)
        items.append({"id": f"hike-bench/{f.stem}", "tenant": TENANT, "policyId": "", "runId": f"g1-hike/{run}" if run else "",
                      "checkpoint": ckpt, "kind": "hike", "courseId": "",
                      "summary": json.dumps({k: v for k, v in e.items() if not isinstance(v, (list, dict))} if isinstance(e, dict) else {}),
                      "detail": json.dumps(e), "at": ts(f.stat().st_mtime)})
    for f in sorted((hike_root / "out" / "hike" / "runs").glob("*/*/run.json")):
        e = json.loads(f.read_text())
        ckpt = _ckpt((e.get("checkpoint") or e.get("ckpt") or "") if isinstance(e, dict) else "")
        run, it = seen("g1-hike", ckpt, f.stat().st_mtime)
        items.append({"id": f"hike-run/{f.parts[-3]}/{f.parent.name}", "tenant": TENANT, "policyId": "",
                      "runId": f"g1-hike/{run}" if run else "", "checkpoint": ckpt, "kind": "hike", "courseId": f.parts[-3],
                      "summary": json.dumps({k: v for k, v in e.items() if not isinstance(v, (list, dict))} if isinstance(e, dict) else {}),
                      "detail": json.dumps({k: v for k, v in e.items() if isinstance(v, (list, dict))} if isinstance(e, dict) else e),
                      "at": ts(f.stat().st_mtime)})
    if only:
        items = [i for i in items if any(o in i["id"] for o in only)]
    parallel(lambda i: db.call("upsert_evaluation", i), items)
    for rid, r in runs.items():
        y, mo, dd, hh, mi, ss = (int(x) for x in re.match(r"(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)(\d\d)", r["run"]).groups())
        started = time.mktime((y, mo, dd, hh, mi, ss, 0, 0, -1))
        pid = next((p for (run, _), p in published.items() if run == r["run"]), "")
        db.call("upsert_train_run", {"id": rid, "tenant": TENANT, "policyId": pid, "trainingSet": r["family"], "hardware": "",
                                     "status": "done", "iterations": r["iterations"], "checkpoint": r["checkpoint"],
                                     "metrics": "{}", "benchmark": "{}",
                                     "notes": "Started at is the run directory's name read as local time; iterations is the newest checkpoint evaluated.",
                                     "startedAt": ts(started)})
    print(f"  evals: {len(items)} evaluations, {len(runs)} training runs", flush=True)


def load_sets(db: Db, root: Path, hike_root: Path, only: set[str]):
    courses = sorted(d.name for d in (root / "out" / "courses").iterdir() if (d / "stats.json").exists())
    hikes = sorted(d.name for d in (hike_root / "out" / "hike" / "courses").iterdir() if (d / "route.json").exists())
    clips = [f"sim/hike-clip/{n}" for n in np.load(hike_root / "out" / "hike" / "clips.npz")["names"]]
    db.call("upsert_training_set", {
        "id": "g1-ski", "tenant": TENANT, "name": "G1 skiing", "activity": "ski", "robot": "unitree_g1_29dof",
        "description": "The tile mosaics the ski policies train on (v1 to v4), the World Cup reference poses they hold, "
                       "and the courses they are ridden on to evaluate.",
        "courses": courses, "terrains": ["ski-tiles", "ski-tiles-v2", "ski-tiles-v3", "ski-tiles-v4"], "recordings": [],
        "referencePoses": ["ski/neutral", "ski/left_turn", "ski/right_turn", "ski/transition"],
        "meta": json.dumps({"trainer": "train/ski_env.py + rsl_rl PPO", "benchmark": "out/terrains, 24 tiles, 12 s per run"})})
    db.call("upsert_training_set", {
        "id": "g1-hike", "tenant": TENANT, "name": "G1 hiking", "activity": "hike", "robot": "unitree_g1_29dof",
        "description": "Lidar trail tiles from Ann Arbor parks, the walking clips the policy imitates, and the trail "
                       "courses it is evaluated on.",
        "courses": hikes, "terrains": ["hike-tiles"], "recordings": clips, "referencePoses": [],
        "meta": json.dumps({"trainer": "train/hike_env.py + rsl_rl PPO on Modal", "curriculum": "+-5 levels over the tile mosaic"})})
    print("  sets: g1-ski, g1-hike", flush=True)


LOADERS = ["courses", "far", "trails", "hike", "tiles", "video", "gopro", "clips", "poses", "policies", "evals", "sets"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", choices=LOADERS + [[]], default=[], help="default: all of them")
    ap.add_argument("--server", default="local", help="local, maincloud or a URL")
    ap.add_argument("--db", default="ground-truth")
    ap.add_argument("--root", type=Path, default=REPO, help="checkout whose out/ and data/ hold the data")
    ap.add_argument("--hike-root", type=Path, default=None, help="checkout whose out/hike holds the hike data (default: --root)")
    ap.add_argument("--only", nargs="*", default=[], help="ids to load within a dataset, e.g. kitzbuhel-streif")
    a = ap.parse_args()
    db = Db(SERVERS.get(a.server, a.server), a.db, cli_token())
    hike_root = a.hike_root or a.root
    if not db.sql("SELECT id FROM public_tenant WHERE id = 'hazard-intelligence'"):
        sys.exit("no hazard-intelligence tenant yet: run `spacetime call <db> claim_platform_admin` once")
    only = set(a.only)
    t0 = time.time()
    for name in a.datasets or LOADERS:
        print(f"{name}:", flush=True)
        if name == "courses": load_courses(db, a.root, only)
        elif name == "far": load_far(db, REPO, only)
        elif name == "trails": load_trails(db, a.root, only)
        elif name == "hike": load_hike(db, hike_root, only)
        elif name == "tiles": load_tiles(db, a.root, hike_root, only)
        elif name == "video": load_video(db, a.root, only)
        elif name == "gopro": load_gopro(db, a.root, only)
        elif name == "clips": load_clips(db, hike_root, only)
        elif name == "poses": load_poses(db, a.root, only)
        elif name == "policies": load_policies(db, REPO, only)
        elif name == "evals": load_evals(db, a.root, hike_root, REPO, only)
        elif name == "sets": load_sets(db, a.root, hike_root, only)
    print(f"done: {db.calls} reducer calls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
