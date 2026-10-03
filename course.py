"""Any ski resort -> a real slope in MuJoCo.

    build_course(resort, run=None) -> Path        writes out/courses/<slug>/
    make_scene(course_dir, robot_xml) -> Path      writes a MuJoCo scene XML into that directory

Runs come from OpenStreetMap (Nominatim geocode + Overpass, ODbL). Elevation comes from AWS Terrain
Tiles (terrarium PNGs). Every download is cached under data/cache.

Course directory contents (all lengths in meters, angles in radians unless the key says _deg):
    elevation.npy    float32 [nrow, ncol], snow surface z in the world frame (see GRID_CONVENTION)
    terrain.bin      the same grid as a MuJoCo custom-format hfield (int32 nrow, int32 ncol, float32 data)
    grid.json        origin lat/lon, z datum, x0/y0/cell, hfield size + geom pos, the convention text
    run.json         full stitched run and the course section, as local polylines [x, y, z, s]
    gates.json       giant slalom gates (poles, panels, normals)
    start_pose.json  start point on the snow, downhill yaw, slope, surface-aligned root quaternion
    stats.json       run and course stats, OSM ids, data sources
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import mujoco
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
COURSES = ROOT / "out" / "courses"
DEFAULT_ROBOT_XML = ROOT / "assets" / "unitree_g1" / "g1.xml"

UA = "mhacks-ski-course/0.1 (hackathon prototype; python-requests)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# G1 is ~1.32 m tall vs a ~1.8 m giant slalom racer; gate geometry scales by this.
ROBOT_SCALE = 0.73
# Human-scale giant slalom geometry (m). Spacing and offset give turn radii in the usual GS range.
GS = dict(spacing=26.0, offset=3.5, gate_width=5.0, panel_width=0.75, panel_height=0.5,
          panel_bottom=1.0, pole_height=1.8, pole_radius=0.016, first_gate=15.0, end_margin=20.0,
          finish_half_width=5.0)
DIFFICULTY_RANK = {"novice": 0, "easy": 1, "intermediate": 2, "advanced": 3, "expert": 4, "freeride": 5,
                   "extreme": 6}

GRID_CONVENTION = (
    "elevation.npy is float32 [nrow, ncol]. grid[r, c] is the snow surface z (m) at x = x0 + c*cell, "
    "y = y0 + r*cell. World frame: x = east, y = north, z = up, origin at the course start (lat0, lon0), "
    "z = MSL elevation - z_datum. Row 0 is the southern edge and rows increase northward; col 0 is the "
    "western edge and cols increase eastward. Between vertices the surface is two triangles per cell, split "
    "along the diagonal from (r, c) to (r+1, c+1): with u = frac((x - x0)/cell), v = frac((y - y0)/cell), "
    "z00 = grid[r, c], z10 = grid[r, c+1], z01 = grid[r+1, c], z11 = grid[r+1, c+1]: "
    "if u >= v: z = z00 + (z10 - z00)*u + (z11 - z10)*v, else: z = z00 + (z11 - z01)*u + (z01 - z00)*v. "
    "This is exactly MuJoCo's hfield triangulation (checked with mj_ray; see course.surface_z). "
    "terrain.bin stores the same array row-major (row 0 first) in MuJoCo's custom hfield format; MuJoCo "
    "normalizes it to [0, 1] on load, and the hfield geom with pos = hfield_geom_pos and "
    "size = hfield_size reproduces grid exactly at the vertices."
)


# --------------------------------------------------------------------------------------------- utils

def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def slugify(*parts: str | None) -> str:
    s = "-".join(_fold(p) for p in parts if p)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:80] or "course"


_session: requests.Session | None = None


def _http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = UA
    return _session


def _cached_json(kind: str, key: str, fetch):
    path = CACHE / kind / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


_last_nominatim = 0.0


def geocode(name: str) -> dict:
    """Nominatim search (cached, at most 1 request per second as its usage policy asks)."""
    def fetch():
        global _last_nominatim
        wait = 1.1 - (time.time() - _last_nominatim)
        if wait > 0:
            time.sleep(wait)
        r = _http().get(NOMINATIM, params={"q": name, "format": "jsonv2", "limit": 5}, timeout=30)
        _last_nominatim = time.time()
        r.raise_for_status()
        return r.json()

    hits = _cached_json("nominatim", name.strip().lower(), fetch)
    if not hits:
        raise LookupError(f"Nominatim found nothing for {name!r}")
    h = hits[0]
    return {"lat": float(h["lat"]), "lon": float(h["lon"]), "display_name": h.get("display_name", name),
            "osm": f'{h.get("osm_type")}/{h.get("osm_id")}', "kind": f'{h.get("category")}/{h.get("type")}'}


def overpass(query: str) -> dict:
    def fetch():
        last = None
        for attempt in range(4):
            for url in OVERPASS:
                try:
                    r = _http().post(url, data={"data": query}, timeout=180)
                    if r.status_code == 200:
                        return r.json()
                    last = f"{url} HTTP {r.status_code}: {r.text[:200]}"
                except requests.RequestException as e:
                    last = f"{url}: {e}"
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Overpass failed: {last}")

    return _cached_json("overpass", query, fetch)


def piste_query(lat=None, lon=None, radius=None, bbox=None) -> str:
    """Overpass QL for downhill pistes (ways and piste route relations) around a point or in a s,w,n,e bbox."""
    if bbox is not None:
        flt = "({:.6f},{:.6f},{:.6f},{:.6f})".format(*bbox)
    else:
        flt = f"(around:{radius:.0f},{lat:.6f},{lon:.6f})"
    return (f'[out:json][timeout:120];(way["piste:type"="downhill"]{flt};'
            f'relation["piste:type"="downhill"]{flt};);out geom;')


class LocalFrame:
    """Equirectangular east/north meters around (lat0, lon0) with WGS84 radii. Fine over a few km."""

    def __init__(self, lat0: float, lon0: float):
        a, f = 6378137.0, 1 / 298.257223563
        e2 = f * (2 - f)
        s = math.sin(math.radians(lat0))
        w = 1 - e2 * s * s
        n_rad = a / math.sqrt(w)
        m_rad = a * (1 - e2) / w ** 1.5
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1) * n_rad * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * m_rad

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_latlon(self, x, y):
        return self.lat0 + np.asarray(y) / self.ky, self.lon0 + np.asarray(x) / self.kx


class Terrain:
    """AWS Terrain Tiles (terrarium encoding), bilinear on pixel centers. elev = R*256 + G + B/256 - 32768."""

    def __init__(self, zoom: int = 15):
        self.zoom = zoom
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    def tile(self, tx: int, ty: int) -> np.ndarray:
        key = (tx, ty)
        if key not in self._tiles:
            path = CACHE / "tiles" / "terrarium" / str(self.zoom) / str(tx) / f"{ty}.png"
            if not path.exists():
                url = TILE_URL.format(z=self.zoom, x=tx, y=ty)
                for attempt in range(4):
                    r = _http().get(url, timeout=60)
                    if r.status_code == 200:
                        break
                    time.sleep(2 * (attempt + 1))
                r.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
            rgb = np.asarray(Image.open(BytesIO(path.read_bytes())).convert("RGB"), dtype=np.float64)
            self._tiles[key] = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        return self._tiles[key]

    def sample(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        shape = lat.shape
        lat, lon = lat.ravel(), lon.ravel()
        n = 2 ** self.zoom * 256
        fx = (lon + 180.0) / 360.0 * n - 0.5
        fy = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n - 0.5
        ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        tx0, tx1 = ix.min() // 256, (ix.max() + 1) // 256
        ty0, ty1 = iy.min() // 256, (iy.max() + 1) // 256
        mosaic = np.empty(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256))
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = self.tile(tx, ty)
        lx, ly = ix - tx0 * 256, iy - ty0 * 256
        u, v = fx - ix, fy - iy
        z = (mosaic[ly, lx] * (1 - u) * (1 - v) + mosaic[ly, lx + 1] * u * (1 - v)
             + mosaic[ly + 1, lx] * (1 - u) * v + mosaic[ly + 1, lx + 1] * u * v)
        return z.reshape(shape)


def surface_z(grid: np.ndarray, x0: float, y0: float, cell: float, x, y) -> np.ndarray:
    """Height of the triangulated surface MuJoCo builds from this grid (see GRID_CONVENTION)."""
    nrow, ncol = grid.shape
    u = (np.asarray(x, dtype=np.float64) - x0) / cell
    v = (np.asarray(y, dtype=np.float64) - y0) / cell
    c = np.clip(np.floor(u).astype(np.int64), 0, ncol - 2)
    r = np.clip(np.floor(v).astype(np.int64), 0, nrow - 2)
    fu, fv = np.clip(u - c, 0, 1), np.clip(v - r, 0, 1)
    g = grid.astype(np.float64)
    z00, z10, z01, z11 = g[r, c], g[r, c + 1], g[r + 1, c], g[r + 1, c + 1]
    return np.where(fu >= fv, z00 + (z10 - z00) * fu + (z11 - z10) * fv,
                    z00 + (z11 - z01) * fu + (z01 - z00) * fv)


def bilinear_z(grid: np.ndarray, x0: float, y0: float, cell: float, x, y) -> np.ndarray:
    nrow, ncol = grid.shape
    u = (np.asarray(x, dtype=np.float64) - x0) / cell
    v = (np.asarray(y, dtype=np.float64) - y0) / cell
    c = np.clip(np.floor(u).astype(np.int64), 0, ncol - 2)
    r = np.clip(np.floor(v).astype(np.int64), 0, nrow - 2)
    fu, fv = np.clip(u - c, 0, 1), np.clip(v - r, 0, 1)
    g = grid.astype(np.float64)
    return (g[r, c] * (1 - fu) * (1 - fv) + g[r, c + 1] * fu * (1 - fv)
            + g[r + 1, c] * (1 - fu) * fv + g[r + 1, c + 1] * fu * fv)


# ------------------------------------------------------------------------------------------- pistes

@dataclass
class Segment:
    way_id: int
    name: str | None
    difficulty: str | None
    ref: str | None
    latlon: np.ndarray  # (N, 2)
    area: bool


def parse_segments(osm: dict) -> list[Segment]:
    """Downhill piste ways, with names/difficulty filled in from piste route relations when the way has none."""
    ways: dict[int, Segment] = {}
    rel_info: dict[int, tuple[str | None, str | None, str | None]] = {}
    rel_geom: dict[int, np.ndarray] = {}

    def latlon_of(geom):
        pts = [(p["lat"], p["lon"]) for p in geom or [] if p]
        return np.array(pts, dtype=np.float64).reshape(-1, 2)

    for el in osm.get("elements", []):
        t = el.get("tags", {})
        if el["type"] == "way" and el.get("geometry"):
            ll = latlon_of(el["geometry"])
            if len(ll) < 2:
                continue
            ways[el["id"]] = Segment(el["id"], t.get("piste:name") or t.get("name"), t.get("piste:difficulty"),
                                     t.get("piste:ref") or t.get("ref"), ll, t.get("area") == "yes")
        elif el["type"] == "relation":
            info = (t.get("piste:name") or t.get("name"), t.get("piste:difficulty"), t.get("piste:ref") or t.get("ref"))
            for m in el.get("members", []):
                if m.get("type") != "way":
                    continue
                if info[0] and m["ref"] not in rel_info:
                    rel_info[m["ref"]] = info
                if m.get("geometry"):
                    rel_geom.setdefault(m["ref"], latlon_of(m["geometry"]))

    for wid, (name, diff, ref) in rel_info.items():
        if wid in ways:
            s = ways[wid]
            s.name = s.name or name
            s.difficulty = s.difficulty or diff
            s.ref = s.ref or ref
        elif wid in rel_geom and len(rel_geom[wid]) >= 2:
            ways[wid] = Segment(wid, name, diff, ref, rel_geom[wid], False)
    return list(ways.values())


def _poly_len(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def stitch(polys: list[np.ndarray], tol: float = 30.0) -> list[tuple[np.ndarray, list[int]]]:
    """Greedily chain polylines (meters) whose endpoints lie within tol. Returns [(chain_xy, member indices)]."""
    left = list(range(len(polys)))
    out = []
    while left:
        seed = max(left, key=lambda k: _poly_len(polys[k]))
        left.remove(seed)
        chain, members = polys[seed].copy(), [seed]
        while True:
            best = None
            for k in left:
                p = polys[k]
                for at_tail in (True, False):
                    end = chain[-1] if at_tail else chain[0]
                    for rev in (False, True):
                        q = p[::-1] if rev else p
                        d = float(np.linalg.norm((q[0] if at_tail else q[-1]) - end))
                        if d <= tol and (best is None or d < best[0]):
                            best = (d, k, at_tail, rev)
            if best is None:
                break
            d, k, at_tail, rev = best
            q = polys[k][::-1] if rev else polys[k]
            if at_tail:
                chain = np.vstack([chain, q[1:] if d < 0.5 else q])
            else:
                chain = np.vstack([q[:-1] if d < 0.5 else q, chain])
            left.remove(k)
            members.append(k)
        out.append((chain, members))
    return out


def resample(xy: np.ndarray, ds: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-6]
    xy = xy[keep]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    n = max(2, int(s[-1] // ds) + 1)
    si = np.linspace(0.0, s[-1], n)
    return np.c_[np.interp(si, s, xy[:, 0]), np.interp(si, s, xy[:, 1])], si


def smooth_polyline(xy: np.ndarray, window_m: float = 30.0, ds: float = 2.0) -> np.ndarray:
    k = max(1, int(round(window_m / ds)) | 1)
    pad = k // 2
    p = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.c_[np.convolve(p[:, 0], ker, "valid"), np.convolve(p[:, 1], ker, "valid")]


@dataclass
class Run:
    name: str
    names: list[str]
    difficulty: str | None
    way_ids: list[int]
    latlon: np.ndarray  # stitched, oriented top -> bottom
    length: float
    z_top: float
    z_bottom: float
    n_segments: int
    n_chains: int

    @property
    def drop(self) -> float:
        return self.z_top - self.z_bottom


def _best_chain(segs: list[Segment], frame: LocalFrame, terrain: Terrain, tol: float = 30.0):
    polys = [np.c_[frame.to_xy(s.latlon[:, 0], s.latlon[:, 1])] for s in segs]
    chains = stitch(polys, tol)
    best = None
    for xy, members in chains:
        lat, lon = frame.to_latlon(xy[:, 0], xy[:, 1])
        z = terrain.sample(lat, lon)
        if z[-1] > z[0]:
            xy, z, lat, lon = xy[::-1], z[::-1], lat[::-1], lon[::-1]
        drop = float(z[0] - z[-1])
        if best is None or drop > best[0]:
            best = (drop, np.c_[lat, lon], float(z[0]), float(z[-1]), _poly_len(xy), members)
    return best, len(chains)


def find_pistes(resort: str, near: tuple[float, float] | None = None, radius: float = 6000.0,
                bbox: tuple[float, float, float, float] | None = None):
    """Geocode the resort (unless near/bbox given) and fetch its downhill pistes from Overpass."""
    if bbox is not None:
        center = {"lat": (bbox[0] + bbox[2]) / 2, "lon": (bbox[1] + bbox[3]) / 2, "display_name": resort,
                  "osm": None, "kind": "bbox"}
        q = piste_query(bbox=bbox)
    else:
        center = ({"lat": near[0], "lon": near[1], "display_name": resort, "osm": None, "kind": "near"}
                  if near else geocode(resort))
        q = piste_query(center["lat"], center["lon"], radius)
    segs = parse_segments(overpass(q))
    return center, segs


def list_runs(resort: str, near=None, radius: float = 6000.0, bbox=None, zoom: int = 13) -> tuple[dict, list[Run]]:
    """Every named run (plus connected unnamed groups), stitched and oriented, with DEM stats at a coarse zoom."""
    center, segs = find_pistes(resort, near, radius, bbox)
    frame = LocalFrame(center["lat"], center["lon"])
    terrain = Terrain(zoom)
    groups: dict[str, list[Segment]] = defaultdict(list)
    unnamed = []
    for s in segs:
        if s.area:
            continue
        if s.name:
            groups[s.name].append(s)
        elif s.ref:
            groups[f"ref {s.ref}"].append(s)
        else:
            unnamed.append(s)
    if unnamed:
        polys = [np.c_[frame.to_xy(s.latlon[:, 0], s.latlon[:, 1])] for s in unnamed]
        for k, (_, members) in enumerate(stitch(polys, 30.0)):
            groups[f"unnamed #{k + 1}"] = [unnamed[i] for i in members]
    runs = []
    for name, ss in groups.items():
        best, nchains = _best_chain(ss, frame, terrain)
        if best is None:
            continue
        drop, latlon, ztop, zbot, length, members = best
        diffs = Counter(s.difficulty for s in ss if s.difficulty)
        runs.append(Run(name, [name], diffs.most_common(1)[0][0] if diffs else None,
                        [ss[i].way_id for i in members], latlon, length, ztop, zbot, len(ss), nchains))
    runs.sort(key=lambda r: -r.drop)
    return center, runs


def pick_run(resort: str, run: str | None = None, near=None, radius: float = 6000.0, bbox=None,
             zoom: int = 15) -> tuple[dict, Run]:
    """run is a case- and accent-insensitive regex. All names that match are stitched together, which joins
    pistes mapped as numbered sections (e.g. 'Stelvio 1'..'Stelvio 4'). Default: the run with the largest
    vertical drop among intermediate/advanced/expert."""
    center, segs = find_pistes(resort, near, radius, bbox)
    frame = LocalFrame(center["lat"], center["lon"])
    lines = [s for s in segs if not s.area]
    if run:
        pat = re.compile(_fold(run))
        chosen = [s for s in lines if s.name and pat.search(_fold(s.name))]
        if not chosen:
            names = sorted({s.name for s in lines if s.name})
            raise LookupError(f"No piste matching {run!r} near {center['display_name']}. Names: {names}")
        terrain = Terrain(zoom)
        best, nchains = _best_chain(chosen, frame, terrain)
        drop, latlon, ztop, zbot, length, members = best
        used = [chosen[i] for i in members]
        diffs = Counter(s.difficulty for s in used if s.difficulty)
        names = list(dict.fromkeys(s.name for s in used))
        return center, Run(" + ".join(names), names, diffs.most_common(1)[0][0] if diffs else None,
                           [s.way_id for s in used], latlon, length, ztop, zbot, len(chosen), nchains)
    center, runs = list_runs(resort, near, radius, bbox, zoom=13)
    ok = [r for r in runs if r.difficulty in ("intermediate", "advanced", "expert") and not r.name.startswith("unnamed")]
    if not ok:
        ok = runs
    if not ok:
        raise LookupError(f"No downhill pistes near {center['display_name']}")
    return pick_run(resort, "^" + re.escape(_fold(ok[0].name)) + "$", near, radius, bbox, zoom)


# ------------------------------------------------------------------------------------------- course

def _tangent(xy: np.ndarray, s: np.ndarray, at: float, half: float = 4.0) -> np.ndarray:
    a = np.array([np.interp(at - half, s, xy[:, 0]), np.interp(at - half, s, xy[:, 1])])
    b = np.array([np.interp(at + half, s, xy[:, 0]), np.interp(at + half, s, xy[:, 1])])
    t = b - a
    return t / np.linalg.norm(t)


def _quat_from_matrix(m: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(m, dtype=np.float64).ravel())
    return q.tolist()


def _r(x, nd=4):
    return [round(float(v), nd) for v in np.ravel(x)]


def set_gates(xy: np.ndarray, s: np.ndarray, zfn, scale: float = ROBOT_SCALE, **over) -> dict:
    """Giant slalom set on the course centerline. Gates alternate left/right of the centerline; the skier
    passes each gate between its turning pole (inner, nearer the centerline) and its outer pole, turning
    around the turning pole. Distances are human GS values times scale."""
    p = {k: v * scale for k, v in GS.items()}
    p.update(over)
    gates = []
    k, sk = 0, p["first_gate"]
    while sk <= s[-1] - p["end_margin"]:
        c = np.array([np.interp(sk, s, xy[:, 0]), np.interp(sk, s, xy[:, 1])])
        t = _tangent(xy, s, sk)
        n = np.array([t[1], -t[0]])  # skier's right when facing downhill
        side = "left" if k % 2 == 0 else "right"
        sign = -1.0 if side == "left" else 1.0
        turn = c + sign * p["offset"] * n
        outer = turn + sign * p["gate_width"] * n
        pole_xy = [turn, turn + sign * p["panel_width"] * n, outer - sign * p["panel_width"] * n, outer]
        poles = [[*q, float(zfn(q[0], q[1]))] for q in pole_xy]
        panels = []
        for a, b in ((poles[0], poles[1]), (poles[2], poles[3])):
            mid = (np.array(a) + np.array(b)) / 2
            mid[2] += p["panel_bottom"] + p["panel_height"] / 2
            panels.append({"center": _r(mid), "normal": _r([-t[0], -t[1], 0.0]),
                           "width": round(p["panel_width"], 4), "height": round(p["panel_height"], 4)})
        gates.append({"id": k, "s": round(float(sk), 3), "side": side, "color": "red" if k % 2 == 0 else "blue",
                      "turn_direction": "right" if side == "left" else "left",
                      "turn_pole": _r(poles[0]), "outer_pole": _r(poles[3]), "poles": [_r(q) for q in poles],
                      "panel_normal": _r([-t[0], -t[1], 0.0]), "panels": panels})
        k += 1
        sk += p["spacing"]
    se = float(s[-1])
    c = np.array([xy[-1, 0], xy[-1, 1]])
    t = _tangent(xy, s, se - 5.0)
    n = np.array([t[1], -t[0]])
    fin = [c - p["finish_half_width"] * n, c + p["finish_half_width"] * n]
    finish = {"s": round(se, 3), "poles": [_r([*q, float(zfn(q[0], q[1]))]) for q in fin],
              "line_normal": _r([-t[0], -t[1], 0.0])}
    params = {k2: round(float(v), 4) for k2, v in p.items()}
    return {"units": "m", "frame": "course world frame (x east, y north, z up), pole xyz is the pole base on the snow",
            "scale": scale, "params": params,
            "rules": "Pass each gate between turn_pole and outer_pole. side = which side of the centerline the gate "
                     "sits on (as seen skiing downhill). panel_normal faces uphill toward the skier.",
            "gates": gates, "finish": finish}


def build_course(resort: str, run: str | None = None, *, near: tuple[float, float] | None = None,
                 radius: float = 6000.0, bbox=None, cell: float = 2.0, smooth_sigma_m: float = 4.0,
                 course_length: float = 1200.0, from_top: bool = False, max_slope_deg: float = 35.0,
                 margin: float = 60.0,
                 scale: float = ROBOT_SCALE, zoom: int = 15, out_root: Path = COURSES, slug: str | None = None,
                 verbose: bool = True) -> Path:
    """Resort name (+ optional run regex) -> out/courses/<slug>/ with grid, hfield, gates, start pose, stats."""
    t0 = time.time()
    center, run_ = pick_run(resort, run, near, radius, bbox, zoom)
    terrain = Terrain(zoom)

    # full run in a frame at its own top, resampled every 2 m and lightly smoothed
    top = run_.latlon[0]
    f_run = LocalFrame(top[0], top[1])
    xy_raw = np.c_[f_run.to_xy(run_.latlon[:, 0], run_.latlon[:, 1])]
    xy_full, s_full = resample(smooth_polyline(resample(xy_raw, 2.0)[0], 30.0, 2.0), 2.0)
    lat_f, lon_f = f_run.to_latlon(xy_full[:, 0], xy_full[:, 1])
    z_full_msl = terrain.sample(lat_f, lon_f)
    if z_full_msl[-1] > z_full_msl[0]:
        xy_full, z_full_msl, lat_f, lon_f = xy_full[::-1], z_full_msl[::-1], lat_f[::-1], lon_f[::-1]
        s_full = s_full[-1] - s_full[::-1]

    # course section: the course_length window with the most vertical drop whose steepest 10 m stays under
    # max_slope_deg (raw DEM); if no window qualifies, the least steep one. Or simply from the top.
    total = float(s_full[-1])
    k10 = 5  # 10 m at 2 m spacing
    sl10 = np.degrees(np.arctan2(z_full_msl[:-k10] - z_full_msl[k10:], s_full[k10:] - s_full[:-k10]))

    def window_max_slope(i, j):
        seg = sl10[i:max(i + 1, j - k10 + 1)]
        return float(seg.max()) if seg.size else 0.0

    if total <= course_length or from_top:
        i0 = 0
        i1 = int(np.searchsorted(s_full, min(course_length, total)))
        rule = "from top" if from_top else "whole run (shorter than course_length)"
    else:
        cands = []
        for i in range(0, len(s_full), 5):
            j = int(np.searchsorted(s_full, s_full[i] + course_length))
            if j >= len(s_full):
                break
            cands.append((float(z_full_msl[i] - z_full_msl[j]), window_max_slope(i, j), i, j))
        ok = [c for c in cands if c[1] <= max_slope_deg]
        if ok:
            _, _, i0, i1 = max(ok)
            rule = f"max vertical drop with steepest 10 m <= {max_slope_deg} deg"
        else:
            _, _, i0, i1 = min(cands, key=lambda c: c[1])
            rule = f"no window under {max_slope_deg} deg; least steep window"
    i1 = min(i1, len(s_full) - 1)
    selection = {"rule": rule, "max_slope_cap_deg": max_slope_deg, "s_start": round(float(s_full[i0]), 1),
                 "s_end": round(float(s_full[i1]), 1),
                 "window_steepest_10m_raw_dem_deg": round(window_max_slope(i0, i1), 2)}
    lat0, lon0 = float(lat_f[i0]), float(lon_f[i0])

    # course frame: origin at the start
    fr = LocalFrame(lat0, lon0)
    x_all, y_all = fr.to_xy(lat_f, lon_f)
    xy_all = np.c_[x_all, y_all]
    xy_c, s_c = resample(xy_all[i0:i1 + 1], 2.0)

    # grid over the course section bbox + margin
    x0 = math.floor((xy_c[:, 0].min() - margin) / cell) * cell
    y0 = math.floor((xy_c[:, 1].min() - margin) / cell) * cell
    x1 = math.ceil((xy_c[:, 0].max() + margin) / cell) * cell
    y1 = math.ceil((xy_c[:, 1].max() + margin) / cell) * cell
    ncol = int(round((x1 - x0) / cell)) + 1
    nrow = int(round((y1 - y0) / cell)) + 1
    if nrow * ncol > 6_000_000:
        raise ValueError(f"grid {nrow}x{ncol} too large; lower course_length or raise cell")
    xx, yy = np.meshgrid(x0 + cell * np.arange(ncol), y0 + cell * np.arange(nrow))  # [r, c]: x = col, y = row
    glat, glon = fr.to_latlon(xx, yy)
    elev = terrain.sample(glat, glon)
    elev_s = gaussian_filter(elev, sigma=smooth_sigma_m / cell, mode="nearest") if smooth_sigma_m > 0 else elev
    z_datum = round(float(surface_z(elev_s, x0, y0, cell, 0.0, 0.0)), 2)
    grid = (elev_s - z_datum).astype(np.float32)

    def zfn(x, y):
        return surface_z(grid, x0, y0, cell, x, y)

    out = Path(out_root) / (slug or slugify(resort, re.sub(r"[\\^$]", "", run) if run else run_.names[0]))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "elevation.npy", grid)
    with open(out / "terrain.bin", "wb") as fh:
        fh.write(struct.pack("<ii", nrow, ncol))
        fh.write(grid.astype("<f4").tobytes())
    zmin, zmax = float(grid.min()), float(grid.max())
    sx, sy = (ncol - 1) * cell / 2, (nrow - 1) * cell / 2
    base = 10.0
    grid_meta = {
        "convention": GRID_CONVENTION,
        "lat0": lat0, "lon0": lon0, "z_datum_msl": z_datum,
        "x0": x0, "y0": y0, "cell": cell, "nrow": nrow, "ncol": ncol,
        "x_range": [x0, x0 + (ncol - 1) * cell], "y_range": [y0, y0 + (nrow - 1) * cell],
        "z_min": zmin, "z_max": zmax,
        "hfield_size": [sx, sy, zmax - zmin, base],
        "hfield_geom_pos": [x0 + sx, y0 + sy, zmin],
        "smoothing_sigma_m": smooth_sigma_m, "dem": f"AWS Terrain Tiles terrarium z{zoom}, bilinear on pixel centers",
        "projection": "equirectangular around (lat0, lon0) with WGS84 meridional/prime-vertical radii",
    }
    (out / "grid.json").write_text(json.dumps(grid_meta, indent=1))

    # polylines (z on the MuJoCo surface where the grid covers it, else raw DEM - datum)
    inside = ((xy_all[:, 0] >= x0) & (xy_all[:, 0] <= x0 + (ncol - 1) * cell)
              & (xy_all[:, 1] >= y0) & (xy_all[:, 1] <= y0 + (nrow - 1) * cell))
    z_all = np.where(inside, zfn(xy_all[:, 0], xy_all[:, 1]), z_full_msl - z_datum)
    z_c = zfn(xy_c[:, 0], xy_c[:, 1])
    (out / "run.json").write_text(json.dumps({
        "frame": "course world frame (x east, y north, z up, m); columns x, y, z, s",
        "full_run": np.round(np.c_[xy_all, z_all, s_full], 3).tolist(),
        "course": np.round(np.c_[xy_c, z_c, s_c], 3).tolist(),
        "course_s_offset_in_full_run": round(float(s_full[i0]), 3),
    }))

    gates = set_gates(xy_c, s_c, zfn, scale)
    (out / "gates.json").write_text(json.dumps(gates, indent=1))

    # start pose: skis flat on the snow, pointing down the course
    t = _tangent(xy_c, s_c, 4.0)
    yaw = math.atan2(t[1], t[0])
    zs = float(zfn(0.0, 0.0))
    z8 = float(zfn(8 * t[0], 8 * t[1]))
    pitch = math.atan2(zs - z8, 8.0)
    gy, gx = np.gradient(grid.astype(np.float64), cell)
    r0, c0 = int(round((0 - y0) / cell)), int(round((0 - x0) / cell))
    nrm = np.array([-gx[r0, c0], -gy[r0, c0], 1.0])
    nrm /= np.linalg.norm(nrm)
    xb = np.array([t[0], t[1], 0.0])
    xb -= xb.dot(nrm) * nrm
    xb /= np.linalg.norm(xb)
    rot = np.c_[xb, np.cross(nrm, xb), nrm]
    start = {"snow_xyz": [0.0, 0.0, round(zs, 4)], "yaw": round(yaw, 6), "yaw_deg": round(math.degrees(yaw), 3),
             "pitch": round(pitch, 6), "slope_deg_along_heading": round(math.degrees(pitch), 3),
             "surface_normal": _r(nrm, 6), "quat_surface_wxyz": _r(_quat_from_matrix(rot), 6),
             "note": "yaw is CCW from +x (east) and points down the course. quat_surface aligns the root x axis "
                     "with the course heading and z with the snow normal (skis flat on the snow). make_scene adds "
                     "a robot-specific root position for that robot under robots/<name>."}
    (out / "start_pose.json").write_text(json.dumps(start, indent=1))

    # stats
    def line_stats(xy, z, s):
        seg2 = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        l2 = float(seg2.sum())
        l3 = float(np.sqrt(seg2 ** 2 + np.diff(z) ** 2).sum())
        drop = float(z[0] - z[-1])
        k = max(1, int(round(10.0 / 2.0)))
        sl = np.degrees(np.arctan2(z[:-k] - z[k:], s[k:] - s[:-k])) if len(z) > k else np.array([0.0])
        return {"length_2d_m": round(l2, 1), "length_3d_m": round(l3, 1), "vertical_drop_m": round(drop, 1),
                "mean_slope_deg": round(math.degrees(math.atan2(drop, l2)), 2),
                "max_slope_10m_deg": round(float(sl.max()), 2)}

    slope_grid = np.degrees(np.arctan(np.hypot(gx, gy)))
    rr = np.clip(np.round((xy_c[:, 1] - y0) / cell).astype(int), 0, nrow - 1)
    cc = np.clip(np.round((xy_c[:, 0] - x0) / cell).astype(int), 0, ncol - 1)
    fall = slope_grid[rr, cc]
    full = line_stats(xy_all, z_full_msl, s_full)
    full.update({"top_elevation_msl_m": round(float(z_full_msl[0]), 1),
                 "bottom_elevation_msl_m": round(float(z_full_msl[-1]), 1)})
    crs = line_stats(xy_c, z_c, s_c)
    crs.update({"start_elevation_msl_m": round(z_datum + float(z_c[0]), 1),
                "end_elevation_msl_m": round(z_datum + float(z_c[-1]), 1),
                "fall_line_slope_on_centerline_deg": {"mean": round(float(fall.mean()), 2),
                                                      "max": round(float(fall.max()), 2)},
                "gates": len(gates["gates"])})
    stats = {
        "resort_query": resort, "run_query": run, "geocode": center,
        "run": {"name": run_.name, "osm_names": run_.names, "difficulty": run_.difficulty,
                "osm_way_ids": run_.way_ids, "segments_matched": run_.n_segments, "chains": run_.n_chains},
        "full_run": full, "course": crs, "course_selection": selection,
        "grid": {"nrow": nrow, "ncol": ncol, "cell_m": cell, "cells": nrow * ncol},
        "sources": {"pistes": "OpenStreetMap via Overpass API (ODbL)",
                    "geocode": "Nominatim (OpenStreetMap)",
                    "elevation": "AWS Terrain Tiles (Mapzen terrarium), see README for attribution"},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_seconds": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    if verbose:
        print(f"[course] {resort} / {run_.name}: run {full['length_2d_m']} m, drop {full['vertical_drop_m']} m; "
              f"course {crs['length_2d_m']} m, drop {crs['vertical_drop_m']} m, {crs['gates']} gates, "
              f"grid {nrow}x{ncol} -> {out}")
    return out


# -------------------------------------------------------------------------------------------- scene

G1_STANCES = {
    # mjlab KNEES_BENT_KEYFRAME from unitree_rl_mjlab g1_constants.py
    "knees_bent": {r".*_hip_pitch_joint": -0.312, r".*_knee_joint": 0.669, r".*_ankle_pitch_joint": -0.363,
                   r".*_elbow_joint": 0.6, r"left_shoulder_roll_joint": 0.2, r"left_shoulder_pitch_joint": 0.2,
                   r"right_shoulder_roll_joint": -0.2, r"right_shoulder_pitch_joint": 0.2},
    # mjlab HOME_KEYFRAME
    "home": {r".*_hip_pitch_joint": -0.1, r".*_knee_joint": 0.3, r".*_ankle_pitch_joint": -0.2,
             r".*_shoulder_pitch_joint": 0.35, r".*_elbow_joint": 0.87, r"left_shoulder_roll_joint": 0.18,
             r"right_shoulder_roll_joint": -0.18},
}
# visual ski: box under each ankle roll link sole (G1 foot capsules bottom out at z = -0.035 in that frame)
SKI = dict(bodies=("left_ankle_roll_link", "right_ankle_roll_link"), center=(0.04, 0.0, -0.035),
           half=(0.5, 0.04, 0.006), rgba=(0.85, 0.12, 0.1, 1.0))


def _set_light_type(light, directional: bool):
    try:
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if directional else mujoco.mjtLightType.mjLIGHT_SPOT
    except AttributeError:
        light.directional = directional


def make_scene(course_dir: Path | str, robot_xml: Path | str = DEFAULT_ROBOT_XML, *, stance: str | dict = "knees_bent",
               skis: bool = True, disable_foot_collision: bool = True, overlays: dict | None = None,
               name: str | None = None, root_body: str = "pelvis") -> Path:
    """Write <course_dir>/<name>.xml: terrain hfield + gates + the robot on visual skis at the start (keyframe
    'start'). overlays={'run': True, 'gates': True, 'tracks': [Nx3 arrays]} adds big visual-only markers in
    geom group 4 for map-scale renders (hidden in the default viewer)."""
    course_dir = Path(course_dir).resolve()
    robot_xml = Path(robot_xml).resolve()
    grid = np.load(course_dir / "elevation.npy")
    gm = json.loads((course_dir / "grid.json").read_text())
    gates = json.loads((course_dir / "gates.json").read_text())
    start = json.loads((course_dir / "start_pose.json").read_text())
    runj = json.loads((course_dir / "run.json").read_text())
    x0, y0, cell = gm["x0"], gm["y0"], gm["cell"]

    def zfn(x, y):
        return surface_z(grid, x0, y0, cell, x, y)

    spec = mujoco.MjSpec.from_file(str(robot_xml))
    robot_mesh_dir = (robot_xml.parent / (spec.meshdir or "")).resolve()
    for m in spec.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = str(robot_mesh_dir / m.file)
    tex_dir = (robot_xml.parent / (spec.texturedir or "")).resolve()
    for tx in spec.textures:
        if tx.file and not os.path.isabs(tx.file):
            tx.file = str(tex_dir / tx.file)
    spec.meshdir = ""
    spec.texturedir = ""
    spec.modelname = f"{course_dir.name}_{robot_xml.stem}"

    # robot edits: visual skis, optional foot collision off
    if skis:
        for bname in SKI["bodies"]:
            b = spec.body(bname)
            if b is None:
                continue
            side = bname.split("_")[0]
            cx, cy, cz = SKI["center"]
            hx, hy, hz = SKI["half"]
            g = b.add_geom(name=f"{side}_ski_visual", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                           pos=[cx, cy, cz - hz], rgba=list(SKI["rgba"]))
            g.contype, g.conaffinity, g.group, g.density = 0, 0, 2, 0.0
            tip = b.add_geom(name=f"{side}_ski_tip_visual", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, hy, hz],
                             pos=[cx + hx + 0.045, cy, cz - hz + 0.016], euler=[0, -0.35, 0], rgba=list(SKI["rgba"]))
            tip.contype, tip.conaffinity, tip.group, tip.density = 0, 0, 2, 0.0
    if disable_foot_collision:
        for g in spec.geoms:
            if g.name and re.fullmatch(r"(left|right)_foot\d+_collision", g.name):
                g.contype, g.conaffinity = 0, 0

    # look
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.quality.shadowsize = 8192
    spec.visual.headlight.ambient = [0.22, 0.22, 0.25]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.map.znear = 0.005
    spec.visual.map.zfar = 1000.0
    spec.stat.extent = 4.0
    spec.stat.center = start["snow_xyz"]
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.45, 0.65, 0.92],
                           rgb2=[0.95, 0.97, 1.0], width=512, height=3072)
    snow_tex = spec.add_texture(name="snow_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.97, 0.98, 1.0],
                                rgb2=[0.92, 0.94, 0.98], width=512, height=512)
    snow = spec.add_material(name="snow", rgba=[1, 1, 1, 1], specular=0.15, shininess=0.2, reflectance=0.0)
    snow.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "snow_grid"
    snow.texrepeat = [gm["hfield_size"][0] / 25.0, gm["hfield_size"][1] / 25.0]  # 50 m checker squares
    del sky, snow_tex

    hpath = course_dir / "terrain.bin"
    spec.add_hfield(name="terrain", file=str(hpath), size=list(gm["hfield_size"]))
    tg = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                 pos=list(gm["hfield_geom_pos"]), material="snow")
    tg.group, tg.condim = 0, 3
    tg.friction = [0.4, 0.005, 0.0001]

    sun = spec.worldbody.add_light(name="sun", pos=[0, 0, 50], dir=[0.5, 0.6, -0.62], diffuse=[0.8, 0.79, 0.75],
                                   specular=[0.1, 0.1, 0.1], ambient=[0.12, 0.12, 0.14])
    _set_light_type(sun, True)
    sun.castshadow = True

    # gates: visual-only poles and panels
    gp = gates["params"]
    col = {"red": [0.86, 0.1, 0.1, 1.0], "blue": [0.1, 0.25, 0.85, 1.0]}
    for gt in gates["gates"]:
        rgba = col[gt["color"]]
        for j, (px, py, pz) in enumerate(gt["poles"]):
            g = spec.worldbody.add_geom(name=f"gate{gt['id']}_pole{j}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                        size=[gp["pole_radius"], gp["pole_height"] / 2, 0],
                                        pos=[px, py, pz + gp["pole_height"] / 2], rgba=rgba)
            g.contype, g.conaffinity, g.group, g.density = 0, 0, 1, 0.0
        for j, pn in enumerate(gt["panels"]):
            nx, ny = pn["normal"][0], pn["normal"][1]
            yaw = math.atan2(ny, nx) - math.pi / 2  # local x along the panel width, local y along the normal
            g = spec.worldbody.add_geom(name=f"gate{gt['id']}_panel{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        size=[pn["width"] / 2, 0.004, pn["height"] / 2], pos=pn["center"],
                                        euler=[0, 0, yaw], rgba=rgba)
            g.contype, g.conaffinity, g.group, g.density = 0, 0, 1, 0.0
    for j, (px, py, pz) in enumerate(gates["finish"]["poles"]):
        g = spec.worldbody.add_geom(name=f"finish_pole{j}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    size=[0.03, 1.0, 0], pos=[px, py, pz + 1.0], rgba=[0.1, 0.1, 0.1, 1])
        g.contype, g.conaffinity, g.group, g.density = 0, 0, 1, 0.0

    # map-scale overlays in group 4
    def capsule_chain(prefix, pts, radius, rgba, stride=1):
        pts = np.asarray(pts)[::stride]
        for j in range(len(pts) - 1):
            a, b = pts[j], pts[j + 1]
            if np.linalg.norm(b - a) < 1e-3:
                continue
            g = spec.worldbody.add_geom(name=f"{prefix}{j}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[radius, 0, 0],
                                        fromto=[*a, *b], rgba=rgba)
            g.contype, g.conaffinity, g.group, g.density = 0, 0, 4, 0.0

    if overlays:
        if overlays.get("run"):
            c = np.array(runj["course"])[:, :3]
            c[:, 2] += 0.6
            capsule_chain("ov_course", c, 0.9, [1.0, 0.75, 0.0, 1.0], stride=5)
        if overlays.get("gates"):
            for gt in gates["gates"]:
                x, y, z = gt["turn_pole"]
                g = spec.worldbody.add_geom(name=f"ov_gate{gt['id']}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                            size=[2.2, 0, 0], pos=[x, y, z + 1.5], rgba=col[gt["color"]])
                g.contype, g.conaffinity, g.group, g.density = 0, 0, 4, 0.0
        for k, tr in enumerate(overlays.get("tracks", [])):
            capsule_chain(f"ov_track{k}_", tr, 0.7, [0.85, 0.1, 0.85, 1.0], stride=1)

    # stance + start placement
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    stance_map = G1_STANCES[stance] if isinstance(stance, str) else stance
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for pat, val in stance_map.items():
            if re.fullmatch(pat, jn):
                qpos[model.jnt_qposadr[j]] = val
    root = model.body(root_body)
    fj = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[j] == root.id]
    if not fj:
        raise ValueError(f"{root_body} has no free joint")
    adr = model.jnt_qposadr[fj[0]]
    quat = np.array(start["quat_surface_wxyz"])
    qpos[adr + 3:adr + 7] = quat
    qpos[adr:adr + 3] = [0.0, 0.0, start["snow_xyz"][2] + 1.0]
    ski_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{b.split('_')[0]}_ski_visual") for b in SKI["bodies"]]
    ski_ids = [i for i in ski_ids if i >= 0]
    foot_ids = [i for i in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_collision")
                and re.search(r"foot\d", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]

    def contact_points():
        pts = []
        if ski_ids:
            for gid in ski_ids:
                hx, hy, hz = model.geom_size[gid]
                rot = data.geom_xmat[gid].reshape(3, 3)
                for sx_ in (-1, 1):
                    for sy_ in (-1, 1):
                        pts.append(data.geom_xpos[gid] + rot @ np.array([sx_ * hx, sy_ * hy, -hz]))
        else:
            for gid in foot_ids:
                pts.append(data.geom_xpos[gid] - np.array([0, 0, model.geom_size[gid][0]]))
        return np.array(pts)

    for _ in range(3):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        pts = contact_points()
        mid = pts[:, :2].mean(axis=0)
        qpos[adr:adr + 2] -= mid  # center the skis over the start point
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        pts = contact_points()
        clearance = pts[:, 2] - zfn(pts[:, 0], pts[:, 1])
        qpos[adr + 2] -= clearance.min() - 0.001
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    pts = contact_points()
    clearance = pts[:, 2] - zfn(pts[:, 0], pts[:, 1])

    spec.add_key(name="start", qpos=qpos.tolist())
    name = name or f"scene_{robot_xml.stem}" + ("_overlay" if overlays else "")
    spec.compile()
    xml = spec.to_xml()
    # make every asset path relative to the scene file so the course directory can move with the repo
    xml = xml.replace(str(hpath), "terrain.bin")
    # to_xml prints 6 significant digits; the terrain must be exact so the hfield matches elevation.npy
    full = lambda v: " ".join(repr(float(x)) for x in v)  # noqa: E731
    xml, n1 = re.subn(r'(<hfield name="terrain"[^>]*?size=")[^"]*(")',
                      lambda mo: mo.group(1) + full(gm["hfield_size"]) + mo.group(2), xml)
    xml, n2 = re.subn(r'(<geom name="terrain"[^>]*?pos=")[^"]*(")',
                      lambda mo: mo.group(1) + full(gm["hfield_geom_pos"]) + mo.group(2), xml)
    assert n1 == 1 and n2 == 1, "terrain hfield/geom not found in written XML"
    xml = xml.replace(str(robot_mesh_dir) + os.sep, os.path.relpath(robot_mesh_dir, course_dir) + "/")
    out = course_dir / f"{name}.xml"
    out.write_text(xml)

    # sanity: the written file loads and matches
    m2 = mujoco.MjModel.from_xml_path(str(out))
    assert m2.nq == model.nq and m2.nkey == 1, "scene reload mismatch"

    if overlays:  # render-only variant; the training scene's record stays authoritative
        return out
    rec = start.setdefault("robots", {})
    rec[robot_xml.stem] = {
        "robot_xml": os.path.relpath(robot_xml, course_dir), "scene_xml": out.name, "keyframe": "start",
        "root_body": root_body, "root_pos": _r(qpos[adr:adr + 3], 5), "root_quat_wxyz": _r(qpos[adr + 3:adr + 7], 6),
        "stance": stance if isinstance(stance, str) else "custom",
        "ski_bottom_clearance_m": {"min": round(float(clearance.min()), 5), "max": round(float(clearance.max()), 5)},
        "skis": "visual only (contype=0, conaffinity=0), main replaces with real ski bodies" if skis else None,
        "foot_collision_disabled": disable_foot_collision,
    }
    (course_dir / "start_pose.json").write_text(json.dumps(start, indent=1))
    return out
