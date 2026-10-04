"""Any trail or road anywhere to an OpenSkiMap style terrain dataset: OpenStreetMap ways plus the best elevation available.

    .venv/bin/python ground.py --place "Ann Arbor, Michigan"                         # trails within 6 km
    .venv/bin/python ground.py --place "Zermatt" --radius 8000
    .venv/bin/python ground.py --lat 42.2806 --lon -83.7483 --radius 3000 --kind roads --name a2-roads
    .venv/bin/python ground.py --bbox 42.27,-83.76,42.30,-83.72 --kind walk           # south, west, north, east

Kinds (OpenStreetMap highway filters):
  trails  paths, tracks, bridleways and unpaved footways
  walk    everything on foot: paths, footways and sidewalks, pedestrian streets, steps, cycleways
  roads   streets and roads for vehicles, motorway down to service road
  all     every highway way

Elevation, best available per point (--dem):
  auto       USGS 3DEP where it has data, AWS Terrain Tiles everywhere else (default)
  3dep       USGS 3DEP only (US): 1 m lidar where flown, else 10 m, fetched as 2 km UTM tiles
  terrarium  AWS Terrain Tiles at zoom 15 only (worldwide: about 5 m pixels, 30 m sources in much of the world)

Writes data/ground/<name>/:
  <kind>.geojson   one feature per OSM way: tags, the park it sits in (trails, walk), an elevation profile
                   every 5 m and slope statistics, like OpenSkiMap runs
  areas.geojson    parks, nature reserves and protected areas holding those ways, with totals (trails, walk)
  meta.json        query, elevation sources used, licenses
Elevation tiles are cached under data/cache (3DEP GeoTIFFs, terrarium PNGs) and reused across runs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from course import CACHE, ROOT, LocalFrame, Terrain, geocode, overpass, slugify

DEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
UNPAVED = "dirt|ground|earth|grass|gravel|fine_gravel|unpaved|compacted|wood|woodchips|mud|sand|rock|pebblestone"
NOT_SIDEWALK = '[footway!~"^(sidewalk|crossing|access_aisle|traffic_island)$"]'
KINDS = {
  "trails": ['[highway~"^(path|track|bridleway)$"][area!=yes]',
             f'[highway=footway]{NOT_SIDEWALK}[surface~"^({UNPAVED})$"]'],
  "walk": ['[highway~"^(path|track|bridleway|footway|pedestrian|steps|cycleway|living_street)$"][area!=yes]'],
  "roads": ['[highway~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|road)(_link)?$"][area!=yes]'],
  "all": ['[highway][area!=yes]'],
}
TAGS = ["name", "ref", "highway", "surface", "smoothness", "tracktype", "sac_scale", "trail_visibility", "incline", "access",
        "bridge", "tunnel", "covered", "layer"]
PROFILE_M = 5        # elevation profile spacing
GRADE_WIN_M = 10     # window for max grade
# Vehicle roads steeper than this over 10 m are almost always the ground under an unmarked bridge, deck or building
# (the steepest public streets in the world are about 35%), so they get flagged rather than trusted.
ROAD_SUSPECT_PCT = 30
FOOT_ONLY = {"path", "track", "bridleway", "footway", "pedestrian", "steps", "cycleway"}


def utm(lat, lon, zone: int):
  """GRS80 to UTM north (Krueger series), vectorized. Matches the 3DEP server's NAD83 UTM to about 1 m."""
  lat = np.asarray(lat, dtype=np.float64); lon = np.asarray(lon, dtype=np.float64)
  a = 6378137.0; f = 1 / 298.257222101; e2 = f * (2 - f); k0 = 0.9996; ep2 = e2 / (1 - e2)
  phi = np.radians(lat); A = np.cos(phi) * np.radians(lon - (zone * 6 - 183))
  N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2); T = np.tan(phi) ** 2; C = ep2 * np.cos(phi) ** 2
  M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
           - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
           + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi) - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
  x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
  y = k0 * (M + N * np.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                   + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
  return x, y


class Tiles:
  """Bilinear sampling over square tiles addressed by global pixel index, with a small LRU so memory stays bounded."""

  def __init__(self, size: int, keep: int):
    self.size = size; self.keep = keep; self.lru: OrderedDict = OrderedDict()

  def load(self, tx: int, ty: int) -> np.ndarray | None:
    raise NotImplementedError

  def tile(self, tx: int, ty: int) -> np.ndarray | None:
    k = (tx, ty)
    if k in self.lru:
      self.lru.move_to_end(k)
    else:
      self.lru[k] = self.load(tx, ty)
      if len(self.lru) > self.keep:
        self.lru.popitem(last=False)
    return self.lru[k]

  def values(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    out = np.full(gx.shape, np.nan)
    tx, ty = gx // self.size, gy // self.size
    key = tx * (1 << 31) + ty
    order = np.argsort(key, kind="stable")
    uniq, start = np.unique(key[order], return_index=True)
    bounds = list(start) + [len(order)]
    for k in range(len(uniq)):
      idx = order[bounds[k]:bounds[k + 1]]
      t = self.tile(int(tx[idx[0]]), int(ty[idx[0]]))
      if t is not None:
        out[idx] = t[gy[idx] - ty[idx] * self.size, gx[idx] - tx[idx] * self.size]
    return out

  def bilinear(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    ix = np.floor(fx).astype(np.int64); iy = np.floor(fy).astype(np.int64); u = fx - ix; v = fy - iy
    return (self.values(ix, iy) * (1 - u) * (1 - v) + self.values(ix + 1, iy) * u * (1 - v)
            + self.values(ix, iy + 1) * (1 - u) * v + self.values(ix + 1, iy + 1) * u * v)


class Terrarium(Tiles):
  """AWS Terrain Tiles, zoom 15, via course.Terrain's cached downloader. Rows run north to south."""
  Z = 15

  def __init__(self):
    super().__init__(256, keep=1024)
    self.src = Terrain(self.Z)

  def load(self, tx, ty):
    t = self.src.tile(tx, ty).astype(np.float32)
    self.src._tiles.pop((tx, ty), None)       # we keep our own bounded cache
    return t

  def sample(self, lat, lon):
    n = 2 ** self.Z * 256
    fx = (np.asarray(lon) + 180.0) / 360.0 * n - 0.5
    fy = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n - 0.5
    return self.bilinear(fx, fy)


class Dep3(Tiles):
  """USGS 3DEP bare earth, 1 m pixels, 2 km tiles on the NAD83 UTM grid. Rows flipped so they run south to north."""
  TILE = 2000          # 3DEP returns HTTP 500 for exports of 3000 px and up

  def __init__(self, zone: int):
    super().__init__(self.TILE, keep=12)
    self.zone = zone; self.epsg = 26900 + zone
    self.dir = CACHE / "3dep" / f"utm{zone}n"; self.dir.mkdir(parents=True, exist_ok=True)

  def path(self, tx, ty) -> Path:
    return self.dir / f"{tx * self.TILE}_{ty * self.TILE}.tif"

  def fetch(self, tx: int, ty: int) -> bool:
    p = self.path(tx, ty)
    if p.exists():
      return True
    xa, ya = tx * self.TILE, ty * self.TILE
    q = urllib.parse.urlencode(dict(bbox=f"{xa},{ya},{xa + self.TILE},{ya + self.TILE}", bboxSR=self.epsg, imageSR=self.epsg,
                                    size=f"{self.TILE},{self.TILE}", format="tiff", pixelType="F32", noData=-9999,
                                    interpolation="RSP_BilinearInterpolation", f="image"))
    for attempt in range(5):
      try:
        raw = urllib.request.urlopen(f"{DEP}/exportImage?{q}", timeout=300).read()
        tmp = p.with_suffix(".part"); tmp.write_bytes(raw); tmp.rename(p)
        return True
      except Exception as e:   # the service throws transient 500s under load
        print(f"  3DEP tile {tx},{ty} failed ({type(e).__name__}), retrying", flush=True)
        time.sleep(5 * (attempt + 1))
    print(f"  3DEP tile {tx},{ty} skipped after 5 failures, terrarium fills it", flush=True)
    return False

  def load(self, tx, ty):
    if not self.fetch(tx, ty):
      return None
    z = np.array(Image.open(self.path(tx, ty)), dtype=np.float32)[::-1]
    z[z < -1000] = np.nan
    return z

  def needed(self, x, y) -> list[tuple[int, int]]:
    ix = np.floor(x - 0.5).astype(np.int64); iy = np.floor(y - 0.5).astype(np.int64)
    s = set()
    for dx in (0, 1):
      for dy in (0, 1):
        s |= set(zip(((ix + dx) // self.TILE).tolist(), ((iy + dy) // self.TILE).tolist()))
    return sorted(s)

  def sample_xy(self, x, y):
    return self.bilinear(np.asarray(x) - 0.5, np.asarray(y) - 0.5)   # pixel centres sit at +0.5 m


# Rough extents 3DEP covers: lower 48, Alaska (both sides of 180), Hawaii, Puerto Rico and the Virgin Islands.
US_BOXES = [(24, -125, 50, -66), (51, -180, 72, -129), (51, 172, 54, 180), (18, -161, 23, -154), (17.5, -68, 18.6, -64)]


def in_us(lat: float, lon: float) -> bool:
  return any(s <= lat <= n and w <= lon <= e for s, w, n, e in US_BOXES)


def dep3_resolution(lat: float, lon: float) -> float | None:
  """3DEP's source resolution at a point, for the record only. None if it has no data or does not answer."""
  g = json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}})
  u = f"{DEP}/getSamples?" + urllib.parse.urlencode(dict(geometry=g, geometryType="esriGeometryPoint", returnFirstValueOnly="true", f="json"))
  for attempt in range(3):
    try:
      s = json.load(urllib.request.urlopen(u, timeout=60))["samples"][0]
      return float(s["resolution"]) if s.get("value") not in (None, "NoData") else None
    except Exception:    # transient 500s and timeouts are common; this number is informational
      time.sleep(3 * (attempt + 1))
  return None


class Elevation:
  def __init__(self, mode: str, lat0: float, lon0: float, max_dep_tiles: int):
    self.mode = mode; self.max_dep_tiles = max_dep_tiles; self.ter = Terrarium(); self.dep = None; self.dep_res = None
    # Inside the US always go for 3DEP tiles. A flaky check must not decide the source: points whose tile
    # fails to download, or lands outside 3DEP's data, fall back to terrarium one by one in sample().
    if mode in ("auto", "3dep") and in_us(lat0, lon0):
      self.dep = Dep3(int((lon0 + 180) // 6) + 1)
      self.dep_res = dep3_resolution(lat0, lon0)
    elif mode == "3dep":
      raise SystemExit("3DEP covers the US only; use --dem auto or --dem terrarium")

  def sample(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.full(lat.shape, np.nan); src = np.zeros(lat.shape, np.int8)    # 1 = 3DEP, 2 = terrarium
    if self.dep is not None:
      x, y = utm(lat, lon, self.dep.zone)
      need = self.dep.needed(x, y)
      if len(need) > self.max_dep_tiles:
        print(f"  {len(need)} 3DEP tiles needed, over --max-3dep-tiles {self.max_dep_tiles}: using terrarium instead", flush=True)
      else:
        print(f"  {len(need)} 3DEP tiles of 2 km ({sum(not self.dep.path(*t).exists() for t in need)} to download)", flush=True)
        with ThreadPoolExecutor(2) as ex:
          list(ex.map(lambda t: self.dep.fetch(*t), need))
        order = np.lexsort((x // Dep3.TILE, y // Dep3.TILE))   # tile by tile, so the LRU holds
        z[order] = self.dep.sample_xy(x[order], y[order]); src[~np.isnan(z)] = 1
    miss = np.isnan(z)
    if miss.any() and self.mode != "3dep":
      z[miss] = self.ter.sample(lat[miss], lon[miss]); src[miss & ~np.isnan(z)] = 2
    return z, src


def rings(e: dict) -> list[list[tuple[float, float]]]:
  """Outer rings of a closed way or a multipolygon relation, as (lat, lon)."""
  if e["type"] == "way":
    return [[(p["lat"], p["lon"]) for p in e["geometry"]]]
  segs = [[(p["lat"], p["lon"]) for p in m["geometry"]] for m in e.get("members", [])
          if m.get("role") == "outer" and m.get("geometry")]
  out = []
  while segs:
    r = segs.pop(0)
    while r[0] != r[-1]:
      for k, s in enumerate(segs):
        if s[0] == r[-1]: r += s[1:]; segs.pop(k); break
        if s[-1] == r[-1]: r += s[::-1][1:]; segs.pop(k); break
      else:
        break
    out.append(r)
  return [r for r in out if len(r) > 3]


def inside(ring, lat: float, lon: float) -> bool:
  c = False
  for (y1, x1), (y2, x2) in zip(ring, ring[1:] + ring[:1]):
    if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
      c = not c
  return c


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--place", help="anything Nominatim can find: a town, park, mountain, trailhead")
  ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
  ap.add_argument("--radius", type=float, default=6000, help="metres around the place or point")
  ap.add_argument("--bbox", help="south,west,north,east instead of a radius")
  ap.add_argument("--kind", choices=KINDS, default="trails")
  ap.add_argument("--dem", choices=["auto", "3dep", "terrarium"], default="auto")
  ap.add_argument("--max-3dep-tiles", type=int, default=150, help="each is 16 MB; above this, use terrarium")
  ap.add_argument("--name", help="output folder under data/ground (default from the place)")
  a = ap.parse_args()

  if a.bbox:
    s, w, n, e = map(float, a.bbox.split(","))
    lat0, lon0 = (s + n) / 2, (w + e) / 2; flt = f"({s},{w},{n},{e})"
  else:
    if a.place:
      g = geocode(a.place); lat0, lon0 = g["lat"], g["lon"]
      print(f"{a.place} -> {g['display_name']} ({lat0:.5f}, {lon0:.5f})")
    elif a.lat is not None and a.lon is not None:
      lat0, lon0 = a.lat, a.lon
    else:
      ap.error("give --place, --lat and --lon, or --bbox")
    flt = f"(around:{a.radius:.0f},{lat0:.6f},{lon0:.6f})"
  name = a.name or slugify(a.place) if (a.name or a.place) else f"{lat0:.4f}_{lon0:.4f}"
  out = ROOT / "data" / "ground" / name; out.mkdir(parents=True, exist_ok=True)
  with_areas = a.kind in ("trails", "walk")

  print(f"{a.kind} from OpenStreetMap")
  q = f"[out:json][timeout:170];\n(" + "".join(f"way{flt}{f};" for f in KINDS[a.kind]) + ")->.w;\n.w out tags geom;\n"
  q += 'rel(bw.w)[route~"^(hiking|foot|walking|bicycle|mtb|road)$"][name];\nout body;\n'
  if with_areas:
    q += (f'(way{flt}[leisure~"^(park|nature_reserve)$"][name];rel{flt}[leisure~"^(park|nature_reserve)$"][name];'
          f'way{flt}[boundary=protected_area][name];rel{flt}[boundary=protected_area][name];);\nout geom;')
  d = overpass(q)
  ways = [e for e in d["elements"] if e["type"] == "way" and e.get("tags", {}).get("highway") and len(e.get("geometry", [])) > 1]
  routes = [e for e in d["elements"] if e["type"] == "relation" and e.get("tags", {}).get("route")]
  areas = [e for e in d["elements"] if e.get("tags", {}).get("leisure") or e.get("tags", {}).get("boundary")]
  route_of: dict[int, set] = {}
  for r in routes:
    for m in r.get("members", []):
      if m["type"] == "way": route_of.setdefault(m["ref"], set()).add(r["tags"]["name"])
  print(f"  {len(ways)} ways, {len(routes)} named routes" + (f", {len(areas)} parks and protected areas" if with_areas else ""))
  if not ways:
    raise SystemExit("nothing found; try a bigger --radius or another --kind")

  elev = Elevation(a.dem, lat0, lon0, a.max_3dep_tiles)
  step = 1.0 if elev.dep is not None else 2.0          # resample spacing, matched to the elevation pixels
  res = f"{elev.dep_res:g} m source at the centre" if elev.dep_res else "source resolution unknown"
  print(f"elevation: " + (f"USGS 3DEP ({res}) with terrarium fill" if elev.dep else "AWS terrarium zoom 15"))

  frame = LocalFrame(lat0, lon0)
  lats, lons, offs, lens = [], [], [0], []
  for w in ways:
    la = np.array([p["lat"] for p in w["geometry"]]); lo = np.array([p["lon"] for p in w["geometry"]])
    x, y = frame.to_xy(la, lo)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]); L = float(s[-1])
    sn = np.linspace(0.0, L, max(2, int(L / step) + 1))
    rla, rlo = frame.to_latlon(np.interp(sn, s, x), np.interp(sn, s, y))
    lats.append(rla); lons.append(rlo); offs.append(offs[-1] + len(sn)); lens.append(L)
  z, src = elev.sample(np.concatenate(lats), np.concatenate(lons))

  area_rings = []
  if with_areas:
    for e in areas:
      rs = rings(e)
      if not rs: continue
      bb = (min(p[0] for r in rs for p in r), min(p[1] for r in rs for p in r), max(p[0] for r in rs for p in r), max(p[1] for r in rs for p in r))
      ext = sum(abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1]))) for r in
                [list(zip(*frame.to_xy(np.array([p[0] for p in r]), np.array([p[1] for p in r])))) for r in rs]) / 2
      area_rings.append((e, rs, bb, ext))

  feats = []
  for k, w in enumerate(ways):
    t = w["tags"]; h = z[offs[k]:offs[k + 1]].copy(); sr = src[offs[k]:offs[k + 1]]; L = lens[k]
    st = {"length_m": round(L, 1)}; profile = None; method = "measured"
    # Bare earth elevation drops bridges and sees through tunnels and covers, so those ways would dip to the river
    # or rail below. Their decks run close to straight between the abutments: draw a line between the end heights.
    if any(t.get(k_, "no") not in ("no", "") for k_ in ("bridge", "tunnel")) or t.get("covered") == "yes":
      if not np.isnan(h[[0, -1]]).any():
        h = np.linspace(h[0], h[-1], len(h)); method = "interpolated"
    if not np.isnan(h).all() and L > 0:
      dt = L / (len(h) - 1); dz = np.diff(h)
      win = max(1, int(round(GRADE_WIN_M / dt)))
      g = np.degrees(np.arctan(np.abs(h[win:] - h[:-win]) / (win * dt))) if len(h) > win else \
          np.array([np.degrees(np.arctan(abs(h[-1] - h[0]) / L))])
      st.update(climb_m=round(float(np.nansum(np.clip(dz, 0, None))), 1), descent_m=round(float(np.nansum(np.clip(-dz, 0, None))), 1),
                min_elevation_m=round(float(np.nanmin(h)), 1), max_elevation_m=round(float(np.nanmax(h)), 1),
                average_grade_deg=round(float(np.degrees(np.arctan(np.nanmean(np.abs(dz)) / dt))), 1),
                max_grade_deg_10m=round(float(np.nanmax(g)), 1),
                max_grade_pct_10m=round(float(np.tan(np.radians(np.nanmax(g))) * 100), 1),
                share_steeper_than_15deg=round(float(np.mean(g > 15)), 3))
      every = max(1, int(round(PROFILE_M / dt)))
      profile = {"heights": [round(float(v), 2) for v in h[::every]], "resolution_m": round(every * dt, 2)}
    used = set(sr[sr > 0].tolist())
    props = {"id": f"way/{w['id']}", **{k_: t.get(k_) for k_ in TAGS},
             "routes": sorted(route_of.get(w["id"], [])) or None,
             "elevation_source": {frozenset({1}): "3dep", frozenset({2}): "terrarium"}.get(frozenset(used), "mixed" if used else None),
             "elevation_method": method,
             "suspect_grade": bool(t["highway"] not in FOOT_ONLY and st.get("max_grade_pct_10m", 0) > ROAD_SUSPECT_PCT),
             "statistics": st, "elevationProfile": profile}
    if with_areas:
      mla, mlo = lats[k][len(lats[k]) // 2], lons[k][len(lons[k]) // 2]
      holders = [(ext, e) for e, rs, bb, ext in area_rings
                 if bb[0] <= mla <= bb[2] and bb[1] <= mlo <= bb[3] and any(inside(r, mla, mlo) for r in rs)]
      ar = min(holders, key=lambda v: v[0])[1] if holders else None     # the smallest area that holds it
      props["area"] = {"id": f"{ar['type']}/{ar['id']}", "name": ar["tags"]["name"]} if ar else None
    feats.append({"type": "Feature", "properties": props,
                  "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in w["geometry"]]}})

  (out / f"{a.kind}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
  agg: dict[str, dict] = {}
  if with_areas:
    for f in feats:
      ar = f["properties"]["area"]
      if not ar: continue
      s = f["properties"]["statistics"]
      v = agg.setdefault(ar["id"], {"name": ar["name"], "ways": 0, "length_m": 0.0, "climb_m": 0.0, "max_grade_deg_10m": 0.0,
                                    "min_elevation_m": 1e9, "max_elevation_m": -1e9})
      v["ways"] += 1; v["length_m"] += s["length_m"]; v["climb_m"] += s.get("climb_m", 0)
      v["max_grade_deg_10m"] = max(v["max_grade_deg_10m"], s.get("max_grade_deg_10m", 0))
      v["min_elevation_m"] = min(v["min_elevation_m"], s.get("min_elevation_m", 1e9))
      v["max_elevation_m"] = max(v["max_elevation_m"], s.get("max_elevation_m", -1e9))
    af = []
    for e, rs, _, _ in area_rings:
      v = agg.get(f"{e['type']}/{e['id']}")
      if not v: continue
      v.update(length_m=round(v["length_m"], 1), climb_m=round(v["climb_m"], 1), vertical_m=round(v["max_elevation_m"] - v["min_elevation_m"], 1))
      af.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[[[lo, la] for la, lo in r]] for r in rs]},
                 "properties": {"id": f"{e['type']}/{e['id']}", "kind": e["tags"].get("leisure") or e["tags"].get("boundary"),
                                "operator": e["tags"].get("operator"), **v}})
    (out / "areas.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": af}))

  srcs = Counter(f["properties"]["elevation_source"] for f in feats)
  (out / "meta.json").write_text(json.dumps({
    "kind": a.kind, "place": a.place, "center": [lat0, lon0], "radius_m": None if a.bbox else a.radius, "bbox": a.bbox,
    "ways": len(feats), "length_km": round(sum(lens) / 1000, 2), "areas": len(agg),
    "elevation": {"3dep_source_resolution_m_at_center": elev.dep_res, "ways_by_source": dict(srcs)},
    "sources": {"ways": "OpenStreetMap contributors, ODbL",
                "elevation": "USGS 3DEP (public domain) and AWS Terrain Tiles (Mapzen/Tilezen, attribution required)"},
  }, indent=1))
  print(f"wrote {len(feats)} ways ({sum(lens) / 1000:.1f} km) to {out / (a.kind + '.geojson')}; elevation by source {dict(srcs)}")
  interp = sum(f["properties"]["elevation_method"] == "interpolated" for f in feats)
  sus = sum(f["properties"]["suspect_grade"] for f in feats)
  if interp or sus:
    print(f"  {interp} bridges, tunnels or covered ways drawn straight between their ends; {sus} roads flagged suspect_grade")
  if agg:
    for v in sorted(agg.values(), key=lambda v: -v["length_m"])[:10]:
      print(f"  {v['name'][:38]:<38} {v['ways']:4d} ways {v['length_m'] / 1000:5.1f} km  climb {v['climb_m']:6.0f} m  "
            f"vertical {v['vertical_m']:5.1f} m  steepest {v['max_grade_deg_10m']:4.1f} deg")
  else:
    by = Counter(); steep = {}
    for f in feats:
      p = f["properties"]; by[p["highway"]] += p["statistics"]["length_m"]
      if not p["suspect_grade"]:
        steep[p["highway"]] = max(steep.get(p["highway"], 0), p["statistics"].get("max_grade_pct_10m", 0))
    for hw, L in by.most_common(10):
      print(f"  {hw:<16} {L / 1000:6.1f} km  steepest {steep.get(hw, 0):5.1f}% over 10 m (not counting flagged)")


if __name__ == "__main__":
  main()
