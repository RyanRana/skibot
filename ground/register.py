"""Tie a phone session to the ground under it: distance, climb (barometer and terrain tiles), the OSM trails it
followed and their tags, cadence, and every gap in the motion stream.

    python -m ground.register out/ground/<session_id>      # prints the summary, writes summary.json there
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import requests

from course import LocalFrame, Terrain, _cached_json, _http
from gopro import nearest_piste
from ground.schema import load_session

MAX_HACC_M = 30.0       # drop GPS fixes worse than this
MATCH_M = 25.0          # map matching radius to a trail centerline
GAP_S = 0.5             # a hole in the 100 Hz motion stream longer than this is a gap
CLIMB_HYST_M = 2.0      # hysteresis when summing elevation gain, so sensor noise does not count as climbing
TRAIL_NAMES = Path(__file__).resolve().parent.parent / "out" / "ground" / "trail_names.json"  # {"<osm way id>": "name"} for ways OSM leaves unnamed
TRIM_M = 200.0          # the shareable track leaves out this much at each end, so no one's home shows up
OVERPASS_MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter"]
OVERPASS_TIMEOUT_S = 25  # a hiker is waiting on the summary text; course.overpass retries for many minutes


def trail_query(bbox) -> str:
    """Overpass QL for walkable ways and hiking route relations in a s,w,n,e bbox, geometry clipped to the bbox."""
    f = "({:.6f},{:.6f},{:.6f},{:.6f})".format(*bbox)
    return (f'[out:json][timeout:{OVERPASS_TIMEOUT_S}];(way["highway"~"^(path|footway|track|bridleway|steps|pedestrian)$"]{f};'
            f'relation["route"~"^(hiking|foot)$"]{f};);out geom{f};')


def fetch_trails(query: str) -> dict:
    """One pass over the mirrors with a short timeout, cached like course.overpass. Raises with every mirror's error."""
    def fetch():
        errs = []
        for url in OVERPASS_MIRRORS:
            try:
                r = _http().post(url, data={"data": query}, timeout=OVERPASS_TIMEOUT_S)
                if r.status_code == 200:
                    return r.json()
                errs.append(f"{url} HTTP {r.status_code}")
            except requests.RequestException as e:
                errs.append(f"{url} {type(e).__name__}")
        raise RuntimeError("all overpass mirrors failed: " + "; ".join(errs))

    return _cached_json("overpass", query, fetch)


def parse_trails(osm: dict) -> list[dict]:
    ways, rel_name = {}, {}
    for el in osm.get("elements", []):
        t = el.get("tags", {})
        if el["type"] == "way" and el.get("geometry"):
            ll = np.array([(p["lat"], p["lon"]) for p in el["geometry"] if p], dtype=np.float64).reshape(-1, 2)
            if len(ll) >= 2:
                ways[el["id"]] = {"id": el["id"], "name": t.get("name"), "highway": t.get("highway"),
                                  "sac_scale": t.get("sac_scale"), "surface": t.get("surface"), "latlon": ll}
        elif el["type"] == "relation" and t.get("name"):
            for m in el.get("members", []):
                if m.get("type") == "way":
                    rel_name.setdefault(m["ref"], t["name"])
    for wid, nm in rel_name.items():
        if wid in ways and not ways[wid]["name"]:
            ways[wid]["name"] = nm
    local = json.loads(TRAIL_NAMES.read_text()) if TRAIL_NAMES.exists() else {}
    for w in ways.values():
        w["name"] = w["name"] or local.get(str(w["id"]))
    return list(ways.values())


def climb(z: np.ndarray, hyst: float = CLIMB_HYST_M) -> tuple[float, float]:
    """Total gain and loss with a hysteresis band."""
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return 0.0, 0.0
    up = down = 0.0
    ref = z[0]
    for v in z[1:]:
        if v - ref >= hyst:
            up += v - ref
            ref = v
        elif ref - v >= hyst:
            down += ref - v
            ref = v
    return up, down


def find_gaps(t: np.ndarray, start: float | None, end: float | None) -> list[list[float]]:
    edges = list(t) if len(t) else []
    if start is not None and end is not None:
        edges = [start] + edges + [end]
    e = np.asarray(edges, dtype=np.float64)
    if len(e) < 2:
        return [[start, end]] if start is not None and end is not None else []
    k = np.nonzero(np.diff(e) > GAP_S)[0]
    return [[float(e[i]), float(e[i + 1])] for i in k]


def cadence_from_acc(imu: np.ndarray, rate: float) -> float | None:
    """Steps per minute from the dominant 1-3.2 Hz frequency of vertical acceleration (user acceleration projected on
    gravity, so it does not depend on how the phone sits in a pocket). Gaps are ignored, fine for a dominant frequency."""
    if len(imu) < rate * 20:
        return None
    acc, grav = imu["acc"].astype(np.float64), imu["grav"].astype(np.float64)
    up = grav / np.maximum(np.linalg.norm(grav, axis=1, keepdims=True), 1e-6)
    a = (acc * up).sum(1)
    a = a - a.mean()
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    f = np.fft.rfftfreq(len(a), 1 / rate)
    band = (f >= 1.0) & (f <= 3.2)
    if not band.any() or spec[band].max() < 3 * np.median(spec[band]):
        return None
    return float(f[band][spec[band].argmax()] * 60)


def register(d: Path) -> dict:
    meta, data, warnings = load_session(d)
    imu, gps, baro = data["imu"], data["gps"], data["baro"]
    start, end = meta.get("started"), meta.get("ended")
    s: dict = {"session": meta["session_id"], "placement": meta.get("placement"), "started": start, "ended": end,
               "duration_s": round(end - start, 1) if start and end else None, "warnings": warnings, "errors": []}

    rate = float(1 / np.median(np.diff(imu["t"]))) if len(imu) > 1 else 0.0
    gaps = find_gaps(imu["t"], start, end)
    s["imu"] = {"samples": int(len(imu)), "rate_hz": round(rate, 1)}
    s["gps"] = {"fixes": int(len(gps))}
    s["baro"] = {"samples": int(len(baro))}
    s["gaps"] = {"count": len(gaps), "total_s": round(sum(b - a for a, b in gaps), 1), "list": gaps}

    good = gps[(gps["hacc"] >= 0) & (gps["hacc"] <= MAX_HACC_M)] if len(gps) else gps
    s["gps"]["good_fixes"] = int(len(good))
    if len(good) >= 2:
        frame = LocalFrame(float(np.median(good["lat"])), float(np.median(good["lon"])))
        x, y = frame.to_xy(good["lat"], good["lon"])
        xy = np.c_[x, y]
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s["distance_m"] = round(float(step.sum()), 1)
        s["gps"]["median_hacc_m"] = round(float(np.median(good["hacc"])), 1)

        rel = baro["rel_alt"][np.isfinite(baro["rel_alt"])] if len(baro) else np.zeros(0)
        if len(rel) >= 2:
            up, down = climb(rel.astype(np.float64))
            s["climb_baro_m"] = {"gain": round(up, 1), "loss": round(down, 1)}
        try:
            dem = Terrain(15).sample(good["lat"], good["lon"])
            up, down = climb(dem)
            s["climb_dem_m"] = {"gain": round(up, 1), "loss": round(down, 1),
                                "start_m": round(float(dem[0]), 1), "max_m": round(float(dem.max()), 1)}
            s["gps_alt_minus_dem_m"] = round(float(np.median(good["alt"] - dem)), 1)
        except Exception as e:  # noqa: BLE001  reported, not hidden
            s["errors"].append(f"terrain tiles: {e}")

        try:
            pad = 0.002
            bbox = (good["lat"].min() - pad, good["lon"].min() - pad, good["lat"].max() + pad, good["lon"].max() + pad)
            trails = parse_trails(fetch_trails(trail_query(bbox)))
            s["trails_in_area"] = len(trails)
            if trails:
                A, B, owner = [], [], []
                for i, tr in enumerate(trails):
                    tx, ty = frame.to_xy(tr["latlon"][:, 0], tr["latlon"][:, 1])
                    p = np.c_[tx, ty]
                    A.append(p[:-1])
                    B.append(p[1:])
                    owner += [i] * (len(p) - 1)
                dist, idx = nearest_piste(xy, np.vstack(A), np.vstack(B))
                own = np.asarray(owner)[np.clip(idx, 0, None)]
                hit = dist <= MATCH_M
                s["trail_match"] = {"matched_share": round(float(hit.mean()), 3),
                                    "median_offset_m": round(float(np.median(dist)), 1)}
                names = Counter(trails[i]["name"] or f"unnamed {trails[i]['highway']}" for i in own[hit])
                s["trail_match"]["top"] = [[n, round(c / len(own), 3)] for n, c in names.most_common(5)]
                s["trail_match"]["ways"] = [[int(trails[i]["id"]), trails[i]["name"] or f"unnamed {trails[i]['highway']}"]
                                            for i, _ in Counter(own[hit].tolist()).most_common(5)]
                for key in ("sac_scale", "surface", "highway"):
                    vals = Counter(trails[i][key] for i in own[hit] if trails[i][key])
                    s["trail_match"][key] = [[v, round(c / len(own), 3)] for v, c in vals.most_common(4)]
        except Exception as e:  # noqa: BLE001
            s["errors"].append(f"trail matching: {e}")

        along = np.r_[0, np.cumsum(step)]
        keep = (along >= TRIM_M) & (along <= along[-1] - TRIM_M)
        s["track_trimmed"] = {"t": np.round(good["t"][keep], 1).tolist(),
                              "lat": np.round(good["lat"][keep], 6).tolist(),
                              "lon": np.round(good["lon"][keep], 6).tolist(), "trim_m": TRIM_M}
    else:
        s["errors"].append(f"not enough good GPS fixes ({len(good)} with hacc <= {MAX_HACC_M} m)")

    ped = meta.get("pedometer") or {}
    if ped.get("steps") and s["duration_s"]:
        s["steps"] = int(ped["steps"])
        s["cadence_spm"] = round(ped["steps"] / (s["duration_s"] / 60), 1)
        s["cadence_source"] = "pedometer"
    else:
        c = cadence_from_acc(imu, rate) if rate else None
        if c:
            s["cadence_spm"], s["cadence_source"] = round(c, 1), "accelerometer"
    return s


def message(s: dict) -> str:
    """The summary text the Photon agent sends back."""
    parts = []
    if s.get("distance_m") is not None:
        parts.append(f"{s['distance_m'] / 1000:.2f} km")
    named = [n for n, _ in (s.get("trail_match") or {}).get("top") or [] if not n.startswith("unnamed")]
    if named:
        parts.append(f"on {named[0]}")
    gain = (s.get("climb_baro_m") or s.get("climb_dem_m") or {}).get("gain")
    if gain is not None:
        parts.append(f"+{gain:.0f} m")
    if s.get("duration_s"):
        m = int(s["duration_s"] // 60)
        parts.append(f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m} min" if m else f"{s['duration_s']:.0f} s")
    line1 = "Hazard Intelligence: " + ", ".join(parts) if parts else "Hazard Intelligence: hike received"
    line2 = f"{s['imu']['samples']:,} motion readings at {s['imu']['rate_hz']:.0f} Hz"
    g = s["gaps"]
    line3 = "no gaps" if g["count"] == 0 else f"{g['count']} gap{'s' if g['count'] > 1 else ''} ({g['total_s']:.0f} s total)"
    out = f"{line1}\n{line2}, {line3}.\nThank you, this hike is now robot training data."
    if s["errors"]:
        out += "\n(Some steps failed on our side. We're on it.)"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    a = ap.parse_args()
    s = register(a.session_dir)
    (a.session_dir / "summary.json").write_text(json.dumps(s))
    print(json.dumps({k: v for k, v in s.items() if k != "track_trimmed"}, indent=1))
    print("\n" + message(s))


if __name__ == "__main__":
    main()
