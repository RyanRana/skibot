"""GoPro telemetry from the Zenodo record "Marmolada Skiing Analysis" (Prochazka, 2026, CC BY 4.0,
doi:10.5281/zenodo.21777004): split each file into descents, then measure speed, distance, which OSM
pistes the descents follow, and GPS altitude (MSL) minus the terrain DEM.

    python gopro.py                      # both CSVs in data/zenodo -> out/gopro/descents.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from course import LocalFrame, Terrain, find_pistes

ROOT = Path(__file__).resolve().parent
ZENODO = ROOT / "data" / "zenodo"
OUT = ROOT / "out" / "gopro"

SKI_SPEED_KMH = 10.0      # moving faster than this
SKI_VZ = -0.15            # m/s, and losing altitude
LIFT_VZ = 0.3             # m/s, climbing faster than this is a lift
MERGE_GAP_S = 20          # join descents split by a short stop
MIN_DESCENT_S = 30
MIN_DROP_M = 30.0
MATCH_M = 30.0            # map-matching radius to an OSM piste centerline


def load(path: Path) -> dict:
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        first = fh.readline().strip().split(",")
    use = [i for i in range(len(header)) if i != 1]  # column 1 is the ISO timestamp
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=use)
    col = {header[i]: a[:, k] for k, i in enumerate(use)}
    return {"t": col["video-time (ms)"] / 1000.0, "lat": col["latitude (deg)"], "lon": col["longitude (deg)"],
            "alt": col["altitude [gps msl] (m)"], "v2d": col["speed [2d gps] (km/h)"],
            "v3d": col["speed [3d gps] (km/h)"], "start_utc": first[1], "header": header,
            "acc": np.c_[col["accelerometer-x (m/s²)"], col["accelerometer-y (m/s²)"], col["accelerometer-z (m/s²)"]]}


def per_second(d: dict) -> dict:
    sec = np.floor(d["t"]).astype(np.int64)
    n = sec.max() + 1
    cnt = np.bincount(sec, minlength=n)
    ok = cnt > 0

    def mean(x):
        return (np.bincount(sec, weights=x, minlength=n)[ok] / cnt[ok])

    return {"t": np.arange(n)[ok].astype(float), "lat": mean(d["lat"]), "lon": mean(d["lon"]), "alt": mean(d["alt"]),
            "v2d": mean(d["v2d"]), "v3d": mean(d["v3d"])}


def segment(s: dict) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    alt = median_filter(s["alt"], size=5, mode="nearest")
    vz = uniform_filter1d(np.gradient(alt, s["t"]), size=11, mode="nearest")
    spd = uniform_filter1d(s["v2d"], size=5, mode="nearest")
    ski = (spd > SKI_SPEED_KMH) & (vz < SKI_VZ)
    lift = vz > LIFT_VZ
    runs, i, n = [], 0, len(ski)
    while i < n:
        if ski[i]:
            j = i
            while j + 1 < n and ski[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= MERGE_GAP_S and not lift[merged[-1][1]:a].any():
            merged[-1][1] = b
        else:
            merged.append([a, b])
    keep = [(a, b) for a, b in merged if s["t"][b] - s["t"][a] >= MIN_DESCENT_S and alt[a] - alt[b] >= MIN_DROP_M]
    return keep, alt, vz, lift


def nearest_piste(xy: np.ndarray, segs_a: np.ndarray, segs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the nearest piste segment and that segment's index."""
    best_d = np.full(len(xy), np.inf)
    best_i = np.full(len(xy), -1)
    ab = segs_b - segs_a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    for k in range(0, len(xy), 256):
        p = xy[k:k + 256, None, :]
        t = np.clip(((p - segs_a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
        q = segs_a[None] + t[..., None] * ab[None]
        dist = np.linalg.norm(p - q, axis=-1)
        j = dist.argmin(1)
        best_d[k:k + 256] = dist[np.arange(len(j)), j]
        best_i[k:k + 256] = j
    return best_d, best_i


def analyze(paths: list[Path]) -> dict:
    files = {p.name: load(p) for p in paths}
    lat_all = np.concatenate([f["lat"] for f in files.values()])
    lon_all = np.concatenate([f["lon"] for f in files.values()])
    good = (np.abs(lat_all) > 1e-6) & (np.abs(lon_all) > 1e-6)
    pad = 0.01
    bbox = (lat_all[good].min() - pad, lon_all[good].min() - pad, lat_all[good].max() + pad, lon_all[good].max() + pad)
    _, segs = find_pistes("gopro-track-area", bbox=bbox)
    frame = LocalFrame((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    A, B, names = [], [], []
    for sg in segs:
        x, y = frame.to_xy(sg.latlon[:, 0], sg.latlon[:, 1])
        p = np.c_[x, y]
        A.append(p[:-1])
        B.append(p[1:])
        names += [sg.name or (f"ref {sg.ref}" if sg.ref else f"unnamed way {sg.way_id}")] * (len(p) - 1)
    A, B, names = np.vstack(A), np.vstack(B), np.array(names, dtype=object)
    terrain = Terrain(15)

    report = {"source": "Zenodo 10.5281/zenodo.21777004 (Prochazka 2026, CC BY 4.0)", "bbox_swne": bbox,
              "osm_pistes_in_bbox": len(segs), "params": {"ski_speed_kmh": SKI_SPEED_KMH, "ski_vz_mps": SKI_VZ,
              "lift_vz_mps": LIFT_VZ, "merge_gap_s": MERGE_GAP_S, "min_descent_s": MIN_DESCENT_S,
              "min_drop_m": MIN_DROP_M, "match_radius_m": MATCH_M}, "files": {}}
    all_diff = []
    name_time = Counter()
    for fname, d in files.items():
        dt = np.diff(d["t"])
        moved = (np.diff(d["lat"]) != 0) | (np.diff(d["lon"]) != 0)
        gps_rate = moved.sum() / (d["t"][-1] - d["t"][0])
        s = per_second(d)
        desc, alt_s, vz, lift = segment(s)
        frec = {"rows": int(len(d["t"])), "duration_s": round(float(d["t"][-1] - d["t"][0]), 1),
                "start_utc": d["start_utc"], "sample_rate_hz": round(float(1 / np.median(dt)), 2),
                "gps_position_updates_per_s": round(float(gps_rate), 2),
                "seconds_lift": int(lift.sum()), "descents": []}
        for k, (a, b) in enumerate(desc):
            lat, lon, alt = s["lat"][a:b + 1], s["lon"][a:b + 1], s["alt"][a:b + 1]
            x, y = frame.to_xy(lat, lon)
            xy = np.c_[x, y]
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            path2d = float(step.sum())
            path3d = float(np.sqrt(step ** 2 + np.diff(alt_s[a:b + 1]) ** 2).sum())
            dur = float(s["t"][b] - s["t"][a])
            raw = (d["t"] >= s["t"][a]) & (d["t"] < s["t"][b] + 1)
            v_raw = d["v2d"][raw]
            dem = terrain.sample(lat, lon)
            diff = alt - dem
            all_diff.append(diff)
            dist, idx = nearest_piste(xy, A, B)
            matched = np.where(dist <= MATCH_M, names[np.clip(idx, 0, None)], None)
            share = Counter(m for m in matched if m)
            for nm, c in share.items():
                name_time[nm] += c
            n_pts = len(matched)
            frec["descents"].append({
                "id": k, "t_start_s": float(s["t"][a]), "t_end_s": float(s["t"][b]), "duration_s": round(dur, 1),
                "alt_start_m": round(float(alt_s[a]), 1), "alt_end_m": round(float(alt_s[b]), 1),
                "vertical_drop_m": round(float(alt_s[a] - alt_s[b]), 1),
                "path_length_2d_m": round(path2d, 1), "path_length_3d_m": round(path3d, 1),
                "speed_max_kmh": round(float(v_raw.max()), 1), "speed_p99_kmh": round(float(np.percentile(v_raw, 99)), 1),
                "speed_mean_kmh": round(path2d / dur * 3.6, 1), "gps_speed_mean_kmh": round(float(v_raw.mean()), 1),
                "alt_minus_dem_m": {"median": round(float(np.median(diff)), 2),
                                    "p25": round(float(np.percentile(diff, 25)), 2),
                                    "p75": round(float(np.percentile(diff, 75)), 2)},
                "piste_match": {"matched_share": round(1 - sum(1 for m in matched if m is None) / n_pts, 3),
                                "top": [[nm, round(c / n_pts, 3)] for nm, c in share.most_common(4)]},
                "track_1hz": {"t": s["t"][a:b + 1].tolist(), "lat": lat.tolist(), "lon": lon.tolist(),
                              "alt": np.round(alt, 2).tolist()},
            })
        report["files"][fname] = frec
    diff = np.concatenate(all_diff) if all_diff else np.array([np.nan])
    report["alt_minus_dem_all_descents_m"] = {
        "n_seconds": int(diff.size), "median": round(float(np.median(diff)), 2),
        "p25": round(float(np.percentile(diff, 25)), 2), "p75": round(float(np.percentile(diff, 75)), 2),
        "iqr": round(float(np.percentile(diff, 75) - np.percentile(diff, 25)), 2),
        "p5": round(float(np.percentile(diff, 5)), 2), "p95": round(float(np.percentile(diff, 95)), 2)}
    report["piste_time_seconds"] = name_time.most_common(15)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", type=Path, default=sorted(ZENODO.glob("*.csv")))
    args = ap.parse_args()
    rep = analyze(args.csv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "descents.json").write_text(json.dumps(rep))
    for fname, f in rep["files"].items():
        print(f"{fname}: {f['rows']} rows, {f['duration_s']} s from {f['start_utc']}, {f['sample_rate_hz']} Hz, "
              f"GPS position updates {f['gps_position_updates_per_s']}/s, {len(f['descents'])} descents, "
              f"{f['seconds_lift']} s classified as lift")
        for dd in f["descents"]:
            print(f"  #{dd['id']} t={dd['t_start_s']:.0f}-{dd['t_end_s']:.0f}s  {dd['duration_s']:.0f} s  "
                  f"drop {dd['vertical_drop_m']} m ({dd['alt_start_m']} -> {dd['alt_end_m']})  "
                  f"path {dd['path_length_2d_m']} m  vmax {dd['speed_max_kmh']} (p99 {dd['speed_p99_kmh']}) km/h  "
                  f"vmean {dd['speed_mean_kmh']} km/h  alt-DEM median {dd['alt_minus_dem_m']['median']} m  "
                  f"pistes {dd['piste_match']['top']} matched {dd['piste_match']['matched_share']}")
    print("alt - DEM over all descents:", rep["alt_minus_dem_all_descents_m"])
    print("piste time (s):", rep["piste_time_seconds"])
    print(f"wrote {OUT / 'descents.json'}")


if __name__ == "__main__":
    main()
