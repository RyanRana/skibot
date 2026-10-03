"""Draw the landing page's two data figures (media/tracks.svg, media/terrain.svg) from local data.

Needs the Marmolada GoPro CSVs in data/zenodo, out/gopro/descents.json (gopro.py) and the built
val-d-isere-bellevarde course (resort.py). Run from the repo root: .venv/bin/python web/tools/build_figures.py
"""
import csv, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SKI = Path(__file__).resolve().parents[2]  # repo root
OUT = Path(__file__).resolve().parents[1] / "media"
ORANGE, GREY = "#111315", "#16191d"  # line colour for tracks and the race line


def path(xs, ys):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


# 1. GoPro GPS tracks from Marmolada (Prochazka 2026, CC BY 4.0), descents highlighted.
desc = json.load(open(SKI / "out/gopro/descents.json"))
s_, w_, n_, e_ = desc["bbox_swne"]
lat0 = (s_ + n_) / 2
k = math.cos(math.radians(lat0))
M = 111_320.0  # metres per degree of latitude
tracks = []
for name, info in desc["files"].items():
    t, lat, lon = [], [], []
    with open(SKI / "data/zenodo" / name) as f:
        r = csv.reader(f)
        h = next(r)
        it, ila, ilo = h.index("video-time (ms)"), h.index("latitude (deg)"), h.index("longitude (deg)")
        for i, row in enumerate(r):
            if i % 100:
                continue  # 1 Hz
            t.append(float(row[it]) / 1000); lat.append(float(row[ila])); lon.append(float(row[ilo]))
    t, lat, lon = map(np.array, (t, lat, lon))
    x, y = (lon - w_) * k * M, (n_ - lat) * M
    spans = [(d["t_start_s"], d["t_end_s"]) for d in info["descents"]]
    tracks.append((t, x, y, spans))

# Break a track wherever the GPS jumps faster than 40 m/s (a lost fix, not a skier).
def pieces(t, x, y, mask):
    out, cur = [], []
    for i in np.flatnonzero(mask):
        if cur:
            j = cur[-1]
            if i != j + 1 or math.hypot(x[i] - x[j], y[i] - y[j]) / max(t[i] - t[j], 1e-3) > 40:
                if len(cur) > 2: out.append(cur)
                cur = []
        cur.append(i)
    if len(cur) > 2: out.append(cur)
    return out

ski_pts = np.concatenate([np.c_[x[(t >= a) & (t <= b)], y[(t >= a) & (t <= b)]]
                          for t, x, y, sp in tracks for a, b in sp])
# 4:3 window around the descents with a margin, so it fills the card.
mid = (ski_pts.max(0) + ski_pts.min(0)) / 2
ext = (ski_pts.max(0) - ski_pts.min(0)) * 1.15
span_x = max(ext[0], ext[1] * 4 / 3)
span_y = span_x * 3 / 4
lo, hi = mid - [span_x / 2, span_y / 2], mid + [span_x / 2, span_y / 2]
W, H = 600, 450
sc = W / span_x
P = lambda xs, ys: ((xs - lo[0]) * sc, (ys - lo[1]) * sc)
ski_paths = []
for t, x, y, spans in tracks:
    inside = (x > lo[0]) & (x < hi[0]) & (y > lo[1]) & (y < hi[1])
    ski = np.zeros_like(inside)
    for a, b in spans:
        ski |= (t >= a) & (t <= b)
    for idx in pieces(t, x, y, inside & ski):
        ski_paths.append(path(*P(x[idx], y[idx])))

# The ground under them: AWS Terrain Tiles via skibot's sampler, 50 m contours.
import sys
sys.path.insert(0, str(SKI))
from course import Terrain
gx = np.linspace(lo[0], hi[0], 300)
gy = np.linspace(lo[1], hi[1], 225)
GX, GY = np.meshgrid(gx, gy)
dem = Terrain(zoom=14).sample(n_ - GY / M, w_ + GX / (k * M))
lv = np.arange(math.floor(dem.min() / 50) * 50, dem.max() + 50, 50)
cs = plt.contour(gx, gy, dem, levels=lv)
cmin, cmaj = [], []
for lev, segs in zip(cs.levels, cs.allsegs):
    for seg in segs:
        if len(seg) > 2:
            (cmaj if int(round(lev)) % 250 == 0 else cmin).append(path(*P(seg[:, 0], seg[:, 1])))
plt.close("all")
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" fill="none" stroke-linecap="round" stroke-linejoin="round">',
       f'<g stroke="{GREY}" stroke-opacity=".12" stroke-width=".8">', *[f'<path d="{p}"/>' for p in cmin], "</g>",
       f'<g stroke="{GREY}" stroke-opacity=".34" stroke-width="1.1">', *[f'<path d="{p}"/>' for p in cmaj], "</g>",
       f'<g stroke="{ORANGE}" stroke-opacity=".8" stroke-width="2">', *[f'<path d="{p}"/>' for p in ski_paths], "</g>",
       "</svg>"]
(OUT / "tracks.svg").write_text("\n".join(svg))
print("tracks", len(ski_paths), "ski pieces", len(cmin) + len(cmaj), "contours", f"dem {dem.min():.0f}..{dem.max():.0f} m")

# 2. Val d'Isere Bellevarde: contours of the course heightfield, the run line and its gates.
course = SKI / "out/courses/val-d-isere-bellevarde"
z = np.load(course / "elevation.npy")  # row 0 = south
g = json.load(open(course / "grid.json"))
cell, x0, y0 = g["cell"], g["x0"], g["y0"]
xs = x0 + np.arange(z.shape[1]) * cell
ys = y0 + np.arange(z.shape[0]) * cell
win = (230.0, 230.0 + 653.0, y0, y0 + 490.0)  # 4:3 window of the middle of the run, metres
Wt = 600
st = Wt / (win[1] - win[0])
T = lambda X, Y: ((np.asarray(X) - win[0]) * st, (win[3] - np.asarray(Y)) * st)
zabs = z + g["z_datum_msl"]
levels = np.arange(math.floor(zabs.min() / 10) * 10, zabs.max() + 10, 10)
cs = plt.contour(xs, ys, zabs, levels=levels)
minor, major = [], []
for lev, segs in zip(cs.levels, cs.allsegs):
    for seg in segs:
        if len(seg) < 3:
            continue
        px, py = T(seg[:, 0], seg[:, 1])
        (major if int(round(lev)) % 50 == 0 else minor).append(path(px, py))
run = np.array(json.load(open(course / "run.json"))["course"])
rx, ry = T(run[:, 0], run[:, 1])
gates = json.load(open(course / "gates.json"))["gates"]
dots = []
for gt in gates:
    gx, gy = T(gt["turn_pole"][0], gt["turn_pole"][1])
    if 0 <= gx <= Wt and 0 <= gy <= 490 * st:
        dots.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="2.4"/>')
Ht = 490 * st
(OUT / "terrain.svg").write_text("\n".join([
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wt} {Ht:.0f}" fill="none" stroke-linejoin="round" stroke-linecap="round">',
    f'<g stroke="{GREY}" stroke-opacity=".12" stroke-width=".8">', *[f'<path d="{p}"/>' for p in minor], "</g>",
    f'<g stroke="{GREY}" stroke-opacity=".34" stroke-width="1.1">', *[f'<path d="{p}"/>' for p in major], "</g>",
    f'<path d="{path(rx, ry)}" stroke="{ORANGE}" stroke-width="2.4"/>',
    f'<g fill="{ORANGE}">', *dots, "</g>",
    "</svg>"]))
print("terrain", len(minor) + len(major), "contours", len(dots), "gates in window")
