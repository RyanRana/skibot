"""Point it at any ski resort and get the real slope in MuJoCo.

    python resort.py "Bormio" --list
    python resort.py "Bormio" --run "stelvio"
    python resort.py "Marmolada" --run "^bellunese$" --track out/gopro/descents.json

--run is a case- and accent-insensitive regex; every piste name that matches is stitched together.
Writes out/courses/<slug>/ (see course.py), checks the MuJoCo hfield against the elevation grid with
mj_ray, and renders out/<slug>_{wide,course,start}.png.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from course import (DEFAULT_ROBOT_XML, LocalFrame, bilinear_z, build_course, list_runs,  # noqa: E402
                    make_scene, surface_z)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def verify(course_dir: Path, scene_xml: Path, n: int = 400, seed: int = 0) -> dict:
    """Cast rays straight down onto the terrain geom only and compare with elevation.npy."""
    grid = np.load(course_dir / "elevation.npy")
    gm = json.loads((course_dir / "grid.json").read_text())
    x0, y0, cell = gm["x0"], gm["y0"], gm["cell"]
    nrow, ncol = grid.shape
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    ztop = float(grid.max()) + 50.0
    gid = np.array([-1], dtype=np.int32)

    def ray(x, y):
        dist = mujoco.mj_ray(m, d, np.array([x, y, ztop]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, gid)
        return (ztop - dist) if (dist >= 0 and gid[0] == tid) else np.nan

    rng = np.random.default_rng(seed)
    r = rng.integers(1, nrow - 1, n)
    c = rng.integers(1, ncol - 1, n)
    zr = np.array([ray(x0 + ci * cell, y0 + ri * cell) for ri, ci in zip(r, c)])
    e_vertex = np.abs(zr - grid[r, c])
    e_flip = np.abs(zr - grid[::-1][r, c])
    xs = rng.uniform(x0 + cell, x0 + (ncol - 2) * cell, n)
    ys = rng.uniform(y0 + cell, y0 + (nrow - 2) * cell, n)
    zi = np.array([ray(x, y) for x, y in zip(xs, ys)])
    e_tri = np.abs(zi - surface_z(grid, x0, y0, cell, xs, ys))
    e_bil = np.abs(zi - bilinear_z(grid, x0, y0, cell, xs, ys))
    misses = int(np.isnan(zr).sum() + np.isnan(zi).sum())

    def st(e):
        return {"max_m": float(np.nanmax(e)), "mean_m": float(np.nanmean(e))}

    return {"n_vertex": n, "n_interior": n, "misses": misses, "vertex_vs_grid": st(e_vertex),
            "interior_vs_triangle_surface": st(e_tri), "interior_vs_bilinear": st(e_bil),
            "control_vertex_vs_row_flipped_grid": st(e_flip)}


def _cam(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def _cam_from_to(pos, lookat):
    d = np.asarray(lookat, float) - np.asarray(pos, float)
    dist = float(np.linalg.norm(d))
    return _cam(lookat, dist, math.degrees(math.atan2(d[1], d[0])), math.degrees(math.asin(d[2] / dist)))


def render(course_dir: Path, scene_xml: Path, slug: str, width: int = 1920, height: int = 1080) -> list[Path]:
    grid = np.load(course_dir / "elevation.npy")
    gm = json.loads((course_dir / "grid.json").read_text())
    start = json.loads((course_dir / "start_pose.json").read_text())
    course = np.array(json.loads((course_dir / "run.json").read_text())["course"])
    x0, y0, cell = gm["x0"], gm["y0"], gm["cell"]
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    rend = mujoco.Renderer(m, height, width)
    heading = math.degrees(start["yaw"])
    xy, s = course[:, :2], course[:, 3]
    L = float(s[-1])
    outs = []

    def shot(name, cam, groups, znear, zfar, shadows):
        m.vis.map.znear, m.vis.map.zfar = znear / m.stat.extent, zfar / m.stat.extent
        # keep the sun's shadows on: with every light's castshadow off, MuJoCo renders part of the hfield black
        m.light_castshadow[:] = 1
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = groups
        rend.update_scene(d, camera=cam, scene_option=opt)
        p = OUT / f"{slug}_{name}.png"
        Image.fromarray(rend.render()).save(p)
        outs.append(p)

    # wide: 3/4 view from below and to the side of the course, overlays on
    mid = xy[len(xy) // 2]
    zmid = float(surface_z(grid, x0, y0, cell, mid[0], mid[1]))
    span = float(np.ptp(xy, axis=0).max())
    shot("wide", _cam([mid[0], mid[1], zmid], 1.05 * max(span, 300.0), heading + 180 - 45, -32),
         [1, 1, 1, 0, 1, 0], 2.0, 8000.0, False)

    # course: racer's view from above and behind the start, looking down the first gates; the camera rises
    # until the sightline to the look point clears the terrain
    fwd = np.array([math.cos(start["yaw"]), math.sin(start["yaw"])])
    pos_xy = -14.0 * fwd
    k = int(np.searchsorted(s, min(55.0, L)))
    look = np.array([xy[k, 0], xy[k, 1], float(surface_z(grid, x0, y0, cell, *xy[k]))])
    zc = max(float(surface_z(grid, x0, y0, cell, *pos_xy)), float(surface_z(grid, x0, y0, cell, 0.0, 0.0))) + 6.0
    for _ in range(60):
        pos = np.array([pos_xy[0], pos_xy[1], zc])
        u = np.linspace(0.02, 0.95, 80)[:, None]
        line = pos + u * (look - pos)
        if np.all(line[:, 2] > surface_z(grid, x0, y0, cell, line[:, 0], line[:, 1]) + 1.0):
            break
        zc += 2.0
    shot("course", _cam_from_to(pos, look), [1, 1, 1, 0, 0, 0], 0.05, 3000.0, True)

    # start: close-up of the robot on its skis
    pelvis = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
    shot("start", _cam(pelvis - np.array([0, 0, 0.25]), 2.6, heading + 180 + 35, -10), [1, 1, 1, 0, 0, 0],
         0.02, 2000.0, True)
    rend.close()
    return outs


def tracks_for(course_dir: Path, track_json: Path) -> list[np.ndarray]:
    """GoPro descents in the course frame, draped 1 m above the snow, clipped to the grid."""
    grid = np.load(course_dir / "elevation.npy")
    gm = json.loads((course_dir / "grid.json").read_text())
    fr = LocalFrame(gm["lat0"], gm["lon0"])
    rep = json.loads(Path(track_json).read_text())
    out = []
    for f in rep["files"].values():
        for dsc in f["descents"]:
            tr = dsc["track_1hz"]
            x, y = fr.to_xy(np.array(tr["lat"]), np.array(tr["lon"]))
            inside = ((x > gm["x_range"][0]) & (x < gm["x_range"][1]) & (y > gm["y_range"][0]) & (y < gm["y_range"][1]))
            if inside.sum() < 5:
                continue
            # split at gaps so the chain doesn't jump across the map
            idx = np.flatnonzero(inside)
            parts = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for p in parts:
                if len(p) >= 5:
                    z = surface_z(grid, gm["x0"], gm["y0"], gm["cell"], x[p], y[p]) + 1.0
                    out.append(np.c_[x[p], y[p], z])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resort")
    ap.add_argument("--list", action="store_true", help="list runs (name, difficulty, length, drop)")
    ap.add_argument("--run", help="piste name regex (default: biggest-drop intermediate/advanced/expert run)")
    ap.add_argument("--near", help="lat,lon to search around instead of geocoding the resort name")
    ap.add_argument("--radius", type=float, default=6000.0)
    ap.add_argument("--bbox", help="s,w,n,e piste search box instead of a radius")
    ap.add_argument("--course-length", type=float, default=1200.0)
    ap.add_argument("--from-top", action="store_true", help="course starts at the top instead of the steepest window")
    ap.add_argument("--max-slope", type=float, default=35.0, help="steepest 10 m allowed in the chosen window (deg)")
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--robot", default=str(DEFAULT_ROBOT_XML))
    ap.add_argument("--track", help="out/gopro/descents.json to draw in the wide render")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--slug")
    a = ap.parse_args()
    near = tuple(float(v) for v in a.near.split(",")) if a.near else None
    bbox = tuple(float(v) for v in a.bbox.split(",")) if a.bbox else None

    if a.list:
        center, runs = list_runs(a.resort, near, a.radius, bbox)
        print(f"{center['display_name']}  ({center['lat']:.5f}, {center['lon']:.5f})  {len(runs)} runs")
        print(f"{'run':48s} {'difficulty':12s} {'len m':>7s} {'drop m':>7s} {'top m':>7s} {'segs':>4s} {'chains':>6s}")
        for r in runs:
            print(f"{r.name[:48]:48s} {str(r.difficulty):12s} {r.length:7.0f} {r.drop:7.0f} {r.z_top:7.0f} "
                  f"{r.n_segments:4d} {r.n_chains:6d}")
        return

    t0 = time.time()
    cdir = build_course(a.resort, a.run, near=near, radius=a.radius, bbox=bbox, cell=a.cell,
                        course_length=a.course_length, from_top=a.from_top, max_slope_deg=a.max_slope,
                        slug=a.slug)
    scene = make_scene(cdir, a.robot)
    v = verify(cdir, scene)
    (cdir / "verify.json").write_text(json.dumps(v, indent=1))
    stats = json.loads((cdir / "stats.json").read_text())
    start = json.loads((cdir / "start_pose.json").read_text())
    print(json.dumps({"run": stats["run"], "full_run": stats["full_run"], "course": stats["course"],
                      "course_selection": stats["course_selection"],
                      "grid": stats["grid"], "start": {k: start[k] for k in ("yaw_deg", "slope_deg_along_heading")},
                      "robot": start["robots"][Path(a.robot).stem], "verify": v}, indent=1))
    if not a.no_render:
        ov = {"run": True, "gates": True}
        if a.track:
            ov["tracks"] = tracks_for(cdir, Path(a.track))
            print(f"track overlay: {len(ov['tracks'])} pieces")
        oscene = make_scene(cdir, a.robot, overlays=ov)
        for p in render(cdir, oscene, cdir.name):
            print("render", p)
    print(f"done in {time.time() - t0:.1f}s -> {cdir}")


if __name__ == "__main__":
    main()
