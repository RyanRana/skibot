"""A Unitree G1 hiking a real Ann Arbor trail in CPU MuJoCo: SONIC's kinematic planner makes the steps, SONIC tracks
them on the 1 m lidar ground, and the planner is steered toward a point a few metres ahead on the trail.

    .venv/bin/python -m hikesim.walk bird-hills-nature-area --start 0 --length 120
    .venv/bin/python -m hikesim.walk bird-hills-nature-area --steepest 100      # the steepest 100 m of the walk

Writes out/hike/runs/<course>/<run>/: run.json (stats), frames.npz (qpos at 25 Hz plus route progress).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from hikesim.planner import Planner
from skisim.scene import g1_model
from skisim.sonic import DEFAULT_ANGLES, SonicController
from skisim.terrain import HeightGrid, from_course

ROOT = Path(__file__).resolve().parents[1]
COURSES = ROOT / "out" / "hike" / "courses"
RUNS = ROOT / "out" / "hike" / "runs"
GROUND_RGBA = (0.52, 0.45, 0.34, 1)   # forest floor tint over the snow texture
TRAIL_RGBA = (0.78, 0.68, 0.5, 1)


def load(course: str) -> tuple[HeightGrid, dict]:
  d = COURSES / course
  return from_course(d), json.loads((d / "route.json").read_text())


def crop(grid: HeightGrid, xy: np.ndarray, margin: float = 30.0) -> HeightGrid:
  """Terrain window around a stretch of trail, so MuJoCo only carries the ground the robot can reach."""
  c0 = int(max(0, (xy[:, 0].min() - margin - grid.x0) // grid.cell))
  c1 = int(min(grid.ncol, (xy[:, 0].max() + margin - grid.x0) // grid.cell + 2))
  r0 = int(max(0, (xy[:, 1].min() - margin - grid.y0) // grid.cell))
  r1 = int(min(grid.nrow, (xy[:, 1].max() + margin - grid.y0) // grid.cell + 2))
  return HeightGrid(z=grid.z[r0:r1, c0:c1], x0=grid.x0 + c0 * grid.cell, y0=grid.y0 + r0 * grid.cell, cell=grid.cell)


def fine(grid: HeightGrid, cell: float = 0.5) -> HeightGrid:
  """Bilinear resample to the 0.5 m surface the training tiles use (1 m lidar cells otherwise show as facets)."""
  from scipy import ndimage
  k = grid.cell / cell
  z = ndimage.zoom(grid.z, k, order=1)
  return HeightGrid(z=z, x0=grid.x0, y0=grid.y0, cell=(grid.ncol - 1) * grid.cell / (z.shape[1] - 1))


def trail_geoms(grid: HeightGrid, xy: np.ndarray, width: float = 0.9, every: int = 2) -> list[str]:
  """Visual-only trail ribbon: thin boxes lying on the ground along the route."""
  out = []
  for i in range(0, len(xy) - every, every):
    a, b = xy[i], xy[i + every]
    m = (a + b) / 2
    h, n = grid.height_normal(np.array([m[0]]), np.array([m[1]]))
    t = np.array([b[0] - a[0], b[1] - a[1], 0.0])
    t[2] = -(n[0][0] * t[0] + n[0][1] * t[1]) / n[0][2]
    t /= np.linalg.norm(t)
    z = n[0] / np.linalg.norm(n[0])
    y = np.cross(z, t)
    R = np.column_stack([t, y, z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    L = float(np.linalg.norm(b - a)) / 2 + 0.05
    out.append(f'<geom type="box" size="{L:.3f} {width / 2} 0.01" pos="{m[0]:.3f} {m[1]:.3f} {h[0] + 0.012:.3f}" '
               f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" rgba="{TRAIL_RGBA[0]} {TRAIL_RGBA[1]} '
               f'{TRAIL_RGBA[2]} 1" contype="0" conaffinity="0" group="0"/>')
  return out


def hiking_model(grid: HeightGrid, xy: np.ndarray, satellite: str | None = None, tex: int = 2048, relief: float = 0.0,
                 origin=None) -> mujoco.MjModel:
  """G1 on the course ground. With `satellite` (a course slug) the ground wears the course's aerial photo at `tex`
  pixels with the trail painted in, and the sun is lowered so roots and rocks cast shadows; otherwise a tan ground
  with a trail ribbon."""
  if satellite is None:
    model = g1_model(grid, skis=False, extras=trail_geoms(grid, xy))
    snow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "snow")
    model.mat_rgba[snow] = GROUND_RGBA
    model.mat_specular[snow] = 0.05
    return model
  import re
  import tempfile
  from hikesim.sat import drape
  model = g1_model(grid, skis=False)
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "scene.xml"
    mujoco.mj_saveLastXML(str(path), model)   # the compiled scene back as XML, to resize the ground texture
    xml = path.read_text()
  xml = re.sub(r'<texture[^>]*name="snowtex"[^>]*/>',
               f'<texture type="2d" name="snowtex" builtin="flat" rgb1="0.45 0.4 0.3" width="{tex}" height="{tex}"/>', xml)
  xml = re.sub(r'(<light name="sun"[^>]*?)dir="[^"]*"', r'\1dir="0.62 0.38 -0.55"', xml)
  model = mujoco.MjModel.from_xml_string(xml)
  model.hfield_data[:] = grid.hfield()["data"].ravel()
  drape(model, grid, satellite, route=xy, relief=relief, origin=origin)
  return model


def drop(model, data, grid: HeightGrid, xy, yaw: float) -> None:
  """Stand the robot at xy facing yaw, feet just touching the ground."""
  data.qpos[:] = 0
  data.qpos[7:] = DEFAULT_ANGLES
  data.qpos[0:3] = (xy[0], xy[1], 0.0)
  data.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
  data.qvel[:] = 0
  mujoco.mj_kinematics(model, data)
  feet = [g for g in range(model.ngeom) if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
  gap = min(data.geom_xpos[g][2] - model.geom_size[g][0]
            - grid.height_normal(data.geom_xpos[g][None, 0], data.geom_xpos[g][None, 1])[0][0] for g in feet)
  data.qpos[2] -= gap - 0.002
  mujoco.mj_forward(model, data)


def steepest(route: dict, length: int) -> int:
  g = np.abs(np.array(route["grade_deg"]))
  if len(g) <= length:
    return 0
  win = np.convolve(g, np.ones(length) / length, mode="valid")
  return int(np.argmax(win))


def hike(course: str, start: int = 0, length: int = 120, mode: str = "walk", speed: float = 1.0,
         careful_above: float = 0.0, lookahead: int = 3, seconds: float | None = None, planner: Planner | None = None,
         name: str | None = None, log=print) -> dict:
  grid, route = load(course)
  rxy = np.stack([route["x"], route["y"]], 1)
  end = min(len(rxy) - 1, start + length)
  seg = rxy[start:end + 1]
  g = crop(grid, seg)
  model = hiking_model(g, seg)
  data = mujoco.MjData(model)
  d0 = seg[min(4, len(seg) - 1)] - seg[0]
  yaw0 = float(np.arctan2(d0[1], d0[0]))
  drop(model, data, g, seg[0], yaw0)
  sonic = SonicController(model)
  planner = planner or Planner()
  sonic.reset(data, planner.start(seg[0], yaw0))
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  decim = int(round(0.02 / model.opt.timestep))
  seconds = seconds or length / max(speed, 0.3) * 2.5 + 5
  grades = np.abs(np.array(route["grade_deg"]))

  idx, cur_mode, fell, frames, prog, t_plan = 0, None, None, [], [], 0.0
  t_wall = time.time()
  ticks = int(seconds * 50)
  for tick in range(ticks):
    p = data.xpos[pelvis]
    lo, hi = max(0, idx - 5), min(len(seg), idx + 40)
    idx = lo + int(np.argmin(np.linalg.norm(seg[lo:hi] - p[:2], axis=1)))
    if idx >= len(seg) - 2:
      break
    want = mode
    if careful_above > 0 and grades[start + idx: start + idx + 8].max(initial=0) > careful_above:
      want = "careful"
    if tick == 50 or (tick > 50 and (tick % 25 == 0 or want != cur_mode)):  # stand 1 s, then walk, replan at 2 Hz
      tgt = seg[min(len(seg) - 1, idx + lookahead)]
      direction = tgt - p[:2]
      t0 = time.time()
      ref = planner.replan(sonic.tick, want, direction, speed=speed)
      t_plan += time.time() - t0
      sonic.ref, sonic.tick, cur_mode = ref, 0, want
    sonic.step(data)
    for _ in range(decim):
      mujoco.mj_step(model, data)
    up = data.xmat[pelvis].reshape(3, 3)[:, 2]
    h = g.height_normal(data.xpos[pelvis][None, 0], data.xpos[pelvis][None, 1])[0][0]
    if tick % 2 == 0:
      frames.append(data.qpos.copy())
      prog.append(idx)
    if not np.isfinite(data.qpos).all() or up[2] < 0.5 or data.xpos[pelvis][2] - h < 0.4:
      fell = round(tick / 50, 2)
      break
  t_sim = (tick + 1) / 50
  walked = float(idx)
  res = {
    "course": course, "area": route["area"], "start_m": start, "segment_m": int(end - start), "mode": mode,
    "speed_cmd": speed, "careful_above_deg": careful_above,
    "result": f"fell at {fell} s" if fell else ("finished" if idx >= len(seg) - 2 else f"upright {t_sim:.0f} s"),
    "walked_m": round(walked, 1), "sim_s": round(t_sim, 1), "mean_speed_mps": round(walked / max(t_sim - 1, 1e-3), 2),
    "max_grade_deg": round(float(grades[start:start + idx + 1].max(initial=0)), 1),
    "mean_abs_grade_deg": round(float(grades[start:start + idx + 1].mean()), 1),
    "wall_s": round(time.time() - t_wall, 1), "planner_s": round(t_plan, 1),
  }
  out = RUNS / course / (name or f"s{start}_l{length}_{mode}")
  out.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out / "frames.npz", qpos=np.array(frames, np.float32), progress=np.array(prog, np.int32),
                      grid_origin=np.array([g.x0, g.y0, g.cell]), start=start, end=end)
  (out / "run.json").write_text(json.dumps(res, indent=1))
  log(json.dumps(res))
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("course")
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--length", type=int, default=120)
  ap.add_argument("--steepest", type=int, default=0, help="walk the steepest stretch of this many metres instead")
  ap.add_argument("--mode", default="walk")
  ap.add_argument("--speed", type=float, default=1.0)
  ap.add_argument("--careful-above", type=float, default=0.0, help="switch to the careful gait above this grade")
  ap.add_argument("--seconds", type=float, default=None)
  ap.add_argument("--name", default=None)
  a = ap.parse_args()
  start, length = a.start, a.length
  if a.steepest:
    _, route = load(a.course)
    start, length = steepest(route, a.steepest), a.steepest
  hike(a.course, start, length, a.mode, a.speed, a.careful_above, seconds=a.seconds, name=a.name)


if __name__ == "__main__":
  main()
