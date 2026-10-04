"""Showcase: one continuous run down a real course through its giant slalom gates, filmed like TV coverage.

The trained policy skis the full course (real gates from gates.json); a broadcast-style caption names the move
as it happens (carving left/right, edge change, skid check, glide, air, landing, crash), with speed, slope, edge
angles, lateral g and gate count. Cameras cut between a level side camera (the slope angle shows against the
horizon), a chase cam, a low front camera and a wide shot (skisim.cinema: never under the snow, never inside a
tree or gate pole). Carving tracks are drawn in the snow. The look is render-only (skisim.scene, pretty=Look):
groomed corduroy piste through natural snow, rock on steep slopes, sun shadows, haze, a mountain sky, 8-bit trees.
--plain renders the old flat look.

Run: PYTHONPATH=. .venv-train/bin/python -m train.showcase --course out/courses/bormio-stelvio --ckpt runs_live/<run>/model_N.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from skisim.cinema import Director, Obstacles, lens_position, piste_mask, snow_lift
from skisim.draw import draw_ski_poles, draw_trees
from skisim.scene import Look, aim_sun, g1_model, render_pretty
from skisim.trees import course_trees
from skisim.ski import SkiParams
from skisim.terrain import HeightGrid
from train.ski_env import CANOPY, ROOT, SkiEnv, mat_to_quat
from train.train import load_policy, runner_cfg

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
W, H, FPS = 1280, 720, 30


def font(size, bold=False):
  for p in (f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf", "/System/Library/Fonts/Helvetica.ttc",
            f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"):  # Linux (Modal)
    if Path(p).exists():
      return ImageFont.truetype(p, size)
  return ImageFont.load_default()


def course_mosaic(course: Path, out: Path, snow: str = "groomed"):
  """A one-tile mosaic around the course so SkiEnv can load it, with that snow (skisim.perturb.SNOW)."""
  from skisim.perturb import SNOW

  out.mkdir(parents=True, exist_ok=True)
  g = json.loads((course / "grid.json").read_text())
  z = np.load(course / "elevation.npy").astype(np.float32)
  np.save(out / "mosaic_elevation.npy", z)
  (out / "mosaic.json").write_text(json.dumps({"x0": g["x0"], "y0": g["y0"], "cell": g["cell"], "nrow": z.shape[0],
                                               "ncol": z.shape[1], "tiles": [{"index": 0, "y_center_m": 0.0, "source": course.name,
                                                                               "snow": snow, "snow_params": SNOW[snow],
                                                                               "mean_slope_deg": 0.0}]}))
  return HeightGrid(z=z.astype(float), x0=g["x0"], y0=g["y0"], cell=g["cell"])


class Course:
  def __init__(self, course: Path):
    run = json.loads((course / "run.json").read_text())
    self.line = np.asarray(run["course"], float)  # x, y, z, s
    self.length = float(self.line[-1, 3])
    gates = json.loads((course / "gates.json").read_text())["gates"]
    self.gates = gates
    self.turn = np.array([g["turn_pole"][:2] for g in gates])
    self.outer = np.array([g["outer_pole"][:2] for g in gates])
    self.s = np.array([g["s"] for g in gates])
    self._set_aim()

  def _set_aim(self):
    # Aim at the middle of each gate: the policy tracks heading to about 10 degrees, so a racer's line 0.8 m off the
    # turning pole left no margin and most gates were missed.
    d = self.outer - self.turn
    self.aim = self.turn + d * 0.5
    self.width = np.linalg.norm(d, axis=1)

  def respace(self, spacing: float, offset: float = 2.5, width: float = 3.65, first: float = 20.0):
    """Replace the course's own gates with alternating gates every `spacing` m along the run (rounder turns)."""
    gates, k = [], 0
    s = first
    while s < self.length - 20:
      h = self.heading_at(s)
      c = self.point_at(s)
      nrm = np.array([-np.sin(h), np.cos(h)])  # left of the course direction
      side = 1 if k % 2 == 0 else -1
      turn = c + side * offset * nrm
      outer = turn + side * width * nrm
      poles = [np.append(turn, 0), np.append(turn + side * 0.55 * nrm, 0), np.append(outer - side * 0.55 * nrm, 0), np.append(outer, 0)]
      gates.append({"id": k, "s": s, "color": "red" if k % 2 == 0 else "blue", "turn_pole": list(np.append(turn, 0)),
                    "outer_pole": list(np.append(outer, 0)), "poles": [list(p) for p in poles]})
      s += spacing
      k += 1
    self.gates = gates
    self.turn = np.array([g["turn_pole"][:2] for g in gates])
    self.outer = np.array([g["outer_pole"][:2] for g in gates])
    self.s = np.array([g["s"] for g in gates])
    self._set_aim()

  def s_of(self, xy):
    d = np.linalg.norm(self.line[:, :2] - xy, axis=1)
    return float(self.line[np.argmin(d), 3])

  def heading_at(self, s):
    i = int(np.clip(np.searchsorted(self.line[:, 3], s), 1, len(self.line) - 1))
    v = self.line[i, :2] - self.line[i - 1, :2]
    return float(np.arctan2(v[1], v[0]))

  def point_at(self, s):
    return np.array([np.interp(s, self.line[:, 3], self.line[:, 0]), np.interp(s, self.line[:, 3], self.line[:, 1])])


def clear_gates_of_trees(trees: np.ndarray, course: "Course", gap: float = 5.0) -> np.ndarray:
  """Drop trees whose canopy comes within `gap` m of a gate center: trees stay on the piste between gates, where the
  planner steers around them, but never block the gate itself."""
  if len(trees) == 0 or len(course.turn) == 0:
    return trees
  mid = (course.turn + course.outer) / 2
  d = np.hypot(trees[:, None, 0] - mid[None, :, 0], trees[:, None, 1] - mid[None, :, 1]).min(1) - CANOPY * trees[:, 2]
  return trees[d >= gap]


def steer_around_trees(xy, aim, trees, look=20.0, clear=2.5):
  """If a canopy (CANOPY x the tree radius, as drawn) lies within `clear` m of the straight line to the aim point
  (within `look` m), aim beside it."""
  if len(trees) == 0:
    return aim
  d = aim - xy
  L = float(np.linalg.norm(d))
  if L < 1e-3:
    return aim
  u = d / L
  rel = trees[:, :2] - xy
  along = rel @ u
  side = rel @ np.array([-u[1], u[0]])
  near = (along > 0.5) & (along < min(look, L + 3)) & (np.abs(side) < CANOPY * trees[:, 2] + clear)
  if not near.any():
    return aim
  j = np.flatnonzero(near)[np.argmin(along[near])]
  # Pass on the side with more room: away from the trunk, toward where the aim already leans.
  sgn = -np.sign(side[j]) if abs(side[j]) > 0.3 else (1.0 if (aim - xy) @ np.array([-u[1], u[0]]) >= 0 else -1.0)
  offset = CANOPY * trees[j, 2] + clear + 0.4
  return xy + u * along[j] + np.array([-u[1], u[0]]) * (side[j] + sgn * offset)


def spawn(env: SkiEnv, xy, heading, speed):
  ids = torch.tensor([0])
  env.reset_idx(ids)
  env.use_gates[0] = False
  x, y = torch.tensor([float(xy[0])]), torch.tensor([float(xy[1])])
  _, nrm = env.grid.height_normal(x, y)
  fwd = torch.tensor([[np.cos(heading), np.sin(heading), 0.0]], dtype=torch.float32)
  xa = fwd - (fwd * nrm).sum(-1, keepdim=True) * nrm
  xa = xa / xa.norm(dim=-1, keepdim=True)
  R = torch.stack([xa, torch.cross(nrm, xa, dim=-1), nrm], -1)
  pts = torch.einsum("nij,kj->nki", R, env.ski_pts_pelvis)
  h, _ = env.grid.height_normal(x[:, None] + pts[..., 0], y[:, None] + pts[..., 1])
  env.qpos[0, 0:3] = torch.tensor([x[0], y[0], (h - pts[..., 2]).max() + 0.003])
  env.qpos[0, 3:7] = mat_to_quat(R)[0]
  env.qvel[0] = 0
  env.qvel[0, 0:3] = xa[0] * speed
  env._forward()
  env._refresh_history(ids, fill=True)
  env.obs = env._observations()


def keep_above_snow(cam, grid: HeightGrid, clearance: float = 1.2):
  """Raise the camera (steeper down-angle) until it is clear of the terrain under it.

  On a steep pitch a chase camera behind and above the skier can end up inside the slope, which renders the
  underside of the snow as long white sheets across the frame. The line of sight near the lens is checked too, and
  if even a steep down-angle is not enough the camera moves in (skisim.cinema.snow_lift). The rendered shots go
  through skisim.cinema.Director, which also keeps the lens out of trees and gate poles.
  """
  cam.elevation, cam.distance = snow_lift(grid, cam.lookat, cam.distance, cam.azimuth, cam.elevation, clearance)


def draw_gate_geoms(scene, course: Course, grid: HeightGrid, next_gate: int):
  for k, g in enumerate(course.gates):
    col = [0.86, 0.08, 0.1, 1.0] if g.get("color", "red") == "red" else [0.08, 0.24, 0.92, 1.0]
    if k < next_gate:
      col[3] = 0.35
    poles = np.asarray(g["poles"])
    for p in poles:
      if scene.ngeom >= scene.maxgeom - 4:
        return
      h, _ = grid.height_normal(np.array([p[0]]), np.array([p[1]]))
      mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.022, 0.66, 0]),
                          np.array([p[0], p[1], h[0] + 0.66]), np.eye(3).ravel(), np.array(col, np.float32))
      scene.ngeom += 1
    for a, b in ((poles[0], poles[1]), (poles[2], poles[3])):  # a panel between each pole pair
      if scene.ngeom >= scene.maxgeom - 4:
        return
      c = (a + b) / 2
      h, _ = grid.height_normal(np.array([c[0]]), np.array([c[1]]))
      ang = np.arctan2(b[1] - a[1], b[0] - a[0])
      Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
      half = np.linalg.norm(b[:2] - a[:2]) / 2
      mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([half, 0.006, 0.17]),
                          np.array([c[0], c[1], h[0] + 0.95]), Rz.ravel(), np.array(col, np.float32))
      scene.ngeom += 1


def draw_tracks(scene, tracks):
  """Carved grooves: thin flat ribbons lying on the snow, a shade darker than it."""
  for trail in tracks:
    pts = list(trail)
    for a, b in zip(pts[:-1], pts[1:]):
      if scene.ngeom >= scene.maxgeom - 2:
        return
      d = b - a
      L = float(np.linalg.norm(d))
      if L < 1e-3 or L > 2.0:  # skip duplicates and the gap after a restart
        continue
      x = d / L
      z = np.array([0.0, 0.0, 1.0])
      y = np.cross(z, x)
      y /= np.linalg.norm(y) + 1e-9
      z = np.cross(x, y)
      mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([L / 2, 0.022, 0.002]),
                          (a + b) / 2, np.column_stack([x, y, z]).ravel(), np.array([0.70, 0.75, 0.84, 1.0], np.float32))
      scene.ngeom += 1


class MoveNamer:
  """Names the move from the physics: edges, ski loads, sideways slip, yaw rate and contact."""

  def __init__(self):
    self.edge_sign_hist = deque(maxlen=12)
    self.air_t = 0.0
    self.since_air = 9.0
    self.label = "Start"

  def update(self, N, edge, v_side, yaw_rate, speed, fell, dt):
    if fell:
      self.label = "Crash"
      return self.label
    airborne = N.max() < 15
    if airborne:
      self.air_t += dt
      self.since_air = 0.0
    else:
      if self.air_t > 0.08:
        self.since_air = 0.0
      self.air_t = 0.0
      self.since_air += dt
    e = float(np.mean(edge))
    sign = int(np.sign(e)) if abs(e) > np.radians(6) else 0
    self.edge_sign_hist.append(sign)
    changed = len({s for s in self.edge_sign_hist if s != 0}) > 1
    if self.air_t > 0.08:
      self.label = "Air"
    elif self.since_air < 0.4:
      self.label = "Landing"
    elif changed and abs(e) < np.radians(14):
      self.label = "Edge change"
    elif abs(e) > np.radians(10) and np.sign(edge[0]) == np.sign(edge[1]) and abs(yaw_rate) > 0.25:
      self.label = "Carving left" if e > 0 else "Carving right"
    elif np.mean(np.abs(v_side)) > 0.8 and speed > 2:
      self.label = "Skid check"
    elif speed > 3:
      self.label = "Gliding"
    else:
      self.label = "Setting off"
    return self.label


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--course", default=str(ROOT / "out/courses/bormio-stelvio"))
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--speed", type=float, default=9.0)
  ap.add_argument("--out", default=str(ROOT / "out/showcase"))
  ap.add_argument("--no-render", action="store_true")
  ap.add_argument("--gate-spacing", type=float, default=0.0, help="re-space gates along the run (0 keeps the course's own)")
  ap.add_argument("--gate-offset", type=float, default=2.5)
  ap.add_argument("--free", action="store_true", help="gates only set the turn rhythm; not drawn or scored, HUD counts turns")
  ap.add_argument("--no-trees", action="store_true")
  ap.add_argument("--trees-in-run", type=int, default=14)
  ap.add_argument("--plain", action="store_true", help="the old flat look: no sky, haze, corduroy, rock or sun shadows")
  ap.add_argument("--snow", default="groomed", help="groomed, hardpack, ice, soft, slush or powder (skisim.perturb.SNOW)")
  ap.add_argument("--start-s", type=float, default=3.0, help="start this many meters down the course")
  ap.add_argument("--style", default="default", help="'racing' for checkpoints trained with --style racing (s3 on)")
  ap.add_argument("--governor", type=float, default=0.5, help="speed governor gain (0 = always ask for --speed; 1.5 asked for near-stops the policy falls doing)")
  ap.add_argument("--sidecut", type=float, default=0.0, help="ski sidecut radius in m (0 = current default; 17 for checkpoints before v20)")
  args = ap.parse_args()
  course_dir = Path(args.course)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  grid = course_mosaic(course_dir, out / f"_mosaic_{course_dir.name}", args.snow)
  course = Course(course_dir)
  trees = course_trees(course.line, in_run=0 if args.no_trees else args.trees_in_run) if not args.no_trees else np.zeros((0, 5), np.float32)
  trees = clear_gates_of_trees(trees, course)
  np.save(out / f"_mosaic_{course_dir.name}" / "trees.npy", trees)
  if len(trees):  # keep every turn marker at least 4 m from a canopy
    for k in range(len(course.aim)):
      for _ in range(4):
        dd = np.hypot(trees[:, 0] - course.aim[k, 0], trees[:, 1] - course.aim[k, 1]) - CANOPY * trees[:, 2]
        j = int(np.argmin(dd))
        if dd[j] >= 4.0:
          break
        away = course.aim[k] - trees[j, :2]
        course.aim[k] = course.aim[k] + away / (np.linalg.norm(away) + 1e-6) * (4.0 - dd[j] + 0.2)
  if args.gate_spacing > 0:
    course.respace(args.gate_spacing, args.gate_offset)
  stats = json.loads((course_dir / "stats.json").read_text())

  env = SkiEnv(num_envs=1, device="cpu", use_graph=False, seed=3, mosaic=out / f"_mosaic_{course_dir.name}", backend="mujoco")
  env.auto_reset = False
  from rsl_rl.runners import OnPolicyRunner

  policy = load_policy(env, args.ckpt)
  it = int(Path(args.ckpt).stem.split("_")[-1])

  env.set_style(args.style)
  if args.sidecut > 0:
    env.p.sidecut_radius = args.sidecut  # shared with env.contact
  start = course.point_at(args.start_s)
  spawn(env, start, course.heading_at(args.start_s), 2.0)

  model = None
  if not args.no_render:
    # Render-only look: a groomed piste (corduroy along the course) through natural snow, rock on steep slopes off
    # the piste, sun shadows, haze and a mountain sky. Physics is unaffected: SkiEnv has its own model and grid.
    run = course.line[-1, :2] - course.line[0, :2]
    look_spec = Look(groom_deg=float(np.degrees(np.arctan2(run[1], run[0]))), piste=piste_mask(grid, course.line[:, :2]))
    model = g1_model(grid, SkiParams(), pretty=False if args.plain else look_spec)
  data = mujoco.MjData(model) if model else None
  renderer = mujoco.Renderer(model, H, W, max_geom=20000) if model else None
  poles = None if args.free or not course.gates else np.array([p[:2] for g in course.gates for p in g["poles"]], float)
  director = Director(grid, Obstacles(grid, trees, poles))
  lens_snow, lens_gap = [], []  # per frame: lens height above the snow, lens distance to the nearest tree/pole footprint
  cams = ["wide", "chase", "side", "front", "side", "chase"]
  shot_len = 6.0
  frames_dir = out / f"_frames_{course_dir.name}"
  if frames_dir.exists():
    shutil.rmtree(frames_dir)
  frames_dir.mkdir()
  big, mid, small = font(54, True), font(26, True), font(22)
  namer = MoveNamer()
  tracks = [deque(maxlen=500), deque(maxlen=500)]
  next_gate, hits, crashes, crash_t = int(np.searchsorted(course.s, args.start_s)), 0, 0, 0.0
  first_gate = next_gate
  turns, last_carve = 0, ""
  tree_crashes = 0
  cmd = 0.0
  t, frame = 0.0, 0
  prev_v = env.qvel[0, 0:3].numpy().copy()
  log = []
  crash_events = []
  speed = 2.0
  steps_per_frame = max(1, round(1 / (FPS * 0.02)))
  substep_t = 0.02
  look = None
  while t < args.seconds:
    xy = env.qpos[0, 0:2].numpy()
    s_now = course.s_of(xy)
    # Gate bookkeeping: a gate is passed once the skier is past it along the course.
    while next_gate < len(course.s) and s_now > course.s[next_gate]:
      c = (course.turn[next_gate] + course.outer[next_gate]) / 2
      hits += int(np.linalg.norm(xy - c) < course.width[next_gate] / 2 + 0.6)
      next_gate += 1
    if s_now > course.length - 15 or next_gate >= len(course.s):
      break
    # Command: aim outside the next turning pole, looking ahead to the gate after it near the gate.
    a = course.aim[next_gate]
    if next_gate + 1 < len(course.aim):
      w = np.clip(1 - (course.s[next_gate] - s_now) / 4.0, 0, 1) * 0.3  # start the next turn late, through the gate
      a = (1 - w) * a + w * course.aim[next_gate + 1]
    a = steer_around_trees(xy, a, trees, look=float(np.clip(2.5 * speed, 12.0, 35.0)))  # about 2.5 s ahead
    _, nrm = env.grid.height_normal(torch.tensor([float(xy[0])]), torch.tensor([float(xy[1])]))
    # Commands are relative to the run direction, exactly as in training: the training tiles are cut along the run
    # (a third of them fall 30+ degrees sideways), not along the fall line. Clamping around the fall line instead
    # pulled the robot straight downhill off every traverse, into the trees and past the gates.
    run_dir = course.heading_at(s_now)
    fall = float(np.arctan2(nrm[0, 1], nrm[0, 0]))  # for the camera director
    bearing = float(np.arctan2(a[1] - xy[1], a[0] - xy[0]))
    slope_deg = float(np.degrees(np.arccos(np.clip(float(nrm[0, 2]), -1.0, 1.0))))
    hmax = 0.9 + 0.4 * float(np.clip((slope_deg - 12.0) / 15.0, 0.0, 1.0))
    rel = np.clip(np.arctan2(np.sin(bearing - run_dir), np.cos(bearing - run_dir)), -hmax, hmax)
    target = run_dir + rel
    # Only ask for turns the skis can make: heading rate capped by lateral grip (about 6 m/s^2, so 1.5 rad/s at 4 m/s
    # and 0.6 rad/s at 10 m/s), and the command kept within 35 degrees of the actual travel. Asking for a 1.5 rad/s
    # turn at 10 m/s left the robot sliding 50 to 75 degrees outside the command, and that was nearly every crash.
    rate = min(1.5, 6.0 / max(speed, 1.0)) * substep_t
    cmd += np.clip(np.arctan2(np.sin(target - cmd), np.cos(target - cmd)), -rate, rate) if t > 0 else target - cmd
    if t > 0 and speed > 2.0:
      travel_now = float(np.arctan2(env.qvel[0, 1], env.qvel[0, 0]))
      cmd = travel_now + float(np.clip(np.arctan2(np.sin(cmd - travel_now), np.cos(cmd - travel_now)), -0.6, 0.6))
    env.cmd_heading[0] = cmd
    # Speed governor: ask for less on steep pitches, and when already too fast ask for much less (down to a stop
    # command), which triggers the braking the policy learned from stop commands.
    target_v = args.speed * float(np.clip(1.0 - (slope_deg - 12.0) / 30.0, 0.6, 1.0))
    env.cmd_speed[0] = float(np.clip(target_v - args.governor * max(0.0, speed - target_v), 0.0, target_v))
    env.cmd_timer[0] = 1e9
    with torch.no_grad():
      act = policy(env.get_observations())
    env.step(act)
    t += substep_t
    fell = bool(env.last_fell[0])
    d = env.ski_diag
    v = env.qvel[0, 0:3].numpy().copy()
    acc = (v - prev_v) / substep_t
    prev_v = v
    speed = float(np.linalg.norm(v[:2]))
    lat_g = float(abs(np.cross(np.append(v[:2] / max(speed, 1e-3), 0), np.append(acc[:2], 0))[2]) / 9.81)
    N, edge, v_side = d["N"][0].numpy(), d["edge"][0].numpy(), d["v_l"][0].numpy()
    label = namer.update(N, edge, v_side, float(env.qvel[0, 5]), speed, fell, substep_t)
    if label.startswith("Carving") and label != last_carve:
      turns += int(last_carve != "")
      last_carve = label
    log.append({"t": round(t, 2), "move": label, "speed": round(speed, 2), "s": round(s_now, 1), "gate": next_gate})
    # Ski tracks: bottom center of each ski on the snow.
    for k, b in enumerate(env.ski_ids.tolist()):
      if N[k] > 20:
        tracks[k].append(env.xpos[0, b].numpy().copy() - np.array([0, 0, 0.015]))
    if fell:
      travel = float(np.arctan2(v[1], v[0]))
      crash_events.append({"t": round(t, 1), "s": round(s_now, 1), "speed": round(speed, 1), "slope": round(slope_deg, 1),
                           "tree": bool(env.last_tree_hit[0]), "cmd_vs_fall_deg": round(float(np.degrees(rel)), 0),
                           "travel_err_deg": round(float(np.degrees(np.arctan2(np.sin(travel - cmd), np.cos(travel - cmd)))), 0),
                           "off_line_m": round(float(np.linalg.norm(xy - course.point_at(s_now))), 1), "move": label})
      crashes += 1
      tree_crashes += int(bool(env.last_tree_hit[0]))
      crash_t = t
      for _ in range(int(1.0 / 0.02)):  # hold the crash on screen for a second, physics still running
        env.step(torch.zeros(1, env.num_actions))
      # Restart 4 m back, or 12 m past the spot if it already crashed there (no crash loops at one bad pitch or tree).
      repeat = any(abs(e["s"] - s_now) < 15.0 for e in crash_events[:-1])
      s_back = min(s_now + 12.0, course.length - 20.0) if repeat else max(s_now - 4.0, 3.0)
      spawn(env, course.point_at(s_back), course.heading_at(s_back), 4.0)
      cmd = course.heading_at(s_back)
      for tr in tracks:
        tr.append(tr[-1] + np.array([0, 0, 100.0]) if tr else np.zeros(3))  # break the trail
    if renderer is None or (int(t / 0.02) % steps_per_frame):
      continue
    # Render.
    data.qpos[:] = env.qpos[0].numpy()
    mujoco.mj_forward(model, data)
    p = data.xpos[1].copy() + v * 0.06  # lead the skier slightly so fast runs stay framed
    look = p if look is None or np.linalg.norm(p - look) > 6 else 0.45 * look + 0.55 * p
    travel = np.degrees(np.arctan2(v[1], v[0])) if speed > 1 else np.degrees(cmd)
    shot = cams[int(t // shot_len) % len(cams)]
    # Chase, side, front or wide (skisim.cinema.SHOTS), kept above the snow and clear of trees and poles; the
    # skier sits above the caption bar.
    cam = director.frame(shot, look + np.array([0, 0, -0.45]), travel, np.degrees(fall), steps_per_frame * substep_t)
    lens = lens_position(cam.lookat, cam.distance, cam.azimuth, cam.elevation)
    lens_snow.append(float(lens[2] - grid.height_normal(lens[:1], lens[1:2])[0][0]))
    lens_gap.append(director.obs.gap(lens))
    if not args.plain:
      aim_sun(model, data, look)  # the sun's shadow map travels with the skier
    renderer.update_scene(data, cam)
    if not args.free:
      draw_gate_geoms(renderer.scene, course, grid, next_gate)
    draw_trees(renderer.scene, trees, grid, p[:2], radius=120)
    draw_ski_poles(renderer.scene, model, data)
    if shot in ("wide", "side"):  # near-lens tracks in chase/front shots read as clutter
      draw_tracks(renderer.scene, tracks)
    img = Image.fromarray(renderer.render() if args.plain else render_pretty(renderer))
    dr = ImageDraw.Draw(img, "RGBA")
    slope_here = float(np.degrees(np.arccos(float(nrm[0, 2]))))
    dr.rectangle([0, 0, W, 54], fill=(10, 16, 30, 170))
    dr.text((22, 12), f"{course_dir.name.replace('-', ' ').upper()}", font=mid, fill=(255, 255, 255))
    dr.text((W - 520, 15), f"Unitree G1 · trained policy iteration {it}", font=small, fill=(200, 210, 225))
    shown = "Crash" if t - crash_t < 1.2 and crashes else label
    col = (240, 80, 95) if shown == "Crash" else (255, 255, 255)
    dr.rectangle([0, H - 150, W, H], fill=(10, 16, 30, 185))
    dr.rectangle([22, H - 138, 30, H - 74], fill=(214, 32, 42, 255))
    dr.text((44, H - 142), shown.upper(), font=big, fill=col)
    tally = f"turns {turns}" if args.free else f"gate {min(next_gate + 1, len(course.s))}/{len(course.s)}    clean gates {hits}"
    line = (f"{speed * 3.6:4.0f} km/h    slope {slope_here:4.1f}°    edges {np.degrees(edge[0]):+4.0f}° / {np.degrees(edge[1]):+4.0f}°    "
            f"{lat_g:3.1f} g    {tally}    crashes {crashes}")
    dr.text((44, H - 62), line, font=small, fill=(215, 225, 238))
    img.save(frames_dir / f"{frame:05d}.jpg", quality=90)
    frame += 1
  summary = {"course": course_dir.name, "checkpoint": args.ckpt, "seconds": round(t, 1), "distance_along_course_m": round(course.s_of(env.qpos[0, 0:2].numpy()), 1),
             "gates_reached": int(next_gate - first_gate), "clean_gates": int(hits), "crashes": crashes, "tree_crashes": tree_crashes,
             "trees": int(len(trees)), "frames": frame, "turns": turns,
             "moves": {m: sum(1 for r in log if r["move"] == m) for m in sorted({r["move"] for r in log})},
             "snow": args.snow,
             "speed_mean": round(float(np.mean([r["speed"] for r in log])), 1) if log else 0.0,
             "speed_p90": round(float(np.percentile([r["speed"] for r in log], 90)), 1) if log else 0.0,
             "crash_events": crash_events,
             "course_mean_slope_deg": stats["course"]["mean_slope_deg"], "course_max_slope_deg": stats["course"]["max_slope_10m_deg"]}
  if lens_snow:  # camera checks over every rendered frame
    summary["camera"] = {"min_lens_above_snow_m": round(min(lens_snow), 2),
                         "min_lens_gap_to_tree_or_pole_m": round(min(lens_gap), 2) if np.isfinite(min(lens_gap)) else None}
  (out / f"{course_dir.name}.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps(summary), flush=True)
  if frame:
    mp4 = out / f"{course_dir.name}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(mp4)], check=True)
    print("video:", mp4, flush=True)


if __name__ == "__main__":
  main()
