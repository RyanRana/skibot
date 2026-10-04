"""Evaluate a hiking policy on the real Ann Arbor courses, in exactly the training env (same physics, SONIC wiring and
observations) with the course's lidar ground in place of the tiles and the trail itself to follow.

    PYTHONPATH=. .venv-hike/bin/python -m train.hike_play --ckpt runs_hike/<run>/model_1500.pt --course bird-hills-nature-area --steepest 100
    PYTHONPATH=. .venv-hike/bin/python -m train.hike_play --ckpt runs_hike/<run>/model_1500.pt --bench --workers 3
    PYTHONPATH=. .venv-hike/bin/python -m train.hike_play --zero --bench          # SONIC alone, same env, for comparison

Runs land in out/hike/runs/<course>/<name>/ (frames.npz + run.json), which hikesim.render and hikesim.export_web read.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hikesim.course import COURSES, slug
from hikesim.rough import roughen
from hikesim.walk import crop, fine, load, steepest
from skisim.ski_torch import TorchGrid
from skisim.terrain import HeightGrid
from train.hike_env import HikeEnv, wrap

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "out" / "hike" / "runs"


class CourseEnv(HikeEnv):
  """One robot walking a stretch of a real course: trail following instead of the tile line, no curriculum."""

  def __init__(self, course: str, start: int, length: int, clip: str = "walk_1", bumps: float = 1.0, **kw):
    self.course, self.seg_start, self.seg_len, self.clip_name, self.bumps = course, start, length, clip, bumps
    super().__init__(num_envs=1, device="cpu", use_graph=False, **kw)
    self.auto_reset = False
    self.max_episode_length = 10 ** 9
    names = list(np.load(kw.get("clips", ROOT / "out/hike/clips.npz"))["names"])
    self.clip_id = names.index(clip)
    q, dq = self.clip_q[self.clip_id], self.clip_dq[self.clip_id]
    # Loop the clip for long walks: jump from near its end back to the walking frame that matches it best.
    T = self.clip_T
    self.loop_from = T - 60
    cost = (q[300:T - 400] - q[self.loop_from]).norm(dim=-1) + 0.05 * (dq[300:T - 400] - dq[self.loop_from]).norm(dim=-1)
    self.loop_to = 300 + int(cost.argmin())
    self.reset_idx(torch.arange(1))
    self._forward()
    self._refresh_history(torch.arange(1), fill=True)
    self._update_command(rate_limit=False)
    self.obs = self._observations()

  def _load_terrain(self, mosaic) -> HeightGrid:
    grid, route = load(self.course)
    rxy = np.stack([route["x"], route["y"]], 1)
    self.end = min(len(rxy) - 1, self.seg_start + self.seg_len)
    seg = rxy[self.seg_start:self.end + 1]
    origin = json.loads((ROOT / "out/hike/courses" / self.course / "grid.json").read_text())["utm_origin"]
    g = roughen(crop(grid, seg), origin=origin, amp=self.bumps) if self.bumps > 0 else fine(crop(grid, seg))
    self.route_info = route
    self.seg = torch.tensor(seg, dtype=torch.float32)
    self.prog = 0
    self.grid = TorchGrid(g.z, g.x0, g.y0, g.cell, self.device)
    grades = np.array(route["grade_deg"][self.seg_start:self.end + 1])
    self.tile_y = torch.zeros(1)
    self.tile_grade = torch.tensor([float(np.mean(grades))])
    self.n_tiles = 1
    self.finish_x, self.half_width = 1e9, 1e9
    return g

  def _progress(self):
    p = self.qpos[0, :2]
    lo, hi = max(0, self.prog - 5), min(len(self.seg), self.prog + 40)
    self.prog = lo + int((self.seg[lo:hi] - p).norm(dim=-1).argmin())
    return self.prog

  def _update_command(self, rate_limit=True):
    i = self._progress()
    tgt = self.seg[min(i + 3, len(self.seg) - 1)]
    d = tgt - self.qpos[0, :2]
    self.cmd_heading = torch.atan2(d[1], d[0])[None]
    if rate_limit:
      step = 0.8 * self.dt * self.decimation
      self.ref_heading = self.ref_heading + torch.clamp(wrap(self.cmd_heading - self.ref_heading), -step, step)
    else:
      self.ref_heading = self.cmd_heading.clone()

  def _trail_offset(self):
    i = min(self.prog, len(self.seg) - 2)
    t = self.seg[i + 1] - self.seg[i]
    t = t / t.norm().clamp_min(1e-6)
    d = self.qpos[0, :2] - self.seg[i]
    return (t[0] * d[1] - t[1] * d[0])[None]

  def _finished(self):
    return torch.tensor([self.prog >= len(self.seg) - 3])

  def _off_course(self):
    return self._trail_offset().abs() > 4.0

  def reset_idx(self, ids):
    if not hasattr(self, "clip_id"):  # base __init__ calls this before the clip is known
      super().reset_idx(ids)
      return
    p0, p1 = self.seg[0], self.seg[min(4, len(self.seg) - 1)]
    yaw = torch.atan2(p1[1] - p0[1], p1[0] - p0[0])
    ox = torch.tensor([-0.12, 0.15, -0.12, 0.15]); oy = torch.tensor([-0.15, -0.15, 0.15, 0.15])
    c, s = torch.cos(yaw), torch.sin(yaw)
    h, _ = self._hn(p0[0] + c * ox - s * oy, p0[1] + s * ox + c * oy)
    self.qpos[0, 0:3] = torch.stack([p0[0], p0[1], h.max() + self.stand_h + 0.01])
    self.qpos[0, 3:7] = torch.stack([torch.cos(yaw / 2), torch.tensor(0.0), torch.tensor(0.0), torch.sin(yaw / 2)])
    self.qpos[0, 7:] = self.default
    self.qvel[0] = 0
    self.qacc_warm[0] = 0
    self.ctrl[0] = self.default
    self.clip[0], self.phase[0], self.walk_from[0] = self.clip_id, 0, 50
    self.ref_heading[0] = yaw
    self.y_target[0] = 0
    self.start_x[0] = p0[0]
    self.lean[0] = 0
    self.episode_length_buf[0] = 0
    self.last_action[0] = 0
    self.prev_action[0] = 0
    self.sonic_last[0] = 0
    self.prog = 0

  def step(self, actions):
    out = super().step(actions)
    if int(self.phase[0]) >= self.loop_from:
      self.phase[0] = self.loop_to
    return out


def walkable_start(course: str, start: int, max_deg: float = 15.0, radius: float = 1.5, back: int = 30) -> int:
  """A spawn point the robot can stand on: ground within `radius` m gentler than max_deg. Looks back up to `back` m
  first (a lead-in, so a steep stretch keeps its climb), then forward. Route ends sit on embankments now and then
  (Barton's trailhead is a 43 deg bank), and a robot spawned there slides before its first step."""
  grid, route = load(course)
  gy, gx = np.gradient(grid.z, grid.cell)
  slope = np.degrees(np.arctan(np.hypot(gx, gy)))
  k = int(np.ceil(radius / grid.cell))

  def ok(i):
    c = int(round((route["x"][i] - grid.x0) / grid.cell)); r = int(round((route["y"][i] - grid.y0) / grid.cell))
    return slope[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1].max() < max_deg

  for i in list(range(start, max(-1, start - back - 1), -1)) + list(range(start + 1, len(route["x"]) - 10)):
    if ok(i):
      return i
  return start


def load_policy(env, ckpt: str):
  from rsl_rl.runners import OnPolicyRunner
  from train.hike_train import runner_cfg
  runner = OnPolicyRunner(env, runner_cfg(1, "tensorboard"), None, device="cpu")
  runner.load(ckpt, map_location="cpu")
  return runner.get_inference_policy(device="cpu")


def evaluate(course: str, start: int, length: int, ckpt: str | None, clip: str = "walk_1", name: str | None = None,
             bumps: float = 1.0,
             max_s: float | None = None) -> dict:
  t_wall = time.time()
  s0 = walkable_start(course, start)
  length -= s0 - start  # keep the same finish line
  env = CourseEnv(course, s0, max(length, 10), clip, bumps=bumps)
  policy = (lambda o: torch.zeros(1, env.num_actions)) if ckpt is None else load_policy(env, ckpt)
  speed = float(env.clip_speed[env.clip_id])
  max_s = max_s or length / max(speed, 0.2) * 2.5 + 10
  obs = env.get_observations()
  frames, prog, result = [], [], None
  for k in range(int(max_s / 0.02)):
    with torch.no_grad():
      obs, *_ = env.step(policy(obs))
    if k % 2 == 0:
      frames.append(env.qpos[0].numpy().copy())
      prog.append(env.prog)
    if bool(env.last_fell[0]):
      result = f"fell at {(k + 1) * 0.02:.1f} s"
      break
    if bool(env._finished()[0]):
      result = "finished"
      break
    if bool(env._off_course()[0]):
      result = f"left the trail at {(k + 1) * 0.02:.1f} s"
      break
  t_sim = (k + 1) * 0.02
  result = result or f"upright {t_sim:.0f} s"
  grades = np.abs(np.array(env.route_info["grade_deg"]))
  s0 = env.seg_start
  label = "sonic_only" if ckpt is None else Path(ckpt).parent.name + "_" + Path(ckpt).stem
  res = {
    "course": course, "area": env.route_info["area"], "start_m": s0, "segment_m": int(env.end - s0), "mode": f"{label} ({clip})",
    "speed_cmd": round(speed, 2), "result": result, "walked_m": float(env.prog), "sim_s": round(t_sim, 1),
    "mean_speed_mps": round(env.prog / max(t_sim - 1.8, 1e-3), 2),
    "max_grade_deg": round(float(grades[s0:s0 + env.prog + 1].max(initial=0)), 1),
    "segment_max_grade_deg": round(float(grades[s0:env.end + 1].max(initial=0)), 1),
    "mean_abs_grade_deg": round(float(grades[s0:s0 + env.prog + 1].mean()), 1),
    "wall_s": round(time.time() - t_wall, 1), "ckpt": ckpt, "bumps": bumps,
  }
  out = RUNS / course / (name or f"{label}_s{s0}_l{length}")
  out.mkdir(parents=True, exist_ok=True)
  g = env.grid
  np.savez_compressed(out / "frames.npz", qpos=np.array(frames, np.float32), progress=np.array(prog, np.int32),
                      grid_origin=np.array([g.x0, g.y0, g.cell]), start=s0, end=env.end)
  (out / "run.json").write_text(json.dumps(res, indent=1))
  return res


def _job(a):
  course, which, start, length, ckpt, clip = a
  tag = "sonic_only" if ckpt is None else Path(ckpt).parent.name + "_" + Path(ckpt).stem
  return {"stretch": which, **evaluate(course, start, length, ckpt, clip, name=f"eval_{tag}_{which}")}


def bench(ckpt: str | None, length: int, workers: int, clip: str) -> list[dict]:
  jobs = []
  for c in COURSES:
    s = slug(c)
    _, route = load(s)
    n = len(route["x"])
    for which, start in (("first", 0), ("middle", max(0, n // 2 - length // 2)), ("steepest", steepest(route, length))):
      jobs.append((s, which, start, length, ckpt, clip))
  rows = []
  with ProcessPoolExecutor(workers) as ex:
    for r in ex.map(_job, jobs):
      rows.append(r)
      print(f"{r['area'][:28]:28s} {r['stretch']:9s} {r['result']:24s} {r['walked_m']:6.1f}/{r['segment_m']} m  "
            f"{r['mean_speed_mps']:4.2f} m/s  steepest {r['segment_max_grade_deg']:4.1f} deg", flush=True)
  fin = sum(r["result"] == "finished" for r in rows)
  print(f"\n{'SONIC alone' if ckpt is None else ckpt}: finished {fin}/{len(rows)}, "
        f"mean {np.mean([r['walked_m'] for r in rows]):.1f} m of {length}, "
        f"steepest stretches finished {sum(r['result'] == 'finished' for r in rows if r['stretch'] == 'steepest')}/{len(COURSES)}")
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default=None)
  ap.add_argument("--zero", action="store_true", help="no policy: SONIC alone on the clips, as the baseline")
  ap.add_argument("--course", default="bird-hills-nature-area")
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--length", type=int, default=100)
  ap.add_argument("--steepest", type=int, default=0)
  ap.add_argument("--clip", default="walk_1")
  ap.add_argument("--bench", action="store_true")
  ap.add_argument("--workers", type=int, default=3)
  ap.add_argument("--name", default=None)
  a = ap.parse_args()
  ckpt = None if a.zero else a.ckpt
  if a.bench:
    rows = bench(ckpt, a.length, a.workers, a.clip)
    tag = "sonic_only" if ckpt is None else Path(ckpt).parent.name + "_" + Path(ckpt).stem
    out = ROOT / "out" / "hike" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=1))
    return
  start, length = a.start, a.length
  if a.steepest:
    _, route = load(a.course)
    start, length = steepest(route, a.steepest), a.steepest
  print(json.dumps(evaluate(a.course, start, length, ckpt, a.clip, a.name)))


if __name__ == "__main__":
  main()
