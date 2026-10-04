"""Many G1s on one real Ann Arbor trail, the way parallel RL sees the world. Untrained: PD stance plus random leans.

A robot that falls respawns at its start slot with a new random lean, like an RL environment reset.
Run: .venv/bin/python -m hikesim.fleet bird-hills-nature-area --n 1000 --cols 40 --seconds 6
Writes out/hike/fleet/: overview.mp4, overview_*.png, stats.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from hikesim.walk import GROUND_RGBA, crop, fine, load
from skisim.scene import g1_model
from skisim.sonic import DEFAULT_ANGLES

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
NQ_ROBOT, NV_ROBOT = 36, 35  # free joint + 29 hinges


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("course", nargs="?", default="bird-hills-nature-area")
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--cols", type=int, default=10)
  ap.add_argument("--spacing", type=float, default=2.5)
  ap.add_argument("--seconds", type=float, default=6.0)
  ap.add_argument("--fps", type=int, default=12)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--lean", type=float, default=0.25, help="scale of the random hip and ankle offsets, radians")
  ap.add_argument("--distance", type=float, default=0.0, help="camera distance, m (0 = fit the grid)")
  ap.add_argument("--elevation", type=float, default=-22.0)
  ap.add_argument("--azimuth", type=float, default=-35.0, help="degrees off looking straight up the grid")
  ap.add_argument("--out", default=str(ROOT / "out/hike/fleet"))
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  grid, route = load(args.course)
  xy = np.column_stack([route["x"], route["y"]])
  heading0 = np.arctan2(*(xy[40] - xy[0])[::-1])
  fwd = np.array([np.cos(heading0), np.sin(heading0)])
  left = np.array([-np.sin(heading0), np.cos(heading0)])
  rows = -(-args.n // args.cols)
  slots = [xy[0] + fwd * (args.spacing * (i // args.cols)) + left * (args.spacing * (i % args.cols - (args.cols - 1) / 2))
           for i in range(args.n)]
  grid = fine(crop(grid, np.array(slots), margin=15.0))

  t0 = time.time()
  model = g1_model(grid, skis=False, n_robots=args.n)
  snow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "snow")
  model.mat_rgba[snow] = GROUND_RGBA
  model.mat_specular[snow] = 0.05
  data = mujoco.MjData(model)
  print(f"built {args.n} robots ({rows} rows) in {time.time() - t0:.1f}s")

  root = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r{i}_floating_base_joint") for i in range(args.n)]
  qadr, dadr = model.jnt_qposadr[root], model.jnt_dofadr[root]
  pelvis = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"r{i}_pelvis") for i in range(args.n)])
  feet = [[g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"r{i}_")
           and "foot" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)] for i in range(args.n)]
  # Actuators are cloned in robot order, 29 each, in the same joint order as DEFAULT_ANGLES.
  ctrl = np.tile(DEFAULT_ANGLES, args.n).astype(float)
  heading = np.zeros(args.n)

  def spawn(i):
    """Stand robot i at its slot, feet just touching the ground, with a fresh random lean."""
    a, d = qadr[i], dadr[i]
    heading[i] = heading0 + np.radians(rng.uniform(-30, 30))
    data.qpos[a : a + 3] = (*slots[i], 0.0)
    data.qpos[a + 3 : a + 7] = (np.cos(heading[i] / 2), 0, 0, np.sin(heading[i] / 2))
    data.qpos[a + 7 : a + NQ_ROBOT] = DEFAULT_ANGLES
    data.qvel[d : d + NV_ROBOT] = 0
    mujoco.mj_kinematics(model, data)
    gap = min(data.geom_xpos[g][2] - model.geom_size[g][0]
              - grid.height_normal(data.geom_xpos[g][None, 0], data.geom_xpos[g][None, 1])[0][0] for g in feet[i])
    data.qpos[a + 2] -= gap - 0.002
    c = ctrl[29 * i : 29 * (i + 1)]
    c[:] = DEFAULT_ANGLES
    pitch, roll = args.lean * rng.uniform(-1, 1, 2)
    c[[0, 6]] += pitch          # hip pitch
    c[[4, 10]] -= 0.5 * pitch   # ankle pitch
    c[[1, 7]] += roll           # hip roll
    c[[5, 11]] -= 0.5 * roll    # ankle roll

  for i in range(args.n):
    spawn(i)
  mujoco.mj_forward(model, data)

  W, H = 960, 540
  overview = mujoco.Renderer(model, H, W, max_geom=60000)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  ov_dir = out / "frames_overview"
  ov_dir.mkdir(exist_ok=True)
  for f in ov_dir.glob("*.png"):
    f.unlink()

  steps_per_frame = int(round(1 / (args.fps * model.opt.timestep)))
  n_frames = int(args.seconds * args.fps)
  ep_start = np.zeros(args.n)
  episodes, ep_lengths = 0, []
  lookat = np.array([*np.mean(slots, 0), 0.0])
  lookat[2] = grid.height_normal(lookat[None, 0], lookat[None, 1])[0][0]
  t_sim = t_render = 0.0
  for f in range(n_frames):
    ts = time.time()
    for _ in range(steps_per_frame):
      data.ctrl[:] = ctrl
      mujoco.mj_step(model, data)
    up_z = data.xmat[pelvis][:, 8]
    h, _ = grid.height_normal(data.xpos[pelvis][:, 0], data.xpos[pelvis][:, 1])
    down = (up_z < np.cos(np.radians(60))) | (data.xpos[pelvis][:, 2] - h < 0.35) | ~np.isfinite(data.xpos[pelvis]).all(1)
    for i in np.flatnonzero(down):
      episodes += 1
      ep_lengths.append(data.time - ep_start[i])
      spawn(i)
      ep_start[i] = data.time
    if down.any():
      mujoco.mj_forward(model, data)
    t_sim += time.time() - ts

    tr = time.time()
    cam.lookat[:] = lookat
    cam.distance = args.distance or 0.55 * args.spacing * max(rows, args.cols)
    cam.azimuth, cam.elevation = np.degrees(heading0) + args.azimuth, args.elevation
    overview.update_scene(data, cam)
    Image.fromarray(overview.render()).save(ov_dir / f"{f:04d}.png")
    t_render += time.time() - tr
    if f % args.fps == 0:
      print(f"t={data.time:5.2f}s resets {episodes:4d}  sim {t_sim:.0f}s render {t_render:.0f}s", flush=True)

  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(ov_dir / "%04d.png"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(out / "overview.mp4")], check=True)
  for k in (0, n_frames // 2, n_frames - 1):
    shutil.copy(ov_dir / f"{k:04d}.png", out / f"overview_{k:04d}.png")
  stats = {"course": args.course, "robots": args.n, "seconds": args.seconds, "resets": episodes,
           "mean_episode_s": float(np.mean(ep_lengths)) if ep_lengths else None,
           "wall_sim_s": round(t_sim, 1), "wall_render_s": round(t_render, 1)}
  (out / "stats.json").write_text(json.dumps(stats, indent=1))
  print(json.dumps(stats, indent=1))


if __name__ == "__main__":
  main()
