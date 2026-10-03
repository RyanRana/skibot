"""Many G1s on one real slope, the way parallel RL sees the world. Untrained: PD stance plus random commands.

A robot that falls respawns at its start slot with a new random command, like an RL environment reset.
Run: .venv/bin/python -m skisim.fleet --course out/courses/bormio-stelvio --n 100 --seconds 8
Writes out/fleet/: overview.mp4, overview.gif, sheet.gif, stills, stats.json.
"""

from __future__ import annotations

import os
import shutil
import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from skisim.scene import SKI_STANCE, g1_model, place, stance_qpos
from skisim.ski import SkiParams
from skisim.ski_batch import BatchSkiContact
from skisim.terrain import from_course

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
NQ_ROBOT, NV_ROBOT = 36, 35  # free joint + 29 hinges


def actuator_index(model, prefix):
  """Unprefixed joint name -> actuator id for one robot."""
  a = {}
  for i in range(model.nu):
    jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
    if jn.startswith(prefix):
      a[jn[len(prefix) :]] = i
  return a


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--course", default=str(ROOT / "out/courses/bormio-stelvio"))
  ap.add_argument("--n", type=int, default=100)
  ap.add_argument("--cols", type=int, default=10)
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--fps", type=int, default=12)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--perturb", type=float, default=0.6, help="scale of the random edge and heading commands")
  ap.add_argument("--stiff-frac", type=float, default=0.75)
  ap.add_argument("--out", default=str(ROOT / "out/fleet"))
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  course = Path(args.course)
  grid = from_course(course)
  start = json.loads((course / "start_pose.json").read_text())
  sx, sy = start["snow_xyz"][:2]
  yaw = start["yaw"]
  fwd = np.array([np.cos(yaw), np.sin(yaw)])
  left = np.array([-np.sin(yaw), np.cos(yaw)])

  p = SkiParams()
  t0 = time.time()
  model = g1_model(grid, p, n_robots=args.n)
  data = mujoco.MjData(model)
  solo = g1_model(grid, p)  # one robot, used to draw the chase-cam tiles cheaply
  solo_data = mujoco.MjData(solo)
  print(f"built {args.n} robots: nq={model.nq} nu={model.nu} in {time.time() - t0:.1f}s")

  q0 = stance_qpos(model, SKI_STANCE)
  data.qpos[:] = q0
  root = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r{i}_floating_base_joint") for i in range(args.n)]
  qadr = model.jnt_qposadr[root]
  dadr = model.jnt_dofadr[root]
  acts = [actuator_index(model, f"r{i}_") for i in range(args.n)]

  # Fixed per slot: gains. Soft slots keep mjlab's gains and fold, like a policy that hasn't learned to brace.
  stiff = rng.random(args.n) < args.stiff_frac
  for i in np.flatnonzero(stiff):
    ids = list(acts[i].values())
    model.actuator_gainprm[ids, 0] *= 6
    model.actuator_biasprm[ids, 1] *= 6
    model.actuator_biasprm[ids, 2] *= 3

  slots = []
  for i in range(args.n):
    r, c = divmod(i, args.cols)
    slots.append(np.array([sx, sy]) + fwd * (8 + 7.0 * r) + left * (5.0 * (c - (args.cols - 1) / 2)))

  ctrl = np.zeros(model.nu)
  heading = np.zeros(args.n)

  def spawn(i):
    """Reset robot i at its slot with a fresh random command."""
    pre = f"r{i}_"
    data.qpos[qadr[i] + 7 : qadr[i] + NQ_ROBOT] = q0[qadr[i] + 7 : qadr[i] + NQ_ROBOT]
    data.qvel[dadr[i] : dadr[i] + NV_ROBOT] = 0
    xy = slots[i]
    _, nrm = grid.height_normal(np.array([xy[0]]), np.array([xy[1]]))
    fall_line = np.arctan2(nrm[0, 1], nrm[0, 0])  # the normal's horizontal part points downhill
    heading[i] = fall_line + args.perturb * np.radians(rng.uniform(-25, 25))
    place(model, data, grid, tuple(xy), heading[i], ski_bodies=(pre + "left_ski", pre + "right_ski"),
          params=p, root_joint=pre + "floating_base_joint")
    act = acts[i]
    for jn, ai in act.items():
      ctrl[ai] = q0[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + jn)]]
    hip_roll = args.perturb * rng.uniform(-0.18, 0.18)
    ankle_roll = np.clip(hip_roll * rng.uniform(0.3, 1.0), -0.25, 0.25)
    ctrl[act["left_hip_roll_joint"]] += hip_roll
    ctrl[act["right_hip_roll_joint"]] += hip_roll
    ctrl[act["left_ankle_roll_joint"]] -= ankle_roll
    ctrl[act["right_ankle_roll_joint"]] -= ankle_roll
    if rng.random() < 0.15 * args.perturb:  # snowplow
      ctrl[act["left_hip_yaw_joint"]] += 0.25
      ctrl[act["right_hip_yaw_joint"]] -= 0.25

  for i in range(args.n):
    spawn(i)
  mujoco.mj_forward(model, data)

  skis = [f"r{i}_{s}_ski" for i in range(args.n) for s in ("left", "right")]
  pelvis_names = [f"r{i}_pelvis" for i in range(args.n)]
  contact = BatchSkiContact(model, grid, skis, drag_bodies=pelvis_names, params=p)
  pelvis = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nme) for nme in pelvis_names])

  W, H = 960, 540
  tile_w, tile_h, tiles_x, tiles_y = 240, 180, 4, 3
  overview = mujoco.Renderer(model, H, W, max_geom=30000)
  tile = mujoco.Renderer(solo, tile_h, tile_w)
  sheet_ids = rng.choice(args.n, min(args.n, tiles_x * tiles_y), replace=False)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  ov_dir, sh_dir = out / "frames_overview", out / "frames_sheet"
  for d in (ov_dir, sh_dir):
    d.mkdir(exist_ok=True)
    for f in d.glob("*.png"):
      f.unlink()

  steps_per_frame = int(round(1 / (args.fps * model.opt.timestep)))
  n_frames = int(args.seconds * args.fps)
  ep_start = np.zeros(args.n)
  episodes, ep_lengths, ep_dist = 0, [], []
  start_xy = np.array(slots)
  lookat = data.xpos[pelvis].mean(0)
  t_sim = t_render = 0.0
  for f in range(n_frames):
    ts = time.time()
    for _ in range(steps_per_frame):
      data.ctrl[:] = ctrl
      contact.apply(data)
      mujoco.mj_step(model, data)
    up_z = data.xmat[pelvis][:, 8]
    h, _ = grid.height_normal(data.xpos[pelvis][:, 0], data.xpos[pelvis][:, 1])
    down = (up_z < np.cos(np.radians(60))) | (data.xpos[pelvis][:, 2] - h < 0.35) | ~np.isfinite(data.xpos[pelvis]).all(1)
    for i in np.flatnonzero(down):
      episodes += 1
      ep_lengths.append(data.time - ep_start[i])
      ep_dist.append(float(np.linalg.norm(data.xpos[pelvis[i], :2] - start_xy[i])))
      spawn(i)
      ep_start[i] = data.time
    if down.any():
      mujoco.mj_forward(model, data)
    t_sim += time.time() - ts

    tr = time.time()
    alive = data.time - ep_start > 1.0
    target = data.xpos[pelvis][alive].mean(0) if alive.any() else data.xpos[pelvis].mean(0)
    lookat = 0.9 * lookat + 0.1 * target if f else target
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 70.0, np.degrees(np.median(heading)) - 30, -30.0
    overview.update_scene(data, cam)
    Image.fromarray(overview.render()).save(ov_dir / f"{f:04d}.png")

    canvas = Image.new("RGB", (tile_w * tiles_x, tile_h * tiles_y))
    for k, i in enumerate(sheet_ids):
      solo_data.qpos[:] = data.qpos[qadr[i] : qadr[i] + NQ_ROBOT]
      mujoco.mj_forward(solo, solo_data)
      c2 = mujoco.MjvCamera()
      c2.type = mujoco.mjtCamera.mjCAMERA_FREE
      c2.lookat[:] = data.xpos[pelvis[i]]
      vel = data.cvel[pelvis[i], 3:]
      head = np.degrees(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 0.5 else np.degrees(heading[i])
      c2.distance, c2.azimuth, c2.elevation = 4.5, head - 25, -15.0
      tile.update_scene(solo_data, c2)
      canvas.paste(Image.fromarray(tile.render()), ((k % tiles_x) * tile_w, (k // tiles_x) * tile_h))
    canvas.save(sh_dir / f"{f:04d}.png")
    t_render += time.time() - tr
    if f % args.fps == 0:
      speed = np.linalg.norm(data.cvel[pelvis][:, 3:], axis=1)
      print(f"t={data.time:5.2f}s episodes ended {episodes:4d}  mean speed {speed.mean():.1f} m/s  max {speed.max():.1f}  "
            f"sim {t_sim:.0f}s render {t_render:.0f}s", flush=True)

  enc = lambda d, name, scale: subprocess.run(
    [FFMPEG, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(d / "%04d.png"),
     "-vf", f"scale={scale}:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
     str(out / f"{name}.gif")], check=True)
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(ov_dir / "%04d.png"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out / "overview.mp4")], check=True)
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(sh_dir / "%04d.png"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out / "sheet.mp4")], check=True)
  enc(ov_dir, "overview", "640:-2")
  enc(sh_dir, "sheet", "720:-2")
  for f in (0, n_frames // 2, n_frames - 1):
    Image.open(ov_dir / f"{f:04d}.png").save(out / f"overview_{f:04d}.png")
  Image.open(sh_dir / f"{n_frames // 2:04d}.png").save(out / "sheet_mid.png")

  speed = np.linalg.norm(data.cvel[pelvis][:, 3:], axis=1)
  open_len = data.time - ep_start
  stats = {
    "robots": args.n,
    "course": course.name,
    "sim_seconds": round(data.time, 2),
    "episodes_ended_by_fall": episodes,
    "median_fall_episode_s": round(float(np.median(ep_lengths)), 2) if ep_lengths else None,
    "robots_never_fell": int((ep_start == 0).sum()),
    "never_fell_stiff": int(((ep_start == 0) & stiff).sum()),
    "stiff_robots": int(stiff.sum()),
    "longest_open_episode_s": round(float(open_len.max()), 2),
    "end_speed_mean_mps": round(float(speed.mean()), 2),
    "end_speed_max_mps": round(float(speed.max()), 2),
    "wall_sim_s": round(t_sim, 1),
    "wall_render_s": round(t_render, 1),
  }
  (out / "stats.json").write_text(json.dumps(stats, indent=1))
  print(json.dumps(stats, indent=1))


if __name__ == "__main__":
  main()
