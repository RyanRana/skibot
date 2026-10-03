"""Evaluate a trained checkpoint on every terrain tile and render it.

Runs the same SkiEnv (CPU, MuJoCo Warp) with one robot per tile, records poses, scores each tile, then draws
the robots into a CPU MuJoCo scene on the same mosaic for an overview and chase-cam videos.
Run: PYTHONPATH=. .venv-train/bin/python -m train.play --ckpt runs/<run>/model_1499.pt
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
import torch
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

from skisim.scene import g1_model
from skisim.ski import SkiParams
from skisim.terrain import HeightGrid
from train.ski_env import ROOT, SkiEnv
from train.train import runner_cfg

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")


def rollout(ckpt: str | None, seconds: float, per_tile: int = 1, zero_policy: bool = False):
  env = SkiEnv(num_envs=24 * per_tile, device="cpu", use_graph=False, seed=1)
  n = env.num_envs
  env.level[:] = torch.arange(n) % env.n_tiles
  env.reset_idx(torch.arange(n))
  env._forward()
  env._refresh_history(torch.arange(n), fill=True)
  env.obs = env._observations()
  if zero_policy:
    policy = lambda obs: torch.zeros(n, env.num_actions)
  else:
    runner = OnPolicyRunner(env, runner_cfg(1, "tensorboard"), None, device="cpu")
    runner.load(ckpt, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
  obs = env.get_observations()
  frames, first_fall, dist = [], np.full(n, np.nan), np.zeros(n)
  start_x = env.qpos[:, 0].clone()
  alive = torch.ones(n, dtype=torch.bool)
  herr_sum, herr_cnt = np.zeros(n), np.zeros(n)
  gap_sum, air_sum, width_sum = np.zeros(n), np.zeros(n), np.zeros(n)
  steps = int(seconds / 0.02)
  for k in range(steps):
    with torch.no_grad():
      act = policy(obs)
    obs, rew, done, extras = env.step(act)
    v = env.qvel[:, 0:2]
    travel = torch.atan2(v[:, 1], v[:, 0])
    herr = torch.atan2(torch.sin(travel - env.cmd_heading), torch.cos(travel - env.cmd_heading)).abs()
    herr_sum += (herr * alive).numpy()
    herr_cnt += alive.numpy()
    sx = env.xmat[:, env.ski_ids, :, 0]
    sy = torch.atan2(sx[..., 1], sx[..., 0])
    gap = torch.atan2(torch.sin(sy[:, 0] - sy[:, 1]), torch.cos(sy[:, 0] - sy[:, 1])).abs()
    Rp = env.xmat[:, env.pelvis]
    rel = (Rp.transpose(1, 2)[:, None] @ (env.xpos[:, env.ski_ids] - env.qpos[:, None, 0:3])[..., None])[..., 0]
    width = rel[:, 0, 1] - rel[:, 1, 1]
    air = (env.ski_diag["N"] < 15).any(-1).float()
    a = alive.numpy()
    gap_sum += gap.numpy() * a
    air_sum += air.numpy() * a
    width_sum += width.numpy() * a
    fell = done & ~extras["time_outs"]
    newly = (fell & alive).numpy()
    first_fall[newly] = (k + 1) * 0.02
    still = alive & ~done  # done envs were already respawned, so keep their last distance
    dist[still.numpy()] = (env.qpos[:, 0] - start_x)[still].numpy().clip(min=0)
    alive = still
    if k % 2 == 0:
      frames.append(env.qpos.numpy().copy())
  meta = json.loads((ROOT / "out/terrains/mosaic.json").read_text())
  tiles = meta["tiles"]
  per = []
  for i in range(n):
    t = tiles[i % env.n_tiles]
    per.append({
      "tile": i % env.n_tiles, "source": t["source"], "mean_slope_deg": t["mean_slope_deg"], "roughness_m": t["roughness_m"],
      "result": "fell" if np.isfinite(first_fall[i]) else "clean",
      "time_s": float(first_fall[i]) if np.isfinite(first_fall[i]) else seconds,
      "distance_m": round(float(dist[i]), 1),
      "heading_err_deg": round(float(np.degrees(herr_sum[i] / max(herr_cnt[i], 1))), 1),
      "ski_gap_deg": round(float(np.degrees(gap_sum[i] / max(herr_cnt[i], 1))), 1),
      "ski_airborne_frac": round(float(air_sum[i] / max(herr_cnt[i], 1)), 2),
      "stance_width_m": round(float(width_sum[i] / max(herr_cnt[i], 1)), 2),
    })
  return env, frames, per


def render(env, frames, out: Path, fps=25):
  z = np.load(ROOT / "out/terrains/mosaic_elevation.npy").astype(float)
  meta = json.loads((ROOT / "out/terrains/mosaic.json").read_text())
  grid = HeightGrid(z=z, x0=meta["x0"], y0=meta["y0"], cell=meta["cell"])
  n = env.num_envs
  model = g1_model(grid, SkiParams(), n_robots=n)
  data = mujoco.MjData(model)
  roots = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"r{i}_floating_base_joint") for i in range(n)]
  qadr = model.jnt_qposadr[roots]
  pel = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"r{i}_pelvis") for i in range(n)]
  solo = g1_model(grid, SkiParams())
  sd = mujoco.MjData(solo)
  ov = mujoco.Renderer(model, 540, 960, max_geom=30000)
  tile = mujoco.Renderer(solo, 180, 240)
  show = list(range(0, n, max(1, n // 12)))[:12]
  dirs = {k: out / f"frames_{k}" for k in ("overview", "sheet")}
  for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
      f.unlink()
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  for fi, q in enumerate(frames):
    for i in range(n):
      data.qpos[qadr[i] : qadr[i] + 36] = q[i]
    mujoco.mj_forward(model, data)
    c = data.xpos[pel].mean(0)
    cam.lookat[:] = c
    cam.distance, cam.azimuth, cam.elevation = 230.0, 200.0, -35.0
    ov.update_scene(data, cam)
    Image.fromarray(ov.render()).save(dirs["overview"] / f"{fi:04d}.png")
    canvas = Image.new("RGB", (240 * 4, 180 * 3))
    for k, i in enumerate(show):
      sd.qpos[:] = q[i]
      mujoco.mj_forward(solo, sd)
      c2 = mujoco.MjvCamera()
      c2.type = mujoco.mjtCamera.mjCAMERA_FREE
      c2.lookat[:] = sd.xpos[1]
      c2.distance, c2.azimuth, c2.elevation = 4.5, 155.0, -15.0
      tile.update_scene(sd, c2)
      canvas.paste(Image.fromarray(tile.render()), ((k % 4) * 240, (k // 4) * 180))
    canvas.save(dirs["sheet"] / f"{fi:04d}.png")
  for k, d in dirs.items():
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(d / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", str(out / f"{k}.mp4")], check=True)
  Image.open(dirs["sheet"] / f"{len(frames) // 2:04d}.png").save(out / "sheet_mid.jpg", quality=85)
  Image.open(dirs["overview"] / f"{len(frames) // 2:04d}.png").save(out / "overview_mid.jpg", quality=85)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default="")
  ap.add_argument("--seconds", type=float, default=12.0)
  ap.add_argument("--zero-policy", action="store_true", help="SONIC alone, no residual (baseline)")
  ap.add_argument("--out", default="")
  ap.add_argument("--no-render", action="store_true")
  args = ap.parse_args()
  name = "sonic_only" if args.zero_policy else Path(args.ckpt).parent.name + "_" + Path(args.ckpt).stem
  out = Path(args.out or ROOT / "out/play" / name)
  out.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  env, frames, per = rollout(args.ckpt or None, args.seconds, zero_policy=args.zero_policy)
  clean = sum(p["result"] == "clean" for p in per)
  summary = {
    "checkpoint": args.ckpt or "none (SONIC + zero residual)",
    "tiles": len(per), "clean_runs": clean,
    "mean_distance_m": round(float(np.mean([p["distance_m"] for p in per])), 1),
    "mean_heading_err_deg": round(float(np.mean([p["heading_err_deg"] for p in per])), 1),
    "mean_ski_gap_deg": round(float(np.mean([p["ski_gap_deg"] for p in per])), 1),
    "mean_ski_airborne_frac": round(float(np.mean([p["ski_airborne_frac"] for p in per])), 2),
    "mean_stance_width_m": round(float(np.mean([p["stance_width_m"] for p in per])), 2),
    "per_tile": per,
  }
  (out / "eval.json").write_text(json.dumps(summary, indent=1))
  print(f"{name}: {clean}/{len(per)} tiles clean for {args.seconds}s, mean distance {summary['mean_distance_m']} m, "
        f"heading err {summary['mean_heading_err_deg']} deg, ski gap {summary['mean_ski_gap_deg']} deg, "
        f"a ski airborne {summary['mean_ski_airborne_frac']:.0%} of the time, stance {summary['mean_stance_width_m']} m  "
        f"({time.time() - t0:.0f}s)", flush=True)
  for p in per:
    print(f"  tile {p['tile']:2d} {p['mean_slope_deg']:5.1f}° rough {p['roughness_m']:.2f}  {p['result']:5s} {p['time_s']:5.1f}s  {p['distance_m']:6.1f} m  head err {p['heading_err_deg']:5.1f}°")
  if not args.no_render:
    render(env, frames, out)
    print("rendered", out, flush=True)


if __name__ == "__main__":
  main()
