"""Failure-mode diagnostic: run a checkpoint for many episodes and say how and where it fails.

Run: PYTHONPATH=. .venv-train/bin/python -m train.diagnose --ckpt runs_live/<run>/model_N.pt --terrain out/terrains_v3
Each fall is classified from the state at the moment of the fall: tree strike, landing (a ski left the snow in the
last 0.5 s), forward or backward fall, falling into or out of the turn (sideways, relative to the commanded turn),
with speed and slope. Success is broken down by slope, snow and terrain features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from train.ski_env import ROOT, SkiEnv
from train.train import load_policy, runner_cfg


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--terrain", default=str(ROOT / "out/terrains_v3"))
  ap.add_argument("--envs", type=int, default=32)
  ap.add_argument("--episodes", type=int, default=96)
  ap.add_argument("--max-level", type=int, default=60, help="sample tiles 0..max-level (where training is)")
  ap.add_argument("--out", default="")
  ap.add_argument("--spawn-speed", type=float, nargs=2, default=None)
  ap.add_argument("--device", default="cpu", help="cuda runs hundreds of envs at once (see modal_train.py diag)")
  args = ap.parse_args()
  env = SkiEnv(num_envs=args.envs, device=args.device, use_graph=False, seed=11, mosaic=args.terrain)
  env.auto_reset = False
  env.explore_frac = 0.0
  if args.spawn_speed:
    env.spawn_speed = tuple(args.spawn_speed)
  meta = json.loads((Path(args.terrain) / "mosaic.json").read_text())
  policy = load_policy(env, args.ckpt, device=args.device)
  rng = np.random.default_rng(0)
  hi = min(args.max_level, env.n_tiles - 1)

  def new_episode(ids):
    ids_t = torch.tensor(ids, dtype=torch.long, device=env.device)
    env.level[ids_t] = torch.tensor(rng.integers(0, hi + 1, len(ids)), dtype=torch.long, device=env.device)
    env.reset_idx(ids_t)
    env._forward()
    env._refresh_history(ids_t, fill=True)
    env._update_gate_command(rate_limit=False)
    for i in ids:
      ep_start[i] = step
      air[i] = 1000
      off_run[i] = 0

  N = args.envs
  ep_start = np.zeros(N, int)
  air = np.full(N, 1000, int)  # steps since the last real flight (capped)
  off_run = np.zeros(N, int)
  step = 0
  new_episode(list(range(N)))
  env.obs = env._observations()
  results = []
  cmd_prev = env.cmd_heading.clone()
  while len(results) < args.episodes:
    with torch.no_grad():
      act = policy(env.get_observations())
    env.step(act)
    step += 1
    d = env.ski_diag
    both_off = (d["N"] < 15).all(-1).cpu().numpy()
    off_run = np.where(both_off, off_run + 1, 0)
    air = np.where(off_run >= 3, 0, np.minimum(air + 1, 1000))  # steps since a real flight (both skis off 3+ steps)
    R = env.xmat[:, env.pelvis].cpu().numpy()
    up = R[:, :, 2]
    v = env.qvel[:, 0:3].cpu().numpy()
    speed = np.linalg.norm(v[:, :2], axis=1)
    vhat = np.concatenate([v[:, :2] / np.maximum(speed[:, None], 1e-3), np.zeros((N, 1))], 1)
    left = np.cross(np.array([0, 0, 1.0]), vhat)
    turn_dir = np.sign((env.cmd_heading - torch.atan2(env.qvel[:, 1], env.qvel[:, 0])).cpu().numpy())
    fell = env.last_fell.cpu().numpy()
    tree = env.last_tree_hit.cpu().numpy()
    x, y = env.qpos[:, 0].cpu().numpy(), env.qpos[:, 1].cpu().numpy()
    off = (np.abs(y - env.tile_y[env.tile].cpu().numpy()) > 22) | (x > env.tile_len - 8)
    timeout = (step - ep_start) * 0.02 >= 15.0
    done = fell | off | timeout
    ids = np.flatnonzero(done)
    for i in ids:
      t = env.tile[i].item()
      tile = meta["tiles"][t]
      rec = {"tile": t, "slope": tile.get("mean_slope_deg"), "snow": tile.get("snow", "groomed"),
             "jumps": bool(tile.get("jumps")), "chute": bool(tile.get("chute_extra_deg")),
             "seconds": round((step - ep_start[i]) * 0.02, 2), "distance": round(float(x[i] - env.start_x[i].item()), 1),
             "speed": round(float(speed[i]), 1), "gates": bool(env.use_gates[i]), "fell": bool(fell[i]),
             "dy": round(float(y[i] - env.tile_y[t].item()), 1), "off_tile": bool(off[i] and not fell[i])}
      if fell[i]:
        fwd_tilt = float(up[i] @ vhat[i])
        side_tilt = float(up[i] @ left[i])  # + = leaning left
        if tree[i]:
          kind = "tree strike"
        elif air[i] < 25:
          kind = "landing / airborne"
        elif abs(fwd_tilt) > abs(side_tilt):
          kind = "fell forward" if fwd_tilt > 0 else "fell backward"
        else:
          into_turn = np.sign(side_tilt) == turn_dir[i] if turn_dir[i] != 0 else False
          kind = "fell into the turn" if into_turn else "fell out of the turn"
        rec["kind"] = kind
      results.append(rec)
    if len(ids):
      new_episode(list(ids))
    env.obs = env._observations()
  falls = [r for r in results if r["fell"]]
  out = {"checkpoint": args.ckpt, "episodes": len(results), "fall_rate": round(len(falls) / len(results), 3),
         "mean_distance_m": round(float(np.mean([r["distance"] for r in results])), 1),
         "failure_kinds": dict(Counter(r["kind"] for r in falls).most_common()),
         "median_fall_time_s": float(np.median([r["seconds"] for r in falls])) if falls else None,
         "falls_in_first_3s": sum(1 for r in falls if r["seconds"] < 3.0)}
  def success_by(key, bins=None):
    g = defaultdict(list)
    for r in results:
      k = r[key] if bins is None else next((f"{a}-{b} deg" for a, b in bins if a <= r[key] < b), "other")
      g[k].append(not r["fell"])
    return {k: f"{100 * np.mean(v):.0f}% clean of {len(v)}" for k, v in sorted(g.items())}
  out["by_slope"] = success_by("slope", [(0, 10), (10, 15), (15, 20), (20, 25), (25, 45)])
  out["by_snow"] = success_by("snow")
  out["with_jumps"] = success_by("jumps")
  out["with_chute"] = success_by("chute")
  out["gate_vs_free"] = success_by("gates")
  print(json.dumps(out, indent=1), flush=True)
  if args.out:
    Path(args.out).write_text(json.dumps({"summary": out, "episodes": results}, indent=1))


if __name__ == "__main__":
  main()
