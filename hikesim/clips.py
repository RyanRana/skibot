"""Walking clips from SONIC's kinematic planner, precomputed for batched training (the planner is CPU only).

Each clip starts standing, then walks straight along +x at one speed and gait for the rest of the clip, replanned
every 0.5 s exactly like hikesim.walk. Stored at 50 Hz: joints (MuJoCo order) and root quaternions with the clip's
heading removed, so the env can point the reference wherever its command says.

    .venv/bin/python -m hikesim.clips          # -> out/hike/clips.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hikesim.planner import Planner

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [("slowWalk", 0.4), ("walk", 0.6), ("walk", 0.8), ("walk", 1.0), ("walk", 1.2), ("careful", -1.0)]


def strip_yaw(q: np.ndarray, smooth: int = 50) -> np.ndarray:
  """wxyz quaternions with their travel heading removed (yaw low-passed over 1 s), keeping the pelvis twist of each
  stride, pitch and roll."""
  w, x, y, z = q.T
  fx, fy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
  yaw = np.unwrap(np.arctan2(fy, fx))
  if smooth > 1:
    yaw = np.convolve(np.pad(yaw, smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="same")[smooth // 2: smooth // 2 + len(q)]
  c, s = np.cos(-yaw / 2), np.sin(-yaw / 2)  # left-multiply by yaw(-yaw)
  return np.stack([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], 1)


def make(seconds: float = 40.0, seed: int = 0) -> dict:
  pl = Planner(seed=seed)
  out = {"names": [], "q": [], "quat": [], "speed": []}
  for mode, speed in CLIPS:
    ref = pl.start((0.0, 0.0), 0.0)
    q, quat, xy = [], [], []
    tick = 0
    for t in range(int(seconds * 50)):
      if t >= 50 and (t - 50) % 25 == 0:
        ref = pl.replan(tick, mode, (1.0, 0.0), speed=speed)
        tick = 0
      f = ref.frame(tick)
      q.append(ref.q[f]); quat.append(ref.quat[f]); xy.append(pl.plan[f, :2].copy())
      tick += 1
    xy = np.array(xy)
    v = np.linalg.norm(xy[-1] - xy[250]) / ((len(xy) - 251) / 50)
    out["names"].append(f"{mode}_{speed:g}")
    out["q"].append(np.array(q, np.float32))
    out["quat"].append(strip_yaw(np.array(quat)).astype(np.float32))
    out["speed"].append(v)
    print(f"  {mode} {speed:g}: {v:.2f} m/s planned", flush=True)
  return {"names": np.array(out["names"]), "q": np.stack(out["q"]), "quat": np.stack(out["quat"]),
          "speed": np.array(out["speed"], np.float32)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seconds", type=float, default=40.0)
  ap.add_argument("--out", type=Path, default=ROOT / "out" / "hike" / "clips.npz")
  a = ap.parse_args()
  d = make(a.seconds)
  a.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(a.out, **d)
  print(f"wrote {a.out}: {len(d['names'])} clips x {d['q'].shape[1]} frames")


if __name__ == "__main__":
  main()
