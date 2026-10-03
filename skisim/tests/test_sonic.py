"""SONIC zero-shot: first standing on plain ground (checks our observation wiring), then on skis.

Run: .venv/bin/python -m skisim.tests.test_sonic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np

from skisim.scene import g1_model, place
from skisim.ski import SkiContact, SkiParams
from skisim.sonic import DEFAULT_ANGLES, JOINTS, Reference, SonicController
from skisim.terrain import slope

ROOT = Path(__file__).resolve().parents[2]


def walk_reference() -> Reference:
  """SONIC's sample walking clip (30 fps, MuJoCo order) resampled to 50 Hz."""
  d = joblib.load(ROOT / "assets/sonic/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")
  m = next(iter(d.values()))
  q, rot, fps = m["dof"], m["root_rot"], m["fps"]
  t_src = np.arange(len(q)) / fps
  t = np.arange(0, t_src[-1], 1 / 50)
  qi = np.stack([np.interp(t, t_src, q[:, j]) for j in range(29)], 1)
  idx = np.clip(np.round(t * fps).astype(int), 0, len(rot) - 1)
  quat = rot[idx][:, [3, 0, 1, 2]]  # xyzw -> wxyz
  return Reference(qi, quat, loop=False)


def run(theta=0.0, skis=False, ref=None, seconds=10.0, label="", record=None):
  p = SkiParams()
  grid = slope(theta, length=600, width=80, cell=1.0)
  model = g1_model(grid, p, skis=skis)
  data = mujoco.MjData(model)
  sonic = SonicController(model)
  q0 = np.zeros(model.nq)
  q0[sonic.qadr] = DEFAULT_ANGLES
  if skis:
    place(model, data, grid, (20.0, 0.0), 0.0, joint_qpos=q0, params=p)
  else:
    data.qpos[:] = q0
    data.qpos[0:3] = (20.0, 0.0, 0.8)
    data.qpos[3:7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    # Drop onto the ground: lower until the lowest foot geom touches.
    h, _ = grid.height_normal(np.array([20.0]), np.array([0.0]))
    feet = [g for g in range(model.ngeom) if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    low = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in feet)
    data.qpos[2] -= low - h[0] - 0.002
    mujoco.mj_forward(model, data)
  ref = ref or Reference.static(DEFAULT_ANGLES)
  sonic.reset(data, ref)
  ski = SkiContact(model, grid, p) if skis else None
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  decim = int(round(0.02 / model.opt.timestep))
  fell, dq_rms, frames = None, [], []
  for k in range(int(seconds / model.opt.timestep)):
    if k % decim == 0:
      sonic.step(data)
      dq_rms.append(np.sqrt(np.mean(data.qvel[sonic.dadr] ** 2)))
      if record is not None and k % (decim * 2) == 0:
        frames.append(data.qpos.copy())
    if ski:
      ski.apply(data)
    mujoco.mj_step(model, data)
    up = data.xmat[pelvis].reshape(3, 3)[:, 2]
    h, _ = grid.height_normal(data.xpos[pelvis][None, 0], data.xpos[pelvis][None, 1])
    if not np.isfinite(data.qpos).all() or up[2] < 0.5 or data.xpos[pelvis][2] - h[0] < 0.35:
      fell = round(data.time, 2)
      break
  disp = np.linalg.norm(data.qpos[0:2] - (20.0, 0.0))
  speed = np.linalg.norm(data.qvel[0:3])
  res = {
    "case": label, "slope_deg": theta, "skis": skis,
    "result": f"fell at {fell}s" if fell else f"upright {seconds}s",
    "moved_m": round(float(disp), 2), "end_speed_mps": round(float(speed), 2),
    "joint_vel_rms": round(float(np.median(dq_rms)), 3),
  }
  if ski:
    res["ski_N"] = [round(s.normal_force) for s in ski.state]
  print(json.dumps(res), flush=True)
  if record is not None:
    record.update(model=model, frames=frames, grid=grid)
  return res


if __name__ == "__main__":
  which = sys.argv[1] if len(sys.argv) > 1 else "all"
  if which in ("all", "ground"):
    run(0.0, skis=False, label="stand, plain ground")
    run(0.0, skis=False, ref=walk_reference(), seconds=25.0, label="walk clip 25 s, plain ground (clip walks 5.6 m by 20 s)")
  if which in ("all", "skis"):
    for th in (0.0, 8.0, 15.0):
      run(th, skis=True, seconds=15.0, label="stand ref, skis")
