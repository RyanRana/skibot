"""G1 on skis holding the ski stance with its PD gains only, no balance controller.

Run: .venv/bin/python -m skisim.tests.test_g1
"""

from __future__ import annotations

import mujoco
import numpy as np

from skisim.scene import g1_model, place, stance_qpos
from skisim.ski import SkiContact, SkiParams
from skisim.terrain import slope


def hold_stance(theta: float, seconds: float = 8.0, timestep: float = 0.002):
  p = SkiParams()
  grid = slope(theta, length=500, width=80, cell=1.0)
  model = g1_model(grid, p, timestep=timestep)
  data = mujoco.MjData(model)
  q0 = stance_qpos(model)
  place(model, data, grid, (15.0, 0.0), 0.0, joint_qpos=q0, params=p)
  ctrl = np.array([q0[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(model.nu)])
  ski = SkiContact(model, grid, p)
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  fell_at = None
  for _ in range(int(seconds / timestep)):
    data.ctrl[:] = ctrl
    ski.apply(data)
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
      fell_at = "NaN"
      break
    up = data.xmat[pelvis].reshape(3, 3)[:, 2]
    h, _ = grid.height_normal(data.xpos[pelvis][None, 0], data.xpos[pelvis][None, 1])
    if up[2] < np.cos(np.radians(60)) or data.xpos[pelvis][2] - h[0] < 0.35:
      fell_at = f"{data.time:.2f}s"
      break
  speed = np.linalg.norm(data.qvel[0:3])
  print(
    f"slope {theta:4.1f} deg, dt {timestep}: {'FELL at ' + fell_at if fell_at else 'upright'} after {data.time:.2f}s, "
    f"speed {speed:.2f} m/s, ski N = {ski.state[0].normal_force:.0f}/{ski.state[1].normal_force:.0f} N, "
    f"edges {ski.state[0].edge_deg:.1f}/{ski.state[1].edge_deg:.1f} deg, total mass {model.body_subtreemass[1]:.1f} kg"
  )
  return fell_at


if __name__ == "__main__":
  for theta in (0.0, 8.0, 15.0, 25.0):
    hold_stance(theta)
