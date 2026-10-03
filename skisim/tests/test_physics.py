"""Checks the ski force laws on a rigid sled, where the answers are known in closed form.

Run: .venv/bin/python -m skisim.tests.test_physics
"""

from __future__ import annotations

import mujoco
import numpy as np

from skisim.scene import place, sled_model
from skisim.ski import SkiContact, SkiParams
from skisim.terrain import slope

G = 9.81


def run(model, grid, xy, heading, seconds, v0=None, params=None):
  params = params or SkiParams()
  data = mujoco.MjData(model)
  place(model, data, grid, xy, heading, params=params)
  if v0 is not None:
    data.qvel[0:3] = v0
  ski = SkiContact(model, grid, params, drag_body="sled")
  t, pos, vel = [], [], []
  steps = int(seconds / model.opt.timestep)
  for _ in range(steps):
    ski.apply(data)
    mujoco.mj_step(model, data)
    t.append(data.time)
    pos.append(data.qpos[0:3].copy())
    vel.append(data.qvel[0:3].copy())
  return np.array(t), np.array(pos), np.array(vel), data, ski


def mass(model):
  return float(model.body_subtreemass[1])


def glide():
  theta = 15.0
  p = SkiParams()
  grid = slope(theta, length=400, width=60, cell=1.0)
  model = sled_model(grid, p)
  t, pos, vel, _, _ = run(model, grid, (10.0, 0.0), 0.0, 8.0, params=p)
  m = mass(model)
  # Expected: dv/dt = g (sin - mu cos) - rho CdA v^2 / (2 m), integrated with the same step.
  th = np.radians(theta)
  v, dt = 0.0, model.opt.timestep
  for _ in range(len(t)):
    v += dt * (G * (np.sin(th) - p.mu_glide * np.cos(th)) - p.rho_air * p.cda * v * v / (2 * m))
  speed = np.linalg.norm(vel[-1])
  lateral = abs(pos[-1, 1] - pos[0, 1])
  print(f"glide 15deg 8s: speed {speed:.2f} m/s vs expected {v:.2f} (ratio {speed / v:.3f}), drift {lateral:.3f} m")
  return speed / v


def carve():
  edge = 30.0
  p = SkiParams()
  grid = slope(0.0, length=200, width=200, cell=1.0)
  grid.x0, grid.y0 = -100.0, -100.0
  model = sled_model(grid, p, ski_roll=-np.radians(edge))  # negative roll about x puts the left edge down
  t, pos, vel, _, ski = run(model, grid, (-60.0, -40.0), 0.0, 3.0, v0=(10.0, 0.0, 0.0), params=p)
  xy = pos[len(pos) // 3 :, :2]
  # Algebraic circle fit.
  A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
  b = (xy**2).sum(1)
  cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
  R = np.sqrt(c + cx**2 + cy**2)
  turned_left = (np.cross(np.append(vel[0, :2], 0), np.append(vel[-1, :2], 0))[2]) > 0
  expected = p.sidecut_radius * np.cos(np.radians(edge))
  st = ski.state[0]
  print(
    f"carve 30deg edge: radius {R:.2f} m vs sidecut*cos(edge) {expected:.2f} (ratio {R / expected:.3f}), "
    f"turned {'left' if turned_left else 'right'}, measured edge {st.edge_deg:.1f} deg, side slip {st.v_side:.3f} m/s"
  )
  return R / expected, turned_left


def skid():
  p = SkiParams()
  grid = slope(0.0, length=200, width=200, cell=1.0)
  grid.x0, grid.y0 = -100.0, -100.0
  model = sled_model(grid, p)
  t, pos, vel, _, _ = run(model, grid, (0.0, -60.0), 0.0, 3.0, v0=(0.0, 5.0, 0.0), params=p)
  speed = np.linalg.norm(vel[:, :2], axis=1)
  stop = t[np.argmax(speed < 0.05)] if (speed < 0.05).any() else np.inf
  expected = 5.0 / (p.mu_skid * G)
  print(f"sideways skid from 5 m/s: stopped in {stop:.2f} s vs 5/(mu_skid g) {expected:.2f} s (ratio {stop / expected:.3f})")
  return stop / expected


def sideslip_and_hold():
  theta = 20.0
  p = SkiParams()
  grid = slope(theta, length=200, width=200, cell=1.0)
  grid.y0 = -100.0
  out = {}
  for name, roll in (("flat", 0.0), ("uphill edge 10deg", np.radians(10.0))):
    # Heading +y across the hill; uphill is -x, which is the sled's left, so left edge down holds.
    model = sled_model(grid, p, ski_roll=-roll)
    t, pos, vel, _, _ = run(model, grid, (100.0, -60.0), np.pi / 2, 2.0, params=p)
    downhill = pos[-1, 0] - pos[0, 0]
    out[name] = downhill
    print(f"traverse 20deg, {name}: slid {downhill:.2f} m down the fall line in 2 s")
  expected_flat = 0.5 * G * (np.sin(np.radians(theta)) - p.mu_skid * np.cos(np.radians(theta))) * 4
  print(f"  flat sideslip expected about {expected_flat:.2f} m (gravity minus skid friction)")
  return out


if __name__ == "__main__":
  glide()
  carve()
  skid()
  sideslip_and_hold()
