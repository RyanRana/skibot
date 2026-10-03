"""Ski-snow contact as applied forces, so the same model can run in CPU MuJoCo and in batched GPU training.

Stock MuJoCo friction is the same in every sliding direction, which can't describe a ski: it glides
along its length, grips sideways once edged, and an edged ski bends into an arc that steers it
(carving radius = sidecut radius * cos(edge angle)). So the ski geoms don't touch the terrain through
MuJoCo at all. Each ski is a set of spring-damper points against the HeightGrid for the normal force,
and the tangential forces are computed here and written to data.xfrc_applied every physics step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from skisim.terrain import HeightGrid


@dataclass
class SkiParams:
  length: float = 1.2  # m, robot-scale GS ski (a 1.85 m racer on ~1.9 m skis, scaled to the 1.32 m G1)
  width: float = 0.07
  thickness: float = 0.015
  sidecut_radius: float = 17.0  # m, FIS GS skis are 30 m at full size
  mu_glide: float = 0.04  # along the ski, waxed ski on groomed snow
  mu_skid: float = 0.25  # sideways when the ski is flat and skidding
  mu_carve: float = 1.5  # sideways grip limit once the edge is fully engaged
  edge_on_deg: float = 3.0  # edge starts to bite
  edge_full_deg: float = 15.0  # edge fully engaged
  k_normal: float = 40000.0  # N/m per ski, ~5 mm of snow compression under half the G1's weight
  c_normal: float = 400.0  # N s/m per ski
  k_lateral: float = 500.0  # N s/m per ski, sideways slip damping below the grip limit
  c_yaw: float = 20.0  # N m s/rad per ski, pulls the ski's yaw rate toward the carved arc
  n_long: int = 9  # contact points along each edge
  glide_v_eps: float = 0.1  # m/s, smooths the glide friction sign at standstill
  rho_air: float = 1.0  # kg/m^3, roughly 2000 m altitude
  cda: float = 0.25  # m^2, drag area of an upright G1


def smoothstep(x, lo, hi):
  t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
  return t * t * (3 - 2 * t)


@dataclass
class SkiState:
  normal_force: float = 0.0
  edge_deg: float = 0.0
  v_along: float = 0.0
  v_side: float = 0.0
  grip: float = 0.0
  curvature: float = 0.0
  points_down: int = 0


@dataclass
class SkiContact:
  model: mujoco.MjModel
  terrain: HeightGrid
  params: SkiParams = field(default_factory=SkiParams)
  ski_bodies: tuple[str, ...] = ("left_ski", "right_ski")
  drag_body: str | None = "pelvis"

  def __post_init__(self):
    p = self.params
    self.ski_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.ski_bodies]
    if min(self.ski_ids) < 0:
      raise ValueError(f"missing ski bodies {self.ski_bodies}")
    self.drag_id = (
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.drag_body) if self.drag_body else -1
    )
    xs = np.linspace(-0.48, 0.48, p.n_long) * p.length
    self.local_pts = np.array([[x, y, -p.thickness / 2] for x in xs for y in (-p.width / 2, p.width / 2)])
    self.k_pt = p.k_normal / len(self.local_pts)
    self.c_pt = p.c_normal / len(self.local_pts)
    self.state = [SkiState() for _ in self.ski_ids]
    self._vel = np.zeros(6)

  def grip(self, edge_rad):
    p = self.params
    s = smoothstep(abs(edge_rad), np.radians(p.edge_on_deg), np.radians(p.edge_full_deg))
    return p.mu_skid + (p.mu_carve - p.mu_skid) * s, s

  def curvature(self, edge_rad):
    """Signed arc curvature of an edged ski: left edge down (positive edge angle) carves left."""
    p = self.params
    _, s = self.grip(edge_rad)
    return np.sign(edge_rad) * s / (p.sidecut_radius * max(np.cos(edge_rad), 0.2))

  def apply(self, data: mujoco.MjData) -> None:
    p = self.params
    for k, b in enumerate(self.ski_ids):
      st = self.state[k]
      R = data.xmat[b].reshape(3, 3)
      com = data.xipos[b]
      mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, b, self._vel, 0)
      w, v = self._vel[:3], self._vel[3:]

      P = data.xpos[b] + self.local_pts @ R.T
      h, n = self.terrain.height_normal(P[:, 0], P[:, 1])
      depth = (h - P[:, 2]) * n[:, 2]
      down = depth > 0
      force = np.zeros(3)
      torque = np.zeros(3)
      st.points_down = int(down.sum())
      if not down.any():
        st.normal_force = 0.0
        data.xfrc_applied[b] = 0.0
        continue

      P, n, depth = P[down], n[down], depth[down]
      r = P - com
      vp = v + np.cross(w, r)
      fn = np.maximum(self.k_pt * depth - self.c_pt * np.einsum("ij,ij->i", vp, n), 0.0)
      Fn = fn[:, None] * n
      force += Fn.sum(0)
      torque += np.cross(r, Fn).sum(0)

      N = fn.sum()
      st.normal_force = float(N)
      if N > 1e-6:
        cop = (fn[:, None] * P).sum(0) / N
        nbar = (fn[:, None] * n).sum(0)
        nbar /= np.linalg.norm(nbar)
        t = R[:, 0] - R[:, 0].dot(nbar) * nbar
        t /= np.linalg.norm(t)
        lat = np.cross(nbar, t)
        edge = np.arctan2(R[:, 2].dot(lat), R[:, 2].dot(nbar))

        vc = v + np.cross(w, cop - com)
        v_t, v_l = vc.dot(t), vc.dot(lat)
        mu_lat, _ = self.grip(edge)
        G = mu_lat * N
        kappa = self.curvature(edge)

        F_tan = -p.mu_glide * N * np.tanh(v_t / p.glide_v_eps) * t
        F_tan += -np.clip(p.k_lateral * v_l, -G, G) * lat
        force += F_tan
        torque += np.cross(cop - com, F_tan)
        yaw_err = w.dot(nbar) - kappa * v_t
        torque += -np.clip(p.c_yaw * yaw_err, -G * p.length / 4, G * p.length / 4) * nbar

        st.edge_deg = float(np.degrees(edge))
        st.v_along, st.v_side = float(v_t), float(v_l)
        st.grip, st.curvature = float(mu_lat), float(kappa)

      data.xfrc_applied[b, :3] = force
      data.xfrc_applied[b, 3:] = torque

    if self.drag_id >= 0:
      mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.drag_id, self._vel, 0)
      v = self._vel[3:]
      data.xfrc_applied[self.drag_id, :3] = -0.5 * p.rho_air * p.cda * np.linalg.norm(v) * v
