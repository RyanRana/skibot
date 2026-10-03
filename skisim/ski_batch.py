"""SkiContact vectorized over every ski in the model: same force laws as skisim.ski, one numpy pass per step.

This is also the reference the batched torch version for GPU training is checked against.
"""

from __future__ import annotations

import mujoco
import numpy as np

from skisim.ski import SkiParams, smoothstep
from skisim.terrain import HeightGrid


def _unit(v):
  return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _dot(a, b):
  return np.einsum("...i,...i->...", a, b)


class BatchSkiContact:
  def __init__(
    self,
    model: mujoco.MjModel,
    terrain: HeightGrid,
    ski_bodies: list[str],
    drag_bodies: list[str] = (),
    params: SkiParams | None = None,
  ):
    self.model, self.terrain = model, terrain
    self.p = p = params or SkiParams()
    name2id = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    self.ski = np.array([name2id(n) for n in ski_bodies])
    self.drag = np.array([name2id(n) for n in drag_bodies], dtype=int)
    if (self.ski < 0).any() or (self.drag < 0).any():
      raise ValueError("missing ski or drag body")
    self.ski_root = model.body_rootid[self.ski]
    self.drag_root = model.body_rootid[self.drag]
    xs = np.linspace(-0.48, 0.48, p.n_long) * p.length
    self.local = np.array([[x, y, -p.thickness / 2] for x in xs for y in (-p.width / 2, p.width / 2)])
    self.k_pt = p.k_normal / len(self.local)
    self.c_pt = p.c_normal / len(self.local)
    S = len(self.ski)
    self.normal_force = np.zeros(S)
    self.edge = np.zeros(S)
    self.v_along = np.zeros(S)
    self.v_side = np.zeros(S)

  def _vel(self, data, ids, roots, at):
    w = data.cvel[ids, :3]
    v = data.cvel[ids, 3:] + np.cross(w, at - data.subtree_com[roots])
    return w, v

  def apply(self, data: mujoco.MjData) -> None:
    p = self.p
    R = data.xmat[self.ski].reshape(-1, 3, 3)
    com = data.xipos[self.ski]
    w, v = self._vel(data, self.ski, self.ski_root, com)

    P = data.xpos[self.ski][:, None, :] + np.einsum("sij,kj->ski", R, self.local)
    h, n = self.terrain.height_normal(P[..., 0], P[..., 1])
    depth = (h - P[..., 2]) * n[..., 2]
    r = P - com[:, None, :]
    vp = v[:, None, :] + np.cross(w[:, None, :], r)
    fn = np.where(depth > 0, np.maximum(self.k_pt * depth - self.c_pt * _dot(vp, n), 0.0), 0.0)
    Fn = fn[..., None] * n
    force = Fn.sum(1)
    torque = np.cross(r, Fn).sum(1)

    N = fn.sum(1)
    ok = N > 1e-6
    Ns = np.where(ok, N, 1.0)
    cop = (fn[..., None] * P).sum(1) / Ns[:, None]
    nbar = _unit((fn[..., None] * n).sum(1) + (~ok)[:, None] * np.array([0.0, 0.0, 1.0]))
    xs, zs = R[:, :, 0], R[:, :, 2]
    t = _unit(xs - _dot(xs, nbar)[:, None] * nbar)
    lat = np.cross(nbar, t)
    edge = np.arctan2(_dot(zs, lat), _dot(zs, nbar))

    vc = v + np.cross(w, cop - com)
    v_t, v_l = _dot(vc, t), _dot(vc, lat)
    s = smoothstep(np.abs(edge), np.radians(p.edge_on_deg), np.radians(p.edge_full_deg))
    G = (p.mu_skid + (p.mu_carve - p.mu_skid) * s) * N
    kappa = np.sign(edge) * s / (p.sidecut_radius * np.maximum(np.cos(edge), 0.2))

    F_tan = -(p.mu_glide * N * np.tanh(v_t / p.glide_v_eps))[:, None] * t
    F_tan -= np.clip(p.k_lateral * v_l, -G, G)[:, None] * lat
    yaw = -np.clip(p.c_yaw * (_dot(w, nbar) - kappa * v_t), -G * p.length / 4, G * p.length / 4)
    force += ok[:, None] * F_tan
    torque += ok[:, None] * (np.cross(cop - com, F_tan) + yaw[:, None] * nbar)

    data.xfrc_applied[self.ski, :3] = force
    data.xfrc_applied[self.ski, 3:] = torque
    self.normal_force, self.edge, self.v_along, self.v_side = N, np.where(ok, edge, 0.0), v_t, v_l

    if len(self.drag):
      dcom = data.xipos[self.drag]
      _, vd = self._vel(data, self.drag, self.drag_root, dcom)
      data.xfrc_applied[self.drag, :3] = -0.5 * p.rho_air * p.cda * np.linalg.norm(vd, axis=1, keepdims=True) * vd
      data.xfrc_applied[self.drag, 3:] = 0.0
