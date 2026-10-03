"""Batched torch version of skisim.ski_batch for GPU training: same force laws, any number of worlds.

Shapes: B worlds, S skis per world. Terrain is a TorchGrid (one shared mosaic).
"""

from __future__ import annotations

import math

import torch

from skisim.ski import SkiParams


class TorchGrid:
  def __init__(self, z, x0: float, y0: float, cell: float, device):
    self.z = torch.as_tensor(z, dtype=torch.float32, device=device)
    self.x0, self.y0, self.cell = x0, y0, cell
    self.nrow, self.ncol = self.z.shape

  def height_normal(self, x, y):
    fx = (x - self.x0) / self.cell
    fy = (y - self.y0) / self.cell
    c = torch.clamp(torch.floor(fx), 0, self.ncol - 2).long()
    r = torch.clamp(torch.floor(fy), 0, self.nrow - 2).long()
    tx = torch.clamp(fx - c, 0, 1)
    ty = torch.clamp(fy - r, 0, 1)
    z00 = self.z[r, c]
    z01 = self.z[r, c + 1]
    z10 = self.z[r + 1, c]
    z11 = self.z[r + 1, c + 1]
    h = z00 * (1 - tx) * (1 - ty) + z01 * tx * (1 - ty) + z10 * (1 - tx) * ty + z11 * tx * ty
    dzdx = ((z01 - z00) * (1 - ty) + (z11 - z10) * ty) / self.cell
    dzdy = ((z10 - z00) * (1 - tx) + (z11 - z01) * tx) / self.cell
    n = torch.stack([-dzdx, -dzdy, torch.ones_like(h)], -1)
    return h, n / n.norm(dim=-1, keepdim=True)


def _smoothstep(x, lo, hi):
  t = torch.clamp((x - lo) / (hi - lo), 0, 1)
  return t * t * (3 - 2 * t)


def _unit(v):
  return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _dot(a, b):
  return (a * b).sum(-1)


class TorchSkiContact:
  def __init__(self, local_pts_count: int | None = None, params: SkiParams | None = None, device="cpu"):
    p = self.p = params or SkiParams()
    xs = torch.linspace(-0.48, 0.48, p.n_long) * p.length
    pts = [[float(x), y, -p.thickness / 2] for x in xs for y in (-p.width / 2, p.width / 2)]
    self.local = torch.tensor(pts, dtype=torch.float32, device=device)  # (K, 3)
    self.k_pt = p.k_normal / len(pts)
    self.c_pt = p.c_normal / len(pts)
    self.edge_on = math.radians(p.edge_on_deg)
    self.edge_full = math.radians(p.edge_full_deg)

  def forces(self, grid: TorchGrid, xpos, xmat, com, w, v, snow: dict | None = None):
    """xpos, com, w, v: (..., 3) world-frame ski body origin, COM, angular and COM linear velocity.
    xmat: (..., 3, 3). snow: optional per-world tensors shaped (B, 1) for mu_glide, mu_skid, mu_carve, k_normal.
    Returns force, torque at COM (..., 3) and diagnostics."""
    p = self.p
    K = self.local.shape[0]
    if snow is None:
      mu_glide, mu_skid, mu_carve, k_pt, c_pt = p.mu_glide, p.mu_skid, p.mu_carve, self.k_pt, self.c_pt
    else:
      mu_glide, mu_skid, mu_carve = snow["mu_glide"], snow["mu_skid"], snow["mu_carve"]
      k = snow["k_normal"]
      k_pt = (k / K)[..., None]
      c_pt = (p.c_normal * torch.sqrt(k / p.k_normal) / K)[..., None]  # keep the damping ratio
    P = xpos[..., None, :] + torch.einsum("...ij,kj->...ki", xmat, self.local)  # (..., K, 3)
    h, n = grid.height_normal(P[..., 0], P[..., 1])
    depth = (h - P[..., 2]) * n[..., 2]
    r = P - com[..., None, :]
    vp = v[..., None, :] + torch.cross(w[..., None, :].expand_as(r), r, dim=-1)
    fn = torch.where(depth > 0, torch.clamp(k_pt * depth - c_pt * _dot(vp, n), min=0), torch.zeros_like(depth))
    Fn = fn[..., None] * n
    force = Fn.sum(-2)
    torque = torch.cross(r, Fn, dim=-1).sum(-2)

    N = fn.sum(-1)
    ok = N > 1e-6
    Ns = torch.where(ok, N, torch.ones_like(N))
    cop = (fn[..., None] * P).sum(-2) / Ns[..., None]
    up = torch.zeros_like(cop)
    up[..., 2] = 1
    nbar = _unit((fn[..., None] * n).sum(-2) + (~ok)[..., None] * up)
    xs, zs = xmat[..., :, 0], xmat[..., :, 2]
    t = _unit(xs - _dot(xs, nbar)[..., None] * nbar)
    lat = torch.cross(nbar, t, dim=-1)
    edge = torch.atan2(_dot(zs, lat), _dot(zs, nbar))
    vc = v + torch.cross(w, cop - com, dim=-1)
    v_t, v_l = _dot(vc, t), _dot(vc, lat)
    s = _smoothstep(edge.abs(), self.edge_on, self.edge_full)
    G = (mu_skid + (mu_carve - mu_skid) * s) * N
    kappa = torch.sign(edge) * s / (p.sidecut_radius * torch.clamp(torch.cos(edge), min=0.2))
    F_tan = -(mu_glide * N * torch.tanh(v_t / p.glide_v_eps))[..., None] * t
    F_tan = F_tan - torch.clamp(p.k_lateral * v_l, -G, G)[..., None] * lat
    yaw_lim = G * p.length / 4
    yaw = -torch.clamp(p.c_yaw * (_dot(w, nbar) - kappa * v_t), -yaw_lim, yaw_lim)
    okf = ok[..., None].float()
    force = force + okf * F_tan
    torque = torque + okf * (torch.cross(cop - com, F_tan, dim=-1) + yaw[..., None] * nbar)
    # Touch: normal force split into front/rear x left/right edge (points are ordered x-major, then right, left).
    K = fn.shape[-1]
    fz = fn.reshape(*fn.shape[:-1], K // 2, 2)
    half = (K // 2) // 2
    zones = torch.stack([fz[..., half:, 1].sum(-1), fz[..., half:, 0].sum(-1), fz[..., :half, 1].sum(-1), fz[..., :half, 0].sum(-1)], -1)
    return force, torque, {"N": N, "edge": torch.where(ok, edge, torch.zeros_like(edge)), "v_t": v_t, "v_l": v_l, "zones": zones}


def drag(v, p: SkiParams):
  return -0.5 * p.rho_air * p.cda * v.norm(dim=-1, keepdim=True) * v
