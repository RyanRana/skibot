"""Camera work for rendered rides: TV-style shots that never go under the snow or into a tree or gate pole.

Director.frame() turns a shot name and the skier's state into a MuJoCo free camera. Every frame it checks the lens
against the terrain (1.2 m clearance, and the line of sight near the lens must clear the snow too) and against trees
and gate poles (the lens stays outside their footprint, and the view of the skier must not pass through them). When
a shot's ideal spot is blocked, the camera swings around the skier, tilts down or moves in, taking the smallest
change that works, and eases between choices so it does not jump.
"""

from __future__ import annotations

import mujoco
import numpy as np

from skisim.terrain import HeightGrid

# Shot: distance (m), elevation (deg), azimuth reference ("travel" = the skier's direction of travel, "fall" = the
# fall line) and the azimuth offset from it (deg).
SHOTS = {
  "chase": (4.8, -14.0, "travel", 0.0),  # behind the skier
  "side": (8.5, -4.0, "fall", 90.0),  # level across the hill, so the slope shows against the horizon
  "front": (6.5, 4.0, "travel", 200.0),  # low and ahead, looking back up the hill
  "wide": (26.0, -24.0, "fall", 60.0),
}

# Offsets (azimuth deg, elevation deg, distance scale) tried when the ideal spot is blocked.
_TRY = [(0, 0, 1.0), (10, 0, 1.0), (-10, 0, 1.0), (0, -8, 1.0), (20, 0, 1.0), (-20, 0, 1.0), (10, -8, 1.0),
        (-10, -8, 1.0), (0, -16, 1.0), (32, 0, 1.0), (-32, 0, 1.0), (0, 0, 0.75), (20, -10, 0.8), (-20, -10, 0.8),
        (45, -6, 0.9), (-45, -6, 0.9), (0, -26, 0.85), (60, -10, 0.8), (-60, -10, 0.8), (0, -40, 0.7),
        (90, -8, 0.9), (-90, -8, 0.9), (130, -6, 0.9), (-130, -6, 0.9), (180, -4, 1.0), (0, -45, 1.5)]


def _change(a, b) -> float:
  """How big a change between two offsets looks on screen (degrees, roughly)."""
  return abs(_wrap(a[0] - b[0])) + 2 * abs(a[1] - b[1]) + 40 * abs(a[2] - b[2])


def _wrap(a: float) -> float:
  return (a + 180.0) % 360.0 - 180.0


def lens_position(lookat, dist: float, az: float, el: float) -> np.ndarray:
  a, e = np.radians(az), np.radians(el)
  return np.asarray(lookat, float) - dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def snow_lift(grid: HeightGrid, lookat, dist: float, az: float, el: float, clearance: float = 1.2):
  """(elevation, distance) at or below `el` that puts the lens `clearance` m above the snow, with the line of sight
  near the lens clear of the snow as well. If a steep down-angle is not enough, the camera moves in."""
  fr = np.array([0.0, 0.25, 0.5])
  need = clearance * (1.0 - fr)
  look = np.asarray(lookat, float)

  def ok(e, d):
    p = lens_position(look, d, az, e)
    pts = p + fr[:, None] * (look - p)
    h, _ = grid.height_normal(pts[:, 0], pts[:, 1])
    return bool(np.all(pts[:, 2] >= h + need))

  e, d = float(el), float(dist)
  if ok(e, d):
    return e, d
  while not ok(-80.0, d) and d > 1.5:  # even looking almost straight down fails: move in
    d *= 0.9
  lo, hi = -80.0, e  # ok at lo, not at hi: bisect to a quarter degree
  while hi - lo > 0.25:
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if ok(mid, d) else (lo, mid)
  return lo, d


class Obstacles:
  """Trees and gate poles as upright cylinders: the lens stays out of them and the view of the skier does not cross them."""

  def __init__(self, grid: HeightGrid, trees=None, poles=None):
    xs, ys, rs, hs = [], [], [], []
    if trees is not None and len(trees):
      t = np.asarray(trees, float)
      # Voxel canopies are square slabs up to 1.45 r half wide, so their corners reach 2.05 r from the trunk.
      xs += [t[:, 0]]; ys += [t[:, 1]]; rs += [2.05 * t[:, 2] + 0.1]; hs += [t[:, 3] + 0.3]
    if poles is not None and len(poles):
      p = np.asarray(poles, float)
      xs += [p[:, 0]]; ys += [p[:, 1]]; rs += [np.full(len(p), 0.3)]; hs += [np.full(len(p), 1.5)]
    self.x = np.concatenate(xs) if xs else np.zeros(0)
    self.y = np.concatenate(ys) if ys else np.zeros(0)
    self.r = np.concatenate(rs) if rs else np.zeros(0)
    ground = grid.height_normal(self.x, self.y)[0] if len(self.x) else np.zeros(0)
    self.base = ground
    self.top = ground + (np.concatenate(hs) if hs else np.zeros(0))
    self.rmax = float(self.r.max()) if len(self.r) else 0.0

  def crowding(self, p: np.ndarray, look: np.ndarray, near: float, half_fov: float = 40.0) -> float:
    """How much an obstacle near the lens fills the picture: 0 for none, toward 1 for one close up in the middle."""
    if len(self.x) == 0:
      return 0.0
    sel = np.hypot(self.x - p[0], self.y - p[1]) < near + self.rmax
    if not sel.any():
      return 0.0
    fwd = (look - p) / (np.linalg.norm(look - p) + 1e-9)
    worst = 0.0
    for frac in (0.15, 0.5, 0.9):  # low, middle and top of each trunk-and-canopy column
      q = np.stack([self.x[sel], self.y[sel], self.base[sel] + frac * (self.top[sel] - self.base[sel])], 1) - p
      dist = np.linalg.norm(q, axis=1) + 1e-9
      ang = np.degrees(np.arccos(np.clip(q @ fwd / dist, -1, 1))) - np.degrees(np.arctan(self.r[sel] / dist))
      score = np.clip(1 - ang / half_fov, 0, 1) * np.clip(1 - dist / near, 0, 1)
      worst = max(worst, float(score.max()))
    return worst

  def gap(self, p: np.ndarray) -> float:
    """Horizontal distance from p to the edge of the nearest footprint whose height range p is within (inf if none)."""
    if len(self.x) == 0:
      return np.inf
    level = (p[2] < self.top) & (p[2] > self.base - 1.0)
    if not level.any():
      return np.inf
    return float((np.hypot(self.x[level] - p[0], self.y[level] - p[1]) - self.r[level]).min())

  def hits(self, pts: np.ndarray, margin: float) -> bool:
    if len(self.x) == 0:
      return False
    pad = self.rmax + margin
    lo, hi = pts[:, :2].min(0) - pad, pts[:, :2].max(0) + pad
    sel = (self.x > lo[0]) & (self.x < hi[0]) & (self.y > lo[1]) & (self.y < hi[1])
    if not sel.any():
      return False
    dx = pts[:, 0, None] - self.x[sel]
    dy = pts[:, 1, None] - self.y[sel]
    z = pts[:, 2, None]
    inside = (dx * dx + dy * dy < (self.r[sel] + margin) ** 2) & (z < self.top[sel] + margin) & (z > self.base[sel] - 1.0)
    return bool(inside.any())


class Director:
  """Builds each frame's camera for a named shot, keeping it above the snow and clear of trees and poles."""

  def __init__(self, grid: HeightGrid, obstacles: Obstacles | None = None, clearance: float = 1.2):
    self.grid = grid
    self.obs = obstacles or Obstacles(grid)
    self.clearance = clearance
    self.shot = None
    self.off = np.array([0.0, 0.0, 1.0])  # applied (azimuth, elevation, distance scale) offset
    self.lift = 0.0  # degrees the snow check tilted the camera down, released slowly
    self.ref = {}  # smoothed reference azimuths (travel, fall)
    self.last_look = None

  def _check(self, look, az, el, dist):
    """(elevation, distance) after the snow lift, whether the lens is outside every obstacle, and a cost for the
    view: 2 if a tree or pole blocks the skier, plus how much an obstacle crowds the foreground (0 to 1)."""
    el2, d2 = snow_lift(self.grid, look, dist, az, el, self.clearance)
    p = lens_position(look, d2, az, el2)
    hard = not self.obs.hits(p[None], margin=0.6)
    n = max(3, int(d2 / 0.4))
    fr = np.linspace(0.0, max(0.0, 1.0 - 1.2 / d2), n)  # stop 1.2 m short of the skier
    blocked = self.obs.hits(p + fr[:, None] * (np.asarray(look) - p), margin=0.1)
    cost = 2.0 * blocked + self.obs.crowding(p, np.asarray(look), near=max(4.0, 0.6 * d2), half_fov=44.0)  # 44: frame corners
    return el2, d2, hard, cost

  def frame(self, shot: str, lookat, travel_deg: float, fall_deg: float, dt: float) -> mujoco.MjvCamera:
    look = np.asarray(lookat, float)
    cut = shot != self.shot or self.last_look is None or np.linalg.norm(look - self.last_look) > 6.0
    self.shot, self.last_look = shot, look.copy()
    # Smooth the reference directions: carving swings the velocity, bumps swing the slope normal.
    for key, val, tau in (("travel", travel_deg, 0.45), ("fall", fall_deg, 0.8)):
      if cut or key not in self.ref:
        self.ref[key] = val
      else:
        self.ref[key] += _wrap(val - self.ref[key]) * min(1.0, dt / tau)
    dist0, el0, rel, az_off = SHOTS[shot]
    az0 = self.ref[rel] + az_off

    def config(o):
      return az0 + o[0], el0 + o[1], dist0 * o[2]

    if cut:
      self.off[:] = (0.0, 0.0, 1.0)
      self.lift = 0.0
    good = 0.12  # a view this uncluttered needs no search
    cur = self._check(look, *config(self.off))
    if cur[2] and cur[3] < good:
      home = self._check(look, *config((0.0, 0.0, 1.0)))
      target = np.array([0.0, 0.0, 1.0]) if home[2] and home[3] < good else self.off.copy()
    else:
      # Least cluttered spot the lens can be in, preferring small changes from where the camera is now.
      best, target = np.inf, None
      for o in _TRY:
        res = self._check(look, *config(o))
        if not res[2]:
          continue
        score = res[3] + _change(o, self.off) / 300.0 + _change(o, (0.0, 0.0, 1.0)) / 600.0
        if score < best:
          best, target = score, np.array(o, float)
      if target is None:
        target = np.array([0.0, -50.0, 1.0])
    if cut or not cur[2] or _change(target, self.off) > 70.0:
      self.off = target.copy()  # lens inside something, or a big move: cut straight to the new spot
    else:
      diff = target - self.off
      diff[0] = _wrap(diff[0])
      self.off += np.clip(diff, -np.array([50.0, 35.0, 0.6]) * dt, np.array([50.0, 35.0, 0.6]) * dt)
    az, el, dist = config(self.off)
    el_req, dist_req = snow_lift(self.grid, look, dist, az, el, self.clearance)
    if self.obs.hits(lens_position(look, dist_req, az, el_req)[None], margin=0.6):
      self.off = target.copy()  # easing toward the target passed through an obstacle: cut to the target
      az, el, dist = config(self.off)
      el_req, dist_req = snow_lift(self.grid, look, dist, az, el, self.clearance)
    need = el - el_req
    self.lift = need if cut else max(need, self.lift - 25.0 * dt)
    el_out = el - self.lift
    if el_out < el_req and self.obs.hits(lens_position(look, dist_req, az, el_out)[None], margin=0.6):
      el_out = el_req  # easing down would put the lens in a tree: take the plain snow lift
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = dist_req, az, min(el_out, el_req)
    return cam


def piste_mask(grid: HeightGrid, line_xy, half_width: float = 13.0, ramp: float = 4.0, seed: int = 0) -> np.ndarray:
  """(nrow, ncol) in [0, 1]: 1 on the groomed piste along a course centerline, 0 off it, with a soft edge that
  wanders by a meter or two like a real groomed run."""
  xy = np.asarray(line_xy, float)[:, :2]
  s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
  ss = np.arange(0.0, s[-1] + 1e-6, 0.5 * grid.cell)
  pts = np.stack([np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])], 1)
  dist = np.full((grid.nrow, grid.ncol), np.inf)
  k = int(np.ceil((half_width + ramp + 4.0) / grid.cell))
  for px, py in pts:
    c = int(round((px - grid.x0) / grid.cell))
    r = int(round((py - grid.y0) / grid.cell))
    r0, r1, c0, c1 = max(r - k, 0), min(r + k + 1, grid.nrow), max(c - k, 0), min(c + k + 1, grid.ncol)
    if r0 >= r1 or c0 >= c1:
      continue
    yy = grid.y0 + np.arange(r0, r1)[:, None] * grid.cell
    xx = grid.x0 + np.arange(c0, c1)[None, :] * grid.cell
    np.minimum(dist[r0:r1, c0:c1], np.hypot(xx - px, yy - py), out=dist[r0:r1, c0:c1])
  rng = np.random.default_rng(seed)
  fy = np.fft.fftfreq(grid.nrow)[:, None]
  fx = np.fft.fftfreq(grid.ncol)[None, :]
  wob = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(dist.shape)) * np.exp(-2.0 * (np.hypot(fx, fy) * 40.0 / grid.cell) ** 2)))
  wob = 1.5 * (wob - wob.mean()) / (wob.std() + 1e-9)
  x = np.clip((half_width + wob - dist) / ramp + 0.5, 0.0, 1.0)
  return x * x * (3 - 2 * x)
