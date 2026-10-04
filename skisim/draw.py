"""Scene decorations drawn as extra MuJoCo scene geoms (visual only, no physics): conifer trees.

Two styles:
- "voxel" (default): 8-bit pixel-art pines. A brown box trunk, stepped stacks of green boxes narrowing upward in
  a few shades of green, white snow blocks on the upper steps and a snow tip.
- "smooth": the earlier stacked ellipsoids with a snow cap.
"""

from __future__ import annotations

import functools

import mujoco
import numpy as np

TRUNK = np.array([0.36, 0.25, 0.16, 1.0], np.float32)
NEEDLES = np.array([0.10, 0.25, 0.17, 1.0], np.float32)
SNOW = np.array([0.95, 0.97, 1.0, 1.0], np.float32)

# Voxel palette: four greens from shadowy to sunlit, a pixel-art brown and a slightly blue snow white.
GREENS = np.array([[0.06, 0.25, 0.12, 1.0], [0.09, 0.33, 0.15, 1.0], [0.13, 0.42, 0.18, 1.0], [0.19, 0.50, 0.22, 1.0]], np.float32)
VOXEL_TRUNK = np.array([0.40, 0.25, 0.13, 1.0], np.float32)
VOXEL_SNOW = np.array([0.94, 0.96, 1.0, 1.0], np.float32)


def draw_trees(scene, trees: np.ndarray, grid, center_xy, radius: float = 70.0, max_trees: int = 400, style: str = "voxel"):
  """trees: rows (x, y, radius, height, ...) as in trees.npy. Draws the ones within `radius` of center_xy.

  Positions and sizes come straight from the rows: radius is the solid trunk-and-lower-branches radius the skier must
  clear, height is the drawn height.
  """
  if trees is None or len(trees) == 0:
    return
  d = np.hypot(trees[:, 0] - center_xy[0], trees[:, 1] - center_xy[1])
  near = np.argsort(d)[: max_trees]
  near = near[d[near] < radius]
  if len(near) == 0:
    return
  h0, _ = grid.height_normal(trees[near, 0].astype(float), trees[near, 1].astype(float))
  if style == "smooth":
    _smooth_trees(scene, trees[near, :4], h0)
  elif style == "voxel":
    _voxel_trees(scene, trees[near, :4], h0)
  else:
    raise ValueError(f"unknown tree style {style!r} (voxel or smooth)")


def _smooth_trees(scene, rows, h0):
  eye = np.eye(3).ravel()
  for (x, y, r, h), z in zip(rows, h0):
    if scene.ngeom + 6 > scene.maxgeom:
      return
    trunk_h = 0.18 * h
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.12 * r + 0.05, trunk_h / 2, 0]),
                        np.array([x, y, z + trunk_h / 2]), eye, TRUNK)
    scene.ngeom += 1
    # Three tiers of needles, wide at the bottom and narrowing toward the top, with a snow cap.
    for k, (frac_z, frac_r, frac_h) in enumerate(((0.38, 1.0, 0.22), (0.60, 0.72, 0.18), (0.80, 0.45, 0.14))):
      mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          np.array([r * 1.6 * frac_r, r * 1.6 * frac_r, h * frac_h]), np.array([x, y, z + h * frac_z]), eye, NEEDLES)
      scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([r * 0.45, r * 0.45, h * 0.07]), np.array([x, y, z + h * 0.93]), eye, SNOW)
    scene.ngeom += 1


def voxel_tree_boxes(x: float, y: float, z: float, r: float, h: float):
  """Boxes of one 8-bit pine standing at (x, y) on ground height z: a list of (half sizes, center, rgba), plus its yaw.

  Layers step in by a constant amount (like pixels), get lighter toward the top, and the upper steps carry snow
  blocks that overhang their step by a couple of centimeters.
  """
  seed = int(abs(x * 73.856 + y * 19.349) * 1000) % 2**31
  rng = np.random.default_rng(seed)
  yaw = rng.uniform(0, np.pi / 2)
  shade = int(rng.integers(0, 2))
  n = int(np.clip(round(h / 1.1), 3, 6))
  trunk_h = 0.17 * h
  tip_h = 0.08 * h
  t = (h - trunk_h - tip_h) / n  # layer thickness
  w0 = 1.45 * r  # bottom layer half width
  step = (w0 - 0.28 * w0) / max(n - 1, 1)
  boxes = []
  tw = 0.15 * r + 0.06
  boxes.append(((tw, tw, trunk_h / 2 + 0.12), (x, y, z + trunk_h / 2 - 0.06), VOXEL_TRUNK))
  snow_from = max(1, n // 2 - 1)
  for k in range(n):
    w = w0 - k * step
    zb = z + trunk_h + k * t
    g = GREENS[int(np.clip(round(k / max(n - 1, 1) * 2) + shade, 0, 3))]
    boxes.append(((w, w, t / 2), (x, y, zb + t / 2), g))
    if k >= snow_from:
      s = max(0.06, 0.2 * t)
      boxes.append(((w + 0.025, w + 0.025, s / 2), (x, y, zb + t - 0.3 * s + s / 2), VOXEL_SNOW))
  tw = max(0.12, 0.45 * (w0 - (n - 1) * step))
  boxes.append(((tw, tw, tip_h / 2 + 0.05), (x, y, z + h - tip_h / 2), VOXEL_SNOW))
  return boxes, yaw


@functools.lru_cache(maxsize=16384)
def _voxel_tree_geoms(x: float, y: float, z: float, r: float, h: float):
  """voxel_tree_boxes as ready-made arrays, cached: trees do not move, so each is built once, not every frame."""
  boxes, yaw = voxel_tree_boxes(x, y, z, r, h)
  c, s = np.cos(yaw), np.sin(yaw)
  mat = np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
  return tuple((np.array(size, float), np.array(pos, float), rgba, rgba is VOXEL_SNOW) for size, pos, rgba in boxes), mat


def _voxel_trees(scene, rows, h0):
  box = mujoco.mjtGeom.mjGEOM_BOX
  for (x, y, r, h), z in zip(rows, h0):
    geoms, mat = _voxel_tree_geoms(float(x), float(y), float(z), float(r), float(h))
    if scene.ngeom + len(geoms) > scene.maxgeom:
      return
    for size, pos, rgba, snow in geoms:
      g = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(g, box, size, pos, mat, rgba)
      if snow:  # snow reads white even on faces turned from the sun
        g.emission, g.specular, g.shininess = 0.35, 0.25, 0.3
      else:  # matte needles and bark
        g.specular, g.shininess = 0.08, 0.3
      scene.ngeom += 1


# Ski poles, visual only (no physics). The grip follows each hand; the shaft trails down and back from it in the
# robot's own heading frame, tips behind the boots like a racer's, whatever the arms are doing (a rigid grip made the
# poles stick out sideways whenever the policy held its arms wide).
POLE_GRIP = np.array([0.08, 0.0, 0.0])  # m, from the wrist yaw link to the middle of the fist
POLE_LEN = 0.85  # m, a 1.2 m adult pole at G1 scale
SHAFT = np.array([0.16, 0.17, 0.19, 1.0], np.float32)
GRIP = np.array([0.05, 0.05, 0.06, 1.0], np.float32)
BASKET = np.array([0.08, 0.08, 0.09, 1.0], np.float32)


def _frame_z(u: np.ndarray) -> np.ndarray:
  """Rotation (flattened, row-major) whose z axis is the unit vector u."""
  a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
  x = np.cross(a, u)
  x /= np.linalg.norm(x)
  y = np.cross(u, x)
  return np.stack([x, y, u], 1).ravel()


def draw_ski_poles(scene, model, data, pelvis: str = "pelvis"):
  """A pole in each hand of the G1 in `data` (bodies left/right_wrist_yaw_link)."""
  pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pelvis)
  Rp = data.xmat[pid].reshape(3, 3) if pid >= 0 else np.eye(3)
  fwd = np.array([Rp[0, 0], Rp[1, 0], 0.0])
  fwd /= np.linalg.norm(fwd) + 1e-9
  left = np.array([-fwd[1], fwd[0], 0.0])
  for side, sgn in (("left", 1.0), ("right", -1.0)):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    if bid < 0 or scene.ngeom + 3 > scene.maxgeom:
      continue
    R = data.xmat[bid].reshape(3, 3)
    grip = data.xpos[bid] + R @ POLE_GRIP
    u = -0.55 * fwd - 0.82 * np.array([0, 0, 1.0]) + sgn * 0.12 * left
    u /= np.linalg.norm(u)
    mat = _frame_z(u)
    for size, center, rgba, kind in (
      (np.array([0.009, POLE_LEN / 2, 0]), grip + u * (POLE_LEN / 2 - 0.05), SHAFT, mujoco.mjtGeom.mjGEOM_CAPSULE),
      (np.array([0.016, 0.06, 0]), grip, GRIP, mujoco.mjtGeom.mjGEOM_CAPSULE),
      (np.array([0.04, 0.04, 0.004]), grip + u * (POLE_LEN - 0.12), BASKET, mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
      mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, size, center, mat, rgba)
      scene.ngeom += 1
