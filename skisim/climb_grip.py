"""Gripping real holds with the G1's own hand, the Unitree Dex3-1 (3 fingers, 7 joints: thumb 3, index 2, middle 2;
model from MuJoCo Menagerie `g1_with_hands.xml`, Apache 2.0).

Holds are collision shapes, not markers: a jug is a bar standing off the wall with a gap behind it, a crimp a thin edge,
a sloper a broad rounded dome. Each hold type has the grip a climber uses on it, written as a postural synergy (one
closing parameter drives every finger joint in fixed proportion, after Santello et al. 1998 and GraspIt's eigengrasps):

  jug     fingers wrap over and behind the bar, thumb wraps under       (full closure)
  crimp   proximal joints half flexed, distal joints hard flexed onto the edge, thumb locked over the index
  sloper  open hand, fingers barely flexed, thumb out to the side: friction from palm and finger pads

The hand is closed the way GraspIt's autograsp does it: every finger steps along its synergy and a link stops once it
touches the hold (MuJoCo signed distance between the Dex3-1's collision meshes and the hold geoms), the joints past it
keep closing until the fingertip touches too. Contacts per finger and the closure are reported per frame.
"""

from __future__ import annotations

import mujoco
import numpy as np

KINDS = ("jug", "crimp", "sloper")
FINGERS = {"thumb": ("thumb_0", "thumb_1", "thumb_2"), "index": ("index_0", "index_1"), "middle": ("middle_0", "middle_1")}
# Synergy: fraction of each joint's closing range at full closure (thumb_0 rotates the thumb across the palm).
SYNERGY = {
  "jug": {"thumb_0": 0.0, "thumb_1": 0.55, "thumb_2": 0.7, "index_0": 0.85, "index_1": 0.95, "middle_0": 0.85, "middle_1": 0.95},
  "crimp": {"thumb_0": 0.6, "thumb_1": 0.9, "thumb_2": 0.9, "index_0": 0.45, "index_1": 1.0, "middle_0": 0.45, "middle_1": 1.0},
  "sloper": {"thumb_0": -0.4, "thumb_1": 0.15, "thumb_2": 0.15, "index_0": 0.3, "index_1": 0.25, "middle_0": 0.3, "middle_1": 0.25},
}
OPEN = {"thumb_0": 0.0, "thumb_1": 0.1, "thumb_2": 0.15, "index_0": 0.15, "index_1": 0.15, "middle_0": 0.15, "middle_1": 0.15}
TOUCH = 0.002  # m: a link this close to the hold is in contact


def hold_geoms(kind: str, rng) -> tuple[list[dict], float]:
  """Geoms for one hold in its own frame (x out of the wall toward the climber is -x, z up), centred on the grip edge.
  Returns the geoms and how far the grip edge sits below the top of the hold."""
  dark = [0.18 + 0.1 * rng.random(), 0.2, 0.24, 1]
  if kind == "jug":  # bar standing 15 mm off the wall on two posts
    r, half = 0.016, 0.05 + 0.02 * rng.random()
    return [dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, half, 0], pos=[-0.031 - r, 0, 0],
                 quat=[np.sqrt(0.5), np.sqrt(0.5), 0, 0], rgba=dark),
            dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.024, 0.012, 0.012], pos=[-0.024, half * 0.8, -0.006], rgba=dark),
            dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.024, 0.012, 0.012], pos=[-0.024, -half * 0.8, -0.006], rgba=dark)], r
  if kind == "crimp":  # 18 mm deep edge, 8 mm thick
    return [dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.009, 0.05 + 0.02 * rng.random(), 0.004], pos=[-0.009, 0, 0],
                 rgba=dark)], 0.004
  # sloper: a flattened dome
  a = 0.055 + 0.015 * rng.random()
  return [dict(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.035, a, 0.045], pos=[0.0, 0, -0.02], rgba=dark)], 0.025


class Hand:
  """One Dex3-1 hand on a compiled scene."""

  def __init__(self, m: mujoco.MjModel, side: str):
    self.m, self.side = m, side
    self.qadr, self.close = {}, {}
    for f, joints in FINGERS.items():
      for j in joints:
        jt = m.joint(f"{side}_hand_{j}_joint")
        lo, hi = jt.range
        self.qadr[j] = jt.qposadr[0]
        # closing direction: the end of the range away from zero (thumb_0 is symmetric: + is across the palm)
        self.close[j] = (hi if abs(hi) >= abs(lo) else lo) if j != "thumb_0" else (hi if side == "left" else lo)
    self.link_geoms = {}
    for f, joints in FINGERS.items():
      for j in joints:
        b = m.body(f"{side}_hand_{j}_link").id
        self.link_geoms[j] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]

  def set(self, d, shape: dict):
    for j, v in shape.items():
      d.qpos[self.qadr[j]] = v * self.close[j]

  def grip(self, d, kind: str, hold: list[int], steps=30) -> dict:
    """Autograsp along the synergy for `kind`, stopping each link on contact with any geom in `hold`.
    Leaves the closed hand in d.qpos and returns per-finger contact and closure."""
    m = self.m
    syn = SYNERGY[kind]
    frac = {j: OPEN[j] * np.sign(syn[j]) if syn[j] else 0.0 for j in syn}
    self.set(d, frac)
    stopped = {j: False for j in syn}
    touching = {j: False for j in syn}
    for k in range(1, steps + 1):
      s = k / steps
      for f, joints in FINGERS.items():
        for n, j in enumerate(joints):
          if stopped[j]:
            continue
          frac[j] = max(abs(frac[j]), abs(syn[j]) * s) * np.sign(syn[j]) if syn[j] else 0.0
      self.set(d, frac)
      mujoco.mj_kinematics(m, d)
      for f, joints in FINGERS.items():
        for n, j in enumerate(joints):
          if stopped[j]:
            continue
          dist = min((mujoco.mj_geomDistance(m, d, g, h, 0.05, None) for g in self.link_geoms[j] for h in hold), default=1.0)
          if dist < TOUCH:
            touching[j] = True
            # this link and every joint before it in the finger stop; later joints keep curling around the hold
            for jj in joints[: n + 1]:
              stopped[jj] = True
            if dist < -0.004:  # stepped into the hold: back off half a step
              frac[j] -= 0.5 * abs(syn[j]) / steps * np.sign(syn[j])
              self.set(d, frac)
    mujoco.mj_kinematics(m, d)
    out = {f: any(touching[j] for j in joints) for f, joints in FINGERS.items()}
    out["closure"] = float(np.mean([abs(frac[j]) / max(abs(syn[j]), 1e-6) for j in syn if syn[j]]))
    out["links_in_contact"] = int(sum(touching.values()))
    return out

  def open(self, d):
    self.set(d, OPEN)
