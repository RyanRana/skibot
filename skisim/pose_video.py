"""Pose data video from follow-cam ski footage: the skier with a clean 3D-pose skeleton, the Unitree G1 carving
the same turns beside them, live joint values, and smoothed joint and virtual IMU recordings.

Run: .venv/bin/python -m skisim.pose_video data/videos/gopro/QNkNHEG8qC8.mp4 --title "TED LIGETY · GS FOLLOW CAM"
Writes out/pose/pose_data_<video>.mp4 and out/pose/pose_data_<video>.json (per-frame recordings).

Lean comes from the image (the line from the feet to the hips against the frame's vertical, which a follow cam
from behind keeps close to true vertical). Joint angles come from MediaPipe's 3D world landmarks. The virtual IMU
is the torso's rotation rate from the 3D pose: a gyroscope predicted from video.

A racer leaning 50 degrees is not falling: the turn holds them up. So the G1 does not just hold the pose, it carves.
Its heading turns at g tan(lean) / v, the rate at which that lean is in balance, both skis stay on the snow on their
edges (the lower leg folds so the other ski reaches), and a chase cam follows it down a groomed slope with its tracks
behind it. MediaPipe runs once over the whole clip (cached). Frames where spray hides the legs are kept (MediaPipe
still places them), outliers are dropped, and gaps are filled and smoothed without lag, so the robot moves in step
with the racer on every frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter

from skisim.draw import draw_trees
from skisim.pose_from_video import (EDGES, KEY, L_AN, L_EL, L_HIP, L_KN, L_SH, L_WR, R_AN, R_EL, R_HIP, R_KN, R_SH, R_WR,
                                    frame_angles, to_g1, unit)
from skisim.scene import g1_model
from skisim.ski import SkiParams
from skisim.sonic import DEFAULT_ANGLES, JOINTS
from skisim.terrain import slope

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
CW, CH = 1920, 1080
VW, VH = 1152, 648
GW, GH = CW - VW, VH
TOP = 52
PLOT_S = 6.0
BG, PANEL, RULE = (11, 18, 32), (17, 27, 46), (36, 51, 76)
LEFT_COL, RIGHT_COL, ACC = (70, 200, 255), (255, 160, 60), (240, 90, 100)
LEFT_SET = {L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN, 29, 31}
RIGHT_SET = {R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN, 30, 32}
CORE = [L_SH, R_SH, L_HIP, R_HIP]
LEGS = [L_KN, R_KN, L_AN, R_AN]

G = 9.81
SPEED = 14.0  # m/s. The video gives no speed: a fast free-ski carve (about 60 km/h), Froude-scaled to the G1 (0.73 of a person).
SLOPE_DEG = 14.0
MAX_GAP_S = 1.5  # longer gaps than this ease back to a neutral stance instead of bridging two turns
LEG = {s: [JOINTS.index(f"{s}_{j}_joint") for j in ("hip_pitch", "knee", "ankle_pitch")] for s in ("left", "right")}
WAIST_ROLL = JOINTS.index("waist_roll_joint")
HIP_ROLL = {s: JOINTS.index(f"{s}_hip_roll_joint") for s in ("left", "right")}
HIP_YAW = {s: JOINTS.index(f"{s}_hip_yaw_joint") for s in ("left", "right")}


def font(size, bold=False):
  for p in (f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf", "/System/Library/Fonts/Helvetica.ttc"):
    if Path(p).exists():
      return ImageFont.truetype(p, size)
  return ImageFont.load_default()


def torso_frame(P):
  mid_hip = (P[L_HIP] + P[R_HIP]) / 2
  up = unit((P[L_SH] + P[R_SH]) / 2 - mid_hip)
  right = P[R_HIP] - P[L_HIP]
  right = unit(right - (right @ up) * up)
  return np.stack([np.cross(up, right), -right, up], 1)


# --- Pass 1: MediaPipe over the clip ---------------------------------------------------------------------------------

def detect(video: Path, start: float, seconds: float) -> dict:
  """Image landmarks, 3D world landmarks and visibility for every frame (NaN where no skier was found), cached."""
  cache = ROOT / "out" / "pose" / f"_detect_{video.stem}_{start:g}_{seconds:g}.npz"
  if cache.exists():
    z = np.load(cache)
    return {k: z[k] for k in z.files}
  cap = cv2.VideoCapture(str(video))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  step = 2 if fps > 32 else 1
  if start:
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
  pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.6)
  uv, P, vis, k = [], [], [], 0
  while True:
    ok, img = cap.read()
    if not ok:
      break
    k += 1
    if (k - 1) % step:
      continue
    if seconds and (k - 1) / fps > seconds:
      break
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if res.pose_world_landmarks:
      lm = res.pose_landmarks.landmark
      uv.append([[p.x, p.y] for p in lm])
      vis.append([p.visibility for p in lm])
      P.append([[p.x, -p.y, -p.z] for p in res.pose_world_landmarks.landmark])
    else:
      uv.append(np.full((33, 2), np.nan))
      vis.append(np.zeros(33))
      P.append(np.full((33, 3), np.nan))
    if len(uv) % 100 == 0:
      print(f"  pose {len(uv)} frames", flush=True)
  cap.release()
  pose.close()
  out = {"uv": np.array(uv), "P": np.array(P), "vis": np.array(vis), "fps": np.array(fps), "step": np.array(step),
         "size": np.array([W, H])}
  np.savez_compressed(cache, **out)
  return out


def measure(uv, P, size):
  """One frame's numbers: human joint angles, lean and torso tilt in the image, and torso yaw for the virtual IMU."""
  pts = uv * size
  hip, feet = (pts[L_HIP] + pts[R_HIP]) / 2, (pts[L_AN] + pts[R_AN]) / 2
  sh = (pts[L_SH] + pts[R_SH]) / 2
  fwd = torso_frame(P)[:, 0]
  m = frame_angles(P)
  m["lean"] = float(np.degrees(np.arctan2(hip[0] - feet[0], feet[1] - hip[1])))  # + = leaning to the racer's right
  m["torso"] = float(np.degrees(np.arctan2(sh[0] - hip[0], hip[1] - sh[1])))
  m["yaw"] = float(np.degrees(np.arctan2(fwd[0], -fwd[2])))  # world landmarks: x right, y up, z toward the camera
  for k in ("lean", "torso"):  # Ligety gets his hip to the snow, about 75 deg; past 80 the skeleton is wrong
    if abs(m[k]) > 80:
      m[k] = np.nan
  return m


def clean(t, y, ok, neutral, sigma=1.5):
  """Outliers out (Hampel), gaps filled with a monotone spline, then a light zero-phase smooth: no lag."""
  idx = np.flatnonzero(ok & np.isfinite(y))
  if len(idx) < 4:
    return np.full_like(t, neutral)
  v = y[idx]
  med = median_filter(v, size=9, mode="nearest")
  dev = np.abs(v - med)
  mad = 1.4826 * median_filter(dev, size=9, mode="nearest")
  keep = dev <= 3 * np.maximum(mad, 0.5 * np.median(mad) + 1e-6)
  tx, ty = list(t[idx[keep]]), list(v[keep])
  # Across a long gap the racer has made turns we did not see: ease to neutral rather than bridge them.
  for i in range(len(tx) - 1, 0, -1):
    if tx[i] - tx[i - 1] > MAX_GAP_S:
      tx[i:i] = [tx[i - 1] + 0.6, tx[i] - 0.6]
      ty[i:i] = [neutral, neutral]
  f = PchipInterpolator(tx, ty, extrapolate=False)
  out = f(t)
  out[t < tx[0]], out[t > tx[-1]] = ty[0], ty[-1]
  return gaussian_filter1d(out, sigma, mode="nearest")


def signals(det: dict):
  """Per-frame series for everything the video shows and the robot needs, measured where possible, filled elsewhere."""
  n = len(det["uv"])
  dt = float(det["step"] / det["fps"])
  t = np.arange(n) * dt
  vis = det["vis"]
  # Spray hides the knees and ankles on most carves; MediaPipe still places them, so a low leg score is kept and
  # left to the outlier filter. Only frames without a clear torso are dropped.
  ok = (vis[:, CORE].min(1) > 0.5) & (vis[:, LEGS].mean(1) > 0.15)
  raw = [measure(det["uv"][i], det["P"][i], det["size"]) if ok[i] else None for i in range(n)]
  keys = list(next(r for r in raw if r is not None))
  series = {k: np.array([np.nan if r is None else r[k] for r in raw]) for k in keys}
  neutral = {k: float(np.nanmedian(v)) for k, v in series.items()}
  neutral["lean"] = neutral["torso"] = neutral["feet_right_of_hips"] = 0.0
  s = {k: clean(t, v, ok, neutral[k]) for k, v in series.items() if k != "yaw"}
  # Virtual gyro from frame-to-frame torso yaw. A jump past 12 deg in one frame (360 deg/s) is MediaPipe flipping the
  # torso front to back, not a rotation, so it is dropped rather than differentiated.
  d = (np.diff(series["yaw"]) + 180) % 360 - 180
  rate = np.concatenate([[np.nan], np.where(np.abs(d) < 12, -d / dt, np.nan)])  # + = turning left, about the torso's up axis
  s["gyro"] = clean(t, rate, np.isfinite(rate), 0.0, sigma=3.0)
  return t, dt, ok, s


# --- The G1 carving ----------------------------------------------------------------------------------------------------

def carve(lean_deg, dt, v=SPEED, slope_deg=SLOPE_DEG):
  """Heading and position in the slope plane from lean alone. A skier carving at lean θ turns at ω = g tan θ / v,
  the turn whose centripetal pull balances that lean. Gravity along the slope bends the heading back toward the fall
  line. Returns heading (rad, from the fall line, + to the left), distance down the slope, sideways offset, ω."""
  th = np.radians(np.clip(lean_deg, -65, 65))
  w_lean = -G * np.tan(th) / v
  # The image lean also carries the follower's camera roll and viewing angle, a slow bias that would walk the robot
  # round in circles. Turns last a second or two, so only the part slower than about 3 s is taken out.
  w_lean -= gaussian_filter1d(w_lean, 3.0 / dt, mode="nearest")
  gs = G * np.sin(np.radians(slope_deg))
  n = len(th)
  psi, s, y, w = (np.zeros(n) for _ in range(4))
  for i in range(n):
    w[i] = w_lean[i] - gs * np.sin(psi[i - 1] if i else 0.0) / v
    if i:
      psi[i] = psi[i - 1] + w[i] * dt
      s[i] = s[i - 1] + v * np.cos(psi[i - 1]) * dt
      y[i] = y[i - 1] + v * np.sin(psi[i - 1]) * dt
  return psi, s, y, w


def travel_dir(psi, slope_deg=SLOPE_DEG):
  a = np.radians(slope_deg)
  return np.array([np.cos(a) * np.cos(psi), np.sin(psi), -np.sin(a) * np.cos(psi)])


def body_axes(psi, roll):
  """Forward along the slope at heading psi, up tilted `roll` (+ = to the right) from true vertical about it."""
  t = travel_dir(psi)
  up = np.array([0.0, 0.0, 1.0])
  u = unit(up - (up @ t) * t)
  right = np.cross(t, u)
  z = np.cos(roll) * u + np.sin(roll) * right
  return np.column_stack([t, np.cross(z, t), z])


def groom(model, grid, tile_m=6.0):
  """Corduroy and faint grain on the snow, so speed and turning read on screen (plain snow renders flat white)."""
  t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "snowtex")
  w, h, c, adr = model.tex_width[t], model.tex_height[t], model.tex_nchannel[t], model.tex_adr[t]
  rng = np.random.default_rng(0)
  rows = np.arange(h)[:, None] * np.ones((1, w))
  cord = 0.5 + 0.5 * np.sin(2 * np.pi * rows / (h / 40))  # groomer ridges running down the fall line
  grain = gaussian_filter(rng.normal(size=(h, w)), 2.5, mode="wrap")
  blotch = gaussian_filter(rng.normal(size=(h, w)), 45, mode="wrap")
  shade = 1.0 - 0.05 * cord - 0.02 * grain / grain.std() - 0.025 * blotch / blotch.std()
  img = np.clip(np.array([236.0, 242.0, 252.0]) * shade[..., None], 0, 255).astype(np.uint8)
  model.tex_data[adr: adr + w * h * c] = img[..., :c].ravel()
  mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "snow")
  model.mat_texrepeat[mat] = (grid.ncol * grid.cell / tile_m, grid.nrow * grid.cell / tile_m)


def forest(grid, path_xy, seed=0):
  """Tree lines either side of the run and a few glade trees, all kept clear of the robot's line."""
  rng = np.random.default_rng(seed)
  x0, x1 = grid.x0, grid.x0 + (grid.ncol - 1) * grid.cell
  ymid, half = float(np.median(path_xy[:, 1])), float(np.abs(path_xy[:, 1] - np.median(path_xy[:, 1])).max()) + 14
  rows, x = [], x0
  while x < x1:
    for side in (-1, 1):
      rows.append((x, ymid + side * (half + rng.uniform(0, 6)), rng.uniform(0.5, 0.9), rng.uniform(3.5, 7.0)))
    x += rng.uniform(2.5, 5.0)
  for _ in range(int((x1 - x0) / 6)):
    p = (rng.uniform(x0, x1), ymid + rng.uniform(-half, half))
    if np.hypot(path_xy[:, 0] - p[0], path_xy[:, 1] - p[1]).min() > 7.0:
      rows.append((*p, rng.uniform(0.4, 0.8), rng.uniform(2.5, 5.5)))
  return np.array(rows, dtype=np.float32)


def draw_tracks(scene, trail, normal, color=(0.74, 0.79, 0.88, 1.0)):
  """Carved grooves: thin ribbons lying on the snow behind each ski."""
  pts = list(trail)
  for a, b in zip(pts[:-1], pts[1:]):
    if scene.ngeom >= scene.maxgeom - 2:
      return
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-3:
      continue
    x = d / L
    y = unit(np.cross(normal, x))
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([L / 2 + 0.01, 0.018, 0.002]),
                        (a + b) / 2 + 0.003 * normal, np.column_stack([x, y, np.cross(x, y)]).ravel(),
                        np.array(color, np.float32))
    scene.ngeom += 1


class G1Carver:
  """The G1 on skis carving the racer's turns, filmed from behind by a chase cam that lags a little, like the follower."""

  def __init__(self, path_s, path_y, w=GW, h=GH):
    a = np.radians(SLOPE_DEG)
    self.x_start = 30.0
    X = self.x_start + path_s * np.cos(a)
    self.grid = slope(SLOPE_DEG, length=float(X.max()) + 80, width=2 * (float(np.abs(path_y).max()) + 45), cell=2.0)
    self.p = SkiParams()
    self.model = g1_model(self.grid, self.p)
    groom(self.model, self.grid)
    self.trees = forest(self.grid, np.column_stack([X, path_y]))
    self.data = mujoco.MjData(self.model)
    self.r = mujoco.Renderer(self.model, h, w, max_geom=12000)
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    body = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
    self.ski = [body("left_ski"), body("right_ski")]
    self.hip = [body("left_hip_pitch_link"), body("right_hip_pitch_link")]
    self.ankle = [body("left_ankle_roll_link"), body("right_ankle_roll_link")]
    xs = np.linspace(-0.5, 0.5, 5) * self.p.length
    self.pts = np.array([[x, y, -self.p.thickness / 2] for x in xs for y in (-self.p.width / 2, self.p.width / 2)])
    self.tracks = [deque(maxlen=260), deque(maxlen=260)]
    self.cam_psi = None

  def _set(self, qj, R, pos):
    d = self.data
    d.qpos[:3] = pos
    mujoco.mju_mat2Quat(d.qpos[3:7], R.ravel())
    d.qpos[7:] = qj
    mujoco.mj_kinematics(self.model, d)

  def _gaps(self):
    """Each ski's lowest point above the snow, along the slope normal."""
    d = self.data
    out = []
    for b in self.ski:
      p = d.xpos[b] + self.pts @ d.xmat[b].reshape(3, 3).T
      out.append(float(((p - self.p0) @ self.n).min()))
    return out

  def _leg_line_roll(self, qj):
    """Tilt of the feet-to-hips line in the pelvis's own frontal plane (+ = hips to the right of the feet)."""
    self._set(qj, np.eye(3), np.zeros(3))
    d = self.data
    v = (d.xpos[self.hip[0]] + d.xpos[self.hip[1]]) / 2 - (d.xpos[self.ankle[0]] + d.xpos[self.ankle[1]]) / 2
    return float(np.arctan2(-v[1], v[2]))

  @staticmethod
  def _fold(qj, side, f):
    """Shorten one leg: knee bends 2f, hip and ankle f each, so the ski stays parallel to the pelvis."""
    q = qj.copy()
    hp, kn, an = LEG[side]
    q[kn] += 2 * f
    q[hp] -= f
    q[an] -= f
    return q

  @staticmethod
  def _reach(k, u, f_in, e_out):
    """One reach parameter to leg changes, in order: the inside leg (k) folds, the outside leg straightens, then the
    inside hip draws that foot in under the body. Returns the two leg folds and the inside hip's adduction."""
    f = [0.0, 0.0]
    f[k] = min(u, f_in)
    f[1 - k] = -min(max(u - f_in, 0.0), e_out)
    return f, (k, max(u - f_in - e_out, 0.0))

  def _legs(self, qj, folds, R, base, adduct=(0, 0.0)):
    """Fold the legs, turn each hip so its ski points along the direction of travel, and set the pose."""
    q = self._fold(self._fold(qj, "left", folds[0]), "right", folds[1])
    side, a = adduct
    q[HIP_ROLL[("left", "right")[side]]] += -a if side == 0 else a  # toward the other leg
    for _ in range(3):
      self._set(q, R, base)
      for k, side in enumerate(("left", "right")):
        v = R.T @ self.data.xmat[self.ski[k]].reshape(3, 3)[:, 0]
        q[HIP_YAW[side]] -= np.arctan2(v[1], v[0])
    self._set(q, R, base)
    return q

  def pose(self, q: dict, lean_deg: float, torso_deg: float, psi: float, s: float, y: float):
    """Joint targets from the racer, the racer's lean about the direction of travel, both skis on the snow."""
    a = np.radians(SLOPE_DEG)
    X = self.x_start + s * np.cos(a)
    h, n = self.grid.height_normal(np.array([X]), np.array([y]))
    self.p0, self.n = np.array([X, y, h[0]]), n[0]
    qj = DEFAULT_ANGLES.copy()
    for j, v in q.items():
      qj[JOINTS.index(j)] = v
    # Both hips rolling the same way only slides the feet sideways, and the lean (the feet-to-hips line) already holds
    # that. Keep the stance width and let the whole body incline into the turn instead.
    spread = np.clip((qj[HIP_ROLL["left"]] - qj[HIP_ROLL["right"]]) / 2, -0.03, 0.10)
    qj[HIP_ROLL["left"]], qj[HIP_ROLL["right"]] = spread, -spread
    base = self.p0 + 1.2 * self.n
    folds, adduct = [0.0, 0.0], (0, 0.0)
    for _ in range(2):
      line = self._leg_line_roll(self._legs(qj, folds, np.eye(3), np.zeros(3), adduct))
      roll = float(np.clip(np.radians(lean_deg) - np.clip(line, -0.3, 0.3), -1.15, 1.15))
      R = body_axes(psi, roll)
      self._legs(qj, [0.0, 0.0], R, base)
      g = self._gaps()
      k = int(g[1] < g[0])
      # The lower ski touches first: fold that leg until the other ski reaches the snow too. Past a full fold of the
      # inside leg (knee near 150 deg), the outside leg straightens to make up the rest.
      f_in = min(1.1, (2.6 - qj[LEG[("left", "right")[k]][1]]) / 2)
      e_out = max(0.0, (qj[LEG[("left", "right")[1 - k]][1]] - 0.15) / 2)
      lo, hi = 0.0, f_in + e_out + 0.3
      for _ in range(18):
        mid = (lo + hi) / 2
        f, ad = self._reach(k, mid, f_in, e_out)
        self._legs(qj, f, R, base, ad)
        g = self._gaps()
        lo, hi = (mid, hi) if g[k] < g[1 - k] else (lo, mid)
      folds, adduct = self._reach(k, lo, f_in, e_out)
    qf = self._legs(qj, folds, R, base, adduct)
    qf[WAIST_ROLL] = np.clip(np.radians(torso_deg) - roll, -0.5, 0.5)  # angulation: torso more upright than legs
    self._set(qf, R, base - min(self._gaps()) * self.n)
    mujoco.mj_forward(self.model, self.data)
    for k, b in enumerate(self.ski):  # track: under the middle of each ski
      c = self.data.xpos[b]
      self.tracks[k].append(c - ((c - self.p0) @ self.n) * self.n)
    return {"fold_L": folds[0], "fold_R": folds[1], "adduct": adduct[1], "roll": float(np.degrees(roll))}

  def render(self, psi: float, dt: float):
    lag = 0.35  # s, the follower reacts a beat late, so the robot swings across the frame like the racer does
    self.cam_psi = psi if self.cam_psi is None else self.cam_psi + (psi - self.cam_psi) * min(1.0, dt / lag)
    t = travel_dir(self.cam_psi)
    d = self.data
    self.cam.lookat[:] = d.xpos[1] - 0.12 * self.n
    self.cam.distance = 3.6
    self.cam.azimuth = float(np.degrees(np.arctan2(t[1], t[0])))
    self.cam.elevation = float(np.degrees(np.arcsin(t[2]))) - 9.0
    self.r.update_scene(d, self.cam)
    draw_trees(self.r.scene, self.trees, self.grid, d.xpos[1][:2], radius=110, max_trees=500)
    for trail in self.tracks:
      draw_tracks(self.r.scene, list(trail)[:-1], self.n)
    return Image.fromarray(self.r.render())


# --- Panels ----------------------------------------------------------------------------------------------------------

def plot(dr, box, series, colors, labels, lo, hi, title, unit_s, f_lab, f_title):
  x0, y0, x1, y1 = box
  dr.rectangle(box, fill=PANEL, outline=RULE)
  dr.text((x0 + 14, y0 + 10), title, font=f_title, fill=(205, 215, 230))
  top, bot = y0 + 70, y1 - 14
  for frac in (0.0, 0.5, 1.0):
    y = bot - frac * (bot - top)
    dr.line([(x0 + 12, y), (x1 - 12, y)], fill=RULE)
    dr.text((x1 - 60, y - 18), f"{lo + frac * (hi - lo):.0f}{unit_s}", font=f_lab, fill=(120, 135, 160))
  for k, (vals, col, lab) in enumerate(zip(series, colors, labels)):
    n = len(vals)
    xy = [(x0 + 12 + (x1 - x0 - 84) * i / max(n - 1, 1), bot - (np.clip(v, lo, hi) - lo) / (hi - lo) * (bot - top))
          for i, v in enumerate(vals) if v is not None]
    if len(xy) > 1:
      dr.line(xy, fill=col, width=4, joint="curve")
    last = next((v for v in reversed(vals) if v is not None), None)
    dr.text((x0 + 14 + 190 * k, y0 + 40), f"{lab}  {'' if last is None else f'{last:.0f}{unit_s}'}", font=f_lab, fill=col)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("video")
  ap.add_argument("--title", default="FOLLOW CAM · MEDIAPIPE 3D POSE")
  ap.add_argument("--start", type=float, default=0.0)
  ap.add_argument("--seconds", type=float, default=0.0)
  args = ap.parse_args()
  video = Path(args.video)
  out_dir = ROOT / "out" / "pose"
  out_dir.mkdir(parents=True, exist_ok=True)
  det = detect(video, args.start, args.seconds)
  t, dt, ok, s = signals(det)
  n = len(t)
  psi, path_s, path_y, omega = carve(s["lean"], dt)
  print(f"{n} frames, skier measured in {int(ok.sum())}, path {path_s[-1]:.0f} m, "
        f"heading {np.degrees(psi.min()):.0f} to {np.degrees(psi.max()):.0f} deg", flush=True)

  frames_dir = out_dir / f"_frames_{video.stem}"
  if frames_dir.exists():
    shutil.rmtree(frames_dir)
  frames_dir.mkdir(parents=True)
  carver = G1Carver(path_s, path_y)
  f_head, f_big, f_mid, f_lab, f_title = font(26, True), font(44, True), font(30, True), font(22), font(24, True)
  n_hist = int(PLOT_S / dt)
  hist = {k: deque([None] * n_hist, maxlen=n_hist) for k in ("kL", "kR", "lean", "gyro")}
  cap = cv2.VideoCapture(str(video))
  fps, step = float(det["fps"]), int(det["step"])
  if args.start:
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
  recording = []
  for i in range(n):
    good, img = cap.read()
    for _ in range(step - 1):
      cap.grab()
    if not good:
      break
    a = {k: float(v[i]) for k, v in s.items()}
    q = to_g1(a)
    ang = a["feet_right_of_hips"] * 100
    phase = "LEFT TURN" if a["lean"] < -12 else "RIGHT TURN" if a["lean"] > 12 else "EDGE CHANGE"  # what steers the G1
    vals = {"kL": np.degrees(a["knee_L"]), "kR": np.degrees(a["knee_R"]), "lean": a["lean"], "ang": ang, "gyro": a["gyro"]}
    carver.pose(q, a["lean"], a["torso"], psi[i], path_s[i], path_y[i])
    g1_img = carver.render(psi[i], dt)
    recording.append({"t": round(args.start + t[i], 3), "measured": bool(ok[i]),
                      **{k: round(float(v), 2) for k, v in vals.items()}, "torso": round(a["torso"], 2), "phase": phase,
                      "robot_heading_deg": round(float(np.degrees(psi[i])), 2),
                      "robot_turn_rate_dps": round(float(np.degrees(omega[i])), 2),
                      "g1": {k: round(v, 3) for k, v in q.items()} | {"waist_roll_joint": round(float(carver.data.qpos[7 + WAIST_ROLL]), 3)}})
    for k in hist:
      hist[k].append(vals[k])

    frame = Image.fromarray(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (VW, VH)))
    if ok[i]:  # the skeleton only where MediaPipe actually found the racer on this frame
      dr_f = ImageDraw.Draw(frame)
      pts = {j: (det["uv"][i][j][0] * VW, det["uv"][i][j][1] * VH) for j in range(33)}
      for a_, b_ in EDGES:
        col = LEFT_COL if a_ in LEFT_SET and b_ in LEFT_SET else RIGHT_COL if a_ in RIGHT_SET and b_ in RIGHT_SET else (235, 235, 235)
        dr_f.line([pts[a_], pts[b_]], fill=col, width=3)
      for j in KEY:
        x, y = pts[j]
        dr_f.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255))

    canvas = Image.new("RGB", (CW, CH), BG)
    canvas.paste(frame, (0, TOP))
    canvas.paste(g1_img, (VW, TOP))
    dr = ImageDraw.Draw(canvas)
    dr.text((20, 13), args.title, font=f_head, fill=(255, 255, 255))
    dr.text((VW + 20, 13), "UNITREE G1, SAME POSE, SAME CARVE", font=f_head, fill=(255, 255, 255))
    dr.line([(VW, TOP), (VW, TOP + VH)], fill=RULE, width=2)
    by = TOP + VH + 14
    dr.rectangle([12, by, 560, CH - 14], fill=PANEL, outline=RULE)
    dr.text((30, by + 14), phase, font=f_big, fill=ACC if ok[i] else (150, 165, 190))
    rows = [("Knees  L / R", f"{vals['kL']:.0f}° / {vals['kR']:.0f}°"), ("Lean into turn", f"{abs(vals['lean']):.0f}°"),
            ("Feet out from hips", f"{abs(vals['ang']):.0f}% of leg"), ("Torso gyro (virtual IMU)", f"{vals['gyro']:+.0f} °/s")]
    for r_, (lab, val) in enumerate(rows):
      y = by + 84 + r_ * 66
      dr.text((30, y), lab, font=f_lab, fill=(150, 165, 190))
      dr.text((30, y + 24), val, font=f_mid, fill=(255, 255, 255))
    pw = (CW - 560 - 12 * 3 - 12) // 2
    x1 = 560 + 12
    plot(dr, (x1, by, x1 + pw, CH - 14), [list(hist["kL"]), list(hist["kR"])], [LEFT_COL, RIGHT_COL],
         ["left knee", "right knee"], 0, 120, "Knee flexion, last 6 s", "°", f_lab, f_title)
    x2 = x1 + pw + 12
    plot(dr, (x2, by, x2 + pw, CH - 14), [list(hist["lean"]), list(hist["gyro"])], [(180, 255, 160), (255, 225, 110)],
         ["lean °", "torso gyro °/s"], -120, 120, "Lean and virtual IMU, last 6 s", "", f_lab, f_title)
    canvas.save(frames_dir / f"{i:05d}.jpg", quality=88)
    if i % 100 == 0:
      print(f"  render {i}/{n}", flush=True)
  cap.release()
  mp4 = out_dir / f"pose_data_{video.stem}.mp4"
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", f"{fps / step:.3f}", "-i", str(frames_dir / "%05d.jpg"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(mp4)], check=True)
  (out_dir / f"pose_data_{video.stem}.json").write_text(json.dumps(recording))
  print(f"{mp4}  frames {len(recording)}, skier measured in {int(ok.sum())}", flush=True)


if __name__ == "__main__":
  main()
