"""Climbing from video: the Unitree G1 on a rock wall, its body learned from climbing footage.

Pieces:
- `wall_scene`: a MuJoCo wall of any size with holds anywhere, sandstone texture, the G1 facing it.
- `Retarget`: damped least squares IK on the G1 that puts its pelvis, shoulders, hips, elbows, knees, hands and feet
  where the climber's are on the wall plane (image coordinates scaled to the G1), hands and feet touching the wall.
- `track`: per-frame climber signals from a cached detection (out/climb/_detect_<video>.npz), camera pan removed, so
  limbs are in wall coordinates; `holds` finds where each hand and foot stops and puts a hold there.
- `ClimbPrior` (trained in `train_prior`): an MLP from where the four limbs are relative to the pelvis to all 29 G1
  joints, trained on the IK of every frame of the training videos. At test time it gets only the holds and the pelvis
  of a climber it has never seen and supplies the rest of the body (elbows, knees, hip turn, torso).

Coordinates: the wall face is the plane x = 0, the G1 faces +x into it, +y is the climber's left, +z is up.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "climb"
G1_XML = ROOT / "assets" / "menagerie_unitree_g1" / "g1.xml"
G1_HANDS_XML = ROOT / "assets" / "menagerie_unitree_g1" / "g1_with_hands.xml"

# MediaPipe landmark ids
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_FT, R_FT = 23, 24, 25, 26, 27, 28, 31, 32
CORE = [L_SH, R_SH, L_HIP, R_HIP]
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
LIMB_LM = {"left_hand": L_WR, "right_hand": R_WR, "left_foot": L_AN, "right_foot": R_AN}
MID_LM = {"left_hand": L_EL, "right_hand": R_EL, "left_foot": L_KN, "right_foot": R_KN}

PELVIS_X = -0.30  # pelvis this far off the wall face
HOLD_DEPTH = 0.05  # holds stick out of the wall this far
CONTACT_X = {"left_hand": -0.07, "right_hand": -0.07, "left_foot": -0.10, "right_foot": -0.10}


def unit(v):
  return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


# --- Scene -----------------------------------------------------------------------------------------------------------

def sandstone(path: Path, n=1024, seed=3):
  """Tileable sandstone: warm base, broad blotches, fine grain, a few darker cracks."""
  if path.exists():
    return
  rng = np.random.default_rng(seed)
  big = gaussian_filter(rng.normal(size=(n, n)), 60, mode="wrap")
  mid = gaussian_filter(rng.normal(size=(n, n)), 12, mode="wrap")
  fine = gaussian_filter(rng.normal(size=(n, n)), 1.2, mode="wrap")
  crack = np.abs(gaussian_filter(rng.normal(size=(n, n)), 20, mode="wrap"))
  crack = np.clip(1 - crack / (0.08 * crack.std() + 1e-9), 0, 1) ** 3
  shade = 1 + 0.10 * big / big.std() + 0.06 * mid / mid.std() + 0.05 * fine / fine.std() - 0.25 * crack
  img = np.clip(np.array([214.0, 192.0, 158.0]) * shade[..., None], 0, 255).astype(np.uint8)
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(img).save(path)


def wall_scene(holds: np.ndarray, width=8.0, height=16.0, extra=220, seed=0, kinds=None, xml=G1_XML) -> mujoco.MjModel:
  """The G1 in front of a wall `width` x `height` m. `holds` (N x 2: y, z) are the route; `extra` more are scattered
  over the rest of the wall so it reads as a real wall. Any wall: change the size, the holds or the seed."""
  tex = OUT / "sandstone.png"
  sandstone(tex)
  spec = mujoco.MjSpec.from_file(str(xml))
  spec.option.gravity = [0, 0, -9.81]
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1920
  spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
  spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
  spec.visual.quality.shadowsize = 4096
  t = spec.add_texture(name="sandstone", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex))
  m = spec.add_material(name="wall", texrepeat=[width / 2.0, height / 2.0], specular=0.05, shininess=0.1)
  m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "sandstone"
  sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.55, 0.72, 0.92], rgb2=[0.92, 0.95, 1.0], width=512, height=512)
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.30, 0.34, 0.30], rgb2=[0.27, 0.31, 0.27], width=256, height=256)
  g = spec.add_material(name="ground", texrepeat=[20, 20])
  g.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  w = spec.worldbody
  w.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[-6, 4, 20], dir=[0.55, -0.35, -0.75],
              diffuse=[0.75, 0.72, 0.66], specular=[0.2, 0.2, 0.2], castshadow=True)
  w.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], pos=[0, 0, 0], material="ground")
  w.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, width / 2, height / 2], pos=[0.25, 0, height / 2],
             material="wall", contype=0, conaffinity=0)
  rng = np.random.default_rng(seed)
  colors = [[0.30, 0.30, 0.32, 1], [0.38, 0.33, 0.30, 1], [0.25, 0.27, 0.30, 1]]
  pts = [tuple(h) for h in holds]
  tries = 0
  while len(pts) < len(holds) + extra and tries < 20 * extra:
    tries += 1
    y, z = rng.uniform(-width / 2 + 0.2, width / 2 - 0.2), rng.uniform(0.3, height - 0.2)
    if all((y - a) ** 2 + (z - b) ** 2 > 0.30 ** 2 for a, b in pts):
      pts.append((y, z))
  from skisim.climb_grip import hold_geoms
  for i, (y, z) in enumerate(pts):
    route = i < len(holds)
    if route and kinds is not None:  # real hold shapes the hands grip; (y, z) is the grip edge
      b = w.add_body(name=f"route{i}", pos=[0, y, z])
      for k, gd in enumerate(hold_geoms(kinds[i], rng)[0]):
        b.add_geom(name=f"route{i}_{k}", contype=0, conaffinity=0, **gd)
      continue
    s = rng.uniform(0.025, 0.045) if not route else 0.04
    w.add_geom(name=f"hold{i}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
               size=[0.6 * HOLD_DEPTH, s * rng.uniform(0.9, 1.5), s], pos=[-0.01, y, z], quat=_rand_quat_x(rng),
               rgba=colors[rng.integers(3)] if not route else [0.20, 0.22, 0.26, 1], contype=0, conaffinity=0)
  for gm in spec.geoms:  # the robot is posed kinematically; hand meshes keep contype so grip can measure distance
    if gm.contype and not (gm.name or "").startswith("route"):
      gm.conaffinity = 0
  return spec.compile()


def _rand_quat_x(rng):
  a = rng.uniform(0, np.pi)
  return [np.cos(a / 2), np.sin(a / 2), 0, 0]


# --- Retargeting -----------------------------------------------------------------------------------------------------

class Retarget:
  """Damped least squares IK on the G1. Targets are world points, each with per-axis weights."""

  BODIES = {"pelvis": "pelvis", "left_shoulder": "left_shoulder_roll_link", "right_shoulder": "right_shoulder_roll_link",
            "left_hip": "left_hip_roll_link", "right_hip": "right_hip_roll_link",
            "left_elbow": "left_elbow_link", "right_elbow": "right_elbow_link",
            "left_knee": "left_knee_link", "right_knee": "right_knee_link",
            "left_hand": "left_wrist_yaw_link", "right_hand": "right_wrist_yaw_link",
            "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"}
  # Points fixed in a body: finger base (fingers run along the wrist's +x) and palm (fingers curl toward the palm,
  # -y for the left hand, +y for the right), used to put the hand on a hold palm to the wall, fingers up.
  POINTS = {"left_fingers": ("left_wrist_yaw_link", (0.12, 0.0, 0.0)), "right_fingers": ("right_wrist_yaw_link", (0.12, 0.0, 0.0)),
            "left_palm": ("left_wrist_yaw_link", (0.03, -0.06, 0.0)), "right_palm": ("right_wrist_yaw_link", (0.03, 0.06, 0.0))}

  def __init__(self, model: mujoco.MjModel | None = None):
    self.m = model or mujoco.MjModel.from_xml_path(str(G1_XML))
    self.d = mujoco.MjData(self.m)
    self.bid = {k: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in self.BODIES.items()}
    self.off = {k: np.zeros(3) for k in self.bid}
    for k, (b, o) in self.POINTS.items():
      self.bid[k] = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
      self.off[k] = np.array(o)
    self.nv = self.m.nv
    self.lo = self.m.jnt_range[1:, 0].copy()
    self.hi = self.m.jnt_range[1:, 1].copy()
    q = np.zeros(self.m.nq)
    q[2], q[3] = 1.0, 1.0
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    self.size = self.dims()
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    # A climber's rest shape: knees bent and turned out, arms up, used as the weak pull when nothing else decides.
    self.q_nominal = self.joints({"left_hip_pitch_joint": -0.6, "right_hip_pitch_joint": -0.6, "left_knee_joint": 1.1,
                                  "right_knee_joint": 1.1, "left_hip_roll_joint": 0.35, "right_hip_roll_joint": -0.35,
                                  "left_hip_yaw_joint": 0.3, "right_hip_yaw_joint": -0.3,
                                  "left_shoulder_pitch_joint": -1.8, "right_shoulder_pitch_joint": -1.8,
                                  "left_elbow_joint": 0.6, "right_elbow_joint": 0.6})
    self.q = self.home()

  def joints(self, named: dict) -> np.ndarray:
    q = np.zeros(self.m.nq - 7)
    for k, v in named.items():
      q[self.m.joint(k).qposadr[0] - 7] = v
    return q

  def dims(self):
    p = lambda k: self.d.xpos[self.bid[k]]
    sh = (p("left_shoulder") + p("right_shoulder")) / 2
    hp = (p("left_hip") + p("right_hip")) / 2
    return {"torso": float(np.linalg.norm(sh - hp)), "shoulders": float(np.linalg.norm(p("left_shoulder") - p("right_shoulder"))),
            "hips": float(np.linalg.norm(p("left_hip") - p("right_hip"))), "arm": self._reach("left_elbow_joint", "left_shoulder", "left_hand"),
            "leg": self._reach("left_knee_joint", "left_hip", "left_foot")}

  def _reach(self, joint, root, end):
    """Longest root-to-end distance over the joint's range (the G1's elbow is bent 90 deg at zero)."""
    a = self.m.joint(joint).qposadr[0]
    best = 0.0
    for v in np.linspace(*self.m.joint(joint).range, 80):
      q = self.d.qpos.copy()
      q[7:] = 0
      q[a] = v
      P = self.points(q)
      best = max(best, float(np.linalg.norm(P[end] - P[root])))
    return best

  def home(self, pelvis=(PELVIS_X, 0.0, 1.0)):
    q = np.zeros(self.m.nq)
    q[:3] = pelvis
    q[3] = 1.0
    q[7:] = self.q_nominal
    return q

  def solve(self, targets: dict, q0: np.ndarray | None = None, q_ref: np.ndarray | None = None, iters=25,
            reg=0.02, damping=0.03) -> np.ndarray:
    """targets: name -> (point(3), weights(3)). Pulls weakly toward q_ref joints (default: the climber's rest shape)."""
    m, d = self.m, self.d
    q = (self.q if q0 is None else q0).copy()
    ref = self.q_nominal if q_ref is None else q_ref
    jac = np.zeros((3, self.nv))
    names = list(targets)
    for _ in range(iters):
      d.qpos[:] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      rows, errs = [], []
      for k in names:
        p, wgt = targets[k]
        b = self.bid[k]
        x = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.off[k]
        mujoco.mj_jac(m, d, jac, None, x, b)
        e = (np.asarray(p) - x) * wgt
        rows.append(jac * np.asarray(wgt)[:, None])
        errs.append(e)
      J = np.vstack(rows + [np.hstack([np.zeros((29, 6)), np.sqrt(reg) * np.eye(29)])])
      e = np.concatenate(errs + [np.sqrt(reg) * (ref - q[7:])])
      dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
      mujoco.mj_integratePos(m, q, dq, 1.0)
      q[7:] = np.clip(q[7:], self.lo, self.hi)
    self.q = q
    return q

  def points(self, q):
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    return {k: self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.off[k] for k, b in self.bid.items()}


# --- Video to wall coordinates -----------------------------------------------------------------------------------------

def load(stem: str) -> dict:
  z = np.load(OUT / f"_detect_{stem}.npz")
  return {k: z[k] for k in z.files}


def fill(t, y, ok, sigma=1.0):
  """Gaps filled with a monotone spline, then a light zero-phase smooth."""
  idx = np.flatnonzero(ok & np.isfinite(y))
  if len(idx) < 4:
    return np.full_like(t, np.nanmedian(y) if np.isfinite(y).any() else 0.0)
  f = PchipInterpolator(t[idx], y[idx], extrapolate=False)
  out = f(t)
  out[t < t[idx[0]]], out[t > t[idx[-1]]] = y[idx[0]], y[idx[-1]]
  return gaussian_filter1d(out, sigma, mode="nearest") if sigma else out


def frames_ok(det, thresh=0.5):
  vis = det["vis"]
  return (vis[:, CORE].min(1) > thresh) & np.isfinite(det["uv"][:, 0, 0])


def wall_points(det: dict, g1_torso: float, pan=True, sigma=1.0, limbs: dict | None = None):
  """Every landmark on the wall plane in G1 meters, per frame: (n, 33, 2) as (y, z). Image right is the climber's right
  (-y, the camera is behind them), image down is -z. Camera pan is undone with the accumulated frame shift, and the
  scale is the G1's torso over the climber's (median over 3 s, so a zoom or tilt follows slowly)."""
  W, H = det["size"]
  px = det["uv"] * np.array([W, H])
  n = len(px)
  ok = frames_ok(det)
  if pan:
    sh = det["shift"][:, :2] * np.array([W, H])
    sh[det["shift"][:, 2] < 0.05] = 0.0  # low confidence: probably a cut or blur, call it no move
    sh = np.clip(sh, -0.08 * H, 0.08 * H)
    cam = np.cumsum(sh, 0)  # accumulated content shift in pixels
    px = px - cam[:, None, :]  # back to where it was on the first frame
  mid_sh = (px[:, L_SH] + px[:, R_SH]) / 2
  mid_hp = (px[:, L_HIP] + px[:, R_HIP]) / 2
  torso_px = np.linalg.norm(mid_sh - mid_hp, axis=1)
  torso_px[~ok] = np.nan
  dt = float(det["step"] / det["fps"])
  win = max(3, int(3.0 / dt))
  med = np.array([np.nanmedian(torso_px[max(0, i - win // 2): i + win // 2 + 1]) if np.isfinite(torso_px[max(0, i - win // 2): i + win // 2 + 1]).any() else np.nan for i in range(n)])
  med = fill(np.arange(n, dtype=float), med, np.isfinite(med), 0)
  s = g1_torso / med
  yz = np.stack([-px[..., 0], -px[..., 1]], -1) * s[:, None, None]
  if limbs is not None:
    yz = limb_scale(yz, ok, limbs)
  t = np.arange(n) * dt
  if sigma:
    for j in range(33):
      for a in range(2):
        yz[:, j, a] = fill(t, yz[:, j, a], ok, sigma)
  return t, dt, ok, yz


def limb_scale(yz, ok, g1: dict):
  """People have long arms for their torso, the G1 does not: scaled by torso alone, a reach is out of the robot's range.
  Each arm is rescaled about its shoulder and each leg about its hip so the climber's longest reach in the clip maps
  to the G1's full arm or leg, keeping every direction and the elbow and knee along the same chain."""
  yz = yz.copy()
  for root, mid, end, length in ((L_SH, L_EL, L_WR, g1["arm"]), (R_SH, R_EL, R_WR, g1["arm"]),
                                 (L_HIP, L_KN, L_AN, g1["leg"]), (R_HIP, R_KN, R_AN, g1["leg"])):
    reach = np.linalg.norm(yz[:, end] - yz[:, root], axis=1)
    k = 0.97 * length / np.nanpercentile(reach[ok], 97)
    for j in (mid, end):
      yz[:, j] = yz[:, root] + (yz[:, j] - yz[:, root]) * k
  return yz


def relative_features(yz: np.ndarray) -> np.ndarray:
  """Network input: the four limbs relative to the pelvis plus the shoulder line, in G1 meters (n, 12)."""
  pelvis = (yz[:, L_HIP] + yz[:, R_HIP]) / 2
  f = [yz[:, LIMB_LM[k]] - pelvis for k in LIMBS]
  f += [yz[:, L_SH] - pelvis, yz[:, R_SH] - pelvis]
  return np.concatenate(f, 1)


def targets_from(yz_i: np.ndarray, contact_x=None, full=True, hold=None):
  """IK targets for one frame from wall-plane landmarks. full: also elbows and knees (the training labels); otherwise
  only what a robot would know on a wall: pelvis, shoulders, hips and where its hands and feet are."""
  P = lambda j, x, w: (np.array([x, yz_i[j][0], yz_i[j][1]]), np.array(w))
  pel = (yz_i[L_HIP] + yz_i[R_HIP]) / 2
  T = {"pelvis": (np.array([PELVIS_X, pel[0], pel[1]]), np.array([1.0, 1.0, 1.0])),
       "left_shoulder": P(L_SH, PELVIS_X + 0.05, [0.3, 1, 1]), "right_shoulder": P(R_SH, PELVIS_X + 0.05, [0.3, 1, 1]),
       "left_hip": P(L_HIP, PELVIS_X, [0.3, 0.6, 0.6]), "right_hip": P(R_HIP, PELVIS_X, [0.3, 0.6, 0.6])}
  for k in LIMBS:
    x = CONTACT_X[k] if contact_x is None else contact_x[k]
    pt = yz_i[LIMB_LM[k]] if hold is None or hold.get(k) is None else hold[k]
    T[k] = (np.array([x, pt[0], pt[1]]), np.array([1.5, 2.0, 2.0]))
  if full:
    for k, j in MID_LM.items():
      T[k.replace("hand", "elbow").replace("foot", "knee")] = P(j, 0.0, [0.0, 0.8, 0.8])
  return T


# --- Holds -----------------------------------------------------------------------------------------------------------

def holds(t, yz, ok, speed=0.25, min_s=0.25, merge=0.12):
  """Where each hand and foot stops: a contact is a stretch slower than `speed` m/s for at least `min_s`, its hold the
  median point. Holds closer than `merge` are one hold (a match, or a foot onto a hand hold)."""
  dt = t[1] - t[0]
  contact, hold_xy = {}, []
  per = {}
  for k in LIMBS:
    p = yz[:, LIMB_LM[k]]
    v = np.linalg.norm(np.gradient(gaussian_filter1d(p, 2, axis=0), dt, axis=0), axis=1)
    still = (v < speed) & ok
    still = median_filter(still.astype(np.uint8), size=5).astype(bool)
    segs, i, n = [], 0, len(t)
    while i < n:
      if still[i]:
        j = i
        while j < n and still[j]:
          j += 1
        if (j - i) * dt >= min_s:
          segs.append((i, j))
        i = j
      else:
        i += 1
    per[k] = segs
  ids = {k: np.full(len(t), -1) for k in LIMBS}
  for k in LIMBS:
    for i, j in per[k]:
      c = np.median(yz[i:j, LIMB_LM[k]], 0)
      near = [h for h, q in enumerate(hold_xy) if np.linalg.norm(q - c) < merge]
      if near:
        h = near[0]
      else:
        h = len(hold_xy)
        hold_xy.append(c)
      ids[k][i:j] = h
  return np.array(hold_xy).reshape(-1, 2), ids


# --- Training the climbing prior ---------------------------------------------------------------------------------------

TRAIN = ["66FnK6L6roM", "71-tvRBQQ1E", "E2wlGixTz70", "EeC7HWZG0Zw", "KNyiPlbx_vg", "KxhPIrlUJfs", "LoILfnviiBM",
         "Mo-3eGMg9vI", "R_v7YZQrkFo", "gbDjEO-CqZs", "nWqOeRYTJhk"]


def label_video(stem: str, iters=12):
  """Features and IK joint labels for every frame where the climber's torso is clear. Each run of consecutive frames
  warm starts from the last, a new run starts from the rest shape."""
  det = load(stem)
  r = Retarget()
  t, dt, ok, yz = wall_points(det, r.size["torso"], pan=False, sigma=0.7, limbs=r.size)
  X = relative_features(yz)
  Y, keep, err = [], [], []
  q, prev = None, -2
  for i in np.flatnonzero(ok):
    pel = (yz[i, L_HIP] + yz[i, R_HIP]) / 2
    if i != prev + 1:
      q = r.home((PELVIS_X, pel[0], pel[1]))
      it = 40
    else:
      it = iters
    T = targets_from(yz[i])
    q = r.solve(T, q, iters=it)
    pts = r.points(q)
    e = np.mean([np.linalg.norm(pts[k][1:] - T[k][0][1:]) for k in LIMBS])
    Y.append(q[7:].copy())
    keep.append(i)
    err.append(e)
    prev = i
  return stem, X[keep], np.array(Y), np.array(err)


class ClimbPrior:
  """MLP: 12 limb/shoulder features -> 29 G1 joints. Weights in out/climb/prior.npz (numpy at inference)."""

  def __init__(self, path=OUT / "prior.npz"):
    z = np.load(path)
    self.w = [z[f"w{i}"] for i in range(3)]
    self.b = [z[f"b{i}"] for i in range(3)]
    self.mu, self.sd = z["mu"], z["sd"]

  def __call__(self, x):
    h = (np.atleast_2d(x) - self.mu) / self.sd
    for i in range(3):
      h = h @ self.w[i] + self.b[i]
      if i < 2:
        h = np.maximum(h, 0) + 0.01 * np.minimum(h, 0)
    return h


def train_prior(epochs=400, seed=0):
  import torch
  from concurrent.futures import ProcessPoolExecutor

  with ProcessPoolExecutor(6) as ex:
    data = list(ex.map(label_video, TRAIN + ["IMG_5878"]))
  for stem, X, Y, e in data:
    print(f"  {stem}: {len(X)} frames, IK limb error {100 * np.median(e):.1f} cm", flush=True)
  test = [d for d in data if d[0] == "IMG_5878"][0]
  tr = [d for d in data if d[0] != "IMG_5878"]
  X = np.concatenate([d[1] for d in tr])
  Y = np.concatenate([d[2] for d in tr])
  # Mirror every frame (left <-> right) to double the data and keep the prior symmetric.
  Xm, Ym = mirror_features(X), mirror_joints(Y)
  X, Y = np.concatenate([X, Xm]), np.concatenate([Y, Ym])
  mu, sd = X.mean(0), X.std(0) + 1e-6
  torch.manual_seed(seed)
  net = torch.nn.Sequential(torch.nn.Linear(12, 256), torch.nn.LeakyReLU(0.01), torch.nn.Linear(256, 256),
                            torch.nn.LeakyReLU(0.01), torch.nn.Linear(256, 29))
  opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
  Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
  Yt = torch.tensor(Y, dtype=torch.float32)
  Xv = torch.tensor((test[1] - mu) / sd, dtype=torch.float32)
  Yv = torch.tensor(test[2], dtype=torch.float32)
  for ep in range(epochs):
    perm = torch.randperm(len(Xt))
    for k in range(0, len(Xt), 256):
      b = perm[k:k + 256]
      noise = 0.03 * torch.randn_like(Xt[b])
      loss = torch.nn.functional.smooth_l1_loss(net(Xt[b] + noise), Yt[b], beta=0.1)
      opt.zero_grad()
      loss.backward()
      opt.step()
    sched.step()
    if ep % 50 == 0 or ep == epochs - 1:
      with torch.no_grad():
        v = (net(Xv) - Yv).abs().mean().item()
      print(f"  epoch {ep}: train {loss.item():.4f}, held-out joint error {np.degrees(v):.1f} deg", flush=True)
  lin = [m for m in net if isinstance(m, torch.nn.Linear)]
  np.savez(OUT / "prior.npz", mu=mu, sd=sd, **{f"w{i}": l.weight.detach().numpy().T for i, l in enumerate(lin)},
           **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)})
  r = Retarget()
  base = np.degrees(np.abs(test[2] - r.q_nominal).mean())
  pred = ClimbPrior()(test[1])
  print(f"held out (IMG_5878, {len(test[1])} frames): mean joint error {np.degrees(np.abs(pred - test[2]).mean()):.1f} deg "
        f"vs {base:.1f} deg for the rest shape; train frames {len(X) // 2} (+ mirrored)", flush=True)
  return test


_SWAP = None


def _swap(r=None):
  """Joint index pairs left<->right and which joints flip sign under a left/right mirror."""
  global _SWAP
  if _SWAP is None:
    m = (r or Retarget()).m
    names = [m.joint(i).name for i in range(1, m.njnt)]
    perm = [names.index(n.replace("left", "TMP").replace("right", "left").replace("TMP", "right")) for n in names]
    sign = np.array([-1.0 if any(s in n for s in ("roll", "yaw")) else 1.0 for n in names])
    _SWAP = (np.array(perm), sign)
  return _SWAP


def mirror_joints(Y):
  perm, sign = _swap()
  return Y[:, perm] * sign


def mirror_features(X):
  X = X.reshape(len(X), 6, 2).copy()
  X = X[:, [1, 0, 3, 2, 5, 4]]
  X[..., 0] *= -1
  return X.reshape(len(X), 12)


if __name__ == "__main__":
  train_prior()
