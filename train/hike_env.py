"""Batched hiking environment: Unitree G1 walking real Ann Arbor trail terrain in MuJoCo Warp, with foot contact on
the 1 m lidar, frozen SONIC underneath tracking walking clips from its kinematic planner, residual policy on top.

Per policy step (50 Hz):
  policy action (31) = body lean for SONIC's reference (pitch, roll) + residual on SONIC's 29 raw joint actions
  SONIC encodes 10 future frames of a planner walking clip, pointed along the reference heading and leaned by the
  policy; its decoder turns that token plus 10 frames of proprioception into joint targets; physics substeps of 2 ms (5 ms blew up 7 of 12 robots in 8 s on this terrain).
Task: walk up or down a real 40 m trail stretch (hikesim.tiles) at the clip's speed, following the trail line,
without falling. Tiles are sorted by difficulty and each robot climbs a curriculum.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from tensordict import TensorDict

from skisim.scene import g1_model
from skisim.ski_torch import TorchGrid
from skisim.sonic import DEFAULT_ANGLES, I2M, JOINTS, M2I, sonic_gains
from skisim.sonic_torch import Decoder, G1Encoder
from skisim.terrain import HeightGrid

ROOT = Path(__file__).resolve().parents[1]

FALL_BODIES = ("torso_link", "left_knee_link", "right_knee_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
SCAN_X = (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0)
SCAN_Y = (-0.3, 0.0, 0.3)
LEGS_MJ = list(range(12))


def quat_to_mat(q):
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ], -1).reshape(*q.shape[:-1], 3, 3)


def rot_zyx(yaw, pitch, roll):
  """R = Rz(yaw) Ry(pitch) Rx(roll), batched."""
  cy, sy, cp, sp, cr, sr = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)
  return torch.stack([
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr,
  ], -1).reshape(*yaw.shape, 3, 3)


def wrap(a):
  return torch.atan2(torch.sin(a), torch.cos(a))


# Sole contact points (foot frame, same sole height as the stock capsules): toe corners, heel corners, toe, heel, mid.
SOLE = ((0.12, -0.025), (0.12, 0.025), (-0.045, -0.018), (-0.045, 0.018), (0.13, 0.0), (-0.054, 0.0), (0.04, 0.0))


BOX_HALF = (0.35, 0.35, 0.05)
GROUND_BOXES = [
  f'<body name="ground_{side}" mocap="true" pos="0 0 -10"><geom name="ground_{side}" type="box" size="{BOX_HALF[0]} '
  f'{BOX_HALF[1]} {BOX_HALF[2]}" contype="0" conaffinity="{bit}" condim="3" friction="0.6" rgba="0 0 0 0" group="3"/></body>'
  for side, bit in (("left", 2), ("right", 4))
]


def sphere_feet(mjm: mujoco.MjModel) -> None:
  """Turn each foot's 7 capsules into 7 spheres at the sole points. MuJoCo Warp sends capsule-heightfield pairs through
  general convex collision (one contact, unstable on steep cells); spheres take its sphere-heightfield path."""
  for side in ("left", "right"):
    for k in range(7):
      g = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{k + 1}_collision")
      r = float(mjm.geom_size[g][0])
      mjm.geom_type[g] = mujoco.mjtGeom.mjGEOM_SPHERE
      mjm.geom_size[g] = (r, 0, 0)
      mjm.geom_pos[g] = (SOLE[k][0], SOLE[k][1], -0.025)
      mjm.geom_quat[g] = (1, 0, 0, 0)
      mjm.geom_rbound[g] = r
      mjm.geom_aabb[g] = (0, 0, 0, r, r, r)


class HikeEnv:
  def __init__(self, num_envs=4096, device="cuda", mosaic=ROOT / "out/hike/tiles", clips=ROOT / "out/hike/clips.npz",
               episode_s=20.0, seed=0, use_graph=True, dt=0.002, feet="spheres", iterations=10, ls_iterations=20, start_level=40):
    self.num_envs, self.device = num_envs, torch.device(device)
    self.dt, self.decimation = dt, int(round(0.02 / dt))
    self.max_episode_length = int(episode_s / (self.dt * self.decimation))
    self.cfg = {"episode_s": episode_s}
    self.gen = torch.Generator(device=self.device).manual_seed(seed)
    self.privileged_critic = True
    self.rew_ema = {}
    self.auto_reset = True
    self.explore_frac = 0.2

    grid = self._load_terrain(mosaic)

    # MuJoCo model: G1 with plain feet on the ground, SONIC gains, mjlab's solver settings for the G1.
    mjm = g1_model(grid, skis=False, timestep=self.dt, extras=GROUND_BOXES)
    if feet == "spheres":
      sphere_feet(mjm)
    # Ground contact: MuJoCo Warp's hfield collision reports bogus penetrations of 0.1 to 22 m on this lidar terrain
    # (3.11 and 3.14 alike) and launches robots. So the hfield is visual only, and each foot stands on its own mocap box
    # that follows the lidar surface (height and slope) under it every substep. Only the soles collide, like mjlab's
    # FEET_ONLY_COLLISION for the G1; falls are caught by body height above the terrain.
    for g in range(mjm.ngeom):
      name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      if name == "terrain" or (name.endswith("_collision") and "_foot" not in name):
        mjm.geom_contype[g], mjm.geom_conaffinity[g] = 0, 0
      elif name.startswith("left_foot") and name.endswith("_collision"):
        mjm.geom_contype[g], mjm.geom_conaffinity[g] = 2, 0
      elif name.startswith("right_foot") and name.endswith("_collision"):
        mjm.geom_contype[g], mjm.geom_conaffinity[g] = 4, 0
    mjm.opt.iterations, mjm.opt.ls_iterations = iterations, ls_iterations
    names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, mjm.actuator_trnid[a, 0]) for a in range(mjm.nu)]
    assert names == JOINTS, "actuator order must match SONIC's MuJoCo joint order"
    kp, kd, effort, armature, scale = sonic_gains()
    mjm.actuator_gainprm[:, 0] = kp
    mjm.actuator_biasprm[:, 1] = -kp
    mjm.actuator_biasprm[:, 2] = -kd
    mjm.actuator_forcerange[:] = np.stack([-effort, effort], 1)
    mjm.dof_armature[6:] = armature
    self.mjm = mjm
    bid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
    self.fall_ids = torch.tensor([bid(n) for n in FALL_BODIES], device=self.device)
    self.feet = torch.tensor([bid(n) for n in FEET], device=self.device)

    # Standing pelvis height above the soles, for spawning on uneven ground.
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:3] = 0
    mjd.qpos[7:] = DEFAULT_ANGLES
    mjd.qpos[3:7] = (1, 0, 0, 0)
    mujoco.mj_kinematics(mjm, mjd)
    foot_geoms = [g for g in range(mjm.ngeom) if "foot" in (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    self.stand_h = float(-min(mjd.geom_xpos[g][2] - mjm.geom_size[g][0] for g in foot_geoms))

    wp.init()
    self.wp_device = "cuda:0" if self.device.type == "cuda" else "cpu"
    with wp.ScopedDevice(self.wp_device):
      self.m = mjw.put_model(mjm)
      if hasattr(self.m.opt, "warn_overflow"):  # mujoco-warp >= 3.14 prints solver-limit warnings every step
        self.m.opt.warn_overflow = int(self.m.opt.warn_overflow) & ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
      self.d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=96, njmax=2000)
    T = wp.to_torch
    self.qpos, self.qvel, self.ctrl = T(self.d.qpos), T(self.d.qvel), T(self.d.ctrl)
    self.xpos, self.xmat = T(self.d.xpos), T(self.d.xmat)
    self.qacc_warm = T(self.d.qacc_warmstart)  # zeroed on reset: a world that blew up must not warm-start from NaN
    self.mocap_pos, self.mocap_quat = T(self.d.mocap_pos), T(self.d.mocap_quat)  # (N, 2, 3), (N, 2, 4): left, right
    # Sole sample points in the foot frame (toe corners, heel corners, middle) for fitting the ground plane.
    self.sole_xy = torch.tensor([[0.12, -0.03], [0.12, 0.03], [-0.045, -0.025], [-0.045, 0.025], [0.04, 0.0]],
                                dtype=torch.float32, device=self.device)
    self.graph = None
    self.use_graph = use_graph and self.device.type == "cuda"
    self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
    self.wp_stream = wp.stream_from_torch(self.stream) if self.stream is not None else None

    # Planner walking clips (hikesim.clips): joints in MuJoCo order, heading-free root quats, planned speed.
    c = np.load(clips)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=self.device)
    self.clip_q = f(c["q"][:, :, M2I])                      # (C, T, 29) IsaacLab order, absolute
    self.clip_dq = torch.gradient(self.clip_q, spacing=1 / 50, dim=1)[0]
    self.clip_R = quat_to_mat(f(c["quat"]))                  # (C, T, 3, 3)
    self.clip_legs = f(c["q"][:, :, LEGS_MJ])                # (C, T, 12) for the gait phase observation
    self.clip_speed = f(c["speed"])  # the planned root speed, which SONIC reproduces in physics to within ~0.03 m/s
    self.n_clips, self.clip_T = self.clip_q.shape[0], self.clip_q.shape[1]

    # SONIC (frozen).
    self.enc = G1Encoder().to(self.device).eval().requires_grad_(False)
    self.dec = Decoder().to(self.device).eval().requires_grad_(False)
    self.default = f(DEFAULT_ANGLES)
    self.scale = f(scale)
    self.i2m, self.m2i = torch.tensor(I2M, device=self.device), torch.tensor(M2I, device=self.device)
    self.fut = torch.arange(10, device=self.device) * 5

    N = num_envs
    self.num_actions = 31
    self.hist_ang = torch.zeros(N, 10, 3, device=self.device)
    self.hist_q = torch.zeros(N, 10, 29, device=self.device)
    self.hist_dq = torch.zeros(N, 10, 29, device=self.device)
    self.hist_act = torch.zeros(N, 10, 29, device=self.device)
    self.hist_grav = torch.zeros(N, 10, 3, device=self.device)
    self.sonic_last = torch.zeros(N, 29, device=self.device)
    self.last_action = torch.zeros(N, 31, device=self.device)
    self.prev_action = torch.zeros(N, 31, device=self.device)
    self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
    self.level = torch.randint(0, max(1, min(start_level, self.n_tiles)), (N,), generator=self.gen, device=self.device)
    self.level_step = 5  # 251 tiles: +-2 per episode needed ~3000 iterations to reach the steep ones
    self.tile = self.level.clone()
    self.clip = torch.zeros(N, dtype=torch.long, device=self.device)
    self.phase = torch.zeros(N, dtype=torch.long, device=self.device)
    self.walk_from = torch.zeros(N, dtype=torch.long, device=self.device)  # clip frame where walking starts
    self.ref_heading = torch.zeros(N, device=self.device)
    self.cmd_heading = torch.zeros(N, device=self.device)
    self.y_target = torch.zeros(N, device=self.device)
    self.start_x = torch.zeros(N, device=self.device)
    self.lean = torch.zeros(N, 2, device=self.device)
    self.reset_idx(torch.arange(N, device=self.device))
    if self.stream is not None:
      torch.cuda.synchronize(self.device)
    self._forward()
    if self.stream is not None:
      torch.cuda.synchronize(self.device)
    self._refresh_history(torch.arange(N, device=self.device), fill=True)
    self._update_command(rate_limit=False)
    self.obs = self._observations()

  # ------------------------------------------------------------------------------------------------
  def _load_terrain(self, mosaic) -> HeightGrid:
    """The trail tile mosaic, one hfield shared by every world (worlds never touch each other)."""
    meta = json.loads((Path(mosaic) / "mosaic.json").read_text())
    z = np.load(Path(mosaic) / "mosaic_elevation.npy").astype(np.float64)
    self.grid = TorchGrid(z, meta["x0"], meta["y0"], meta["cell"], self.device)
    self.tile_y = torch.tensor([t["y_center_m"] for t in meta["tiles"]], dtype=torch.float32, device=self.device)
    self.tile_grade = torch.tensor([t["grade_deg"] for t in meta["tiles"]], dtype=torch.float32, device=self.device)
    self.n_tiles = len(self.tile_y)
    self.finish_x = meta["chord_m"] - 1.0
    self.half_width = meta["tile_width_m"] / 2 - 1.5
    return HeightGrid(z=z, x0=meta["x0"], y0=meta["y0"], cell=meta["cell"])

  def _finished(self):
    return self.qpos[:, 0] > self.finish_x

  def _off_course(self):
    return (self.qpos[:, 1] - self.tile_y[self.tile]).abs() > self.half_width

  def _trail_offset(self):
    """Signed distance from the line the robot should walk (critic input and the on_trail penalty)."""
    return self.qpos[:, 1] - self.tile_y[self.tile] - self.y_target

  def _hn(self, x, y):
    """Terrain height and normal. A blown-up world's NaN position must not turn into a wild grid index on the GPU."""
    return self.grid.height_normal(torch.nan_to_num(x, 0.0, 0.0, 0.0), torch.nan_to_num(y, 0.0, 0.0, 0.0))

  def _rand(self, n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=self.gen, device=self.device)

  def _scope(self):
    from contextlib import ExitStack
    st = ExitStack()
    st.enter_context(wp.ScopedDevice(self.wp_device))
    if self.wp_stream is not None:
      st.enter_context(wp.ScopedStream(self.wp_stream))
    return st

  def _forward(self):
    with self._scope():
      mjw.forward(self.m, self.d)
    self._ground_boxes()

  def _ground_boxes(self):
    """Move each foot's ground box under it: a plane fitted through the terrain under 5 sole points, lifted onto the
    highest of them, so a root or rock under the toe or heel tilts the foot like a rigid sole would."""
    p = self.xpos[:, self.feet]                                    # (N, 2, 3)
    R = self.xmat[:, self.feet]                                    # (N, 2, 3, 3)
    pts = p[..., None, :2] + torch.einsum("nfij,kj->nfki", R[..., :2, :2], self.sole_xy)  # (N, 2, 5, 2)
    h, _ = self._hn(pts[..., 0], pts[..., 1])                      # (N, 2, 5)
    m = pts.mean(-2, keepdim=True)
    dx, dy = pts[..., 0] - m[..., 0], pts[..., 1] - m[..., 1]
    hm = h.mean(-1, keepdim=True)
    dz = h - hm
    sxx, syy, sxy = (dx * dx).sum(-1), (dy * dy).sum(-1), (dx * dy).sum(-1)
    sxz, syz = (dx * dz).sum(-1), (dy * dz).sum(-1)
    det = (sxx * syy - sxy * sxy).clamp_min(1e-6)
    gx = ((syy * sxz - sxy * syz) / det).clamp(-0.84, 0.84)        # slope capped at 40 deg
    gy = ((sxx * syz - sxy * sxz) / det).clamp(-0.84, 0.84)
    lift = (dz - gx[..., None] * dx - gy[..., None] * dy).amax(-1).clamp_min(0)
    zc = hm[..., 0] + gx * (p[..., 0] - m[..., 0, 0]) + gy * (p[..., 1] - m[..., 0, 1]) + lift
    n = torch.stack([-gx, -gy, torch.ones_like(gx)], -1)
    n = n / n.norm(dim=-1, keepdim=True)
    top = torch.stack([p[..., 0], p[..., 1], zc], -1)
    self.ground_top, self.ground_n = top, n  # per foot, for sole clearance in the rough-terrain env
    self.mocap_pos[:] = top - n * BOX_HALF[2]
    # quaternion turning +z onto n: axis z x n, angle acos(n_z)
    w = torch.sqrt(torch.clamp((1 + n[..., 2]) / 2, min=1e-8))
    self.mocap_quat[:] = torch.stack([w, -n[..., 1] / (2 * w), n[..., 0] / (2 * w), torch.zeros_like(w)], -1)

  def _physics_step(self):
    self._ground_boxes()
    with self._scope():
      if self.use_graph:
        if self.graph is None:
          mjw.step(self.m, self.d)
          with wp.ScopedCapture() as cap:
            mjw.step(self.m, self.d)
          self.graph = cap.graph
        else:
          wp.capture_launch(self.graph)
      else:
        mjw.step(self.m, self.d)

  def _speed_cmd(self):
    """The clip's walking speed once its stand-up second is over."""
    walking = (self.phase >= self.walk_from + 40).float()
    return self.clip_speed[self.clip] * walking

  def _update_command(self, rate_limit=True):
    """Follow the trail line: aim 3 m ahead at the target lateral offset; the reference heading turns at most 0.8 rad/s."""
    tgt = torch.clamp(torch.atan2(self.tile_y[self.tile] + self.y_target - self.qpos[:, 1], torch.full_like(self.qpos[:, 0], 3.0)), -0.6, 0.6)
    self.cmd_heading = tgt
    if rate_limit:
      step = 0.8 * self.dt * self.decimation
      self.ref_heading = self.ref_heading + torch.clamp(wrap(tgt - self.ref_heading), -step, step)
    else:
      self.ref_heading = tgt.clone()

  def reset_idx(self, ids):
    if len(ids) == 0:
      return
    n = len(ids)
    rand_tile = torch.randint(0, self.n_tiles, (n,), generator=self.gen, device=self.device)
    explore = torch.rand(n, generator=self.gen, device=self.device) < self.explore_frac
    self.tile[ids] = torch.where(explore, rand_tile, self.level[ids])
    x = self._rand(n, 0.0, 0.5)
    y = self.tile_y[self.tile[ids]] + self._rand(n, -0.3, 0.3)
    yaw = self._rand(n, -0.25, 0.25)
    # Highest ground under the feet footprint, then stand on it.
    ox = torch.tensor([-0.12, 0.15, -0.12, 0.15], device=self.device)
    oy = torch.tensor([-0.15, -0.15, 0.15, 0.15], device=self.device)
    h, _ = self._hn(x[:, None] + ox, y[:, None] + oy)
    zc = h.max(-1).values + self.stand_h + 0.01
    self.qpos[ids, 0], self.qpos[ids, 1], self.qpos[ids, 2] = x, y, zc
    self.qpos[ids, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
    self.qpos[ids, 7:] = self.default + 0.03 * (torch.rand(n, 29, generator=self.gen, device=self.device) - 0.5)
    self.qvel[ids] = 0
    self.qacc_warm[ids] = 0
    self.ctrl[ids] = self.default
    # Clip and phase: half the episodes start standing (the clip's own stand-up), half mid-stride at speed.
    self.clip[ids] = torch.randint(0, self.n_clips, (n,), generator=self.gen, device=self.device)
    mid = torch.rand(n, generator=self.gen, device=self.device) < 0.5
    hi = self.clip_T - self.max_episode_length - 60
    self.phase[ids] = torch.where(mid, torch.randint(300, hi, (n,), generator=self.gen, device=self.device), torch.zeros_like(self.phase[ids]))
    self.walk_from[ids] = torch.where(mid, self.phase[ids] - 100, torch.full_like(self.phase[ids], 50))
    v0 = torch.where(mid, self.clip_speed[self.clip[ids]] * 0.8, torch.zeros(n, device=self.device))
    self.qvel[ids, 0], self.qvel[ids, 1] = v0 * torch.cos(yaw), v0 * torch.sin(yaw)
    self.y_target[ids] = self._rand(n, -0.4, 0.4)
    self.ref_heading[ids] = yaw
    self.start_x[ids] = x
    self.lean[ids] = 0
    self.episode_length_buf[ids] = 0
    self.last_action[ids] = 0
    self.prev_action[ids] = 0
    self.sonic_last[ids] = 0

  # ------------------------------------------------------------------------------------------------
  def _state(self):
    R = quat_to_mat(self.qpos[:, 3:7])
    ang = self.qvel[:, 3:6]
    grav = -R[:, 2, :]
    q = (self.qpos[:, 7:] - self.default)[:, self.m2i]
    dq = self.qvel[:, 6:][:, self.m2i]
    return R, ang, grav, q, dq

  def _refresh_history(self, ids=None, fill=False):
    R, ang, grav, q, dq = self._state()
    if fill:
      self.hist_ang[ids] = ang[ids, None].expand(-1, 10, -1).clone()
      self.hist_q[ids] = q[ids, None].expand(-1, 10, -1).clone()
      self.hist_dq[ids] = dq[ids, None].expand(-1, 10, -1).clone()
      self.hist_act[ids] = 0
      self.hist_grav[ids] = grav[ids, None].expand(-1, 10, -1).clone()
      return
    for buf, new in ((self.hist_ang, ang), (self.hist_q, q), (self.hist_dq, dq), (self.hist_act, self.sonic_last), (self.hist_grav, grav)):
      buf[:, :-1] = buf[:, 1:].clone()
      buf[:, -1] = new

  def _sonic(self):
    """Encode 10 future clip frames, pointed along the reference heading and leaned; decode with proprioception."""
    R, *_ = self._state()
    idx = torch.clamp(self.phase[:, None] + self.fut, max=self.clip_T - 1)       # (N, 10)
    ci = self.clip[:, None].expand(-1, 10)
    qf, dqf, Rc = self.clip_q[ci, idx], self.clip_dq[ci, idx], self.clip_R[ci, idx]  # (N,10,29), (N,10,3,3)
    Rh = rot_zyx(self.ref_heading, self.lean[:, 0], self.lean[:, 1])               # (N,3,3)
    Rref = Rh[:, None] @ Rc
    M = R.transpose(1, 2)[:, None] @ Rref
    anchor = torch.stack([M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1], M[..., 2, 0], M[..., 2, 1]], -1)
    with torch.no_grad():
      token = self.enc(torch.cat([qf.flatten(1), dqf.flatten(1)], -1), anchor.flatten(1))
      dec_in = torch.cat([token, self.hist_ang.flatten(1), self.hist_q.flatten(1), self.hist_dq.flatten(1),
                          self.hist_act.flatten(1), self.hist_grav.flatten(1)], -1)
      return self.dec(dec_in)

  # ------------------------------------------------------------------------------------------------
  def get_observations(self):
    critic = torch.cat([self.obs, self._privileged()], -1) if self.privileged_critic else self.obs
    return TensorDict({"actor": self.obs, "critic": critic}, batch_size=[self.num_envs])

  def _privileged(self):
    """Critic only: true slope under the robot, the tile's trail grade, lateral offset from the line, time left."""
    _, nrm = self._hn(self.qpos[:, 0], self.qpos[:, 1])
    off = self._trail_offset()[:, None]
    t_frac = (self.episode_length_buf.float() / self.max_episode_length)[:, None]
    priv = torch.cat([nrm, self.tile_grade[self.tile][:, None] / 30, off, self.qvel[:, 0:3] * 0.5, t_frac], -1)
    return torch.nan_to_num(priv, 0.0, 2.0, -2.0)

  def _feet_contact(self):
    p = self.xpos[:, self.feet]
    h, _ = self._hn(p[..., 0], p[..., 1])
    return ((p[..., 2] - h) < 0.07).float()

  def _observations(self):
    R, ang, grav, q, dq = self._state()
    vb = (R.transpose(1, 2) @ self.qvel[:, 0:3, None])[..., 0]
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    gx, gy = torch.meshgrid(torch.tensor(SCAN_X, device=self.device), torch.tensor(SCAN_Y, device=self.device), indexing="ij")
    gx, gy = gx.flatten(), gy.flatten()
    px = self.qpos[:, 0:1] + c[:, None] * gx - s[:, None] * gy
    py = self.qpos[:, 1:2] + s[:, None] * gx + c[:, None] * gy
    h, _ = self._hn(px, py)
    heights = torch.clamp(h - self.qpos[:, 2:3] + self.stand_h, -1.5, 1.5)
    dh = wrap(self.ref_heading - yaw)
    ref_legs = self.clip_legs[self.clip, torch.clamp(self.phase, max=self.clip_T - 1)]
    obs = torch.cat([
      ang * 0.25, grav, vb * 0.5, q, dq * 0.05, self.last_action, heights,
      torch.stack([torch.sin(dh), torch.cos(dh), self._speed_cmd()], -1), ref_legs, self._feet_contact(),
    ], -1)
    return torch.nan_to_num(obs, 0.0, 10.0, -10.0)

  def step(self, actions):
    if self.stream is None:
      return self._step(actions)
    caller = torch.cuda.current_stream(self.device)
    self.stream.wait_stream(caller)
    with torch.cuda.stream(self.stream):
      out = self._step(actions)
    caller.wait_stream(self.stream)
    return out

  def _step(self, actions):
    actions = torch.clamp(actions, -3, 3)
    self.prev_action = self.last_action
    self.last_action = actions
    self.lean = actions[:, :2] * torch.tensor([0.25, 0.12], device=self.device)  # pitch up to ~0.75 rad, roll ~0.36
    residual = actions[:, 2:] * 0.3
    self._refresh_history()
    total = self._sonic() + residual
    self.sonic_last = total
    self.ctrl[:] = self.default + total[:, self.i2m] * self.scale
    for _ in range(self.decimation):
      self._physics_step()
    self.episode_length_buf += 1
    self.phase = torch.clamp(self.phase + 1, max=self.clip_T - 1)
    self._update_command()

    # Rewards.
    R, ang, grav, q, dq = self._state()
    v = self.qvel[:, 0:3]
    ch, sh = torch.cos(self.cmd_heading), torch.sin(self.cmd_heading)
    fwd = v[:, 0] * ch + v[:, 1] * sh
    lat = -v[:, 0] * sh + v[:, 1] * ch
    vcmd = self._speed_cmd()
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    herr = wrap(yaw - self.ref_heading)
    off = self._trail_offset().abs()
    r = {
      "track_speed": 2.0 * torch.exp(-(fwd - vcmd) ** 2 / 0.1),
      "progress": 1.0 * torch.clamp(fwd, -0.5, 1.0) * (vcmd > 0).float(),
      "still_when_standing": -1.0 * fwd.abs() * (vcmd == 0).float(),
      "heading": 0.5 * torch.exp(-herr ** 2 / 0.15),
      "lateral_vel": -0.5 * lat ** 2,
      "on_trail": -0.3 * off,
      "alive": 1.0 * torch.ones_like(fwd),
      "upright": -1.0 * grav[:, 1] ** 2 - 0.3 * grav[:, 0] ** 2,
      "ang_vel": -0.02 * (ang[:, :2] ** 2).sum(-1),
      "action_rate": -0.01 * ((self.last_action - self.prev_action) ** 2).sum(-1),
      "residual": -0.05 * (residual ** 2).sum(-1),
      "joint_vel": -1e-4 * (dq ** 2).sum(-1),
    }

    # Terminations.
    hp, _ = self._hn(self.qpos[:, 0], self.qpos[:, 1])
    fb = self.xpos[:, self.fall_ids]
    hb, _ = self._hn(fb[..., 0], fb[..., 1])
    blowup = ~torch.isfinite(self.qpos).all(-1) | (self.qvel.abs().max(-1).values > 200)  # diverging (contact spikes reach ~150)
    fell = (R[:, 2, 2] < 0.5) | (self.qpos[:, 2] - hp < 0.45) | (fb[..., 2] - hb < 0.05).any(-1) | blowup
    finished = self._finished()
    off_tile = self._off_course()
    timeout = (self.episode_length_buf >= self.max_episode_length) | finished | off_tile
    r["termination"] = -30.0 * fell.float()
    # A world that blew up gets only the fall penalty: its 1e9 m/s velocities would otherwise swamp the value function.
    for k in r:
      r[k] = torch.where(blowup, torch.zeros_like(r[k]), torch.nan_to_num(r[k], 0.0, 0.0, 0.0))
    r["termination"] = -30.0 * fell.float()
    reward = torch.clamp(sum(r.values()), -60.0, 10.0) * (self.dt * self.decimation)
    for k, val in r.items():
      m = val.mean().item()
      self.rew_ema[k] = m if k not in self.rew_ema else 0.99 * self.rew_ema[k] + 0.01 * m

    self.last_fell = fell
    if not self.auto_reset:
      self.obs = self._observations()
      return self.get_observations(), reward, torch.zeros_like(fell), {"time_outs": torch.zeros_like(fell), "log": {}}
    done = fell | timeout
    ids = done.nonzero().flatten()
    extras = {"time_outs": timeout & ~fell, "log": {}}
    if len(ids):
      dist = torch.nan_to_num(self.qpos[ids, 0] - self.start_x[ids], 0.0, 0.0, 0.0)
      t = self.episode_length_buf[ids].float() * self.dt * self.decimation
      expect = self.clip_speed[self.clip[ids]] * torch.clamp(t - 1.0, min=0.0)
      good = ~fell[ids] & ((dist > 0.6 * expect) | finished[ids]) & (t > 0.5 * self.cfg["episode_s"]) | finished[ids]
      up, down = good, fell[ids]
      self.level[ids] = torch.clamp(self.level[ids] + self.level_step * (up.long() - down.long()), 0, self.n_tiles - 1)
      tg = self.tile_grade[self.tile[ids]]
      extras["log"] = {
        "Episode/fell_frac": fell[ids].float().mean().item(),
        "Episode/blowup_frac": blowup[ids].float().mean().item(),
        "Episode/finished_frac": finished[ids].float().mean().item(),
        "Episode/distance_m": dist.mean().item(),
        "Episode/length_s": t.mean().item(),
        "Curriculum/mean_level": self.level.float().mean().item(),
        "Curriculum/max_level": self.level.float().max().item(),
        "Curriculum/mean_abs_grade": tg.abs().mean().item(),
        "Task/speed_mps": fwd.mean().item(),
        "Task/heading_err_deg": math.degrees(herr.abs().mean().item()),
        **{f"Reward/{k}": v for k, v in self.rew_ema.items()},
      }
      climbs = ids[tg > 8]
      if len(climbs):
        extras["log"]["Episode/fell_frac_climbs_over_8deg"] = fell[climbs].float().mean().item()
      self.reset_idx(ids)
      self._forward()
      self._refresh_history(ids, fill=True)
      snap = self.ref_heading.clone()
      self._update_command(rate_limit=False)
      keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
      keep[ids] = False
      self.ref_heading = torch.where(keep, snap, self.ref_heading)
    self.obs = self._observations()
    return self.get_observations(), reward, done, extras
