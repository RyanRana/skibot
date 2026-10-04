"""Batched skiing environment: Unitree G1 on skis in MuJoCo Warp, frozen SONIC underneath, residual policy on top.

Per policy step (50 Hz):
  policy action (31) = body lean target for SONIC (roll, pitch) + residual on SONIC's 29 raw joint actions
  SONIC encoder tracks a crouched ski stance with that lean; decoder turns its token plus 10 frames of
  proprioception into joint targets; 10 physics substeps of 2 ms with the ski-snow forces applied each substep.
Task: follow a commanded travel heading (resampled every few seconds, so it has to turn) at a commanded speed
without falling, on perturbed real-course terrain tiles with a difficulty curriculum.
The MuJoCo model has contacts disabled: skis touch the snow through the torch contact model, and falls end the episode.
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
from skisim.ski import SkiParams
from skisim.ski_torch import TorchGrid, TorchSkiContact, drag
from skisim.sonic import DEFAULT_ANGLES, I2M, JOINTS, M2I, sonic_gains
from skisim.sonic_torch import Decoder, G1Encoder
from skisim.terrain import slope

ROOT = Path(__file__).resolve().parents[1]

# Crouched ski stance SONIC tracks (MuJoCo order). Best zero-shot result: 93 m on 15 deg.
STANCE = DEFAULT_ANGLES.copy()
for _i, _j in enumerate(JOINTS):
  if _j.endswith("hip_pitch_joint"):
    STANCE[_i] = -0.6
  elif _j.endswith("knee_joint"):
    STANCE[_i] = 1.1
  elif _j.endswith("ankle_pitch_joint"):
    STANCE[_i] = -0.5
  elif _j == "waist_pitch_joint":
    STANCE[_i] = 0.25

POSE_JSON = ROOT / "data" / "pose" / "ski_pose.json"


def athletic_poses():
  """Neutral, left-turn and right-turn G1 poses measured from World Cup GS video (falls back to STANCE)."""
  if not POSE_JSON.exists():
    return STANCE.copy(), STANCE.copy(), STANCE.copy()
  d = json.loads(POSE_JSON.read_text())

  def vec(name):
    q = STANCE.copy()
    for j, v in d.get(name, d["neutral"])["g1"].items():
      q[JOINTS.index(j)] = v
    return q

  return vec("neutral"), vec("left_turn"), vec("right_turn")


# Racer's stance, as offsets on the video poses (rad): knees 17 deg deeper with hips and ankles matched so the feet
# stay flat, torso 7 deg further forward, hands higher, closer in and elbows bent like a GS racer's. Used by
# style="racing"; it moves SONIC's target pose too, so a checkpoint trained with it must be run with it.
RACING_DELTA = {
  "left_knee_joint": 0.30, "right_knee_joint": 0.30, "left_hip_pitch_joint": -0.22, "right_hip_pitch_joint": -0.22,
  "left_ankle_pitch_joint": -0.08, "right_ankle_pitch_joint": -0.08, "waist_pitch_joint": 0.12,
  "left_shoulder_pitch_joint": -0.29, "right_shoulder_pitch_joint": -0.29,
  "left_shoulder_roll_joint": -0.24, "right_shoulder_roll_joint": 0.24, "left_elbow_joint": 0.46, "right_elbow_joint": 0.46,
}

# Tree scan: a horizontal fan of rays in front of the robot, distance to the nearest trunk, like a 2D lidar.
N_RAYS = 15
RAY_FOV_DEG = 200.0
RAY_RANGE = 15.0
RAY_INFLATE = 0.6  # m added to each trunk for sensing; rays are 13 deg apart, 2.3 m at 10 m
# Trees are obstacles as drawn: the 8-bit canopy reaches about 1.5x the trunk-and-branches radius in trees.npy, and
# touching it ends the run. Keeping clear only of the bare radius let the robot ski through the visible branches.
CANOPY = 1.6
BODY_MARGIN = 0.35  # m, half the robot's width at the hips and skis

FALL_BODIES = ("torso_link", "left_knee_link", "right_knee_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
HEIGHT_X = (-1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0)
HEIGHT_Y = (-1.5, 0.0, 1.5)


def quat_to_mat(q):
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ], -1).reshape(*q.shape[:-1], 3, 3)


def mat_to_quat(R):
  # Robust enough for proper rotations built from orthonormal frames.
  w = torch.sqrt(torch.clamp(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], min=1e-8)) / 2
  x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
  y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
  z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
  q = torch.stack([w, x, y, z], -1)
  return q / q.norm(dim=-1, keepdim=True)


def wrap(a):
  return torch.atan2(torch.sin(a), torch.cos(a))


class SkiEnv:
  def __init__(self, num_envs=4096, device="cuda", mosaic=ROOT / "out/terrains", episode_s=15.0, seed=0,
               use_graph=True, backend="warp"):
    self.num_envs, self.device = num_envs, torch.device(device)
    self.dt, self.decimation = 0.002, 10
    self.max_episode_length = int(episode_s / (self.dt * self.decimation))
    self.cfg = {"episode_s": episode_s}
    self.gen = torch.Generator(device=self.device).manual_seed(seed)
    self.p = SkiParams()

    # MuJoCo model: Unitree G1 + skis, SONIC gains, no contacts.
    mjm = g1_model(slope(0.0, 4, 4, 1.0), self.p)
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mjm.opt.timestep = self.dt
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
    self.ski_ids = torch.tensor([bid("left_ski"), bid("right_ski")], device=self.device)
    self.pelvis = bid("pelvis")
    self.fall_ids = torch.tensor([bid(n) for n in FALL_BODIES], device=self.device)

    # Ski contact points in the pelvis frame at the stance pose (for spawn height).
    mjd = mujoco.MjData(mjm)
    mjd.qpos[7:] = athletic_poses()[0]
    mjd.qpos[3:7] = (1, 0, 0, 0)
    mujoco.mj_kinematics(mjm, mjd)
    self.contact = TorchSkiContact(params=self.p, device=self.device)
    loc = self.contact.local.cpu().numpy()
    pts = [mjd.xpos[b] + loc @ mjd.xmat[b].reshape(3, 3).T for b in (bid("left_ski"), bid("right_ski"))]
    self.ski_pts_pelvis = torch.tensor(np.concatenate(pts) - mjd.xpos[self.pelvis], dtype=torch.float32, device=self.device)

    self.backend = backend
    self.auto_reset = True
    self.privileged_critic = True
    self.style = "default"  # "racing": reward a racer's form (low stance, quiet hands forward, angulation, carving)
    self.rew_ema = {}
    self.explore_frac = 0.25  # share of resets sent to a random tile instead of the curriculum level
    self.spawn_speed = (0.0, 9.0)  # m/s along the initial heading: real courses are skied at 8 to 10 m/s for minutes
    self.last_fell = None
    if backend == "mujoco":
      # One world in plain CPU MuJoCo (fast for a single robot); float32 torch mirrors keep the env code identical.
      assert num_envs == 1, "the mujoco backend runs a single world"
      self.cpu_m, self.cpu_d = mjm, mjd
      Z = lambda *shape: torch.zeros(1, *shape)
      nb = mjm.nbody
      self.qpos, self.qvel, self.ctrl = Z(mjm.nq), Z(mjm.nv), Z(mjm.nu)
      self.xpos, self.xmat, self.xipos = Z(nb, 3), Z(nb, 3, 3), Z(nb, 3)
      self.cvel, self.subtree_com, self.xfrc = Z(nb, 6), Z(nb, 3), Z(nb, 6)
      mujoco.mj_forward(mjm, mjd)
      self._pull()
      self.wp_device, self.wp_stream = "cpu", None
    else:
      wp.init()
      self.wp_device = "cuda:0" if self.device.type == "cuda" else "cpu"
      with wp.ScopedDevice(self.wp_device):
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=num_envs)
      T = wp.to_torch
      self.qpos, self.qvel, self.ctrl = T(self.d.qpos), T(self.d.qvel), T(self.d.ctrl)
      self.xpos, self.xmat, self.xipos = T(self.d.xpos), T(self.d.xmat), T(self.d.xipos)
      self.cvel, self.subtree_com, self.xfrc = T(self.d.cvel), T(self.d.subtree_com), T(self.d.xfrc_applied)
    self.ski_root = torch.tensor(mjm.body_rootid[self.ski_ids.cpu().numpy()], device=self.device)
    self.graph = None
    self.use_graph = use_graph and self.device.type == "cuda"
    # The env runs on its own CUDA stream (graph capture is impossible on the default stream); warp shares it so
    # torch writes (xfrc, ctrl, qpos) and physics stay ordered. step() syncs with the caller's stream at both ends.
    self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
    if backend != "mujoco":
      self.wp_stream = wp.stream_from_torch(self.stream) if self.stream is not None else None

    # Terrain mosaic and tile layout.
    meta = json.loads((Path(mosaic) / "mosaic.json").read_text())
    z = np.load(Path(mosaic) / "mosaic_elevation.npy")
    self.grid = TorchGrid(z, meta["x0"], meta["y0"], meta["cell"], self.device)
    self.tile_y = torch.tensor([t["y_center_m"] for t in meta["tiles"]], dtype=torch.float32, device=self.device)
    # Steeper tiles get wider turns: on a 25 degree pitch a skier only controls speed by crossing the fall line.
    slope_deg = torch.tensor([t.get("mean_slope_deg", 10.0) for t in meta["tiles"]], dtype=torch.float32, device=self.device)
    # Max heading off the fall line: 52 deg on gentle tiles up to 75 deg on steep ones. Real pistes traverse the hill;
    # with the old 34 deg cap the robot could not follow a course line across the slope and ran off it.
    self.tile_hmax = 0.9 + 0.4 * torch.clamp((slope_deg - 12.0) / 15.0, 0, 1)
    keys = ("mu_glide", "mu_skid", "mu_carve", "k_normal")
    defaults = {"mu_glide": self.p.mu_glide, "mu_skid": self.p.mu_skid, "mu_carve": self.p.mu_carve, "k_normal": self.p.k_normal}
    self.tile_snow = torch.tensor([[t.get("snow_params", defaults)[k] for k in keys] for t in meta["tiles"]],
                                  dtype=torch.float32, device=self.device)
    self.snow = {k: torch.full((num_envs, 1), defaults[k], device=self.device) for k in keys}
    self.n_tiles = len(self.tile_y)
    # Trees per tile, padded: (n_tiles, K, 3) with x, y, radius. Tiles without trees get far-away dummies.
    tree_file = Path(mosaic) / "trees.npy"
    trees = np.load(tree_file) if tree_file.exists() else np.zeros((0, 5), np.float32)
    per = [trees[trees[:, 4] == k][:, :3] for k in range(self.n_tiles)]
    K = max(1, max((len(p) for p in per), default=1))
    arr = np.zeros((self.n_tiles, K, 3), np.float32)
    arr[..., :2], arr[..., 2] = 1e6, 0.1
    for k, p in enumerate(per):
      arr[k, : len(p)] = p
    self.tile_trees = torch.tensor(arr, device=self.device)
    self.n_trees = len(trees)
    self.ray_angles = torch.deg2rad(torch.linspace(-RAY_FOV_DEG / 2, RAY_FOV_DEG / 2, N_RAYS, device=self.device))
    self.tree_rays = torch.ones(num_envs, N_RAYS, device=self.device)
    self.tile_len = (self.grid.ncol - 1) * self.grid.cell

    # SONIC (frozen).
    self.enc = G1Encoder().to(self.device).eval().requires_grad_(False)
    self.dec = Decoder().to(self.device).eval().requires_grad_(False)
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
    self.default = f(DEFAULT_ANGLES)
    self.scale = f(scale)
    self.i2m, self.m2i = torch.tensor(I2M, device=self.device), torch.tensor(M2I, device=self.device)
    neutral, left, right = athletic_poses()
    self._ref_base = (f(neutral), f(left), f(right))
    self.ref_neutral, self.ref_left, self.ref_right = self._ref_base
    self.stance = self.ref_neutral
    self.ref_cur = self.ref_neutral.expand(num_envs, -1).clone()

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
    self.level = torch.randint(0, min(4, self.n_tiles), (N,), generator=self.gen, device=self.device)
    self.tile = self.level.clone()
    self.start_x = torch.zeros(N, device=self.device)
    self.cmd_heading = torch.zeros(N, device=self.device)
    self.cmd_goal = torch.zeros(N, device=self.device)
    self.shield_on = torch.zeros(N, dtype=torch.bool, device=self.device)  # free-skiing heading before piste keeping and tree avoidance
    self.cmd_speed = torch.zeros(N, device=self.device)
    self.cmd_timer = torch.zeros(N, device=self.device)
    # Gates: alternating checkpoints down the tile. 80% of episodes race gates, 20% follow free heading commands.
    self.G = 14
    self.gate_x = torch.full((N, self.G), 1e6, device=self.device)
    self.gate_y = torch.zeros(N, self.G, device=self.device)
    self.gate_idx = torch.zeros(N, dtype=torch.long, device=self.device)
    self.use_gates = torch.ones(N, dtype=torch.bool, device=self.device)
    self.gates_hit = torch.zeros(N, device=self.device)
    self.gates_crossed = torch.zeros(N, device=self.device)
    self.gate_frac = 0.7
    self.ski_diag = None
    self.episode_sums = {}
    self.reset_idx(torch.arange(N, device=self.device))
    if self.stream is not None:
      torch.cuda.synchronize(self.device)  # initial writes were on the default stream
    self._forward()
    if self.stream is not None:
      torch.cuda.synchronize(self.device)
    self._refresh_history(torch.arange(N, device=self.device), fill=True)
    self._update_gate_command(rate_limit=False)
    self.obs = self._observations()

  # ------------------------------------------------------------------------------------------------
  def _rand(self, n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=self.gen, device=self.device)

  def _scope(self):
    from contextlib import ExitStack
    st = ExitStack()
    st.enter_context(wp.ScopedDevice(self.wp_device))
    if self.wp_stream is not None:
      st.enter_context(wp.ScopedStream(self.wp_stream))
    return st

  def _pull(self):
    d = self.cpu_d
    for dst, src in ((self.qpos, d.qpos), (self.qvel, d.qvel), (self.ctrl, d.ctrl), (self.xpos, d.xpos),
                     (self.xmat, d.xmat.reshape(-1, 3, 3)), (self.xipos, d.xipos), (self.cvel, d.cvel),
                     (self.subtree_com, d.subtree_com)):
      dst[0].copy_(torch.from_numpy(src))

  def _push(self):
    d = self.cpu_d
    d.qpos[:] = self.qpos[0].numpy()
    d.qvel[:] = self.qvel[0].numpy()
    d.ctrl[:] = self.ctrl[0].numpy()
    d.xfrc_applied[:] = self.xfrc[0].numpy()

  def _forward(self):
    if self.backend == "mujoco":
      self._push()
      mujoco.mj_forward(self.cpu_m, self.cpu_d)
      self._pull()
      return
    with self._scope():
      mjw.forward(self.m, self.d)

  def _physics_step(self):
    if self.backend == "mujoco":
      self._push()
      mujoco.mj_step(self.cpu_m, self.cpu_d)
      self._pull()
      return
    with self._scope():
      if self.use_graph:
        if self.graph is None:
          mjw.step(self.m, self.d)  # warm up kernels before capture
          with wp.ScopedCapture() as cap:
            mjw.step(self.m, self.d)
          self.graph = cap.graph
        else:
          wp.capture_launch(self.graph)
      else:
        mjw.step(self.m, self.d)

  def _resample_commands(self, ids):
    n = len(ids)
    # Travel heading relative to the fall line (+x), biased back toward the middle of the run when off to one side.
    off = self.qpos[ids, 1] - self.tile_y[self.tile[ids]]
    hmax = self.tile_hmax[self.tile[ids]]
    self.cmd_goal[ids] = torch.clamp(-0.5 * torch.tanh(off / 10.0) + (2 * self._rand(n, 0.0, 1.0) - 1) * (hmax - 0.1), -hmax, hmax)
    frac = self.tile[ids].float() / max(self.n_tiles - 1, 1)
    gate_speed = 4.0 + 7.0 * frac + self._rand(n, -1.0, 1.5)
    # A quarter of free-skiing commands say stop (0 to 1.5 m/s): explicit braking practice, the skill the falls lack.
    stop = torch.rand(n, generator=self.gen, device=self.device) < 0.25
    free_speed = torch.where(stop, self._rand(n, 0.0, 1.5), self._rand(n, 3.0, 12.0))
    self.cmd_speed[ids] = torch.where(self.use_gates[ids], gate_speed, free_speed)
    self.cmd_timer[ids] = self._rand(n, 2.5, 5.0)

  def _place_gates(self, ids, spacing=None, amp=None):
    n = len(ids)
    k = torch.arange(self.G, device=self.device)[None]
    frac = (self.tile[ids].float() / max(self.n_tiles - 1, 1))[:, None]
    sp = self._rand(n, 10.0, 16.0)[:, None] if spacing is None else torch.full((n, 1), float(spacing), device=self.device)
    steep = ((self.tile_hmax[self.tile[ids]] - 0.9) / 0.4)[:, None]
    am = (1.0 + (0.5 + 3.0 * frac + 4.0 * steep) * torch.rand(n, 1, generator=self.gen, device=self.device)) if amp is None \
      else torch.full((n, 1), float(amp), device=self.device)
    side = torch.where(torch.rand(n, 1, generator=self.gen, device=self.device) < 0.5, -1.0, 1.0)
    gx = self.start_x[ids, None] + self._rand(n, 6.0, 10.0)[:, None] + sp * k
    gy = self.tile_y[self.tile[ids], None] + side * (1.0 - 2.0 * (k % 2)) * am + 0.5 * (torch.rand(n, self.G, generator=self.gen, device=self.device) - 0.5)
    # Keep gates clear of trees: push any gate within 3 m of a trunk sideways, away from that tree.
    trees = self.tile_trees[self.tile[ids]]  # (n, K, 3)
    yc = self.tile_y[self.tile[ids], None]
    for _ in range(3):
      d = torch.linalg.norm(torch.stack([gx, gy], -1)[:, :, None, :] - trees[:, None, :, :2], dim=-1) - CANOPY * trees[:, None, :, 2]
      dmin, j = d.min(-1)  # (n, G)
      ty = torch.gather(trees[..., 1], 1, j)
      away = torch.where(gy >= ty, 1.0, -1.0)
      gy = torch.where(dmin < 4.0, gy + away * (4.0 - dmin + 0.2), gy)
      gy = torch.clamp(gy, yc - 12.0, yc + 12.0)
    beyond = gx > self.tile_len - 12
    self.gate_x[ids] = torch.where(beyond, torch.full_like(gx, 1e6), gx)
    self.gate_y[ids] = torch.where(beyond, self.tile_y[self.tile[ids], None].expand_as(gy), gy)
    self.gate_idx[ids] = 0
    self.gates_hit[ids] = 0
    self.gates_crossed[ids] = 0

  def _next_gate(self):
    i = self.gate_idx.clamp(max=self.G - 1)
    ar = torch.arange(self.num_envs, device=self.device)
    return self.gate_x[ar, i], self.gate_y[ar, i], self.gate_idx < self.G

  def _update_gate_command(self, rate_limit=True):
    gx, gy, any_left = self._next_gate()
    ar = torch.arange(self.num_envs, device=self.device)
    j = (self.gate_idx + 1).clamp(max=self.G - 1)
    nx, ny = self.gate_x[ar, j], self.gate_y[ar, j]
    # Look ahead: inside the last 6 m before a gate, aim partly at the gate after it (a racing line, not a zig-zag).
    w = torch.clamp(1 - (gx - self.qpos[:, 0]) / 6.0, 0, 1) * 0.5 * (nx < 1e5).float()
    ax, ay = (1 - w) * gx + w * nx, (1 - w) * gy + w * ny
    hmax = self.tile_hmax[self.tile]
    bearing = torch.clamp(torch.atan2(ay - self.qpos[:, 1], torch.clamp(ax - self.qpos[:, 0], min=1.5)), -hmax, hmax)
    has_gate = self.use_gates & any_left & (gx < 1e5)
    # Free skiing stays on the piste: the goal heading bends back toward the middle from 8 m off center, fully by 14 m
    # (the tree lines start 17.5 m out).
    off = self.qpos[:, 1] - self.tile_y[self.tile]
    wall = -torch.sign(off) * torch.clamp((off.abs() - 8.0) / 6.0, 0, 1) * 1.2
    free = torch.clamp(self.cmd_goal + wall, -hmax, hmax)
    target = torch.where(has_gate, bearing, torch.where(self.use_gates, torch.zeros_like(bearing), free))
    target = torch.clamp(self._avoid_trees(target), -hmax - 0.3, hmax + 0.3)
    if rate_limit:  # the commanded heading turns at most 1.5 rad/s
      step = 1.5 * self.dt * self.decimation
      target = self.cmd_heading + torch.clamp(target - self.cmd_heading, -step, step)
    # A caller that pins the command (showcase and live set cmd_timer huge) keeps its own free-mode heading.
    auto = self.use_gates | (self.cmd_timer < 1e8)
    self.cmd_heading = torch.where(auto, target, self.cmd_heading)

  def set_style(self, style: str):
    """'default' or 'racing' (racer's-form rewards and the deeper RACING_DELTA stance as SONIC's target)."""
    self.style = style
    delta = torch.zeros_like(self._ref_base[0])
    if style == "racing":
      for j, v in RACING_DELTA.items():
        delta[JOINTS.index(j)] = v
    self.ref_neutral, self.ref_left, self.ref_right = (r + delta for r in self._ref_base)
    self.stance = self.ref_neutral
    self.ref_cur = self.ref_neutral.expand(self.num_envs, -1).clone()

  def _shield(self, horizon_s: float = 2.0):
    """Hard tree constraint on the heading command, whatever set it (training, showcase, live viewer): if the line
    the robot is actually travelling would bring it within a body width of a canopy in the next 2 s, the command
    turns 35 degrees away from that tree, on the side the tree is not, until the line is clear. The planner bends
    the commanded line early; this catches the robot when its real line drifts from the command."""
    v = self.qvel[:, 0:2]
    speed = v.norm(dim=-1)
    u = v / speed.clamp(min=0.5)[:, None]
    trees = self.tile_trees[self.tile]
    rel = trees[..., :2] - self.qpos[:, None, 0:2]
    along = (rel * u[:, None]).sum(-1)
    lat = rel[..., 1] * u[:, None, 0] - rel[..., 0] * u[:, None, 1]
    keep = CANOPY * trees[..., 2] + BODY_MARGIN + 0.5
    reach = (horizon_s * speed).clamp(min=4.0)[:, None]
    danger = (along > 0) & (along < reach) & (lat.abs() < keep)
    j = torch.where(danger, along, torch.full_like(along, 1e9)).argmin(-1, keepdim=True)
    l = lat.gather(1, j)[:, 0]
    travel = torch.atan2(v[:, 1], v[:, 0])
    safe = travel + torch.where(l > 0, -0.6, 0.6)
    self.cmd_heading = torch.where(danger.any(-1) & (speed > 1.0), safe, self.cmd_heading)
    self.shield_on = danger.any(-1) & (speed > 1.0)

  def _avoid_trees(self, heading, clear: float = 2.5):
    """Bend a heading around the first trunk on the line ahead: aim beside it on the side that needs the smaller
    turn. The batched twin of showcase.steer_around_trees; two passes so the new line is checked too. Looks about
    2.5 s ahead (12 to 35 m)."""
    look = torch.clamp(2.5 * self.qvel[:, 0:2].norm(dim=-1), 12.0, 35.0)[:, None]
    trees = self.tile_trees[self.tile]  # (N, K, 3)
    rel = trees[..., :2] - self.qpos[:, None, 0:2]
    need = CANOPY * trees[..., 2] + clear
    for _ in range(2):
      u = torch.stack([torch.cos(heading), torch.sin(heading)], -1)
      along = (rel * u[:, None]).sum(-1)
      lat = rel[..., 1] * u[:, None, 0] - rel[..., 0] * u[:, None, 1]  # + = tree left of the line
      block = (along > 1.0) & (along < look) & (lat.abs() < need)
      j = torch.where(block, along, torch.full_like(along, 1e9)).argmin(-1, keepdim=True)
      a, l, n = along.gather(1, j)[:, 0], lat.gather(1, j)[:, 0], need.gather(1, j)[:, 0]
      aim = l - torch.where(l > 0, n, -n)  # pass on the far side from the tree's offset
      heading = torch.where(block.any(-1), heading + torch.atan2(aim, torch.clamp(a, min=2.0)), heading)
    return heading

  def reset_idx(self, ids):
    if len(ids) == 0:
      return
    n = len(ids)
    rand_tile = torch.randint(0, self.n_tiles, (n,), generator=self.gen, device=self.device)
    explore = torch.rand(n, generator=self.gen, device=self.device) < self.explore_frac
    self.tile[ids] = torch.where(explore, rand_tile, self.level[ids])
    x = self._rand(n, 4.0, 12.0)
    y = self.tile_y[self.tile[ids]] + self._rand(n, -6.0, 6.0)
    heading = self._rand(n, -0.5, 0.5)
    _, nrm = self.grid.height_normal(x, y)
    fwd = torch.stack([torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)], -1)
    xa = fwd - (fwd * nrm).sum(-1, keepdim=True) * nrm
    xa = xa / xa.norm(dim=-1, keepdim=True)
    ya = torch.cross(nrm, xa, dim=-1)
    R = torch.stack([xa, ya, nrm], -1)
    pts = torch.einsum("nij,kj->nki", R, self.ski_pts_pelvis)
    h, _ = self.grid.height_normal(x[:, None] + pts[..., 0], y[:, None] + pts[..., 1])
    z = (h - pts[..., 2]).max(-1).values + 0.003
    self.qpos[ids, 0], self.qpos[ids, 1], self.qpos[ids, 2] = x, y, z
    self.qpos[ids, 3:7] = mat_to_quat(R)
    self.qpos[ids, 7:] = self.stance + 0.03 * (torch.rand(n, 29, generator=self.gen, device=self.device) - 0.5)
    self.qvel[ids] = 0
    v0 = self._rand(n, *self.spawn_speed)
    self.qvel[ids, 0:3] = xa * v0[:, None]
    self.ctrl[ids] = self.stance
    self.start_x[ids] = x
    # Snow for this tile, randomized +-15% per episode.
    for j, k in enumerate(("mu_glide", "mu_skid", "mu_carve", "k_normal")):
      self.snow[k][ids, 0] = self.tile_snow[self.tile[ids], j] * self._rand(n, 0.85, 1.15)
    self.episode_length_buf[ids] = 0
    self.last_action[ids] = 0
    self.prev_action[ids] = 0
    self.sonic_last[ids] = 0
    self.use_gates[ids] = torch.rand(n, generator=self.gen, device=self.device) < self.gate_frac
    self._resample_commands(ids)
    self._place_gates(ids)

  # ------------------------------------------------------------------------------------------------
  def _state(self):
    quat = self.qpos[:, 3:7]
    R = quat_to_mat(quat)
    ang = self.qvel[:, 3:6]  # body frame for a free joint
    grav = -R[:, 2, :]  # world -z expressed in the body frame = -(row 2 of R)
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

  def _sonic(self, lean):
    """lean: (N, 2) roll, pitch target for the pelvis relative to upright. Returns raw SONIC actions (IsaacLab order)."""
    R, *_ = self._state()
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    roll, pitch = lean[:, 0], lean[:, 1]
    cy, sy, cp, sp, cr, sr = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)
    Rref = torch.stack([
      cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
      sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
      -sp, cp * sr, cp * cr,
    ], -1).reshape(-1, 3, 3)
    M = R.transpose(1, 2) @ Rref
    anchor = torch.stack([M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1], M[:, 2, 0], M[:, 2, 1]], -1)
    # Reference posture: the racers' neutral stance, blended toward their left- or right-turn shape with the lean.
    turn = torch.clamp(roll / 0.35, -1, 1)[:, None]
    self.ref_cur = (self.ref_neutral + torch.relu(turn) * (self.ref_left - self.ref_neutral)
                    + torch.relu(-turn) * (self.ref_right - self.ref_neutral))
    ref_i = self.ref_cur[:, self.m2i]
    ref_block = torch.cat([ref_i.repeat(1, 10), torch.zeros(self.num_envs, 290, device=self.device)], -1)
    with torch.no_grad():
      token = self.enc(ref_block, anchor.repeat(1, 10))
      dec_in = torch.cat([token, self.hist_ang.flatten(1), self.hist_q.flatten(1), self.hist_dq.flatten(1),
                          self.hist_act.flatten(1), self.hist_grav.flatten(1)], -1)
      return self.dec(dec_in)

  def _apply_ski_forces(self):
    ids = self.ski_ids
    w = self.cvel[:, ids, :3]
    com = self.xipos[:, ids]
    v = self.cvel[:, ids, 3:] + torch.cross(w, com - self.subtree_com[:, self.ski_root], dim=-1)
    F, Tq, diag = self.contact.forces(self.grid, self.xpos[:, ids], self.xmat[:, ids], com, w, v, snow=self.snow)
    self.xfrc[:, ids, :3] = F
    self.xfrc[:, ids, 3:] = Tq
    pw = self.cvel[:, self.pelvis, :3]
    pv = self.cvel[:, self.pelvis, 3:] + torch.cross(pw, self.xipos[:, self.pelvis] - self.subtree_com[:, self.pelvis], dim=-1)
    self.xfrc[:, self.pelvis, :3] = drag(pv, self.p)
    self.ski_diag = diag

  # ------------------------------------------------------------------------------------------------
  def get_observations(self):
    critic = torch.cat([self.obs, self._privileged()], -1) if self.privileged_critic else self.obs
    return TensorDict({"actor": self.obs, "critic": critic}, batch_size=[self.num_envs])

  def _privileged(self):
    """Critic-only information the robot cannot sense: true snow, slope, gate offset, nearest trees, time left."""
    R, *_ = self._state()
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    def to_body(dx, dy):
      cc, ss = (c, s_) if dx.dim() == 1 else (c[:, None], s_[:, None])
      return cc * dx + ss * dy, -ss * dx + cc * dy
    snow = torch.cat([self.snow["mu_glide"] / 0.1, self.snow["mu_skid"] / 0.5, self.snow["mu_carve"] / 2.0,
                      self.snow["k_normal"] / 80000.0], -1)
    _, nrm = self.grid.height_normal(self.qpos[:, 0], self.qpos[:, 1])
    nx, ny = to_body(nrm[:, 0], nrm[:, 1])
    gx, gy, any_left = self._next_gate()
    has = (self.use_gates & any_left & (gx < 1e5)).float()
    gdx, gdy = to_body((gx - self.qpos[:, 0]) * has, (gy - self.qpos[:, 1]) * has)
    trees = self.tile_trees[self.tile]
    rel = trees[..., :2] - self.qpos[:, None, 0:2]
    d = rel.norm(dim=-1) - CANOPY * trees[..., 2]
    k = min(3, trees.shape[1])
    idx = d.topk(k, largest=False).indices
    near = torch.gather(rel, 1, idx[..., None].expand(-1, -1, 2))
    tdx, tdy = to_body(near[..., 0], near[..., 1])
    tr = torch.gather(trees[..., 2], 1, idx)
    tree_feat = torch.stack([torch.clamp(tdx / 15, -2, 2), torch.clamp(tdy / 15, -2, 2), tr], -1).flatten(1)
    if k < 3:
      tree_feat = torch.cat([tree_feat, torch.zeros(self.num_envs, 3 * (3 - k), device=self.device)], -1)
    t_frac = (self.episode_length_buf.float() / self.max_episode_length)[:, None]
    priv = torch.cat([snow, torch.stack([nx, ny, nrm[:, 2]], -1), torch.stack([gdx / 20, gdy / 20, has], -1), tree_feat, t_frac], -1)
    return torch.nan_to_num(priv, 0.0, 2.0, -2.0)

  def _tree_scan(self, yaw):
    """Ray distances (0..1 of RAY_RANGE) to the nearest tree trunk in each direction, and the closest clearance."""
    trees = self.tile_trees[self.tile]  # (N, K, 3)
    rel = trees[..., :2] - self.qpos[:, None, 0:2]  # (N, K, 2)
    r = trees[..., 2]
    ang = yaw[:, None] + self.ray_angles[None]  # (N, R)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (N, R, 2)
    proj = torch.einsum("nrd,nkd->nrk", u, rel)
    perp2 = (rel**2).sum(-1)[:, None, :] - proj**2
    disc = (r[:, None, :] + RAY_INFLATE) ** 2 - perp2  # inflated trunks: nothing in range fits between two rays
    t = proj - torch.sqrt(torch.clamp(disc, min=0))
    t = torch.where((disc > 0) & (t > 0), t, torch.full_like(t, RAY_RANGE))
    self.tree_rays = torch.clamp(t.min(-1).values, max=RAY_RANGE) / RAY_RANGE
    return self.tree_rays

  def _tree_clearance(self, xy):
    """Distance from points (N, P, 2) to the nearest trunk surface on each env's tile: (N, P)."""
    trees = self.tile_trees[self.tile]
    d = torch.linalg.norm(xy[:, :, None, :] - trees[:, None, :, :2], dim=-1) - CANOPY * trees[:, None, :, 2]
    return d.min(-1).values

  def _observations(self):
    R, ang, grav, q, dq = self._state()
    vb = (R.transpose(1, 2) @ self.qvel[:, 0:3, None])[..., 0]
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    hx = torch.tensor(HEIGHT_X, device=self.device)
    hy = torch.tensor(HEIGHT_Y, device=self.device)
    gx, gy = torch.meshgrid(hx, hy, indexing="ij")
    gx, gy = gx.flatten(), gy.flatten()
    px = self.qpos[:, 0:1] + c[:, None] * gx - s[:, None] * gy
    py = self.qpos[:, 1:2] + s[:, None] * gx + c[:, None] * gy
    h, _ = self.grid.height_normal(px, py)
    heights = torch.clamp(h - self.qpos[:, 2:3], -3, 1)
    d = self.ski_diag
    if d is None:
      ski = torch.zeros(self.num_envs, 16, device=self.device)
    else:
      ski = torch.cat([d["edge"], d["N"] / 200, d["v_t"] * 0.1, d["v_l"], d["zones"].flatten(1) / 175], -1)
    dh = wrap(self.cmd_heading - yaw)
    obs = torch.cat([
      ang * 0.25, grav, vb * 0.2, q, dq * 0.05, self.last_action, ski, heights,
      torch.stack([torch.sin(dh), torch.cos(dh), self.cmd_speed / 10], -1),
      self._tree_scan(yaw),
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
    lean = actions[:, :2] * 0.35
    residual = actions[:, 2:] * 0.5
    self._refresh_history()
    total = self._sonic(lean) + residual
    self.sonic_last = total
    self.ctrl[:] = self.default + total[:, self.i2m] * self.scale
    for _ in range(self.decimation):
      self._apply_ski_forces()
      self._physics_step()
    self.episode_length_buf += 1
    self.cmd_timer -= self.dt * self.decimation
    resample = ((self.cmd_timer <= 0) & ~self.use_gates).nonzero().flatten()
    if len(resample):
      self._resample_commands(resample)
    gx, gy, any_left = self._next_gate()
    crossed = self.use_gates & any_left & (self.qpos[:, 0] >= gx)
    hit = crossed & ((self.qpos[:, 1] - gy).abs() < 1.5)
    self.gate_idx += crossed.long()
    self.gates_hit += hit.float()
    self.gates_crossed += crossed.float()
    self._update_gate_command()
    self._shield()

    # Rewards.
    R, ang, grav, q, dq = self._state()
    v = self.qvel[:, 0:3]
    speed = v[:, :2].norm(dim=-1)
    travel = torch.atan2(v[:, 1], v[:, 0])
    herr = wrap(travel - self.cmd_heading)
    moving = torch.clamp(speed / 2.0, max=1.0)
    # Ski style: parallel, both on the snow, hip-width apart, edges agreeing in a turn.
    sx = self.xmat[:, self.ski_ids, :, 0]  # ski forward axes (N, 2, 3)
    ski_yaw = torch.atan2(sx[..., 1], sx[..., 0])
    yaw_gap = wrap(ski_yaw[:, 0] - ski_yaw[:, 1])
    rel = (R.transpose(1, 2)[:, None] @ (self.xpos[:, self.ski_ids] - self.qpos[:, None, 0:3])[..., None])[..., 0]
    width = rel[:, 0, 1] - rel[:, 1, 1]
    d = self.ski_diag
    airborne = (d["N"] < 15).float().sum(-1)
    e = d["edge"]
    edged = (e.abs() > math.radians(5)).all(-1)
    disagree = (edged & (torch.sign(e[:, 0]) != torch.sign(e[:, 1]))).float()
    z = d["zones"]  # (N, 2 skis, 4): front-left, front-right, rear-left, rear-right normal force
    front, rear = z[..., 0:2].sum((-1, -2)), z[..., 2:4].sum((-1, -2))
    rear_frac = rear / (front + rear + 1e-6)
    hw = torch.clamp(self.cmd_speed / 3.0, 0.25, 1.0)
    on_snow = ((front + rear) > 50.0).float()
    r = {
      "ski_parallel": -1.5 * torch.clamp(yaw_gap.abs() - 0.12, min=0) ** 2 * 10,
      "ski_contact": -0.6 * airborne,
      "ski_width": -20.0 * torch.clamp((width - 0.24).abs() - 0.08, min=0) ** 2,
      "edge_agree": -0.4 * disagree,
      # A stop command frees the heading (weight down to 0.25), so it can brake the way skiers do: turn across the hill.
      "heading": 1.5 * torch.exp(-herr**2 / 0.15) * moving * hw,
      # A sharp second term (sigma 8 deg): the broad one pays 80% at 13 deg off, which misses a 1.5 m gate from 8 m.
      "heading_fine": 1.0 * torch.exp(-herr**2 / 0.02) * moving * hw,
      # Weight on the tails (back seat) is how it falls backward at speed: penalize rear-heavy pressure.
      "fore_aft": -1.0 * torch.clamp(rear_frac - 0.6, min=0) * on_snow,
      "gates": (3.0 * hit.float() - 3.0 * (crossed & ~hit).float()) / (self.dt * self.decimation),
      # Progress along the commanded heading, full marks at the commanded speed (it used to pay for speed up to 14 m/s).
      "toward_target": 0.5 * torch.clamp((v[:, 0] * torch.cos(self.cmd_heading) + v[:, 1] * torch.sin(self.cmd_heading))
                                         / torch.clamp(self.cmd_speed, min=2.0), 0, 1) * (self.cmd_speed > 2.0).float(),
      "speed": 0.5 * torch.exp(-(speed - self.cmd_speed) ** 2 / 8.0),
      # Falls happen at a median 12 m/s against 8.6 m/s for clean runs, so speed control comes first. Slope 0.4 per
      # m/s near the limit, saturating at -1.2: an unbounded penalty made every fast state worse than falling, and
      # the policy learned to fall on purpose (fall rate 0.57 -> 0.80 in 450 iterations).
      "overspeed": -1.2 * torch.tanh(torch.clamp(speed - self.cmd_speed - 1.5, min=0) / 3.0),
      "alive": 0.5 * torch.ones_like(speed),
      "upright": -0.2 * (grav[:, :2] ** 2).sum(-1),
      "action_rate": -0.01 * ((self.last_action - self.prev_action) ** 2).sum(-1),
      "residual": -0.02 * (residual**2).sum(-1),
      # Quiet hands: arms stay near the ski stance (shoulders, elbows, wrists are MuJoCo joints 15..28).
      # Light touch: arms may still balance, they just shouldn't flail (0.3 with a velocity term froze them and broke balance).
      "arms_quiet": -0.05 * ((self.qpos[:, 7 + 15 :] - self.ref_cur[:, 15:]) ** 2).sum(-1),
      "athletic": -0.15 * ((self.qpos[:, 7 : 7 + 15] - self.ref_cur[:, :15]) ** 2).sum(-1),
      "joint_vel": -1e-4 * (dq**2).sum(-1),
    }

    if self.style == "racing":
      # Racer's form. Stance and quiet hands weigh more; the body leans into each turn by as much as the turn's
      # force asks (a racer's inclination, atan(a_lat / g)); edges carve instead of skidding while racing.
      r["athletic"] = r["athletic"] * (0.4 / 0.15)
      r["arms_quiet"] = r["arms_quiet"] * (0.15 / 0.05)
      yaw_rate = (R @ ang[..., None])[:, 2, 0]
      a_lat = speed * yaw_rate
      left = torch.stack([-v[:, 1], v[:, 0], torch.zeros_like(speed)], -1) / speed.clamp(min=0.5)[:, None]
      incl = torch.asin(torch.clamp((R[:, :, 2] * left).sum(-1), -1, 1))
      fast = (speed > 4.0).float()
      r["angulation"] = 0.6 * torch.exp(-(incl - torch.atan(a_lat / 9.81)) ** 2 / 0.03) * fast
      slip = (d["v_l"].abs() / (d["v_t"].abs() + 1.0)).mean(-1)
      r["carve"] = -0.5 * torch.clamp(slip - 0.15, min=0) * fast * (self.cmd_speed > 2.0).float()

    # Terminations.
    hp, _ = self.grid.height_normal(self.qpos[:, 0], self.qpos[:, 1])
    fb = self.xpos[:, self.fall_ids]
    hb, _ = self.grid.height_normal(fb[..., 0], fb[..., 1])
    fell = (R[:, 2, 2] < 0.45) | (self.qpos[:, 2] - hp < 0.4) | (fb[..., 2] - hb < 0.05).any(-1) | ~torch.isfinite(self.qpos).all(-1)
    # Tree strike: pelvis or either ski within 15 cm of a trunk.
    body_xy = torch.cat([self.qpos[:, None, 0:2], self.xpos[:, self.ski_ids, 0:2]], 1)
    clearance = self._tree_clearance(body_xy)
    clr = clearance.min(-1).values
    hit_tree = clr < BODY_MARGIN
    fell = fell | hit_tree
    self.last_tree_hit = hit_tree
    # Barrier: grows fast near a canopy (-0.75 at 1.25 m, -3 at contact), so close shaves never pay.
    r["tree_near"] = -3.0 * torch.clamp(1 - clr / 2.5, min=0) ** 2
    off_tile = ((self.qpos[:, 1] - self.tile_y[self.tile]).abs() > 22) | (self.qpos[:, 0] > self.tile_len - 8)
    timeout = (self.episode_length_buf >= self.max_episode_length) | off_tile
    r["termination"] = -100.0 * fell.float()  # well above the value of an episode, so a fall never pays
    reward = sum(r.values()) * (self.dt * self.decimation)
    reward = torch.nan_to_num(reward, 0.0, 0.0, 0.0)
    for k, val in r.items():  # kept on the device; read out only when logging
      m = val.mean().detach()
      self.rew_ema[k] = m if k not in self.rew_ema else 0.99 * self.rew_ema[k] + 0.01 * m

    self.last_fell = fell
    if not self.auto_reset:  # showcase runs: the caller decides what happens after a fall; no tile limits
      self.obs = self._observations()
      return self.get_observations(), reward, torch.zeros_like(fell), {"time_outs": torch.zeros_like(fell), "log": {}}
    done = fell | timeout
    ids = done.nonzero().flatten()
    extras = {"time_outs": timeout & ~fell, "log": {}}
    if len(ids):
      dist = self.qpos[ids, 0] - self.start_x[ids]
      up = (dist > 100) | (timeout[ids] & ~fell[ids])
      # Any fall short of the 100 m goal moves down (it used to need a fall inside 25 m, so levels kept rising while
      # 80% of runs ended in a fall). Now the frontier sits where about half the runs succeed.
      down = fell[ids] & (dist <= 100)
      self.level[ids] = torch.clamp(self.level[ids] + up.long() - down.long(), 0, self.n_tiles - 1)
      extras["log"] = {
        "Episode/fell_frac": fell[ids].float().mean().item(),
        "Episode/distance_m": dist.mean().item(),
        "Episode/length_s": (self.episode_length_buf[ids].float() * self.dt * self.decimation).mean().item(),
        "Curriculum/mean_level": self.level.float().mean().item(),
        "Task/speed_mps": speed.mean().item(),
        "Task/heading_err_deg": math.degrees(herr.abs().mean().item()),
        "Episode/tree_crash_frac": hit_tree[ids].float().mean().item(),
        "Task/shield_frac": self.shield_on.float().mean().item(),
        **dict(zip((f"Reward/{k}" for k in self.rew_ema), torch.stack(list(self.rew_ema.values())).tolist())),
      }
      g = ids[self.use_gates[ids] & (self.gates_crossed[ids] > 0)]
      if len(g):
        extras["log"]["Gates/hit_rate"] = (self.gates_hit[g].sum() / self.gates_crossed[g].sum()).item()
        extras["log"]["Gates/hits_per_episode"] = self.gates_hit[g].mean().item()
      self.reset_idx(ids)
      self._forward()
      self._refresh_history(ids, fill=True)
      snap = self.cmd_heading.clone()
      self._update_gate_command(rate_limit=False)
      keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
      keep[ids] = False
      self.cmd_heading = torch.where(keep, snap, self.cmd_heading)
    self.obs = self._observations()
    return self.get_observations(), reward, done, extras
