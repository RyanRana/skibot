"""GEAR-SONIC's kinematic planner (planner_sonic.onnx) on CPU: walk/run/careful commands to a 50 Hz reference
that the SONIC controller tracks. Built to match the C++ deploy stack (docs/source/references/planner_onnx.md in
NVlabs/GR00T-WholeBodyControl):

  context  4 qpos frames (36 = root xyz, wxyz quat, 29 joints in MuJoCo order) sampled at 30 Hz from the current
           plan, starting 2 frames (50 Hz) ahead of the frame being played
  output   up to 64 frames at 30 Hz in the world frame, resampled to 50 Hz (lerp joints and root, slerp quats)
  blend    the new plan starts at that look-ahead frame and fades in over 8 frames
Weights: NVIDIA Open Model License (fetched by scripts/fetch_assets.sh into assets/sonic_planner).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skisim.sonic import DEFAULT_ANGLES, Reference

ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "assets" / "sonic_planner" / "planner_sonic.onnx"

MODES = {"idle": 0, "slowWalk": 1, "walk": 2, "run": 3, "squat": 4, "happy": 17, "stealth": 18, "injured": 19,
         "careful": 20, "objectCarrying": 21, "crouch": 22}
LOOK_AHEAD = 2
BLEND = 8
STAND_Z = 0.793


def _slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Row-wise slerp between wxyz quaternions a and b (N, 4) at fractions t (N,)."""
  d = np.sum(a * b, axis=1)
  b = np.where(d[:, None] < 0, -b, b)
  d = np.abs(d)
  th = np.arccos(np.clip(d, -1, 1))
  s = np.sin(th)
  small = s < 1e-6
  wa = np.where(small, 1 - t, np.sin((1 - t) * th) / np.where(small, 1, s))
  wb = np.where(small, t, np.sin(t * th) / np.where(small, 1, s))
  q = wa[:, None] * a + wb[:, None] * b
  return q / np.linalg.norm(q, axis=1, keepdims=True)


def _sample(frames: np.ndarray, t: np.ndarray) -> np.ndarray:
  """qpos frames (T, 36) at fractional indices t: lerp position and joints, slerp the root quaternion."""
  t = np.clip(t, 0, len(frames) - 1)
  i0 = np.floor(t).astype(int)
  i1 = np.minimum(i0 + 1, len(frames) - 1)
  f = t - i0
  out = frames[i0] * (1 - f[:, None]) + frames[i1] * f[:, None]
  out[:, 3:7] = _slerp(frames[i0, 3:7], frames[i1, 3:7], f)
  return out


def yaw_quat(yaw: float) -> np.ndarray:
  return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


class Planner:
  def __init__(self, path: Path = PLANNER, threads: int = 0, seed: int = 0):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
      opts.intra_op_num_threads = threads
    self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    self.inputs = {i.name: i for i in self.sess.get_inputs()}
    self.k = self.inputs["allowed_pred_num_tokens"].shape[1] if "allowed_pred_num_tokens" in self.inputs else 0
    self.rng = np.random.default_rng(seed)
    self.plan: np.ndarray | None = None  # (T, 36) at 50 Hz, frame 0 = the frame SONIC plays at tick 0

  def _run(self, context: np.ndarray, mode: int, direction, facing, speed: float) -> np.ndarray:
    feed = {
      "context_mujoco_qpos": context[None].astype(np.float32),
      "target_vel": np.array([speed if speed > 0 else -1.0], np.float32),
      "mode": np.array([mode], np.int64),
      "movement_direction": np.array([direction], np.float32),
      "facing_direction": np.array([facing], np.float32),
      "height": np.array([-1.0], np.float32),
      "random_seed": np.array([int(self.rng.integers(1 << 30))], np.int64),
      "has_specific_target": np.zeros((1, 1), np.int64),
      "specific_target_positions": np.zeros((1, 4, 3), np.float32),
      "specific_target_headings": np.zeros((1, 4), np.float32),
    }
    if self.k:
      mask = np.zeros((1, self.k), np.int64)
      mask[0, -1] = 1  # always the longest horizon (16 tokens = 64 frames at 30 Hz), so 0.5 s replans never run dry
      feed["allowed_pred_num_tokens"] = mask
    feed = {k: v for k, v in feed.items() if k in self.inputs}
    qpos, n = self.sess.run(None, feed)
    out30 = qpos[0, : int(np.ravel(n)[0])].astype(float)
    t50 = np.arange(0, (len(out30) - 1) * 50 / 30 + 1e-9) * 30 / 50
    return _sample(out30, t50)

  def start(self, xy, yaw: float, joints: np.ndarray = DEFAULT_ANGLES) -> Reference:
    """Standing context at the robot's position and heading, first plan in idle (like Initialize())."""
    frame = np.concatenate([[xy[0], xy[1], STAND_Z], yaw_quat(yaw), joints])
    self.plan = self._run(np.tile(frame, (4, 1)), MODES["idle"], (0, 0, 0), (np.cos(yaw), np.sin(yaw), 0), -1)
    return self.reference()

  def replan(self, tick: int, mode: str | int, direction, facing=None, speed: float = -1.0) -> Reference:
    """New plan from the frame SONIC is playing (tick), blended in from tick + LOOK_AHEAD. Returns the rebased
    reference; the caller resets SONIC's tick to 0."""
    mode = MODES.get(mode, mode) if isinstance(mode, str) else mode
    d = np.array([direction[0], direction[1], 0.0])
    f = d if facing is None else np.array([facing[0], facing[1], 0.0])
    gen = tick + LOOK_AHEAD
    ctx = _sample(self.plan, gen + np.arange(4) * 50 / 30)
    new = self._run(ctx, mode, d, f, speed)
    old = self.plan[tick:]
    if len(old) < LOOK_AHEAD + BLEND:
      old = np.concatenate([old, np.repeat(self.plan[-1:], LOOK_AHEAD + BLEND - len(old), 0)])
    out = np.concatenate([old[:LOOK_AHEAD], new])
    for i in range(BLEND):
      w = i / BLEND
      a, b = old[LOOK_AHEAD + i], new[i]
      out[LOOK_AHEAD + i] = a * (1 - w) + b * w
      out[LOOK_AHEAD + i, 3:7] = _slerp(a[None, 3:7], b[None, 3:7], np.array([w]))[0]
    self.plan = out
    return self.reference()

  def reference(self) -> Reference:
    return Reference(self.plan[:, 7:36], self.plan[:, 3:7], loop=False)
