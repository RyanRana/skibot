"""NVIDIA GEAR-SONIC driving our G1 in CPU MuJoCo, built to match its C++ deploy reference.

Contract (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref, policy_parameters.hpp and g1_deploy_onnx_ref.cpp):
  50 Hz control. Decoder input 994 = token 64 + 10-frame history (oldest first) of
  pelvis angular velocity (body frame, 3), joint positions minus default angles (29, IsaacLab order),
  joint velocities (29, IsaacLab order), last raw actions (29, IsaacLab order), gravity in body frame (3).
  Decoder output: 29 raw actions in IsaacLab order; target (MuJoCo order) = default + action * scale,
  scale = 0.25 * effort_limit / kp. Encoder input 1762, here in "g1" mode (id 0): future reference joint
  positions and velocities (absolute, IsaacLab order, 10 frames 5 ticks apart) and reference root
  orientation relative to the pelvis (first two columns of the rotation matrix, row-major).
Weights: NVIDIA Open Model License (assets/sonic/LICENSE). Code reference: Apache 2.0.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from skisim.scene import _J, _W, _ZETA, _EFFORT

ROOT = Path(__file__).resolve().parents[1]
SONIC_DIR = ROOT / "assets" / "sonic"

# MuJoCo joint order used by SONIC (same as Unitree's g1.xml).
JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# isaaclab_to_mujoco[i]: IsaacLab index of MuJoCo joint i. mujoco_to_isaaclab[i]: MuJoCo index of IsaacLab joint i.
I2M = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
M2I = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
LOWER_BODY_ISAAC = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
WRISTS_ISAAC = [23, 24, 25, 26, 27, 28]

DEFAULT_ANGLES = np.array([
  -0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
  0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0,
])
# Motor group per MuJoCo joint and gain multiplier (hip pitch uses 7520_22 in SONIC; ankles and waist roll/pitch are 2x 5020).
_GROUP = (["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2
          + ["7520_14", "5020", "5020"] + (["5020"] * 5 + ["4010"] * 2) * 2)
_MULT = ([1, 1, 1, 1, 2, 2] * 2 + [1, 2, 2] + [1] * 14)


def sonic_gains():
  kp = np.array([_J[g] * m * _W**2 for g, m in zip(_GROUP, _MULT)])
  kd = np.array([2 * _ZETA * _J[g] * m * _W for g, m in zip(_GROUP, _MULT)])
  effort = np.array([_EFFORT[g] * m for g, m in zip(_GROUP, _MULT)])
  armature = np.array([_J[g] * m for g, m in zip(_GROUP, _MULT)])
  # Action scale uses single-motor stiffness and effort, as written in policy_parameters.hpp.
  kp1 = np.array([_J[g] * _W**2 for g in _GROUP])
  scale = 0.25 * np.array([_EFFORT[g] for g in _GROUP]) / kp1
  return kp, kd, effort, armature, scale


def _quat_conj(q):
  return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def _quat_mat(q):
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, q)
  return m.reshape(3, 3)


def heading_quat(q):
  """Yaw-only part of a wxyz quaternion."""
  R = _quat_mat(q)
  yaw = np.arctan2(R[1, 0], R[0, 0])
  return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


class Reference:
  """A reference motion in MuJoCo joint order at 50 Hz: joint positions (T, 29) and root quats (T, 4)."""

  def __init__(self, qpos: np.ndarray, root_quat: np.ndarray, loop: bool = True):
    self.q = np.asarray(qpos, float)
    self.quat = np.asarray(root_quat, float)
    self.dq = np.gradient(self.q, 1 / 50, axis=0) if len(self.q) > 1 else np.zeros_like(self.q)
    self.loop = loop

  @classmethod
  def static(cls, joints_mj: np.ndarray, frames: int = 50):
    return cls(np.tile(joints_mj, (frames, 1)), np.tile([1.0, 0, 0, 0], (frames, 1)))

  def frame(self, t: int) -> int:
    T = len(self.q)
    return t % T if self.loop else min(t, T - 1)


class SonicController:
  def __init__(self, model: mujoco.MjModel, prefix: str = "", root_joint: str = "floating_base_joint",
               pelvis: str = "pelvis", set_gains: bool = True):
    import onnxruntime as ort  # CPU playback only; training uses skisim.sonic_torch

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    self.enc = ort.InferenceSession(str(SONIC_DIR / "model_encoder.onnx"), opts, providers=["CPUExecutionProvider"])
    self.dec = ort.InferenceSession(str(SONIC_DIR / "model_decoder.onnx"), opts, providers=["CPUExecutionProvider"])
    self.model = model
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j) for j in JOINTS]
    if min(jid) < 0:
      raise ValueError("model is missing a G1 joint")
    self.qadr = model.jnt_qposadr[jid]
    self.dadr = model.jnt_dofadr[jid]
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + root_joint)
    self.root_q, self.root_d = model.jnt_qposadr[root], model.jnt_dofadr[root]
    act_by_joint = {model.actuator_trnid[a, 0]: a for a in range(model.nu)}
    self.act = np.array([act_by_joint[j] for j in jid])
    self.kp, self.kd, self.effort, armature, self.scale = sonic_gains()
    if set_gains:
      model.actuator_gainprm[self.act, 0] = self.kp
      model.actuator_biasprm[self.act, 1] = -self.kp
      model.actuator_biasprm[self.act, 2] = -self.kd
      model.actuator_forcerange[self.act] = np.stack([-self.effort, self.effort], 1)
      model.dof_armature[self.dadr] = armature
    self.hist = deque(maxlen=10)
    self.last_action = np.zeros(29)
    self.tick = 0
    self.ref: Reference | None = None
    self.heading_align = np.array([1.0, 0, 0, 0])
    self.token = np.zeros(64)

  # --- state ------------------------------------------------------------------------------------
  def _state(self, data):
    q = data.qpos[self.qadr]
    dq = data.qvel[self.dadr]
    quat = data.qpos[self.root_q + 3 : self.root_q + 7].copy()
    ang = data.qvel[self.root_d + 3 : self.root_d + 6].copy()  # free joint angular velocity is body-frame
    R = _quat_mat(quat)
    grav = R.T @ np.array([0.0, 0.0, -1.0])
    return {
      "ang": ang,
      "q": (q - DEFAULT_ANGLES)[M2I],
      "dq": dq[M2I],
      "act": self.last_action.copy(),
      "grav": grav,
      "quat": quat,
    }

  def reset(self, data, ref: Reference):
    self.ref = ref
    self.tick = 0
    self.last_action[:] = 0
    s = self._state(data)
    self.hist.clear()
    for _ in range(10):
      self.hist.append(s)
    # Align the reference heading with the robot's current heading, like ComputeApplyDeltaHeading.
    self.heading_align = _quat_mul(heading_quat(s["quat"]), _quat_conj(heading_quat(ref.quat[0])))

  # --- observations --------------------------------------------------------------------------------
  def encoder_obs(self, base_quat):
    ref = self.ref
    fut = [ref.frame(self.tick + 5 * k) for k in range(10)]
    near = [ref.frame(self.tick + k) for k in range(10)]
    qi = ref.q[:, M2I]
    dqi = ref.dq[:, M2I]
    def anchor(f):
      r = _quat_mul(self.heading_align, ref.quat[f])
      M = _quat_mat(_quat_mul(_quat_conj(base_quat), r))
      return np.array([M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[2, 0], M[2, 1]])
    parts = [
      np.zeros(4),  # encoder mode 0 = g1, then zeros
      qi[fut].ravel(), dqi[fut].ravel(),
      np.zeros(10), np.zeros(1),  # root z (not used by the g1 encoder)
      anchor(fut[0]), np.concatenate([anchor(f) for f in fut]),
      qi[fut][:, LOWER_BODY_ISAAC].ravel(), dqi[fut][:, LOWER_BODY_ISAAC].ravel(),
      np.zeros(9), np.zeros(12),  # VR 3-point
      np.zeros(720), np.zeros(60),  # SMPL
      qi[near][:, WRISTS_ISAAC].ravel(),
    ]
    obs = np.concatenate(parts)
    assert obs.size == 1762, obs.size
    return obs

  def decoder_obs(self):
    h = list(self.hist)
    obs = np.concatenate([
      self.token,
      np.concatenate([e["ang"] for e in h]),
      np.concatenate([e["q"] for e in h]),
      np.concatenate([e["dq"] for e in h]),
      np.concatenate([e["act"] for e in h]),
      np.concatenate([e["grav"] for e in h]),
    ])
    assert obs.size == 994, obs.size
    return obs

  # --- control ------------------------------------------------------------------------------------
  def step(self, data, token: np.ndarray | None = None, residual: np.ndarray | None = None) -> np.ndarray:
    """One 50 Hz tick: log state, encode the reference (or use a given token), decode, write ctrl."""
    s = self._state(data)
    self.hist.append(s)
    if token is None:
      enc_in = self.encoder_obs(s["quat"]).astype(np.float32)[None]
      token = self.enc.run(None, {self.enc.get_inputs()[0].name: enc_in})[0][0]
    self.token = np.asarray(token, float)
    dec_in = self.decoder_obs().astype(np.float32)[None]
    action = self.dec.run(None, {self.dec.get_inputs()[0].name: dec_in})[0][0].astype(float)
    if residual is not None:
      action = action + residual
    self.last_action = action
    target = DEFAULT_ANGLES + action[I2M] * self.scale
    data.ctrl[self.act] = target
    self.tick += 1
    return target
