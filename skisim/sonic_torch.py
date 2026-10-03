"""SONIC's g1 encoder and decoder rebuilt in torch from the ONNX weights, batched for GPU training.

Encoder (g1 mode): input = per future frame [58 values from the joint pos/vel block reshaped (10, 58),
6 anchor-orientation values] -> 640 -> MLP (SiLU) -> 64 -> FSQ quantizer in 2 groups of 32.
Decoder: 994 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29, SiLU. Checked against onnxruntime in __main__.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn

SONIC_DIR = Path(__file__).resolve().parents[1] / "assets" / "sonic"


def _consts(graph):
  out = {}
  for n in graph.node:
    if n.op_type == "Constant":
      for a in n.attribute:
        if a.name == "value":
          out[n.output[0]] = numpy_helper.to_array(a.t)
  return out


class Decoder(nn.Module):
  def __init__(self, path=SONIC_DIR / "model_decoder.onnx"):
    super().__init__()
    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consts = _consts(g)
    layers = []
    for n in g.node:
      if n.op_type == "MatMul":
        w = next(init[i] for i in n.input if i in init)
        layers.append([torch.tensor(w.T.copy()), None])
      elif n.op_type == "Add" and layers and layers[-1][1] is None:
        b = next((init.get(i, consts.get(i)) for i in n.input if i in init or i in consts), None)
        layers[-1][1] = torch.tensor(np.asarray(b).copy())
    self.layers = nn.ModuleList()
    for w, b in layers:
      lin = nn.Linear(w.shape[1], w.shape[0])
      lin.weight.data.copy_(w)
      lin.bias.data.copy_(b)
      self.layers.append(lin)

  def forward(self, x):
    for i, lin in enumerate(self.layers):
      x = lin(x)
      if i < len(self.layers) - 1:
        x = x * torch.sigmoid(x)
    return x


class G1Encoder(nn.Module):
  def __init__(self, path=SONIC_DIR / "model_encoder.onnx"):
    super().__init__()
    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consts = _consts(g)
    self.layers = nn.ModuleList()
    for k in (0, 2, 4, 6, 8):
      w = init[f"module.encoders.g1.module.{k}.weight"]
      b = init[f"module.encoders.g1.module.{k}.bias"]
      lin = nn.Linear(w.shape[1], w.shape[0])
      lin.weight.data.copy_(torch.tensor(w))
      lin.bias.data.copy_(torch.tensor(b))
      self.layers.append(lin)
    q = {}
    for n in g.node:
      if n.name.startswith("/quantizer/") and n.op_type in ("Add", "Mul", "Sub", "Div"):
        c = next((consts[i] for i in n.input if i in consts and consts[i].shape == (32,)), None)
        if c is not None and n.op_type not in q:
          q[n.op_type] = torch.tensor(c.copy())
    self.register_buffer("q_shift", q["Add"])
    self.register_buffer("q_half", q["Mul"])
    self.register_buffer("q_offset", q["Sub"])
    self.register_buffer("q_width", q["Div"])

  def forward(self, jointblock_580, anchor_60):
    """jointblock_580: (B, 580) = 10 frames of ref joint pos then 10 frames of ref joint vel (IsaacLab order).
    anchor_60: (B, 60) = 10 frames x 6 anchor-orientation values."""
    B = jointblock_580.shape[0]
    x = torch.cat([jointblock_580.reshape(B, 10, 58), anchor_60.reshape(B, 10, 6)], -1).reshape(B, 640)
    for i, lin in enumerate(self.layers):
      x = lin(x)
      if i < len(self.layers) - 1:
        x = x * torch.sigmoid(x)
    z = x.reshape(B, 2, 32)
    bounded = torch.tanh(z + self.q_shift) * self.q_half - self.q_offset
    quant = bounded + (torch.round(bounded) - bounded).detach()
    return (quant / self.q_width).reshape(B, 64)


def encoder_inputs_from_obs(obs_1762: torch.Tensor):
  """Slice the deploy-layout 1762 vector the way the ONNX graph does (it drops element 0, the mode id)."""
  return obs_1762[:, 4:584], obs_1762[:, 601:661]


if __name__ == "__main__":
  import onnxruntime as ort

  rng = np.random.default_rng(0)
  dec, enc = Decoder().eval(), G1Encoder().eval()
  so = ort.InferenceSession(str(SONIC_DIR / "model_decoder.onnx"), providers=["CPUExecutionProvider"])
  se = ort.InferenceSession(str(SONIC_DIR / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
  worst_d = worst_e = 0.0
  mism = 0
  for _ in range(20):
    xd = rng.normal(0, 0.5, (1, 994)).astype(np.float32)
    ref = so.run(None, {so.get_inputs()[0].name: xd})[0]
    with torch.no_grad():
      got = dec(torch.tensor(xd)).numpy()
    worst_d = max(worst_d, float(np.abs(ref - got).max()))
    xe = np.zeros((1, 1762), np.float32)
    xe[0, 4:584] = rng.normal(0, 0.6, 580)
    xe[0, 601:661] = np.tile([1, 0, 0, 1, 0, 0], 10) + rng.normal(0, 0.1, 60)
    ref = se.run(None, {se.get_inputs()[0].name: xe})[0]
    with torch.no_grad():
      got = enc(*encoder_inputs_from_obs(torch.tensor(xe))).numpy()
    worst_e = max(worst_e, float(np.abs(ref - got).max()))
    mism += int((np.abs(ref - got) > 1e-4).sum())
  print(f"decoder max abs diff vs onnxruntime: {worst_d:.2e}")
  print(f"encoder max abs diff vs onnxruntime: {worst_e:.2e}  (token entries differing > 1e-4: {mism} of {20 * 64})")
