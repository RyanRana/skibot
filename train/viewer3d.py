"""Fast 3D viewer for the live G1 skier: the server streams poses, the browser draws the scene with three.js.

Run: PYTHONPATH=. .venv-train/bin/python -m train.viewer3d     then open http://127.0.0.1:8766 (or your LAN IP on a phone)

- Physics is the exact training env on CPU MuJoCo (backend="mujoco"), paced to real time, newest checkpoint pulled
  from the Modal volume every 60 s (train.live's sync_loop). auto_reset is off: a crash holds on screen for 1.5 s and
  respawns at the top; the end of a tile fades out and respawns at the top.
- Each 50 Hz frame is about 1 KB of binary: world position and quaternion of the 32 bodies with visual geoms (skis
  included), the two ski soles for the tracks, and a few HUD numbers. The robot meshes are sent once (/model.bin),
  terrain and trees once per tile. The page and the websocket (/ws) share the port.

Modes:
  python -m train.viewer3d                                        live, training tiles (gates or free steer)
  python -m train.viewer3d --course out/courses/bormio-stelvio     live run down a real course through its gates
  python -m train.viewer3d --record out/viewer3d/run.npz           record the live run (the page has a Rec button too)
  python -m train.viewer3d --replay out/viewer3d/run.npz           play a recording back, no physics
  python -m train.viewer3d --bake out/viewer3d/bormio.npz --course out/courses/bormio-stelvio --seconds 150
                                                                   record a run offline as fast as the CPU allows
"""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import gzip
import hashlib
import json
import logging
import math
import os
import queue
import signal
import struct
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from websockets.asyncio.server import broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from skisim.scene import g1_model
from skisim.ski import SkiParams
from skisim.terrain import HeightGrid, slope

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web3d"
OUT = ROOT / "out" / "viewer3d"
PORT = 8766
DT = 0.02  # one policy step (50 Hz)
LABELS = ["Start", "Setting off", "Gliding", "Carving left", "Carving right", "Edge change", "Skid check", "Air", "Landing",
          "Crash"]
# Frame header: kind, flags, label, nbody, seq, t, server ms, disc, next gate, gates hit, gates crossed, run, speed,
# fade, commanded heading. Then nbody x (px py pz qw qx qy qz) and 2 skis x (x y z load) as float32.
HEAD = struct.Struct("<BBBBIddIHHHHfff")
F_GATES, F_CRASH, F_REC, F_REPLAY, F_COURSE = 1, 2, 4, 8, 16
CRASH_HOLD, FADE_OUT, FADE_IN = 1.5, 0.45, 0.4  # s
log = logging.getLogger("viewer3d")


# ---------------------------------------------------------------------------------------------------------------------
# Robot model for the browser


def _cluster(v: np.ndarray, f: np.ndarray, cell: float):
  """Vertex-clustering decimation: merge vertices in the same `cell` box, drop faces that collapse."""
  if cell <= 0:
    return v.astype(np.float32), f
  keys, inv = np.unique(np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True)
  inv = inv.reshape(-1)
  nv = np.zeros((len(keys), 3))
  np.add.at(nv, inv, v)
  nv /= np.bincount(inv, minlength=len(keys))[:, None]
  ff = inv[f]
  ff = ff[(ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])]
  _, keep = np.unique(np.sort(ff, 1), axis=0, return_index=True)
  return nv.astype(np.float32), ff[np.sort(keep)]


_GEOM = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
         int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
         int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}


def envelope(kind: int, meta: dict, payload: bytes = b"") -> bytes:
  """Binary message: u8 kind, 3 pad, u32 json length (padded to 4), JSON, then the payload the JSON points into."""
  j = json.dumps(meta, separators=(",", ":")).encode()
  j += b" " * (-len(j) % 4)
  return struct.pack("<BxxxI", kind, len(j)) + j + payload


class Blob:
  def __init__(self):
    self.parts, self.n = [], 0

  def add(self, arr: np.ndarray) -> int:
    b = np.ascontiguousarray(arr).tobytes()
    b += b"\0" * (-len(b) % 4)
    self.parts.append(b)
    off, self.n = self.n, self.n + len(b)
    return off

  def bytes(self) -> bytes:
    return b"".join(self.parts)


def visual_model(m: mujoco.MjModel, cluster: float = 0.002):
  """Visual geoms (groups 0-2) of every body, with their meshes decimated to `cluster` m. Returns (body ids, bytes)."""
  blob, meshes, bodies = Blob(), {}, []
  for b in range(1, m.nbody):
    geoms = []
    for g in np.flatnonzero(m.geom_bodyid == b):
      t = int(m.geom_type[g])
      if m.geom_group[g] > 2 or t not in _GEOM:
        continue
      mat = int(m.geom_matid[g])
      rgba = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]).round(4).tolist()
      if rgba[3] <= 0:
        continue
      e = {"type": _GEOM[t], "size": m.geom_size[g].round(6).tolist(), "pos": m.geom_pos[g].round(6).tolist(),
           "quat": m.geom_quat[g].round(6).tolist(), "rgba": rgba,
           "mat": (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat) or "") if mat >= 0 else ""}
      if _GEOM[t] == "mesh":
        k = int(m.geom_dataid[g])
        if k not in meshes:
          va, nv, fa, nf = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
          v, f = _cluster(m.mesh_vert[va:va + nv].astype(np.float64), m.mesh_face[fa:fa + nf], cluster)
          meshes[k] = {"nv": len(v), "nf": len(f), "v": blob.add(v.astype("<f4")), "f": blob.add(f.astype("<u2"))}
        e["mesh"] = k
      geoms.append(e)
    if geoms:
      bodies.append({"id": b, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), "geoms": geoms})
  meta = {"type": "model", "bodies": bodies, "meshes": {str(k): v for k, v in meshes.items()}, "labels": LABELS}
  return [b["id"] for b in bodies], [b["name"] for b in bodies], envelope(3, meta, blob.bytes())


# ---------------------------------------------------------------------------------------------------------------------
# Terrain and trees for the browser


def _smooth_noise(shape, feature_cells: float, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  fy = np.fft.fftfreq(shape[0])[:, None]
  fx = np.fft.fftfreq(shape[1])[None, :]
  n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * np.exp(-2.0 * (np.hypot(fx, fy) * feature_cells) ** 2)))
  return (n - n.mean()) / (n.std() + 1e-9)


def _smoothstep(x):
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3 - 2 * x)


def crop_to(grid: HeightGrid, k: int, rows: tuple[int, int] | None = None) -> HeightGrid:
  """Sub-grid whose row and column spans are multiples of k cells, so a k-times coarser grid shares its edge vertices."""
  r0, r1 = rows if rows else (0, grid.nrow - 1)
  r1 = r0 + (r1 - r0) // k * k
  c1 = (grid.ncol - 1) // k * k
  return HeightGrid(z=np.asarray(grid.z[r0:r1 + 1, :c1 + 1], float), x0=grid.x0, y0=grid.y0 + r0 * grid.cell, cell=grid.cell)


def extend_terrain(near: HeightGrid, k: int, margin: tuple[float, float, float], walls: tuple[float, float] | None,
                   seed: int = 0) -> HeightGrid:
  """Coarse terrain (cell k * near.cell) around the real patch, so it sits in a mountain valley instead of floating.

  It shares near's edge vertices exactly (the browser cuts a hole where near is and stitches near's border to it).
  Outside: near's smoothed edge plane extrapolated outward (damped), valley walls rising away from the run axis
  (walls = (y axis, rise per m)), and smooth noise that grows with distance from the patch.
  """
  c = near.cell * k
  xmin, xmax, ymin, ymax = near.bounds()
  up, down, side = margin
  nx0, nx1, ny = math.ceil(up / c), math.ceil(down / c), math.ceil(side / c)
  ncol = nx0 + round((xmax - xmin) / c) + nx1 + 1
  nrow = 2 * ny + round((ymax - ymin) / c) + 1
  X0, Y0 = xmin - nx0 * c, ymin - ny * c
  X, Y = np.meshgrid(X0 + c * np.arange(ncol), Y0 + c * np.arange(nrow))
  cx, cy = np.clip(X, xmin, xmax), np.clip(Y, ymin, ymax)
  h, _ = near.height_normal(cx, cy)
  smooth = HeightGrid(z=gaussian_filter(near.z, 6.0 / near.cell, mode="nearest"), x0=near.x0, y0=near.y0, cell=near.cell)
  _, n = smooth.height_normal(cx, cy)
  dx, dy = X - cx, Y - cy
  dist = np.hypot(dx, dy)
  Z = h + (-n[..., 0] / n[..., 2] * dx - n[..., 1] / n[..., 2] * dy) / (1.0 + dist / 160.0)
  if walls is not None:
    axis, rise = walls
    d = np.clip(np.abs(Y - axis) - (ymax - ymin) / 2, 0, None)
    Z += rise * d + 0.0016 * d**2
  Z += 9.0 * _smoothstep(dist / 220.0) * _smooth_noise(Z.shape, 40.0 / c, seed)
  Z += 1.6 * _smoothstep(dist / 50.0) * _smooth_noise(Z.shape, 9.0 / c, seed + 1)
  Z = Z + (gaussian_filter(Z, 1.2, mode="nearest") - Z) * _smoothstep(dist / (3 * c))
  inside = dist == 0
  Z[inside] = h[inside]  # exact on the shared border; the interior is cut out by the browser
  return HeightGrid(z=Z, x0=X0, y0=Y0, cell=c)


def decor_trees(far: HeightGrid, near: HeightGrid, keep_clear, seed: int, spacing: float = 5.5, max_n: int = 4000):
  """Scenery pines on the far terrain and the near patch away from the run: rows (x, y, radius, height)."""
  rng = np.random.default_rng(seed)
  x0, x1, y0, y1 = far.bounds()
  X, Y = np.meshgrid(np.arange(x0 + spacing / 2, x1, spacing), np.arange(y0 + spacing / 2, y1, spacing))
  X = X + rng.uniform(-0.45, 0.45, X.shape) * spacing
  Y = Y + rng.uniform(-0.45, 0.45, Y.shape) * spacing
  forest = _smooth_noise(X.shape, 7.0, seed + 5)  # patches of forest with clearings
  keep = rng.random(X.shape) < _smoothstep((forest + 0.6) / 1.4) * 0.9
  keep &= ~keep_clear(X, Y)
  X, Y = X[keep], Y[keep]
  nx0, nx1, ny0, ny1 = near.bounds()
  on_near = (X >= nx0) & (X <= nx1) & (Y >= ny0) & (Y <= ny1)
  _, n_far = far.height_normal(X, Y)
  _, n_near = near.height_normal(X, Y)
  nz = np.where(on_near, n_near[:, 2], n_far[:, 2])
  ok = nz > np.cos(np.radians(36))  # no trees on cliffs
  X, Y = X[ok], Y[ok]
  cx, cy = (nx0 + nx1) / 2, (ny0 + ny1) / 2
  order = np.argsort(np.hypot(np.clip(np.abs(X - cx) - (nx1 - nx0) / 2, 0, None), np.clip(np.abs(Y - cy) - (ny1 - ny0) / 2, 0, None)))
  X, Y = X[order][:max_n], Y[order][:max_n]
  r = rng.uniform(0.55, 1.0, len(X))
  hgt = rng.uniform(4.0, 8.5, len(X)) * (0.9 + 0.2 * rng.random(len(X)))
  return np.stack([X, Y, r, hgt], 1).astype(np.float32)


def ground(near: HeightGrid, far: HeightGrid, x, y):
  x, y = np.asarray(x, float), np.asarray(y, float)
  nx0, nx1, ny0, ny1 = near.bounds()
  on = (x >= nx0) & (x <= nx1) & (y >= ny0) & (y <= ny1)
  return np.where(on, near.height_normal(x, y)[0], far.height_normal(x, y)[0])


def world_message(wid: int, name: str, kind: str, near: HeightGrid, far: HeightGrid, k: int, trees: np.ndarray,
                  decor: np.ndarray, fall: tuple[float, float], extra: dict | None = None) -> bytes:
  """Terrain (near + far grids) and trees as one binary message. Trees: x, y, ground z, radius, height, real (1/0)."""
  rows = []
  for arr, real in ((trees, 1.0), (decor, 0.0)):
    if len(arr):
      z = ground(near, far, arr[:, 0], arr[:, 1])
      rows.append(np.column_stack([arr[:, 0], arr[:, 1], z, arr[:, 2], arr[:, 3], np.full(len(arr), real)]))
  T = np.concatenate(rows).astype("<f4") if rows else np.zeros((0, 6), "<f4")
  blob = Blob()
  grid = lambda g, off: {"x0": g.x0, "y0": g.y0, "cell": g.cell, "nrow": g.nrow, "ncol": g.ncol, "offset": off}
  meta = {"type": "world", "id": wid, "name": name, "kind": kind, "k": k,
          "near": grid(near, blob.add(near.z.astype("<f4"))), "far": grid(far, blob.add(far.z.astype("<f4"))),
          "trees": {"count": len(T), "offset": blob.add(T), "stride": 6}, "fall": [float(fall[0]), float(fall[1])]}
  meta.update(extra or {})
  return envelope(2, meta, blob.bytes())


def tile_world(wid: int, Z: np.ndarray, meta: dict, tile: int, trees_all: np.ndarray) -> tuple[bytes, HeightGrid]:
  t = meta["tiles"][tile]
  yc, cell = float(t["y_center_m"]), float(meta["cell"])
  full = HeightGrid(z=Z, x0=meta["x0"], y0=meta["y0"], cell=cell)
  r0 = max(0, int(math.floor((yc - 27 - meta["y0"]) / cell)))
  r1 = min(Z.shape[0] - 1, int(math.ceil((yc + 27 - meta["y0"]) / cell)))
  k = 2
  near = crop_to(full, k, (r0, r1))
  far = extend_terrain(near, k, (150.0, 340.0, 230.0), walls=(yc, 0.16), seed=tile)
  ny0, ny1 = near.bounds()[2:]
  clear = lambda X, Y: (np.abs(Y - yc) < 29.0) | ((Y > ny0 - 2) & (Y < ny1 + 2) & (X > -4) & (X < near.bounds()[1] + 4))
  trees = trees_all[trees_all[:, 4] == tile][:, :4] if len(trees_all) else np.zeros((0, 4), np.float32)
  decor = decor_trees(far, near, clear, seed=1000 + tile)
  name = f"Tile {tile}: {t.get('source', '')}"
  extra = {"tile": tile, "yc": yc, "slope": t.get("mean_slope_deg"), "snow": t.get("snow", "groomed")}
  return world_message(wid, name, "tile", near, far, k, trees, decor, (1.0, 0.0), extra), near


def course_world(wid: int, name: str, grid: HeightGrid, line: np.ndarray, trees: np.ndarray) -> bytes:
  k = 4
  near = crop_to(grid, k)
  far = extend_terrain(near, k, (480.0, 480.0, 480.0), walls=None, seed=7)
  kd = cKDTree(line[:, :2])
  clear = lambda X, Y: kd.query(np.stack([X.ravel(), Y.ravel()], 1), distance_upper_bound=30.0)[0].reshape(X.shape) < 30.0
  decor = decor_trees(far, near, clear, seed=11, spacing=6.0, max_n=6000)
  d = line[min(10, len(line) - 1), :2] - line[0, :2]
  fall = d / (np.linalg.norm(d) + 1e-9)
  return world_message(wid, name, "course", near, far, k, trees, decor, (fall[0], fall[1]), {"course": name})


def gates_message(gates: list, next_gate: int = 0) -> str:
  return json.dumps({"type": "gates", "gates": gates, "next": next_gate}, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------------------------------
# Websocket hub (runs on its own asyncio thread)


class Hub:
  def __init__(self, model_bytes: bytes, hello: dict):
    self.loop = asyncio.new_event_loop()
    self.clients = set()
    self.cmds: queue.Queue = queue.Queue()
    self.model = model_bytes
    self.model_gz = gzip.compress(model_bytes, 6)
    self.model_tag = '"' + hashlib.sha1(model_bytes).hexdigest()[:16] + '"'
    self.hello = hello
    self.world = None
    self.gates = None
    self.hud = None
    self.lat = deque(maxlen=1000)  # s, from frame ready on the sim thread to bytes handed to every socket
    self.size = 0
    self.sent = 0

  # Called from the sim thread.
  def post(self, fn, *a, **kw):
    self.loop.call_soon_threadsafe(lambda: fn(*a, **kw))

  def frame(self, data: bytes, t_ready: float):
    live = [c for c in self.clients if c.transport is not None and c.transport.get_write_buffer_size() < 200_000]
    broadcast(live, data)  # a client that falls behind skips frames instead of queueing latency
    self.lat.append(time.perf_counter() - t_ready)
    self.size = len(data)
    self.sent += 1

  def set_world(self, data: bytes):
    self.world = data
    broadcast(self.clients, data)

  def set_gates(self, text: str):
    self.gates = text
    broadcast(self.clients, text)

  def set_hud(self, text: str):
    self.hud = text
    broadcast(self.clients, text)

  def send_to(self, ws, text: str):
    broadcast([ws], text)

  def set_hello(self, **kw):
    self.hello.update(kw)

  # Websocket side.
  async def handler(self, ws):
    try:
      await ws.send(json.dumps(self.hello))
      if self.world is not None:
        await ws.send(self.world)
      if self.gates is not None:
        await ws.send(self.gates)
      if self.hud is not None:
        await ws.send(self.hud)
      self.clients.add(ws)
      async for msg in ws:
        if isinstance(msg, bytes):
          continue
        try:
          d = json.loads(msg)
        except ValueError:
          continue
        if d.get("cmd") == "ping":  # answered right here, without waiting for the sim thread
          await ws.send(json.dumps({"type": "pong", "t": d.get("t"), "srv": time.time() * 1000}))
        else:
          self.cmds.put((ws, d))
    except Exception as e:  # noqa: BLE001 - a dropped phone connection must never take the server down
      if "close" not in type(e).__name__.lower():
        log.debug("client error: %s", e)
    finally:
      self.clients.discard(ws)

  def process_request(self, connection, request):
    path = request.path.split("?", 1)[0]
    if path == "/ws":
      return None
    hdr = [("Date", email.utils.formatdate(usegmt=True)), ("Connection", "close")]

    def reply(status, reason, body: bytes, ctype: str, extra=()):
      h = Headers(hdr + [("Content-Length", str(len(body))), ("Content-Type", ctype)] + list(extra))
      return Response(status, reason, h, body)

    if path == "/model.bin":
      if request.headers.get("If-None-Match") == self.model_tag:
        return Response(304, "Not Modified", Headers(hdr + [("ETag", self.model_tag), ("Content-Length", "0")]), b"")
      gz = "gzip" in (request.headers.get("Accept-Encoding") or "")
      body = self.model_gz if gz else self.model
      extra = [("ETag", self.model_tag), ("Cache-Control", "no-cache")] + ([("Content-Encoding", "gzip")] if gz else [])
      return reply(200, "OK", body, "application/octet-stream", extra)
    if path == "/api/state":
      return reply(200, "OK", (self.hud or "{}").encode(), "application/json", [("Cache-Control", "no-store")])
    name = "index.html" if path in ("/", "") else path.lstrip("/")
    f = (WEB / name).resolve()
    if WEB.resolve() in f.parents and f.is_file():
      ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css",
               ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json"}.get(f.suffix, "application/octet-stream")
      return reply(200, "OK", f.read_bytes(), ctype, [("Cache-Control", "no-cache")])
    return reply(404, "Not Found", b"not found", "text/plain")

  def start(self, host: str, port: int):
    ready = threading.Event()

    def run():
      asyncio.set_event_loop(self.loop)

      async def main():
        async with serve(self.handler, host, port, process_request=self.process_request, compression=None,
                         max_size=2**20, ping_interval=20, ping_timeout=60):
          ready.set()
          await asyncio.Future()

      self.loop.run_until_complete(main())

    threading.Thread(target=run, daemon=True, name="ws").start()
    if not ready.wait(10):
      raise RuntimeError(f"could not start the server on port {port}")


# ---------------------------------------------------------------------------------------------------------------------
# Frames, recording and replay


class Frame:
  """The state behind one 50 Hz frame, packed for the wire or stacked into a recording."""

  __slots__ = ("flags", "label", "disc", "gate_next", "hits", "crossed", "run", "speed", "fade", "heading", "pose", "ski")

  def pack(self, seq: int, t: float) -> bytes:
    head = HEAD.pack(1, self.flags, self.label, len(self.pose), seq & 0xFFFFFFFF, t, time.time() * 1000.0, self.disc & 0xFFFFFFFF,
                     min(self.gate_next, 65535), min(self.hits, 65535), min(self.crossed, 65535), self.run & 0xFFFF,
                     self.speed, self.fade, self.heading)
    return head + self.pose.tobytes() + self.ski.tobytes()


class Recorder:
  FIELDS = ("flags", "label", "disc", "gate_next", "hits", "crossed", "run", "speed", "fade", "heading")

  def __init__(self, path: Path, body_names: list[str], meta: dict):
    self.path = Path(path)
    self.names = body_names
    self.meta = meta
    self.rows = {k: [] for k in self.FIELDS}
    self.pose, self.ski, self.t = [], [], []
    self.worlds, self.gates = [], []  # (frame index, payload)

  @property
  def n(self) -> int:
    return len(self.t)

  def add(self, f: Frame, t: float):
    for k in self.FIELDS:
      self.rows[k].append(getattr(f, k))
    self.pose.append(f.pose)
    self.ski.append(f.ski)
    self.t.append(t)

  def world(self, data: bytes):
    self.worlds.append((self.n, data))

  def gate_list(self, text: str):
    self.gates.append((self.n, text))

  def save(self) -> Path:
    if not self.t:
      return self.path
    self.path.parent.mkdir(parents=True, exist_ok=True)
    dtypes = {"flags": np.uint8, "label": np.uint8, "disc": np.uint32, "gate_next": np.uint16, "hits": np.uint16,
              "crossed": np.uint16, "run": np.uint16, "speed": np.float32, "fade": np.float32, "heading": np.float32}
    arrays = {k: np.asarray(v, dtypes[k]) for k, v in self.rows.items()}
    arrays["t"] = np.asarray(self.t, np.float64) - self.t[0]
    arrays["pose"] = np.stack(self.pose).astype(np.float32)  # (frames, bodies, 7): x y z qw qx qy qz, world frame
    arrays["ski"] = np.stack(self.ski).astype(np.float32)  # (frames, 2, 4): ski sole x y z and snow load N
    for i, (at, data) in enumerate(self.worlds):
      arrays[f"world_{i}"] = np.frombuffer(data, np.uint8)
    arrays["world_at"] = np.asarray([w[0] for w in self.worlds], np.int64)
    arrays["gates_at"] = np.asarray([g[0] for g in self.gates], np.int64)
    arrays["gates_json"] = np.asarray(json.dumps([g[1] for g in self.gates]))
    arrays["body_names"] = np.asarray(json.dumps(self.names))
    arrays["meta"] = np.asarray(json.dumps({**self.meta, "frames": self.n, "seconds": round(self.n * DT, 2), "labels": LABELS,
                                            "saved": time.strftime("%Y-%m-%d %H:%M:%S")}))
    tmp = self.path.with_name(self.path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(self.path)
    return self.path


class Replay:
  """Plays a recording: one recorded frame per 20 ms at 1x, worlds and gates re-sent where they changed."""

  def __init__(self, path: Path):
    d = np.load(path, allow_pickle=False)
    self.path = Path(path)
    self.meta = json.loads(str(d["meta"]))
    self.names = json.loads(str(d["body_names"]))
    self.pose, self.ski = d["pose"], d["ski"]
    self.cols = {k: d[k] for k in Recorder.FIELDS}
    self.n = len(self.pose)
    self.worlds = [(int(at), d[f"world_{i}"].tobytes()) for i, at in enumerate(d["world_at"])]
    self.gates = list(zip((int(a) for a in d["gates_at"]), json.loads(str(d["gates_json"]))))
    self.pos = 0.0
    self.speed = 1.0
    self.paused = False
    self.disc = 1
    self.last = -1

  def _latest(self, items, i):
    best = None
    for at, payload in items:
      if at <= i:
        best = payload
    return best

  def seek(self, frac: float):
    self.pos = float(np.clip(frac, 0, 1)) * (self.n - 1)
    self.last = -1
    self.disc += 1

  def frame_at(self, i: int, order: list[int] | None) -> Frame:
    f = Frame()
    for k in Recorder.FIELDS:
      v = self.cols[k][i]
      setattr(f, k, float(v) if k in ("speed", "fade", "heading") else int(v))
    f.flags = (f.flags & ~F_REC) | F_REPLAY
    f.disc = self.disc * 100003 + f.disc
    pose = self.pose[i] if order is None else self.pose[i][order]
    f.pose, f.ski = np.ascontiguousarray(pose, "<f4"), np.ascontiguousarray(self.ski[i], "<f4")
    return f


# ---------------------------------------------------------------------------------------------------------------------
# Live simulation


class MoveNamerFallback:
  """Same rules as train.showcase.MoveNamer (used only if that import fails mid-edit)."""

  def __init__(self):
    self.edge_sign_hist = deque(maxlen=12)
    self.air_t, self.since_air, self.label = 0.0, 9.0, "Start"

  def update(self, N, edge, v_side, yaw_rate, speed, fell, dt):
    if fell:
      self.label = "Crash"
      return self.label
    if N.max() < 15:
      self.air_t += dt
      self.since_air = 0.0
    else:
      if self.air_t > 0.08:
        self.since_air = 0.0
      self.air_t = 0.0
      self.since_air += dt
    e = float(np.mean(edge))
    self.edge_sign_hist.append(int(np.sign(e)) if abs(e) > np.radians(6) else 0)
    changed = len({s for s in self.edge_sign_hist if s != 0}) > 1
    if self.air_t > 0.08:
      self.label = "Air"
    elif self.since_air < 0.4:
      self.label = "Landing"
    elif changed and abs(e) < np.radians(14):
      self.label = "Edge change"
    elif abs(e) > np.radians(10) and np.sign(edge[0]) == np.sign(edge[1]) and abs(yaw_rate) > 0.25:
      self.label = "Carving left" if e > 0 else "Carving right"
    elif np.mean(np.abs(v_side)) > 0.8 and speed > 2:
      self.label = "Skid check"
    elif speed > 3:
      self.label = "Gliding"
    else:
      self.label = "Setting off"
    return self.label


def move_namer():
  try:
    from train.showcase import MoveNamer

    return MoveNamer()
  except Exception as e:  # noqa: BLE001
    log.warning("train.showcase.MoveNamer unavailable (%s); using the local copy", e)
    return MoveNamerFallback()


class _Shared:
  def __init__(self):
    self.lock = threading.Lock()
    self.heading, self.speed, self.amp, self.gates = 0.0, 9.0, 3.0, True
    self.tile, self.n_tiles, self.reset = 10, 24, True
    self.state, self.ckpt_path, self.ckpt_iter = {}, None, None


_LIVE = None


def live_module():
  """train.live (its Shared state and Modal sync_loop), or a stand-in without sync if it cannot be imported."""
  global _LIVE
  if _LIVE is None:
    try:
      from train import live as m

      _LIVE = m
    except Exception as e:  # noqa: BLE001
      log.warning("train.live unavailable (%s): no checkpoint sync", e)
      import types

      _LIVE = types.SimpleNamespace(S=_Shared())
  return _LIVE


def local_checkpoints() -> list[tuple[Path, int]]:
  """Complete model_N.pt files under runs_live/, newest run first and highest iteration first within a run.

  A checkpoint still being downloaded is not a valid zip yet (torch saves zip archives), so it is skipped.
  """
  out = []
  for run in sorted((p for p in (ROOT / "runs_live").glob("*") if p.is_dir()), reverse=True):
    its = sorted(((int(p.stem.split("_")[-1]), p) for p in run.glob("model_*.pt") if p.stem.split("_")[-1].isdigit()), reverse=True)
    out += [(p, it) for it, p in its if zipfile.is_zipfile(p)]
  return out


def newest_local_checkpoint() -> tuple[Path, int] | None:
  found = local_checkpoints()
  return found[0] if found else None


def _avoid_trees(xy, heading: float, trees: np.ndarray, look: float = 18.0, clear: float = 2.5) -> float:
  """Bend a heading around the first trunk on the line ahead (same rule as the env's own free-ski steering)."""
  if trees is None or len(trees) == 0:
    return heading
  rel = trees[:, :2] - np.asarray(xy)[None]
  need = 1.6 * trees[:, 2] + clear  # canopy as drawn (ski_env.CANOPY)
  for _ in range(2):
    u = np.array([math.cos(heading), math.sin(heading)])
    along = rel @ u
    lat = rel[:, 1] * u[0] - rel[:, 0] * u[1]
    block = (along > 1.0) & (along < look) & (np.abs(lat) < need)
    if not block.any():
      break
    j = np.flatnonzero(block)[np.argmin(along[block])]
    aim = lat[j] - (need[j] if lat[j] > 0 else -need[j])
    heading += math.atan2(aim, max(along[j], 2.0))
  return heading


class Live:
  """The training env on CPU MuJoCo, driven by the newest policy; one call to tick() is one 50 Hz step."""

  def __init__(self, args, hub: Hub, body_ids: list[int], body_names: list[str]):
    import torch

    from train.ski_env import SkiEnv

    live_mod = live_module()

    torch.set_num_threads(args.threads)
    self.torch = torch
    self.args, self.hub = args, hub
    self.S = live_mod.S  # train.live's shared state; its sync_loop writes S.ckpt_path / S.ckpt_iter
    self.live_mod = live_mod
    self.course_dir = Path(args.course) if args.course else None
    if self.course_dir is not None:
      mosaic_dir = self._init_course()
    else:
      self.terrain = Path(args.terrain)
      self.mosaic = json.loads((self.terrain / "mosaic.json").read_text())
      self.Z = np.load(self.terrain / "mosaic_elevation.npy").astype(float)
      tf = self.terrain / "trees.npy"
      self.trees_all = np.load(tf) if tf.exists() else np.zeros((0, 5), np.float32)
      mosaic_dir = self.terrain
    self.env = SkiEnv(num_envs=1, device="cpu", use_graph=False, seed=args.seed, mosaic=mosaic_dir, backend="mujoco")
    self.env.set_style(args.style)  # racing checkpoints (s3 on) need the racing stance as SONIC's target
    env = self.env
    env.auto_reset = False  # the viewer decides what happens after a crash or at the end of the run
    env.explore_frac = 0.0  # respawn on the tile we show, never a random one
    names = [mujoco.mj_id2name(env.mjm, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_ids]
    assert names == body_names, "visual model and env bodies differ"
    self.bids = np.asarray(body_ids)
    self.kd = mujoco.MjData(env.mjm)
    self.ski_ids = env.ski_ids.tolist()
    self.sole = np.array([0.0, 0.0, -SkiParams().thickness / 2])
    self.ids = torch.tensor([0])
    self.S.n_tiles = env.n_tiles
    self.policy, self.loaded, self.iter = None, None, None
    self.failed: dict = {}
    self.world_id, self.world_tile, self.world_cache = 0, None, {}
    self.tile_trees_cache = {}
    self.completed = 0
    self.state, self.state_t = "run", 0.0
    self.disc, self.run, self.falls, self.tree_falls = 0, 0, 0, 0
    self.namer = move_namer()
    self.label = "Start"
    self.cmd = 0.0
    self.slow_t = 0.0
    self.t_run = 0.0
    self.start_x = 0.0
    self.next_gate, self.hits = 0, 0
    self.crash_s, self.crash_streak = None, 0
    self.gates_json = None
    self.sim_steps, self.step_ms = 0, deque(maxlen=200)
    if args.ckpt:
      self.S.ckpt_path, self.S.ckpt_iter = Path(args.ckpt), int(Path(args.ckpt).stem.split("_")[-1])
    else:
      found = newest_local_checkpoint()
      if found:
        self.S.ckpt_path, self.S.ckpt_iter = found
    if not args.no_sync and not args.ckpt and hasattr(live_mod, "sync_loop"):
      threading.Thread(target=live_mod.sync_loop, daemon=True, name="sync").start()
    self._load_policy()
    self.respawn(top=True)

  # -- setup -----------------------------------------------------------------------------------------------------------
  def _init_course(self):
    from skisim.trees import course_trees
    from train.showcase import Course, clear_gates_of_trees, course_mosaic

    args = self.args
    name = self.course_dir.name
    mdir = OUT / f"_mosaic_{name}"
    self.grid = course_mosaic(self.course_dir, mdir)
    self.course = Course(self.course_dir)
    trees = np.zeros((0, 5), np.float32) if args.no_trees else course_trees(self.course.line, in_run=args.trees_in_run)
    trees = clear_gates_of_trees(trees, self.course)
    np.save(mdir / "trees.npy", trees)
    self.course_trees = trees
    c = self.course
    if len(trees):  # keep every turn marker at least 3.5 m from a trunk (as train.showcase does)
      for k in range(len(c.aim)):
        for _ in range(4):
          dd = np.hypot(trees[:, 0] - c.aim[k, 0], trees[:, 1] - c.aim[k, 1]) - 1.6 * trees[:, 2]
          j = int(np.argmin(dd))
          if dd[j] >= 4.0:
            break
          away = c.aim[k] - trees[j, :2]
          c.aim[k] = c.aim[k] + away / (np.linalg.norm(away) + 1e-6) * (4.0 - dd[j] + 0.2)
    if args.gate_spacing > 0:
      c.respace(args.gate_spacing, args.gate_offset)
    self.course_world = course_world(1, name, self.grid, c.line, trees[:, :4])
    gates = []
    for k, g in enumerate(c.gates):
      poles = np.asarray(g["poles"], float)[:, :2]
      z = self.grid.height_normal(poles[:, 0], poles[:, 1])[0]
      gates.append({"c": 0 if g.get("color", "red") == "red" else 1,
                    "p": np.column_stack([poles, z]).round(3).tolist(), "panels": [[0, 1], [2, 3]] if len(poles) >= 4 else []})
    self.course_gates = gates
    return mdir

  def _load_policy(self):
    with self.S.lock:
      want, it = self.S.ckpt_path, self.S.ckpt_iter
    if want is None or want == self.loaded or time.time() < self.failed.get(want, 0):
      return
    from train.train import load_policy

    try:
      self.policy, self.loaded, self.iter = load_policy(self.env, str(want)), want, it
      print(f"now driving with {want}", flush=True)
    except Exception as e:  # noqa: BLE001 - keep the current policy, retry this file in 30 s
      print(f"policy load failed ({want}): {e}", flush=True)
      self.failed[want] = time.time() + 30
      if self.policy is None and not self.args.ckpt:  # nothing driving yet: fall back to an older complete file
        for p, k in local_checkpoints():
          if p != want and time.time() >= self.failed.get(p, 0):
            try:
              self.policy, self.loaded, self.iter = load_policy(self.env, str(p)), p, k
              print(f"now driving with {p} (fallback)", flush=True)
              return
            except Exception as e2:  # noqa: BLE001
              print(f"policy load failed ({p}): {e2}", flush=True)
              self.failed[p] = time.time() + 30

  @property
  def gates_mode(self) -> bool:
    return self.course_dir is None and bool(self.S.gates)

  # -- world, gates, respawn -------------------------------------------------------------------------------------------
  def _send_world(self):
    if self.course_dir is not None:
      if self.world_tile != "course":
        self.world_tile = "course"
        self.hub.post(self.hub.set_world, self.course_world)
        self.on_world(self.course_world)
      return
    tile = int(self.S.tile)
    if tile == self.world_tile:
      return
    if tile not in self.world_cache:
      self.world_id += 1
      self.world_cache = {k: v for k, v in list(self.world_cache.items())[-3:]}
      self.world_cache[tile] = tile_world(self.world_id, self.Z, self.mosaic, tile, self.trees_all)[0]
    msg = self.world_cache[tile]
    self.world_tile = tile
    self.hub.post(self.hub.set_world, msg)
    self.on_world(msg)

  def on_world(self, msg: bytes):  # hook for the recorder
    pass

  def on_gates(self, text: str):  # hook for the recorder
    pass

  def _send_gates(self):
    if self.course_dir is not None:
      gates = self.course_gates
    elif self.gates_mode:
      gx, gy = self.env.gate_x[0].numpy(), self.env.gate_y[0].numpy()
      gates = []
      for k in range(len(gx)):
        if gx[k] > 1e5:
          break
        ys = gy[k] + np.array([-2.05, -1.5, 1.5, 2.05])
        z = self.env.grid.height_normal(self.torch.full((4,), float(gx[k])), self.torch.tensor(ys, dtype=self.torch.float32))[0].numpy()
        gates.append({"c": k % 2, "p": [[round(float(gx[k]), 3), round(float(y), 3), round(float(zz), 3)] for y, zz in zip(ys, z)],
                      "panels": [[0, 1], [2, 3]]})
    else:
      gates = []
    text = gates_message(gates)
    self.gates_json = text
    self.hub.post(self.hub.set_gates, text)
    self.on_gates(text)

  def respawn(self, top: bool = True):
    env, ids, S = self.env, self.ids, self.S
    with S.lock:
      gates_mode, amp, tile = bool(S.gates), float(S.amp), int(S.tile)
    self._send_world()
    if self.course_dir is not None:
      from train.showcase import spawn

      c = self.course
      if top:
        s0, self.crash_s, self.crash_streak = 3.0, None, 0
      else:  # back 4 m from the crash; the third crash at the same spot skips 12 m past it, so a run always finishes
        s_crash = c.s_of(env.qpos[0, 0:2].numpy())
        near = self.crash_s is not None and abs(s_crash - self.crash_s) < 12.0
        self.crash_streak = self.crash_streak + 1 if near else 1
        self.crash_s = s_crash if not near else self.crash_s
        s0 = s_crash - 4.0 if self.crash_streak < 3 else s_crash + 12.0
        if self.crash_streak >= 3:
          print(f"crash loop at {s_crash:.0f} m along the course: skipping to {s0:.0f} m", flush=True)
          self.crash_s, self.crash_streak = None, 0
        s0 = float(np.clip(s0, 3.0, c.length - 20.0))
      spawn(env, c.point_at(s0), c.heading_at(s0), 2.0 if top else 4.0)
      self.cmd = c.heading_at(s0)
      if top:
        self.next_gate, self.hits = 0, 0
    else:
      env.level[0] = tile
      env.gate_frac = 1.0 if gates_mode else 0.0
      env.reset_idx(ids)
      env.tile[0] = tile
      env._forward()
      env._refresh_history(ids, fill=True)
      env.use_gates[0] = gates_mode
      env._place_gates(ids, amp=amp)
      env._update_gate_command(rate_limit=False)
      env.obs = env._observations()
      self.cmd = float(env.cmd_heading[0])
    self.start_x = float(env.qpos[0, 0])
    self.namer = move_namer()
    self.label = "Start"
    self.state, self.state_t, self.slow_t, self.t_run = "run", 0.0, 0.0, 0.0
    self.disc += 1
    self.run += 1
    self._send_gates()

  # -- commands from the page ------------------------------------------------------------------------------------------
  def command(self, d: dict):
    S, c = self.S, d.get("cmd")
    with S.lock:
      if c == "steer":
        if "heading" in d:
          S.heading = float(np.clip(d["heading"], -1.0, 1.0))
        if "speed" in d:
          S.speed = float(np.clip(d["speed"], 3, 14))
        if "amp" in d:
          S.amp = float(np.clip(d["amp"], 1.5, 5.0))
      elif c == "mode":
        S.gates = bool(d.get("gates", True))
        S.reset = True
      elif c == "tile":
        n = int(S.n_tiles)
        S.tile = int(np.clip(d["index"] if "index" in d else S.tile + int(d.get("d", 1)), 0, n - 1))
        S.reset = True
      elif c == "reset":
        S.reset = True

  # -- one step --------------------------------------------------------------------------------------------------------
  def tick(self) -> Frame:
    torch, env, S = self.torch, self.env, self.S
    if self.sim_steps % 25 == 0:
      self._load_policy()
    with S.lock:
      do_reset, S.reset = S.reset, False
      heading, speed = S.heading, S.speed
    if do_reset:
      self.respawn(top=True)
    a = time.perf_counter()
    xy = env.qpos[0, 0:2].numpy().astype(float)
    if self.state == "crash":
      self._crash_step()
    else:
      self._drive(xy, heading, speed)
      with torch.no_grad():
        act = self.policy(env.get_observations()) if self.policy is not None else torch.zeros(1, env.num_actions)
      env.step(act)
      self.sim_steps += 1
      self.t_run += DT
    self.state_t += DT
    self.step_ms.append((time.perf_counter() - a) * 1000)
    fell = bool(env.last_fell[0]) if env.last_fell is not None else False
    d = env.ski_diag
    v = env.qvel[0, 0:3].numpy()
    spd = float(np.hypot(v[0], v[1]))
    if d is not None and self.state != "crash":
      self.label = self.namer.update(d["N"][0].numpy(), d["edge"][0].numpy(), d["v_l"][0].numpy(), float(env.qvel[0, 5]), spd, fell, DT)
    # Run state: crash -> hold, then respawn at the top; end of the run -> fade, respawn at the top.
    if self.state == "run":
      if fell:
        self.state, self.state_t, self.label = "crash", 0.0, "Crash"
        self.falls += 1
        self.tree_falls += int(bool(getattr(env, "last_tree_hit", torch.zeros(1))[0]))
      elif self._run_over(spd):
        self.state, self.state_t = "end", 0.0
    elif self.state == "end" and self.state_t >= FADE_OUT:
      self.completed += 1
      self.respawn(top=True)
    elif self.state == "crash" and self.state_t >= CRASH_HOLD:
      self.respawn(top=self.course_dir is None)
    return self._frame(spd)

  def _drive(self, xy, heading: float, speed: float):
    env = self.env
    env.cmd_speed[0] = speed
    if self.course_dir is not None:
      self.cmd = self._course_heading(xy)
      env.cmd_heading[0] = self.cmd
    elif not self.gates_mode:
      tile = int(env.tile[0])
      hmax = float(env.tile_hmax[tile]) if hasattr(env, "tile_hmax") else 0.6
      off = float(xy[1] - env.tile_y[tile])
      # The page sends -1..1; keep the robot on the piste (bend back from 8 m off center) and around trunks.
      goal = float(np.clip(heading, -1, 1)) * hmax
      goal += -math.copysign(1.0, off) * float(np.clip((abs(off) - 8.0) / 6.0, 0, 1)) * 1.2
      goal = float(np.clip(goal, -hmax, hmax))
      if tile not in self.tile_trees_cache:
        self.tile_trees_cache[tile] = self.trees_all[self.trees_all[:, 4] == tile][:, :3] if len(self.trees_all) else None
      trees = self.tile_trees_cache[tile]
      goal = float(np.clip(_avoid_trees(xy, goal, trees), -hmax - 0.3, hmax + 0.3))
      self.cmd += float(np.clip(goal - self.cmd, -1.5 * DT, 1.5 * DT))
      env.cmd_heading[0] = self.cmd
    else:
      self.cmd = float(env.cmd_heading[0])
    env.cmd_timer[0] = 1e9  # pinned: the env keeps our heading (gates mode steers itself either way)

  def _course_heading(self, xy) -> float:
    from train.showcase import steer_around_trees

    c, env = self.course, self.env
    s_now = c.s_of(xy)
    while self.next_gate < len(c.s) and s_now > c.s[self.next_gate]:
      mid = (c.turn[self.next_gate] + c.outer[self.next_gate]) / 2
      self.hits += int(np.linalg.norm(xy - mid) < c.width[self.next_gate] / 2 + 0.6)
      self.next_gate += 1
    g = min(self.next_gate, len(c.aim) - 1)
    a = c.aim[g]
    if g + 1 < len(c.aim):
      w = np.clip(1 - (c.s[g] - s_now) / 4.0, 0, 1) * 0.3
      a = (1 - w) * a + w * c.aim[g + 1]
    spd = float(np.linalg.norm(env.qvel[0, 0:2].numpy()))
    a = steer_around_trees(xy, a, self.course_trees, look=float(np.clip(2.5 * spd, 12.0, 35.0)))
    _, nrm = env.grid.height_normal(self.torch.tensor([float(xy[0])]), self.torch.tensor([float(xy[1])]))
    # Same command frame as training and showcase.py: relative to the run direction, wider turns on steep pitches.
    run_dir = c.heading_at(s_now)
    slope_deg = float(np.degrees(np.arccos(np.clip(float(nrm[0, 2]), -1.0, 1.0))))
    hmax = 0.9 + 0.4 * float(np.clip((slope_deg - 12.0) / 15.0, 0.0, 1.0))
    bearing = float(np.arctan2(a[1] - xy[1], a[0] - xy[0]))
    target = run_dir + float(np.clip(np.arctan2(np.sin(bearing - run_dir), np.cos(bearing - run_dir)), -hmax, hmax))
    if self.t_run == 0.0:
      return target
    return self.cmd + float(np.clip(np.arctan2(np.sin(target - self.cmd), np.cos(target - self.cmd)), -0.03, 0.03))

  def _run_over(self, spd: float) -> bool:
    env = self.env
    x, y = float(env.qpos[0, 0]), float(env.qpos[0, 1])
    if self.course_dir is not None:
      c = self.course
      return c.s_of(np.array([x, y])) > c.length - 15 or self.next_gate >= len(c.s)
    self.slow_t = self.slow_t + DT if (spd < 0.4 and self.t_run > 3.0) else 0.0
    tile = int(env.tile[0])
    return x > env.tile_len - 22 or abs(y - float(env.tile_y[tile])) > 24 or self.slow_t > 5.0

  def _crash_step(self):
    """Hold the crash: let it fall a little further with no policy, but freeze before anything sinks into the snow
    (only the skis touch the snow in this model)."""
    env, torch = self.env, self.torch
    if self.state_t > 0.6:
      return
    fb = env.xpos[0, env.fall_ids]
    hb, _ = env.grid.height_normal(fb[:, 0], fb[:, 1])
    hp, _ = env.grid.height_normal(env.qpos[0, 0:1], env.qpos[0, 1:2])
    if bool((fb[:, 2] - hb < 0.02).any()) or float(env.qpos[0, 2] - hp[0]) < 0.25 or not bool(torch.isfinite(env.qpos).all()):
      return
    env.step(torch.zeros(1, env.num_actions))

  def _frame(self, spd: float) -> Frame:
    env, kd = self.env, self.kd
    kd.qpos[:] = env.qpos[0].numpy()
    mujoco.mj_kinematics(env.mjm, kd)
    f = Frame()
    f.pose = np.ascontiguousarray(np.concatenate([kd.xpos[self.bids], kd.xquat[self.bids]], 1), "<f4")
    d = env.ski_diag
    N = d["N"][0].numpy() if d is not None else np.zeros(2)
    ski = np.zeros((2, 4), "<f4")
    for k, b in enumerate(self.ski_ids):
      ski[k, :3] = kd.xpos[b] + kd.xmat[b].reshape(3, 3) @ self.sole
      ski[k, 3] = 0.0 if self.state == "crash" else N[k]
    f.ski = ski
    f.flags = (F_GATES if self.gates_mode else 0) | (F_CRASH if self.state == "crash" else 0) | (F_COURSE if self.course_dir else 0)
    f.label = LABELS.index(self.label) if self.label in LABELS else 2
    f.disc, f.run, f.speed, f.heading = self.disc, self.run, spd, self.cmd
    if self.course_dir is not None:
      f.gate_next, f.hits, f.crossed = self.next_gate, self.hits, self.next_gate
    elif self.gates_mode:
      f.gate_next, f.hits, f.crossed = int(env.gate_idx[0]), int(env.gates_hit[0]), int(env.gates_crossed[0])
    else:
      f.gate_next = f.hits = f.crossed = 0
    if self.state == "end":
      f.fade = min(1.0, self.state_t / FADE_OUT)
    elif self.state == "crash":
      f.fade = float(np.clip((self.state_t - (CRASH_HOLD - FADE_OUT)) / FADE_OUT, 0, 1))
    else:
      f.fade = float(np.clip(1.0 - self.t_run / FADE_IN, 0, 1)) if self.run > 1 else 0.0
    return f

  def hud(self) -> dict:
    env, S = self.env, self.S
    d = env.ski_diag
    with S.lock:
      it, ck, cmd_speed, amp = S.ckpt_iter, S.ckpt_path, S.speed, S.amp
    if self.course_dir is not None:
      mode, where = "course", self.course_dir.name.replace("-", " ")
      gates = f"{self.hits} clean of {self.next_gate} (of {len(self.course.s)})"
      s_now = self.course.s_of(env.qpos[0, 0:2].numpy())
      extra = {"distance": round(s_now, 1), "course_m": round(self.course.length, 1)}
    else:
      tile = int(env.tile[0])
      t = self.mosaic["tiles"][tile]
      feats = ", ".join(n for n, on in (("jumps", t.get("jumps")), ("compressions", t.get("compressions")),
                                        ("chute", t.get("chute_extra_deg")), ("flat", t.get("flat_section"))) if on) or "no features"
      mode = "gates" if self.gates_mode else "free"
      where = f"tile {tile} of {self.S.n_tiles}: {t.get('source', '')}, {t.get('mean_slope_deg')}°, {t.get('snow', 'groomed')}, {feats}"
      gates = f"{int(env.gates_hit[0])} of {int(env.gates_crossed[0])}" if self.gates_mode else "free steer"
      extra = {"tile": tile, "n_tiles": int(self.S.n_tiles), "amp": round(amp, 2)}
    return {"type": "hud", "mode": mode, "where": where, "label": self.label, "state": self.state,
            "policy": f"policy iteration {self.iter}" if self.policy is not None else "SONIC only (no checkpoint yet)",
            "iter": self.iter, "ckpt": f"{ck.parent.name}/{ck.name}" if ck else None, "pending": it if it != self.iter else None,
            "gates": gates, "run": self.run, "falls": self.falls, "tree_falls": self.tree_falls,
            "distance": round(float(env.qpos[0, 0]) - self.start_x, 1), "cmd_speed": round(cmd_speed, 1),
            "edges": [round(float(np.degrees(e)), 1) for e in (d["edge"][0].tolist() if d is not None else [0, 0])],
            "zones": (d["zones"][0] / 175).clamp(0, 1.5).numpy().round(2).tolist() if d is not None else [[0] * 4] * 2,
            "step_ms": round(float(np.mean(self.step_ms)), 1) if self.step_ms else None, **extra}


# ---------------------------------------------------------------------------------------------------------------------
# Main loop


class App:
  def __init__(self, args):
    self.args = args
    m = g1_model(slope(0.0, 4, 4, 1.0), SkiParams())
    self.body_ids, self.body_names, model_bytes = visual_model(m, args.mesh_cluster)
    self.hub = Hub(model_bytes, {"type": "hello", "bodies": len(self.body_ids), "labels": LABELS, "live": not args.replay,
                                 "source": "starting", "recordings": self.recordings()})
    self.live: Live | None = None
    self.replay: Replay | None = None
    self.rec: Recorder | None = None
    self.seq = 0
    self.t = 0.0
    self.rt_win = deque(maxlen=100)
    self.hud_t = 0.0
    self.saving = None

  @staticmethod
  def recordings() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(OUT.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "mb": round(p.stat().st_size / 1e6, 1)} for p in files if not p.name.endswith(".tmp.npz")]

  # -- recording -------------------------------------------------------------------------------------------------------
  def start_recording(self, path: Path | None = None):
    if self.rec is not None or self.live is None:
      return
    path = Path(path) if path else OUT / f"{'course-' + self.live.course_dir.name if self.live.course_dir else 'live'}-{time.strftime('%Y%m%d-%H%M%S')}.npz"
    self.rec = Recorder(path, self.body_names, {"source": "course" if self.live.course_dir else "tiles",
                                                 "course": str(self.live.course_dir or ""), "ckpt": str(self.live.loaded or ""),
                                                 "iter": self.live.iter, "terrain": str(getattr(self.live, "terrain", ""))})
    w = self.hub.world
    if w is not None:
      self.rec.world(w)
    if self.live.gates_json:
      self.rec.gate_list(self.live.gates_json)
    print(f"recording to {path}", flush=True)

  def stop_recording(self, block: bool = False):
    rec, self.rec = self.rec, None
    if rec is None:
      return

    def save():
      p = rec.save()
      print(f"saved {rec.n} frames ({rec.n * DT:.1f} s) to {p}", flush=True)
      self.hub.post(self.hub.set_hello, recordings=self.recordings())

    if block:
      save()
    else:
      self.saving = threading.Thread(target=save, daemon=False)
      self.saving.start()

  # -- source switching ------------------------------------------------------------------------------------------------
  def open_replay(self, name: str):
    p = (OUT / Path(name).name) if not Path(name).exists() else Path(name)
    try:
      rp = Replay(p)
    except Exception as e:  # noqa: BLE001
      print(f"cannot replay {p}: {e}", flush=True)
      return
    if rp.names != self.body_names:
      idx = {n: i for i, n in enumerate(rp.names)}
      if not all(n in idx for n in self.body_names):
        print(f"cannot replay {p}: different body set", flush=True)
        return
      rp.order = [idx[n] for n in self.body_names]
    else:
      rp.order = None
    self.stop_recording()
    self.replay = rp
    self.hub.post(self.hub.set_hello, source=f"replay {p.name}")
    print(f"replaying {p} ({rp.n} frames)", flush=True)

  def back_to_live(self):
    if self.live is None:
      return
    self.replay = None
    self.live.world_tile = None  # re-send the live world and gates
    self.live.respawn(top=True)
    self.hub.post(self.hub.set_hello, source="live")

  # -- the loop --------------------------------------------------------------------------------------------------------
  def handle_commands(self):
    while True:
      try:
        ws, d = self.hub.cmds.get_nowait()
      except queue.Empty:
        return
      c = d.get("cmd")
      if c == "record":
        self.start_recording() if d.get("on", self.rec is None) else self.stop_recording()
      elif c == "list":
        self.hub.post(self.hub.send_to, ws, json.dumps({"type": "recordings", "files": self.recordings()}))
      elif c == "replay":
        self.open_replay(d["file"]) if d.get("file") else self.back_to_live()
      elif c == "replay_ctl" and self.replay is not None:
        rp = self.replay
        if "pause" in d:
          rp.paused = bool(d["pause"])
          rp.disc += 1
        if "speed" in d:
          rp.speed = float(np.clip(d["speed"], 0.1, 4.0))
        if "seek" in d:
          rp.seek(float(d["seek"]))
      elif self.live is not None and self.replay is None:
        self.live.command(d)

  def emit(self, f: Frame, t_ready: float):
    self.seq += 1
    if self.rec is not None:
      f.flags |= F_REC
      self.rec.add(f, self.t)
      if self.rec.n >= self.args.max_record_s / DT:
        self.stop_recording()
    data = f.pack(self.seq, self.t)
    self.hub.post(self.hub.frame, data, t_ready)

  def replay_tick(self) -> bool:
    rp = self.replay
    if rp.paused:
      return False
    i = int(rp.pos)
    if i >= rp.n:
      rp.pos, i = 0.0, 0
      rp.disc += 1
      rp.last = -1
    if i == rp.last:  # slow motion: wait for the next recorded frame
      rp.pos += rp.speed
      return False
    w = rp._latest(rp.worlds, i)
    if w is not None and w is not getattr(rp, "_sent_world", None):
      rp._sent_world = w
      self.hub.post(self.hub.set_world, w)
    g = rp._latest(rp.gates, i)
    if g is not None and g is not getattr(rp, "_sent_gates", None):
      rp._sent_gates = g
      self.hub.post(self.hub.set_gates, g)
    t_ready = time.perf_counter()
    f = rp.frame_at(i, rp.order)
    rp.last = i
    rp.pos += rp.speed
    self.emit(f, t_ready)
    return True

  def hud(self, now: float):
    if now - self.hud_t < 0.12:
      return
    self.hud_t = now
    lat = np.asarray(self.hub.lat) * 1000 if self.hub.lat else np.zeros(1)
    rt = (len(self.rt_win) - 1) * DT / max(self.rt_win[-1] - self.rt_win[0], 1e-6) if len(self.rt_win) > 2 else None
    base = {"rt": round(rt, 2) if rt else None, "send_ms": round(float(np.mean(lat)), 2), "send_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "bytes": self.hub.size, "clients": len(self.hub.clients), "recording": self.rec is not None,
            "rec_s": round(self.rec.n * DT, 1) if self.rec else 0, "recordings": None}
    if self.replay is not None:
      rp = self.replay
      h = {"type": "hud", "mode": "replay", "where": f"replay {rp.path.name}", "label": LABELS[int(rp.cols['label'][max(rp.last, 0)])],
           "policy": f"policy iteration {rp.meta.get('iter')}", "iter": rp.meta.get("iter"), "frame": max(rp.last, 0), "frames": rp.n,
           "paused": rp.paused, "speed_x": rp.speed, "gates": f"{int(rp.cols['hits'][max(rp.last, 0)])} of {int(rp.cols['crossed'][max(rp.last, 0)])}",
           "run": int(rp.cols["run"][max(rp.last, 0)])}
    else:
      h = self.live.hud()
    h.update({k: v for k, v in base.items() if v is not None})
    self.hub.post(self.hub.set_hud, json.dumps(h, separators=(",", ":")))

  def run(self):
    args = self.args
    self.hub.start(args.host, args.port)
    print(f"viewer3d on http://127.0.0.1:{args.port}  (websocket /ws)", flush=True)
    if args.replay:
      self.open_replay(args.replay)
      if self.replay is None:
        sys.exit(1)
    else:
      self.live = Live(args, self.hub, self.body_ids, self.body_names)
      self.live.on_world = lambda msg: self.rec.world(msg) if self.rec else None
      self.live.on_gates = lambda text: self.rec.gate_list(text) if self.rec else None
      self.hub.post(self.hub.set_hello, source="live")
      if args.record:
        self.start_recording(Path(args.record))
    next_t = time.perf_counter()
    last_log = time.perf_counter()
    try:
      while True:
        self.handle_commands()
        if self.replay is not None:
          emitted = self.replay_tick()
          if emitted:
            self.t += DT
        else:
          f = self.live.tick()
          self.t += DT
          self.emit(f, time.perf_counter())
        now = time.perf_counter()
        self.rt_win.append(now)
        self.hud(now)
        if now - last_log > 30:
          last_log = now
          lat = np.asarray(self.hub.lat) * 1000 if self.hub.lat else np.zeros(1)
          print(f"frames {self.hub.sent}, {self.hub.size} B/frame, send {np.mean(lat):.2f} ms (p95 {np.percentile(lat, 95):.2f}), "
                f"clients {len(self.hub.clients)}", flush=True)
        next_t += DT
        sleep = next_t - time.perf_counter()
        if sleep > 0:
          time.sleep(sleep)
        elif sleep < -0.25:  # running behind (slow CPU): no catch-up burst, the page slows its clock instead
          next_t = time.perf_counter()
    except KeyboardInterrupt:
      pass
    finally:
      self.stop_recording(block=True)


def bake(args):
  """Record a run offline as fast as the CPU allows (no server, no pacing)."""
  app = App(args)

  class _NullHub:
    world = gates = None

    def post(self, fn, *a, **kw):
      fn(*a, **kw)

    def set_world(self, data):
      self.world = data

    def set_gates(self, text):
      self.gates = text

    def set_hello(self, **kw):
      pass

  app.hub = _NullHub()
  live = Live(args, app.hub, app.body_ids, app.body_names)
  app.live = live
  rec = Recorder(Path(args.bake), app.body_names, {"source": "course" if live.course_dir else "tiles", "course": str(live.course_dir or ""),
                                                   "ckpt": str(live.loaded or ""), "iter": live.iter, "baked": True})
  if app.hub.world is not None:
    rec.world(app.hub.world)
  if live.gates_json:
    rec.gate_list(live.gates_json)
  live.on_world = rec.world
  live.on_gates = rec.gate_list
  t, t0 = 0.0, time.time()
  n = int(args.seconds / DT)
  for i in range(n):
    f = live.tick()
    rec.add(f, t)
    t += DT
    if args.stop_at_end and live.completed > 0:
      break
    if i % 500 == 0:
      print(f"{i}/{n} frames, {time.time() - t0:.0f} s, run {live.run}, falls {live.falls}, x {float(live.env.qpos[0, 0]):.1f}", flush=True)
  p = rec.save()
  print(f"baked {rec.n} frames ({rec.n * DT:.1f} s) in {time.time() - t0:.0f} s to {p}", flush=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--terrain", default=str(ROOT / "out/terrains_v3"), help="tile mosaic for live tiles")
  ap.add_argument("--course", default="", help="a course folder (out/courses/<slug>) to ski end to end instead of tiles")
  ap.add_argument("--style", default="racing", help="'racing' (default, matches the current training run) or 'default'")
  ap.add_argument("--ckpt", default="", help="pin a checkpoint (default: newest in runs_live/, then synced from Modal)")
  ap.add_argument("--no-sync", action="store_true", help="do not pull checkpoints from the Modal volume")
  ap.add_argument("--record", default="", help="record the live run to this .npz (stops at --max-record-s)")
  ap.add_argument("--replay", default="", help="replay a recording instead of running physics")
  ap.add_argument("--bake", default="", help="record offline as fast as possible to this .npz, no server")
  ap.add_argument("--seconds", type=float, default=120.0, help="--bake length")
  ap.add_argument("--stop-at-end", action="store_true", help="--bake: stop when a course run reaches the end")
  ap.add_argument("--max-record-s", type=float, default=900.0)
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--port", type=int, default=PORT)
  ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--tile", type=int, default=10)
  ap.add_argument("--speed", type=float, default=9.0)
  ap.add_argument("--free", action="store_true", help="start in free-steer mode instead of gates")
  ap.add_argument("--gate-spacing", type=float, default=0.0, help="course: re-space gates every N m (0 keeps the course's own)")
  ap.add_argument("--gate-offset", type=float, default=2.5)
  ap.add_argument("--no-trees", action="store_true", help="course: no trees in or along the run")
  ap.add_argument("--trees-in-run", type=int, default=14)
  ap.add_argument("--mesh-cluster", type=float, default=0.002, help="robot mesh decimation cell, m (0 keeps full meshes)")
  args = ap.parse_args()
  logging.basicConfig(level=logging.WARNING)
  logging.getLogger("websockets").setLevel(logging.ERROR)
  sys.setswitchinterval(0.001)  # the websocket thread gets the GIL quickly between sim steps
  signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))  # kill/pkill still saves a recording in progress
  OUT.mkdir(parents=True, exist_ok=True)
  if not args.replay:
    S = live_module().S
    with S.lock:
      S.tile, S.speed, S.gates = args.tile, args.speed, not args.free
  if args.bake:
    args.no_sync = True
    bake(args)
    return
  App(args).run()


if __name__ == "__main__":
  main()
