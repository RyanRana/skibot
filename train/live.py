"""Live skier: the newest policy from the continuous Modal run, on slalom gates or steered by touch.

Run: PYTHONPATH=. .venv-train/bin/python -m train.live   then open http://127.0.0.1:8766 (or your LAN IP on a phone)
- Physics runs in CPU MuJoCo with the exact training env code (backend="mujoco"), paced to real time.
- Pulls the newest checkpoint from the Modal volume g1-ski-runs every 60 s and hot-swaps it.
- Gates mode: drag up/down for speed, left/right for how curvy the course is (applies on the next run).
  Free mode: drag left/right to steer, up/down for speed.
- Renders only the current tile, with the gates as red and blue poles.
"""

from __future__ import annotations

import argparse
import os
import shutil
import io
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from skisim.draw import draw_trees
from skisim.scene import g1_model
from skisim.ski import SkiParams
from skisim.terrain import HeightGrid
from train.ski_env import ROOT, SkiEnv
from train.train import load_policy, runner_cfg

MODAL = os.environ.get("MODAL_BIN", shutil.which("modal") or str(Path.home() / "mhacks-spikes/.venv/bin/modal"))
LIVE_DIR = ROOT / "runs_live"
PORT = 8766
W, H = 640, 360


class Shared:
  def __init__(self):
    self.lock = threading.Lock()
    self.jpeg = b""
    self.heading = 0.0
    self.speed = 9.0
    self.amp = 3.0
    self.gates = True
    self.tile = 10
    self.n_tiles = 24
    self.reset = True
    self.state = {}
    self.ckpt_path = None
    self.ckpt_iter = None


S = Shared()


def newest_remote_checkpoint():
  """(run_dir, path, iteration) of the highest model_N.pt in the newest run on the volume."""
  name = lambda e: e.get("filename") or e.get("Filename") or ""
  out = subprocess.run([MODAL, "volume", "ls", "g1-ski-runs", "--json"], capture_output=True, text=True, timeout=60)
  runs = sorted(name(e) for e in json.loads(out.stdout or "[]") if str(e.get("type") or e.get("Type")).lower() in ("dir", "directory", "2"))
  if not runs:
    return None
  run = runs[-1]
  out = subprocess.run([MODAL, "volume", "ls", "g1-ski-runs", run, "--json"], capture_output=True, text=True, timeout=60)
  best = None
  for e in json.loads(out.stdout or "[]"):
    m = re.search(r"model_(\d+)\.pt$", name(e))
    if m and (best is None or int(m.group(1)) > best[2]):
      best = (run, name(e) if "/" in name(e) else f"{run}/{name(e)}", int(m.group(1)))
  return best


def sync_loop():
  LIVE_DIR.mkdir(exist_ok=True)
  while True:
    try:
      found = newest_remote_checkpoint()
      if found and (S.ckpt_path is None or found[1] != f"{S.ckpt_path.parent.name}/{S.ckpt_path.name}"):
        run, fname, it = found
        dest = LIVE_DIR / Path(run).name
        dest.mkdir(exist_ok=True)
        subprocess.run([MODAL, "volume", "get", "--force", "g1-ski-runs", fname, str(dest) + "/"], check=True,
                       capture_output=True, timeout=300)
        local = dest / Path(fname).name
        if local.exists():
          with S.lock:
            S.ckpt_path, S.ckpt_iter = local, it
          print(f"pulled {run} iteration {it}", flush=True)
    except Exception as e:
      print("sync:", e, flush=True)
    time.sleep(60)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>G1 Ski Live</title>
<style>
:root{--bg:#0b1220;--panel:#111b2e;--ink:#e6edf6;--muted:#94a4b9;--rule:#24334c;--red:#f0596d;--blue:#7aa2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:14px;grid-template-columns:minmax(0,2fr) minmax(0,1fr)}
@media(max-width:820px){.wrap{grid-template-columns:1fr}}
h1{margin:0;font:700 26px/1 system-ui;letter-spacing:-.01em}.muted{color:var(--muted)}
img{width:100%;border:1px solid var(--rule);background:#000;display:block;aspect-ratio:16/9}
.pad{position:relative;height:220px;border:1px solid var(--rule);background:var(--panel);touch-action:none;user-select:none}
.pad .dot{position:absolute;width:28px;height:28px;margin:-14px 0 0 -14px;border-radius:50%;background:var(--red)}
.pad .axis{position:absolute;left:8px;bottom:6px;font-size:12px;color:var(--muted)}
.row{display:flex;gap:8px;flex-wrap:wrap}button{background:var(--panel);color:var(--ink);border:1px solid var(--rule);padding:8px 12px;font:inherit;cursor:pointer}
button[aria-pressed=true]{border-color:var(--red)}button:focus-visible{outline:2px solid var(--blue)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td{padding:4px 0;border-bottom:1px solid var(--rule)}td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}
.skis{display:grid;grid-template-columns:1fr 1fr;gap:10px}.ski{display:grid;grid-template-columns:1fr 1fr;gap:3px}
.z{height:30px;background:var(--panel);border:1px solid var(--rule);position:relative}.z i{position:absolute;inset:auto 0 0 0;background:var(--blue)}
</style></head><body><div class="wrap">
<div style="display:grid;gap:10px;min-width:0">
<h1>G1 Ski Live</h1><div class="muted" id="sub">loading</div>
<img src="/stream" alt="Live G1 skiing">
</div>
<div style="display:grid;gap:12px;align-content:start;min-width:0">
<div class="row"><button id="mode" aria-pressed="true">Gates</button><button id="prev">Easier tile</button><button id="next">Harder tile</button><button id="reset">New run</button></div>
<div class="pad" id="pad"><div class="dot" id="dot"></div><div class="axis" id="axis"></div></div>
<div><div class="muted" style="margin-bottom:6px">Ski touch (front / rear, inside / outside edge)</div>
<div class="skis"><div><div class="muted">Left ski</div><div class="ski" id="zl"></div></div><div><div class="muted">Right ski</div><div class="ski" id="zr"></div></div></div></div>
<table id="t"></table>
</div></div>
<script>
const pad=document.getElementById('pad'),dot=document.getElementById('dot'),axis=document.getElementById('axis'),modeBtn=document.getElementById('mode');
let cmd={gates:true,heading:0,speed:9,amp:3};
function labels(){axis.textContent=cmd.gates?'drag: left/right course curviness (next run), up/down speed':'drag: left/right steer, up/down speed';modeBtn.textContent=cmd.gates?'Gates':'Free steer';modeBtn.setAttribute('aria-pressed',cmd.gates)}
function place(){const r=pad.getBoundingClientRect();const fx=cmd.gates?(cmd.amp-1.5)/3.5:(cmd.heading/0.6+1)/2;dot.style.left=(fx*r.width)+'px';dot.style.top=((1-(cmd.speed-3)/11)*r.height)+'px'}
function send(){fetch('/cmd',{method:'POST',body:JSON.stringify(cmd)}).catch(()=>{})}
function move(e){const r=pad.getBoundingClientRect();const x=Math.min(Math.max((e.clientX-r.left)/r.width,0),1),y=Math.min(Math.max((e.clientY-r.top)/r.height,0),1);
if(cmd.gates)cmd.amp=+(1.5+x*3.5).toFixed(2);else cmd.heading=+((x*2-1)*0.6).toFixed(3);cmd.speed=+(3+(1-y)*11).toFixed(2);place();send()}
pad.addEventListener('pointerdown',e=>{pad.setPointerCapture(e.pointerId);move(e)});pad.addEventListener('pointermove',e=>{if(e.buttons)move(e)});
modeBtn.onclick=()=>{cmd.gates=!cmd.gates;labels();place();send();fetch('/reset',{method:'POST'})};
for(const [id,u] of [['prev','/tile?d=-1'],['next','/tile?d=1'],['reset','/reset']])document.getElementById(id).onclick=()=>fetch(u,{method:'POST'});
function zones(el,z){el.innerHTML=z.map(v=>`<div class="z"><i style="height:${Math.min(100,v*100).toFixed(0)}%"></i></div>`).join('')}
async function poll(){try{const s=await (await fetch('/state')).json();
document.getElementById('sub').textContent=`${s.policy} · tile ${s.tile}: ${s.source}, ${s.slope}°, ${s.snow}, ${s.features} · ${s.rt}x real time`;
zones(document.getElementById('zl'),s.zones[0]);zones(document.getElementById('zr'),s.zones[1]);
const rows=[['Speed',s.speed+' m/s ('+Math.round(s.speed*3.6)+' km/h)'],['Target speed',s.cmd_speed+' m/s'],['Gates this run',s.gates],['Edge angles L / R',s.edges.join(' / ')+'°'],['Run distance',s.distance+' m'],['Falls this session',s.falls]];
document.getElementById('t').innerHTML=rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('')}catch(e){}setTimeout(poll,300)}
labels();place();poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
  def log_message(self, *a):
    pass

  def _json(self, obj):
    b = json.dumps(obj).encode()
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(b)))
    self.end_headers()
    self.wfile.write(b)

  def do_GET(self):
    if self.path == "/":
      b = PAGE.encode()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(b)))
      self.end_headers()
      self.wfile.write(b)
    elif self.path == "/stream":
      self.send_response(200)
      self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      last = None
      try:
        while True:
          with S.lock:
            jpg = S.jpeg
          if jpg and jpg is not last:
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            last = jpg
          time.sleep(0.01)
      except (BrokenPipeError, ConnectionResetError):
        return
    elif self.path == "/state":
      with S.lock:
        self._json(S.state)
    else:
      self.send_error(404)

  def do_POST(self):
    n = int(self.headers.get("Content-Length") or 0)
    body = self.rfile.read(n) if n else b""
    with S.lock:
      if self.path == "/cmd":
        d = json.loads(body or b"{}")
        S.heading = float(np.clip(d.get("heading", 0), -0.6, 0.6))
        S.speed = float(np.clip(d.get("speed", 9), 3, 14))
        S.amp = float(np.clip(d.get("amp", 3), 1.5, 5.0))
        S.gates = bool(d.get("gates", True))
      elif self.path.startswith("/tile"):
        S.tile = int(np.clip(S.tile + (1 if "d=1" in self.path else -1), 0, S.n_tiles - 1))
        S.reset = True
      elif self.path == "/reset":
        S.reset = True
    self._json({"ok": True})


def tile_render_model(Z, meta, env, tile):
  """MuJoCo model of just one tile (fast to draw) in the same world coordinates as the mosaic."""
  cell = meta["cell"]
  yc = float(env.tile_y[tile])
  r0 = max(0, int((yc - 34) / cell))
  r1 = min(Z.shape[0], int((yc + 34) / cell) + 1)
  grid = HeightGrid(z=Z[r0:r1], x0=meta["x0"], y0=meta["y0"] + r0 * cell, cell=cell)
  model = g1_model(grid, SkiParams())
  return model, mujoco.MjData(model), mujoco.Renderer(model, H, W)


def keep_above_snow(cam, grid: HeightGrid, clearance: float = 1.2):
  """Lift the camera until it, and the line of sight near it, clear the snow (as in train/showcase.py).

  A chase camera behind the skier on a steep pitch ends up inside the slope, where the hfield is invisible from
  below. Raise the elevation first; if even a steep down-angle is not enough, shorten the distance.
  """
  fr = np.array([0.0, 0.25, 0.5, 0.75])
  need = clearance * (1.0 - fr)  # full clearance at the lens, tapering toward the skier
  look = np.asarray(cam.lookat, float)
  for _ in range(60):
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    pos = look - cam.distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pts = pos + fr[:, None] * (look - pos)
    h, _ = grid.height_normal(pts[:, 0], pts[:, 1])
    if np.all(pts[:, 2] >= h + need):
      return
    if cam.elevation > -80:
      cam.elevation = max(cam.elevation - 2.0, -80.0)
    elif cam.distance > 1.0:
      cam.distance *= 0.85
    else:
      return


def draw_gates(scene, env):
  """Gate poles as extra scene geoms: two poles per gate, red and blue alternating, passed gates faded."""
  if not bool(env.use_gates[0]):
    return
  gx, gy = env.gate_x[0].numpy(), env.gate_y[0].numpy()
  idx = int(env.gate_idx[0])
  for k in range(len(gx)):
    if gx[k] > 1e5:
      break
    for dy in (-1.5, 1.5):
      if scene.ngeom >= scene.maxgeom:
        return
      x, y = float(gx[k]), float(gy[k] + dy)
      h, _ = env.grid.height_normal(torch.tensor([x]), torch.tensor([y]))
      col = [0.85, 0.1, 0.1, 1.0] if k % 2 == 0 else [0.1, 0.25, 0.9, 1.0]
      if k < idx:
        col[3] = 0.25
      mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.03, 0.65, 0]),
                          np.array([x, y, float(h[0]) + 0.65]), np.eye(3).ravel(), np.array(col, dtype=np.float32))
      scene.ngeom += 1


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--terrain", default=str(ROOT / "out/terrains_v2"))
  args = ap.parse_args()
  terrain = Path(args.terrain)
  threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever(), daemon=True).start()
  threading.Thread(target=sync_loop, daemon=True).start()
  print(f"live viewer on http://127.0.0.1:{PORT}", flush=True)

  env = SkiEnv(num_envs=1, device="cpu", use_graph=False, seed=7, mosaic=terrain, backend="mujoco")
  S.n_tiles = env.n_tiles
  meta = json.loads((terrain / "mosaic.json").read_text())
  Z = np.load(terrain / "mosaic_elevation.npy").astype(float)
  tree_file = terrain / "trees.npy"
  trees_all = np.load(tree_file) if tree_file.exists() else np.zeros((0, 5), np.float32)
  draw_grid = HeightGrid(z=Z, x0=meta["x0"], y0=meta["y0"], cell=meta["cell"])
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  policy, loaded, falls, look = None, None, 0, None
  model = data = renderer = None
  render_tile = None
  start_x = 0.0
  t0, sim_t = time.time(), 0.0
  step = 0
  ids = torch.tensor([0])

  def apply_mode(gates_mode, amp):
    env.use_gates[0] = gates_mode
    env._place_gates(ids, amp=amp)
    env._update_gate_command()
    env.obs = env._observations()

  while True:
    with S.lock:
      want, tile, do_reset = S.ckpt_path, S.tile, S.reset
      S.reset = False
      heading, speed, gates_mode, amp = S.heading, S.speed, S.gates, S.amp
    if want is not None and want != loaded:
      try:
        from rsl_rl.runners import OnPolicyRunner

        policy, loaded = load_policy(env, str(want)), want
        print("now driving with", want, flush=True)
      except Exception as e:
        print("policy load failed:", e, flush=True)
        loaded = want
    if do_reset:
      env.level[0] = tile
      env.gate_frac = 1.0 if gates_mode else 0.0
      env.reset_idx(ids)
      env.tile[0] = tile
      env._forward()
      env._refresh_history(ids, fill=True)
      apply_mode(gates_mode, amp)
      start_x = float(env.qpos[0, 0])
      t0, sim_t = time.time(), 0.0
    if render_tile != tile:
      if renderer is not None:
        renderer.close()
      model, data, renderer = tile_render_model(Z, meta, env, tile)
      renderer.close()
      renderer = mujoco.Renderer(model, H, W, max_geom=6000)
      render_tile = tile
    env.cmd_speed[0] = speed
    if not gates_mode:
      env.cmd_heading[0] = heading
    env.cmd_timer[0] = 1e9
    obs = env.get_observations()
    with torch.no_grad():
      act = policy(obs) if policy is not None else torch.zeros(1, env.num_actions)
    _, _, done, extras = env.step(act)
    sim_t += 0.02
    step += 1
    if bool(done[0]):
      if not bool(extras["time_outs"][0]):
        falls += 1
      env.level[0] = tile
      apply_mode(gates_mode, amp)
      start_x = float(env.qpos[0, 0])
    if step % 2 == 0:
      data.qpos[:] = env.qpos[0].numpy()
      mujoco.mj_forward(model, data)
      v3 = env.qvel[0, 0:3].numpy()
      p = data.xpos[1].copy() + v3 * 0.06  # lead the skier so fast runs stay framed
      look = p if look is None or bool(done[0]) or np.linalg.norm(p - look) > 6 else 0.45 * look + 0.55 * p
      v = v3[:2]
      head = np.degrees(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 0.5 else 0.0
      cam.lookat[:] = look + np.array([0.0, 0.0, -0.3])  # skier sits slightly above image center
      cam.distance, cam.azimuth, cam.elevation = 5.0, head - 28, -16
      keep_above_snow(cam, draw_grid)
      renderer.update_scene(data, cam)
      draw_gates(renderer.scene, env)
      draw_trees(renderer.scene, trees_all[trees_all[:, 4] == tile] if len(trees_all) else trees_all, draw_grid, p[:2], radius=60)
      buf = io.BytesIO()
      Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=72)
      d = env.ski_diag
      zones = (d["zones"][0] / 175).clamp(0, 1.5).numpy().round(2).tolist() if d is not None else [[0] * 4] * 2
      t = meta["tiles"][tile]
      feats = ", ".join(n for n, on in (("jumps", t.get("jumps")), ("compressions", t.get("compressions")),
                                        ("chute", t.get("chute_extra_deg")), ("flat", t.get("flat_section"))) if on) or "no features"
      with S.lock:
        S.jpeg = buf.getvalue()
        S.state = {
          "policy": f"policy iteration {S.ckpt_iter}" if policy is not None else "SONIC only (waiting for a checkpoint)",
          "tile": tile, "slope": t["mean_slope_deg"], "snow": t.get("snow", "groomed"), "features": feats,
          "source": t.get("source", ""), "rt": round(sim_t / max(time.time() - t0, 1e-6), 2),
          "speed": round(float(np.linalg.norm(v)), 1), "cmd_speed": round(speed, 1),
          "gates": f"{int(env.gates_hit[0])} of {int(env.gates_crossed[0])}" if gates_mode else "free steer",
          "edges": [round(float(np.degrees(e)), 1) for e in (d["edge"][0].tolist() if d is not None else [0, 0])],
          "zones": zones, "distance": round(float(env.qpos[0, 0]) - start_x, 1), "falls": falls,
        }
    ahead = sim_t - (time.time() - t0)  # pace to real time
    if ahead > 0:
      time.sleep(ahead)


if __name__ == "__main__":
  main()
