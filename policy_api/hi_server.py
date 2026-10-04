"""Hazard Intelligence policy server.

Serves trained runtime policies the way Physical Intelligence's openpi serves pi0: a websocket that sends the
policy's metadata on connect, then answers each msgpack-numpy observation with an action, so openpi_client's
WebsocketClientPolicy works against it unchanged. Plain HTTP endpoints sit next to it for curl and browsers, and
every policy can be downloaded as ONNX to run inside your own physics loop at full rate.

    uvicorn hi_server:app --port 8000          # from policy_api/, needs fastapi uvicorn numpy msgpack

  GET  /v1/policies                  list
  GET  /v1/policies/{id}             observation and action layout, control rate, benchmark, provenance
  GET  /v1/policies/{id}/onnx        runtime policy for local inference
  GET  /v1/policies/{id}/weights     the same network as numpy arrays (npz)
  POST /v1/policies/{id}/infer       {"obs": [...] or [[...], ...]} -> {"actions": ...}
  WS   /v1/policies/{id}/ws          openpi protocol: metadata, then {"obs": array} -> {"actions": array}
  POST /v1/missions                  {policy, route: {x, y}, target_s | target_xy, speed?, arrive_radius?} -> mission
  POST /v1/missions/{mid}/step       {obs, pose: {x, y, yaw}} -> actions with the server's own route command
  GET  /v1/missions/{mid}            progress, distance to target, arrival time; ?log=1 adds every step
  WS   /                             same, default policy (or ?policy=<id>), so openpi_client can point at the host

Keys and usage live in the team's SpacetimeDB (DATABASE.md). A key goes in `Authorization: Bearer hi_...`,
`Authorization: Api-Key hi_...` (openpi_client's api_key) or `X-Api-Key`. Without one, the public policies still work
and the call counts for the public tenant; a wrong or revoked key gets 401. Every call is logged to api_call in
batches. STDB_TOKEN (the API's platform-service identity, from tools/stdb_admin.py api-identity), STDB_HTTP and
STDB_DB configure it; with no token, nothing is checked or logged.
"""
from __future__ import annotations

import asyncio
import atexit
import functools
import hashlib
import json
import os
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import msgpack
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import mission as missions

POLICY_DIR = Path(os.environ.get("HI_POLICY_DIR", Path(__file__).parent / "policies"))
DEFAULT_POLICY = os.environ.get("HI_DEFAULT_POLICY", "g1-ski")


# msgpack-numpy, byte-compatible with openpi_client.msgpack_numpy.
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class Policy:
    """Normalizer + ELU MLP in numpy: the rsl_rl actor's deterministic mean action."""

    def __init__(self, d: Path):
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        w = np.load(d / "weights.npz")
        self.mean, self.std, self.eps = w["mean"], w["std"], float(w["eps"])
        self.layers = [(w[f"W{k}"].T.copy(), w[f"b{k}"]) for k in range(4)]
        self.obs_dim = self.meta["obs_dim"]
        self.clip = self.meta["action"]["clip"]

    def infer(self, obs: dict) -> dict:
        x = np.asarray(obs["obs"] if isinstance(obs, dict) else obs, dtype=np.float32)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        if x.shape[1] != self.obs_dim:
            raise ValueError(f"{self.meta['id']} expects obs of size {self.obs_dim}, got {x.shape[1]}")
        h = (np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0) - self.mean) / (self.std + self.eps)
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < 3:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
        a = np.clip(h, *self.clip).astype(np.float32)
        return {"actions": a[0] if single else a}


POLICIES = {p.name: Policy(p) for p in sorted(POLICY_DIR.iterdir()) if (p / "meta.json").exists()}

app = FastAPI(title="Hazard Intelligence policy API", version="0.1")


# ------------------------------------------------------------------------------------- keys and usage (SpacetimeDB)
STDB_HTTP = os.environ.get("STDB_HTTP", "https://maincloud.spacetimedb.com").rstrip("/")
STDB_DB = os.environ.get("STDB_DB", "ground-truth")
STDB_TOKEN = os.environ.get("STDB_TOKEN", "")
KEY_TTL_S = 60  # how long a key's state is trusted before asking again
_keys: dict[str, tuple[float, bool]] = {}  # sha256 -> (checked at, valid)
_calls: list[dict] = []
_calls_lock = threading.Lock()


def _stdb(path: str, body: bytes, ctype: str) -> bytes:
    req = urllib.request.Request(f"{STDB_HTTP}/v1/database/{STDB_DB}/{path}", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {STDB_TOKEN}", "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def key_of(headers) -> str:
    auth = headers.get("authorization") or ""
    for scheme in ("Bearer ", "Api-Key "):
        if auth.startswith(scheme):
            return auth[len(scheme):].strip()
    return (headers.get("x-api-key") or "").strip()


def key_valid(key: str) -> bool | None:
    """True or False for a key, None when there is no database to ask (then the key is not enforced)."""
    if not STDB_TOKEN:
        return None
    h = hashlib.sha256(key.encode()).hexdigest()
    hit = _keys.get(h)
    if hit and time.time() - hit[0] < KEY_TTL_S:
        return hit[1]
    try:
        rows = json.loads(_stdb("sql", f"SELECT revoked FROM service_api_key WHERE hash = '{h}'".encode(), "text/plain"))[0]["rows"]
    except Exception as e:  # noqa: BLE001  the database is down: serve, and ask again next time
        print(f"[api] key check failed: {e}")
        return None
    ok = bool(rows) and not rows[0][0]
    _keys[h] = (time.time(), ok)
    return ok


def log_call(key: str, path: str, status: int, ms: float):
    if not STDB_TOKEN:
        return
    parts = path.strip("/").split("/")
    if parts[:2] == ["v1", "policies"]:
        pid = parts[2] if len(parts) > 2 else ""
        route = parts[3] if len(parts) > 3 else ("spec" if pid else "list")
    elif parts[:2] == ["v1", "missions"]:
        pid, route = "", "missions"
    else:
        pid, route = "", parts[0] or "root"
    with _calls_lock:
        _calls.append({"key_hash": hashlib.sha256(key.encode()).hexdigest() if key else "", "policy_id": pid,
                       "route": route, "status": int(status), "ms": round(float(ms), 3), "at_ms": int(time.time() * 1000)})
        del _calls[:-5000]  # if the database is unreachable for long, keep the newest


def flush_calls():
    with _calls_lock:
        batch, _calls[:] = _calls[:1000], _calls[1000:]
    if not batch:
        return
    try:
        _stdb("call/log_api_calls", json.dumps({"calls": batch}).encode(), "application/json")
    except Exception as e:  # noqa: BLE001
        print(f"[api] usage log failed, keeping {len(batch)} calls: {e}")
        with _calls_lock:
            _calls[:0] = batch


def _flusher():
    while True:
        time.sleep(2)
        flush_calls()


if STDB_TOKEN:
    threading.Thread(target=_flusher, daemon=True).start()
    atexit.register(flush_calls)


@app.middleware("http")
async def keys_and_usage(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in ("/", "/healthz"):
        return await call_next(request)
    t0 = time.perf_counter()
    key = key_of(request.headers)
    if key and await asyncio.to_thread(key_valid, key) is False:
        resp = JSONResponse({"detail": "invalid or revoked API key"}, status_code=401)
    else:
        resp = await call_next(request)
    log_call(key, request.url.path, resp.status_code, (time.perf_counter() - t0) * 1e3)
    return resp


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get(pid: str) -> Policy:
    if pid not in POLICIES:
        raise HTTPException(404, f"no policy {pid!r}; have {sorted(POLICIES)}")
    return POLICIES[pid]


@app.get("/")
def root():
    return {"service": "hazard-intelligence-policy", "policies": sorted(POLICIES), "default": DEFAULT_POLICY,
            "docs": "/docs", "websocket": "/v1/policies/{id}/ws (openpi protocol)"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/policies")
def list_policies():
    keys = ("id", "name", "robot", "skill", "obs_dim", "action_dim", "control_hz", "benchmark")
    return [{k: p.meta[k] for k in keys} for p in POLICIES.values()]


@app.get("/v1/policies/{pid}")
def policy_meta(pid: str):
    return get(pid).meta


@app.get("/v1/policies/{pid}/onnx")
def policy_onnx(pid: str):
    return FileResponse(get(pid).dir / "policy.onnx", media_type="application/octet-stream", filename=f"{pid}.onnx")


@app.get("/v1/policies/{pid}/weights")
def policy_weights(pid: str):
    return FileResponse(get(pid).dir / "weights.npz", media_type="application/octet-stream", filename=f"{pid}.npz")


class InferRequest(BaseModel):
    obs: list[float] | list[list[float]]


@app.post("/v1/policies/{pid}/infer")
def policy_infer(pid: str, req: InferRequest):
    p = get(pid)
    t0 = time.perf_counter()
    try:
        out = p.infer({"obs": req.obs})
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"policy": pid, "actions": out["actions"].tolist(), "server_timing": {"infer_ms": (time.perf_counter() - t0) * 1e3}}


async def _serve_ws(ws: WebSocket, p: Policy):
    key = key_of(ws.headers) or ws.query_params.get("key", "")
    if key and await asyncio.to_thread(key_valid, key) is False:
        await ws.close(code=4401)
        log_call(key, f"/v1/policies/{p.meta['id']}/ws", 401, 0.0)
        return
    t_open = time.perf_counter()
    await ws.accept()
    try:
        await _serve_ws_loop(ws, p)
    finally:
        log_call(key, f"/v1/policies/{p.meta['id']}/ws", 101, (time.perf_counter() - t_open) * 1e3)


async def _serve_ws_loop(ws: WebSocket, p: Policy):
    await ws.send_bytes(packb(p.meta))
    prev_total = None
    while True:
        try:
            t0 = time.monotonic()
            obs = unpackb(await ws.receive_bytes())
            t1 = time.monotonic()
            out = p.infer(obs)
            timing = {"infer_ms": (time.monotonic() - t1) * 1e3}
            if prev_total is not None:
                timing["prev_total_ms"] = prev_total * 1e3
            out["server_timing"] = timing
            await ws.send_bytes(packb(out))
            prev_total = time.monotonic() - t0
        except WebSocketDisconnect:
            return
        except Exception:
            await ws.send_text(traceback.format_exc())
            await ws.close(code=1011)
            return


@app.websocket("/v1/policies/{pid}/ws")
async def policy_ws(ws: WebSocket, pid: str):
    if pid not in POLICIES:
        await ws.close(code=4404)
        return
    await _serve_ws(ws, POLICIES[pid])


@app.websocket("/")
async def root_ws(ws: WebSocket):
    pid = ws.query_params.get("policy", DEFAULT_POLICY)
    if pid not in POLICIES:
        await ws.close(code=4404)
        return
    await _serve_ws(ws, POLICIES[pid])


MISSIONS: dict[str, missions.Mission] = {}


class Route(BaseModel):
    x: list[float]
    y: list[float]


class MissionRequest(BaseModel):
    policy: str = DEFAULT_POLICY
    route: Route
    target_s: float | None = None
    target_xy: list[float] | None = None
    speed: float = 3.0
    arrive_radius: float = 3.0


class Pose(BaseModel):
    x: float
    y: float
    yaw: float


class StepRequest(BaseModel):
    obs: list[float]
    pose: Pose
    t: float | None = None


def command_slice(p: Policy) -> slice:
    f = next(o for o in p.meta["observation"] if o["name"] == "command")
    return slice(f["start"], f["start"] + f["size"])


@app.post("/v1/missions")
def create_mission(req: MissionRequest):
    get(req.policy)
    try:
        m = missions.make(req.policy, req.route.x, req.route.y, req.target_s, req.target_xy,
                          speed=req.speed, arrive_radius=req.arrive_radius)
    except ValueError as e:
        raise HTTPException(422, str(e))
    MISSIONS[m.id] = m
    return m.status()


def get_mission(mid: str) -> missions.Mission:
    if mid not in MISSIONS:
        raise HTTPException(404, f"no mission {mid!r}")
    return MISSIONS[mid]


@app.post("/v1/missions/{mid}/step")
def mission_step(mid: str, req: StepRequest):
    m = get_mission(mid)
    p = get(m.policy)
    obs = np.asarray(req.obs, dtype=np.float32)
    if obs.shape != (p.obs_dim,):
        raise HTTPException(422, f"{p.meta['id']} expects obs of size {p.obs_dim}, got {obs.shape}")
    cmd = m.command(req.pose.x, req.pose.y, req.pose.yaw, req.t)
    obs[command_slice(p)] = cmd.pop("command")
    return {"mission": mid, "actions": p.infer({"obs": obs})["actions"].tolist(), **cmd}


@app.get("/v1/missions/{mid}")
def mission_status(mid: str, log: bool = False):
    m = get_mission(mid)
    return {**m.status(), **({"log": m.log} if log else {})}
