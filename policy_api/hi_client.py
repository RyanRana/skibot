"""Client for the Hazard Intelligence policy API.

Pull a policy and run it in your own physics loop (no network on the hot path, full 50 Hz):

    from hi_client import pull
    policy = pull("https://<host>", "g1-ski")        # downloads meta + ONNX once, caches under ~/.cache/hi
    policy.meta["architecture"]                       # layers, activation, normalizer
    actions = policy.infer(obs)                       # obs (141,) or (B, 141) -> (31,) or (B, 31)

Or ask the server every step over a websocket (handy for quick tests, adds a network round trip):

    from hi_client import HiPolicy
    policy = HiPolicy("wss://<host>", "g1-ski")
    actions = policy.infer({"obs": obs})["actions"]

Needs numpy, plus onnxruntime for pull() or msgpack + websockets for HiPolicy.
openpi_client's WebsocketClientPolicy(host="wss://<host>") also talks to the server's root websocket.
"""
from __future__ import annotations

import functools
import json
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "hi"


class PulledPolicy:
    def __init__(self, meta: dict, onnx_path: Path):
        import onnxruntime as ort

        self.meta = meta
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.clip = meta["action"]["clip"]

    def infer(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        a = self.session.run(["actions"], {"obs": np.atleast_2d(x)})[0]
        a = np.clip(a, *self.clip)
        return a[0] if single else a


def pull(host: str, policy: str = "g1-ski", refresh: bool = False) -> PulledPolicy:
    base = host.rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base
    d = CACHE / policy
    d.mkdir(parents=True, exist_ok=True)
    if refresh or not (d / "policy.onnx").exists():
        with urllib.request.urlopen(f"{base}/v1/policies/{policy}") as r:
            (d / "meta.json").write_bytes(r.read())
        with urllib.request.urlopen(f"{base}/v1/policies/{policy}/onnx") as r:
            (d / "policy.onnx").write_bytes(r.read())
    return PulledPolicy(json.loads((d / "meta.json").read_text()), d / "policy.onnx")


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
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


class HiPolicy:
    def __init__(self, host: str = "ws://127.0.0.1:8000", policy: str = "g1-ski"):
        import msgpack
        from websockets.sync.client import connect

        self._packb = functools.partial(msgpack.packb, default=_pack_array)
        self._unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
        base = host.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        if not base.startswith("ws"):
            base = "wss://" + base
        self.uri = f"{base}/v1/policies/{policy}/ws"
        self.ws = connect(self.uri, compression=None, max_size=None)
        self.metadata = self._unpackb(self.ws.recv())

    def infer(self, obs: dict) -> dict:
        self.ws.send(self._packb({k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}))
        reply = self.ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"policy server error:\n{reply}")
        return self._unpackb(reply)

    def close(self):
        self.ws.close()
