"""Check every path to the policy against rsl_rl's own actions on a real rollout (parity.npz).

    policy_api/.venv/bin/python policy_api/test_api.py http://127.0.0.1:8791
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hi_client import HiPolicy, pull  # noqa: E402
from hi_server import POLICIES  # noqa: E402

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8791"
ref = np.load(Path(__file__).parent / "parity.npz")
obs, act = ref["obs"].astype(np.float32), np.clip(ref["actions"], -3, 3)


def report(name, a):
    err = np.abs(a - act).max()
    print(f"{name:34s} max |diff| vs rsl_rl = {err:.2e}  {'OK' if err < 1e-4 else 'MISMATCH'}")


# 1. numpy inference in the server process
report("server numpy (g1-ski-trees)", POLICIES["g1-ski-trees"].infer({"obs": obs})["actions"])

# 2. HTTP
body = json.dumps({"obs": obs[:50].tolist()}).encode()
req = urllib.request.Request(f"{base}/v1/policies/g1-ski-trees/infer", body, {"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req).read())
err = np.abs(np.array(r["actions"]) - act[:50]).max()
print(f"{'HTTP POST /infer (50 obs batch)':34s} max |diff| vs rsl_rl = {err:.2e}  {'OK' if err < 1e-4 else 'MISMATCH'}")

# 3. websocket, one step at a time like a control loop
ws = HiPolicy(base, "g1-ski-trees")
t0 = time.perf_counter()
a = np.stack([ws.infer({"obs": o})["actions"] for o in obs])
dt = (time.perf_counter() - t0) / len(obs) * 1e3
report(f"websocket per step ({dt:.2f} ms/step)", a)
ws.close()

# 4. pull: download ONNX + meta, run locally
p = pull(base, "g1-ski-trees", refresh=True)
t0 = time.perf_counter()
a = np.stack([p.infer(o) for o in obs])
dt = (time.perf_counter() - t0) / len(obs) * 1e3
report(f"pull() + local ONNX ({dt:.3f} ms/step)", a)
print("architecture:", p.meta["architecture"]["layers"], p.meta["architecture"]["activation"])

# 5. the 141-input policy: same layout without the tree rays, numpy vs its ONNX
q = pull(base, "g1-ski", refresh=True)
x = obs[:, :141]
d = np.abs(q.infer(x) - POLICIES["g1-ski"].infer({"obs": x})["actions"]).max()
print(f"{'g1-ski numpy vs ONNX':34s} max |diff| = {d:.2e}  {'OK' if d < 1e-4 else 'MISMATCH'}")

# 6. openpi_client compatibility, if installed
try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    host = base.replace("http://", "ws://").replace("https://", "wss://")
    c = WebsocketClientPolicy(host=host + "/?policy=g1-ski-trees")
    a = np.stack([c.infer({"obs": o})["actions"] for o in obs[:20]])
    e = np.abs(a - act[:20]).max()
    print(f"{'openpi_client WebsocketClientPolicy':34s} max |diff| vs rsl_rl = {e:.2e}  {'OK' if e < 1e-4 else 'MISMATCH'}")
except ImportError:
    print("openpi_client not installed, skipped")
