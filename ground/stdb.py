"""Writes hikes and the app's people into the team's SpacetimeDB (module: spacetime/spacetimedb/src/index.ts) over its
HTTP API.

The phone server owns one SpacetimeDB identity, saved in out/ground/stdb_identity.json, and claims the module's
hike writer slot with it at startup; the hike reducers refuse every other identity. The same identity writes the app's
users, invites, messages, sessions and raw phone recordings as a service of the app's organisation (GT_TENANT,
default ground-truth-app). Every failure raises StdbError with what the database said.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import requests

CHUNK = 60  # seconds per hike_chunk row
MAX_VALUES = 40_000  # f32 values per recording chunk: about 400 KB of JSON, well under the 2 MiB request cap


class StdbError(RuntimeError):
    pass


class Stdb:
    def __init__(self, base: str, db: str, identity_path: Path):
        self.base, self.db, self.identity_path = base.rstrip("/"), db, Path(identity_path)
        self._tok: str | None = None

    def __repr__(self):
        return f"{self.base}/{self.db}"

    def token(self) -> str:
        if self._tok:
            return self._tok
        saved = json.loads(self.identity_path.read_text()) if self.identity_path.exists() else {}
        if saved.get(self.base):  # identities are per server: local and maincloud each get their own
            self._tok = saved[self.base]
            return self._tok
        r = requests.post(f"{self.base}/v1/identity", timeout=20)
        if r.status_code != 200:
            raise StdbError(f"could not create an identity on {self.base}: HTTP {r.status_code} {r.text[:200]}")
        saved[self.base] = self._tok = r.json()["token"]
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        self.identity_path.write_text(json.dumps(saved))
        return self._tok

    def identity(self) -> str:
        """This server's identity as 0x-prefixed hex, the form reducers take (read from the token's claims)."""
        import base64
        payload = self.token().split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return "0x" + claims["hex_identity"]

    def call(self, reducer: str, args: dict):
        try:
            r = requests.post(f"{self.base}/v1/database/{self.db}/call/{reducer}", data=json.dumps(args, allow_nan=False),
                              timeout=60, headers={"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"})
        except requests.RequestException as e:
            raise StdbError(f"{reducer}: spacetimedb unreachable at {self.base}: {e}") from e
        if r.status_code != 200:
            raise StdbError(f"{reducer}: HTTP {r.status_code} {r.text.strip()[:300]}")

    def sql(self, query: str) -> list[dict]:
        r = requests.post(f"{self.base}/v1/database/{self.db}/sql", data=query, timeout=30,
                          headers={"Authorization": f"Bearer {self.token()}"})
        if r.status_code != 200:
            raise StdbError(f"sql: HTTP {r.status_code} {r.text.strip()[:300]}")
        out = []
        for stmt in r.json():
            cols = [c["name"]["some"] if isinstance(c.get("name"), dict) else c.get("name")
                    for c in stmt["schema"]["elements"]]
            out += [dict(zip(cols, row)) for row in stmt["rows"]]
        return out

    # ------------------------------------------------------------------ hike reducers
    def claim(self):
        self.call("claim_hike_writer", {})

    def record_hike(self, **fields):
        self.call("record_hike", fields)

    def push_samples(self, key: str, samples: list[dict]) -> int:
        n = 0
        for seq, i in enumerate(range(0, len(samples), CHUNK)):
            self.call("push_hike_chunk", {"hikeKey": key, "seq": seq, "samples": samples[i:i + CHUNK]})
            n += 1
        return n

    def label(self, key: str, note: str):
        self.call("label_hike", {"key": key, "note": note})

    def delete_hikes(self, join_code: str):
        self.call("delete_hikes", {"joinCode": join_code})

    # ------------------------------------------------------------------ the app's people
    def member_of(self, tenant: str) -> str | None:
        """This identity's role in the organisation, or None."""
        rows = self.sql(f"SELECT role FROM my_tenant WHERE id = '{tenant}'")
        return rows[0]["role"] if rows else None

    def join(self, invite_code: str):
        self.call("redeem_invite", {"code": invite_code, "name": "ground truth phone server"})

    def app_user(self, tenant: str, handle: str, code: str = "", **f):
        self.call("upsert_app_user", {"handle": handle, "tenant": tenant, "code": code or "", "name": f.get("name", ""),
                                      "optedOut": bool(f.get("optedOut")), "pendingLabel": f.get("pendingLabel") or "",
                                      "pendingDelete": bool(f.get("pendingDelete")), "invites": int(f.get("invites") or 0),
                                      "notes": [str(n) for n in f.get("notes") or []], "firstSeenMs": int(f.get("firstSeenMs") or 0)})

    def app_message(self, tenant: str, key: str, handle: str, direction: str, kind: str, text: str, session: str = "",
                    status: str = "", error: str = "", at: float = 0):
        self.call("upsert_app_message", {"key": key, "handle": handle or "", "tenant": tenant, "direction": direction,
                                         "kind": kind or "", "text": text or "", "session": session or "", "status": status or "",
                                         "error": error or "", "atMs": int(at * 1000)})

    def app_invite(self, tenant: str, token: str, handle: str, code: str, link: str, created: float = 0):
        self.call("upsert_app_invite", {"token": token, "tenant": tenant, "handle": handle or "", "code": code or "",
                                        "link": link or "", "createdAtMs": int(created * 1000)})

    def app_session(self, tenant: str, sid: str, handle: str, code: str, token: str, status: str, consent: dict | None = None,
                    meta: dict | None = None, summary: dict | None = None, labels: list[str] | None = None, error: str = "",
                    created: float = 0):
        self.call("upsert_app_session", {"id": sid, "tenant": tenant, "handle": handle or "", "code": code or "", "token": token or "",
                                         "status": status, "consent": json.dumps(consent or {}), "meta": json.dumps(meta or {}),
                                         "summary": json.dumps(summary or {}, default=str), "labels": labels or [],
                                         "error": error or "", "createdAtMs": int(created * 1000)})

    def delete_app_user(self, tenant: str, handle: str):
        self.call("delete_app_user", {"tenant": tenant, "handle": handle})

    def phone_recordings(self, tenant: str, sid: str, data: dict[str, np.ndarray], meta: dict) -> int:
        """The raw streams of a session as recordings phone/<sid>/imu|gps|baro (ground/schema.py layouts). GPS becomes
        metres east and north of the first fix, since f32 cannot hold a latitude to the metre. Returns frames written."""
        imu, gps, baro = data.get("imu"), data.get("gps"), data.get("baro")
        t0 = min(float(a["t"][0]) for a in (imu, gps, baro) if a is not None and len(a))
        base = {"session": sid, "device": meta.get("device"), "placement": meta.get("placement"), "start_unix_s": t0,
                "layout": "ground/schema.py", "writer": "ground/server.py"}
        n = 0
        if imu is not None and len(imu):
            cols = np.column_stack([imu["acc"], imu["grav"], imu["gyro"], imu["quat"], imu["mag"]])
            n += self._recording(tenant, f"phone/{sid}/imu", sid, ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
                                  "gyro_x", "gyro_y", "gyro_z", "quat_x", "quat_y", "quat_z", "quat_w", "mag_x", "mag_y", "mag_z"],
                                 ["m/s2"] * 6 + ["rad/s"] * 3 + ["unit"] * 4 + ["uT"] * 3, imu["t"] - t0, cols,
                                 "iPhone motion, 100 Hz", {**base, "mag": "0 until the magnetometer is calibrated"})
        if gps is not None and len(gps):
            lat0, lon0 = float(gps["lat"][0]), float(gps["lon"][0])
            east = np.radians(gps["lon"] - lon0) * 6371008.8 * math.cos(math.radians(lat0))
            north = np.radians(gps["lat"] - lat0) * 6371008.8
            cols = np.column_stack([east, north, gps["alt"], gps["hacc"], gps["vacc"], gps["speed"], gps["course"]])
            n += self._recording(tenant, f"phone/{sid}/gps", sid, ["east", "north", "alt", "hacc", "vacc", "speed", "course"],
                                 ["m", "m", "m", "m", "m", "m/s", "deg"], gps["t"] - t0, cols, "iPhone GPS",
                                 {**base, "origin_lat": lat0, "origin_lon": lon0})
        if baro is not None and len(baro):
            cols = np.column_stack([baro["rel_alt"], baro["kpa"], baro["abs_alt"], baro["abs_acc"]])
            n += self._recording(tenant, f"phone/{sid}/baro", sid, ["rel_alt", "kpa", "abs_alt", "abs_acc"],
                                 ["m", "kPa", "m", "m"], baro["t"] - t0, cols, "iPhone barometer",
                                 {**base, "pairs": "relative and absolute altitude arrive separately: the other pair is 0"})
        return n

    def _recording(self, tenant: str, key: str, ref: str, channels: list[str], units: list[str], t_rel: np.ndarray,
                   frames: np.ndarray, title: str, meta: dict) -> int:
        frames = np.asarray(frames, dtype=np.float64)
        bad = ~np.isfinite(frames)
        if bad.any():
            meta = {**meta, "non_finite_values_set_to_0": int(bad.sum())}
            frames = np.where(bad, 0.0, frames)
        self.call("put_recording", {"key": key, "tenant": tenant, "source": "phone", "activity": "hike", "ref": ref, "uri": "",
                                    "channels": channels, "units": units, "rateHz": 0.0, "title": title, "license": "",
                                    "meta": json.dumps(meta, default=str)})
        t_ms = np.round(np.asarray(t_rel) * 1000).astype(np.int64)
        per = max(1, MAX_VALUES // len(channels))
        for seq, i in enumerate(range(0, len(frames), per)):
            base = int(t_ms[i])
            self.call("push_recording_chunk", {"recordingKey": key, "seq": seq, "t0Ms": base,
                                               "tMs": [int(v) - base for v in t_ms[i:i + per]],
                                               "data": np.round(frames[i:i + per].ravel(), 5).tolist()})
        self.call("close_recording", {"key": key})
        return len(frames)
