"""Writes hikes into the team's SpacetimeDB (module: spacetime/spacetimedb/src/index.ts) over its HTTP API.

The phone server owns one SpacetimeDB identity, saved in out/ground/stdb_identity.json, and claims the module's
hike writer slot with it at startup; the hike reducers refuse every other identity. Every failure raises
StdbError with what the database said.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

CHUNK = 60  # seconds per hike_chunk row


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

    def call(self, reducer: str, args: dict):
        try:
            r = requests.post(f"{self.base}/v1/database/{self.db}/call/{reducer}", json=args, timeout=30,
                              headers={"Authorization": f"Bearer {self.token()}"})
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
