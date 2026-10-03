"""Ground Truth server: invites, consent, uploads from the iOS recorder, registration, and the outbox the Photon
agent sends from.

    .venv/bin/python -m ground.server --public https://<name>.trycloudflare.com
    cloudflared tunnel --url http://127.0.0.1:8770        # in another terminal; gives the https URL above

- POST /api/invite {phone?}             new token + link (the agent calls this when someone texts "hike")
- GET  /g/<token>                       the page the text links to; one button opens the app
- POST /api/consent {token, session_id, ...}
- PUT  /api/session/<id>/file/<name>    raw file bytes, header X-Token. meta.json lists every file and its size;
                                        the session is finalized once all of them are here
- GET  /api/session/<id>                status and summary
- GET  /api/outbox, POST /api/outbox/<id>/sent|failed   messages waiting for the Photon agent
- GET  /                                dashboard (dashboard and outbox answer only on this Mac, not the tunnel)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import secrets
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from ground.schema import FILE_RE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "ground"
PORT = 8770
CONSENT_VERSION = "gt-consent-1"
SID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")

LOCK = threading.RLock()
PUBLIC = {"url": None}


def _load(name: str, default):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def _save(name: str, obj, d: Path = OUT):
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(d / name)


def new_invite(phone: str | None) -> dict:
    if not PUBLIC["url"]:
        raise RuntimeError("public url not set: start the server with --public https://<your tunnel>")
    with LOCK:
        inv = _load("invites.json", {})
        token = secrets.token_urlsafe(9)
        inv[token] = {"phone": phone, "created": time.time(), "link": f"{PUBLIC['url']}/g/{token}"}
        _save("invites.json", inv)
        return {"token": token, **inv[token]}


def enqueue(to: str | None, text: str, session: str | None = None) -> dict | None:
    if not to:
        return None
    with LOCK:
        box = _load("outbox.json", [])
        msg = {"id": secrets.token_hex(6), "to": to, "text": text, "session": session, "status": "pending",
               "created": time.time()}
        box.append(msg)
        _save("outbox.json", box)
        return msg


def session_status(sid: str) -> dict:
    d = OUT / "sessions" / sid
    if not d.exists():
        return {"session": sid, "status": "unknown"}
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"status": "consented"}
    st["session"] = sid
    if (d / "summary.json").exists():
        st["summary"] = {k: v for k, v in json.loads((d / "summary.json").read_text()).items() if k != "track_trimmed"}
    return st


def try_finalize(sid: str):
    """Once meta.json and every file it lists (at its listed size) are here, register the session in a thread."""
    d = OUT / "sessions" / sid
    if not (d / "meta.json").exists():
        return
    meta = json.loads((d / "meta.json").read_text())
    missing = [n for n, size in meta["files"].items() if not (d / n).exists() or (d / n).stat().st_size != size]
    with LOCK:
        st = session_status(sid)
        if st.get("status") in ("registering", "done", "failed"):
            return
        if missing:
            _save("status.json", {"status": "uploading", "missing": missing}, d)
            return
        _save("status.json", {"status": "registering"}, d)
    threading.Thread(target=_finalize, args=(sid,), daemon=True).start()


def _finalize(sid: str):
    from ground.register import message, register  # heavy imports (scipy, PIL, mujoco via course) only when needed
    d = OUT / "sessions" / sid
    consent = json.loads((d / "consent.json").read_text())
    phone = _load("invites.json", {}).get(consent["token"], {}).get("phone")
    try:
        s = register(d)
        _save("summary.json", s, d)
        text = message(s)
        msg = enqueue(phone, text, sid)
        _save("status.json", {"status": "done", "message": text, "outbox_id": msg and msg["id"],
                              "message_status": "pending" if msg else "no phone on invite, nothing to send"}, d)
        print(f"[ground] {sid} done\n{text}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        _save("status.json", {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": tb}, d)
        print(f"[ground] {sid} REGISTRATION FAILED\n{tb}")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ground truth</title><style>
:root{--ink:#1a1a1a;--mute:#8a8a8a;--rule:#d8d8d4;--bg:#f7f7f4}
body{margin:0;background:var(--bg);color:var(--ink);font:300 17px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:420px;margin:0 auto;padding:56px 24px}
h1{font-weight:300;font-size:28px;margin:0 0 8px;letter-spacing:-.01em}
p{color:var(--mute);margin:0 0 28px}
a.b{display:block;text-align:center;padding:14px;border:1px solid var(--ink);color:var(--ink);text-decoration:none;margin-bottom:14px}
.s{font-size:13px;color:var(--mute);border-top:1px solid var(--rule);padding-top:16px;margin-top:28px}
</style></head><body><main>
<h1>ground truth</h1>
<p>record your hike with your phone's motion sensors so rescue robots can learn to move on real ground. you choose, you can delete it anytime.</p>
<a class="b" href="__APP__">open ground truth</a>
<div class="s">nothing happens? the app isn't installed on this phone yet. ask the team for the install link.</div>
</main></body></html>"""


DASH = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ground truth</title><meta http-equiv="refresh" content="5"><style>
body{margin:0;background:#f7f7f4;color:#1a1a1a;font:300 14px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 20px}h1{font-weight:300;font-size:22px;margin:0 0 4px}
.m{color:#8a8a8a;margin-bottom:24px}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #d8d8d4;padding:8px 10px 8px 0;text-align:left;vertical-align:top;font-weight:300}
th{color:#8a8a8a}.bad{color:#b3261e}.w{overflow-x:auto}pre{margin:0;white-space:pre-wrap;font:12px ui-monospace,monospace}
</style></head><body><main><h1>ground truth</h1><div class="m">__INFO__</div>
<div class="w"><table><tr><th>session</th><th>status</th><th>motion</th><th>gaps</th><th>summary / errors</th><th>text</th></tr>__ROWS__</table></div>
<h1 style="margin-top:36px">outbox</h1><div class="w"><table><tr><th>to</th><th>status</th><th>text</th></tr>__OUT__</table></div>
</main></body></html>"""


def dashboard() -> str:
    rows = []
    sdir = OUT / "sessions"
    for d in sorted(sdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if sdir.exists() else []:
        st = session_status(d.name)
        s = st.get("summary") or {}
        status = st.get("status", "?")
        bad = status == "failed" or s.get("errors") or s.get("gaps", {}).get("count")
        detail = st.get("error") or "; ".join(s.get("errors", [])) or (
            f"{s.get('distance_m', '?')} m, trails {((s.get('trail_match') or {}).get('top') or [])[:2]}" if s else
            ("missing " + ", ".join(st.get("missing", [])) if st.get("missing") else ""))
        imu = s.get("imu", {})
        g = s.get("gaps", {})
        rows.append(f"<tr><td>{html.escape(d.name[:8])}</td><td class='{'bad' if status == 'failed' else ''}'>{status}</td>"
                    f"<td>{imu.get('samples', '')} @ {imu.get('rate_hz', '')} hz</td>"
                    f"<td class='{'bad' if g.get('count') else ''}'>{g.get('count', '')} ({g.get('total_s', '')} s)</td>"
                    f"<td class='{'bad' if bad else ''}'>{html.escape(str(detail))}</td>"
                    f"<td>{html.escape(str(st.get('message_status', '')))}</td></tr>")
    outs = []
    for m in reversed(_load("outbox.json", [])[-30:]):
        cls = "bad" if m["status"] != "sent" else ""
        outs.append(f"<tr><td>{html.escape(m['to'])}</td><td class='{cls}'>{html.escape(m['status'])}"
                    f"{html.escape(' ' + m.get('error', '')) if m.get('error') else ''}</td>"
                    f"<td><pre>{html.escape(m['text'])}</pre></td></tr>")
    agent = _load("agent.json", {})
    age = time.time() - agent.get("seen", 0)
    info = (f"public url {PUBLIC['url'] or 'NOT SET'} · photon agent "
            + (f"polled {age:.0f} s ago" if age < 30 else "<span class='bad'>not running</span>"))
    return DASH.replace("__INFO__", info).replace("__ROWS__", "".join(rows)).replace("__OUT__", "".join(outs))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print(f"[ground] {self.command} {self.path} -> {a[1] if len(a) > 1 else ''}")

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, code: int, msg: str):
        print(f"[ground] ERROR {code} {self.path}: {msg}")
        self._json({"error": msg}, code)

    def _local(self) -> bool:
        """Dashboard and outbox hold phone numbers: only from this Mac, never through the tunnel."""
        return "CF-Connecting-IP" not in self.headers and self.client_address[0] in ("127.0.0.1", "::1")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/api/outbox") and not self._local():
            return self._err(404, "not found")
        if p == "/":
            self._send(200, dashboard().encode(), "text/html; charset=utf-8")
        elif m := re.fullmatch(r"/g/([A-Za-z0-9_-]+)", p):
            if m.group(1) not in _load("invites.json", {}):
                return self._send(404, b"<h1>this link has expired or is wrong</h1>", "text/html; charset=utf-8")
            app = f"groundtruth://join?token={m.group(1)}&server={quote(PUBLIC['url'] or '', safe='')}"
            self._send(200, PAGE.replace("__APP__", html.escape(app)).encode(), "text/html; charset=utf-8")
        elif m := re.fullmatch(r"/api/session/([A-Za-z0-9-]+)", p):
            self._json(session_status(m.group(1)))
        elif p == "/api/outbox":
            with LOCK:
                _save("agent.json", {"seen": time.time()})
                self._json([m for m in _load("outbox.json", []) if m["status"] == "pending"])
        else:
            self._err(404, "not found")

    def do_POST(self):
        p = self.path.split("?")[0]
        try:
            body = json.loads(self._body() or b"{}")
        except json.JSONDecodeError:
            return self._err(400, "body is not json")
        if p == "/api/invite":
            try:
                self._json(new_invite(body.get("phone")))
            except RuntimeError as e:
                self._err(500, str(e))
        elif p == "/api/consent":
            tok, sid = body.get("token"), body.get("session_id", "")
            if tok not in _load("invites.json", {}):
                return self._err(403, "unknown invite token")
            if not SID_RE.match(sid):
                return self._err(400, "bad session_id")
            if body.get("consent_version") != CONSENT_VERSION:
                return self._err(400, f"consent_version must be {CONSENT_VERSION}")
            rec = {**body, "received": time.time(), "ip": self.headers.get("CF-Connecting-IP") or self.client_address[0]}
            _save("consent.json", rec, OUT / "sessions" / sid)
            self._json({"ok": True, "session_id": sid})
        elif p.startswith("/api/outbox/") and not self._local():
            self._err(404, "not found")
        elif m := re.fullmatch(r"/api/outbox/([0-9a-f]+)/(sent|failed)", p):
            with LOCK:
                box = _load("outbox.json", [])
                for msg in box:
                    if msg["id"] == m.group(1):
                        msg["status"] = m.group(2)
                        msg["error"] = body.get("error")
                        msg["at"] = time.time()
                        if msg.get("session"):
                            d = OUT / "sessions" / msg["session"]
                            st = json.loads((d / "status.json").read_text())
                            st["message_status"] = m.group(2) + (f": {body['error']}" if body.get("error") else "")
                            _save("status.json", st, d)
                        break
                else:
                    return self._err(404, "no such message")
                _save("outbox.json", box)
            self._json({"ok": True})
        else:
            self._err(404, "not found")

    def do_PUT(self):
        m = re.fullmatch(r"/api/session/([A-Za-z0-9-]+)/file/([A-Za-z0-9_.]+)", self.path.split("?")[0])
        if not m:
            return self._err(404, "not found")
        sid, name = m.groups()
        d = OUT / "sessions" / sid
        if not (d / "consent.json").exists():
            return self._err(403, "no consent on file for this session")
        if json.loads((d / "consent.json").read_text())["token"] != self.headers.get("X-Token"):
            return self._err(403, "token does not match this session")
        if not (FILE_RE.match(name) or name == "meta.json"):
            return self._err(400, f"unexpected file name {name}")
        data = self._body()
        if name == "meta.json":
            try:
                meta = json.loads(data)
                assert isinstance(meta.get("files"), dict) and meta.get("session_id") == sid
            except Exception:  # noqa: BLE001
                return self._err(400, "meta.json must be json with session_id and files")
        tmp = d / (name + ".part")
        tmp.write_bytes(data)
        tmp.replace(d / name)
        try_finalize(sid)
        self._json({"ok": True, "bytes": len(data), **{k: v for k, v in session_status(sid).items() if k != "summary"}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", help="https base URL phones reach this server at (the cloudflared tunnel)")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    PUBLIC["url"] = a.public.rstrip("/") if a.public else None
    OUT.mkdir(parents=True, exist_ok=True)
    if not PUBLIC["url"]:
        print("[ground] WARNING: no --public url, invites will fail until you restart with one")
    print(f"[ground] http://127.0.0.1:{a.port}  public {PUBLIC['url']}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
