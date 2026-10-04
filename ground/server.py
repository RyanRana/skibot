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
- GET  /api/user/<phone>                that person's hikes (the agent answers "how did my hike go" from this)
- POST /api/trigger {phone}             new invite + a "want to record this one?" text (later: the app's geofence)
- POST /api/label {session, text}       something the hiker told the agent about a hike ("slipped near the creek")
- POST /api/delete {phone}              removes that person's hikes, invites and messages
- GET  /                                dashboard
Everything that touches phone numbers (dashboard, outbox, user, trigger, label, delete) answers only on this Mac,
never through the tunnel.
"""
from __future__ import annotations

import argparse
import html
import json
import os
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
STDB = {"db": None}  # ground.stdb.Stdb when --stdb is given: hikes go to the team's SpacetimeDB


def _load(name: str, default):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def _save(name: str, obj, d: Path = OUT):
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(d / name)


def new_invite(phone: str | None, code: str | None = None) -> dict:
    """code: the person's join code from the agent. SpacetimeDB only ever sees the code, never the phone."""
    if not PUBLIC["url"]:
        raise RuntimeError("public url not set: start the server with --public https://<your tunnel>")
    with LOCK:
        inv = _load("invites.json", {})
        token = secrets.token_urlsafe(9)
        inv[token] = {"phone": phone, "code": (code or "").strip().upper() or None, "created": time.time(),
                      "link": f"{PUBLIC['url']}/g/{token}"}
        _save("invites.json", inv)
        return {"token": token, **inv[token]}


def enqueue(to: str | None, text: str, session: str | None = None, kind: str = "summary") -> dict | None:
    """kind: "summary" (hike registered, the agent follows up with a question) or "nudge" (invite to record)."""
    if not to:
        return None
    with LOCK:
        box = _load("outbox.json", [])
        msg = {"id": secrets.token_hex(6), "to": to, "text": text, "session": session, "kind": kind,
               "status": "pending", "created": time.time()}
        box.append(msg)
        _save("outbox.json", box)
        return msg


def same_phone(a: str | None, b: str | None) -> bool:
    """+1 (919) 555-0123, 19195550123 and +19195550123 are one person; emails compare case-insensitively."""
    if not a or not b:
        return False
    if "@" in a or "@" in b:
        return a.strip().lower() == b.strip().lower()
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    return da == db or (len(da) >= 10 and len(db) >= 10 and da[-10:] == db[-10:])


def tokens_of(phone: str) -> set[str]:
    return {t for t, v in _load("invites.json", {}).items() if same_phone(v.get("phone"), phone)}


def sessions_of(phone: str) -> list[str]:
    toks = tokens_of(phone)
    sdir = OUT / "sessions"
    out = []
    for d in sdir.iterdir() if sdir.exists() else []:
        c = d / "consent.json"
        if c.exists() and json.loads(c.read_text()).get("token") in toks:
            out.append(d.name)
    return out


def user_hikes(phone: str) -> list[dict]:
    """Newest first: what the agent may tell a person about their own hikes. Numbers only from the summary."""
    hikes = []
    for sid in sessions_of(phone):
        st = session_status(sid)
        s = st.get("summary") or {}
        top = [n for n, _ in (s.get("trail_match") or {}).get("top") or [] if not n.startswith("unnamed")]
        lp = OUT / "sessions" / sid / "labels.json"
        labels = json.loads(lp.read_text()) if lp.exists() else []
        hikes.append({"session": sid, "status": st.get("status"), "started": s.get("started"),
                      "duration_s": s.get("duration_s"), "distance_m": s.get("distance_m"),
                      "climb_m": (s.get("climb_baro_m") or s.get("climb_dem_m") or {}).get("gain"),
                      "trail": top[0] if top else None, "motion_samples": (s.get("imu") or {}).get("samples"),
                      "gaps": (s.get("gaps") or {}).get("count"), "cadence_spm": s.get("cadence_spm"),
                      "labels": [x["text"] for x in labels], "error": st.get("error")})
    return sorted(hikes, key=lambda h: h["started"] or 0, reverse=True)


def delete_user(phone: str) -> dict:
    import shutil
    with LOCK:
        if STDB["db"]:  # the database first: if it fails, nothing local is gone and the hiker can retry
            for code in {v.get("code") for v in _load("invites.json", {}).values()
                         if same_phone(v.get("phone"), phone) and v.get("code")}:
                STDB["db"].delete_hikes(code)
        sids = sessions_of(phone)
        for sid in sids:
            shutil.rmtree(OUT / "sessions" / sid)
        inv = _load("invites.json", {})
        gone = [t for t, v in inv.items() if same_phone(v.get("phone"), phone)]
        for t in gone:
            del inv[t]
        _save("invites.json", inv)
        box = _load("outbox.json", [])
        kept = [m for m in box if not same_phone(m["to"], phone)]
        _save("outbox.json", kept)
    print(f"[ground] deleted {len(sids)} hikes, {len(gone)} invites, {len(box) - len(kept)} messages for a user")
    return {"hikes": len(sids), "invites": len(gone), "messages": len(box) - len(kept)}


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
    invite = _load("invites.json", {}).get(consent["token"], {})
    phone, code = invite.get("phone"), invite.get("code")
    try:
        s = register(d)
        _save("summary.json", s, d)
        text = message(s)
        st = {"status": "done", "message": text}
        if STDB["db"]:
            try:
                n = to_stdb(d, s, code, text)
                st["stdb"] = f"recorded in {STDB['db']} ({n} chunks)"
                st["message_status"] = "queued in spacetimedb outbox" if code else "no join code on invite, nothing to send"
            except Exception as e:  # noqa: BLE001  the hike is safe on disk; the dashboard shows this in red
                st["stdb_error"] = f"{type(e).__name__}: {e}"
                print(f"[ground] {sid} SPACETIMEDB WRITE FAILED: {e}")
        if "message_status" not in st:  # no database, or it failed: text through the local outbox
            msg = enqueue(phone, text, sid)
            st["outbox_id"] = msg and msg["id"]
            st["message_status"] = "pending" if msg else "no phone on invite, nothing to send"
        _save("status.json", st, d)
        print(f"[ground] {sid} done\n{text}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        _save("status.json", {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": tb}, d)
        print(f"[ground] {sid} REGISTRATION FAILED\n{tb}")


def to_stdb(d: Path, s: dict, code: str | None, text: str) -> int:
    """The registered hike and its per-second rows, written to SpacetimeDB. Returns the number of chunks."""
    from ground.motion import per_second
    from ground.schema import load_session
    meta, data, _ = load_session(d)
    tm = s.get("trail_match") or {}
    named = [n for n, _ in tm.get("top") or [] if not n.startswith("unnamed")]
    climb = (s.get("climb_baro_m") or s.get("climb_dem_m") or {}).get("gain") or 0.0
    db = STDB["db"]
    db.record_hike(key=d.name, joinCode=code or "", placement=s.get("placement") or "", startedS=s.get("started") or 0,
                   endedS=s.get("ended") or 0, distanceM=s.get("distance_m") or 0.0, climbM=climb,
                   trail=named[0] if named else "", sacScale=(tm.get("sac_scale") or [[""]])[0][0],
                   surface=(tm.get("surface") or [[""]])[0][0], motionSamples=s["imu"]["samples"],
                   rateHz=s["imu"]["rate_hz"], gaps=s["gaps"]["count"], gapS=s["gaps"]["total_s"],
                   cadenceSpm=s.get("cadence_spm") or 0.0, message=text if code else "")
    rows = per_second(meta, data)
    return db.push_samples(d.name, rows) if rows else 0


# The Hazard Intelligence look (web/site.css), so the page the text links to feels like the site.
SITE_CSS = """
:root{--bg:#fff;--panel:#f7f7f5;--ink:#111315;--muted:#5f656b;--faint:#9a9fa5;--line:#e6e7e8;
--serif:"Times New Roman",Times,serif;--sans:"Inter",system-ui,-apple-system,sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:400 16px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.wrap{max-width:560px;margin:0 auto;padding:0 clamp(20px,6vw,32px)}
header{display:flex;align-items:baseline;justify-content:space-between;padding:22px 0}
.logo{font:400 34px/1 var(--serif);letter-spacing:-.01em}
.k{font:500 11px/1 var(--mono);letter-spacing:.14em;color:var(--faint);text-transform:uppercase}
h1{font:400 clamp(40px,11vw,56px)/1.02 var(--serif);letter-spacing:-.02em;margin-top:clamp(28px,7vh,64px)}
.lede{margin-top:18px;color:var(--muted);font-size:17px}
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Truth · Hazard Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>__CSS__
.list{margin-top:36px;border-top:1px solid var(--ink)}
.list div{display:flex;justify-content:space-between;gap:20px;align-items:baseline;padding:16px 0;border-bottom:1px solid var(--line)}
.list b{font:400 21px/1.15 var(--serif)}
.list span{color:var(--muted);font-size:15px;text-align:right}
a.go{display:block;margin-top:32px;padding:16px;text-align:center;background:var(--ink);color:var(--bg);font-weight:500}
.foot{margin:28px 0 48px;padding-top:18px;border-top:1px solid var(--line);color:var(--faint);font-size:13px}
</style></head><body><div class="wrap">
<header><span class="logo">Hi.</span><span class="k">Ground Truth</span></header>
<h1>Your hike can teach a rescue robot.</h1>
<p class="lede">Your phone records how you move on real ground while you hike. We rebuild that ground in physics and train robots on it.</p>
<div class="list">
<div><b>Recorded</b><span>Motion, GPS, altitude, steps</span></div>
<div><b>Only while</b><span>You're on a hike you started</span></div>
<div><b>Never</b><span>Contacts, photos, audio</span></div>
<div><b>Delete</b><span>Anytime, just text us</span></div>
</div>
<a class="go" href="__APP__">Open Ground Truth</a>
<p class="foot">Nothing happens? The app isn't on this phone yet. Ask the team for the install link.</p>
</div></body></html>""".replace("__CSS__", SITE_CSS)

EXPIRED = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Truth · Hazard Intelligence</title><style>__CSS__</style></head><body><div class="wrap">
<header><span class="logo">Hi.</span><span class="k">Ground Truth</span></header>
<h1>This link has expired.</h1><p class="lede">Text us "hike" and we'll send you a fresh one.</p>
</div></body></html>""".replace("__CSS__", SITE_CSS)


DASH = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Truth · dashboard</title><meta http-equiv="refresh" content="5">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>__CSS__
.wrap{max-width:1120px}
h1{margin-top:28px;font-size:clamp(36px,5vw,56px)}
h2{font:400 30px/1.1 var(--serif);margin-top:56px}
.m{margin-top:14px;color:var(--muted);font-size:14px}
.w{overflow-x:auto;margin-top:18px;border-top:1px solid var(--ink)}
table{border-collapse:collapse;width:100%;font-size:14px}
td,th{border-bottom:1px solid var(--line);padding:12px 14px 12px 0;text-align:left;vertical-align:top;font-weight:400}
th{font:500 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--faint);padding-top:16px}
td:first-child{font-family:var(--mono);font-size:12.5px}
.bad{color:#b3261e}
pre{margin:0;white-space:pre-wrap;font:12.5px/1.6 var(--mono);color:var(--muted)}
footer{margin:56px 0 40px;padding-top:18px;border-top:1px solid var(--line);color:var(--faint);font-size:13px}
</style></head><body><div class="wrap">
<header><span class="logo">Hi.</span><span class="k">Ground Truth · dashboard</span></header>
<h1>Hikes</h1><div class="m">__INFO__</div>
<div class="w"><table><tr><th>Session</th><th>Status</th><th>Motion</th><th>Gaps</th><th>Summary / errors</th><th>Text</th></tr>__ROWS__</table></div>
<h2>Outbox</h2>
<div class="w"><table><tr><th>To</th><th>Status</th><th>Text</th></tr>__OUT__</table></div>
<footer>Refreshes every 5 s. Only visible on this Mac.</footer>
</div></body></html>""".replace("__CSS__", SITE_CSS)


def dashboard() -> str:
    rows = []
    sdir = OUT / "sessions"
    for d in sorted(sdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if sdir.exists() else []:
        st = session_status(d.name)
        s = st.get("summary") or {}
        status = st.get("status", "?")
        bad = status == "failed" or s.get("errors") or s.get("gaps", {}).get("count") or st.get("stdb_error")
        detail = st.get("error") or (st.get("stdb_error") and "SPACETIMEDB WRITE FAILED: " + st["stdb_error"]) or \
            "; ".join(s.get("errors", [])) or (
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
    info = (f"public url {PUBLIC['url'] or 'NOT SET'} · spacetimedb {STDB['db'] or 'off (hikes stay on this mac)'} · photon agent "
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
        if (p in ("/", "/api/outbox") or p.startswith("/api/user/")) and not self._local():
            return self._err(404, "not found")
        if p == "/":
            self._send(200, dashboard().encode(), "text/html; charset=utf-8")
        elif m := re.fullmatch(r"/g/([A-Za-z0-9_-]+)", p):
            if m.group(1) not in _load("invites.json", {}):
                return self._send(404, EXPIRED.encode(), "text/html; charset=utf-8")
            app = f"groundtruth://join?token={m.group(1)}&server={quote(PUBLIC['url'] or '', safe='')}"
            self._send(200, PAGE.replace("__APP__", html.escape(app)).encode(), "text/html; charset=utf-8")
        elif m := re.fullmatch(r"/api/session/([A-Za-z0-9-]+)", p):
            self._json(session_status(m.group(1)))
        elif m := re.fullmatch(r"/api/user/([^/]+)", p):
            from urllib.parse import unquote
            self._json({"hikes": user_hikes(unquote(m.group(1)))})
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
                self._json(new_invite(body.get("phone"), body.get("code")))
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
        elif (p.startswith("/api/outbox/") or p in ("/api/trigger", "/api/label", "/api/delete")) and not self._local():
            self._err(404, "not found")
        elif p == "/api/trigger":
            if not body.get("phone"):
                return self._err(400, "phone required")
            try:
                inv = new_invite(body["phone"], body.get("code"))
            except RuntimeError as e:
                return self._err(500, str(e))
            text = body.get("text") or (f"Looks like you're heading out on a trail. Want this one to help train rescue "
                                        f"robots? Tap to start recording: {inv['link']}")
            self._json({"invite": inv, "message": enqueue(body["phone"], text, kind="nudge")})
        elif p == "/api/label":
            sid, text = body.get("session", ""), (body.get("text") or "").strip()
            d = OUT / "sessions" / sid
            if not SID_RE.match(sid) or not d.exists():
                return self._err(404, "no such session")
            if not text:
                return self._err(400, "text required")
            with LOCK:
                labels = json.loads((d / "labels.json").read_text()) if (d / "labels.json").exists() else []
                labels.append({"text": text[:2000], "at": time.time(), "source": body.get("source", "agent")})
                _save("labels.json", labels, d)
            st = session_status(sid)
            if STDB["db"] and st.get("stdb"):
                try:
                    STDB["db"].label(sid, text)
                except Exception as e:  # noqa: BLE001
                    return self._err(502, f"label saved on this server but not in spacetimedb: {e}")
            self._json({"ok": True, "labels": len(labels)})
        elif p == "/api/delete":
            if not body.get("phone"):
                return self._err(400, "phone required")
            try:
                self._json({"ok": True, "deleted": delete_user(body["phone"])})
            except Exception as e:  # noqa: BLE001
                self._err(502, f"nothing deleted, spacetimedb refused: {e}")
        elif m := re.fullmatch(r"/api/outbox/([0-9a-f]+)/(sent|failed)", p):
            with LOCK:
                box = _load("outbox.json", [])
                for msg in box:
                    if msg["id"] == m.group(1):
                        msg["status"] = m.group(2)
                        msg["error"] = body.get("error")
                        msg["at"] = time.time()
                        if msg.get("session") and (OUT / "sessions" / msg["session"] / "status.json").exists():
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
    ap.add_argument("--stdb", default=os.environ.get("STDB_HTTP"),
                    help="SpacetimeDB to write hikes to: http://127.0.0.1:3000 or https://maincloud.spacetimedb.com")
    ap.add_argument("--stdb-db", default=os.environ.get("STDB_DB", "ground-truth"))
    a = ap.parse_args()
    PUBLIC["url"] = a.public.rstrip("/") if a.public else None
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stdb:
        from ground.stdb import Stdb
        STDB["db"] = Stdb(a.stdb, a.stdb_db, OUT / "stdb_identity.json")
        STDB["db"].claim()  # fails loudly if another identity already owns hike writes
        print(f"[ground] spacetimedb {STDB['db']}: hike writer claimed")
    if not PUBLIC["url"]:
        print("[ground] WARNING: no --public url, invites will fail until you restart with one")
    print(f"[ground] http://127.0.0.1:{a.port}  public {PUBLIC['url']}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
