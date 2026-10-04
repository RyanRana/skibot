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
- POST /api/signup {phone, consent}     public: the website's phone box. queues a welcome text with the app link;
                                        rate limited per IP and per number, US numbers only
- POST /api/trigger {phone}             new invite + a "want to record this one?" text (later: the app's geofence)
- POST /api/label {session, text}       something the hiker told the agent about a hike ("slipped near the creek")
- POST /api/delete {phone}              removes that person's hikes, invites and messages
- GET  /                                dashboard
Everything that touches phone numbers (dashboard, outbox, user, trigger, label, delete) answers only on this Mac,
never through the tunnel.

With --stdb, the files here stay this server's working copy and SpacetimeDB gets every change too: people, invites,
messages, sessions and each session's raw IMU, GPS and barometer streams, under the private app organisation
(GT_TENANT, default ground-truth-app). At startup it copies over whatever it already has. Its identity must be a
service of that organisation: set GT_SERVICE_INVITE to an invite code once, or have an owner add it (DATABASE.md).
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
SIGNUP_ORIGINS = {o.strip() for o in os.environ.get("GT_SIGNUP_ORIGINS", "*").split(",") if o.strip()}
SIGNUP_IP_MAX, SIGNUP_IP_WINDOW = 5, 3600  # sign-ups per IP per hour
SIGNUP_PHONE_WINDOW = 86400  # one welcome text per number per day
SIGNUPS: dict[str, list[float]] = {}  # ip -> recent sign-up times (memory only)

LOCK = threading.RLock()
PUBLIC = {"url": None}
STDB = {"db": None, "app": False}  # ground.stdb.Stdb when --stdb is given: hikes go to the team's SpacetimeDB
TENANT = os.environ.get("GT_TENANT", "ground-truth-app")  # the organisation that owns the app's people and sessions


def handle_key(h: str | None) -> str:
    """The agent's key for a person (ground/agent/state.mjs): +1 and ten digits, + and digits, or a lowercase email."""
    if not h:
        return ""
    if "@" in h:
        return h.strip().lower()
    d = re.sub(r"\D", "", h)
    return "+1" + d if len(d) == 10 else "+" + d


def mirror(what: str, fn):
    """Copies one change into SpacetimeDB. Best effort: the files here are already saved, so a failure is logged and
    the next startup sync fills the gap."""
    if not (STDB["db"] and STDB["app"]):
        return
    try:
        fn(STDB["db"])
    except Exception as e:  # noqa: BLE001
        print(f"[ground] SPACETIMEDB {what} FAILED: {e}")


def mirror_session(sid: str, status: str | None = None, **extra):
    """The session's row: who (from its invite), consent, the phone's meta.json, the summary and labels."""
    d = OUT / "sessions" / sid
    if not (STDB["db"] and STDB["app"]) or not (d / "consent.json").exists():
        return
    consent = json.loads((d / "consent.json").read_text())
    inv = _load("invites.json", {}).get(consent.get("token"), {})
    st = session_status(sid)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    labels = [x["text"] for x in json.loads((d / "labels.json").read_text())] if (d / "labels.json").exists() else []
    mirror(f"session {sid}", lambda db: db.app_session(
        TENANT, sid, handle_key(inv.get("phone")), inv.get("code") or "", consent.get("token", ""), status or st.get("status", "consented"),
        consent=consent, meta=meta, summary=extra.get("summary", st.get("summary")), labels=labels,
        error=extra.get("error") or st.get("error") or st.get("stdb_error") or "", created=consent.get("received", 0)))


def sync_to_stdb():
    """Everything already on this Mac, copied into SpacetimeDB once at startup. Re-sending a row replaces it."""
    if not (STDB["db"] and STDB["app"]):
        return
    db = STDB["db"]
    invites, box = _load("invites.json", {}), _load("outbox.json", [])
    for tok, v in invites.items():
        mirror(f"invite {tok}", lambda db, tok=tok, v=v: db.app_invite(TENANT, tok, handle_key(v.get("phone")), v.get("code") or "",
                                                                       v.get("link", ""), v.get("created", 0)))
    for m in box:
        mirror(f"message {m['id']}", lambda db, m=m: db.app_message(TENANT, m["id"], handle_key(m.get("to")), "out", m.get("kind", ""),
                                                                    m.get("text", ""), m.get("session") or "", m.get("status", ""),
                                                                    m.get("error") or "", m.get("created", 0)))
    sdir = OUT / "sessions"
    done = 0
    for d in sorted(sdir.iterdir()) if sdir.exists() else []:
        mirror_session(d.name)
        if session_status(d.name).get("status") == "done":
            try:
                have = db.sql(f"SELECT key FROM my_recording WHERE ref = '{d.name}'")
                if not have:
                    from ground.schema import load_session
                    meta, data, _ = load_session(d)
                    db.phone_recordings(TENANT, d.name, data, meta)
                    done += 1
            except Exception as e:  # noqa: BLE001
                print(f"[ground] SPACETIMEDB raw streams for {d.name} FAILED: {e}")
    print(f"[ground] spacetimedb sync: {len(invites)} invites, {len(box)} messages, "
          f"{len(list(sdir.iterdir())) if sdir.exists() else 0} sessions, {done} raw uploads")


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
        out = {"token": token, **inv[token]}
    mirror("invite", lambda db: db.app_invite(TENANT, token, handle_key(phone), out["code"] or "", out["link"], out["created"]))
    return out


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
    mirror("message", lambda db: db.app_message(TENANT, msg["id"], handle_key(to), "out", kind, text, session or "", "pending",
                                                at=msg["created"]))
    return msg


def us_phone(raw) -> str | None:
    """(919) 555-0123, 919-555-0123, +1 919 555 0123 -> +19195550123. Anything else is refused."""
    d = re.sub(r"\D", "", str(raw or ""))
    if len(d) == 11 and d[0] == "1":
        d = d[1:]
    return f"+1{d}" if len(d) == 10 and d[0] in "23456789" else None


def signup(phone: str, ip: str) -> dict:
    """The website's phone box. Same welcome as texting the agent: a link that opens the app, then it records ambiently."""
    now = time.time()
    with LOCK:
        recent = [t for t in SIGNUPS.get(ip, []) if now - t < SIGNUP_IP_WINDOW]
        if len(recent) >= SIGNUP_IP_MAX:
            raise PermissionError("too many sign-ups from this network, try again in an hour")
        for m in _load("outbox.json", []):
            if m.get("kind") == "welcome" and same_phone(m["to"], phone) and now - m["created"] < SIGNUP_PHONE_WINDOW:
                return {"ok": True, "already": True}
        SIGNUPS[ip] = recent + [now]
        inv = new_invite(phone)
        enqueue(phone, "hi, it's ground truth. you signed up on the site to help teach rescue robots how people move on "
                       f"real terrain. tap once to set up the app, then it records your hikes and runs on its own: {inv['link']}  "
                       "text me anything about it, STOP to pause, or \"delete my data\" anytime.", kind="welcome")
        return {"ok": True}


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
        if STDB["db"] and STDB["app"]:  # the person, their messages, sessions, raw streams and hikes, in one transaction
            STDB["db"].delete_app_user(TENANT, handle_key(phone))
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
            new = "uploading"
        else:
            _save("status.json", {"status": "registering"}, d)
            new = "registering"
    if new != st.get("status"):
        mirror_session(sid, new)
    if new == "registering":
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
        if STDB["db"] and STDB["app"]:  # the raw streams, then the session row that points at them
            def raw(db):
                from ground.schema import load_session
                meta, data, _ = load_session(d)
                db.phone_recordings(TENANT, sid, data, meta)
            mirror(f"raw streams {sid}", raw)
        mirror_session(sid, "done", summary=s)
        print(f"[ground] {sid} done\n{text}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        _save("status.json", {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": tb}, d)
        mirror_session(sid, "failed", error=f"{type(e).__name__}: {e}")
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

    def _cors(self):
        origin = self.headers.get("Origin", "")
        if self.path.split("?")[0] == "/api/signup" and ("*" in SIGNUP_ORIGINS or origin in SIGNUP_ORIGINS):
            self.send_header("Access-Control-Allow-Origin", origin or "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self._cors()
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
                return self._send(404, b"<h1>this link has expired or is wrong</h1>", "text/html; charset=utf-8")
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
        elif p == "/api/signup":
            phone = us_phone(body.get("phone"))
            if not phone:
                return self._err(400, "enter a US mobile number")
            if body.get("consent") is not True:
                return self._err(400, "consent required")
            try:
                self._json(signup(phone, self.headers.get("CF-Connecting-IP") or self.client_address[0]))
            except PermissionError as e:
                self._err(429, str(e))
            except RuntimeError as e:
                self._err(503, str(e))
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
            mirror_session(sid, "consented")
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
            text = body.get("text") or (f"looks like you're heading out on a trail. want this one to help train rescue "
                                        f"robots? tap to start recording: {inv['link']}")
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
            mirror_session(sid)
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
            mirror("message status", lambda db: db.app_message(TENANT, msg["id"], handle_key(msg.get("to")), "out", msg.get("kind", ""),
                                                               msg.get("text", ""), msg.get("session") or "", msg["status"],
                                                               msg.get("error") or "", msg.get("created", 0)))
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
        db = STDB["db"]
        try:
            if not db.member_of(TENANT) and os.environ.get("GT_SERVICE_INVITE"):
                db.join(os.environ["GT_SERVICE_INVITE"])
            role = db.member_of(TENANT)
        except Exception as e:  # noqa: BLE001
            role = None
            print(f"[ground] spacetimedb membership check failed: {e}")
        STDB["app"] = role in ("owner", "admin", "service")
        if STDB["app"]:
            print(f"[ground] spacetimedb: writing people, messages and sessions as {role} of {TENANT}")
            threading.Thread(target=sync_to_stdb, daemon=True).start()
        else:
            print(f"[ground] WARNING: {db.identity()} is not a service of {TENANT} in spacetimedb, so people, messages and "
                  f"sessions stay on this Mac. Fix: GT_SERVICE_INVITE=<code> or python tools/stdb_admin.py add-service "
                  f"{db.identity()}")
    if not PUBLIC["url"]:
        print("[ground] WARNING: no --public url, invites will fail until you restart with one")
    print(f"[ground] http://127.0.0.1:{a.port}  public {PUBLIC['url']}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
