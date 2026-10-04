"""Admin for the team's SpacetimeDB: the app's organisation, service identities and Model API keys. Runs as the
identity the spacetime CLI is logged in as (the platform admin and owner of hazard-intelligence).

    python tools/stdb_admin.py setup-app                          # the private ground-truth-app organisation
    python tools/stdb_admin.py add-service 0xc200...              # a phone server or agent identity, as its service
    python tools/stdb_admin.py invite --role service              # an invite code instead (GT_SERVICE_INVITE)
    python tools/stdb_admin.py api-identity                       # a new identity for the policy API, printed once
    python tools/stdb_admin.py key create --label "demo laptop"   # a Model API key, printed once
    python tools/stdb_admin.py key list
    python tools/stdb_admin.py key revoke hi_AbCd12

Add --server local to work against `spacetime start` instead of maincloud.
"""
from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.stdb_load import SERVERS, Db, cli_token  # noqa: E402

APP = "ground-truth-app"
HI = "hazard-intelligence"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="maincloud")
    ap.add_argument("--db", default="ground-truth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup-app")
    s = sub.add_parser("add-service")
    s.add_argument("identity")
    s.add_argument("--tenant", default=APP)
    s.add_argument("--role", default="service")
    s = sub.add_parser("invite")
    s.add_argument("--tenant", default=APP)
    s.add_argument("--role", default="service")
    s.add_argument("--uses", type=int, default=1)
    sub.add_parser("api-identity")
    k = sub.add_parser("key")
    k.add_argument("action", choices=["create", "list", "revoke"])
    k.add_argument("which", nargs="?", help="revoke: the key, its prefix or its hash")
    k.add_argument("--tenant", default=HI)
    k.add_argument("--label", default="")
    a = ap.parse_args()
    base = SERVERS.get(a.server, a.server)
    db = Db(base, a.db, cli_token())

    if a.cmd == "setup-app":
        if db.sql(f"SELECT id FROM my_tenant WHERE id = '{APP}'"):
            print(f"{APP} already exists")
        else:
            db.call("create_tenant", {"id": APP, "name": "Ground Truth app", "kind": "team", "visibility": "private"})
            print(f"created {APP} (private); you are its owner")
    elif a.cmd == "add-service":
        ident = a.identity if a.identity.startswith("0x") else "0x" + a.identity
        db.call("add_member_identity", {"tenantId": a.tenant, "identity": {"__identity__": ident}, "role": a.role})
        print(f"{ident} is now {a.role} of {a.tenant}")
    elif a.cmd == "invite":
        before = {r[0] for r in db.sql(f"SELECT code FROM my_invite WHERE tenant_id = '{a.tenant}'")}
        db.call("create_invite", {"tenantId": a.tenant, "role": a.role, "maxUses": a.uses})
        code = [r[0] for r in db.sql(f"SELECT code FROM my_invite WHERE tenant_id = '{a.tenant}'") if r[0] not in before]
        print(code[0] if code else "created, but could not read the code back")
    elif a.cmd == "api-identity":
        import requests
        r = requests.post(f"{base}/v1/identity", timeout=20).json()
        ident = "0x" + r["identity"]
        db.call("add_platform_service", {"identity": {"__identity__": ident}, "name": "policy API (Modal)", "kind": "api"})
        print(f"identity {ident} is a platform service. Its token (store it as a secret, it is shown once):")
        print(r["token"])
    elif a.cmd == "key":
        if a.action == "create":
            key = "hi_" + secrets.token_urlsafe(30)
            db.call("add_api_key", {"tenantId": a.tenant, "hash": hashlib.sha256(key.encode()).hexdigest(),
                                    "prefix": key[:10], "label": a.label})
            print(f"Model API key for {a.tenant} (shown once, store it now):\n{key}")
        elif a.action == "list":
            for h, tenant, prefix, label, calls, revoked in db.sql(
                    f"SELECT hash, tenant, prefix, label, calls, revoked FROM my_api_key WHERE tenant = '{a.tenant}'"):
                print(f"{prefix}...  {label or '-':20}  {calls:>8} calls  {'REVOKED' if revoked else 'active'}  {h[:12]}")
        else:
            w = a.which or sys.exit("which key? the key, its prefix or its hash")
            rows = db.sql(f"SELECT hash, prefix FROM my_api_key WHERE tenant = '{a.tenant}'")
            h = hashlib.sha256(w.encode()).hexdigest() if w.startswith("hi_") and len(w) > 20 else None
            hit = [r[0] for r in rows if r[0] == h or r[0] == w or r[1] == w[:10]]
            if len(hit) != 1:
                sys.exit(f"{len(hit)} keys match {w!r}")
            db.call("revoke_api_key", {"hash": hit[0]})
            print(f"revoked {hit[0][:12]}")


if __name__ == "__main__":
    main()
