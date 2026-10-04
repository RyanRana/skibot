#!/usr/bin/env bash
# Run the Presage relay on this laptop and give it a public address, so the live website can read pulses.
#
#   scripts/presage-relay.sh https://your-site.vercel.app
#
# Uses PRESAGE_API_KEY from game/.env.local (never printed, never sent to the site). Runs the relay in Docker
# when Docker is up, otherwise with Node, then opens a free Cloudflare tunnel and prints the wss:// address
# to put in the Vercel project as VITE_PRESAGE_URL (then redeploy). Keep this window open while people play.
set -euo pipefail
cd "$(dirname "$0")/.."
SITE="${1:?usage: scripts/presage-relay.sh https://your-site.example (the address of the live site)}"
KEY=$(sed -n 's/^PRESAGE_API_KEY=//p' game/.env.local 2>/dev/null | tr -d '\r')
[ -n "$KEY" ] || { echo "PRESAGE_API_KEY missing in game/.env.local"; exit 1; }
ORIGINS="${SITE%/},http://localhost:5173"
if docker info >/dev/null 2>&1; then
  docker build -q -t presage-relay presage-relay >/dev/null
  docker rm -f presage-relay >/dev/null 2>&1 || true
  docker run -d --name presage-relay -p 8787:8787 -e PRESAGE_API_KEY="$KEY" -e ALLOWED_ORIGINS="$ORIGINS" presage-relay >/dev/null
  echo "[relay] running in Docker (docker logs -f presage-relay)"
else
  (cd presage-relay && [ -d node_modules ] || npm install --no-audit --no-fund)
  (cd presage-relay && PRESAGE_API_KEY="$KEY" ALLOWED_ORIGINS="$ORIGINS" nohup node server.mjs > /tmp/presage-relay.log 2>&1 &)
  echo "[relay] running with Node (tail -f /tmp/presage-relay.log)"
fi
sleep 2
curl -sf http://localhost:8787/health >/dev/null && echo "[relay] healthy on :8787, allowing $ORIGINS"
echo "[tunnel] starting; copy the https://...trycloudflare.com address it prints and set in Vercel:"
echo "         VITE_PRESAGE_URL = wss://<that-host>/presage     (then redeploy)"
exec cloudflared tunnel --url http://localhost:8787
