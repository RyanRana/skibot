#!/bin/zsh
# Starts a public https tunnel (cloudflared) and the Ground Truth server behind it.
# Ctrl-C stops both. Every run gets a new trycloudflare.com address; new invites carry it automatically.
set -e
cd "$(dirname "$0")/.."
LOG=$(mktemp -t gt-tunnel)
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8770 > "$LOG" 2>&1 &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null' EXIT INT TERM
for i in {1..60}; do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1)
  [[ -n "$URL" ]] && break
  sleep 0.5
done
if [[ -z "$URL" ]]; then echo "TUNNEL FAILED, cloudflared said:"; cat "$LOG"; exit 1; fi
echo "public url: $URL   dashboard: http://127.0.0.1:8770"
.venv/bin/python -u -m ground.server --public "$URL"
