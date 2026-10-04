#!/usr/bin/env bash
# One command for the demo table: local SpacetimeDB, the module, the game server and the iMessage agent.
#
#   scripts/dev.sh            local SpacetimeDB on :3000, game on http://<LAN IP>:5173, board at /board.html
#   STDB=maincloud scripts/dev.sh   same, but the game and agent talk to the maincloud database (run `spacetime login` once)
#
# The iMessage agent is ground/agent (it replaced agent/, which is no longer started): it needs SPECTRUM_PROJECT_ID and
# SPECTRUM_PROJECT_SECRET in ground/agent/.env to text real phones; without them it runs in terminal mode.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
STDB="${STDB:-local}"
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)

if [ "$STDB" = "local" ]; then
  if ! curl -s -o /dev/null http://127.0.0.1:3000/v1/ping; then
    echo "[dev] starting local SpacetimeDB"; (spacetime start > /tmp/spacetime-local.log 2>&1 &); sleep 3
  fi
  (cd spacetime && spacetime publish ground-truth --server local --yes)
  URI="ws://127.0.0.1:3000"
else
  (cd spacetime && spacetime publish ground-truth --server maincloud --yes)
  URI="wss://maincloud.spacetimedb.com"
fi
spacetime generate --lang typescript --out-dir game/src/module_bindings --module-path spacetime/spacetimedb >/dev/null
spacetime generate --lang typescript --out-dir ground/agent/module_bindings --module-path spacetime/spacetimedb >/dev/null

[ -d game/node_modules ] || (cd game && npm install)
[ -d ground/agent/node_modules ] || (cd ground/agent && npm install)
[ -d spacetime/spacetimedb/node_modules ] || (cd spacetime/spacetimedb && npm install)
[ -f game/.env ] || cp game/.env.example game/.env
[ -f ground/agent/.env ] || cp ground/agent/.env.example ground/agent/.env

echo "[dev] game:  http://${LAN_IP}:5173/   board: http://${LAN_IP}:5173/board.html   spacetimedb: ${URI}"
(cd game && VITE_STDB_URI="$URI" npx vite --host --port 5173 &)
sleep 2
(cd ground/agent && STDB_URI="$URI" GAME_URL="http://${LAN_IP}:5173/" npx tsx agent.mjs)
