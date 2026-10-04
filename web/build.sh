#!/usr/bin/env bash
# One deployable site: the landing pages at /, the body-controlled ski game at /game (when game/ is present).
#
#   bash web/build.sh        ->  web/dist/
#
# The game fetches /courses/... and /g1/... from the site root; vercel.json rewrites those into /game.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=web/dist
rm -rf "$OUT" && mkdir -p "$OUT"
cp web/*.html web/site.css "$OUT"/
cp -R web/media web/vendor "$OUT"/
rm -f "$OUT"/media/.gitignore
cp policy_api/hi_client.py "$OUT"/hi_client.py
if [ -f game/package.json ]; then
  # the deployed game talks to the team database on maincloud, not a local spacetime
  export VITE_STDB_URI="${VITE_STDB_URI:-wss://maincloud.spacetimedb.com}" VITE_STDB_DB="${VITE_STDB_DB:-ground-truth}"
  (cd game && npm ci --no-audit --no-fund && npx vite build --base=/game/ --outDir "../$OUT/game" --emptyOutDir)
fi
echo "built $OUT"
