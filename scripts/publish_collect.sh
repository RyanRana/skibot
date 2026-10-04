#!/usr/bin/env bash
# Publishes the capture tables to maincloud and regenerates bindings. Run by whoever owns ground-truth:
#   spacetime login   (once)
#   scripts/publish_collect.sh
# Additive only: existing ski and hike data is kept. Never add --delete-data.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v spacetime >/dev/null || { echo "install the spacetime CLI first: curl -sSf https://install.spacetimedb.com | sh"; exit 1; }
spacetime login show >/dev/null 2>&1 || { echo "run: spacetime login"; exit 1; }
(cd spacetime/spacetimedb && npm ci --silent)
# Explicit confirmations: a change that would break clients still asks first (see DATABASE.md).
spacetime publish ground-truth --no-config -s maincloud -p spacetime/spacetimedb --yes=remote,migrate
spacetime generate --lang typescript --out-dir ground/agent/module_bindings --module-path spacetime/spacetimedb
spacetime generate --lang typescript --out-dir game/src/module_bindings --module-path spacetime/spacetimedb
git add ground/agent/module_bindings game/src/module_bindings
git diff --cached --quiet || git commit -m "Regenerate bindings for the capture tables"
git push origin HEAD
echo "published. start collecting: cd ground/agent && STDB_URI=wss://maincloud.spacetimedb.com npm run collect"
