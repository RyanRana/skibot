# Ground Truth phone recorder

A hiker opts in over iMessage and records a hike with the ground truth iOS app, which keeps recording with the screen locked. At the end the app uploads the hike. The server ties it to the ground under it, and the Photon agent texts back a summary, then asks about the trail and files the answer on that hike.

```
photon agent (ground/agent)  --text with link-->  hiker's iPhone
        ^                                             | tap link -> /g/<token> -> opens the app
        | outbox (summary text)                       | consent, start, lock phone, hike, hold to end
ground/server.py  <--------- background upload ------- ios/GroundTruth
        | when every file is in
ground/register.py: distance, climb (barometer + terrain tiles), OSM trail match, cadence, gaps
```

## What it records

The app keeps itself alive in the background through iOS background location (blue pill in the status bar). It records:

- **Motion at 100 Hz:** user acceleration, gravity, rotation rate, attitude quaternion, magnetic field.
- **GPS at about 1 Hz.**
- **Barometer:** relative and absolute altitude.
- **Steps and floors** at the end.

The record layout is in `ground/schema.py`, and `ios/GroundTruth/Sources/SessionWriter.swift` writes it byte for byte.

Limits:
- A fully powered-off phone records nothing.
- Swiping the app closed stops recording. Relaunching offers "keep recording", and the hole is reported as a gap.

## Run it

One-time setup:

1. `brew install xcodegen cloudflared`
2. In Xcode:
   - Settings → Accounts: add your Apple ID. A free Personal Team is fine. Builds expire after 7 days, so press run again weekly.
   - Settings → Components: download the iOS platform. Without it Xcode can't build to a phone.
3. On the iPhone: Settings → Privacy & Security → Developer Mode on. Plug the phone into the Mac and tap Trust.
4. `cd ios/GroundTruth && ./setup.sh && open GroundTruth.xcodeproj`. Pick your iPhone and press run.
   - On first launch, trust the developer under Settings → General → VPN & Device Management.
   - If Xcode says the bundle id is taken, use `GT_BUNDLE_ID=com.<you>.groundtruth ./setup.sh`.
5. Agent:
   - In `ground/agent`, run `npm install`, then `cp .env.example .env`.
   - Fill in `SPECTRUM_PROJECT_ID` and `SPECTRUM_PROJECT_SECRET` from your project settings at app.photon.codes.
   - Fill in `XAI_API_KEY` for Grok. Without it, replies are scripted, and the agent prints a warning at startup.

Each session:

```bash
ground/run.sh                                  # https tunnel + server, prints the public url; dashboard at http://127.0.0.1:8770
cd ground/agent && npm start                   # the agent on iMessage through Photon Spectrum
npm run chat                                   # same agent in this terminal, no Photon account needed
node agent.mjs nudge +1XXXXXXXXXX              # text someone "heading out? record this one" with their link
npm test                                       # checks the conversation against the running server
```

No Photon yet? Open the app's "testing" field, paste the public url and tap "get a test invite".

The tunnel address changes every time `run.sh` starts. New invites carry the new address. If a hike is still waiting to upload, tap the newest invite link and the app retries with the new address.

## Checks

- **On a real phone:** follow [PHONE_TEST.md](PHONE_TEST.md), eight short recordings. Then `python -m ground.validate ...` checks the sensor numbers against known answers (rest noise and drift, 100 counted steps, 3 turns, a known distance). `python -m ground.validate report` turns the results into one table.

- **Simulator, end to end:** `-GTAutopilot <s> -GTJoin <link>` launch arguments (debug builds only) run consent → record → end → upload with no taps. Feed it a walk with `xcrun simctl location <sim> start ...`. The simulator has no motion sensors or barometer, and the app shows that in red.
- **On the phone:**
  1. Start, lock, pocket, walk 15 min, then unlock.
  2. Motion samples should keep climbing at about 100 per second, and gaps should read "none".
  3. Hold to end. The dashboard should show the session as done, and the summary text should arrive.

## The agent (ground/agent)

It runs on Photon Spectrum (`spectrum-ts`). On the free and Pro plans, people text a number from Photon's shared pool. The Business plan gets one dedicated number. Spectrum allows 50 new conversations per line per day.

- **Who it can text.** We only text people who texted us first, or who were nudged after opting in. Their iMessage handle comes from their message, so the agent has the number before it ever writes to them.
- **What it does.**
  - Explains ground truth.
  - Sends each person their own app link.
  - Answers questions about their hikes, using only numbers from the server.
  - Remembers facts about them across conversations.
  - After every hike, sends the summary and asks how the ground was. The answer is filed as a label on that hike, in `labels.json` in the session folder.
- **Stop and delete are plain code, not the model.**
  - "stop" blocks every outbound text until they text "start".
  - "delete my data" needs an explicit yes, then removes their hikes, invites and messages from the server.
- **Files.**
  - Memory lives in `out/ground/agent_state.json`.
  - `brain.mjs` holds the conversation and the Grok tools.
  - `agent.mjs` is the Spectrum transport and the outbox sender.
  - `server.mjs` calls `ground/server.py`.

## SpacetimeDB (the team database)

Hikes go into the team's SpacetimeDB module (`spacetime/spacetimedb/src/index.ts`), the same database as the ski game.

**What the module stores for hikes.** These tables and reducers are additive only: no ski table changed, so publishing keeps existing data.
- `hike`: one row per hike. It holds the join code, never a phone number.
- `hike_chunk`: per-second features, minus the first and last 200 m.
- `hike_stats`: running totals.
- Reducers: `claim_hike_writer`, `record_hike`, `push_hike_chunk`, `label_hike`, `delete_hikes`. They refuse every identity except the phone server's.

**The app's people.** With `--stdb`, the server also writes every person, invite, message, consent, session and raw phone stream to the private `ground-truth-app` organisation, and the agent writes people and messages. Both need their identity added as that organisation's service once; see [DATABASE.md](../DATABASE.md).

**How hikes get there.**
- `ground/server.py --stdb <url>` (or `STDB_HTTP=<url> ground/run.sh`) writes each registered hike over SpacetimeDB's HTTP API.
- The code is in `ground/stdb.py`. The per-second rows come from `ground/motion.py`.
- The server's identity is saved in `out/ground/stdb_identity.json`.
- The summary text goes into the module's `outbox`, keyed by join code.

**Who texts.** `ground/agent` is the one iMessage agent. It replaced `agent/`, which nothing starts anymore.
- It connects with `STDB_URI` and sends every outbox row: ski joins, results with the finish photo, challenges, hike summaries.
- It answers SKI, TOP, STATS, MAP, CHALLENGE and ME, plus the hike flow.
- Each person gets one join code. It's stored only in `out/ground/agent_state.json`, next to their phone number.

Local:

```
spacetime start                                              # SpacetimeDB on :3000
cd spacetime && spacetime publish ground-truth --server local --yes
STDB_HTTP=http://127.0.0.1:3000 ground/run.sh                # phone server, writes hikes
cd ground/agent && npm start                                 # STDB_URI=ws://127.0.0.1:3000 in .env
```

Demo (hosted). This needs a browser login once:

```
spacetime login
cd spacetime && spacetime publish ground-truth --server maincloud --yes
STDB_HTTP=https://maincloud.spacetimedb.com ground/run.sh
# ground/agent/.env: STDB_URI=wss://maincloud.spacetimedb.com, then restart the agent
```

After changing the module, regenerate the agent's bindings:

```
spacetime generate --lang typescript --out-dir ground/agent/module_bindings --module-path spacetime/spacetimedb
```

The game uses `game/src/module_bindings`; `scripts/dev.sh` regenerates both.
