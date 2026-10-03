# Ground Truth phone recorder

A hiker opts in over iMessage and records a hike with the ground truth iOS app, which keeps recording with the screen locked. At the end the app uploads the hike. The server ties it to the ground under it, and the Photon agent texts back a summary.

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
5. In `ground/agent`, run `npm install`. Then give your terminal app Full Disk Access (System Settings → Privacy & Security), because Photon's kit reads the Messages database.

Each session:

```bash
ground/run.sh                                  # https tunnel + server, prints the public url; dashboard at http://127.0.0.1:8770
cd ground/agent && node agent.mjs              # Photon agent: replies to "hike" texts with the link, sends summaries
node agent.mjs invite +1XXXXXXXXXX             # or text one person the link right now
```

No Photon yet? Open the app's "testing" field, paste the public url and tap "get a test invite".

The tunnel address changes every time `run.sh` starts. New invites carry the new address. If a hike is still waiting to upload, tap the newest invite link and the app retries with the new address.

## Checks

- **Simulator, end to end:** `-GTAutopilot <s> -GTJoin <link>` launch arguments (debug builds only) run consent → record → end → upload with no taps. Feed it a walk with `xcrun simctl location <sim> start ...`. The simulator has no motion sensors or barometer, and the app shows that in red.
- **On the phone:**
  1. Start, lock, pocket, walk 15 min, then unlock.
  2. Motion samples should keep climbing at about 100 per second, and gaps should read "none".
  3. Hold to end. The dashboard should show the session as done, and the summary text should arrive.
