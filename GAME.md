# Ground Truth: Ski

Stand in front of a laptop and ski the Streif with your body. A Unitree G1 copies you down the real Hahnenkamm, everyone who opens the page is on the same mountain at the same time, and every run grows a dataset of human motion registered to the ground it happened on. Built at MHacks 2026 for the theme "build something that grows", on top of skibot.

## What it looks like

**The mountain.** The real Streif at Kitzbühel, twice over. The piste is a 2 m heightfield from OpenStreetMap piste lines and AWS elevation tiles, the same grid the robot trains on in MuJoCo; around it the real Alps out to 10 km from a coarser DEM, coloured by altitude and slope, so the ridgeline on the horizon is the actual Kitzbühel skyline. A dawn sky with a low warm sun and a lens flare, thin high cloud, valley haze that thickens below the course, and a soft bloom on the sun and the sparkle. The snow shades blue in the shade and warm toward the sun, carries corduroy down the groomed corridor and wind texture off it, and glints where it faces the eye. Dark three-tier conifers with snow caps thicken away from the piste. Orange safety nets line the outside of every turn, orange markers run down both edges, race banners (Ground Truth, MHacks, SpacetimeDB, Photon, Kitzbühel) stand along the right, and a chairlift with moving chairs climbs the slope beside the course. The race starts 90 m down the line where the slope tips over, under a START banner beside a timing hut, and by default runs 650 m through 33 gates so a walk-up run takes about a minute (`?length=full` races the whole 1.2 km). The finish is a red arch with a crowd of spectators along the last 70 m who bounce when the room cheers.

**The skier.** The actual Unitree G1, decimated from the repo's STL meshes and rebuilt in the browser from its MJCF kinematic tree, on red skis with poles and a race bib carrying your name and colour. It is posed by you: knee bend, hip angulation, chest lean, arm position and elbow bend are retargeted from MediaPipe's 3D landmarks with the same recipe skibot uses to turn race video into robot joint targets. Lean and the G1 rolls into the turn, crouch and it tucks, hop and it pulls its knees up in the air. Two ribbons of track stay in the snow behind the skis; an edged ski at speed throws a plume of spray to the outside of the turn.

**The gates.** Red and blue giant slalom gates with fluttering panels, placed by the course builder along the real line. The next gate glows, passed gates fade, missed gates fade further and cost 1.5 s. Making a gate gives a 40 ms hit-stop, a small camera kick, a burst of sparks in the gate's colour and a flash; the crowd rings cowbells when it is excited.

**The camera and the feel.** A chase camera over the right shoulder that drops lower, pulls back and widens its field of view from 60° to 85° as you get faster, rolls with your edge, and kicks on landings and wipeouts. At speed the edges of the frame pull into a radial blur with speed lines, a little chromatic aberration and a vignette. While you calibrate it faces the robot so you can watch it copy you. Press C for front and side views. Joining plays a six-second flyover of the course with a title card.

**The race on TV.** At every gate a split flashes in the centre against the fastest run on the course, green when you are up and red when you are down, and slides into a stack on the left. The leader skis beside you as a translucent gold ghost replayed from their recorded trace, with "▲ 1.3 m ahead" under the clock. Six clean gates in a row spell STREIF under the clock and earn a push. The finish is a 150 ms freeze, 1.2 s of slow motion with the camera orbiting the skier, confetti from the arch, a fanfare and the crowd, then a result card: time, rank, delta to the leader, gates, top speed, samples added to Ground Truth and cheers received. A speech announcer calls the start, big air, splits every six gates and the result. A finish photo is captured and, if you joined by text, arrives on your phone.

**The HUD.** Dark glass panels. Top left: course and the split stack. Top centre: clock, gap to the leader, speed, the STREIF streak and a row of gate tiles that fill red and blue. Top right: the Ground Truth counter ticking up live from SpacetimeDB, and cheers as they arrive with a crowd meter. Right: today's fastest. Bottom left: your webcam, mirrored, skeleton in amber while calibrating and mint once tracking. Bottom centre: lean and tuck meters. Bottom right: the shared feed. Before you join, the start screen is a mirror: your silhouette with the skeleton over it, and raising both hands for two seconds puts you on the snow without touching the keyboard.

**The board** (`/board.html`, for a second screen or a phone). A hillshaded top-down map of the real terrain with contour lines, the piste corridor, gates and finish, a live dot and trail for every skier with their speed, the dataset counter in large type, who is on the mountain, the leaderboard, the feed, a QR code and the iMessage number to join, and three big buttons: cowbell, clap, fire. A tap on a phone rings in the game within a frame, names the sender on screen, bounces the crowd and feeds the crowd meter.

## How it plays

1. Text **SKI** to the iMessage agent (optional). It replies with a four-letter code and a link.
2. On the laptop, type your name and the code, and step onto the snow. Stand about two metres from the camera with your whole body in frame; hold still for a second and the skeleton turns mint.
3. Raise both hands. Three, two, one, go.
4. **Lean** left or right to carve (shoulders over hips, hips over feet). **Crouch** to tuck for speed, stand tall to slow. **Hop** to jump. Make the gates.
5. At the finish you see your time, rank, gates, top speed, and how many motion samples you just added to Ground Truth. If you texted SKI, the same arrives on your phone with a link to the board. Reply **CHALLENGE <name>** and the agent carries your time to another skier; the game shows them the challenge on the start line and the agent texts you both the result.

Keyboard fallback: arrows lean, down crouches, space jumps, Enter starts or skips the intro, R restarts, K recalibrates, C changes camera, V toggles the announcer, B rings a test cowbell. URL flags: `?autopilot=1` makes the G1 ski the course by itself (attract mode for the table; it makes every gate at about 45 km/h), `?length=full` or `?length=400` sets the race length in metres, `?lowfx=1` turns off shadows for a weak laptop, `?nocam=1` skips the camera, `?name=…&code=…&auto=1` prefills and submits the start form, `?cam=front` starts on the front camera.

## How it works

```
game/            Vite + TypeScript + Three.js + MediaPipe Tasks Vision
  src/course.ts  heightfield sampling, centerline progress, gate geometry (pure, tested in Node)
  src/physics.ts carving model: edge angle from lean, carve radius sidecut x cos(edge), grip across the
                 skis, glide friction, tuck-dependent drag, hops and natural air off rollers (pure, tested)
  src/pose.ts    MediaPipe Pose Landmarker -> lean, crouch, hop, hands up, knee angles; auto-calibration
  src/g1.ts      G1 kinematic tree from the MJCF + decimated GLB; MediaPipe -> G1 joint retarget
  src/world.ts   near and far terrain, snow shader, sky and flare, trees, fences, banners, chairlift, gates, start, finish, crowd, trails, spray, confetti, bursts
  src/post.ts    post-processing: bloom, speed blur, chromatic aberration, speed lines, vignette, flashes
  src/net.ts     SpacetimeDB client: identity in localStorage, 8 Hz state stream, queued calls, cache reads
  src/main.ts    game loop, phases, camera, ghosts of other skiers, dataset upload
  src/board.ts   the live map
spacetime/       SpacetimeDB module (TypeScript, runs inside the database)
  player, skier (live state), run (with per-gate splits), trace_chunk (the dataset), dataset_stats, feed, challenge,
  outbox, cheer (event table), run_photo, tick_timer (drops stale skiers every 2 s); reducers join, set_state, start_run,
  push_trace, finish_run, create_challenge, link_code, mark_sent, send_cheer, push_photo
agent/           Photon Spectrum iMessage agent; brain.ts is the logic, agent.ts the wiring, test.ts a scripted conversation
tools/           export_course.py (built course -> browser heightfield), export_g1.py (STL -> 1 MB GLB + kinematic JSON)
```

**Why SpacetimeDB.** The mountain is the database. Every browser streams its skier row ~8 times a second through a reducer and subscribes to everyone else's; ghosts, the leaderboard, the feed and the dataset counter are all just table subscriptions, with no game server in between. A scheduled reducer sweeps skiers whose tab died. The dataset itself is rows: each `trace_chunk` is 50 samples of body state (lean, crouch, knee angles) with the position, speed and slope of the real ground under it, which is exactly the record Ground Truth exists to collect.

**Why the iMessage agent.** It is how a stranger at a table joins and leaves with something: a code from one text, a finish photo and result on their phone before they have walked away, and a way to pull a friend in. Phone numbers never reach the database; the agent holds the code-to-conversation map and the module only writes to an outbox keyed by code.

## Run it

```bash
curl -sSf https://install.spacetimedb.com | sh        # spacetime CLI (once)
scripts/dev.sh                                        # local SpacetimeDB, publish, bindings, game on :5173, agent in terminal mode
```

Then open `http://<laptop LAN IP>:5173/` on the laptop (the webcam needs localhost or HTTPS, so play on the laptop itself) and `/board.html` on a second screen or a phone.

For the demo on the hosted database: `spacetime login`, then `STDB=maincloud scripts/dev.sh`. For real iMessage: create a project at app.photon.codes, add the teammates' and judges' numbers under Users (free tier: 10), put `PHOTON_PROJECT_ID` and `PHOTON_PROJECT_SECRET` in `agent/.env`, and put the assigned number in `VITE_AGENT_NUMBER` in `game/.env`.

Rebuilding the assets: `.venv/bin/python resort.py "Kitzbühel" --run streif` then `.venv/bin/python tools/export_course.py kitzbuhel-streif`; `.venv/bin/python tools/export_g1.py`.

Tests: `cd game && node --experimental-strip-types test/physics.test.ts` drives an autopilot down the course without a browser and `test/retarget.test.ts` checks the body-to-G1 mapping; `cd agent && npx tsx src/test.ts` runs a scripted iMessage conversation against the live database, including a finished run and a challenge.

## Data and attribution

Piste lines © OpenStreetMap contributors (ODbL). Elevation: AWS Terrain Tiles (Mapzen terrarium). Unitree G1 model: unitree_rl_mjlab (Apache 2.0). Pose: MediaPipe Pose Landmarker (lite). Contributors can be removed from the dataset on request; traces carry no phone numbers or precise home locations, only positions on the piste.
