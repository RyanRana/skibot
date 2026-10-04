# Hazard Intelligence

The website is **Hazard Intelligence** (the code and the database still use the project's working name, Ground Truth).

Stand in front of a laptop and ski the Streif with your body. A Unitree G1 copies you down the real Hahnenkamm, everyone who opens the page is on the same mountain at the same time, and every run grows a dataset of human motion registered to the ground it happened on. Built at MHacks 2026 for the theme "build something that grows", on top of skibot. (The iMessage side of the project lives on the `photon` branch.)

## What it looks like

**The mountain.** The real Streif at Kitzbühel, twice over. The piste is a 2 m heightfield from OpenStreetMap piste lines and AWS elevation tiles, the same grid the robot trains on in MuJoCo; around it the real Alps out to 10 km from a 30 m DEM (zoom-12 terrain tiles), so the ridgeline on the horizon is the actual Kitzbühel skyline. The far mountains are shaded smooth and surfaced in the shader: snow where the slope lets it hold, banded rock on the faces, patchy spruce forest with white clearings up to a ragged treeline at about 1800 m, fields on the valley floor, valleys sitting in their own shade, cloud shadows drifting across, and aerial perspective that warms toward the sun and cools away from it. A dawn sky with a low sun raking in from behind the skier's right shoulder, streaks of cirrus and a broken layer of altocumulus lit on the sun side, valley haze that thickens below the course, and a soft bloom on the sun and the sparkle. The snow shades blue in the shade and warm toward the sun, carries corduroy down the groomed corridor and wind-packed sastrugi off it, shows rock where the ground gets steep, and glints where it faces the eye. A spruce forest thickens away from the piste in stands with glades between them: three procedural tree shapes (slender, full, young and heavily loaded) built from drooping, ragged branch tiers with clumpy snow on every tier and dark needles underneath, swaying a little in the wind. Snow-capped boulders break out of the slope, more of them where it is steep, and snow lances lean over both edges of the course. The low sun's shadows from the real ridgelines are baked into both terrains at load by marching toward the sun over the 30 m DEM, so the far Alps stand in layers and the glitter on the snow only flashes where the sun actually reaches. Orange diamond-mesh B-nets on posts with yellow pads line the outside of every turn, orange markers run down both edges, fluttering race banners (Ground Truth, MHacks, SpacetimeDB, Photon, Kitzbühel) stand along the right, and a chairlift with moving chairs climbs lattice-steel towers beside the course. The race starts 90 m down the line where the slope tips over, under a START banner readable from both sides, beside a plank-and-shingle timing hut, and by default runs 650 m through 33 gates so a walk-up run takes about a minute (`?length=full` races the whole 1.2 km). The finish is a red inflatable arch with FINISH and GROUND TRUTH banners, national flags waving behind a 3D crowd along the last 70 m: spectators in ski jackets and beanies who face the piste, bounce on their toes and throw their arms up when the room cheers.

**The skier.** The actual Unitree G1, decimated from the repo's STL meshes and rebuilt in the browser from its MJCF kinematic tree, in a clear-coated white shell with dark anodised joints that reflect the sky (a prefiltered environment map baked from the sky shader) and a cool rim light that separates it from the snow, wearing a race helmet with mirrored iridescent goggles, on shaped race skis (sidecut, a rising shovel, steel edges, a binding plate, a printed top sheet) with poles and a race bib carrying your name and colour. It is posed by you: knee bend, hip angulation, chest lean, arm position and elbow bend are retargeted from MediaPipe's 3D landmarks with the same recipe skibot uses to turn race video into robot joint targets. Lean and the G1 rolls into the turn, crouch and it tucks, hop and it pulls its knees up in the air. Two carved grooves stay in the snow behind the skis (a bright lip of thrown-up snow and a shaded channel); an edged ski at speed throws soft powder that billows, grows and fades to the outside of the turn.

**The way down.** Two blue ribbons mark the edges of the piste from the start to the finish, light-blue chevrons flow down the middle of the line, and the next gate carries a soft light column, a bobbing arrow and a translucent doorway between its panels. When the next gate is more than about 25° off your heading a big arrow appears in the middle of the screen with the distance; it never points behind you, because a gate you have skied past is counted as missed at once.

**The gates.** Red and blue giant slalom gates with fluttering panels, placed by the course builder along the real line. The next gate glows, passed gates fade, missed gates fade further and cost 1.5 s. Making a gate gives a 40 ms hit-stop, a small camera kick, a burst of sparks in the gate's colour and a flash; the crowd rings cowbells when it is excited.

**Obstacles.** Trees, the timing hut, lift towers and the finish pillars are solid. Hit one and it is a wipeout: a burst of snow and needles, the tree shakes and sheds its snow, the screen kicks and flashes, the announcer winces, and you are back on your skis a second later facing down the course. Ski into a safety net and it catches you: a soft bounce back onto the piste, orange sparks, and a toast.

**The camera and the feel.** A chase camera over the right shoulder that drops lower, pulls back and widens its field of view from 60° to 85° as you get faster, rolls with your edge, and kicks on landings and wipeouts. At speed the edges of the frame pull into a radial blur with speed lines, a little chromatic aberration and a vignette. While you calibrate it faces the robot so you can watch it copy you. Press C for front and side views. Joining plays a six-second flyover of the course with the HUD out of the way; the flyover is the title card. Behind the start screen the camera hangs over the robot in the start gate looking down the course, and the robot already copies whoever stands in front of the camera.

**The race on TV.** At every gate a split against the fastest run on the course lands on top of a stack under the course tag, green when you are up and red when you are down. The leader skis beside you as a translucent gold ghost replayed from their recorded trace, with "▲ 1.3 m ahead" under the clock. Six clean gates in a row spell STREIF under the clock and earn a push. Things that happen mid-run (air, misses, wipeouts, nets, boosts, powerups) appear as small coloured callouts under the clock, never in the middle of the frame. The finish is a 150 ms freeze, 1.2 s of slow motion with the camera orbiting the skier, confetti from the arch, a fanfare and the crowd, then a result card: time, rank, delta to the leader, gates, top speed, samples added to Ground Truth and cheers received. A speech announcer calls the start, big air, splits every six gates and the result. A finish photo of every run is kept in the database.

**The HUD.** A race broadcast that gets out of the way. A soft scrim at the top keeps white type readable over white snow without boxing everything in; numerals are Saira (a variable sports face bundled locally, so it works offline) in heavy italic with tabular figures. During a run only these stay up: top left, the course tag (resort and start altitude, run name, length, drop, gates) with the split stack under it; top centre, one timing cluster (a speed ring that fills to 90 km/h and turns amber past 60, the clock with the gap to the leader under it, gates made out of 33 with the STREIF streak, a gate track that fills red and blue, white for the next gate, amber-ringed for a miss), with the active powerup chips and the event callouts under it; cheers along the right edge; and bottom left, a small picture of your body (the mirrored webcam with the skeleton, lean and tuck bars over it) that shrinks while you race. The live Ground Truth counter, today's fastest and the mountain feed come back between runs. The middle of the frame is kept for the start prompt, the countdown, the tutorial and the arrow to the next gate. The start screen is one input and one button over the live 3D mountain, with your mirror on the right; raising both hands for two seconds puts you on the snow without touching anything, and the camera picker is folded away under a small settings toggle.

**The look.** ACES tone mapping (`?tone=agx` or `?tone=neutral` to compare), then a grade in display space: cool shadows, warm highlights, a little more colour, a gentle S-curve and fine grain (`?nograde` turns it off). The snow glitters like real snow in sunshine: every few centimetres an ice crystal gets its own facet normal, and the few that mirror the sun into your eye flash bright enough to bloom, changing as you move. A blue-white sheen lifts the snow at grazing angles.

**The board** (`/board.html`, for a second screen or a phone). A hillshaded top-down map of the real terrain with contour lines, the piste corridor, gates and finish, a live dot and trail for every skier with their speed, the dataset counter in large type, who is on the mountain, the leaderboard, the feed, a QR code to join, and three big buttons: cowbell, clap, fire. A tap on a phone rings in the game within a frame, names the sender on screen, bounces the crowd and feeds the crowd meter.

## How the body is read

Pose comes from MediaPipe Pose Landmarker (the full model, GPU with a CPU fallback) at 640×480. MediaPipe already runs a One Euro filter on its landmarks, so the game filters only the derived signals, each with its own One Euro (lean 1.2 Hz / β 4, crouch 1 Hz / β 2) timestamped by video frame, plus half a frame of prediction. Tracking works in tiers from whatever the camera sees, with hysteresis and a short dwell so it does not flicker:

- **Head and shoulders:** lean from the tilt of the shoulder line and the head carried to the side; crouch from the shoulders sinking; hop from the shoulders popping up.
- **Shoulders and hips:** lean from shoulders-across-hips normalised by torso length (the sine of trunk flexion, stable when you turn), plus the tilt; crouch from the torso sinking.
- **Full body:** feet offset, knee angles and leg span as well, and the G1's legs are driven directly.

Lean goes through an expo curve (flat near zero, no dead-zone jump) and is scaled by the player's own comfortable range, learned as a slowly decaying peak (full lock at 80 % of your biggest recent lean, anywhere from 0.12 to 0.8 raw), so a timid player still steers and a big leaner keeps proportional control instead of slamming between full left and full right. A hard sideways lean drops the hip midpoint a little in the image, so that is taken back out of the crouch, and the "turned away" check sits below the shoulder foreshortening a big lean causes. The standing baselines that crouch and hop are measured against never learn from a crouch, so a held tuck stays a full tuck (it used to fade to half within a few seconds, and standing up afterwards read as a hop). Frames where the detector re-centres (shoulders jump more than 0.3 torso or shoulder width changes 25 %) are held for up to three frames. If you turn away from the camera the lean holds and the label says "face the camera"; it also says "step closer" and "more light would help" when those are the problem. Pole plants are wrists dropping fast past the shoulder line (measured in shoulder widths per second), a double pole is both within 150 ms, and the start gesture reads arm direction so raised hands can leave the frame. Calibration is 0.8 s of standing upright and re-arms when you raise your arms. The standing baselines are never learned with an arm in the air (the start gesture lifts the shoulders); before the race they simply follow the player, and from the countdown on they hold through a tuck.

**Hops.** A hop is the whole body going up and coming straight back down within half a second, measured on the head and shoulders (always in view) against their own recent path, not against a standing baseline. Tuned on a player's real hops (head and shoulders up 0.1-0.3 torso lengths, hips just as much, shoulder width swinging 12-44 % with the arms): when the hips are seen they must rise with the shoulders (a shrug lifts only the shoulders), otherwise the head and shoulders must rise clearly; raising the arms is not a hop (in a labelled test the wrists rose 1.0-2.5 shoulder widths over the shoulders when the arms went up and 0.2-0.7 in hops, so a rise over 0.85, or an arm clearly up, rules it out, unless the hips went well up too, which only a big jump does); the torso length must match before the take-off and after the landing (walking toward or away from the camera changes it); and the body has to start coming back down (standing up from a tuck does not). It fires about a tenth of a second after the top of the hop. Pole plants are ignored mid-hop. A hop always leaves the snow once you are moving (higher the faster you go, plus a skating push when slow), and at the start gate the robot hops on the spot so you can see it was read; the tutorial has a HOP step.

**Tutorial.** On the first start line of a session the gold G1 stands beside you and demonstrates: lean left, lean right, get low, plant your poles. Each step completes when your signal matches for half a second (a plant counts at once), with a chime and sparks; then both arms up starts the race. Enter skips it, T replays it, `?tutorial=0` disables it. A gauge on the snow under the skier shows the lean needle and tuck bar the game is reading, green when tracking, amber while calibrating, grey when it sees nobody.

## How it plays

1. On the laptop, type your name and step onto the snow. Stand about two metres from the camera with your whole body in frame; hold still for a second and the skeleton turns mint.
2. Raise both hands. Three, two, one, go.
3. **Lean** left or right to carve (shoulders over hips, hips over feet). **Get low** to push off and tuck: a deep crouch pumps you forward up to about 36 km/h and cuts drag above that; stand tall to slow. **Hop** to skate when slow, or to jump once you're fast. Make the gates.
4. Ski into the powerups on the way down (no buttons; they trigger on contact).
5. At the finish you see your time, rank, gates, top speed, and how many motion samples you just added to Ground Truth.

**Powerups.** Fixed placements, the same on every run so times stay comparable: at load the game simulates one clean autopilot run through every gate (the physics is deterministic) and puts the pickups on that racing line. Every pickup answers with light, never a sound: a burst in its colour, a shockwave ring, a tint over the screen, a chip under the clock with a ring that drains as it runs out.

- **Slipstream rings** (cyan, four on the course): a 3.6 m hoop across the line with chevrons streaming through it and a pillar of light above. Ski through for a kick and two seconds of thrust; the field of view opens, the speed blur deepens and your ski tracks glow cyan.
- **Time crystals** (teal, in threes, 27 in all): a little off the line, so you choose. Each takes 0.1 s off your time; the bonus floats up off the clock.
- **Shield** (white-blue bubble, two): a hex-patterned bubble rides on the robot until it forgives your next crash (you smash through the tree in a shower of snow and keep most of your speed) or your next missed gate.
- **Magnet** (violet, two): the next three gates count a pass well outside the poles; they glow violet, the doorway between the panels widens and a violet ring turns on the snow under you.
- **Tuck charge** (a move, not a pickup): hold a low, straight tuck and it charges in three steps (sparks off the ski tails go blue, orange, pink); stand up to fire a boost worth the charge. It rewards real tuck technique and uses only what the camera already reads.

The attract autopilot skis the line cleanly and collects what lies on it, but skips the tuck charge.

Keyboard fallback: left/right lean, up accelerates, down tucks, space hops, P plants poles, Enter starts or skips the intro or tutorial, T replays the tutorial, R restarts, K recalibrates, C changes camera, V toggles the announcer, B rings a test cowbell. URL flags: `?autopilot=1` makes the G1 ski the course by itself (attract mode for the table; it makes every gate at about 45 km/h), `?length=full` or `?length=400` sets the race length in metres, `?lowfx=1` turns off shadows for a weak laptop, `?nocam=1` skips the camera, `?name=…&code=…&auto=1` prefills and submits the start form, `?cam=front` starts on the front camera.

## Hazard Intelligence: where hearts race

The name of the site is the idea. Ground Truth records how people move on real terrain so a robot can learn from it; Presage adds how that terrain *felt*. During every run the game stores, about once a second, a fresh and confident heart-rate reading registered to the exact spot on the real Streif (position, metres along the course, heart rate, breathing, and the rise over that skier's own resting rate) in SpacetimeDB, beside the motion trace (`vital_chunk` table, `push_vitals` reducer). Pooled across everyone, by 25 m stretch of course, that is a live hazard map of the mountain: the human stress response as a label on the ground, which tells a robot (or a course designer, or a ski patrol) which stretches people find dangerous.

- **On the snow:** stretches where hearts ran 6 %+ over resting carry a translucent band across the piste (amber, red past 12 %) and a warning sign at their entry: "HAZARD ♥ +14 % · hearts race here".
- **In the race:** entering one flashes "HAZARD ZONE · hearts +14 %" and the commentator calls it.
- **On the board:** the course line is coloured by the hazard map, with the number of heart readings, runs and hazard zones.
- **In the HUD:** the live counter shows heart readings pooled and hazard zones flagged; the result card shows how many readings your run added.

## The commentator (ElevenLabs)

A race commentator voiced by ElevenLabs (Flash model, a commentator-style voice from your account or `ELEVENLABS_VOICE_ID`): the welcome, the start, splits against the leader, streaks, powerups, big air, crashes and nets, missed gates, speed milestones, your heart rate (resting rate, "breathe" when it races, "ice in the veins" when it does not), hazard zones and the finish with your peak heart rate; when the booth goes quiet it fills in with your speed, gates to go, the gap to the leader or your pulse. Big moments cut in, routine calls wait their turn, stale chatter is dropped. The game asks `GET /api/tts?t=<line>` for each line and the key never reaches the page. Locally the dev server answers it (`ELEVENLABS_API_KEY` in `game/.env.local`, loopback only, every line cached in `game/.tts-cache/`, gitignored). On the deployed site the Vercel function `api/tts.js` answers it: set `ELEVENLABS_API_KEY` (and optionally `ELEVENLABS_VOICE_ID`) in the Vercel project's environment variables, never in the repo. The function only serves same-site requests, short plain-text lines and a per-visitor rate limit, and Vercel's CDN caches each line for a year, so a repeated line costs nothing; put a credit cap on the key as well. Without a key the browser's voice reads the same lines. Presage needs a long-lived WebSocket and a native SDK, which Vercel cannot host, so on the live site it runs on the Presage relay (`presage-relay/`, a small Node server with a Dockerfile): the game sends it JPEG frames at `VITE_PRESAGE_URL` (set in the Vercel project, then redeploy), and the key lives only in the relay's environment. `scripts/presage-relay.sh https://<site>` runs it on a laptop with a free Cloudflare tunnel for a demo; see `presage-relay/README.md` for a proper host. The start screen tells visitors their camera picture goes to the pulse server. Without `VITE_PRESAGE_URL` the live game runs without vitals.

## Your heart in the game (Presage)

Presage reads the player's heart rate and breathing from the same webcam, contactless. The game learns your resting rate while you stand in the start gate and reacts to how far above it you are during the run:

- **On the robot:** a light on the G1's chest pulses with your real heartbeat, teal when calm, amber, then red when your heart races.
- **On the screen:** the edges of the frame thump on every beat as your heart rate climbs, and close in red when it races.
- **Ice veins:** stay within about 5 % of your resting rate at speed and time crystals are worth double.
- **Steady:** if your heart races (+20 % over resting) the gates quietly widen a little and the announcer says "Breathe"; it eases off as you calm down.
- **At the finish:** average and peak heart rate, your resting rate and breathing rate on the result card.

The HUD shows a heart beating at your measured rate with the bpm, your resting rate and breaths per minute. Vitals are also logged in `?watch` recordings.

How it is wired: Presage has no in-browser SDK, so the game streams the webcam frames the pose tracker already uses (raw RGBA from a worker, timestamped from `requestVideoFrameCallback`) over a localhost WebSocket (`/__presage`) to the Vite dev server, where the official Node SDK (`@smartspectra/node-sdk`) measures pulse and breathing on this machine; Presage's server is only contacted to check the API key. Setup: put `PRESAGE_API_KEY=...` in `game/.env.local` (gitignored, read only by the dev server, never in the client bundle) and run the game with `npm run dev`. The endpoint only accepts connections from this machine. A reading needs your face in view and roughly still: the pulse warms up in about 12 s and breathing in about 30 s, so the start gate is where your resting rate is learned; mid-run the last good reading holds. `?nopresage` turns it off. The dev server logs a `[presage]` line every few seconds with the current readings and any hint (face the camera, hold still, more light).

## How it works

```
game/            Vite + TypeScript + Three.js + MediaPipe Tasks Vision
  src/course.ts  heightfield sampling, centerline progress, gate geometry (pure, tested in Node)
  src/physics.ts carving model: edge angle from lean, carve radius sidecut x cos(edge), grip across the
                 skis, glide friction, tuck-dependent drag, hops and natural air off rollers (pure, tested)
  src/pose.ts    MediaPipe Pose Landmarker -> lean, crouch, hop, hands up, knee angles; auto-calibration
  src/g1.ts      G1 kinematic tree from the MJCF + decimated GLB; MediaPipe -> G1 joint retarget
  src/world.ts   near and far terrain (with baked sun shadows), snow and mountain shaders, sky, forest, rocks, snow lances, nets, banners, chairlift, gates, start, finish arch, flags, crowd, trails, spray, confetti, bursts, the environment map
  src/props.ts   procedural models: snow-laden spruces, boulders, spectators, snow lances, flags
  src/powerups.ts the racing line, pickup placement, tokens, rings, crystals, shockwaves, the shield bubble and magnet aura
  src/post.ts    post-processing: bloom, speed blur, chromatic aberration, speed lines, vignette, flashes, colour grade and grain
  src/net.ts     SpacetimeDB client: identity in localStorage, 8 Hz state stream, queued calls, cache reads
  src/main.ts    game loop, phases, camera, ghosts of other skiers, dataset upload
  src/vitals.ts  heart rate and breathing: smoothing, the resting baseline, arousal, the heartbeat clock
  src/recorder.ts ?watch: pose log, events and side-by-side frames posted to the dev server (game/.rec/)
  src/board.ts   the live map
spacetime/       SpacetimeDB module (TypeScript, runs inside the database)
  player, skier (live state), run (with per-gate splits), trace_chunk (the dataset), dataset_stats, feed, challenge,
  outbox and run_photo (left in place for the team's iMessage agent on the `photon` branch), cheer (event table),
  tick_timer (drops stale skiers every 2 s); reducers join, set_state, start_run, push_trace, finish_run,
  create_challenge, link_code, mark_sent, send_cheer, push_photo
tools/           export_course.py (built course -> browser heightfield), export_g1.py (STL -> 1 MB GLB + kinematic JSON)
```

**Why SpacetimeDB.** The mountain is the database. Every browser streams its skier row ~8 times a second through a reducer and subscribes to everyone else's; ghosts, the leaderboard, the feed and the dataset counter are all just table subscriptions, with no game server in between. A scheduled reducer sweeps skiers whose tab died. The dataset itself is rows: each `trace_chunk` is 50 samples of body state (lean, crouch, knee angles) with the position, speed and slope of the real ground under it, which is exactly the record Ground Truth exists to collect.

## Run it

```bash
curl -sSf https://install.spacetimedb.com | sh        # spacetime CLI (once)
scripts/dev.sh                                        # local SpacetimeDB, publish, bindings, game on :5173
```

Then open `http://<laptop LAN IP>:5173/` on the laptop (the webcam needs localhost or HTTPS, so play on the laptop itself) and `/board.html` on a second screen or a phone.

For the demo on the hosted database: `spacetime login`, then `STDB=maincloud scripts/dev.sh`.

Rebuilding the assets: `.venv/bin/python resort.py "Kitzbühel" --run streif` then `.venv/bin/python tools/export_course.py kitzbuhel-streif`; `.venv/bin/python tools/export_far.py kitzbuhel-streif --cell 30 --zoom 12` for the horizon; `.venv/bin/python tools/export_g1.py`.

Tests: `cd game && node --experimental-strip-types test/physics.test.ts` drives an autopilot down the course without a browser, `test/powerups.test.ts` runs the attract autopilot with the powerups on the racing line (it should make every gate and pass through every ring), and `test/retarget.test.ts` checks the body-to-G1 mapping.

## Data and attribution

Piste lines © OpenStreetMap contributors (ODbL). Elevation: AWS Terrain Tiles (Mapzen terrarium). Unitree G1 model: unitree_rl_mjlab (Apache 2.0). Pose: MediaPipe Pose Landmarker (lite). Contributors can be removed from the dataset on request; traces carry no phone numbers or precise home locations, only positions on the piste.
