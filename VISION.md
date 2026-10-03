# Ground Truth

**Every step a person takes on real ground becomes something a robot can learn from.**

skibot taught a Unitree G1 to ski real World Cup pistes in simulation. Ground Truth is where that goes next: a data collector that records how people move on known terrain, from a phone, a GoPro or a video, and ties every sample to the exact ground under it. The robot then learns from that data on the same ground in physics, and the dataset grows with every run, hike and walk anyone records.

## The problem

Robots learn to move from people. The human motion they learn from was mostly captured in studios, on flat floors, and none of it records what was underfoot. GPS apps know where millions of people hiked and skied, but not how their bodies moved. So a humanoid that leaves the lab meets snow, scree, mud and stairs with no human example of moving on that exact ground.

We have not found anyone collecting motion and terrain together, at scale, outside a lab.

## The idea

Three ways in, one place everything lands.

| Source | What it gives | What it misses |
|---|---|---|
| Phone, GoPro or watch on a known trail or piste | Position, speed, turns, steps, slips and orientation, at up to 100 Hz | Full body motion: it sees one point on the body |
| Video, from a webcam or YouTube | Full body motion | The ground under it |
| skibot | The same place rebuilt as physics, with a robot that can try it | Anything human: it needs the other two |

Every sample is registered to the ground: elevation and slope from terrain data, plus the trail or piste it sits on and that trail's tags (surface, difficulty) from OpenStreetMap. Video gives the body, the phone gives the ground and the timing, and the simulator closes the loop:

1. **Capture.** A person skis, hikes or walks with a phone, or uploads a video.
2. **Register.** Each sample is tied to the ground under it.
3. **Replay.** skibot rebuilds that ground in MuJoCo and the G1 tries the motion.
4. **Learn.** Training across many people and many places teaches the robot to move on ground it has never seen.
5. **Compare.** The robot's run is scored against the human's on the same ground, and the gaps say what to collect next.

## Why skiing first

Skiing is the hardest version of real ground: low friction, high speed, steep slopes, and a body that has to stay balanced the whole way down. If the loop works on a World Cup piste, a hiking trail is easier. Most of the pieces already exist.

## What exists today

From skibot (the README has the numbers and how to run it):

- **Ski physics** that matches closed form answers on a test sled within 2%, from straight glide to carve radius to sideways skid.
- **Any resort to a course.** A resort name becomes a MuJoCo course built from OpenStreetMap pistes and elevation tiles. 21 courses so far, from Bormio Stelvio to the Streif.
- **A robot that learned to ski.** NVIDIA GEAR-SONIC underneath, a PPO policy on top, trained on thousands of robots at once on one H100. Clean runs on the 24 tile benchmark went from 1 with SONIC alone to 19 with the best checkpoint so far.
- **Real human ski data on known terrain.** About 79 minutes of GoPro accelerometer, orientation and GPS from Marmolada (Prochazka, CC BY 4.0), split into descents and matched to the OpenStreetMap pistes they followed. This is exactly the record Ground Truth collects, so it is the first entry.
- **Body motion from video.** MediaPipe 3D pose on World Cup giant slalom runs, turned into G1 joint targets for neutral and turning stances.
- **Live control.** A local server runs the newest checkpoint in real time and takes steering from a phone.

## What we build at MHacks 2026

Three stations at the table:

1. **Webcam.** Ski in front of the laptop and the G1 copies you on a real slope.
2. **Phone.** Scan a QR code, walk the hall or the stairs, and watch your trace land on a shared live map with the ground under it. The dataset counter goes up while you walk.
3. **YouTube.** Paste a World Cup run. Gemini finds the clean skiing segments, GVHMR on Modal pulls out the 3D body motion, and the G1 skis it in physics on the same piste.

Built with SpacetimeDB for the live shared map, Gemini for reading video, Modal for training and video processing, and a Fetch.ai agent that takes a video or a hike file and hands back motion a robot can use.

## How we will know it works

Two tests, not run yet:

- **The phone as a terrain sensor.** Estimate slope from the GoPro motion data alone, and score it against the elevation data on descents held out of training.
- **Data a robot can use.** Out of a set of YouTube runs, how many the G1 completes in physics.

## Where it goes

- **Every trail.** Passive data from hikers becomes live conditions: where people slowed, slipped or turned back means ice, mud or a washout. Land managers see which trails need work first.
- **Every robot.** A motion library tagged by terrain, so a robot can train for a place before it goes there: a ski slope, a mountain trail, a stairwell, a rescue site.
- **Every sport.** The same pipeline works for trail running, climbing approaches and backcountry touring, anywhere the ground decides how people move.
- **Open, with consent.** Contributors opt in and can delete their data, and the start and end of every trace are trimmed so no one's home shows up on a map.

The MHacks 2026 theme is "build something that grows." Ground Truth grows with every step someone records, and the robot grows with it.

## Honest limits

- A phone sees one point on the body. Full body motion needs video.
- In a phone browser, iOS gives motion sensors, compass and GPS only after a tap and over HTTPS, at about 60 Hz, with no barometer, and the sensors stop when the screen locks. Long hikes come in as files from a sensor logging app or a GoPro.
- GVHMR uses the SMPL-X body model, which is free for research but needs registration and is not licensed for commercial use.
- Everything robotic is in simulation. There is no real G1 on a real slope yet.
- Race videos are used to measure motion and are not redistributed.
