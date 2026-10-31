# skibot

data collection & rl envs for humanoid training. skiing and hiking.

tooling to collect human motion on real terrain and turn it into training envs for a unitree g1. the terrain is real (openstreetmap + elevation tiles + lidar), the body is a pretrained whole body controller (nvidia gear-sonic), and a small ppo policy on top learns the activity.

## data collection

every recording is registered to the ground it happened on, so motion, terrain and vitals line up with the training envs.

- **phone recorder** (`ground/`, `ios/GroundTruth/`). an iphone app records a hike in the background (100 hz motion, gps, barometer, steps) and uploads it. the server ties it to the trail under it: distance, climb, osm trail match, cadence, gaps. opt in and summaries go over imessage (photon agent). see [ground/README.md](ground/README.md).
- **browser capture** (`game/`, `presage-relay/`). webcam pose in the browser on the real streif, retargeted onto the g1 live, many people at once on one shared course. each session stores human motion against real terrain, plus heart rate and breathing read from the same camera (presage), pooled into a hazard map of where on the terrain people's hearts race. live at [hazard-intelligence-seven.vercel.app](https://hazard-intelligence-seven.vercel.app). see [GAME.md](GAME.md).
- **gopro runs** (`gopro.py`). gopro imu/gps ski runs (zenodo) matched to the terrain.
- **race video** (`skisim/pose_from_video.py`). world cup giant slalom video to g1 joint targets for athletic posture.
- **one database.** everything that isn't code (maps, terrain, recordings, training inputs, the policy registry) lives in one spacetimedb database. see [DATABASE.md](DATABASE.md).

## training envs

- **skiing** (`skisim/`, `train/`). ski physics mujoco can't do on its own, any resort as a course, perturbed terrain with a curriculum, 4096 robots on one h100.
- **hiking** (`hikesim/`, `train/hike_*.py`). trails from usgs 1 m lidar, rocks and roots added, sonic's kinematic planner for steps. see [HIKING.md](HIKING.md).
- **policy api** (`policy_api/`). serves trained policies over http and an openpi-compatible websocket, so any physics loop can run them zero shot.

## results so far

**skiing.** fixed benchmark: 24 terrain tiles (real course windows and synthetic slopes, 8 to 31 deg), 12 s per run.

| | sonic alone | policy v1 (iter 900) | policy v2 (iter 1325) |
|---|---|---|---|
| runs without a fall | 1 / 24 | 18 / 24 | 17 / 24 |
| mean distance | 11.4 m | 52.6 m | 81.9 m |
| heading error | 29.8 deg | 16.5 deg | 8.6 deg |
| ski alignment gap | | 31.2 deg | 3.1 deg |
| a ski in the air | | 21% | 7% |
| stance width (hips about 0.24 m) | | 0.44 m | 0.27 m |

v1 learned to stay up with crossed, scissored skis. v2 added parallel ski, both skis down, stance width and edge agreement rewards and fixed the look in about 250 iterations.

ski physics on a rigid test sled vs closed form answers:

| check | simulated | expected |
|---|---|---|
| straight glide, 15 deg, 8 s | 14.91 m/s | 14.90 m/s |
| carve radius at a 30 deg edge | 14.58 m | 14.72 m (sidecut x cos edge) |
| sideways skid from 5 m/s | stops in 2.00 s | 2.04 s |
| flat ski across a 20 deg slope, 2 s | slides 2.10 m | 2.10 m |

**hiking.** sonic alone walks easy ground well and falls once a climb passes about 12 deg. the trained policy's benchmark isn't in yet.

## how the ski env works

- **ski physics** (`skisim/ski.py`, `ski_batch.py`, `ski_torch.py`). mujoco friction is the same in every direction, which can't describe a ski. each ski touches snow through spring points along both edges, with glide friction along the ski, edge grip that grows with edge angle, and sidecut steering (an edged ski carves sidecut x cos(edge)). six snow conditions (groomed, hardpack, ice, soft, slush, powder) change grip, glide and sink. numpy and torch versions match the reference to 1e-13 and 1.7e-4.
- **any resort** (`resort.py`, `course.py`). resort name to nominatim, piste lines from openstreetmap via overpass, elevation from aws terrain tiles, stitched into a 1.2 km course with giant slalom gates at g1 scale. 21 courses so far, including bormio stelvio, cortina olimpia delle tofane, kitzbuehel streif, adelboden, garmisch kandahar, val d'isere bellevarde, beaver creek golden eagle, whistler, jackson hole, zermatt and hakuba.
- **terrain perturbation** (`skisim/perturb.py`). windows cut along real runs plus synthetic slopes, perturbed with pitch scale, side tilt, bumps, moguls, rollers, jumps, compressions, chutes and flats, packed into one map with a difficulty curriculum. the steep bank has 128 tiles from 4.5 to 42 deg.
- **pretrained body** (`skisim/sonic.py`, `skisim/sonic_torch.py`). gear-sonic's encoder and decoder wired to match its c++ deploy reference (50 hz, 10 frame history, isaaclab joint order), rebuilt in torch for batched gpu use. zero shot on skis it holds a straight run: 96 m on 8 deg, 93 m on 15 deg before falling.
- **policy** (`train/ski_env.py`, `train/train.py`). frozen sonic underneath. the policy outputs a body lean target plus residual joint offsets (31 actions) from 141 observations: proprioception, ski touch, edge angles, terrain heights ahead, target heading and speed. mujoco warp runs 4096 robots on one h100 at about 63k robot steps per second.
- **live** (`train/live.py`). runs the newest checkpoint in real time, pulls new ones every minute, steer by touch from a laptop or phone. `train/autoeval.py` scores and renders each checkpoint on the benchmark.

## layout

```
resort.py, course.py      any resort to a mujoco course (osm + elevation)
ground.py                 trails and lidar for a place (hiking)
gopro.py                  gopro imu/gps ski runs matched to the terrain
skisim/                   ski physics, terrain, scenes, sonic, perturbation, pose from video
hikesim/                  hiking courses, planner, rough ground, training tiles, web viewer
train/                    batched envs, ppo launchers (local + modal), live viewer, eval
ground/                   phone recorder server, trail registration, photon imessage agent
ios/GroundTruth/          ios app that records a hike in the background
game/                     browser capture: webcam pose and vitals on real terrain
web/, api/                site, api docs and tts function
presage-relay/            heart rate and breathing relay for browser capture
spacetime/, tools/        spacetimedb module and load/export tools
policy_api/               serves trained policies
assets/unitree_g1/        unitree g1 29 dof model (unitree_rl_mjlab, apache 2.0)
scripts/                  asset fetch, dev server
```

## run it

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mujoco numpy scipy pillow requests onnxruntime joblib "mediapipe==0.10.14" opencv-python-headless
uv venv .venv-train --python 3.12 && uv pip install --python .venv-train/bin/python mjlab==1.6.0 onnx tensorboard
scripts/fetch_assets.sh

.venv/bin/python -m skisim.tests.test_physics                        # ski physics checks
.venv/bin/python resort.py "Bormio" --run stelvio                    # any resort to out/courses/<slug>
.venv/bin/python -m skisim.perturb --n 128 --cols 8 --steep 1.5 --out out/terrains_v3
PYTHONPATH=. .venv-train/bin/python -m train.train --device cpu --num-envs 16 --iters 2 --terrain out/terrains_v3   # cpu smoke test
modal run --detach train/modal_train.py --num-envs 4096 --terrain out/terrains_v3                                  # h100 training
PYTHONPATH=. .venv-train/bin/python -m train.live --terrain out/terrains_v3                                        # http://127.0.0.1:8766
PYTHONPATH=. .venv-train/bin/python -m train.play --ckpt runs_live/<run>/model_<iter>.pt                          # benchmark + videos
```

hiking: [HIKING.md](HIKING.md). browser capture: `scripts/dev.sh`, see [GAME.md](GAME.md). phone recorder: [ground/README.md](ground/README.md).

## data and licenses

- unitree g1 model and meshes: unitree_rl_mjlab, apache 2.0 (`assets/unitree_g1/LICENSE`).
- nvidia gear-sonic weights and kinematic planner: nvidia open model license, not stored here, fetched by `scripts/fetch_assets.sh`. reference code: nvlabs/gr00t-wholebodycontrol, apache 2.0.
- piste lines and trails: openstreetmap contributors, odbl. elevation: aws terrain tiles (mapzen / tilezen sources, attribution required) and usgs 3dep 1 m lidar (public domain). aerial photos: usda naip, public domain.
- gopro ski runs on marmolada: prochazka, cc by 4.0, doi:10.5281/zenodo.21777004 (optional, not stored here).
- race videos aren't stored here. they're only used locally to measure posture.
- mujoco, mujoco warp, mjlab, rsl_rl and mediapipe under their own licenses.
