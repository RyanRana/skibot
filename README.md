# skibot

Unitree G1 humanoids skiing real Olympic and World Cup pistes in MuJoCo. Type a resort name and the run's real slope becomes a MuJoCo heightfield. A pretrained whole body controller (NVIDIA GEAR-SONIC) keeps the robot moving like a person, and a small policy trained with PPO on thousands of parallel robots teaches it to ski: edge, carve through gates, hold speed and stay upright on terrain it has never seen.

## MHacks 2026: Ground Truth

The hackathon build on top of this repo: ski the real Streif in the browser with your body (webcam pose), with the Unitree G1 copying you, everyone live on one shared mountain through SpacetimeDB, every run adding to a shared motion-on-terrain dataset, and an iMessage agent on Photon that hands out join codes and texts results. See [GAME.md](GAME.md) for what it looks like, how it plays and how to run it (`scripts/dev.sh`). The long version of the idea is in [VISION.md](VISION.md).

## Results so far

Fixed benchmark: 24 terrain tiles (real course windows and synthetic slopes, 8 to 31 degrees), 12 s per run.

| | SONIC alone | Policy v1 (iter 900) | Policy v2 (iter 1325) |
|---|---|---|---|
| Runs without a fall | 1 / 24 | 18 / 24 | 17 / 24 |
| Mean distance | 11.4 m | 52.6 m | 81.9 m |
| Heading error | 29.8 deg | 16.5 deg | 8.6 deg |
| Ski alignment gap | | 31.2 deg | 3.1 deg |
| A ski in the air | | 21% | 7% |
| Stance width (hips about 0.24 m) | | 0.44 m | 0.27 m |

v1 learned to stay up with crossed, scissored skis. v2 added parallel ski, both skis down, stance width and edge agreement rewards and fixed the look in about 250 iterations.

Ski physics checks on a rigid test sled against closed form answers:

| Check | Simulated | Expected |
|---|---|---|
| Straight glide, 15 deg, 8 s | 14.91 m/s | 14.90 m/s |
| Carve radius at a 30 deg edge | 14.58 m | 14.72 m (sidecut x cos edge) |
| Sideways skid from 5 m/s | stops in 2.00 s | 2.04 s |
| Flat ski across a 20 deg slope, 2 s | slides 2.10 m | 2.10 m |

## How it works

- **Ski physics** (`skisim/ski.py`, `ski_batch.py`, `ski_torch.py`). MuJoCo friction is the same in every direction, which cannot describe a ski. Each ski touches the snow through spring points along both edges, and the model adds glide friction along the ski, edge grip that grows with edge angle, and sidecut steering (an edged ski carves a radius of sidecut x cos(edge)). Six snow conditions (groomed, hardpack, ice, soft, slush, powder) change grip, glide and how far the skis sink. The numpy and torch versions match the reference to 1e-13 and 1.7e-4.
- **Any resort** (`resort.py`, `course.py`). Resort name to Nominatim, piste lines from OpenStreetMap via Overpass, elevation from AWS Terrain Tiles, stitched into a 1.2 km course with giant slalom gates at G1 scale. 21 courses built so far, including Bormio Stelvio, Cortina Olimpia delle Tofane, Kitzbuehel Streif, Adelboden, Garmisch Kandahar, Val d'Isere Bellevarde, Beaver Creek Golden Eagle, Whistler, Jackson Hole, Zermatt and Hakuba.
- **Terrain perturbation** (`skisim/perturb.py`). Windows cut along the real runs plus synthetic slopes, perturbed with pitch scale, side tilt, rolling bumps, moguls, rollers, jumps, compressions, chutes and flat stretches, packed side by side into one map for training with a difficulty curriculum. The steep bank has 128 tiles with mean slopes from 4.5 to 42 degrees.
- **Pretrained body** (`skisim/sonic.py`, `skisim/sonic_torch.py`). GEAR-SONIC's encoder and decoder, wired to match its C++ deploy reference (50 Hz, 10 frame history, IsaacLab joint order), then rebuilt in torch for batched GPU use. Zero shot on skis it already holds a straight run: 96 m on 8 deg, 93 m on 15 deg before falling.
- **Policy** (`train/ski_env.py`, `train/train.py`). Frozen SONIC underneath. The policy outputs a body lean target for SONIC plus residual joint offsets (31 actions) from 141 observations: proprioception, ski touch (pressure under the front and rear of each edge), edge angles, terrain heights ahead, and the target heading and speed. Episodes race alternating gates or follow free heading commands. MuJoCo Warp runs 4096 robots on one H100 at about 63k robot steps per second.
- **Athletic posture from race video** (`skisim/pose_from_video.py`). MediaPipe 3D pose on World Cup giant slalom runs, filtered to skiing frames, turned into G1 joint targets for neutral, left turn and right turn stances. SONIC tracks that posture and blends between the turn shapes as the policy leans.
- **Live** (`train/live.py`). A local server runs the newest checkpoint in real time, pulls new checkpoints from the training run every minute, and lets you steer by touch (gates mode or free steer) from a laptop or phone. `train/autoeval.py` scores and renders each new checkpoint on the benchmark.

## Layout

```
resort.py, course.py      any resort to a MuJoCo course (OSM + elevation)
gopro.py                  GoPro IMU/GPS ski runs (Zenodo) matched to the terrain
build_resorts.sh          builds a list of courses
skisim/                   ski physics, terrain, scenes, SONIC, perturbation, fleet renders, pose from video
skisim/tests/             physics and controller checks
train/                    batched env, PPO launcher, Modal launcher, live viewer, eval and renders
assets/unitree_g1/        Unitree G1 29 DoF model (from unitree_rl_mjlab, Apache 2.0)
scripts/fetch_assets.sh   downloads SONIC weights (and optional data)
```

## Run it

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mujoco numpy scipy pillow requests onnxruntime joblib "mediapipe==0.10.14" opencv-python-headless
uv venv .venv-train --python 3.12 && uv pip install --python .venv-train/bin/python mjlab==1.6.0 onnx tensorboard
scripts/fetch_assets.sh

.venv/bin/python -m skisim.tests.test_physics                        # ski physics checks
.venv/bin/python resort.py "Bormio" --run stelvio                    # any resort to out/courses/<slug>
.venv/bin/python -m skisim.perturb --n 128 --cols 8 --steep 1.5 --out out/terrains_v3
PYTHONPATH=. .venv-train/bin/python -m train.train --device cpu --num-envs 16 --iters 2 --terrain out/terrains_v3   # CPU smoke test
modal run --detach train/modal_train.py --num-envs 4096 --terrain out/terrains_v3                                  # H100 training
PYTHONPATH=. .venv-train/bin/python -m train.live --terrain out/terrains_v3                                        # http://127.0.0.1:8766
PYTHONPATH=. .venv-train/bin/python -m train.play --ckpt runs_live/<run>/model_<iter>.pt                          # benchmark + videos
```

## Data and licenses

- Unitree G1 model and meshes: unitree_rl_mjlab, Apache 2.0 (`assets/unitree_g1/LICENSE`).
- NVIDIA GEAR-SONIC weights: NVIDIA Open Model License, not stored here, fetched by `scripts/fetch_assets.sh`. Reference code: NVlabs/GR00T-WholeBodyControl, Apache 2.0.
- Piste lines: OpenStreetMap contributors, ODbL. Elevation: AWS Terrain Tiles (Mapzen / Tilezen sources, attribution required).
- GoPro ski runs on Marmolada: Prochazka, CC BY 4.0, doi:10.5281/zenodo.21777004 (optional, not stored here).
- Race videos are not stored here. They are only used locally to measure posture.
- MuJoCo, MuJoCo Warp, mjlab, rsl_rl and MediaPipe under their own licenses.
