# Hiking

The skibot stack, on foot: a Unitree G1 hiking real Ann Arbor trails in MuJoCo. Trails come from OpenStreetMap, the ground from USGS 1 m lidar, and the steps from NVIDIA GEAR-SONIC's kinematic planner. A residual policy trained with PPO on thousands of robots teaches it to climb. Everything lives next to the ski code in the same layout: `hikesim/` beside `skisim/`, `train/hike_*.py` beside the ski trainer.

## Results

Fill in from `out/hike/bench/*.json` once the trained policy is benchmarked.

| | SONIC alone | Trained policy |
|---|---|---|
| 100 m stretches finished (6 courses x first, middle, steepest) | | |
| Steepest stretches finished | | |
| Mean distance before a fall | | |

## How it works

- **Trails and terrain** (`ground.py --place "Ann Arbor, Michigan"`). 1,309 trail ways (164 km) in 72 parks, each with an elevation profile from USGS 3DEP 1 m lidar, cached as 2 km UTM tiles under `data/cache/3dep/utm17n`.
- **Courses** (`hikesim/course.py`). The longest walk through one park's trail network, cut out of the lidar as a MuJoCo heightfield: Bird Hills, Nichols Arboretum, Bluffs, Kuebler Langford, Barton and County Farm, 0.8 to 1.8 km each, steepest grades 11 to 29 deg.
- **Steps from SONIC's planner** (`hikesim/planner.py`). `planner_sonic.onnx` turns walk, slow walk or careful commands plus a direction into whole body motion at 30 Hz. The wrapper follows NVIDIA's C++ deploy stack: context from the current plan, 30 to 50 Hz resampling, an 8 frame crossfade, replans every 0.5 s. The planner walks at about 0.64x the commanded speed with this cadence, and SONIC reproduces the planned speed in physics within 0.03 m/s.
- **Zero shot hiking** (`hikesim/walk.py`). SONIC tracking the planner on the real course in CPU MuJoCo, steered at a point 3 m ahead on the trail. It walks well on easy ground and falls once a climb passes about 12 deg, at the same spot whatever the speed or gait.
- **Bumpy ground** (`hikesim/rough.py`). 1 m lidar is smooth, real trails are not: fractal roughness (about 2 cm RMS), root ridges up to 8 cm and rocks up to 15 cm, added on a 0.25 m grid. The noise is hashed from absolute position, so the physics, the renders and any crop of a course see the same rocks. Each foot stands on a plane fitted through 5 points under its sole and lifted onto the highest, so a rock under the toe tilts the foot.
- **Satellite floor** (`hikesim/sat.py`). Renders drape the course's USGS NAIP aerial photo (0.3 m) on the ground at 2048 px with the trail painted in as packed dirt and leaf-litter grain, and a low sun so roots and rocks cast shadows.
- **Training terrain** (`hikesim/tiles.py`). Straight 40 m stretches of every trail, turned so the trail runs along +x, in both directions, plus 1.25x and 1.5x steepened copies. Copies whose walking strip has a cell over 45 deg are dropped (2x copies turned hillside trails into 60 deg side slopes). Each tile gets bumps at a random strength (0.3x to 1.5x). 219 tiles at 0.25 m from -26 to +26 deg, sorted by difficulty (grade, steep spots, bumps) for a curriculum.
- **Walking clips** (`hikesim/clips.py`). The planner is CPU only, so six 40 s clips (slow walk, walk at four speeds, careful) are precomputed on Modal and pointed along each robot's heading in training.
- **Policy** (`train/hike_env.py`, `train/hike_train.py`). Frozen SONIC tracks the clip; the policy adds a body lean (pitch, roll) to SONIC's reference and a residual on its 29 joint actions, from 142 observations (proprioception, terrain heights ahead, heading and speed command, the clip's leg pose, foot contact). Rewards: speed along the trail, heading, staying on the line, upright, smooth; a fall ends the episode. MuJoCo Warp, 4096 robots on one H100 (`train/hike_modal.py`), about 2.5 s per PPO iteration.
- **Evaluation on the real courses** (`train/hike_play.py`). The training env with a course's lidar in place of the tiles and the trail itself to follow, so the policy sees exactly what it trained on. `--zero` runs SONIC alone in the same env as the baseline.
- **Viewer** (`hikesim/export_web.py`, `hikesim/web/index.html`). three.js in the browser: the whole area with every trail coloured by steepness over the NAIP aerial photo, each course at 2 m with the photo at 0.5 m, and every run replayed on the real G1 meshes with a chase camera. A second run can ride along as a ghost, so the trained policy and SONIC alone climb the same stretch side by side.

## Things we learned the hard way

- **MuJoCo Warp heightfield contact is not safe on lidar terrain.** Foot spheres or capsules against the hfield occasionally get a contact depth of 0.1 to 22 m in a single 2 ms substep (mujoco-warp 3.11 and 3.14), which launches the robot. Smaller steps, more solver iterations, smoothing, gentler tiles, sphere feet and feet-only collision did not fix it. What did: the hfield is visual only, and each foot stands on its own mocap box that follows the lidar surface (height and slope) under it every substep. Blow-ups went from 6 to 9 in 32 robots per 6 s to 0, and training went from 15 to 28% of episodes ending in a blow-up to 0%.
- **A blown-up world needs a full reset.** Zero `qacc_warmstart` with `qpos` and `qvel`, sanitise positions before any terrain lookup (a NaN became a wild index and a CUDA assert), and give the world only the fall penalty so 1e9 m/s velocities never reach the value function.
- **mujoco-warp 3.14 prints a line search warning every step**, which floods Modal's log rate limit and halves throughput. The env turns that warning off.

## Run it

```bash
uv venv .venv-hike --python 3.12 && uv pip install --python .venv-hike/bin/python mujoco==3.14.0 mujoco-warp==3.14.0 torch rsl-rl-lib==5.4.2 tensordict onnx onnxruntime tensorboard scipy pillow
scripts/fetch_assets.sh                                   # SONIC weights and the 774 MB planner
.venv/bin/python ground.py --place "Ann Arbor, Michigan"  # trails and lidar
.venv/bin/python -m hikesim.course --all                  # six courses
.venv/bin/python -m hikesim.walk bird-hills-nature-area --length 60      # SONIC alone, CPU MuJoCo
.venv/bin/python -m hikesim.tiles                          # training mosaic
modal run train/hike_modal.py::clips                       # planner walking clips
modal run --detach train/hike_modal.py --iters 3000        # PPO on an H100
PYTHONPATH=. .venv-hike/bin/python -m train.hike_play --ckpt runs_hike/<run>/model_<n>.pt --bench
.venv/bin/python -m hikesim.render out/hike/runs/<course>/<run>
.venv/bin/python -m hikesim.export_web && python3 -m http.server 8770 -d out/hike/web
```

## Data and licenses

- Trails: OpenStreetMap contributors, ODbL. Elevation: USGS 3DEP 1 m lidar, public domain. Aerial photos: USDA NAIP via USGS The National Map, public domain.
- NVIDIA GEAR-SONIC weights and kinematic planner: NVIDIA Open Model License, not stored here, fetched by `scripts/fetch_assets.sh`.
- Unitree G1 model and meshes: unitree_rl_mjlab, Apache 2.0.
