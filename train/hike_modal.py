"""Train the G1 hiking policy on a Modal H100. Checkpoints and tensorboard logs land in the Modal volume g1-hike-runs
(separate from the ski run's g1-ski-runs).

Walking clips first (planner on 16 CPU cores, ~1 min):  ~/mhacks-spikes/.venv/bin/modal run train/hike_modal.py::clips
Launch training (detached):  ~/mhacks-spikes/.venv/bin/modal run --detach train/hike_modal.py::main --iters 3000
  (leave the local client running: killing it, e.g. with `timeout`, cancels the run even with --detach)
Benchmark on the courses:    ~/mhacks-spikes/.venv/bin/modal run train/hike_modal.py::bench --ckpt <run>/model_<n>.pt  (empty = SONIC alone)
Pull results:                ~/mhacks-spikes/.venv/bin/modal volume get g1-hike-runs <run>/ runs_hike/
"""

from __future__ import annotations

import modal

ROOT = "/root/hike"

image = (
  modal.Image.debian_slim(python_version="3.12")
  .apt_install("git", "libegl1", "libgl1", "libglib2.0-0")
  .pip_install("mujoco==3.14.0", "mujoco-warp==3.14.0", "torch==2.14.1", "rsl-rl-lib==5.4.2", "tensordict", "onnx==1.23.1",
               "tensorboard", "scipy", "pillow")
  .env({"PYTHONPATH": ROOT, "PYTHONUNBUFFERED": "1", "GIT_PYTHON_REFRESH": "quiet", "MUJOCO_GL": "egl"})
  .add_local_dir("skisim", f"{ROOT}/skisim", ignore=["__pycache__", "*.pyc", "tests"])
  .add_local_dir("hikesim", f"{ROOT}/hikesim", ignore=["__pycache__", "*.pyc", "web"])
  .add_local_dir("train", f"{ROOT}/train", ignore=["__pycache__", "*.pyc"])
  .add_local_dir("assets/unitree_g1", f"{ROOT}/assets/unitree_g1")
  .add_local_file("assets/sonic/model_encoder.onnx", f"{ROOT}/assets/sonic/model_encoder.onnx")
  .add_local_file("assets/sonic/model_decoder.onnx", f"{ROOT}/assets/sonic/model_decoder.onnx")
  .add_local_dir("out/hike/tiles", f"{ROOT}/out/hike/tiles")
  .add_local_dir("out/hike/courses", f"{ROOT}/out/hike/courses", ignore=["*.json.bak", "canopy.npy"])
)
clip_image = (
  modal.Image.debian_slim(python_version="3.12")
  .apt_install("curl")
  .pip_install("mujoco==3.3.7", "numpy", "scipy", "pillow", "onnxruntime==1.23.2")
  .run_commands("mkdir -p /root/hike/assets/sonic_planner && curl -fsL -o /root/hike/assets/sonic_planner/planner_sonic.onnx "
                "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/planner_sonic.onnx")
  .env({"PYTHONPATH": ROOT, "PYTHONUNBUFFERED": "1"})
  .add_local_dir("skisim", f"{ROOT}/skisim", ignore=["__pycache__", "*.pyc", "tests"])
  .add_local_dir("hikesim", f"{ROOT}/hikesim", ignore=["__pycache__", "*.pyc", "web"])
  .add_local_dir("assets/unitree_g1", f"{ROOT}/assets/unitree_g1")
)
vol = modal.Volume.from_name("g1-hike-runs", create_if_missing=True)
app = modal.App("g1-hike-train", image=image)


@app.function(image=clip_image, cpu=16, memory=8192, timeout=1800, volumes={"/runs": vol})
def clips(seconds: float = 40.0):
  import subprocess
  import sys
  subprocess.run([sys.executable, "-m", "hikesim.clips", "--seconds", str(seconds), "--out", "/runs/clips.npz"], cwd=ROOT, check=True)
  vol.commit()


@app.function(gpu="H100", timeout=6 * 3600, volumes={"/runs": vol})
def train(iters: int = 3000, num_envs: int = 4096, resume: str = "", init_std: float = 0.0, start_level: int = 40, env: str = "sonic"):
  import subprocess
  import sys
  import threading
  import time

  def commit_loop():
    while True:
      time.sleep(60)
      try:
        vol.commit()
      except Exception as e:
        print("volume commit failed:", e, flush=True)

  threading.Thread(target=commit_loop, daemon=True).start()
  cmd = [sys.executable, "-m", "train.hike_train", "--device", "cuda", "--num-envs", str(num_envs), "--iters", str(iters),
         "--log-dir", "/runs", "--clips", "/runs/clips.npz", "--init-std", str(init_std),
         "--start-level", str(start_level), "--env", env]
  if resume:
    cmd += ["--resume", resume]
  subprocess.run(cmd, cwd=ROOT, check=True)
  vol.commit()


@app.function(cpu=4, memory=8192, timeout=3600, volumes={"/runs": vol})
def evaluate_one(job: dict) -> dict:
  """One course stretch on CPU (the course env is a single robot), so a benchmark fans out across containers."""
  import io
  import os
  import shutil
  import sys
  import numpy as np
  sys.path.insert(0, ROOT)
  os.chdir(ROOT)
  os.makedirs(f"{ROOT}/out/hike", exist_ok=True)
  shutil.copy("/runs/clips.npz", f"{ROOT}/out/hike/clips.npz")
  from train.hike_play import RUNS, evaluate
  ckpt = f"/runs/{job['ckpt']}" if job.get("ckpt") else None
  res = evaluate(job["course"], job["start"], job["length"], ckpt, job.get("clip", "walk_1"), name=job["name"])
  d = RUNS / job["course"] / job["name"]
  return {"stretch": job["stretch"], **res, "_frames": (d / "frames.npz").read_bytes()}


@app.local_entrypoint()
def bench(ckpt: str = "", length: int = 100, clip: str = "walk_1", tag: str = ""):
  """Benchmark a checkpoint (path inside the volume, e.g. 20261003-215725/model_800.pt) or SONIC alone (empty)."""
  import json
  from pathlib import Path
  import numpy as np
  # Runs in the Modal CLI's Python: read the courses directly instead of importing hikesim (PIL, scipy, mujoco).
  label = tag or ("sonic_only" if not ckpt else ckpt.replace("/", "_").replace(".pt", ""))
  courses = sorted(p.parent.name for p in Path("out/hike/courses").glob("*/route.json"))
  jobs = []
  for s in courses:
    route = json.loads((Path("out/hike/courses") / s / "route.json").read_text())
    n = len(route["x"])
    g = np.abs(np.array(route["grade_deg"]))
    steep = int(np.argmax(np.convolve(g, np.ones(length) / length, mode="valid"))) if n > length else 0
    for which, start in (("first", 0), ("middle", max(0, n // 2 - length // 2)), ("steepest", steep)):
      jobs.append({"course": s, "stretch": which, "start": start, "length": length, "ckpt": ckpt, "clip": clip,
                   "name": f"eval_{label}_{which}"})
  rows = []
  for r in evaluate_one.map(jobs):
    frames = r.pop("_frames")
    d = Path("out/hike/runs") / r["course"] / f"eval_{label}_{r['stretch']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames.npz").write_bytes(frames)
    (d / "run.json").write_text(json.dumps({k: v for k, v in r.items() if k != "stretch"}, indent=1))
    rows.append(r)
    print(f"{r['area'][:28]:28s} {r['stretch']:9s} {r['result']:24s} {r['walked_m']:6.1f}/{r['segment_m']} m  "
          f"{r['mean_speed_mps']:4.2f} m/s  steepest {r['segment_max_grade_deg']:4.1f} deg", flush=True)
  fin = sum(r["result"] == "finished" for r in rows)
  steep = sum(r["result"] == "finished" for r in rows if r["stretch"] == "steepest")
  print(f"\n{label}: finished {fin}/{len(rows)}, steepest finished {steep}/{len(courses)}, "
        f"mean {sum(r['walked_m'] for r in rows) / len(rows):.1f} m of {length}")
  out = Path("out/hike/bench")
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{label}.json").write_text(json.dumps(rows, indent=1))


@app.local_entrypoint()
def main(iters: int = 3000, num_envs: int = 4096, resume: str = "", init_std: float = 0.0, start_level: int = 40,
         env: str = "sonic"):
  train.remote(iters=iters, num_envs=num_envs, resume=resume, init_std=init_std, start_level=start_level, env=env)
