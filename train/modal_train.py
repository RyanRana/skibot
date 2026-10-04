"""Train the G1 ski policy on a Modal GPU. Checkpoints and tensorboard logs land in the Modal volume g1-ski-runs.

Launch (detached):  ~/mhacks-spikes/.venv/bin/modal run --detach train/modal_train.py::main --iters 1500 --num-envs 4096
Pull results:       ~/mhacks-spikes/.venv/bin/modal volume get g1-ski-runs <run>/ runs/
Full rides:         ~/mhacks-spikes/.venv/bin/modal run train/modal_train.py::rides --ckpt /runs/<run>/model_N.pt --courses a,b,c [--render]
                    (train/showcase.py on real courses, one container each, in parallel; results land in out/rides/)
Failure report:     ~/mhacks-spikes/.venv/bin/modal run train/modal_train.py::diag --ckpt /runs/<run>/model_N.pt
                    (train/diagnose.py on a GPU: 512 episodes in a couple of minutes; JSON lands in out/diag/)
"""

from __future__ import annotations

import modal

ROOT = "/root/ski"

image = (
  modal.Image.debian_slim(python_version="3.12")
  .apt_install("git", "libegl1", "libgl1", "libglib2.0-0")
  .pip_install("mjlab==1.6.0", "onnx==1.23.1", "tensorboard")
  .apt_install("ffmpeg", "fonts-liberation")  # showcase videos
  .env({"PYTHONPATH": ROOT, "PYTHONUNBUFFERED": "1", "GIT_PYTHON_REFRESH": "quiet", "MUJOCO_GL": "egl"})
  .add_local_dir("skisim", f"{ROOT}/skisim", ignore=["__pycache__", "*.pyc"])
  .add_local_dir("train", f"{ROOT}/train", ignore=["__pycache__", "*.pyc"])
  .add_local_dir("assets/unitree_g1", f"{ROOT}/assets/unitree_g1")
  .add_local_file("assets/sonic/model_encoder.onnx", f"{ROOT}/assets/sonic/model_encoder.onnx")
  .add_local_file("assets/sonic/model_decoder.onnx", f"{ROOT}/assets/sonic/model_decoder.onnx")
  .add_local_dir("out/terrains", f"{ROOT}/out/terrains")
  .add_local_dir("out/terrains_v2", f"{ROOT}/out/terrains_v2")
  .add_local_dir("out/terrains_v3", f"{ROOT}/out/terrains_v3")
  .add_local_file("data/pose/ski_pose.json", f"{ROOT}/data/pose/ski_pose.json")
  .add_local_file("out/terrains_v3/trees.npy", f"{ROOT}/out/terrains_v3/trees.npy")
  .add_local_dir("out/terrains_v4", f"{ROOT}/out/terrains_v4")
  .add_local_dir("out/courses", f"{ROOT}/out/courses")
  .add_local_dir("out/textures", f"{ROOT}/out/textures")
)
vol = modal.Volume.from_name("g1-ski-runs", create_if_missing=True)
app = modal.App("g1-ski-train", image=image)


@app.function(gpu="H100", timeout=12 * 3600, volumes={"/runs": vol})
def train(iters: int = 100000, num_envs: int = 4096, resume: str = "", terrain: str = "out/terrains", init_std: float = 0.0,
          start_level: int = 0, style: str = "default"):
  import subprocess
  import sys
  import threading
  import time

  def commit_loop():
    while True:
      time.sleep(60)
      try:
        vol.commit()
      except Exception as e:  # keep training even if a commit races
        print("volume commit failed:", e, flush=True)

  threading.Thread(target=commit_loop, daemon=True).start()
  cmd = [sys.executable, "-m", "train.train", "--device", "cuda", "--num-envs", str(num_envs), "--iters", str(iters),
         "--log-dir", "/runs", "--terrain", f"{ROOT}/{terrain}", "--init-std", str(init_std), "--start-level", str(start_level),
         "--style", style]
  if resume:
    cmd += ["--resume", resume]
  subprocess.run(cmd, cwd=ROOT, check=True)
  vol.commit()


@app.function(gpu="L40S", timeout=1800, volumes={"/runs": vol})
def diagnose(ckpt: str, terrain: str, envs: int, episodes: int, max_level: int, spawn_speed: str) -> str:
  import subprocess
  import sys

  cmd = [sys.executable, "-m", "train.diagnose", "--device", "cuda", "--ckpt", ckpt, "--terrain", f"{ROOT}/{terrain}",
         "--envs", str(envs), "--episodes", str(episodes), "--max-level", str(max_level), "--out", "/tmp/diag.json"]
  if spawn_speed:
    cmd += ["--spawn-speed", *spawn_speed.split(",")]
  subprocess.run(cmd, cwd=ROOT, check=True)
  return open("/tmp/diag.json").read()


@app.function(gpu="L4", cpu=4, timeout=3600, volumes={"/runs": vol})
def ride(course: str, ckpt: str, seconds: float, speed: float, render: bool, extra: str) -> dict:
  return _ride(course, ckpt, seconds, speed, render, extra)


@app.function(cpu=2, timeout=3600, volumes={"/runs": vol})
def ride_dry(course: str, ckpt: str, seconds: float, speed: float, render: bool, extra: str) -> dict:
  return _ride(course, ckpt, seconds, speed, False, extra)


def _ride(course: str, ckpt: str, seconds: float, speed: float, render: bool, extra: str) -> dict:
  import subprocess
  import sys
  from pathlib import Path

  out = Path("/tmp/ride")
  cmd = [sys.executable, "-m", "train.showcase", "--course", f"{ROOT}/out/courses/{course}", "--ckpt", ckpt,
         "--seconds", str(seconds), "--speed", str(speed), "--out", str(out)] + ([] if render else ["--no-render"]) + extra.split()
  r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
  if r.returncode != 0:
    return {"error": r.stderr[-3000:]}
  return {p.name: p.read_bytes() for p in out.glob(f"{course}.*")}


@app.local_entrypoint()
def rides(ckpt: str, courses: str, seconds: float = 60.0, speed: float = 8.0, render: bool = False, extra: str = "",
          out: str = "out/rides"):
  import json
  from pathlib import Path

  names = [c.strip() for c in courses.split(",") if c.strip()]
  tag = "_".join(extra.replace("--", "").split())
  dest = Path(out) / f"{Path(ckpt).parent.name}_{Path(ckpt).stem}{'_' + tag if tag else ''}"
  dest.mkdir(parents=True, exist_ok=True)
  fn = ride if render else ride_dry  # dry runs need no GPU
  results = list(fn.starmap([(c, ckpt, seconds, speed, render, extra) for c in names]))
  for name, res in zip(names, results):
    if "error" in res:
      print(f"{name}: FAILED\n{res['error']}")
      continue
    for fn, data in res.items():
      (dest / fn).write_bytes(data)
    j = json.loads(res[f"{name}.json"])
    print(f"{name[:34]:34s} {j['distance_along_course_m']:6.0f} m  crashes {j['crashes']:2d}  tree {j['tree_crashes']}  turns {j['turns']:3d}"
          f"  slope {j.get('course_mean_slope_deg')}" + ("  video saved" if f"{name}.mp4" in res else ""))
  print("saved to", dest)


@app.local_entrypoint()
def diag(ckpt: str, terrain: str = "out/terrains_v3", envs: int = 256, episodes: int = 512, max_level: int = 60,
         spawn_speed: str = "", out: str = ""):
  import json
  from pathlib import Path

  res = diagnose.remote(ckpt=ckpt, terrain=terrain, envs=envs, episodes=episodes, max_level=max_level, spawn_speed=spawn_speed)
  dest = Path(out or f"out/diag/{Path(ckpt).parent.name}_{Path(ckpt).stem}{'_s' + spawn_speed.replace(',', '-') if spawn_speed else ''}.json")
  dest.parent.mkdir(parents=True, exist_ok=True)
  dest.write_text(res)
  print(json.dumps(json.loads(res)["summary"], indent=1))
  print("saved", dest)


@app.local_entrypoint()
def main(iters: int = 100000, num_envs: int = 4096, resume: str = "", terrain: str = "out/terrains", init_std: float = 0.0,
         start_level: int = 0, style: str = "default"):
  train.remote(iters=iters, num_envs=num_envs, resume=resume, terrain=terrain, init_std=init_std, start_level=start_level,
               style=style)
