"""Train the G1 ski policy on a Modal GPU. Checkpoints and tensorboard logs land in the Modal volume g1-ski-runs.

Launch (detached):  ~/mhacks-spikes/.venv/bin/modal run --detach train/modal_train.py --iters 1500 --num-envs 4096
Pull results:       ~/mhacks-spikes/.venv/bin/modal volume get g1-ski-runs <run>/ runs/
"""

from __future__ import annotations

import modal

ROOT = "/root/ski"

image = (
  modal.Image.debian_slim(python_version="3.12")
  .apt_install("git", "libegl1", "libgl1", "libglib2.0-0")
  .pip_install("mjlab==1.6.0", "onnx==1.23.1", "tensorboard")
  .env({"PYTHONPATH": ROOT, "PYTHONUNBUFFERED": "1", "GIT_PYTHON_REFRESH": "quiet", "MUJOCO_GL": "egl"})
  .add_local_dir("skisim", f"{ROOT}/skisim", ignore=["__pycache__", "*.pyc"])
  .add_local_dir("train", f"{ROOT}/train", ignore=["__pycache__", "*.pyc"])
  .add_local_dir("assets/unitree_g1", f"{ROOT}/assets/unitree_g1")
  .add_local_file("assets/sonic/model_encoder.onnx", f"{ROOT}/assets/sonic/model_encoder.onnx")
  .add_local_file("assets/sonic/model_decoder.onnx", f"{ROOT}/assets/sonic/model_decoder.onnx")
  .add_local_dir("out/terrains", f"{ROOT}/out/terrains")
  .add_local_dir("out/terrains_v2", f"{ROOT}/out/terrains_v2")
)
vol = modal.Volume.from_name("g1-ski-runs", create_if_missing=True)
app = modal.App("g1-ski-train", image=image)


@app.function(gpu="H100", timeout=4 * 3600, volumes={"/runs": vol})
def train(iters: int = 100000, num_envs: int = 4096, resume: str = "", terrain: str = "out/terrains"):
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
         "--log-dir", "/runs", "--terrain", f"{ROOT}/{terrain}"]
  if resume:
    cmd += ["--resume", resume]
  subprocess.run(cmd, cwd=ROOT, check=True)
  vol.commit()


@app.local_entrypoint()
def main(iters: int = 100000, num_envs: int = 4096, resume: str = "", terrain: str = "out/terrains"):
  train.remote(iters=iters, num_envs=num_envs, resume=resume, terrain=terrain)
