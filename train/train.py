"""PPO on the skiing env with rsl_rl (settings from mjlab's G1 velocity task).

Local smoke test: .venv-train/bin/python -m train.train --device cpu --num-envs 8 --iters 2
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import torch
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rl.runners import OnPolicyRunner

from train.ski_env import SkiEnv

ROOT = Path(__file__).resolve().parents[1]


def runner_cfg(iters: int, logger: str) -> dict:
  cfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.005, learning_rate=5e-4, num_learning_epochs=5, num_mini_batches=4,
                                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name="g1_ski", num_steps_per_env=24, max_iterations=iters, save_interval=25, logger=logger,
    upload_model=False,
  )
  d = dataclasses.asdict(cfg)
  for key in ("actor", "critic"):  # same cleanup mjlab's runner does before rsl_rl sees the config
    for opt in ("cnn_cfg", "distribution_cfg"):
      if d[key].get(opt) is None:
        d[key].pop(opt, None)
    if d[key].get("rnn_type") is None:
      for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        d[key].pop(opt, None)
  return d


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--iters", type=int, default=1500)
  ap.add_argument("--log-dir", default=str(ROOT / "runs"))
  ap.add_argument("--logger", default="tensorboard")
  ap.add_argument("--resume", default="")
  ap.add_argument("--terrain", default=str(ROOT / "out/terrains"))
  args = ap.parse_args()
  t0 = time.time()
  env = SkiEnv(num_envs=args.num_envs, device=args.device, mosaic=args.terrain)
  print(f"env ready in {time.time() - t0:.1f}s: {args.num_envs} envs, obs {env.obs.shape[1]}, actions {env.num_actions}", flush=True)
  log_dir = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
  runner = OnPolicyRunner(env, runner_cfg(args.iters, args.logger), str(log_dir), device=args.device)
  if args.resume:
    runner.load(args.resume)
  runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
  print(f"done in {time.time() - t0:.0f}s, logs at {log_dir}", flush=True)


if __name__ == "__main__":
  main()
