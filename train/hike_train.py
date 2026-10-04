"""PPO on the hiking env with rsl_rl (same settings as the ski run, including the capped action std).

Local smoke test: PYTHONPATH=. .venv-hike/bin/python -m train.hike_train --device cpu --num-envs 8 --iters 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from train.hike_env import HikeEnv

ROOT = Path(__file__).resolve().parents[1]


def runner_cfg(iters: int, logger: str, rough: bool = False) -> dict:
  """rsl_rl 5.4 runner config, written out from mjlab's RslRlOnPolicyRunnerCfg (G1 velocity settings, capped action
  std) so this branch runs on mujoco-warp 3.14 without mjlab, whose 1.6 pin (mujoco-warp 3.11) has a heightfield
  contact bug that reports 20+ m penetrations and launches robots."""
  mlp = {"hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True, "class_name": "MLPModel"}
  if rough:  # from scratch, mjlab velocity-task settings: wide exploration, higher entropy and learning rate
    std, entropy, lr = {"init_std": 1.0, "std_range": [0.05, 1.5]}, 0.005, 1e-3
  else:
    std, entropy, lr = {"init_std": 0.5, "std_range": [0.05, 1.0]}, 0.001, 5e-4
  cfg = {
    "seed": 42, "num_steps_per_env": 24, "max_iterations": iters, "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
    "save_interval": 25, "experiment_name": "g1_hike_rough" if rough else "g1_hike", "run_name": "", "logger": logger, "wandb_project": "mjlab",
    "wandb_tags": [], "resume": False, "load_run": ".*", "load_checkpoint": "model_.*.pt", "clip_actions": None,
    "upload_model": False, "class_name": "OnPolicyRunner",
    "actor": {**mlp, "distribution_cfg": {"class_name": "GaussianDistribution", "std_type": "scalar", **std}},
    "critic": dict(mlp),
    "algorithm": {"num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": lr, "schedule": "adaptive",
                  "gamma": 0.99, "lam": 0.95, "entropy_coef": entropy, "desired_kl": 0.01, "max_grad_norm": 1.0,
                  "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                  "normalize_advantage_per_mini_batch": False, "optimizer": "adam", "share_cnn_encoders": False,
                  "class_name": "PPO"},
  }
  return cfg


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--iters", type=int, default=1500)
  ap.add_argument("--log-dir", default=str(ROOT / "runs_hike"))
  ap.add_argument("--logger", default="tensorboard")
  ap.add_argument("--resume", default="")
  ap.add_argument("--terrain", default=str(ROOT / "out/hike/tiles"))
  ap.add_argument("--clips", default=str(ROOT / "out/hike/clips.npz"))
  ap.add_argument("--init-std", type=float, default=0.0)
  ap.add_argument("--env", choices=["sonic", "rough"], default="sonic", help="rough = end-to-end joints, 5 ms physics")
  ap.add_argument("--start-level", type=int, default=40, help="robots start spread over the easiest N tiles (raise on resume)")
  args = ap.parse_args()
  t0 = time.time()
  if args.env == "rough":
    from train.hike_rough_env import HikeRoughEnv
    env = HikeRoughEnv(num_envs=args.num_envs, device=args.device, mosaic=args.terrain, clips=args.clips,
                       start_level=args.start_level, dt=0.005)
  else:
    env = HikeEnv(num_envs=args.num_envs, device=args.device, mosaic=args.terrain, clips=args.clips, start_level=args.start_level)
  print(f"env ready in {time.time() - t0:.1f}s: {args.num_envs} envs, obs {env.obs.shape[1]}, actions {env.num_actions}", flush=True)
  log_dir = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
  runner = OnPolicyRunner(env, runner_cfg(args.iters, args.logger, rough=args.env == "rough"), str(log_dir), device=args.device)
  if args.resume:
    runner.load(args.resume, map_location=args.device)
    if args.init_std > 0:
      dist = runner.alg.actor.distribution
      if hasattr(dist, "std_param"):
        dist.std_param.data.fill_(args.init_std)
      else:
        dist.log_std_param.data.fill_(float(torch.log(torch.tensor(args.init_std))))
  runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
  print(f"done in {time.time() - t0:.0f}s, logs at {log_dir}", flush=True)


if __name__ == "__main__":
  main()
