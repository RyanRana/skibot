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
                        # Capped exploration noise: an uncapped std drifted from 2 to 3.8 in v7 and turned residuals into noise.
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar",
                                          "std_range": (0.05, 1.0)}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.001, learning_rate=5e-4, num_learning_epochs=5, num_mini_batches=4,
                                   gamma=0.995, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),  # ~4 s horizon at 50 Hz
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


def load_policy(env, ckpt: str, device: str = "cpu"):
  """Inference policy from a checkpoint; the env's critic setup is matched to the checkpoint (old or privileged)."""
  ck = torch.load(ckpt, map_location=device, weights_only=False)
  env.privileged_critic = ck["critic_state_dict"]["mlp.0.weight"].shape[1] > ck["actor_state_dict"]["mlp.0.weight"].shape[1]
  runner = OnPolicyRunner(env, runner_cfg(1, "tensorboard"), None, device=device)
  runner.load(ckpt, map_location=device)
  return runner.get_inference_policy(device=device)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--iters", type=int, default=1500)
  ap.add_argument("--log-dir", default=str(ROOT / "runs"))
  ap.add_argument("--logger", default="tensorboard")
  ap.add_argument("--resume", default="")
  ap.add_argument("--terrain", default=str(ROOT / "out/terrains"))
  ap.add_argument("--init-std", type=float, default=0.0, help="reset the policy's action std after loading (0 keeps it)")
  ap.add_argument("--start-level", type=int, default=0, help="start the tile curriculum near this level (on resume)")
  ap.add_argument("--style", default="default", help="'racing' adds a racer's-form reward (see SkiEnv.style)")
  args = ap.parse_args()
  t0 = time.time()
  env = SkiEnv(num_envs=args.num_envs, device=args.device, mosaic=args.terrain)
  env.set_style(args.style)
  print(f"env ready in {time.time() - t0:.1f}s: {args.num_envs} envs, obs {env.obs.shape[1]}, actions {env.num_actions}", flush=True)
  log_dir = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
  runner = OnPolicyRunner(env, runner_cfg(args.iters, args.logger), str(log_dir), device=args.device)
  if args.resume:
    ck = torch.load(args.resume, map_location=args.device, weights_only=False)
    obs0 = env.get_observations()
    dims = {"actor_state_dict": obs0["actor"].shape[1], "critic_state_dict": obs0["critic"].shape[1]}
    old_dims = {p: ck[p]["mlp.0.weight"].shape[1] for p in dims}
    old_dim, new_dim = old_dims["actor_state_dict"], dims["actor_state_dict"]
    if any(old_dims[p] < dims[p] for p in dims):
      # New observations were appended (e.g. the tree scan): widen the first layer with zero weights and pad the
      # normalizers, so the policy starts exactly where the checkpoint was and learns to use the new inputs.
      for part in ("actor_state_dict", "critic_state_dict"):
        sd = ck[part]
        pad = dims[part] - old_dims[part]
        if pad <= 0:
          continue
        for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
          sd[key] = torch.cat([sd[key], torch.full((1, pad), fill, device=sd[key].device)], 1)
        w = sd["mlp.0.weight"]
        sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], pad, device=w.device)], 1)
      runner.alg.actor.load_state_dict(ck["actor_state_dict"])
      runner.alg.critic.load_state_dict(ck["critic_state_dict"])
      runner.current_learning_iteration = ck["iter"]
      print(f"widened checkpoint inputs: actor {old_dims['actor_state_dict']} -> {dims['actor_state_dict']}, critic {old_dims['critic_state_dict']} -> {dims['critic_state_dict']}", flush=True)
    else:
      runner.load(args.resume, map_location=args.device)
    if args.init_std > 0:
      dist = runner.alg.actor.distribution
      if hasattr(dist, "std_param"):
        dist.std_param.data.fill_(args.init_std)
      else:
        dist.log_std_param.data.fill_(float(torch.log(torch.tensor(args.init_std))))
      print(f"action std reset to {args.init_std}", flush=True)
  if args.start_level > 0:
    # Pick the tile curriculum up near where the last run left it instead of climbing back up from tile 0.
    hi = min(args.start_level, env.n_tiles - 1)
    env.level[:] = torch.randint(max(0, hi - 8), hi + 1, (env.num_envs,), generator=env.gen, device=env.device)
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)
    env._forward()
    env._refresh_history(ids, fill=True)
    env._update_gate_command(rate_limit=False)
    env.obs = env._observations()
    print(f"curriculum starts at levels {max(0, hi - 8)}..{hi}", flush=True)
  runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
  print(f"done in {time.time() - t0:.0f}s, logs at {log_dir}", flush=True)


if __name__ == "__main__":
  main()
