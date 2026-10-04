"""Record real observations and rsl_rl's own actions from a CPU SkiEnv rollout, for checking the server's numpy
inference and the exported ONNX against the trainer.

    PYTHONPATH=. .venv-train/bin/python policy_api/record_parity.py      # from the repo root
"""
from pathlib import Path

import numpy as np
import torch

from train.ski_env import ROOT, SkiEnv
from train.train import load_policy

CKPT = ROOT / "runs_live/20261003-201557/model_5475.pt"  # g1-ski-trees, 156 obs (the env's current layout)

env = SkiEnv(num_envs=1, device="cpu", use_graph=False, seed=3, mosaic=ROOT / "out/terrains", backend="mujoco")
policy = load_policy(env, str(CKPT))
obs_log, act_log = [], []
for _ in range(300):
    obs = env.get_observations()
    with torch.no_grad():
        act = policy(obs)
    obs_log.append(obs["actor"][0].numpy().copy())
    act_log.append(act[0].numpy().copy())
    env.step(act)
out = Path(__file__).parent / "parity.npz"
np.savez(out, obs=np.array(obs_log), actions=np.array(act_log))
print("saved", out, np.array(obs_log).shape)
