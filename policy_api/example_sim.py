"""Drive skibot's physics with a policy pulled from the API: one robot, zero training on the client.

    PYTHONPATH=. .venv-train/bin/python policy_api/example_sim.py http://127.0.0.1:8791 [tiles...]

Runs 12 s per benchmark tile with the pulled ONNX policy in the loop (no network per step) and prints how far
the G1 got and whether it fell. The skibot repo provides the physics (ski-snow contact, SONIC underneath).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.append(str(next((HERE / ".venv/lib").glob("python3.*/site-packages"))))  # onnxruntime for pull()
sys.path.insert(0, str(HERE))
from hi_client import pull  # noqa: E402
from train.ski_env import ROOT, SkiEnv  # noqa: E402

host = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8791"
tiles = [int(t) for t in sys.argv[2:]] or [2, 10, 18]
policy = pull(host, "g1-ski-trees")
env = SkiEnv(num_envs=1, device="cpu", use_graph=False, seed=11, mosaic=ROOT / "out/terrains", backend="mujoco")
assert env.obs.shape[1] == policy.meta["obs_dim"], (env.obs.shape, policy.meta["obs_dim"])

ids = torch.tensor([0])
for tile in tiles:
    env.level[0] = tile
    env.reset_idx(ids)
    env.tile[0] = tile
    env._forward()
    env._refresh_history(ids, fill=True)
    env.obs = env._observations()
    x0 = float(env.qpos[0, 0])
    fell = None
    for step in range(600):  # 12 s at 50 Hz
        obs = env.get_observations()["actor"][0].numpy()
        act = policy.infer(obs)
        _, _, done, extras = env.step(torch.from_numpy(act)[None])
        if bool(done[0]):
            fell = None if bool(extras["time_outs"][0]) else step * 0.02
            break
    dist = float(env.qpos[0, 0]) - x0
    print(f"tile {tile}: {dist:.1f} m downhill in {(step + 1) * 0.02:.1f} s, "
          + (f"fell at {fell:.1f} s" if fell is not None else "no fall"))
