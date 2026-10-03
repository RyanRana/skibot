"""Export skibot checkpoints into what the Hazard Intelligence policy server serves.

For each policy: weights.npz (normalizer + MLP, numpy inference), policy.onnx (the same network for local runtime),
and meta.json (observation layout, action layout, control rate, provenance). Run with skibot's training venv:

    PYTHONPATH=. .venv-train/bin/python policy_api/export_policies.py      # from the repo root
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from skisim.sonic import I2M, JOINTS
from train.ski_env import HEIGHT_X, HEIGHT_Y, N_RAYS, RAY_FOV_DEG

SKI = Path(__file__).resolve().parents[1]  # repo root; checkpoints live in runs_live/ (not committed)
OUT = Path(__file__).parent / "policies"
EPS = 1e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)

POLICIES = [
    {
        "id": "g1-ski",
        "name": "G1 ski, best benchmark",
        "checkpoint": "runs_live/20261003-182406/model_1900.pt",
        "benchmark": {"clean_runs": 19, "of": 24, "seconds_per_run": 12, "baseline_sonic_only": 1},
    },
    {
        "id": "g1-ski-trees",
        "name": "G1 ski with tree avoidance, newest",
        "checkpoint": "runs_live/20261003-201557/model_5475.pt",
        "benchmark": None,
    },
]

ISAAC_JOINTS = [None] * 29
for mj, isaac in enumerate(I2M):
    ISAAC_JOINTS[isaac] = JOINTS[mj]


def obs_layout(dim: int) -> list[dict]:
    parts = [
        ("base_ang_vel", 3, "body-frame angular velocity, rad/s x 0.25"),
        ("gravity", 3, "gravity direction in the body frame"),
        ("base_lin_vel", 3, "body-frame linear velocity, m/s x 0.2"),
        ("joint_pos", 29, "joint angles, rad, MuJoCo order"),
        ("joint_vel", 29, "joint velocities, rad/s x 0.05, MuJoCo order"),
        ("last_action", 31, "previous action from this policy"),
        ("ski_edge", 2, "edge angle per ski, rad"),
        ("ski_normal_force", 2, "snow normal force per ski, N / 200"),
        ("ski_v_along", 2, "velocity along each ski, m/s x 0.1"),
        ("ski_v_lateral", 2, "sideways velocity per ski, m/s"),
        ("ski_zones", 8, "pressure under front/rear of inner/outer edge per ski, / 175"),
        ("height_scan", len(HEIGHT_X) * len(HEIGHT_Y),
         f"terrain height minus pelvis height, clamped to [-3, 1] m, at body-frame x in {list(HEIGHT_X)} m by y in {list(HEIGHT_Y)} m (x major)"),
        ("command", 3, "sin and cos of heading error, target speed m/s / 10"),
    ]
    if dim == 156:
        parts.append(("tree_rays", N_RAYS, f"{N_RAYS} rays over {RAY_FOV_DEG:.0f} deg, free distance fraction"))
    out, i = [], 0
    for name, n, desc in parts:
        out.append({"name": name, "start": i, "size": n, "desc": desc})
        i += n
    assert i == dim, (i, dim)
    return out


class Actor(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].float())
        self.register_buffer("std", sd["obs_normalizer._std"].float())
        dims = [sd["mlp.0.weight"].shape[1], 512, 256, 128, 31]
        layers = []
        for k, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            lin = torch.nn.Linear(a, b)
            lin.weight.data, lin.bias.data = sd[f"mlp.{2 * k}.weight"].float(), sd[f"mlp.{2 * k}.bias"].float()
            layers += [lin, torch.nn.ELU()] if k < 3 else [lin]
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp((obs - self.mean) / (self.std + EPS))


def main():
    index = []
    for p in POLICIES:
        ck = torch.load(SKI / p["checkpoint"], map_location="cpu", weights_only=False)
        sd = ck["actor_state_dict"]
        actor = Actor(sd).eval()
        dim = actor.mean.shape[1]
        d = OUT / p["id"]
        d.mkdir(parents=True, exist_ok=True)
        lin = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
        np.savez(d / "weights.npz", mean=actor.mean.numpy()[0], std=actor.std.numpy()[0], eps=np.float32(EPS),
                 **{f"W{k}": l.weight.detach().numpy() for k, l in enumerate(lin)},
                 **{f"b{k}": l.bias.detach().numpy() for k, l in enumerate(lin)})
        torch.onnx.export(actor, torch.zeros(1, dim), d / "policy.onnx", input_names=["obs"], output_names=["actions"],
                          dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17, dynamo=False)
        meta = {
            "id": p["id"], "name": p["name"], "robot": "unitree_g1_29dof", "skill": "ski",
            "obs_dim": dim, "action_dim": 31, "control_hz": 50, "physics_dt": 0.002, "decimation": 10,
            "architecture": {
                "type": "mlp", "layers": [dim, 512, 256, 128, 31], "activation": "elu",
                "normalizer": "obs_n = (obs - mean) / (std + 0.01), mean and std in weights.npz",
                "output": "deterministic mean action, clipped to [-3, 3]",
                "params": int(sum(v.numel() for k, v in sd.items() if k.startswith("mlp."))),
                "files": {"onnx": "policy.onnx (input obs [B, obs_dim], output actions [B, 31])",
                          "weights": "weights.npz: mean, std, eps, W0..W3 [out, in], b0..b3"},
            },
            "observation": obs_layout(dim),
            "action": {
                "clip": [-3.0, 3.0],
                "layout": [
                    {"name": "lean", "start": 0, "size": 2, "scale": 0.35,
                     "desc": "body lean target (roll, pitch) handed to SONIC as its command, rad after scale"},
                    {"name": "joint_residual", "start": 2, "size": 29, "scale": 0.5,
                     "desc": "offsets added to SONIC's joint targets, rad after scale, SONIC (IsaacLab) joint order",
                     "joints": ISAAC_JOINTS},
                ],
                "downstream": "joint_target = default_angle + (sonic(lean) + residual) * action_scale, then PD at 500 Hz",
            },
            "controller": "NVIDIA GEAR-SONIC encoder/decoder, frozen (NVIDIA Open Model License); not included here",
            "benchmark": p["benchmark"],
            "provenance": {"checkpoint": p["checkpoint"], "iteration": int(ck.get("iter", -1)), "trainer": "rsl_rl PPO"},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        index.append({k: meta[k] for k in ("id", "name", "robot", "skill", "obs_dim", "action_dim", "control_hz", "benchmark")})
        print(p["id"], "obs", dim, "iter", meta["provenance"]["iteration"])
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
