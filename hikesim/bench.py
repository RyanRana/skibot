"""Hiking benchmark: every course, three 100 m stretches (first, middle, steepest), one row per gait setting.

    .venv/bin/python -m hikesim.bench --workers 3
    .venv/bin/python -m hikesim.bench --configs walk1.0 careful12 --length 60

Writes out/hike/bench/<tag>.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from hikesim.course import COURSES, slug

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
  "walk1.0": dict(mode="walk", speed=1.0),
  "walk0.7": dict(mode="walk", speed=0.7),
  "careful12": dict(mode="walk", speed=0.9, careful_above=12.0),
}


def stretches(course: str, length: int) -> list[tuple[str, int]]:
  from hikesim.walk import load, steepest
  _, route = load(course)
  n = len(route["x"])
  return [("first", 0), ("middle", max(0, n // 2 - length // 2)), ("steepest", steepest(route, length))]


def _one(args):
  course, which, start, length, cfg_name = args
  from hikesim.walk import hike
  return {"stretch": which, "config": cfg_name,
          **hike(course, start, length, name=f"bench_{cfg_name}_{which}", log=lambda *_: None, **CONFIGS[cfg_name])}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
  ap.add_argument("--length", type=int, default=100)
  ap.add_argument("--workers", type=int, default=3)
  ap.add_argument("--tag", default="zero_shot")
  a = ap.parse_args()
  jobs = [(slug(c), w, s, a.length, cfg) for c in COURSES for w, s in stretches(slug(c), a.length) for cfg in a.configs]
  rows = []
  with ProcessPoolExecutor(a.workers) as ex:
    for r in ex.map(_one, jobs):
      rows.append(r)
      print(f"{r['area'][:28]:28s} {r['stretch']:9s} {r['config']:10s} {r['result']:18s} {r['walked_m']:6.1f} m "
            f"{r['mean_speed_mps']:4.2f} m/s  max {r['max_grade_deg']:4.1f} deg", flush=True)
  out = ROOT / "out" / "hike" / "bench"
  out.mkdir(parents=True, exist_ok=True)
  (out / f"{a.tag}.json").write_text(json.dumps(rows, indent=1))
  print("\nconfig      finished  mean walked  mean speed")
  for cfg in a.configs:
    rs = [r for r in rows if r["config"] == cfg]
    print(f"{cfg:10s} {sum(r['result'] == 'finished' for r in rs):3d}/{len(rs):<4d} {np.mean([r['walked_m'] for r in rs]):8.1f} m "
          f"{np.mean([r['mean_speed_mps'] for r in rs]):8.2f} m/s")


if __name__ == "__main__":
  main()
