"""Average the last N iterations of an rsl_rl console log.

Run: python train/logstats.py runs_modal_vN.log [N]
"""

import re
import sys
from collections import defaultdict

KEYS = ["Mean action std", "Curriculum/mean_level", "Episode/fell_frac", "Episode/tree_crash_frac", "Episode/distance_m",
        "Task/speed_mps", "Task/heading_err_deg", "Gates/hits_per_episode", "Gates/hit_rate", "Task/shield_frac", "Iteration time"]


def main():
  log, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
  blocks = open(log, errors="ignore").read().split("Learning iteration ")[1:]
  if not blocks:
    print("no iterations yet")
    return
  it = [int(re.match(r"(\d+)", b).group(1)) for b in blocks]
  tail = blocks[-n:]
  acc = defaultdict(list)
  for b in tail:
    for k, v in re.findall(r"^\s*([A-Za-z][\w/ ]+?):\s+(-?[\d.]+)\s*$", b, re.M):
      acc[k.strip()].append(float(v))
  avg = lambda k: sum(acc[k]) / len(acc[k])
  print(f"iters {it[-len(tail)]}..{it[-1]}: " + " | ".join(f"{k.split('/')[-1]} {avg(k):.3g}" for k in KEYS if acc.get(k)))
  rk = sorted(k for k in acc if k.startswith("Reward/"))
  if rk:
    print("rewards/step: " + " ".join(f"{k[7:]} {avg(k):+.3f}" for k in rk))


if __name__ == "__main__":
  main()
