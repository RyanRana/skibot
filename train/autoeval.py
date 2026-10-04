"""Progress log: evaluate and render each new checkpoint, parse training curves, publish to the local page.

Run: PYTHONPATH=. nohup .venv-train/bin/python -m train.autoeval &
Reads checkpoints that train.live already syncs into runs_live/, evaluates every --every iterations on the fixed
24-tile benchmark (out/terrains), and writes out/site/progress.json and out/site/curves.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "out" / "site"
METRICS = {
  "Mean reward": "reward",
  "Episode/fell_frac": "fell_frac",
  "Episode/distance_m": "distance_m",
  "Task/heading_err_deg": "heading_err_deg",
  "Task/speed_mps": "speed_mps",
  "Curriculum/mean_level": "level",
}


def checkpoints():
  out = []
  for p in (ROOT / "runs_live").glob("*/model_*.pt"):
    m = re.search(r"model_(\d+)\.pt$", p.name)
    if m:
      out.append((int(m.group(1)), p))
  return sorted(out)


def parse_curves():
  """Training curves from every Modal run log: iteration -> metrics (last value seen per iteration)."""
  curves = {}
  logs = sorted(ROOT.glob("runs_modal_v*.log"), key=lambda p: p.stat().st_mtime)
  for log in logs[-1:]:
    it = None
    for line in log.read_text(errors="ignore").splitlines():
      m = re.search(r"Learning iteration (\d+)/", line)
      if m:
        it = int(m.group(1))
        continue
      if it is None:
        continue
      for key, name in METRICS.items():
        if key in line:
          try:
            val = float(line.split(":")[-1].strip().split()[0])
          except ValueError:
            continue
          curves.setdefault(it, {})[name] = val
  pts = [{"iter": k, **v} for k, v in sorted(curves.items())]
  # Thin to at most ~600 points for the page.
  step = max(1, len(pts) // 600)
  return pts[::step]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--every", type=int, default=150)
  ap.add_argument("--poll", type=int, default=120)
  args = ap.parse_args()
  prog_path = SITE / "progress.json"
  progress = json.loads(prog_path.read_text()) if prog_path.exists() else []
  done = {e["checkpoint"] for e in progress}
  (SITE / "media" / "runs").mkdir(parents=True, exist_ok=True)
  while True:
    try:
      (SITE / "curves.json").write_text(json.dumps(parse_curves()))
      cps = checkpoints()
      def last_for(run):
        return max([e["iteration"] for e in progress if e["run"] == run], default=-10**9)
      todo = [(it, p) for it, p in cps if str(p) not in done and it >= last_for(p.parent.name) + args.every]
      if todo:
        it, p = todo[-1]  # newest first; skip intermediate ones
        name = f"{p.parent.name}_{p.stem}"
        out = ROOT / "out" / "play" / name
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "train.play", "--ckpt", str(p), "--seconds", "12", "--out", str(out)],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if not (out / "eval.json").exists():
          print(f"eval of {p} failed:\n{r.stderr[-1500:]}", flush=True)
          done.add(str(p))
          continue
        media = {}
        for k in ("sheet", "overview"):
          if (out / f"{k}.mp4").exists():
            dst = SITE / "media" / "runs" / f"{name}_{k}.mp4"
            shutil.copy(out / f"{k}.mp4", dst)
            media[k] = f"media/runs/{dst.name}"
        if (out / "sheet_mid.jpg").exists():
          shutil.copy(out / "sheet_mid.jpg", SITE / "media" / "runs" / f"{name}_sheet.jpg")
          media["poster"] = f"media/runs/{name}_sheet.jpg"
        ev = json.loads((out / "eval.json").read_text())
        diag = None
        try:
          subprocess.run([sys.executable, "-m", "train.diagnose", "--ckpt", str(p), "--terrain", str(ROOT / "out/terrains_v3"),
                          "--envs", "32", "--episodes", "64", "--max-level", "40", "--out", str(out / "diag.json")],
                         cwd=ROOT, capture_output=True, text=True, timeout=2400, env={**os.environ, "PYTHONPATH": str(ROOT)})
          dd = json.loads((out / "diag.json").read_text())["summary"]
          diag = {k: dd[k] for k in ("fall_rate", "mean_distance_m", "failure_kinds", "by_slope")}
        except Exception as e:
          print("diagnose failed:", repr(e), flush=True)
        entry = {
          "time": time.strftime("%Y-%m-%d %H:%M"), "run": p.parent.name, "iteration": it, "checkpoint": str(p),
          **{k: v for k, v in ev.items() if k not in ("per_tile", "checkpoint")}, "media": media,
          "eval_seconds": round(time.time() - t0), "diag": diag,
        }
        progress.append(entry)
        done.add(str(p))
        prog_path.write_text(json.dumps(progress, indent=1))
        print(f"[{entry['time']}] iter {it}: {ev['clean_runs']}/{ev['tiles']} clean, {ev['mean_distance_m']} m, "
              f"gap {ev.get('mean_ski_gap_deg')} deg", flush=True)
        if r.returncode:
          print(r.stderr[-800:], flush=True)
    except Exception as e:
      print("autoeval:", repr(e), flush=True)
    time.sleep(args.poll)


if __name__ == "__main__":
  main()
