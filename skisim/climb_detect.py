"""MediaPipe pose over climbing videos, cached per video: image landmarks, world landmarks, visibility and the camera's
frame-to-frame shift (phase correlation on the whole frame, which is mostly wall), so limb positions can be put back in
wall coordinates when the camera pans up after the climber.

Run: .venv/bin/python -m skisim.climb_detect data/videos/climb/*.mp4 data/videos/IMG_5878.mp4 [--fps 15]
Writes out/climb/_detect_<video>.npz
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "climb"


def detect(video: str, fps_out: float = 15.0, complexity: int = 1) -> str:
  import mediapipe as mp

  video = Path(video)
  cache = OUT / f"_detect_{video.stem}.npz"
  if cache.exists():
    return f"{video.stem}: cached"
  cap = cv2.VideoCapture(str(video))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  step = max(1, round(fps / fps_out))
  pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=complexity, smooth_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
  uv, P, vis, shift, idx = [], [], [], [], []
  prev, k = None, -1
  while True:
    ok, img = cap.read()
    if not ok:
      break
    k += 1
    if k % step:
      continue
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (W // 4, H // 4)).astype(np.float32)
    if prev is None:
      shift.append((0.0, 0.0, 1.0))
    else:
      (dx, dy), resp = cv2.phaseCorrelate(prev, g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F))
      shift.append((4 * dx / W, 4 * dy / H, resp))  # in units of frame width / height
    prev = g
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if res.pose_world_landmarks:
      uv.append([[p.x, p.y] for p in res.pose_landmarks.landmark])
      vis.append([p.visibility for p in res.pose_landmarks.landmark])
      P.append([[p.x, -p.y, -p.z] for p in res.pose_world_landmarks.landmark])
    else:
      uv.append(np.full((33, 2), np.nan))
      vis.append(np.zeros(33))
      P.append(np.full((33, 3), np.nan))
    idx.append(k)
  cap.release()
  pose.close()
  OUT.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(cache, uv=np.array(uv), P=np.array(P), vis=np.array(vis), shift=np.array(shift),
                      frame=np.array(idx), fps=np.array(fps), step=np.array(step), size=np.array([W, H]))
  found = int((np.array(vis)[:, 11:13].min(1) > 0.5).sum())
  return f"{video.stem}: {len(uv)} frames, climber found in {found}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("videos", nargs="+")
  ap.add_argument("--fps", type=float, default=15.0)
  ap.add_argument("--workers", type=int, default=6)
  args = ap.parse_args()
  with ProcessPoolExecutor(args.workers) as ex:
    for msg in ex.map(detect, args.videos, [args.fps] * len(args.videos)):
      print(msg, flush=True)


if __name__ == "__main__":
  main()
