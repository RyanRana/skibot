"""Athletic ski posture from real race video: MediaPipe 3D pose -> joint angles -> G1 reference poses.

Run: .venv/bin/python -m skisim.pose_from_video data/videos/*.mp4
Writes data/pose/ski_pose.json (per-phase G1 joint targets and the human angle stats behind them),
out/pose/overlays/*.jpg (detected skeletons on the race frames) and out/pose/g1_<phase>.png.

Phases come from where the feet sit relative to the hips in the skier's own pelvis frame:
feet pushed out to the right of the hips means the skier is turning left, and the reverse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parents[1]
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_HE, R_HE, L_FT, R_FT = 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
KEY = [L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
EDGES = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_SH, L_HIP), (R_SH, R_HIP),
         (L_HIP, R_HIP), (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN), (L_AN, L_HE), (L_HE, L_FT),
         (L_AN, L_FT), (R_AN, R_HE), (R_HE, R_FT), (R_AN, R_FT)]


def angle(a, b, c):
  """Angle at b between b->a and b->c."""
  u, v = a - b, c - b
  return float(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))


def unit(v):
  return v / (np.linalg.norm(v) + 1e-9)


def frame_angles(P):
  """Human joint angles from 33 world landmarks (meters)."""
  mid_hip = (P[L_HIP] + P[R_HIP]) / 2
  mid_sh = (P[L_SH] + P[R_SH]) / 2
  up = unit(mid_sh - mid_hip)
  right = P[R_HIP] - P[L_HIP]
  right = unit(right - (right @ up) * up)
  fwd = np.cross(up, right)
  out = {}
  for side, (sh, el, wr, hip, kn, an, he, ft) in (("L", (L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN, L_HE, L_FT)),
                                                   ("R", (R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN, R_HE, R_FT))):
    out[f"knee_{side}"] = np.pi - angle(P[hip], P[kn], P[an])
    out[f"torso_thigh_{side}"] = np.pi - angle(mid_sh, P[hip], P[kn])
    shank, foot = P[kn] - P[an], P[ft] - P[he]
    out[f"dorsi_{side}"] = np.pi / 2 - angle(P[kn], P[an], P[an] + foot)
    out[f"elbow_{side}"] = np.pi - angle(P[sh], P[el], P[wr])
    U = P[el] - P[sh]
    out[f"arm_fwd_{side}"] = float(np.arctan2(U @ fwd, -(U @ up)))
    outward = -right if side == "L" else right
    out[f"arm_abd_{side}"] = float(np.arctan2(U @ outward, -(U @ up)))
    thigh = P[kn] - P[hip]
    out[f"leg_frontal_{side}"] = float(np.arctan2(thigh @ right, -(thigh @ up)))  # + = knee toward the skier's right
  leg_len = np.linalg.norm(P[L_HIP] - P[L_AN]) + 1e-6
  mid_an = (P[L_AN] + P[R_AN]) / 2
  out["feet_right_of_hips"] = float(((mid_an - mid_hip) @ right) / leg_len)
  out["stance_width"] = float(abs((P[L_AN] - P[R_AN]) @ right) / (np.linalg.norm(P[L_HIP] - P[R_HIP]) + 1e-6))
  return out


def to_g1(a: dict) -> dict:
  """Human angles -> Unitree G1 joint targets (radians, G1 sign conventions).

  Feet stay flat under an upright pelvis (hip + knee + ankle = 0), the extra forward torso lean goes to the waist,
  and the frontal leg angle becomes hip roll (both hips negative when the feet are pushed to the skier's right).
  """
  q = {}
  for side, s in (("L", "left"), ("R", "right")):
    knee = float(np.clip(a[f"knee_{side}"], 0.2, 2.0))
    dorsi = float(np.clip(a[f"dorsi_{side}"], -0.1, 0.8))
    q[f"{s}_knee_joint"] = knee
    q[f"{s}_ankle_pitch_joint"] = -dorsi
    q[f"{s}_hip_pitch_joint"] = -(knee - dorsi)
    q[f"{s}_hip_roll_joint"] = float(np.clip(-a[f"leg_frontal_{side}"], -0.45, 0.45))
    q[f"{s}_elbow_joint"] = float(np.clip(a[f"elbow_{side}"], 0.1, 1.9))
    roll = float(np.clip(a[f"arm_abd_{side}"], -0.2, 1.2))
    q[f"{s}_shoulder_roll_joint"] = roll if side == "L" else -roll
    q[f"{s}_shoulder_pitch_joint"] = float(np.clip(-a[f"arm_fwd_{side}"], -1.6, 0.6))  # G1: negative raises the arm forward
  extra = np.mean([a["torso_thigh_L"] - a["knee_L"] + a["dorsi_L"], a["torso_thigh_R"] - a["knee_R"] + a["dorsi_R"]])
  q["waist_pitch_joint"] = float(np.clip(extra, 0.0, 0.5))
  return q


def process(video: Path, every_s=0.2, overlay_dir=None, max_overlays=6):
  landmarker = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
  cap = cv2.VideoCapture(str(video))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  step = max(1, int(round(fps * every_s)))
  frames, k, saved = [], 0, 0
  while True:
    ok, img = cap.read()
    if not ok:
      break
    if k % step:
      k += 1
      continue
    ts = int(k / fps * 1000)
    k += 1
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = landmarker.process(rgb)
    if not res.pose_world_landmarks:
      continue
    lm2d = res.pose_landmarks.landmark
    vis = np.array([lm2d[i].visibility for i in KEY])
    ys = np.array([lm2d[i].y for i in KEY])
    if vis.min() < 0.6 or not 0.18 < (ys.max() - ys.min()) < 0.6:  # every key joint seen; racer framed, not a close-up
      continue
    P = np.array([[p.x, -p.y, -p.z] for p in res.pose_world_landmarks.landmark])  # to a right-handed, y-up frame
    a = frame_angles(P)
    # Skiing, not standing in the start house or finish area: knees and hips clearly flexed.
    if min(a["knee_L"], a["knee_R"]) < np.radians(25) or min(a["torso_thigh_L"], a["torso_thigh_R"]) < np.radians(25):
      continue
    a["t"] = ts / 1000
    a["video"] = video.stem
    frames.append(a)
    if overlay_dir is not None and saved < max_overlays and len(frames) % 12 == 0:
      h, w = img.shape[:2]
      pts = {i: (int(lm2d[i].x * w), int(lm2d[i].y * h)) for i in range(33)}
      for i, j in EDGES:
        cv2.line(img, pts[i], pts[j], (40, 220, 255), 3, cv2.LINE_AA)
      for i in KEY:
        cv2.circle(img, pts[i], 5, (60, 60, 240), -1, cv2.LINE_AA)
      phase = "turning left" if a["feet_right_of_hips"] > 0.12 else "turning right" if a["feet_right_of_hips"] < -0.12 else "transition"
      cv2.putText(img, f"knees {np.degrees(a['knee_L']):.0f}/{np.degrees(a['knee_R']):.0f} deg  {phase}", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
      cv2.imwrite(str(overlay_dir / f"{video.stem}_{saved}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
      saved += 1
  cap.release()
  landmarker.close()
  return frames


def summarize(frames):
  keys = [k for k in frames[0] if k not in ("t", "video")]
  def med(fs):
    return {k: float(np.median([f[k] for f in fs])) for k in keys}
  lean = np.array([f["feet_right_of_hips"] for f in frames])
  phases = {
    "neutral": frames,
    "left_turn": [f for f, l in zip(frames, lean) if l > 0.12],
    "right_turn": [f for f, l in zip(frames, lean) if l < -0.12],
    "transition": [f for f, l in zip(frames, lean) if abs(l) <= 0.12],
  }
  out = {}
  for name, fs in phases.items():
    if len(fs) < 5:
      continue
    h = med(fs)
    out[name] = {"frames": len(fs), "human_deg": {k: round(float(np.degrees(v)), 1) if not k.startswith(("feet", "stance")) else round(v, 3) for k, v in h.items()},
                 "g1": {k: round(v, 3) for k, v in to_g1(h).items()}}
  return out


def main():
  videos = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "data" / "videos").glob("*.mp4"))
  overlay_dir = ROOT / "out" / "pose" / "overlays"
  overlay_dir.mkdir(parents=True, exist_ok=True)
  all_frames, per_video = [], {}
  for v in videos:
    fr = process(v, overlay_dir=overlay_dir)
    per_video[v.stem] = len(fr)
    all_frames += fr
    print(f"{v.name}: {len(fr)} usable racer frames", flush=True)
  (ROOT / "data" / "pose").mkdir(parents=True, exist_ok=True)
  (ROOT / "data" / "pose" / "frames.json").write_text(json.dumps(all_frames))
  summary = summarize(all_frames)
  summary["_meta"] = {"videos": per_video, "total_frames": len(all_frames), "sample_every_s": 0.2}
  out = ROOT / "data" / "pose"
  out.mkdir(parents=True, exist_ok=True)
  (out / "ski_pose.json").write_text(json.dumps(summary, indent=1))
  for name in ("neutral", "left_turn", "right_turn"):
    if name in summary:
      h = summary[name]["human_deg"]
      print(f"{name:10s} n={summary[name]['frames']:4d}  knees {h['knee_L']}/{h['knee_R']}  torso-thigh {h['torso_thigh_L']}/{h['torso_thigh_R']}  "
            f"dorsi {h['dorsi_L']}/{h['dorsi_R']}  elbows {h['elbow_L']}/{h['elbow_R']}  arm fwd {h['arm_fwd_L']}/{h['arm_fwd_R']}  "
            f"stance {h['stance_width']}  feet->right {h['feet_right_of_hips']}")


if __name__ == "__main__":
  main()
