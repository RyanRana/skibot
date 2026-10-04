"""Side by side: a climbing video with its 3D pose skeleton on the left, the Unitree G1 climbing the same route on a
rock wall on the right, live joint values and recordings below.

The robot gets what it could know on a wall: where its pelvis and shoulders are and which hold each hand and foot is
on (holds are rebuilt from where the climber's limbs stopped). The rest of its body (elbows, knees, hip turn, torso)
comes from the climbing prior trained on other climbing videos (`skisim.climb`), which never saw this one.

Run: .venv/bin/python -m skisim.climb_video IMG_5878 --title "RYAN · NIGHT TOWER · MEDIAPIPE 3D POSE"
Writes out/climb/climb_<video>.mp4 and .json (per-frame recordings).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from skisim.climb_grip import KINDS, Hand
from skisim.climb import (G1_HANDS_XML, CONTACT_X, L_AN, L_EL, L_HIP, L_SH, L_WR, LIMB_LM, LIMBS, OUT, PELVIS_X, R_AN, R_EL, R_HIP,
                          R_SH, R_WR, ROOT, ClimbPrior, Retarget, holds, load, relative_features, targets_from,
                          wall_points, wall_scene)
from skisim.pose_from_video import EDGES, KEY
from skisim.pose_video import ACC, BG, LEFT_COL, LEFT_SET, PANEL, RIGHT_COL, RIGHT_SET, RULE, font, plot

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
CW, CH, TOP = 1920, 1080, 52
PH = CH - TOP - 12
PW = int(PH * 9 / 16) // 2 * 2
PLOT_S = 6.0
INSET = 300
NAMES = {"left_hand": "L hand", "right_hand": "R hand", "left_foot": "L foot", "right_foot": "R foot"}


def phase_of(on: dict, moving: dict) -> str:
  hands = [k for k in ("left_hand", "right_hand") if moving[k]]
  feet = [k for k in ("left_foot", "right_foot") if moving[k]]
  if not any(on.values()):
    return "LOWERING OFF"
  if hands and not feet:
    return f"REACHING {'BOTH HANDS' if len(hands) == 2 else NAMES[hands[0]].upper()}"
  if feet and not hands:
    return f"STEPPING {'BOTH FEET' if len(feet) == 2 else NAMES[feet[0]].upper()}"
  if hands and feet:
    return "MOVING"
  return "ALL FOUR ON" if all(on.values()) else "RESTING"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("stem")
  ap.add_argument("--video", default=None)
  ap.add_argument("--title", default="CLIMBER · MEDIAPIPE 3D POSE")
  ap.add_argument("--frames", type=int, default=0)
  ap.add_argument("--still", type=int, default=-1, help="render one frame to out/climb/still.jpg and stop")
  args = ap.parse_args()
  video = Path(args.video or ROOT / "data" / "videos" / f"{args.stem}.mp4")
  det = load(args.stem)
  r0 = Retarget()
  t, dt, ok, yz = wall_points(det, r0.size["torso"], limbs=r0.size)
  H, ids = holds(t, yz, ok)
  # Wall coordinates: the lowest foot a little off the ground, the route centred on the wall.
  feet_z = np.concatenate([yz[:, L_AN, 1], yz[:, R_AN, 1]])
  off = np.array([-np.median((yz[:, L_HIP, 0] + yz[:, R_HIP, 0]) / 2), 0.25 - np.percentile(feet_z, 1)])
  yz = yz + off
  H = H + off
  n = len(t) if not args.frames else min(args.frames, len(t))
  height = float(max(yz[:, :, 1].max() + 3, 8.0))
  # Hold types: the video shows where the hands stopped, not the hold's shape, so each route hold gets a type,
  # weighted the way a gym route is set (mostly jugs, some crimps and slopers). Feet stand on whatever is there.
  hrng = np.random.default_rng(7)
  kinds = list(hrng.choice(KINDS, size=len(H), p=[0.5, 0.3, 0.2]))
  # The hold's grip edge sits just above the wrist landmark (the fingers are over the edge, the wrist below it).
  hand_holds = set(int(h) for k in ("left_hand", "right_hand") for h in np.unique(ids[k]) if h >= 0)
  E = H.copy()
  for h in hand_holds:
    E[h, 1] += 0.07
  model = wall_scene(E, width=8.0, height=height, kinds=kinds, xml=G1_HANDS_XML)
  data = mujoco.MjData(model)
  r = Retarget()
  body_map = [(model.joint(r.m.joint(j).name).qposadr[0], r.m.joint(j).qposadr[0]) for j in range(1, r.m.njnt)]
  hands = {k: Hand(model, k.split("_")[0]) for k in ("left_hand", "right_hand")}
  route_geoms = [[g for g in range(model.ngeom) if model.geom(g).name.startswith(f"route{h}_")] for h in range(len(H))]
  prior = ClimbPrior()
  print(f"{len(t)} frames, {len(H)} holds rebuilt, climb {yz[:, L_HIP, 1].max() - yz[0, L_HIP, 1]:.1f} m (G1 scale)", flush=True)

  rend = mujoco.Renderer(model, PH, PW)
  crend = mujoco.Renderer(model, INSET, INSET)
  ccam = mujoco.MjvCamera()
  ccam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  opt = mujoco.MjvOption()
  frames_dir = OUT / f"_frames_{args.stem}"
  if args.still < 0:
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
  f_head, f_big, f_mid, f_lab, f_title = font(26, True), font(40, True), font(30, True), font(22), font(24, True)
  n_hist = int(PLOT_S / dt)
  hist = {k: deque([None] * n_hist, maxlen=n_hist) for k in ("eL", "eR", "kL", "kR", "h", "gyro")}
  W, Hh = det["size"]
  cap = cv2.VideoCapture(str(video))
  frame_ids = det["frame"]
  q = None
  cam_z = None
  prev_R = None
  base_z = None
  rec = []
  jidx = lambda nm: model.joint(nm).qposadr[0]
  rng = range(n) if args.still < 0 else [args.still]
  k_frame = -1
  for i in rng:
    target = frame_ids[i]
    if args.still >= 0:
      cap.set(cv2.CAP_PROP_POS_FRAMES, target)
      k_frame = target - 1
    while k_frame < target:
      good, img = cap.read()
      k_frame += 1
    y = yz[i].copy()
    on = {k: ids[k][i] >= 0 for k in LIMBS}
    for k in LIMBS:  # a limb on a hold sits exactly on it
      if on[k]:
        y[LIMB_LM[k]] = H[ids[k][i]]
    q_ref = prior(relative_features(y[None]))[0]
    pel = (y[L_HIP] + y[R_HIP]) / 2
    if q is None:
      q = r.home((PELVIS_X, pel[0], pel[1]))
      q[7:] = q_ref
    cx = {k: CONTACT_X[k] if on[k] else CONTACT_X[k] - 0.08 for k in LIMBS}  # a free limb swings off the wall
    T = targets_from(y, cx, full=False)
    for k in ("left_hand", "right_hand"):  # palm to the wall under the edge, finger bases just over it
      if on[k]:
        ey, ez = E[ids[k][i]]
        side = k.split("_")[0]
        # spaced exactly as on the Dex3-1 (finger base 0.12 m along the wrist, palm 0.06 m off it), weighted high
        # so the hand turns palm to the wall even when the arm has to give a little elsewhere
        T[k] = (np.array([-0.085, ey, ez - 0.108]), np.array([2.0, 3.0, 3.0]))
        T[f"{side}_fingers"] = (np.array([-0.085, ey, ez + 0.012]), np.array([6.0, 6.0, 6.0]))
        T[f"{side}_palm"] = (np.array([-0.025, ey, ez - 0.078]), np.array([6.0, 6.0, 6.0]))
    q = r.solve(T, q, q_ref=q_ref, iters=40 if i == rng[0] else 12, reg=0.08)
    data.qpos[:7] = q[:7]
    for a_, b_ in body_map:
      data.qpos[a_] = q[b_]
    grips = {}
    for k, hd in hands.items():
      if on[k]:
        h = int(ids[k][i])
        grips[k] = {"hold": h, "kind": kinds[h], **hd.grip(data, kinds[h], route_geoms[h])}
      else:
        hd.open(data)
    mujoco.mj_forward(model, data)
    torso = data.body("torso_link")
    R = torso.xmat.reshape(3, 3).copy()
    if prev_R is None:
      gyro = 0.0
    else:
      dR = prev_R.T @ R
      gyro = float(np.degrees(np.arctan2(dR[1, 0] - dR[0, 1], dR[0, 0] + dR[1, 1]) / dt))  # about the torso's up axis
    prev_R = R
    p_z = float(data.qpos[2])
    base_z = p_z if base_z is None else base_z
    vals = {"eL": np.degrees(data.qpos[jidx("left_elbow_joint")]), "eR": np.degrees(data.qpos[jidx("right_elbow_joint")]),
            "kL": np.degrees(data.qpos[jidx("left_knee_joint")]), "kR": np.degrees(data.qpos[jidx("right_knee_joint")]),
            "h": p_z - base_z, "gyro": gyro}
    moving = {k: not on[k] for k in LIMBS}
    phase = phase_of(on, moving)
    if args.still < 0:
      for k in hist:
        hist[k].append(vals[k])
    rec.append({"t": round(float(t[i]), 3), "phase": phase, "on_hold": {k: int(ids[k][i]) for k in LIMBS}, "grip": grips,
                **{k: round(float(v), 2) for k, v in vals.items()},
                "g1_qpos": [round(float(v), 4) for v in data.qpos]})

    # Robot view: behind and a little below, following the pelvis up the wall with some lag, like the phone did.
    cam_z = p_z if cam_z is None else cam_z + (p_z - cam_z) * min(1.0, dt / 0.6)
    cam.lookat[:] = [0.0, data.qpos[1] * 0.6, cam_z + 0.15]
    cam.distance, cam.azimuth, cam.elevation = 3.2, 30.0, -6.0
    rend.update_scene(data, cam, opt)
    g1_img = Image.fromarray(rend.render())
    gk = next((k for k in ("left_hand", "right_hand") if k in grips), None)
    if gk is not None:  # close-up of the gripping Dex3-1 hand, from below and to the side, so the fingers show on the hold
      hc = data.body(f"{gk.split('_')[0]}_wrist_yaw_link").xpos
      ey, ez = E[grips[gk]["hold"]]
      ccam.lookat[:] = [-0.05, ey, ez - 0.03]
      ccam.distance, ccam.azimuth, ccam.elevation = 0.38, (60.0 if gk == "left_hand" else -60.0), 20.0
      crend.update_scene(data, ccam, opt)
      inset = Image.fromarray(crend.render())
      di = ImageDraw.Draw(inset)
      di.rectangle([0, 0, INSET - 1, INSET - 1], outline=(255, 225, 110), width=3)
      di.text((10, 8), f"Dex3-1 · {grips[gk]['kind'].upper()}", font=f_lab, fill=(255, 255, 255))
      g1_img.paste(inset, (10, PH - INSET - 10))

    frame = Image.fromarray(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (PW, PH)))
    if ok[i]:
      dr_f = ImageDraw.Draw(frame)
      pts = {j: (det["uv"][i][j][0] * PW, det["uv"][i][j][1] * PH) for j in range(33)}
      for a_, b_ in EDGES:
        col = LEFT_COL if a_ in LEFT_SET and b_ in LEFT_SET else RIGHT_COL if a_ in RIGHT_SET and b_ in RIGHT_SET else (235, 235, 235)
        dr_f.line([pts[a_], pts[b_]], fill=col, width=3)
      for j in KEY + [L_WR, R_WR, L_EL, R_EL]:
        x, y_ = pts[j]
        dr_f.ellipse([x - 4, y_ - 4, x + 4, y_ + 4], fill=(255, 255, 255))
      for k in LIMBS:  # a ring on each limb that is on a hold
        if on[k]:
          x, y_ = pts[LIMB_LM[k]]
          dr_f.ellipse([x - 11, y_ - 11, x + 11, y_ + 11], outline=(120, 255, 140), width=3)

    canvas = Image.new("RGB", (CW, CH), BG)
    canvas.paste(frame, (12, TOP))
    canvas.paste(g1_img, (12 + PW + 8, TOP))
    dr = ImageDraw.Draw(canvas)
    dr.text((20, 13), args.title, font=f_head, fill=(255, 255, 255))
    dr.text((12 + PW + 8 + 8, 13), "UNITREE G1, SAME ROUTE", font=f_head, fill=(255, 255, 255))
    x0 = 12 + 2 * PW + 8 + 14
    bw = CW - x0 - 12
    box_h = 300
    dr.rectangle([x0, TOP, x0 + bw, TOP + box_h], fill=PANEL, outline=RULE)
    dr.text((x0 + 18, TOP + 12), phase, font=f_big, fill=ACC)
    dr.text((x0 + 18, TOP + 66), "On holds", font=f_lab, fill=(150, 165, 190))
    for c, k in enumerate(LIMBS):
      xx = x0 + 18 + c * 118
      col = (120, 255, 140) if on[k] else (90, 100, 120)
      dr.rounded_rectangle([xx, TOP + 94, xx + 108, TOP + 128], 8, fill=PANEL, outline=col, width=2)
      dr.text((xx + 12, TOP + 99), NAMES[k], font=f_lab, fill=col)
    for c, k in enumerate(("left_hand", "right_hand")):
      g = grips.get(k)
      txt = f"{NAMES[k]}: open" if g is None else (f"{NAMES[k]}: {g['kind'].upper()} · "
            f"{sum(g[f] for f in ('thumb', 'index', 'middle'))}/3 fingers on")
      dr.text((x0 + 18 + c * (bw // 2), TOP + 136), txt, font=f_lab, fill=(255, 225, 110) if g else (120, 135, 160))
    rows = [("Elbows  L / R", f"{vals['eL']:.0f}° / {vals['eR']:.0f}°"), ("Knees  L / R", f"{vals['kL']:.0f}° / {vals['kR']:.0f}°"),
            ("Height climbed", f"{vals['h']:.1f} m"), ("Torso gyro (virtual IMU)", f"{vals['gyro']:+.0f} °/s")]
    for c, (lab, val) in enumerate(rows):
      xx, yy = x0 + 18 + (c % 2) * (bw // 2), TOP + 168 + (c // 2) * 62
      dr.text((xx, yy), lab, font=f_lab, fill=(150, 165, 190))
      dr.text((xx, yy + 24), val, font=f_mid, fill=(255, 255, 255))
    ph = (CH - 12 - (TOP + box_h + 12) - 24) // 3
    y1 = TOP + box_h + 12
    plot(dr, (x0, y1, x0 + bw, y1 + ph), [list(hist["eL"]), list(hist["eR"])], [LEFT_COL, RIGHT_COL],
         ["left elbow", "right elbow"], 0, 150, "Elbow flexion, last 6 s", "°", f_lab, f_title)
    y2 = y1 + ph + 12
    plot(dr, (x0, y2, x0 + bw, y2 + ph), [list(hist["kL"]), list(hist["kR"])], [LEFT_COL, RIGHT_COL],
         ["left knee", "right knee"], 0, 150, "Knee flexion, last 6 s", "°", f_lab, f_title)
    y3 = y2 + ph + 12
    plot(dr, (x0, y3, x0 + bw, CH - 12), [[None if v is None else v * 20 for v in hist["h"]], list(hist["gyro"])],
         [(180, 255, 160), (255, 225, 110)], ["height x20", "torso gyro °/s"], -120, 120,
         "Height and virtual IMU, last 6 s", "", f_lab, f_title)
    if args.still >= 0:
      canvas.save(OUT / "still.jpg", quality=90)
      print(OUT / "still.jpg")
      return
    canvas.save(frames_dir / f"{i:05d}.jpg", quality=88)
    if i % 200 == 0:
      print(f"  render {i}/{n}", flush=True)
  cap.release()
  mp4 = OUT / f"climb_{args.stem}.mp4"
  fps = float(det["fps"]) / int(det["step"])
  subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", str(frames_dir / "%05d.jpg"),
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(mp4)], check=True)
  (OUT / f"climb_{args.stem}.json").write_text(json.dumps({"holds": H.round(3).tolist(), "frames": rec}))
  print(mp4, flush=True)


if __name__ == "__main__":
  main()
