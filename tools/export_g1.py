"""Unitree G1 for the browser: decimated meshes in one GLB plus the kinematic tree as JSON.

Run: .venv/bin/python tools/export_g1.py
Writes game/public/g1/g1.glb (every visual mesh, named, at the origin) and game/public/g1/g1.json
(bodies with parent, pos, quat wxyz, joint axis/range, and the visual geoms per body).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import fast_simplification

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/unitree_g1/g1.xml"
OUT = ROOT / "game/public/g1"
TARGET_FACES = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

root = ET.parse(XML).getroot()
meshdir = XML.parent / root.find("compiler").get("meshdir", "assets")
mesh_files = {m.get("name"): m.get("file") for m in root.iter("mesh")}
materials = {m.get("name"): [float(v) for v in m.get("rgba").split()] for m in root.iter("material")}


def vec(s, n, default):
  return [float(v) for v in s.split()] if s else default


bodies = []


def walk(b, parent):
  entry = {
    "name": b.get("name"), "parent": parent,
    "pos": vec(b.get("pos"), 3, [0, 0, 0]), "quat": vec(b.get("quat"), 4, [1, 0, 0, 0]),
    "joint": None, "geoms": [],
  }
  j = b.find("joint")
  if j is not None and j.get("type") != "free":
    entry["joint"] = {"name": j.get("name"), "axis": vec(j.get("axis"), 3, [0, 0, 1]), "range": vec(j.get("range"), 2, [-3.14, 3.14])}
  for g in b.findall("geom"):
    if g.get("mesh"):
      entry["geoms"].append({"mesh": g.get("mesh"), "pos": vec(g.get("pos"), 3, [0, 0, 0]), "quat": vec(g.get("quat"), 4, [1, 0, 0, 0]),
                             "rgba": materials.get(g.get("material", "silver"), [0.7, 0.7, 0.7, 1])})
  bodies.append(entry)
  for c in b.findall("body"):
    walk(c, entry["name"])


for b in root.find("worldbody").findall("body"):
  walk(b, None)

scene = trimesh.Scene()
total_in = total_out = 0
for name, file in mesh_files.items():
  m = trimesh.load(meshdir / file, force="mesh")
  total_in += len(m.faces)
  if len(m.faces) > TARGET_FACES:
    reduction = 1 - TARGET_FACES / len(m.faces)
    v, f = fast_simplification.simplify(np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int64), target_reduction=reduction)
    m = trimesh.Trimesh(v, f, process=True)
  total_out += len(m.faces)
  scene.add_geometry(m, node_name=name, geom_name=name)

OUT.mkdir(parents=True, exist_ok=True)
scene.export(OUT / "g1.glb")
(OUT / "g1.json").write_text(json.dumps({"bodies": bodies, "meshes": list(mesh_files)}, indent=0))
print(f"{len(mesh_files)} meshes, {total_in} -> {total_out} faces, {(OUT / 'g1.glb').stat().st_size / 1e6:.2f} MB glb, {len(bodies)} bodies")
