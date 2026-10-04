"""Exports the Ann Arbor hiking data for the 3D viewer (hikesim/web/index.html) into out/hike/web.

    .venv/bin/python -m hikesim.export_web            # area overview, every built course, every recorded run
    python3 -m http.server 8770 -d out/hike/web        # then open http://127.0.0.1:8770

Area: lidar terrain at 10 m, USGS NAIP aerial photo, every trail with its steepness. Courses: terrain at 2 m, aerial at
0.5 m, the walk. Runs: G1 body poses at 25 Hz for the meshes in web/meshes (Unitree G1, Apache 2.0).
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from hikesim.course import DEM, TRAILS, slug
from hikesim.geo import Dem, fill_nan, utm17
from skisim.scene import G1_DIR, g1_model
from skisim.terrain import slope

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "out" / "hike" / "web"
NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"


def naip(x0: float, y0: float, x1: float, y1: float, px: float, out: Path, epsg: int = 26917) -> bool:
  """USGS NAIP orthoimagery (public domain) for a UTM box (NAD83, `epsg`) at about px metres per pixel, max 4000 px a side."""
  if out.exists():
    return True
  w, h = int(min(4000, (x1 - x0) / px)), int(min(4000, (y1 - y0) / px))
  q = urllib.parse.urlencode(dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=epsg, imageSR=epsg, size=f"{w},{h}",
                                  format="jpg", compressionQuality=85, f="image"))
  for _ in range(3):
    try:
      raw = urllib.request.urlopen(f"{NAIP}?{q}", timeout=240).read()
      if raw[:2] == b"\xff\xd8":
        out.write_bytes(raw)
        return True
    except Exception as e:
      print(f"  NAIP failed ({type(e).__name__}), retrying")
  return False


def block_mean(z: np.ndarray, k: int) -> np.ndarray:
  ny, nx = z.shape[0] // k * k, z.shape[1] // k * k
  return z[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def write_height(z: np.ndarray, path: Path) -> dict:
  path.write_bytes(z.astype(np.float32).tobytes())
  return {"file": path.name, "nx": int(z.shape[1]), "ny": int(z.shape[0]), "zmin": float(z.min()), "zmax": float(z.max())}


def area(trails_path: Path, dem_dir: Path, cell: int = 10) -> dict:
  feats = json.loads(Path(trails_path).read_text())["features"]
  xy = np.array([utm17(la, lo) for f in feats for lo, la in f["geometry"]["coordinates"]])
  x0, y0 = int(xy[:, 0].min() // cell * cell - 200), int(xy[:, 1].min() // cell * cell - 200)
  x1, y1 = int(xy[:, 0].max() // cell * cell + 200), int(xy[:, 1].max() // cell * cell + 200)
  print(f"area {x1 - x0} x {y1 - y0} m at {cell} m")
  z = block_mean(fill_nan(Dem(dem_dir).window(x0, y0, x1 - x0, y1 - y0)), cell)
  d = WEB / "area"
  d.mkdir(parents=True, exist_ok=True)
  meta = {"utm_origin": [x0, y0], "cell": cell, "size": [x1 - x0, y1 - y0], "height": write_height(z, d / "height.f32")}
  if naip(x0, y0, x1, y1, 3.0, d / "aerial.jpg"):
    meta["aerial"] = "aerial.jpg"
  out = []
  for f in feats:
    p = f["properties"]
    pts = np.array([utm17(la, lo) for lo, la in f["geometry"]["coordinates"]]) - (x0, y0)
    st = p.get("statistics") or {}
    out.append({"id": p["id"], "name": p.get("name"), "area": (p.get("area") or {}).get("name"),
                "surface": p.get("surface"), "length_m": st.get("length_m"), "max_grade": st.get("max_grade_deg_10m"),
                "xy": np.round(pts, 1).ravel().tolist()})
  (d / "trails.json").write_text(json.dumps(out))
  meta["trails"] = "trails.json"
  return meta


def course(cdir: Path, cell: int = 2) -> dict:
  route = json.loads((cdir / "route.json").read_text())
  g = json.loads((cdir / "grid.json").read_text())
  z = np.load(cdir / "elevation.npy")
  d = WEB / "courses" / cdir.name
  d.mkdir(parents=True, exist_ok=True)
  zc = block_mean(z, cell)
  ox, oy = g["utm_origin"]
  meta = {"slug": cdir.name, "area": route["area"], "trails": route["trails"], "utm_origin": [ox, oy], "cell": cell,
          "size": [zc.shape[1] * cell, zc.shape[0] * cell], "height": write_height(zc, d / "height.f32"),
          **{k: route[k] for k in ("length_m", "climb_m", "vertical_m", "max_grade_deg", "share_steeper_than_10deg")}}
  if naip(ox, oy, ox + zc.shape[1] * cell, oy + zc.shape[0] * cell, 0.5, d / "aerial.jpg"):
    meta["aerial"] = "aerial.jpg"
  r = np.stack([route["x"], route["y"], route["z"], route["grade_deg"]], 1)[::2]
  meta["route"] = np.round(r, 2).ravel().tolist()
  return meta


def robot() -> tuple[mujoco.MjModel, dict]:
  """Visual mesh geoms of the G1 with their body and local transform, and the STL files copied for the browser."""
  m = g1_model(slope(0.0, 4, 4, 1.0), skis=False)
  files = {e.get("name"): e.get("file") for e in ET.parse(G1_DIR / "g1.xml").getroot().iter("mesh")}
  mats = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i): m.mat_rgba[i].tolist() for i in range(m.nmat)}
  (WEB / "meshes").mkdir(parents=True, exist_ok=True)
  geoms, bodies = [], []
  for gi in range(m.ngeom):
    if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[gi] == 3:
      continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gi])
    f = files.get(name)
    if not f:
      continue
    if not (WEB / "meshes" / f).exists():
      shutil.copy(G1_DIR / "assets" / f, WEB / "meshes" / f)
    b = int(m.geom_bodyid[gi])
    if b not in bodies:
      bodies.append(b)
    rgba = m.mat_rgba[m.geom_matid[gi]].tolist() if m.geom_matid[gi] >= 0 else m.geom_rgba[gi].tolist()
    # MuJoCo recentres meshes on their inertial frame; mesh_pos/quat undo that so the raw STL lines up.
    geoms.append({"mesh": f, "body": bodies.index(b), "pos": m.geom_pos[gi].tolist(), "quat": m.geom_quat[gi].tolist(),
                  "mesh_pos": m.mesh_pos[m.geom_dataid[gi]].tolist(), "mesh_quat": m.mesh_quat[m.geom_dataid[gi]].tolist(),
                  "rgba": rgba})
  return m, {"bodies": bodies, "geoms": geoms, "materials": mats}


def runs(m: mujoco.MjModel, bodies: list[int], course_slug: str) -> list[dict]:
  out = []
  d = mujoco.MjData(m)
  for run in sorted((ROOT / "out" / "hike" / "runs" / course_slug).glob("*/run.json")):
    info = json.loads(run.read_text())
    f = np.load(run.parent / "frames.npz")
    poses = np.zeros((len(f["qpos"]), len(bodies), 7), np.float32)
    for i, q in enumerate(f["qpos"]):
      d.qpos[:] = q
      mujoco.mj_kinematics(m, d)
      poses[i, :, :3] = d.xpos[bodies]
      poses[i, :, 3:] = d.xquat[bodies]
    name = run.parent.name
    dst = WEB / "courses" / course_slug / f"run_{name}.f32"
    dst.write_bytes(poses.tobytes())
    out.append({"name": name, "file": dst.name, "frames": len(poses), "fps": 25, **info})
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--trails", type=Path, default=TRAILS)
  ap.add_argument("--dem", type=Path, default=DEM)
  ap.add_argument("--skip-area", action="store_true")
  a = ap.parse_args()
  WEB.mkdir(parents=True, exist_ok=True)
  shutil.copy(Path(__file__).parent / "web" / "index.html", WEB / "index.html")
  old = json.loads((WEB / "manifest.json").read_text()) if (WEB / "manifest.json").exists() else {}
  manifest = {"area": old.get("area") if a.skip_area and old.get("area") else area(a.trails, a.dem), "courses": []}
  m, rb = robot()
  (WEB / "robot.json").write_text(json.dumps(rb))
  for cdir in sorted((ROOT / "out" / "hike" / "courses").iterdir()):
    if not (cdir / "route.json").exists():
      continue
    meta = course(cdir)
    meta["runs"] = runs(m, rb["bodies"], cdir.name)
    manifest["courses"].append(meta)
    print(f"  {meta['area']}: {len(meta['runs'])} runs")
  (WEB / "manifest.json").write_text(json.dumps(manifest))
  print(f"wrote {WEB}")
  from hikesim import pipeline   # refresh the viewer's Pipeline drawer so it matches what was just exported
  pipeline.main()


if __name__ == "__main__":
  main()
