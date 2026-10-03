"""Builds MuJoCo scenes on a HeightGrid: a rigid test sled, or Unitree's G1 (unitree_rl_mjlab model) on skis."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from skisim.ski import SkiParams
from skisim.terrain import HeightGrid

ROOT = Path(__file__).resolve().parents[1]
G1_DIR = ROOT / "assets" / "unitree_g1"


def _reflected_inertia(rotor, gears):
  # Two-stage planetary, as in mjlab.utils.actuator.
  return rotor[0] * (gears[1] * gears[2]) ** 2 + rotor[1] * gears[2] ** 2 + rotor[2]


# Motor groups and gains from unitree_rl_mjlab g1_constants.py: kp = J w^2, kd = 2 zeta J w, 10 Hz, zeta 2.
_J = {
  "5020": _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16)),
  "7520_14": _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)),
  "7520_22": _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)),
  "4010": _reflected_inertia((0.068e-4, 0.0, 0.0), (1, 5, 5)),
}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_W = 10 * 2 * np.pi
_ZETA = 2.0


def g1_motor(joint: str) -> tuple[float, float, float, float]:
  """(kp, kd, effort limit, armature) for a G1 joint."""
  if joint.endswith(("hip_pitch_joint", "hip_yaw_joint")) or joint == "waist_yaw_joint":
    g, mult = "7520_14", 1
  elif joint.endswith(("hip_roll_joint", "knee_joint")):
    g, mult = "7520_22", 1
  elif joint.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
    g, mult = "4010", 1
  elif joint in ("waist_pitch_joint", "waist_roll_joint") or "ankle" in joint:
    g, mult = "5020", 2  # 4-bar linkage driven by two 5020 motors
  else:
    g, mult = "5020", 1
  J = _J[g] * mult
  return J * _W**2, 2 * _ZETA * J * _W, _EFFORT[g] * mult, J


# Athletic ski stance: shins forward, hips over the feet, chest forward, hands forward and out.
SKI_STANCE = {
  "hip_pitch_joint": -0.55,
  "knee_joint": 1.0,
  "ankle_pitch_joint": -0.45,
  "waist_pitch_joint": 0.25,
  "shoulder_pitch_joint": -0.25,
  "left_shoulder_roll_joint": 0.3,
  "right_shoulder_roll_joint": -0.3,
  "elbow_joint": 0.9,
}


def stance_qpos(model: mujoco.MjModel, stance: dict[str, float] = SKI_STANCE) -> np.ndarray:
  q = np.zeros(model.nq)
  for j in range(model.njnt):
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for key, val in stance.items():
      if name == key or (not key.startswith(("left_", "right_")) and name.endswith(key)):
        q[model.jnt_qposadr[j]] = val
  return q


def _world(grid: HeightGrid, timestep: float, extras: list[str]) -> str:
  hf = grid.hfield()
  rx, ry, zr, base = hf["size"]
  px, py, pz = hf["pos"]
  return f"""
  <option timestep="{timestep}" integrator="implicitfast"/>
  <statistic extent="5" center="0 0 0"/>
  <visual>
    <map znear="0.01" zfar="2000" haze="0.15"/>
    <quality shadowsize="8192"/>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.42 0.62 0.9" rgb2="0.92 0.95 1" width="512" height="3072"/>
    <texture name="snowtex" type="2d" builtin="flat" rgb1="0.94 0.96 1" rgb2="0.9 0.93 0.98" mark="random" random="0.08" markrgb="1 1 1" width="512" height="512"/>
    <material name="snow" texture="snowtex" texrepeat="40 40" specular="0.25" shininess="0.15" reflectance="0"/>
    <material name="ski_red" rgba="0.8 0.1 0.08 1" specular="0.6" shininess="0.6"/>
    <material name="gate_red" rgba="0.85 0.08 0.08 1"/>
    <material name="gate_blue" rgba="0.08 0.2 0.85 1"/>
    <hfield name="terrain" nrow="{hf['nrow']}" ncol="{hf['ncol']}" size="{rx} {ry} {zr} {base}"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 50" dir="0.35 0.25 -1" diffuse="0.75 0.75 0.72" castshadow="true"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="{px} {py} {pz}" material="snow"
          contype="0" conaffinity="1" condim="3" friction="0.6"/>
    {''.join(extras)}
  </worldbody>"""


def _ski_body(name: str, pos: tuple[float, float, float], roll: float, p: SkiParams) -> str:
  L, W, T = p.length, p.width, p.thickness
  m = 1.2
  ixx = m * (W**2 + T**2) / 12
  iyy = m * (L**2 + T**2) / 12
  izz = m * (L**2 + W**2) / 12
  quat = (np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0)
  return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
      <inertial pos="0 0 0" mass="{m}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="box" size="{L / 2} {W / 2} {T / 2}" material="ski_red" contype="0" conaffinity="0" group="1"/>
      <geom type="capsule" fromto="{L / 2 - 0.02} {-W / 2} {T / 2} {L / 2 + 0.05} {-W / 2} {T / 2 + 0.035}" size="{T / 2}"
            material="ski_red" contype="0" conaffinity="0" group="1"/>
      <geom type="capsule" fromto="{L / 2 - 0.02} {W / 2} {T / 2} {L / 2 + 0.05} {W / 2} {T / 2 + 0.035}" size="{T / 2}"
            material="ski_red" contype="0" conaffinity="0" group="1"/>
    </body>"""


def _compile(xml: str, grid: HeightGrid) -> mujoco.MjModel:
  model = mujoco.MjModel.from_xml_string(xml)
  hf = grid.hfield()
  model.hfield_data[:] = hf["data"].ravel()
  return model


def sled_model(
  grid: HeightGrid, params: SkiParams | None = None, ski_roll: float = 0.0, timestep: float = 0.002
) -> mujoco.MjModel:
  """A rigid 35 kg sled on two skis, for checking the ski force laws without a controller in the loop."""
  p = params or SkiParams()
  skis = _ski_body("left_ski", (0, 0.2, -0.15), ski_roll, p) + _ski_body("right_ski", (0, -0.2, -0.15), ski_roll, p)
  body = f"""
    <body name="sled" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="32.6" diaginertia="2.0 2.0 1.0"/>
      <geom type="box" size="0.3 0.25 0.05" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="0"/>
      {skis}
    </body>"""
  xml = f'<mujoco model="sled">{_world(grid, timestep, [body])}</mujoco>'
  return _compile(xml, grid)


def g1_model(
  grid: HeightGrid,
  params: SkiParams | None = None,
  extras: list[str] | None = None,
  timestep: float = 0.002,
  n_robots: int = 1,
  skis: bool = True,
) -> mujoco.MjModel:
  """Unitree G1 29-DoF (unitree_rl_mjlab g1.xml) with mjlab motor gains, skis bolted under both feet.

  With n_robots > 1 the robot is cloned with name prefixes r0_, r1_, ... (no robot-robot collisions).
  """
  p = params or SkiParams()
  tree = ET.parse(G1_DIR / "g1.xml")
  root = tree.getroot()
  root.find("compiler").set("meshdir", str(G1_DIR / "assets"))

  for geom in root.iter("geom"):
    name = geom.get("name", "")
    if not name.endswith("_collision"):
      continue
    if "_foot" in name:
      if skis:
        geom.set("contype", "0")  # the skis carry all snow contact
        geom.set("conaffinity", "0")
      else:
        geom.set("contype", "1")  # plain feet on the ground, as in unitree_rl_mjlab
        geom.set("conaffinity", "0")
        geom.set("condim", "3")
        geom.set("friction", "0.6")
    else:
      geom.set("contype", "1")  # bodies hit the snow when it falls, never each other
      geom.set("conaffinity", "0")
      geom.set("condim", "3")
      geom.set("friction", "0.6")

  actuators = ET.Element("actuator")
  for joint in root.iter("joint"):
    name = joint.get("name")
    if joint.tag != "joint" or name == "floating_base_joint" or joint.get("type") == "free":
      continue
    kp, kd, effort, armature = g1_motor(name)
    joint.set("armature", f"{armature:.6g}")
    ET.SubElement(
      actuators,
      "position",
      name=name.removesuffix("_joint"),
      joint=name,
      kp=f"{kp:.6g}",
      kv=f"{kd:.6g}",
      forcerange=f"{-effort} {effort}",
    )

  # Foot capsules sit at z=-0.025 with radius 0.01, so the sole is at -0.035; 2 cm binding plate below it.
  ski_z = -0.035 - 0.02 - p.thickness / 2
  for side in ("left", "right") if skis else ():
    foot = next(b for b in root.iter("body") if b.get("name") == f"{side}_ankle_roll_link")
    foot.append(ET.fromstring(_ski_body(f"{side}_ski", (0.04, 0.0, ski_z), 0.0, p)))

  if n_robots > 1:
    worldbody = root.find("worldbody")
    pelvis = worldbody.find("body")
    worldbody.remove(pelvis)
    template = list(actuators)
    for a in template:
      actuators.remove(a)
    for i in range(n_robots):
      pre = f"r{i}_"
      clone = copy.deepcopy(pelvis)
      for parent in list(clone.iter()):
        for light in parent.findall("light"):
          parent.remove(light)
      for e in clone.iter():
        if "name" in e.attrib:
          e.set("name", pre + e.get("name"))
      worldbody.append(clone)
      for a in template:
        a2 = copy.deepcopy(a)
        a2.set("name", pre + a.get("name"))
        a2.set("joint", pre + a.get("joint"))
        actuators.append(a2)
    # Contact excludes and IMU sensors name robot parts, so each clone gets its own prefixed copy.
    for section in ("contact", "sensor"):
      sec = root.find(section)
      if sec is None:
        continue
      items = list(sec)
      for it in items:
        sec.remove(it)
      for i in range(n_robots):
        for it in items:
          it2 = copy.deepcopy(it)
          for attr in ("name", "body1", "body2", "site", "body"):
            if attr in it2.attrib:
              it2.set(attr, f"r{i}_" + it2.get(attr))
          sec.append(it2)

  # Splice the robot into our world: keep its defaults, assets and bodies, add ours.
  ours = ET.fromstring(f'<mujoco model="g1_ski">{_world(grid, timestep, extras or [])}</mujoco>')
  for tag in ("option", "statistic", "visual"):
    root.append(copy.deepcopy(ours.find(tag)))
  root.find("asset").extend(list(ours.find("asset")))
  root.find("worldbody").extend(list(ours.find("worldbody")))
  root.append(actuators)
  return _compile(ET.tostring(root, encoding="unicode"), grid)


def place(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  grid: HeightGrid,
  xy: tuple[float, float],
  heading: float,
  joint_qpos: np.ndarray | None = None,
  ski_bodies: tuple[str, ...] = ("left_ski", "right_ski"),
  params: SkiParams | None = None,
  clearance: float = 0.003,
  root_joint: str | None = None,
) -> None:
  """Put a free root at xy, aligned with the slope normal and facing `heading`, skis just touching snow.

  root_joint picks which free joint to move when the model holds several robots (default: the first).
  """
  p = params or SkiParams()
  _, n = grid.height_normal(np.array([xy[0]]), np.array([xy[1]]))
  z_ax = n[0]
  fwd = np.array([np.cos(heading), np.sin(heading), 0.0])
  x_ax = fwd - fwd.dot(z_ax) * z_ax
  x_ax /= np.linalg.norm(x_ax)
  y_ax = np.cross(z_ax, x_ax)
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.column_stack([x_ax, y_ax, z_ax]).ravel())

  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint) if root_joint else 0
  qa, va = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
  if joint_qpos is not None:
    data.qpos[:] = joint_qpos
  data.qvel[va : va + 6] = 0
  data.qpos[qa : qa + 3] = (xy[0], xy[1], 0.0)
  data.qpos[qa + 3 : qa + 7] = quat
  mujoco.mj_kinematics(model, data)

  # Lowest clearance of any ski contact point above the terrain, then shift the root by it along z.
  xs = np.linspace(-0.48, 0.48, p.n_long) * p.length
  local = np.array([[x, y, -p.thickness / 2] for x in xs for y in (-p.width / 2, p.width / 2)])
  gap = np.inf
  for name in ski_bodies:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    P = data.xpos[b] + local @ data.xmat[b].reshape(3, 3).T
    h, _ = grid.height_normal(P[:, 0], P[:, 1])
    gap = min(gap, float((P[:, 2] - h).min()))
  data.qpos[qa + 2] -= gap - clearance
  mujoco.mj_forward(model, data)
