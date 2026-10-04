// The Unitree G1 in Three.js: the kinematic tree from the MJCF, decimated meshes from the GLB, and a
// retarget from MediaPipe world landmarks to G1 joint angles (a port of skisim/pose_from_video.py).
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { LM, type LandmarkLike as Landmark } from './landmarks.ts';

interface BodyDef { name: string; parent: string | null; pos: number[]; quat: number[]; joint: { name: string; axis: number[]; range: number[] } | null; geoms: { mesh: string; pos: number[]; quat: number[]; rgba: number[] }[] }
interface G1Def { bodies: BodyDef[]; meshes: string[] }

let cache: Promise<{ def: G1Def; meshes: Map<string, THREE.BufferGeometry> }> | null = null;

export function loadG1Assets(base = '/g1') {
  if (!cache) cache = (async () => {
    const [def, gltf] = await Promise.all([
      fetch(`${base}/g1.json`).then(r => r.json() as Promise<G1Def>),
      new GLTFLoader().loadAsync(`${base}/g1.glb`),
    ]);
    const meshes = new Map<string, THREE.BufferGeometry>();
    gltf.scene.traverse(o => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        // trimesh writes the MuJoCo z-up coordinates unchanged, and the world is z-up too.
        const g = m.geometry.clone();
        g.computeVertexNormals();
        // Name from the node, falling back to the mesh name trimesh assigned.
        meshes.set(o.name || m.name, g);
        if (o.parent && o.parent.name && !meshes.has(o.parent.name)) meshes.set(o.parent.name, g);
      }
    });
    return { def, meshes };
  })();
  return cache;
}

const mjQuat = (q: number[]) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

export class G1 {
  root = new THREE.Group();            // pelvis frame, z-up, x forward
  joints = new Map<string, { node: THREE.Object3D; axis: THREE.Vector3; base: THREE.Quaternion; range: number[]; q: number }>();
  bodies = new Map<string, THREE.Object3D>();
  skis: THREE.Mesh[] = [];
  private tmpQ = new THREE.Quaternion();

  constructor(def: G1Def, meshes: Map<string, THREE.BufferGeometry>, tint: THREE.Color | null = null, ghost = false) {
    const silver = new THREE.MeshStandardMaterial({ color: tint ? tint.clone().lerp(new THREE.Color('#dfe6ee'), 0.55) : '#dfe6ee', roughness: 0.45, metalness: 0.25, transparent: ghost, opacity: ghost ? 0.55 : 1 });
    const black = new THREE.MeshStandardMaterial({ color: '#23272e', roughness: 0.6, metalness: 0.2, transparent: ghost, opacity: ghost ? 0.55 : 1 });
    const skiMat = new THREE.MeshStandardMaterial({ color: tint ?? '#e0312f', roughness: 0.35, metalness: 0.1, transparent: ghost, opacity: ghost ? 0.6 : 1 });
    for (const b of def.bodies) {
      const node = new THREE.Object3D();
      node.name = b.name;
      node.position.set(b.pos[0], b.pos[1], b.pos[2]);
      node.quaternion.copy(mjQuat(b.quat));
      if (b.parent === null) { node.position.set(0, 0, 0); this.root.add(node); }
      else this.bodies.get(b.parent)!.add(node);
      this.bodies.set(b.name, node);
      if (b.joint) this.joints.set(b.joint.name, { node, axis: new THREE.Vector3(...b.joint.axis).normalize(), base: node.quaternion.clone(), range: b.joint.range, q: 0 });
      for (const g of b.geoms) {
        const geo = meshes.get(g.mesh);
        if (!geo) continue;
        const dark = g.rgba[0] < 0.4;
        const m = new THREE.Mesh(geo, dark ? black : silver);
        m.position.set(g.pos[0], g.pos[1], g.pos[2]);
        m.quaternion.copy(mjQuat(g.quat));
        m.castShadow = !ghost;
        node.add(m);
      }
    }
    // Skis under the ankle roll links (same numbers as course.SKI): box 1.0 x 0.08 m with a turned-up tip.
    for (const side of ['left', 'right']) {
      const ankle = this.bodies.get(`${side}_ankle_roll_link`)!;
      const ski = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.08, 0.012), skiMat);
      ski.position.set(0.04, 0, -0.035);
      ski.castShadow = !ghost;
      const tip = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.08, 0.012), skiMat);
      tip.position.set(0.58, 0, 0.02); tip.rotation.y = -0.6;
      ski.add(tip);
      ankle.add(ski);
      this.skis.push(ski);
    }
    // Ski poles: hang down and back in the skier's own frame, re-attached to the hands every frame.
    const poleMat = new THREE.MeshStandardMaterial({ color: '#1b1f27', roughness: 0.5, metalness: 0.5, transparent: ghost, opacity: ghost ? 0.5 : 1 });
    for (const side of ['left', 'right']) {
      const pole = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.006, 0.95, 6), poleMat);
      shaft.position.y = -0.42;
      const basket = new THREE.Mesh(new THREE.TorusGeometry(0.035, 0.005, 6, 10), poleMat);
      basket.position.y = -0.82; basket.rotation.x = Math.PI / 2;
      const grip = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.012, 0.1, 6), skiMat);
      pole.add(shaft, basket, grip);
      pole.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0.45, 0, 1).normalize()); // tip down-back, grip up-forward
      this.root.add(pole);
      this.poles.push(pole);
      this.hands.push(this.bodies.get(`${side}_wrist_yaw_link`)!);
    }
    // Race bib on the chest.
    this.bib = new THREE.Mesh(new THREE.PlaneGeometry(0.17, 0.15), new THREE.MeshStandardMaterial({ color: tint ?? '#ffffff', roughness: 0.8, transparent: ghost, opacity: ghost ? 0.6 : 1 }));
    this.bib.position.set(0.095, 0, 0.19);
    this.bib.rotation.y = Math.PI / 2; this.bib.rotation.z = Math.PI / 2; // face +x, upright
    this.bodies.get('torso_link')!.add(this.bib);
    this.setBib('', tint ? '#' + tint.getHexString() : '#ff3b4e', 1);
  }

  poles: THREE.Group[] = [];
  hands: THREE.Object3D[] = [];
  bib!: THREE.Mesh;
  private _hp = new THREE.Vector3();

  /** Move the poles to the hands (call after the root has been placed and matrices updated). */
  updatePoles() {
    for (let i = 0; i < this.poles.length; i++) {
      this.hands[i].getWorldPosition(this._hp);
      this.root.worldToLocal(this._hp);
      this.poles[i].position.copy(this._hp);
    }
  }

  /** Paint the bib: a number, the name, a colour stripe. */
  setBib(name: string, color: string, number = 1) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 224;
    const g = c.getContext('2d')!;
    g.fillStyle = '#f7f9fc'; g.fillRect(0, 0, 256, 224);
    g.fillStyle = color; g.fillRect(0, 0, 256, 34); g.fillRect(0, 190, 256, 34);
    g.fillStyle = '#0b1220'; g.font = '900 120px system-ui,sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(String(number), 128, 104);
    g.font = '700 30px system-ui,sans-serif'; g.fillStyle = '#fff';
    g.fillText((name || 'GROUND TRUTH').toUpperCase().slice(0, 14), 128, 17);
    g.fillText('GROUND TRUTH', 128, 207);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
    const m = this.bib.material as THREE.MeshStandardMaterial;
    m.map = tex; m.color.set('#ffffff'); m.needsUpdate = true;
  }

  set(joint: string, value: number) {
    const j = this.joints.get(joint);
    if (!j) return;
    const v = Math.max(j.range[0], Math.min(j.range[1], value));
    j.q = v;
    j.node.quaternion.copy(j.base).multiply(this.tmpQ.setFromAxisAngle(j.axis, v));
  }

  /** Blend joints toward a target pose. */
  apply(target: Record<string, number>, alpha = 1) {
    for (const [k, v] of Object.entries(target)) {
      const j = this.joints.get(k);
      if (!j) continue;
      this.set(k, j.q + (v - j.q) * alpha);
    }
  }
}

// Athletic ski stance (skisim/scene.py SKI_STANCE), used when there is no body to copy.
export const SKI_STANCE: Record<string, number> = {
  left_hip_pitch_joint: -0.55, right_hip_pitch_joint: -0.55, left_knee_joint: 1.0, right_knee_joint: 1.0,
  left_ankle_pitch_joint: -0.45, right_ankle_pitch_joint: -0.45, waist_pitch_joint: 0.25,
  left_shoulder_pitch_joint: -0.25, right_shoulder_pitch_joint: -0.25, left_shoulder_roll_joint: 0.3, right_shoulder_roll_joint: -0.3,
  left_elbow_joint: 0.9, right_elbow_joint: 0.9,
};

/** A plausible pose from only lean/crouch, for ghosts of other skiers. */
export function stanceFrom(lean: number, crouch: number): Record<string, number> {
  const knee = 0.7 + 0.9 * crouch;
  const roll = -0.35 * lean; // both hips roll into the turn
  return {
    ...SKI_STANCE,
    left_knee_joint: knee, right_knee_joint: knee,
    left_hip_pitch_joint: -(knee - 0.4), right_hip_pitch_joint: -(knee - 0.4),
    left_ankle_pitch_joint: -0.4, right_ankle_pitch_joint: -0.4,
    left_hip_roll_joint: roll, right_hip_roll_joint: roll,
    waist_pitch_joint: 0.2 + 0.3 * crouch, waist_roll_joint: 0.25 * lean,
    left_shoulder_roll_joint: 0.3 + 0.3 * Math.max(0, -lean), right_shoulder_roll_joint: -0.3 - 0.3 * Math.max(0, lean),
    left_elbow_joint: 0.9 + 0.4 * crouch, right_elbow_joint: 0.9 + 0.4 * crouch,
  };
}

const v = (p: Landmark) => new THREE.Vector3(p.x, -p.y, -p.z); // right-handed, y up, z toward the camera
const unit = (a: THREE.Vector3) => a.clone().normalize();
const angle = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3) => a.clone().sub(b).angleTo(c.clone().sub(b));
const clip = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/**
 * MediaPipe world landmarks -> G1 joint targets. Same recipe as pose_from_video.py: joint angles in the
 * skier's own pelvis frame, feet kept flat under the pelvis (hip + knee + ankle = 0), extra chest lean to
 * the waist, frontal leg angle to hip roll, arms from the upper-arm direction.
 */
export function retarget(w: Landmark[]): Record<string, number> {
  const P = w.map(v);
  const midHip = P[LM.lhip].clone().add(P[LM.rhip]).multiplyScalar(0.5);
  const midSh = P[LM.lsh].clone().add(P[LM.rsh]).multiplyScalar(0.5);
  const up = unit(midSh.clone().sub(midHip));
  let right = P[LM.rhip].clone().sub(P[LM.lhip]);
  right = unit(right.sub(up.clone().multiplyScalar(right.dot(up))));
  const fwd = new THREE.Vector3().crossVectors(up, right);
  const q: Record<string, number> = {};
  const extra: number[] = [];
  for (const [side, s, sh, el, wr, hip, kn, an, he, ft] of [
    ['L', 'left', LM.lsh, LM.lel, LM.lwr, LM.lhip, LM.lkn, LM.lan, LM.lhe, LM.lft],
    ['R', 'right', LM.rsh, LM.rel, LM.rwr, LM.rhip, LM.rkn, LM.ran, LM.rhe, LM.rft],
  ] as const) {
    const knee = Math.PI - angle(P[hip], P[kn], P[an]);
    const torsoThigh = Math.PI - angle(midSh, P[hip], P[kn]);
    const foot = P[ft].clone().sub(P[he]);
    const dorsi = Math.PI / 2 - angle(P[kn], P[an], P[an].clone().add(foot));
    const elbow = Math.PI - angle(P[sh], P[el], P[wr]);
    const U = P[el].clone().sub(P[sh]);
    const armFwd = Math.atan2(U.dot(fwd), -U.dot(up));
    const outward = side === 'L' ? right.clone().negate() : right;
    const armAbd = Math.atan2(U.dot(outward), -U.dot(up));
    const thigh = P[kn].clone().sub(P[hip]);
    const legFrontal = Math.atan2(thigh.dot(right), -thigh.dot(up));
    const k = clip(knee, 0.2, 2.0), d = clip(dorsi, -0.1, 0.8);
    q[`${s}_knee_joint`] = k;
    q[`${s}_ankle_pitch_joint`] = -d;
    q[`${s}_hip_pitch_joint`] = -(k - d);
    q[`${s}_hip_roll_joint`] = clip(-legFrontal, -0.45, 0.45);
    q[`${s}_elbow_joint`] = clip(elbow, 0.1, 1.9);
    const roll = clip(armAbd, -0.2, 1.2);
    q[`${s}_shoulder_roll_joint`] = side === 'L' ? roll : -roll;
    q[`${s}_shoulder_pitch_joint`] = clip(-armFwd, -1.6, 0.6);
    extra.push(torsoThigh - knee + dorsi);
  }
  q.waist_pitch_joint = clip((extra[0] + extra[1]) / 2, 0, 0.5);
  return q;
}
