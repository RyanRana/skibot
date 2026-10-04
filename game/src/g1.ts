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

/** Heart light colours: calm teal, worked-up amber, racing red. */
const HEART = [new THREE.Color('#3dffc8'), new THREE.Color('#ffb340'), new THREE.Color('#ff3b5c')];

const mjQuat = (q: number[]) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

export class G1 {
  /** Set once from the world's prefiltered sky so the shell and the visor reflect the mountain. */
  static env: THREE.Texture | null = null;
  root = new THREE.Group();            // pelvis frame, z-up, x forward
  joints = new Map<string, { node: THREE.Object3D; axis: THREE.Vector3; base: THREE.Quaternion; range: number[]; q: number }>();
  bodies = new Map<string, THREE.Object3D>();
  skis: THREE.Mesh[] = [];
  private tmpQ = new THREE.Quaternion();

  constructor(def: G1Def, meshes: Map<string, THREE.BufferGeometry>, tint: THREE.Color | null = null, ghost = false) {
    const env = G1.env, fade = { transparent: ghost, opacity: ghost ? 0.55 : 1, envMap: env };
    // the shell: clear-coated white composite; the joints and the face: dark anodised aluminium
    const silver = ghost
      ? new THREE.MeshStandardMaterial({ color: tint ? tint.clone().lerp(new THREE.Color('#dfe6ee'), 0.55) : '#dfe6ee', roughness: 0.4, metalness: 0.1, emissive: tint ?? '#000000', emissiveIntensity: 0.25, ...fade })
      : new THREE.MeshPhysicalMaterial({ color: '#eef2f6', roughness: 0.38, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.12, envMapIntensity: 0.9, ...fade });
    const black = ghost
      ? new THREE.MeshStandardMaterial({ color: '#23272e', roughness: 0.5, metalness: 0.3, ...fade })
      : new THREE.MeshPhysicalMaterial({ color: '#1a1f27', roughness: 0.32, metalness: 0.75, clearcoat: 0.4, clearcoatRoughness: 0.3, envMapIntensity: 1.0, ...fade });
    const skiMat = new THREE.MeshPhysicalMaterial({ map: skiTexture(tint ? '#' + tint.getHexString() : '#e0312f'), roughness: 0.25, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.1, transparent: ghost, opacity: ghost ? 0.6 : 1, envMap: env });
    const edgeMat = new THREE.MeshStandardMaterial({ color: '#9aa3ad', roughness: 0.25, metalness: 1, envMap: env, transparent: ghost, opacity: ghost ? 0.6 : 1 });
    this.mats = [silver, black];
    // a cool rim light so the white robot separates from the white snow
    if (!ghost) for (const m of [silver, black]) {
      m.onBeforeCompile = sh => {
        sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          float rimF = pow(1.0 - saturate(dot(normal, normalize(vViewPosition))), 3.0);
          totalEmissiveRadiance += vec3(0.62, 0.78, 1.0) * rimF * 0.45;`);
      };
      m.customProgramCacheKey = () => 'g1-rim';
    }
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
    // Skis under the ankle roll links (same numbers as course.SKI): 1.0 m race skis with sidecut, a rising
    // shovel, steel edges and a binding plate.
    const skiGeo = skiGeometry();
    const bindGeo = new THREE.BoxGeometry(0.2, 0.06, 0.03).translate(0, 0, 0.017);
    const bindMat = new THREE.MeshStandardMaterial({ color: '#15181d', roughness: 0.45, metalness: 0.4, envMap: env, transparent: ghost, opacity: ghost ? 0.6 : 1 });
    for (const side of ['left', 'right']) {
      const ankle = this.bodies.get(`${side}_ankle_roll_link`)!;
      const ski = new THREE.Mesh(skiGeo, [skiMat, edgeMat]);
      ski.position.set(0.04, 0, -0.035);
      ski.castShadow = !ghost;
      const bind = new THREE.Mesh(bindGeo, bindMat); ski.add(bind);
      ankle.add(ski);
      this.skis.push(ski);
    }
    // A race helmet over the head and mirrored goggles: the robot is a racer.
    if (!ghost) {
      const torso = this.bodies.get('torso_link')!;
      const helmetMat = new THREE.MeshPhysicalMaterial({ color: tint ?? '#e0312f', roughness: 0.25, metalness: 0.2, clearcoat: 1, clearcoatRoughness: 0.06, envMap: env, envMapIntensity: 1.1 });
      this.helmetMat = helmetMat;
      const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.118, 32, 20, 0, Math.PI * 2, 0, Math.PI * 0.56), helmetMat);
      helmet.rotation.x = Math.PI / 2; helmet.rotation.y = 0; // pole up the torso's z
      helmet.position.set(-0.004, 0, 0.392); helmet.scale.set(1.04, 1, 1.12);
      helmet.castShadow = true; torso.add(helmet);
      const stripe = new THREE.Mesh(new THREE.TorusGeometry(0.104, 0.007, 6, 40, Math.PI), new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.3, envMap: env }));
      stripe.position.set(-0.004, 0, 0.392); stripe.rotation.x = Math.PI / 2; stripe.rotation.z = Math.PI / 2; stripe.scale.set(1.12, 1.04, 1); // front to back over the crown
      torso.add(stripe);
      const visorMat = new THREE.MeshPhysicalMaterial({ color: '#1a1206', roughness: 0.05, metalness: 1, iridescence: 1, iridescenceIOR: 1.6, iridescenceThicknessRange: [250, 700], envMap: env, envMapIntensity: 1.6 });
      const goggles = new THREE.Mesh(new THREE.CylinderGeometry(0.104, 0.1, 0.05, 32, 1, true, -1.15, 2.3), visorMat);
      goggles.rotation.x = Math.PI / 2; goggles.rotation.y = Math.PI / 2; // axis up the torso's z, the arc facing +x
      goggles.position.set(0.006, 0, 0.405);
      torso.add(goggles);
      const strap = new THREE.Mesh(new THREE.CylinderGeometry(0.106, 0.102, 0.03, 32, 1, true, 1.15, Math.PI * 2 - 2.3), new THREE.MeshStandardMaterial({ color: '#111318', roughness: 0.8, side: THREE.DoubleSide }));
      strap.rotation.x = Math.PI / 2; strap.rotation.y = Math.PI / 2;
      strap.position.set(0.006, 0, 0.405);
      torso.add(strap);
    }
    // Ski poles: hang down and back in the skier's own frame, re-attached to the hands every frame.
    const poleMat = new THREE.MeshStandardMaterial({ color: '#2b313b', roughness: 0.3, metalness: 0.8, envMap: env, transparent: ghost, opacity: ghost ? 0.5 : 1 });
    const gripMat = new THREE.MeshStandardMaterial({ color: '#111318', roughness: 0.7, transparent: ghost, opacity: ghost ? 0.5 : 1 });
    for (const side of ['left', 'right']) {
      const pole = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.006, 0.95, 6), poleMat);
      shaft.position.y = -0.42;
      const basket = new THREE.Mesh(new THREE.TorusGeometry(0.035, 0.005, 6, 10), poleMat);
      basket.position.y = -0.82; basket.rotation.x = Math.PI / 2;
      const grip = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.012, 0.11, 8), gripMat);
      pole.add(shaft, basket, grip);
      pole.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0.45, 0, 1).normalize()); // tip down-back, grip up-forward
      this.root.add(pole);
      this.poles.push(pole);
      this.hands.push(this.bodies.get(`${side}_wrist_yaw_link`)!);
    }
    // A heart light above the bib: it pulses with the player's real heartbeat (Presage).
    if (!ghost) {
      this.heartMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#ff3b5c'), transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
      const heart = new THREE.Mesh(new THREE.CircleGeometry(0.022, 20), this.heartMat);
      heart.position.set(0.1, 0, 0.27); heart.rotation.y = Math.PI / 2; heart.rotation.z = Math.PI / 2;
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.026, 0.032, 24), this.heartMat);
      heart.add(ring);
      this.bodies.get('torso_link')!.add(heart);
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
  mats: THREE.Material[] = [];
  helmetMat: THREE.MeshPhysicalMaterial | null = null;
  heartMat: THREE.MeshBasicMaterial | null = null;
  /** Heartbeat light: `k` 0..1 (the beat), colour from calm teal to racing red, HDR so it blooms. */
  heart(k: number, arousal: number) {
    if (!this.heartMat) return;
    this.heartMat.opacity = k > 0 ? 0.25 + 0.75 * Math.min(1, k) : 0;
    const a = Math.max(0, Math.min(1, arousal));
    if (a < 0.5) this.heartMat.color.copy(HEART[0]).lerp(HEART[1], a * 2); else this.heartMat.color.copy(HEART[1]).lerp(HEART[2], (a - 0.5) * 2);
    this.heartMat.color.multiplyScalar(1 + 2.5 * Math.min(1, k));
  }
  bib!: THREE.Mesh;
  private _hp = new THREE.Vector3();

  /** Ghosts only: scale every part's opacity (a ghost right at the lens fades out instead of filling the frame). */
  fade(k: number) {
    if (Math.abs(k - this._fade) < 0.02) return;
    this._fade = k;
    this.root.traverse(o => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (!m) return;
      for (const mm of Array.isArray(m) ? m : [m]) {
        if (!mm.transparent) continue;
        mm.userData.base ??= mm.opacity;
        mm.opacity = mm.userData.base * k;
        mm.visible = k > 0.03;
      }
    });
  }
  private _fade = 1;

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
    g.fillStyle = '#fff';
    fit(g, (name || 'HAZARD INTEL').toUpperCase().slice(0, 14), 128, 17, 240, 30, 700);
    fit(g, 'HAZARD INTELLIGENCE', 128, 207, 240, 30, 700);
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

/**
 * A skier's pose from only lean and crouch: used for ghosts, for other players, and for your own legs when the
 * camera does not see them (a laptop camera usually stops at the hips). Carving: both legs incline into the
 * turn from the hips while the upper body stays quieter (angulation), the inside knee bends more than the
 * outside one, the outside arm opens for balance and the inside hand drives forward. Crouch blends into a
 * downhill tuck: deep knees, chest down over the thighs, hands together in front of the face.
 */
export function stanceFrom(lean: number, crouch: number): Record<string, number> {
  const l = Math.max(-1, Math.min(1, lean)), c = Math.max(0, Math.min(1, crouch));
  const tuck = Math.max(0, (c - 0.45) / 0.55);         // 0 until a real tuck, 1 in a full one
  const inR = Math.max(0, l), inL = Math.max(0, -l);    // + lean = right turn: the right leg is the inside leg
  const knee = 0.72 + 0.95 * c;
  const kneeL = knee + 0.38 * inL - 0.12 * inR, kneeR = knee + 0.38 * inR - 0.12 * inL;
  const ankle = -0.38 - 0.12 * c;
  const fold = 0.12 * tuck;                             // hips fold a little further than the knees in a tuck
  const roll = -0.36 * l;                               // both hips roll into the turn
  const armOut = (inside: number, outside: number) => 0.3 + 0.35 * outside - 0.1 * inside;
  return {
    ...SKI_STANCE,
    left_knee_joint: kneeL, right_knee_joint: kneeR,
    left_hip_pitch_joint: -(kneeL - 0.42) - fold, right_hip_pitch_joint: -(kneeR - 0.42) - fold,
    left_ankle_pitch_joint: ankle, right_ankle_pitch_joint: ankle,
    left_hip_roll_joint: roll, right_hip_roll_joint: roll,
    waist_pitch_joint: 0.2 + 0.25 * c + 0.25 * tuck, waist_roll_joint: 0.28 * l,
    // arms: athletic and wide when carving, forward and tucked in a tuck
    left_shoulder_pitch_joint: (-0.3 - 0.4 * inL) * (1 - tuck) + -1.05 * tuck,
    right_shoulder_pitch_joint: (-0.3 - 0.4 * inR) * (1 - tuck) + -1.05 * tuck,
    left_shoulder_roll_joint: armOut(inL, inR) * (1 - tuck) + 0.12 * tuck,
    right_shoulder_roll_joint: -(armOut(inR, inL) * (1 - tuck) + 0.12 * tuck),
    left_elbow_joint: (0.9 + 0.3 * c) * (1 - tuck) + 1.75 * tuck,
    right_elbow_joint: (0.9 + 0.3 * c) * (1 - tuck) + 1.75 * tuck,
  };
}

const v = (p: Landmark) => new THREE.Vector3(p.x, -p.y, -p.z); // right-handed, y up, z toward the camera
const unit = (a: THREE.Vector3) => a.clone().normalize();
const angle = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3) => a.clone().sub(b).angleTo(c.clone().sub(b));
const clip = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** Arms only (shoulders, elbows, wrists), for when the camera sees the upper body but not the legs. */
export function retargetArms(w: Landmark[]): Record<string, number> {
  const P = w.map(v);
  const midHip = P[LM.lhip].clone().add(P[LM.rhip]).multiplyScalar(0.5);
  const midSh = P[LM.lsh].clone().add(P[LM.rsh]).multiplyScalar(0.5);
  const up = unit(midSh.clone().sub(midHip));
  let right = P[LM.rsh].clone().sub(P[LM.lsh]);
  right = unit(right.sub(up.clone().multiplyScalar(right.dot(up))));
  const fwd = new THREE.Vector3().crossVectors(up, right);
  const q: Record<string, number> = {};
  for (const [side, s, sh, el, wr] of [['L', 'left', LM.lsh, LM.lel, LM.lwr], ['R', 'right', LM.rsh, LM.rel, LM.rwr]] as const) {
    const elbow = Math.PI - angle(P[sh], P[el], P[wr]);
    const U = P[el].clone().sub(P[sh]);
    const armFwd = Math.atan2(U.dot(fwd), -U.dot(up));
    const outward = side === 'L' ? right.clone().negate() : right;
    const armAbd = Math.atan2(U.dot(outward), -U.dot(up));
    q[`${s}_elbow_joint`] = clip(elbow, 0.1, 1.9);
    const roll = clip(armAbd, -0.2, 1.4);
    q[`${s}_shoulder_roll_joint`] = side === 'L' ? roll : -roll;
    q[`${s}_shoulder_pitch_joint`] = clip(-armFwd, -2.4, 0.8);
  }
  return q;
}

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

/** A 1.0 m race ski along +x, 12 mm thick: sidecut (wider at the shovel and tail), a shovel that rises over
 *  the last 15 cm and a slightly raised tail. Group 0 is the top sheet and base, group 1 the steel edges. */
function skiGeometry() {
  const N = 40, L = 1.0, x0 = -0.46;
  const halfW = (u: number) => 0.04 + 0.009 * Math.pow(Math.abs(u - 0.47) / 0.53, 1.6) - (u > 0.94 ? (u - 0.94) * 0.25 : 0);
  const rise = (u: number) => u > 0.85 ? Math.pow((u - 0.85) / 0.15, 2) * 0.055 : u < 0.06 ? Math.pow((0.06 - u) / 0.06, 2) * 0.012 : 0;
  const pos: number[] = [], uv: number[] = [], top: number[] = [], edge: number[] = [];
  const T = 0.012;
  for (let i = 0; i <= N; i++) {
    const u = i / N, x = x0 + u * L, w = halfW(u), z = rise(u);
    // 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
    pos.push(x, w, z + T / 2, x, -w, z + T / 2, x, w, z - T / 2, x, -w, z - T / 2);
    uv.push(u, 1, u, 0, u, 1, u, 0);
  }
  for (let i = 0; i < N; i++) {
    const a = i * 4, b = a + 4;
    top.push(a, a + 1, b + 1, a, b + 1, b);             // top sheet
    top.push(a + 2, b + 2, b + 3, a + 2, b + 3, a + 3); // base
    edge.push(a, b, b + 2, a, b + 2, a + 2);            // left edge
    edge.push(a + 1, a + 3, b + 3, a + 1, b + 3, b + 1); // right edge
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex([...top, ...edge]);
  g.addGroup(0, top.length, 0); g.addGroup(top.length, edge.length, 1);
  g.computeVertexNormals();
  return g;
}

/** Top sheet: the racer's colour, a white stripe, the project name down the ski. */
function skiTexture(color: string) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 64;
  const g = c.getContext('2d')!;
  const gr = g.createLinearGradient(0, 0, 512, 0); gr.addColorStop(0, '#0d1118'); gr.addColorStop(0.3, color); gr.addColorStop(1, color);
  g.fillStyle = gr; g.fillRect(0, 0, 512, 64);
  g.fillStyle = 'rgba(255,255,255,.9)'; g.fillRect(0, 28, 512, 3);
  g.fillStyle = '#ffffff'; g.font = 'italic 900 21px system-ui,sans-serif'; g.textBaseline = 'middle'; g.fillText('HAZARD INTELLIGENCE', 246, 33);
  g.fillStyle = '#0d1118'; g.fillRect(170, 0, 60, 64); // binding zone
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}

function fit(g: CanvasRenderingContext2D, text: string, x: number, y: number, maxW: number, size: number, weight = 900) {
  do { g.font = `${weight} ${size}px system-ui,sans-serif`; size -= 1; } while (g.measureText(text).width > maxW && size > 8);
  g.fillText(text, x, y);
}
