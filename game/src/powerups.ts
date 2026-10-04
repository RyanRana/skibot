// Powerups on the Streif. Fixed placements (the same for every run, so times stay comparable), picked up
// by skiing into them, no buttons:
//   Slipstream rings  cyan hoops you ski through: two seconds of thrust and the trails glow
//   Time crystals     teal shards in threes, a little off the line: each takes 0.1 s off the clock
//   Shield            a white-blue bubble that forgives the next crash or missed gate
//   Magnet            violet: the next three gates count a pass well outside the poles
// Everything glows above the bloom threshold so it reads from far away on white snow, and every pickup
// answers with light (burst, shockwave, screen tint), never a sound.
import * as THREE from 'three';
import type { Course } from './course.ts';
import { initialState, step, GateTracker } from './physics.ts';

export type PowerKind = 'ring' | 'crystal' | 'shield' | 'magnet';
export interface Pickup { kind: PowerKind; x: number; y: number; z: number; s: number; yaw: number; taken: boolean; ph: number; obj: THREE.Object3D | null; slot: number }

export const POWER_COLOR: Record<PowerKind | 'tuck', string> = { ring: '#38e8ff', crystal: '#3dffc8', shield: '#cfe8ff', magnet: '#b07cff', tuck: '#ffb340' };
const hdr = (hex: string, k: number) => new THREE.Color(hex).multiplyScalar(k);

/** Additive Fresnel shell: bright rim, clear middle, an optional hex pattern drifting over it. */
export function fresnelMaterial(color: THREE.Color, opts: { hex?: boolean; alpha?: number } = {}) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uColor: { value: color }, uTime: { value: 0 }, uAlpha: { value: opts.alpha ?? 1 }, uHex: { value: opts.hex ? 1 : 0 } },
    vertexShader: `varying vec3 vN; varying vec3 vV; varying vec3 vP;
      void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = -mv.xyz; vP = position; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uTime, uAlpha, uHex; varying vec3 vN; varying vec3 vV; varying vec3 vP;
      float hexd(vec2 p){ p = abs(p); return max(dot(p, normalize(vec2(1.0, 1.732))), p.x); }
      void main(){
        float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
        float h = 0.0;
        if (uHex > 0.5) {
          vec2 p = vec2(atan(vP.y, vP.x) * 2.2, vP.z * 4.0 + uTime * 0.4);
          vec2 r = vec2(1.0, 1.732), hh = r * 0.5;
          vec2 a = mod(p, r) - hh, b = mod(p - hh, r) - hh;
          vec2 g = dot(a, a) < dot(b, b) ? a : b;
          h = smoothstep(0.42, 0.48, hexd(g)) * (0.5 + 0.5 * sin(uTime * 3.0 + vP.z * 6.0));
        }
        float a = (f * 1.2 + h * 0.35 + 0.04) * uAlpha;
        gl_FragColor = vec4(uColor * a, a);
      }`,
  });
}

function shieldShape() {
  const s = new THREE.Shape();
  s.moveTo(0, 0.32); s.quadraticCurveTo(0.14, 0.27, 0.26, 0.28); s.lineTo(0.25, 0.02); s.quadraticCurveTo(0.22, -0.2, 0, -0.34);
  s.quadraticCurveTo(-0.22, -0.2, -0.25, 0.02); s.lineTo(-0.26, 0.28); s.quadraticCurveTo(-0.14, 0.27, 0, 0.32);
  return s;
}

/** The world side of the powerups: tokens, rings, crystals, halos on the snow, and pickup shockwaves. */
export class Powerups {
  pickups: Pickup[] = [];
  private group = new THREE.Group();
  private crystals: THREE.InstancedMesh | null = null;
  private crystalIdx: number[] = [];
  private halos: THREE.InstancedMesh;
  private fresnels: THREE.ShaderMaterial[] = [];
  private chevronMats: THREE.ShaderMaterial[] = [];
  private waves: { mesh: THREE.Mesh; t: number }[] = [];
  private t = 0;
  private m = new THREE.Matrix4(); private q = new THREE.Quaternion(); private p = new THREE.Vector3(); private s = new THREE.Vector3(); private z = new THREE.Vector3(0, 0, 1);

  private course: Course;
  constructor(scene: THREE.Scene, course: Course, env: THREE.Texture | null) {
    this.course = course;
    scene.add(this.group);
    this.place();
    // crystals: one instanced mesh, faceted, glowing teal
    const cr = this.pickups.filter(p => p.kind === 'crystal');
    if (cr.length) {
      const geo = new THREE.OctahedronGeometry(0.17, 0).scale(1, 1, 1.8);
      const mat = new THREE.MeshStandardMaterial({ color: '#7fffe0', emissive: hdr(POWER_COLOR.crystal, 1), emissiveIntensity: 2.4, roughness: 0.08, metalness: 0.3, envMap: env, flatShading: true });
      this.crystals = new THREE.InstancedMesh(geo, mat, cr.length);
      this.crystals.frustumCulled = false;
      cr.forEach((p, i) => { p.slot = i; this.crystalIdx.push(this.pickups.indexOf(p)); });
      this.group.add(this.crystals);
    }
    for (const p of this.pickups) if (p.kind !== 'crystal') p.obj = this.token(p, env);
    // a glowing ring on the snow under every pickup
    const haloMat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
    this.halos = new THREE.InstancedMesh(new THREE.RingGeometry(0.62, 0.8, 40), haloMat, this.pickups.length);
    this.halos.frustumCulled = false;
    this.pickups.forEach((p, i) => this.halos.setColorAt(i, hdr(POWER_COLOR[p.kind], 1.4)));
    this.group.add(this.halos);
    for (let i = 0; i < 4; i++) {
      const w = new THREE.Mesh(new THREE.TorusGeometry(1, 0.05, 8, 48), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
      w.visible = false; this.group.add(w); this.waves.push({ mesh: w, t: 1 });
    }
    this.update(0, 0, 0, 0);
  }

  /** Fixed layout on the racing line: a clean run through the gates is simulated once at load (the
   *  physics is deterministic, so every player gets the same layout), rings go right on that line between
   *  gates, crystals in threes just off it, a shield and a magnet a step to the side. */
  private place() {
    const c = this.course, G = c.gates;
    const line = racingLine(c);
    const at = (sm: number) => { let i = 0; while (i < line.length - 1 && line[i].s < sm) i++; return line[i]; };
    const add = (kind: PowerKind, x: number, y: number, sAlong: number, yaw: number, lift: number) => {
      if (!c.inside(x, y, 3)) return;
      this.pickups.push({ kind, x, y, z: c.sample(x, y).h + lift, s: sAlong, yaw, taken: false, ph: this.pickups.length * 1.7, obj: null, slot: -1 });
    };
    const plan: Record<number, PowerKind> = {};
    for (const k of [1, 3, 6, 9, 14, 18, 22, 26, 30]) plan[k] = 'crystal';
    for (const k of [4, 12, 20, 28]) plan[k] = 'ring';
    for (const k of [10, 24]) plan[k] = 'shield';
    for (const k of [7, 16]) plan[k] = 'magnet';
    for (let k = 0; k < G.length - 1; k++) {
      const kind = plan[k]; if (!kind || !line.length) continue;
      const sm = (G[k].s + G[k + 1].s) / 2, side = k % 2 ? 1 : -1;
      if (kind === 'crystal') for (const j of [-1, 0, 1]) {
        const p = at(sm + j * 2.4), lx = -Math.sin(p.h), ly = Math.cos(p.h);
        add('crystal', p.x + lx * side * 1.8, p.y + ly * side * 1.8, p.s, p.h, 0.85);
      } else {
        const p = at(sm), lx = -Math.sin(p.h), ly = Math.cos(p.h), off = kind === 'ring' ? 0 : 1.6 * side;
        add(kind, p.x + lx * off, p.y + ly * off, p.s, p.h, kind === 'ring' ? 0 : 1.05);
      }
    }
  }

  private token(p: Pickup, env: THREE.Texture | null): THREE.Object3D {
    const g = new THREE.Group();
    const col = POWER_COLOR[p.kind];
    if (p.kind === 'ring') {
      // a hoop standing across the line, chevrons streaming through it, a pillar of light above
      const hoop = new THREE.Mesh(new THREE.TorusGeometry(1.8, 0.1, 14, 72), new THREE.MeshBasicMaterial({ color: hdr(col, 2.6) }));
      const outer = new THREE.Mesh(new THREE.TorusGeometry(1.95, 0.025, 8, 72), new THREE.MeshBasicMaterial({ color: hdr('#ffffff', 2.2) }));
      const chev = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        uniforms: { uTime: { value: 0 }, uColor: { value: hdr(col, 1) } },
        vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        fragmentShader: `uniform float uTime; uniform vec3 uColor; varying vec2 vUv;
          void main(){ vec2 d = vUv - 0.5; float r = length(d) * 2.0; if (r > 1.0) discard;
            float c = fract(vUv.y * 2.5 - abs(vUv.x - 0.5) * 1.6 + uTime * 1.8);
            float a = smoothstep(0.55, 0.62, c) * (1.0 - smoothstep(0.8, 0.86, c)) * 0.55 + 0.06;
            a *= smoothstep(1.0, 0.85, r);
            gl_FragColor = vec4(uColor * a, a); }`,
      });
      this.chevronMats.push(chev);
      const disc = new THREE.Mesh(new THREE.CircleGeometry(1.75, 48), chev);
      const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.4, 30, 12, 1, true).translate(0, 15, 0), new THREE.MeshBasicMaterial({ color: hdr(col, 0.6), transparent: true, opacity: 0.18, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
      pillar.position.y = 1.9;
      // the ring's own frame: x across the line, y up, z along the line (the way through)
      const inner = new THREE.Group(); inner.add(hoop, outer, disc, pillar);
      inner.rotation.x = Math.PI / 2; // y-up shapes into the z-up world
      g.add(inner);
      g.position.set(p.x, p.y, p.z + 1.95);
      g.rotation.z = p.yaw - Math.PI / 2; // hoop plane faces along the line
    } else {
      const shell = fresnelMaterial(hdr(col, 2.2), { hex: p.kind === 'shield' });
      this.fresnels.push(shell);
      g.add(new THREE.Mesh(new THREE.IcosahedronGeometry(0.62, 3), shell));
      const core = new THREE.MeshStandardMaterial({ color: col, emissive: hdr(col, 1), emissiveIntensity: 2.2, roughness: 0.15, metalness: 0.6, envMap: env });
      if (p.kind === 'shield') {
        const icon = new THREE.Mesh(new THREE.ExtrudeGeometry(shieldShape(), { depth: 0.06, bevelEnabled: true, bevelSize: 0.02, bevelThickness: 0.02, bevelSegments: 2 }).translate(0, 0, -0.03), core);
        icon.rotation.x = Math.PI / 2; g.add(icon);
      } else {
        const horse = new THREE.Group();
        horse.add(new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.07, 10, 24, Math.PI), core));
        for (const sx of [-0.2, 0.2]) {
          const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.2, 10), core); leg.position.set(sx, -0.1, 0); horse.add(leg);
          const tip = new THREE.Mesh(new THREE.CylinderGeometry(0.072, 0.072, 0.07, 10), new THREE.MeshBasicMaterial({ color: hdr('#ffffff', 2.5) })); tip.position.set(sx, -0.23, 0); horse.add(tip);
        }
        horse.rotation.x = Math.PI / 2; horse.position.z = 0.05; g.add(horse);
      }
      g.position.set(p.x, p.y, p.z);
    }
    g.userData.baseZ = g.position.z;
    this.group.add(g);
    return g;
  }

  /** Every pickup back for a new run. */
  reset() { for (const p of this.pickups) { p.taken = false; if (p.obj) p.obj.visible = true; } }

  /** Skier at (x, y, z) this frame: returns what was picked up. */
  collect(x: number, y: number, z: number): Pickup[] {
    const out: Pickup[] = [];
    for (const p of this.pickups) {
      if (p.taken) continue;
      const dx = x - p.x, dy = y - p.y, r = p.kind === 'ring' ? 2.0 : p.kind === 'crystal' ? 1.25 : 1.5;
      if (dx * dx + dy * dy > r * r || Math.abs(z + 0.8 - p.z) > (p.kind === 'ring' ? 3.5 : 2.2)) continue;
      p.taken = true; out.push(p);
      if (p.obj) p.obj.visible = false;
      this.shockwave(p);
    }
    return out;
  }

  private shockwave(p: Pickup) {
    const w = this.waves.find(v => v.t >= 1) ?? this.waves[0];
    w.t = 0; w.mesh.visible = true;
    (w.mesh.material as THREE.MeshBasicMaterial).color.copy(hdr(POWER_COLOR[p.kind], 2.5));
    w.mesh.position.set(p.x, p.y, p.kind === 'ring' ? p.z + 1.95 : p.z);
    w.mesh.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), p.yaw).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2));
    w.mesh.userData.big = p.kind === 'ring' ? 2.2 : 1;
  }

  update(dt: number, camX: number, camY: number, camZ: number) {
    this.t += dt;
    const t = this.t, { m, q, p, s, z } = this;
    for (const f of this.fresnels) f.uniforms.uTime.value = t;
    for (const c of this.chevronMats) c.uniforms.uTime.value = t;
    // tokens bob and turn; rings breathe
    for (const pk of this.pickups) {
      if (!pk.obj || pk.taken) continue;
      if (pk.kind === 'ring') { const k = 1 + 0.03 * Math.sin(t * 3 + pk.ph); pk.obj.scale.setScalar(k); }
      else { pk.obj.position.z = pk.obj.userData.baseZ + Math.sin(t * 2.2 + pk.ph) * 0.12; pk.obj.rotation.z = t * 1.6 + pk.ph; }
    }
    if (this.crystals) {
      this.crystalIdx.forEach((pi, i) => {
        const pk = this.pickups[pi];
        q.setFromAxisAngle(z, t * 2.4 + pk.ph);
        p.set(pk.x, pk.y, pk.z + Math.sin(t * 2.6 + pk.ph) * 0.1);
        s.setScalar(pk.taken ? 0 : 1);
        m.compose(p, q, s); this.crystals!.setMatrixAt(i, m);
      });
      this.crystals.instanceMatrix.needsUpdate = true;
    }
    // halos pulse on the snow, a little brighter when you are near
    this.pickups.forEach((pk, i) => {
      const near = Math.max(0, 1 - Math.hypot(camX - pk.x, camY - pk.y) / 40);
      const k = pk.taken ? 0 : (pk.kind === 'ring' ? 2.6 : pk.kind === 'crystal' ? 0.55 : 1.2) * (1 + 0.12 * Math.sin(t * 4 + pk.ph) + 0.15 * near);
      const sm = this.course.sample(pk.x, pk.y);
      p.set(pk.x, pk.y, sm.h + 0.04); s.setScalar(k);
      q.setFromUnitVectors(z, new THREE.Vector3(sm.n[0], sm.n[1], sm.n[2]));
      m.compose(p, q, s); this.halos.setMatrixAt(i, m);
    });
    this.halos.instanceMatrix.needsUpdate = true;
    void camZ;
    // shockwaves expand and fade
    for (const w of this.waves) {
      if (w.t >= 1) continue;
      w.t = Math.min(1, w.t + dt * 2.2);
      const k = (0.6 + w.t * 3.2) * (w.mesh.userData.big ?? 1);
      w.mesh.scale.set(k, k, k);
      (w.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - w.t) * (1 - w.t);
      if (w.t >= 1) w.mesh.visible = false;
    }
  }
}

/** The shield bubble that rides on the robot while a shield is held. */
export function shieldBubble() {
  const mat = fresnelMaterial(hdr(POWER_COLOR.shield, 1.8), { hex: true, alpha: 0.9 });
  const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 4), mat);
  mesh.scale.set(0.85, 0.85, 1.05);
  mesh.visible = false;
  return { mesh, mat };
}

/** A violet aura on the snow under the skier while the magnet is on. */
export function magnetAura() {
  const mat = new THREE.MeshBasicMaterial({ color: hdr(POWER_COLOR.magnet, 1.6), transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
  const g = new THREE.Group();
  const a = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.08, 64), mat);
  const b = new THREE.Mesh(new THREE.RingGeometry(1.25, 1.3, 6, 1), mat); // a hexagon turning the other way
  g.add(a, b); g.visible = false;
  return { group: g, inner: a, outer: b, mat };
}

/** Where a clean skier goes: the autopilot's line through every gate, one sample per ~0.5 m along the course. */
export function racingLine(course: Course) {
  const s = initialState(course), gt = new GateTracker(course), cl = course.meta.centerline;
  const out: { x: number; y: number; s: number; h: number }[] = [];
  const inp = { lean: 0, crouch: 0, jump: false, push: 0 };
  const dt = 1 / 60;
  for (let t = 0; t < 240; t += dt) {
    let tx = cl[Math.min(cl.length - 1, s.clIndex + 10)][0], ty = cl[Math.min(cl.length - 1, s.clIndex + 10)][1];
    if (Math.abs(s.lateral) > 12) { const b = cl[Math.min(cl.length - 1, s.clIndex + 5)]; tx = b[0]; ty = b[1]; }
    else if (!gt.done) { const g = course.gates[gt.next]; if (g.s - s.sAlong < 45) { tx = (g.turn_pole[0] + g.outer_pole[0]) / 2; ty = (g.turn_pole[1] + g.outer_pole[1]) / 2; } }
    let want = Math.atan2(ty - s.y, tx - s.x);
    inp.jump = false;
    if (s.speed < 1.5 && !s.air) { const [gx, gy] = course.grad(s.x, s.y); const fall = Math.atan2(-gy, -gx); if (Math.cos(fall - want) < 0.1) want = fall; inp.jump = true; }
    const err = Math.atan2(Math.sin(want - s.heading), Math.cos(want - s.heading));
    inp.lean = Math.max(-1, Math.min(1, -err * 2.5 * Math.min(1, 12 / Math.max(6, s.speed))));
    inp.crouch = Math.abs(inp.lean) < 0.25 ? 0.9 : 0.3; inp.push = Math.max(0, (inp.crouch - 0.45) / 0.55);
    step(s, inp, dt, course);
    gt.update(s.x, s.y, s.sAlong);
    if (!out.length || s.sAlong - out[out.length - 1].s > 0.5) out.push({ x: s.x, y: s.y, s: s.sAlong, h: Math.atan2(s.vy, s.vx) });
    if (gt.crossedFinish(s.x, s.y) && s.progress > 0.9) break;
  }
  return out;
}
