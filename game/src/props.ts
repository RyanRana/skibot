// Procedural models for the mountain: snow-laden spruces with drooping branch tiers, snowy rocks, a 3D
// crowd, snow guns and flags. Everything is built from code (no downloaded assets), z-up, unit-sized,
// and carries per-vertex `color` (base colour with ambient occlusion baked in) and `aSnow` (0..1, how much
// snow sits there), which the shared snowy material turns into clumpy snow over needles, bark or stone.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Give a geometry flat per-vertex colour and snow so it can be merged with the rest of a model. */
function paint(g: THREE.BufferGeometry, color: THREE.Color | ((i: number, p: THREE.Vector3, n: THREE.Vector3) => [number, number, number]), snow: number | ((i: number, p: THREE.Vector3, n: THREE.Vector3) => number)) {
  const geo = g.index ? g.toNonIndexed() : g;
  geo.deleteAttribute('uv');
  if (!geo.attributes.normal) geo.computeVertexNormals();
  const n = geo.attributes.position.count;
  const col = new Float32Array(n * 3), sn = new Float32Array(n);
  const p = new THREE.Vector3(), nr = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    p.fromBufferAttribute(geo.attributes.position as THREE.BufferAttribute, i);
    nr.fromBufferAttribute(geo.attributes.normal as THREE.BufferAttribute, i);
    const c = typeof color === 'function' ? color(i, p, nr) : [color.r, color.g, color.b];
    col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
    sn[i] = typeof snow === 'function' ? snow(i, p, nr) : snow;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aSnow', new THREE.BufferAttribute(sn, 1));
  return geo;
}

/**
 * A Norway spruce, 1 unit tall, base at z = 0. Tiers of drooping branches whose tips poke out in a ragged
 * star, each tier loaded with snow on top and dark needles underneath, a closed underside so a low camera
 * never sees inside, a bare leader at the top and a bit of trunk at the bottom.
 */
export function spruceGeometry(seed: number, o: { tiers?: number; width?: number; seg?: number; snow?: number } = {}) {
  const R = rng(seed);
  const tiers = o.tiers ?? 7, width = o.width ?? 0.36, seg = o.seg ?? 12, snowLoad = o.snow ?? 1;
  const pos: number[] = [], col: number[] = [], snow: number[] = [], idx: number[] = [];
  const needleDark = [0.035, 0.075, 0.055], needleTip = [0.07, 0.13, 0.085], under = [0.02, 0.04, 0.03];
  const push = (x: number, y: number, z: number, c: number[], s: number, ao: number) => {
    pos.push(x, y, z); col.push(c[0] * ao, c[1] * ao, c[2] * ao); snow.push(s); return pos.length / 3 - 1;
  };
  for (let t = 0; t < tiers; t++) {
    const u = t / tiers;
    const zb = 0.1 + u * 0.8, zt = Math.min(0.98, zb + 0.27 * (1 - 0.45 * u));
    const r = width * Math.pow(1 - u * 0.88, 0.95) * (0.9 + 0.2 * R());
    const rot = R() * Math.PI * 2, ao = 0.55 + 0.45 * u; // lower tiers sit in the shade of the ones above
    const load = Math.min(1, snowLoad * (0.75 + 0.35 * R()));
    const top: number[] = [], mid: number[] = [], bot: number[] = [];
    for (let k = 0; k < seg; k++) {
      const a = rot + (k / seg) * Math.PI * 2;
      const tip = k % 2 === 0, jag = tip ? 1 + 0.12 * R() : 0.72 + 0.1 * R();
      const ca = Math.cos(a), sa = Math.sin(a);
      top.push(push(ca * r * 0.06, sa * r * 0.06, zt, needleDark, load, ao));
      const rm = r * 0.58 * (tip ? 1.04 : 0.96);
      mid.push(push(ca * rm, sa * rm, zt - (zt - zb) * 0.42 + (tip ? 0 : 0.01), needleDark, load * (tip ? 0.95 : 0.8), ao));
      // tips droop under the snow, the gaps between them sit higher and darker
      const rb = r * jag, droop = tip ? 0.035 + 0.02 * R() : -0.01;
      bot.push(push(ca * rb, sa * rb, zb - droop, tip ? needleTip : needleDark, tip ? load * 0.35 : 0.0, tip ? ao * 1.1 : ao * 0.8));
    }
    for (let k = 0; k < seg; k++) {
      const k1 = (k + 1) % seg;
      idx.push(top[k], mid[k], mid[k1], top[k], mid[k1], top[k1]);
      idx.push(mid[k], bot[k], bot[k1], mid[k], bot[k1], mid[k1]);
    }
    // underside: a shallow dark cone up to the trunk
    const c = push(0, 0, zb + (zt - zb) * 0.35, under, 0, 1);
    const ub: number[] = [];
    for (let k = 0; k < seg; k++) { const a = rot + (k / seg) * Math.PI * 2, rb = r * (k % 2 === 0 ? 1.0 : 0.74); ub.push(push(Math.cos(a) * rb, Math.sin(a) * rb, zb - (k % 2 === 0 ? 0.035 : -0.01), under, 0, 1)); }
    for (let k = 0; k < seg; k++) idx.push(c, ub[(k + 1) % seg], ub[k]);
  }
  const foliage = new THREE.BufferGeometry();
  foliage.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  foliage.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  foliage.setAttribute('aSnow', new THREE.Float32BufferAttribute(snow, 1));
  foliage.setIndex(idx);
  foliage.computeVertexNormals();
  const fol = foliage.toNonIndexed(); // smooth normals are kept from the indexed pass
  const trunk = new THREE.CylinderGeometry(0.012, 0.032, 0.5, 6, 1, true).rotateX(Math.PI / 2).translate(0, 0, 0.2);
  const leader = new THREE.ConeGeometry(0.018, 0.1, 5).rotateX(Math.PI / 2).translate(0, 0, 1.0);
  return mergeGeometries([
    fol,
    paint(trunk, new THREE.Color(0.11, 0.07, 0.045), 0),
    paint(leader, new THREE.Color(0.05, 0.1, 0.07), 0.6),
  ])!;
}

/** A frost-shattered boulder, unit radius, snow on whatever faces up. */
export function rockGeometry(seed: number) {
  const R = rng(seed);
  const g = new THREE.IcosahedronGeometry(1, 2);
  const p = g.attributes.position as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  const f1 = new THREE.Vector3(R() - 0.5, R() - 0.5, R() - 0.5).normalize(), f2 = new THREE.Vector3(R() - 0.5, R() - 0.5, R() - 0.5).normalize();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    // a couple of cleavage planes and lumpy noise, then squash it into the snow
    let k = 1 + 0.18 * Math.sin(v.x * 3.1 + seed) * Math.cos(v.y * 2.7 - seed) + 0.1 * Math.sin(v.z * 5.3 + v.x * 2.0);
    k -= 0.25 * Math.max(0, v.dot(f1) - 0.55) + 0.2 * Math.max(0, v.dot(f2) - 0.6);
    v.multiplyScalar(k); v.z = v.z * 0.62 + 0.15;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  const tone = 0.13 + R() * 0.06;
  return paint(g, (_i, pp) => { const ao = 0.6 + 0.4 * THREE.MathUtils.clamp((pp.z + 0.3) / 1.0, 0, 1); return [tone * ao, tone * 0.98 * ao, tone * 1.03 * ao]; }, (_i, _pp, n) => THREE.MathUtils.smoothstep(n.z, 0.35, 0.75));
}

/** Spectator parts, 1.75 m tall person in metres: jacket body, head, beanie with a pompom, one arm (used twice). */
export function personGeometries() {
  const body = new THREE.CapsuleGeometry(0.2, 0.62, 4, 10).rotateX(Math.PI / 2).translate(0, 0, 0.92);
  const legs = new THREE.CapsuleGeometry(0.15, 0.55, 3, 8).rotateX(Math.PI / 2).translate(0, 0, 0.42).scale(1.15, 0.8, 1);
  const head = new THREE.SphereGeometry(0.12, 12, 10).translate(0, 0, 1.47);
  const hat = mergeGeometries([
    new THREE.SphereGeometry(0.128, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.55).translate(0, 0, 1.5),
    new THREE.SphereGeometry(0.05, 8, 6).translate(0, 0, 1.64),
  ])!;
  const arm = new THREE.CapsuleGeometry(0.055, 0.5, 3, 6).rotateX(Math.PI / 2).translate(0, 0, -0.27); // hangs from the shoulder pivot
  return { body, legs, head, hat, arm };
}

/** A snow gun on a tilted lance: the tall kind that stands along the edges of a race piste. */
export function snowLanceGeometry() {
  const pole = new THREE.CylinderGeometry(0.04, 0.06, 7, 8).rotateX(Math.PI / 2).translate(0, 0, 3.5);
  const head = new THREE.CylinderGeometry(0.13, 0.11, 0.55, 10).rotateX(Math.PI / 2).translate(0, 0, 7.1);
  const nozzle = new THREE.CylinderGeometry(0.05, 0.08, 0.22, 8).translate(0, 0.32, 7.05);
  const base = new THREE.CylinderGeometry(0.22, 0.28, 0.3, 10).rotateX(Math.PI / 2).translate(0, 0, 0.15);
  const g = mergeGeometries([
    paint(pole, new THREE.Color(0.55, 0.57, 0.6), 0),
    paint(head, new THREE.Color(0.85, 0.32, 0.05), 0.15),
    paint(nozzle, new THREE.Color(0.08, 0.08, 0.09), 0),
    paint(base, new THREE.Color(0.25, 0.26, 0.28), 0.4),
  ])!;
  g.rotateY(0.32); // lances lean out over the piste
  return g;
}

/** A unit flag (1.5 x 1, hoist edge at x = 0) with enough segments to wave. */
export function flagGeometry() {
  return new THREE.PlaneGeometry(1.5, 1, 10, 4).translate(0.75, 0, 0).rotateX(Math.PI / 2);
}
