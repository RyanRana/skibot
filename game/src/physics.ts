// Ski physics on a heightfield, simple enough to tune by feel and honest enough to carve.
// Skis point along `heading`. Velocity across the skis is killed by edge grip, velocity along them
// is driven by gravity down the slope minus glide friction and air drag. An edged ski carves a
// radius that shrinks with edge angle (sidecut x cos(edge) in the full sim), so turn rate grows
// with lean and with speed, exactly like real carving.
import { Course, gateSide, gateSpan, type Vec3 } from './course.ts';

export const G = 9.81;

export interface Input {
  lean: number;    // -1 (left) .. 1 (right)
  crouch: number;  // 0 standing .. 1 full tuck
  jump: boolean;   // edge-triggered hop
}

export interface SkierState {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  heading: number;      // radians CCW from +x, where the skis point
  speed: number;        // along-ski speed (m/s)
  edge: number;         // signed edge angle (radians), + = right edge (turning right)
  air: boolean;
  airTime: number;
  slip: number;         // across-ski speed being scrubbed off (m/s), for spray and sound
  normal: Vec3;
  slopeDeg: number;
  crashed: number;      // seconds left in a wipeout, 0 = skiing
  skateCd: number;      // seconds until the next skating push is allowed
  clIndex: number;      // nearest centerline sample
  sAlong: number;       // metres along the full centerline
  progress: number;     // 0..1 from the race start to the finish
  lateral: number;      // signed metres off the centerline
  offPiste: boolean;
}

export interface Tuning {
  sidecut: number;      // metres (robot scale GS ski)
  maxEdge: number;      // radians
  mu: number;           // glide friction
  dragStand: number;    // v^2 drag coefficients (per metre)
  dragTuck: number;
  gripTau: number;      // seconds to scrub across-ski velocity
  pivotRate: number;    // rad/s of skidded steering available at low speed
  jumpV: number;        // m/s
  pisteHalfWidth: number;
}

export const DEFAULT_TUNING: Tuning = {
  sidecut: 14, maxEdge: 1.05, mu: 0.04, dragStand: 0.014, dragTuck: 0.006, gripTau: 0.12, pivotRate: 1.3, jumpV: 3.4, pisteHalfWidth: 22,
};

export function initialState(course: Course, push = 3.5): SkierState {
  const i = course.startIndex, p = course.meta.centerline[i];
  const yaw = course.headingAt(i);
  const s = course.sample(p[0], p[1]);
  // A racer leaves the start with a push.
  return {
    x: p[0], y: p[1], z: s.h, vx: push * Math.cos(yaw), vy: push * Math.sin(yaw), vz: 0, heading: yaw, speed: push, edge: 0,
    air: false, airTime: 0, slip: 0, normal: [...s.n] as Vec3, slopeDeg: course.slopeDeg(p[0], p[1]),
    crashed: 0, skateCd: 0, clIndex: i, sAlong: p[3], progress: 0, lateral: 0, offPiste: false,
  };
}

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** Advance the skier by dt seconds. Mutates and returns `s`. */
export function step(s: SkierState, inp: Input, dt: number, course: Course, tune: Tuning = DEFAULT_TUNING) {
  dt = Math.min(dt, 0.05);
  const smp = course.sample(s.x, s.y);
  s.normal = smp.n; s.slopeDeg = Math.acos(smp.n[2]) * 180 / Math.PI;

  if (s.crashed > 0) {
    s.crashed = Math.max(0, s.crashed - dt);
    const k = Math.exp(-4 * dt);
    s.vx *= k; s.vy *= k; s.speed = Math.hypot(s.vx, s.vy); s.edge *= k; s.slip = 0;
    s.x += s.vx * dt; s.y += s.vy * dt; s.z = course.sample(s.x, s.y).h;
    if (s.crashed === 0) { s.heading = course.headingAt(s.clIndex); const v = 3; s.vx = v * Math.cos(s.heading); s.vy = v * Math.sin(s.heading); s.speed = v; }
    track(s, course, tune);
    return s;
  }

  // Edge angle follows the lean with a little lag (legs take time to angulate).
  const targetEdge = Math.max(-1, Math.min(1, inp.lean)) * tune.maxEdge;
  s.edge += (targetEdge - s.edge) * (1 - Math.exp(-dt / 0.09));

  const hx = Math.cos(s.heading), hy = Math.sin(s.heading);
  let vAlong = s.vx * hx + s.vy * hy;
  let vAcross = -s.vx * hy + s.vy * hx; // + = to the skier's left

  if (!s.air) {
    // Gravity along the surface, projected on the skis.
    const [gx, gy] = course.grad(s.x, s.y);
    const denom = 1 + gx * gx + gy * gy;
    const ax = -G * gx / denom, ay = -G * gy / denom;
    const aAlong = ax * hx + ay * hy, aAcross = -ax * hy + ay * hx;
    const drag = tune.dragStand + (tune.dragTuck - tune.dragStand) * Math.max(0, Math.min(1, inp.crouch));
    const nForce = G * smp.n[2];
    const friction = tune.mu * nForce * (1 + 0.6 * Math.abs(Math.sin(s.edge))); // an edged ski glides a little worse
    const sgn = vAlong > 0.05 ? 1 : vAlong < -0.05 ? -1 : 0;
    // Skis pointed uphill: a skier sits back and digs in rather than sliding backwards.
    const fr = vAlong < 0 ? friction * 6 : friction;
    vAlong += (aAlong - sgn * fr - drag * vAlong * Math.abs(vAlong)) * dt;
    if (vAlong < 0 && aAlong > -fr) vAlong = Math.min(0, vAlong + fr * dt);
    if (Math.abs(vAlong) < 0.05 && Math.abs(aAlong) < friction) vAlong = 0;
    // Across the skis: gravity pushes, the edge holds. Grip is weaker when the ski is flat.
    const grip = tune.gripTau * (1.3 - 0.8 * Math.abs(Math.sin(s.edge)));
    vAcross += aAcross * dt;
    const scrub = vAcross * (1 - Math.exp(-dt / grip));
    vAcross -= scrub;
    s.slip = Math.abs(scrub) / dt;
    // Carving: turn rate = v * tan(edge) / sidecut, toward the edged side. At low speed the skier pivots
    // the skis instead (rotary steering), which fades out as carving takes over.
    const lean = Math.max(-1, Math.min(1, inp.lean));
    const pivot = lean * tune.pivotRate * Math.max(0, 1 - vAlong / 7);
    const turn = (vAlong * Math.tan(s.edge)) / tune.sidecut + pivot;
    s.heading = wrap(s.heading - turn * dt);
    // Off piste: deep snow drags.
    if (s.offPiste) vAlong -= vAlong * 0.9 * dt;
    // Hop at speed; at a near standstill the same move is a skating push.
    s.skateCd = Math.max(0, s.skateCd - dt);
    if (inp.jump && vAlong > 2) { s.air = true; s.airTime = 0; s.vz = tune.jumpV + Math.abs(s.vz); }
    else if (inp.jump && s.skateCd === 0) { vAlong += 1.6; s.skateCd = 0.7; }
    // Wipeout: too much sideways speed at an aggressive edge angle.
    if (s.slip > 9.5 && Math.abs(s.edge) > 0.95 && vAlong > 13) { s.crashed = 1.0; }
  } else {
    s.airTime += dt;
    s.slip = 0;
  }

  const nhx = Math.cos(s.heading), nhy = Math.sin(s.heading);
  s.vx = vAlong * nhx - vAcross * nhy;
  s.vy = vAlong * nhy + vAcross * nhx;
  s.speed = Math.max(0, vAlong);

  // Vertical: follow the snow, or fly.
  const nx = s.x + s.vx * dt, ny = s.y + s.vy * dt;
  const ground = course.sample(nx, ny).h;
  if (s.air) {
    s.vz -= G * dt;
    const nz = s.z + s.vz * dt;
    if (nz <= ground) {
      s.air = false; s.z = ground;
      if (s.vz < -9 && Math.abs(s.edge) > 0.8) s.crashed = 1.0; // hard landing on edge
      s.vz = 0;
    } else s.z = nz;
  } else {
    const dz = ground - s.z;
    const vzTerrain = dz / dt;
    // Airborne when the ground falls away faster than free fall would carry the skier, by more than the
    // legs can absorb (a roller or a jump, not a change of pitch at a grid cell).
    const zFree = s.z + s.vz * dt - 0.5 * G * dt * dt;
    if (zFree - ground > 0.35 && s.speed > 5) { s.air = true; s.airTime = 0; s.vz = Math.max(s.vz, -1.5); s.z = Math.max(ground, s.z + s.vz * dt); }
    else { s.z = ground; s.vz = s.vz + (vzTerrain - s.vz) * Math.min(1, dt * 18); }
  }
  if (course.inside(nx, ny)) { s.x = nx; s.y = ny; }
  else { s.vx = s.vy = 0; s.speed = 0; }

  track(s, course, tune);
  return s;
}

function track(s: SkierState, course: Course, tune: Tuning) {
  const n = course.nearest(s.x, s.y, s.clIndex);
  s.clIndex = n.index;
  s.sAlong = n.s;
  s.progress = Math.max(0, (n.s - course.startS) / course.raceLength);
  s.lateral = course.lateral(s.x, s.y, n.index);
  s.offPiste = Math.abs(s.lateral) > tune.pisteHalfWidth;
}

/** Gate bookkeeping: which gates were made, which were missed, which is next. */
export class GateTracker {
  next = 0;
  hit: boolean[] = [];
  missed: boolean[] = [];
  lastSide: number[] = [];
  private course: Course;
  constructor(course: Course) {
    this.course = course;
    for (const g of course.gates) { this.hit.push(false); this.missed.push(false); this.lastSide.push(1); }
  }
  get total() { return this.course.gates.length; }
  get made() { return this.hit.filter(Boolean).length; }
  get done() { return this.next >= this.total; }
  /** Returns 'hit' | 'miss' | null for this step. */
  update(x: number, y: number, s: number): 'hit' | 'miss' | null {
    let result: 'hit' | 'miss' | null = null;
    const gates = this.course.gates;
    // Check the next two gates so a wide line that skips one still registers the one after.
    for (let k = this.next; k < Math.min(this.next + 2, this.total); k++) {
      const g = gates[k];
      const side = gateSide(g, x, y);
      if (this.lastSide[k] > 0 && side <= 0) {
        const u = gateSpan(g, x, y);
        if (u > -0.45 && u < 1.45) {
          this.hit[k] = true;
          for (let j = this.next; j < k; j++) if (!this.hit[j]) this.missed[j] = true;
          this.next = k + 1; result = 'hit';
          this.lastSide[k] = side; break;
        } else if (k === this.next && u > -3 && u < 4) {
          // Crossed the gate line just outside the poles: a straddle or a near miss. Missed, move on.
          this.missed[k] = true; this.next = k + 1; result = 'miss';
          this.lastSide[k] = side; break;
        }
      }
      this.lastSide[k] = side;
    }
    // Passed well below the next gate without crossing between its poles: a miss.
    if (result === null && this.next < this.total && s > gates[this.next].s + 9) {
      this.missed[this.next] = true; this.next++; result = 'miss';
    }
    return result;
  }
  crossedFinish(x: number, y: number) {
    const f = this.course.finish;
    const d = (x - f.poles[0][0]) * f.line_normal[0] + (y - f.poles[0][1]) * f.line_normal[1];
    return d <= 0;
  }
}
