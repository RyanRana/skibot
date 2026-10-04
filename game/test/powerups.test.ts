// The attract-mode autopilot with powerups, no browser: pickups sit on the simulated racing line, so a clean
// run should pass through every ring and still make every gate. Mirrors main.ts (autopilotInput, onPickup,
// tuckCharge) closely enough to tune them.
// Run: node --experimental-strip-types test/powerups.test.ts
import { readFileSync } from 'node:fs';
import { Course, type CourseMeta } from '../src/course.ts';
import { initialState, step, GateTracker, type Input } from '../src/physics.ts';
import { Powerups } from '../src/powerups.ts';
import * as THREE from 'three';
const dir = new URL('../public/courses/kitzbuhel-streif', import.meta.url).pathname;
const meta = JSON.parse(readFileSync(dir + '/course.json', 'utf8')) as CourseMeta;
const buf = readFileSync(dir + '/heights.bin');
const course = new Course(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const cl = meta.centerline;

{
  const pw = new Powerups(new THREE.Scene(), course, null);
  const s = initialState(course), gt = new GateTracker(course);
  let t = 0, crashes = 0, wasCrash = false, boostT = 0, tuckHold = 0, tuckLevel = 0, tuckBoostT = 0, tuckFired = 0, bonus = 0, fires = 0;
  const got: Record<string, number> = {};
  const minD = new Map<any, number>(), dz = new Map<any, number>();
  const dt = 1 / 60;
  const inp: Input = { lean: 0, crouch: 0, jump: false };
  while (t < 200) {
    // autopilot (main.ts)
    const look = cl[Math.min(cl.length - 1, s.clIndex + 10)];
    let tx = look[0], ty = look[1];
    if (Math.abs(s.lateral) > 12) { const b = cl[Math.min(cl.length - 1, s.clIndex + 5)]; tx = b[0]; ty = b[1]; }
    else if (!gt.done) {
      const g = course.gates[gt.next]; if (g.s - s.sAlong < 45) { tx = (g.turn_pole[0] + g.outer_pole[0]) / 2; ty = (g.turn_pole[1] + g.outer_pole[1]) / 2; }
    }
    let want = Math.atan2(ty - s.y, tx - s.x);
    inp.jump = false;
    if (s.speed < 1.5 && !s.air) { const [gx, gy] = course.grad(s.x, s.y); const fall = Math.atan2(-gy, -gx); if (Math.cos(fall - want) < 0.1) want = fall; inp.jump = true; }
    const err = Math.atan2(Math.sin(want - s.heading), Math.cos(want - s.heading));
    inp.lean = Math.max(-1, Math.min(1, -err * 2.5 * Math.min(1, 12 / Math.max(6, s.speed))));
    inp.crouch = Math.abs(inp.lean) < 0.25 ? 0.9 : 0.3;
    inp.push = Math.max(0, (inp.crouch - 0.45) / 0.55);
    // tuck charge (main.ts)
    const tucked = inp.crouch > 0.7 && Math.abs(inp.lean) < 0.3 && !s.air && s.crashed === 0 && s.speed > 6;
    if (tucked) { tuckHold += dt; tuckLevel = Math.max(tuckLevel, Math.min(3, Math.floor(tuckHold / 0.7))); }
    else if (tuckLevel > 0 && inp.crouch < 0.5 && Math.abs(inp.lean) < 0.6) { tuckFired = tuckLevel; tuckBoostT = 1.2; tuckHold = 0; tuckLevel = 0; fires++; }
    else if (!tucked) { tuckHold = Math.max(0, tuckHold - dt * 2); tuckLevel = Math.min(tuckLevel, Math.floor(tuckHold / 0.7)); }
    inp.boost = (boostT > 0 ? 0.6 : 0) + (tuckBoostT > 0 ? 0.15 * tuckFired : 0);
    step(s, inp, dt, course);
    boostT = Math.max(0, boostT - dt); tuckBoostT = Math.max(0, tuckBoostT - dt);
    gt.update(s.x, s.y, s.sAlong);
    for (const p of pw.pickups) { const d = Math.hypot(p.x - s.x, p.y - s.y); minD.set(p, Math.min(minD.get(p) ?? 1e9, d)); if (d < 3) dz.set(p, s.z + 0.8 - p.z); }
    for (const p of pw.collect(s.x, s.y, s.z)) {
      got[p.kind] = (got[p.kind] ?? 0) + 1;
      if (p.kind === 'ring') { boostT = 2; s.vx += Math.cos(s.heading) * 0.8; s.vy += Math.sin(s.heading) * 0.8; }
      if (p.kind === 'crystal') bonus += 100;
      if (p.kind === 'magnet') gt.assist = 3;
    }
    if (s.crashed > 0 && !wasCrash) crashes++; wasCrash = s.crashed > 0;
    t += dt;
    if (gt.crossedFinish(s.x, s.y) && s.progress > 0.9) break;
  }
  if (process.env.DBG) console.log(pw.pickups.map(p => `${p.kind[0]}${minD.get(p)?.toFixed(1)}/${dz.get(p)?.toFixed(1)}`).join(' '));
  const total = pw.pickups.reduce((a, p) => (a[p.kind] = (a[p.kind] ?? 0) + 1, a), {} as Record<string, number>);
  console.log(`time ${t.toFixed(1)}s gates ${gt.made}/${gt.total} crashes ${crashes} tuck fires ${fires} bonus ${bonus}ms got ${JSON.stringify(got)} of ${JSON.stringify(total)}`);
}
