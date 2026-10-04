// Autopilot down the real course, no browser: checks the carving model finishes, how fast, and that nothing explodes.
// Run: node --experimental-strip-types test/physics.test.ts
import { readFileSync } from 'node:fs';
import { Course, type CourseMeta } from '../src/course.ts';
import { initialState, step, GateTracker } from '../src/physics.ts';
const dir = new URL('../public/courses/kitzbuhel-streif', import.meta.url).pathname;
const meta = JSON.parse(readFileSync(dir + '/course.json', 'utf8')) as CourseMeta;
const buf = readFileSync(dir + '/heights.bin');
const course = new Course(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
// slope profile along the centerline per 100 m
const cl = meta.centerline; const prof: string[] = [];
for (let s0 = 0; s0 < course.length; s0 += 100) { const i = cl.findIndex(p => p[3] >= s0), j = cl.findIndex(p => p[3] >= s0 + 100); if (i < 0 || j < 0) break; prof.push(`${s0}:${(Math.atan2(cl[i][2] - cl[j][2], 100) * 180 / Math.PI).toFixed(0)}°`); }
console.log('along-course slope per 100 m:', prof.join(' '));
for (const mode of ['centerline', 'gates', 'straight']) {
  const s = initialState(course); const gt = new GateTracker(course);
  let t = 0, maxV = 0, finished = false, airs = 0, wasAir = false, crashes = 0, wasCrash = false, maxLat = 0;
  const dt = 1 / 60;
  while (t < 300) {
    let lean = 0, jump = false;
    if (mode !== 'straight') {
      let tx: number, ty: number;
      const look = cl[Math.min(cl.length - 1, s.clIndex + 10)];
      tx = look[0]; ty = look[1];
      if (Math.abs(s.lateral) > 12) { const b = cl[Math.min(cl.length - 1, s.clIndex + 5)]; tx = b[0]; ty = b[1]; }
      else if (mode === 'gates' && !gt.done) { const g = course.gates[gt.next]; if (g.s - s.sAlong < 45) { tx = (g.turn_pole[0] + g.outer_pole[0]) / 2; ty = (g.turn_pole[1] + g.outer_pole[1]) / 2; } }
      let want = Math.atan2(ty - s.y, tx - s.x);
      if (s.speed < 1.5 && !s.air) { const [gx, gy] = course.grad(s.x, s.y); const fall = Math.atan2(-gy, -gx); if (Math.cos(fall - want) < 0.1) want = fall; jump = true; }
      const err = Math.atan2(Math.sin(want - s.heading), Math.cos(want - s.heading));
      lean = Math.max(-1, Math.min(1, -err * 2.5 * Math.min(1, 12 / Math.max(6, s.speed))));
    }
    step(s, { lean, crouch: Math.abs(lean) < 0.25 ? 0.9 : 0.3, jump }, dt, course); jump = false;
    gt.update(s.x, s.y, s.sAlong);
    maxV = Math.max(maxV, s.speed); t += dt; maxLat = Math.max(maxLat, Math.abs(s.lateral));
    if (s.air && !wasAir) airs++; wasAir = s.air;
    if (s.crashed > 0 && !wasCrash) crashes++; wasCrash = s.crashed > 0;
    if (gt.crossedFinish(s.x, s.y) && s.progress > 0.9) { finished = true; break; }
  }
  console.log(mode.padEnd(10), `time ${t.toFixed(1)}s finished=${finished} gates ${gt.made}/${gt.total} missed ${gt.missed.filter(Boolean).length} max ${(maxV * 3.6).toFixed(0)} km/h airs ${airs} crashes ${crashes} progress ${s.progress.toFixed(2)} maxLateral ${maxLat.toFixed(0)} end (${s.x.toFixed(0)},${s.y.toFixed(0)})`);
}
