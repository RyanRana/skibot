// MediaPipe-shaped synthetic poses through the G1 retarget: finite joints, knees pass through, lean flips hip roll.
// Run: node --experimental-strip-types test/retarget.test.ts
import { retarget, stanceFrom } from '../src/g1.ts';
// Synthetic MediaPipe world landmarks (metres, hips at origin, y DOWN, z toward... MediaPipe world: x right(of image), y down, z toward camera negative).
function body(knee = 0, lean = 0, crouchDrop = 0) {
  const P: { x: number; y: number; z: number }[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
  const set = (i: number, x: number, y: number, z: number) => { P[i] = { x, y, z }; };
  // pelvis at origin; y grows downward in MediaPipe. Shoulders 0.5 m up. Lean shifts shoulders in x.
  set(23, 0.1, 0, 0); set(24, -0.1, 0, 0);
  set(11, 0.18 + lean * 0.3, -0.5, 0); set(12, -0.18 + lean * 0.3, -0.5, 0);
  set(0, lean * 0.35, -0.7, 0);
  // knees: thigh 0.45 m, shank 0.45 m; flex forward (z toward camera is negative z)
  const th = 0.45, sh = 0.45;
  for (const [hip, kn, an, he, ft, s] of [[23, 25, 27, 29, 31, 1], [24, 26, 28, 30, 32, -1]] as const) {
    const kx = P[hip].x, ky = P[hip].y + th * Math.cos(knee / 2), kz = -th * Math.sin(knee / 2);
    set(kn, kx, ky, kz);
    set(an, kx, ky + sh * Math.cos(knee / 2), kz + sh * Math.sin(knee / 2));
    set(he, P[an].x, P[an].y + 0.05, P[an].z + 0.05); set(ft, P[an].x, P[an].y + 0.07, P[an].z - 0.15);
    // arms hanging slightly forward
    set(hip === 23 ? 13 : 14, P[hip === 23 ? 11 : 12].x + 0.05 * s, -0.25, -0.1);
    set(hip === 23 ? 15 : 16, P[hip === 23 ? 11 : 12].x + 0.08 * s, -0.15, -0.3);
  }
  return P;
}
const deg = (r: number) => (r * 180 / Math.PI).toFixed(0);
for (const [label, P] of [['standing', body(0, 0)], ['knees 60deg', body(Math.PI / 3, 0)], ['lean right 0.5', body(0.5, 0.5)], ['lean left 0.5', body(0.5, -0.5)]] as const) {
  const q = retarget(P as any);
  const bad = Object.entries(q).filter(([, v]) => !isFinite(v));
  console.log(label.padEnd(16), `kneeL ${deg(q.left_knee_joint)} kneeR ${deg(q.right_knee_joint)} hipPitchL ${deg(q.left_hip_pitch_joint)} hipRollL ${deg(q.left_hip_roll_joint)} hipRollR ${deg(q.right_hip_roll_joint)} waistPitch ${deg(q.waist_pitch_joint)} shPitchL ${deg(q.left_shoulder_pitch_joint)} shRollL ${deg(q.left_shoulder_roll_joint)} elbowL ${deg(q.left_elbow_joint)}`, bad.length ? `NON-FINITE ${bad.map(b => b[0])}` : '');
}
console.log('stanceFrom(0.5, 0.8)', Object.fromEntries(Object.entries(stanceFrom(0.5, 0.8)).map(([k, v]) => [k, +v.toFixed(2)])));
