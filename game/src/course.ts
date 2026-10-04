// A real piste for the browser: heightfield sampling, centerline progress and gate geometry.
// World frame matches the course builder: x east, y north, z up, metres, origin at the course start.

export type Vec3 = [number, number, number];

export interface Gate {
  id: number;
  s: number;
  side: 'left' | 'right';
  color: 'red' | 'blue';
  turn_pole: Vec3;
  outer_pole: Vec3;
  poles: Vec3[];
  panel_normal: Vec3;
  panels: { center: Vec3; normal: Vec3; width: number; height: number }[];
}

export interface CourseMeta {
  slug: string; resort: string; run: string; difficulty: string;
  lat0: number; lon0: number; z_datum_msl: number;
  x0: number; y0: number; cell: number; nrow: number; ncol: number; z_min: number; z_max: number;
  centerline: [number, number, number, number][]; // x y z s
  gates: { params: Record<string, number>; gates: Gate[]; finish: { s: number; poles: Vec3[]; line_normal: Vec3 } };
  start: { snow_xyz: Vec3; yaw: number; pitch: number; surface_normal: Vec3 };
  stats: { length_m: number; drop_m: number; mean_slope_deg: number; max_slope_deg: number; start_elevation_msl_m: number; gates: number };
}

export class Course {
  readonly z: Float32Array;
  readonly meta: CourseMeta;
  /** Where the race starts along the centerline (m). The first metres of the OSM line can be a flat traverse. */
  startS: number;
  startIndex: number;
  /** Where the race ends (m along the centerline). Shorter than the course for a demo-length run. */
  finishS: number;
  finishIndex: number;
  constructor(meta: CourseMeta, heights: ArrayBuffer, startS = 90, raceLength = 650) {
    this.meta = meta;
    const total = meta.centerline[meta.centerline.length - 1][3];
    this.startS = Math.min(startS, total * 0.5);
    this.startIndex = Math.max(0, meta.centerline.findIndex(p => p[3] >= this.startS));
    this.finishS = Math.min(total, this.startS + Math.max(100, raceLength));
    this.finishIndex = meta.centerline.findIndex(p => p[3] >= this.finishS - 1e-6);
    if (this.finishIndex < 0) this.finishIndex = meta.centerline.length - 1;
    this.z = new Float32Array(heights);
    if (this.z.length !== meta.nrow * meta.ncol) throw new Error(`heights ${this.z.length} != ${meta.nrow}x${meta.ncol}`);
  }

  get x0() { return this.meta.x0; }
  get y0() { return this.meta.y0; }
  get cell() { return this.meta.cell; }
  get nrow() { return this.meta.nrow; }
  get ncol() { return this.meta.ncol; }
  get xMax() { return this.x0 + (this.ncol - 1) * this.cell; }
  get yMax() { return this.y0 + (this.nrow - 1) * this.cell; }

  at(r: number, c: number) { return this.z[r * this.ncol + c]; }

  /** Height and unit normal at world (x, y), bilinear, clamped at the edges. */
  sample(x: number, y: number, out?: { h: number; n: Vec3 }) {
    const fx = (x - this.x0) / this.cell, fy = (y - this.y0) / this.cell;
    const c = Math.min(Math.max(Math.floor(fx), 0), this.ncol - 2);
    const r = Math.min(Math.max(Math.floor(fy), 0), this.nrow - 2);
    const tx = Math.min(Math.max(fx - c, 0), 1), ty = Math.min(Math.max(fy - r, 0), 1);
    const z00 = this.at(r, c), z01 = this.at(r, c + 1), z10 = this.at(r + 1, c), z11 = this.at(r + 1, c + 1);
    const h = z00 * (1 - tx) * (1 - ty) + z01 * tx * (1 - ty) + z10 * (1 - tx) * ty + z11 * tx * ty;
    const dzdx = ((z01 - z00) * (1 - ty) + (z11 - z10) * ty) / this.cell;
    const dzdy = ((z10 - z00) * (1 - tx) + (z11 - z01) * tx) / this.cell;
    const inv = 1 / Math.hypot(dzdx, dzdy, 1);
    const o = out ?? { h: 0, n: [0, 0, 1] as Vec3 };
    o.h = h; o.n[0] = -dzdx * inv; o.n[1] = -dzdy * inv; o.n[2] = inv;
    return o;
  }

  /** Gradient of the surface (dz/dx, dz/dy) at world (x, y). */
  grad(x: number, y: number): [number, number] {
    const s = this.sample(x, y);
    return [-s.n[0] / s.n[2], -s.n[1] / s.n[2]];
  }

  slopeDeg(x: number, y: number) {
    const [gx, gy] = this.grad(x, y);
    return Math.atan(Math.hypot(gx, gy)) * 180 / Math.PI;
  }

  inside(x: number, y: number, margin = 2) {
    return x > this.x0 + margin && x < this.xMax - margin && y > this.y0 + margin && y < this.yMax - margin;
  }

  /** Nearest centerline sample starting the search at `hint`, searching a window around it (centerline points are 2 m apart). */
  nearest(x: number, y: number, hint = 0, window = 40) {
    const cl = this.meta.centerline;
    let best = hint, bd = Infinity;
    const lo = Math.max(0, hint - 6), hi = Math.min(cl.length - 1, hint + window);
    for (let i = lo; i <= hi; i++) {
      const d = (cl[i][0] - x) ** 2 + (cl[i][1] - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return { index: best, dist: Math.sqrt(bd), s: cl[best][3] };
  }

  /** Signed lateral offset from the centerline at index i: positive = skier's right when facing downhill. */
  lateral(x: number, y: number, i: number) {
    const cl = this.meta.centerline;
    const a = cl[Math.max(0, i - 1)], b = cl[Math.min(cl.length - 1, i + 1)];
    const tx = b[0] - a[0], ty = b[1] - a[1], tl = Math.hypot(tx, ty) || 1;
    // right-hand perpendicular of the downhill tangent
    return ((x - cl[i][0]) * ty - (y - cl[i][1]) * tx) / tl;
  }

  /** Downhill heading (radians, CCW from +x) of the centerline at index i. */
  headingAt(i: number) {
    const cl = this.meta.centerline;
    const a = cl[Math.max(0, i - 2)], b = cl[Math.min(cl.length - 1, i + 2)];
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  }

  get length() { return this.meta.centerline[this.meta.centerline.length - 1][3]; }
  /** Race length from the start point to the finish. */
  get raceLength() { return this.finishS - this.startS; }
  get fullCourse() { return this.finishS >= this.length - 1e-6; }
  /** Gates on the raced part of the course. */
  get gates() { return this.meta.gates.gates.filter(g => g.s > this.startS + 15 && g.s < this.finishS - 12); }
  /** The finish line: the course builder's when racing the whole course, else a line across the centerline at finishS. */
  get finish(): { s: number; poles: Vec3[]; line_normal: Vec3 } {
    if (this.fullCourse) return this.meta.gates.finish;
    const i = this.finishIndex, p = this.meta.centerline[i], h = this.headingAt(i);
    const nx = Math.sin(h), ny = -Math.cos(h), w = 3.65; // right-hand normal, half width as the builder's
    const a: Vec3 = [p[0] - nx * w, p[1] - ny * w, this.sample(p[0] - nx * w, p[1] - ny * w).h];
    const b: Vec3 = [p[0] + nx * w, p[1] + ny * w, this.sample(p[0] + nx * w, p[1] + ny * w).h];
    return { s: this.finishS, poles: [a, b], line_normal: [-Math.cos(h), -Math.sin(h), 0] };
  }
}

export async function loadCourse(slug: string, base = '/courses', startS = 90, raceLength = 650): Promise<Course> {
  const [meta, bin] = await Promise.all([
    fetch(`${base}/${slug}/course.json`).then(r => r.json() as Promise<CourseMeta>),
    fetch(`${base}/${slug}/heights.bin`).then(r => r.arrayBuffer()),
  ]);
  return new Course(meta, bin, startS, raceLength);
}

/** Side of a gate line the point is on: > 0 is uphill of the gate (toward the skier before passing). */
export function gateSide(g: Gate, x: number, y: number) {
  return (x - g.turn_pole[0]) * g.panel_normal[0] + (y - g.turn_pole[1]) * g.panel_normal[1];
}

/** Where along the pole-to-pole axis a point falls: 0 at the turn pole, 1 at the outer pole. */
export function gateSpan(g: Gate, x: number, y: number) {
  const ax = g.outer_pole[0] - g.turn_pole[0], ay = g.outer_pole[1] - g.turn_pole[1];
  const l2 = ax * ax + ay * ay || 1;
  return ((x - g.turn_pole[0]) * ax + (y - g.turn_pole[1]) * ay) / l2;
}
