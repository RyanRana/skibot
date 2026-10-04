// The mountain. Real terrain twice: the piste at 2 m cells as a snow mesh, and the Alps around it at
// 30 m cells out to 10 km for the horizon. A dawn sky with a low sun and clouds, snow that shades
// blue in the shade and sparkles toward the eye, trees, a chairlift, safety nets on the turns, race
// banners, a start house, a finish arch with a crowd, confetti, gate bursts, ski trails and spray.
// World is z-up in metres, the same frame as the course data.
import * as THREE from 'three';
import { ShaderChunk } from 'three';
import type { Course, Gate } from './course.ts';
import { spruceGeometry, rockGeometry, personGeometries, snowLanceGeometry, flagGeometry, rng } from './props.ts';

export const FOG = new THREE.Color('#e3ebf4');
const SUN_AZ = 158 * Math.PI / 180;   // from the south-south-east: raking across the slope from the skier's right-rear, out of the forward view
const SUN_EL = 21 * Math.PI / 180;
export const SUN_DIR = new THREE.Vector3(Math.cos(SUN_EL) * Math.cos(SUN_AZ), Math.cos(SUN_EL) * Math.sin(SUN_AZ), Math.sin(SUN_EL)).normalize();

const hash2 = (x: number, y: number) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
const timeU = { value: 0 };

export type ObstacleKind = 'tree' | 'hut' | 'tower' | 'pillar' | 'rock' | 'lance';
export interface FarTerrain { z: Float32Array; x0: number; y0: number; cell: number; nrow: number; ncol: number; z_min: number; z_max: number; z_datum_msl: number }

// GLSL noise shared by the sky, the mountains and the snow.
const NOISE_GLSL = `
float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f); return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(13.1, 7.7); a *= 0.5; } return s; }`;
const SUNV = `vec3(${SUN_DIR.x.toFixed(4)}, ${SUN_DIR.y.toFixed(4)}, ${SUN_DIR.z.toFixed(4)})`;

// Shared fog chunk: distance fog, a valley haze that thickens below the course, and aerial perspective
// that warms toward the sun and cools away from it, so far ridges read as distance rather than grey paint.
const FOG_PARS = `#include <fog_pars_fragment>\nvarying vec3 vWPos2; varying vec3 vWNrm;`;
const FOG_FRAG = `
  #ifdef USE_FOG
    float fd = vFogDepth;
    float ff = 1.0 - exp( - fogDensity * fogDensity * fd * fd );
    float haze = smoothstep(-200.0, -700.0, vWPos2.z) * 0.22 * smoothstep(150.0, 3000.0, fd);
    ff = clamp(ff + haze, 0.0, 1.0);
    vec3 vd2 = normalize(vWPos2 - cameraPosition);
    float sunF = pow(max(dot(vd2, ${SUNV}), 0.0), 6.0);
    vec3 fc = mix(fogColor, fogColor * vec3(1.04, 0.97, 0.88) + vec3(0.08, 0.03, 0.0), sunF * 0.7);
    fc = mix(fc, fc * vec3(0.93, 0.96, 1.03), smoothstep(800.0, 6000.0, fd) * (1.0 - sunF) * 0.6);
    gl_FragColor.rgb = mix( gl_FragColor.rgb, fc, ff );
  #endif`;
const V_PARS = `#include <common>\nvarying vec3 vWPos2; varying vec3 vWNrm;`;
const V_BODY = `#include <begin_vertex>\nvWPos2 = (modelMatrix * vec4(position, 1.0)).xyz; vWNrm = normalize(mat3(modelMatrix) * objectNormal);`;

/** The shared material for everything that carries snow (trees, rocks, lances): per-vertex base colour and
 *  `aSnow`, turned into clumpy snow that sits on up-facing surfaces, with a little blue in it from the sky,
 *  surface detail from noise, and an optional sway for the trees. */
function snowyMaterial(key: string, o: { detail?: number; sway?: number } = {}) {
  const detail = (o.detail ?? 6).toFixed(2), sway = (o.sway ?? 0).toFixed(4);
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0 });
  mat.onBeforeCompile = sh => {
    sh.uniforms.uTime = timeU;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\nattribute float aSnow; varying float vSnow; varying vec3 vWP; varying vec3 vWN; uniform float uTime;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vSnow = aSnow;
        #ifdef USE_INSTANCING
          vec3 ip = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
        #else
          vec3 ip = vec3(0.0);
        #endif
        float sw = ${sway} * position.z * position.z * (sin(uTime * 1.1 + ip.x * 0.11 + ip.y * 0.07) + 0.4 * sin(uTime * 2.7 + ip.y * 0.3));
        transformed.x += sw; transformed.y += sw * 0.6;`)
      .replace('#include <worldpos_vertex>', `#include <worldpos_vertex>
        vec4 wp4 = vec4(transformed, 1.0); vec3 wn3 = objectNormal;
        #ifdef USE_INSTANCING
          wp4 = instanceMatrix * wp4; wn3 = mat3(instanceMatrix) * wn3;
        #endif
        vWP = (modelMatrix * wp4).xyz; vWN = normalize(mat3(modelMatrix) * wn3);`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\nvarying float vSnow; varying vec3 vWP; varying vec3 vWN;\n${NOISE_GLSL}`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        float nz = normalize(vWN).z;
        float clump = vnoise(vWP.xy * 3.1 + vWP.z * 2.3) * 0.6 + vnoise(vWP.xy * 11.0 - vWP.z * 9.0) * 0.4;
        float sAmt = smoothstep(0.40, 0.62, vSnow * (0.5 + 0.65 * smoothstep(-0.15, 0.7, nz)) + (clump - 0.5) * 0.6);
        float det = 0.7 + 0.6 * vnoise(vWP.xy * ${detail} + vWP.z * ${detail} * 0.7);
        diffuseColor.rgb = mix(diffuseColor.rgb * det, vec3(0.88, 0.92, 0.97) * (0.94 + 0.06 * clump), sAmt);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        totalEmissiveRadiance += vec3(0.018, 0.04, 0.085) * sAmt;`);
  };
  mat.customProgramCacheKey = () => 'snowy-' + key;
  return mat;
}

const HIDE = new Set((new URLSearchParams(location.search).get('hide') || '').split(',').filter(Boolean));

export class World {
  scene = new THREE.Scene();
  sun: THREE.DirectionalLight;
  terrain!: THREE.Mesh;
  gateGroups: { g: Gate; mats: THREE.MeshStandardMaterial[]; glow: THREE.Mesh[]; kick: { value: number } }[] = [];
  trailL: Trail; trailR: Trail;
  spray: Spray;
  flakes: Flakes;
  confetti: Confetti;
  burst: Burst;
  crowd: Crowd | null = null;
  lift: Lift | null = null;
  pisteDist!: Float32Array;
  lowfx: boolean;
  /** Things you can hit. Trees, hut, towers, pillars: a wipeout. Fence segments: a soft bounce. */
  obstacles: { x: number; y: number; r: number; kind: ObstacleKind; idx: number }[] = [];
  fences: { ax: number; ay: number; bx: number; by: number }[] = [];
  private obsHash = new Map<string, number[]>();
  private fenceHash = new Map<string, number[]>();
  private treeSpots: { x: number; y: number; z: number; h: number; w: number; r: number; mesh: THREE.InstancedMesh; i: number }[] = [];
  private treeShake = new Map<number, number>();
  beacon!: THREE.Mesh; beaconArrow!: THREE.Mesh; doorway!: THREE.Mesh; chevrons!: THREE.InstancedMesh;
  private chevronMat!: THREE.MeshBasicMaterial;

  constructor(public course: Course, far: FarTerrain | null, lowfx = false) {
    this.lowfx = lowfx;
    const sc = this.scene;
    sc.fog = new THREE.FogExp2(FOG.getHex(), 0.00011);
    sc.background = FOG;
    sc.add(new THREE.HemisphereLight('#a9c8f2', '#dfe6ee', 0.78)); // sky light: snow in shade stays bright and blue
    this.sun = new THREE.DirectionalLight('#fff0d8', 1.7);
    this.sun.castShadow = !lowfx;
    this.sun.shadow.mapSize.set(lowfx ? 1024 : 2048, lowfx ? 1024 : 2048);
    const sc2 = this.sun.shadow.camera as THREE.OrthographicCamera;
    sc2.left = -50; sc2.right = 50; sc2.top = 50; sc2.bottom = -50; sc2.near = 1; sc2.far = 600;
    this.sun.shadow.bias = -0.0005; this.sun.shadow.normalBias = 0.04; this.sun.shadow.radius = 3;
    sc.add(this.sun); sc.add(this.sun.target);
    const fill = new THREE.DirectionalLight('#a6c0e8', 0.3); // cool bounce from the opposite side
    fill.position.set(-SUN_DIR.x * 100, -SUN_DIR.y * 100, 60); sc.add(fill);

    this.buildSky();
    this.far = far;
    if (far) this.buildFar(far); else this.buildRidges();
    this.buildTerrain();
    if (!HIDE.has('trees')) this.buildTrees();
    if (!HIDE.has('rocks')) this.buildRocks();
    if (!HIDE.has('lances')) this.buildLances();
    this.buildGuides();
    if (!HIDE.has('markers')) this.buildMarkers();
    if (!HIDE.has('fences')) this.buildFences();
    if (!HIDE.has('banners')) this.buildBanners();
    if (!HIDE.has('gates')) this.buildGates();
    if (!HIDE.has('start')) this.buildStart();
    if (!HIDE.has('finish')) this.buildFinish();
    if (!HIDE.has('lift')) this.lift = new Lift(this, course);
    this.trailL = new Trail(sc); this.trailR = new Trail(sc);
    this.spray = new Spray(sc);
    this.flakes = new Flakes(sc, HIDE.has('flakes') ? 1 : lowfx ? 250 : 700);
    this.confetti = new Confetti(sc);
    this.burst = new Burst(sc);
  }

  /** A prefiltered environment from the sky shader over a snowfield, for reflections on the robot, the
   *  arch and anything else that is glossy. Call once with the renderer. */
  makeEnv(renderer: THREE.WebGLRenderer) {
    const sky = this.scene.children.find(o => (o as THREE.Mesh).isMesh && (o as THREE.Mesh).geometry.type === 'SphereGeometry') as THREE.Mesh | undefined;
    const env = new THREE.Scene();
    if (sky) { const m = new THREE.Mesh(new THREE.SphereGeometry(50, 48, 24), sky.material); env.add(m); }
    const ground = new THREE.Mesh(new THREE.CircleGeometry(49, 48), new THREE.MeshBasicMaterial({ color: '#cfdcea' }));
    ground.position.z = -3; env.add(ground);
    const pm = new THREE.PMREMGenerator(renderer);
    const rt = pm.fromScene(env, 0.02, 0.1, 120);
    pm.dispose();
    this.env = rt.texture;
    return this.env;
  }
  env: THREE.Texture | null = null;

  /** Keep the sun and its shadow box over the skier. */
  follow(x: number, y: number, z: number) {
    this.sun.position.set(x + SUN_DIR.x * 180, y + SUN_DIR.y * 180, z + SUN_DIR.z * 180);
    this.sun.target.position.set(x, y, z);
    this.sun.target.updateMatrixWorld();
  }

  addObstacle(o: { x: number; y: number; r: number; kind: ObstacleKind; idx: number }) {
    const i = this.obstacles.push(o) - 1;
    const k = `${Math.floor(o.x / 10)},${Math.floor(o.y / 10)}`;
    (this.obsHash.get(k) ?? this.obsHash.set(k, []).get(k)!).push(i);
  }
  /** Obstacles within reach of (x, y). */
  nearObstacles(x: number, y: number) {
    const out: typeof this.obstacles = [];
    const cx = Math.floor(x / 10), cy = Math.floor(y / 10);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const i of this.obsHash.get(`${cx + dx},${cy + dy}`) ?? []) out.push(this.obstacles[i]);
    return out;
  }
  nearFences(x: number, y: number) {
    const out: typeof this.fences = [];
    const cx = Math.floor(x / 10), cy = Math.floor(y / 10);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const i of this.fenceHash.get(`${cx + dx},${cy + dy}`) ?? []) out.push(this.fences[i]);
    return out;
  }
  /** A tree was hit: it shakes and sheds its snow. */
  hitTree(idx: number) {
    this.treeShake.set(idx, 1);
    const t = this.treeSpots[idx];
    if (t) { const p = new THREE.Vector3(t.x, t.y, t.z + t.h * 0.55), v = new THREE.Vector3(0, 0, -1); this.spray.emit(p, v, 80, t.w * 0.5); }
  }

  private buildSky() {
    const geo = new THREE.SphereGeometry(9000, 64, 32);
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { sunDir: { value: SUN_DIR.clone() }, uTime: timeU },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform vec3 sunDir; uniform float uTime; varying vec3 vDir;
        ${NOISE_GLSL}
        void main(){
          vec3 d = normalize(vDir);
          float h = clamp(d.z, -0.1, 1.0);
          // where round the horizon we are looking: 1 toward the sun, 0 away from it
          float az = 0.5 + 0.5 * dot(normalize(d.xy + vec2(1e-5)), normalize(sunDir.xy));
          vec3 zenith = vec3(0.13, 0.34, 0.76), mid = vec3(0.46, 0.67, 0.92), haze = vec3(0.90, 0.92, 0.95);
          vec3 horizon = mix(vec3(0.80, 0.86, 0.94), vec3(0.99, 0.86, 0.72), az * az);
          vec3 c = h < 0.0 ? haze : h < 0.18 ? mix(horizon, mid, smoothstep(0.0, 0.18, h)) : mix(mid, zenith, smoothstep(0.18, 0.85, h));
          // warm glow around the sun, strongest near the horizon, and the disc
          float s = max(dot(d, sunDir), 0.0);
          float hor = 1.0 - smoothstep(0.0, 0.35, h);
          c += vec3(1.0, 0.72, 0.42) * (pow(s, 4.0) * 0.10 + pow(s, 24.0) * 0.18) * (0.6 + 0.4 * hor);
          c += vec3(1.0, 0.95, 0.85) * pow(s, 400.0) * 1.2;
          // clouds on a flat sheet (planar projection, so they foreshorten toward the horizon)
          vec2 uv = d.xy / max(d.z + 0.10, 0.03);
          // high cirrus: long streaks drawn out along the wind
          float cir = fbm(uv * vec2(0.9, 2.6) + vec2(uTime * 0.005, 0.0));
          cir = smoothstep(0.50, 0.80, cir) * smoothstep(0.02, 0.18, h) * (1.0 - smoothstep(0.45, 0.9, h));
          // altocumulus: a broken mid layer, grey underneath, lit on the sun side, thin edges brightest
          float cu = fbm(uv * 0.5 + vec2(uTime * 0.008, uTime * 0.002));
          float cuM = smoothstep(0.47, 0.66, cu) * smoothstep(0.015, 0.10, h) * (1.0 - smoothstep(0.22, 0.55, h));
          float edge = 1.0 - smoothstep(0.50, 0.78, cu);
          vec3 cuCol = mix(vec3(0.66, 0.72, 0.84), vec3(1.0, 0.95, 0.90), clamp(0.35 + 0.5 * az + 0.4 * edge, 0.0, 1.0));
          c = mix(c, vec3(0.97, 0.97, 0.98), cir * 0.55);
          c = mix(c, cuCol, cuM * 0.9);
          // the haze band sits on top of everything at the horizon
          c = mix(c, haze, (1.0 - smoothstep(-0.02, 0.05, h)) * 0.6);
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    const sky = new THREE.Mesh(geo, mat);
    sky.frustumCulled = false;
    this.scene.add(sky);
    // The sun itself is drawn by the sky shader; no lens flare (it bloomed and mis-projected at the frame edge).
  }

  /** The real mountains around the course: a smooth-shaded DEM under a procedural winter surface
   *  (snow where the slope lets it hold, banded rock on the faces, patchy spruce up to a ragged treeline,
   *  fields on the valley floor), cloud shadows drifting across it, and aerial perspective. */
  private far: FarTerrain | null = null;

  /** How much of the sun reaches (x, y, h): march toward the sun over the real 30 m DEM out to 6 km and
   *  keep the closest the ray comes to the ground, soft with distance. Baked once, so the long shadows of
   *  the ridges at a low winter sun cost nothing per frame. */
  private sunVis(x: number, y: number, h: number) {
    const f = this.far; if (!f) return 1;
    const hl = Math.hypot(SUN_DIR.x, SUN_DIR.y), ux = SUN_DIR.x / hl, uy = SUN_DIR.y / hl, tanE = SUN_DIR.z / hl;
    let lit = 1, d = f.cell;
    for (let i = 0; i < 70 && d < 6000; i++) {
      const c = Math.round((x + ux * d - f.x0) / f.cell), r = Math.round((y + uy * d - f.y0) / f.cell);
      if (c < 0 || r < 0 || c >= f.ncol || r >= f.nrow) break;
      const v = (h + d * tanE - f.z[r * f.ncol + c]) / (d * 0.03);
      if (v < lit) { lit = Math.max(0, v); if (lit === 0) break; }
      d = d * 1.07 + 8;
    }
    return lit;
  }

  private buildFar(far: FarTerrain) {
    const { nrow, ncol, cell, x0, y0 } = far;
    const n = nrow * ncol;
    const pos = new Float32Array(n * 3), relief = new Float32Array(n);
    const c = this.course;
    // relief = height above the local (~350 m) mean: ridges positive, valley floors negative
    const W = ncol + 1, sat = new Float64Array((nrow + 1) * W);
    for (let r = 0; r < nrow; r++) { let acc = 0; for (let cc = 0; cc < ncol; cc++) { acc += far.z[r * ncol + cc]; sat[(r + 1) * W + cc + 1] = sat[r * W + cc + 1] + acc; } }
    const R = Math.max(2, Math.round(350 / cell));
    for (let r = 0; r < nrow; r++) for (let cc = 0; cc < ncol; cc++) {
      const k = r * ncol + cc;
      pos[k * 3] = x0 + cc * cell; pos[k * 3 + 1] = y0 + r * cell; pos[k * 3 + 2] = far.z[k] - 4; // a little under the near terrain
      const r0 = Math.max(0, r - R), r1 = Math.min(nrow, r + R + 1), c0 = Math.max(0, cc - R), c1 = Math.min(ncol, cc + R + 1);
      const sum = sat[r1 * W + c1] - sat[r0 * W + c1] - sat[r1 * W + c0] + sat[r0 * W + c0];
      relief[k] = THREE.MathUtils.clamp((far.z[k] - sum / ((r1 - r0) * (c1 - c0))) / 140, -1, 1);
    }
    // hole where the near terrain sits (shrunk so there is no gap)
    const hx0 = c.x0 + 40, hx1 = c.xMax - 40, hy0 = c.y0 + 40, hy1 = c.yMax - 40;
    const idx = new Uint32Array((nrow - 1) * (ncol - 1) * 6); let q = 0;
    for (let r = 0; r < nrow - 1; r++) for (let cc = 0; cc < ncol - 1; cc++) {
      const x = x0 + cc * cell, y = y0 + r * cell;
      if (x >= hx0 && x + cell <= hx1 && y >= hy0 && y + cell <= hy1) continue;
      const a = r * ncol + cc, b = a + 1, d = a + ncol, e = d + 1;
      idx[q++] = a; idx[q++] = b; idx[q++] = e; idx[q++] = a; idx[q++] = e; idx[q++] = d;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aRelief', new THREE.BufferAttribute(relief, 1));
    const sun = new Float32Array(n);
    for (let k = 0; k < n; k++) sun[k] = this.sunVis(pos[k * 3], pos[k * 3 + 1], far.z[k] + 2);
    geo.setAttribute('aSun', new THREE.BufferAttribute(sun, 1));
    geo.setIndex(new THREE.BufferAttribute(idx.subarray(0, q), 1)); geo.computeVertexNormals();
    const treeLine = '1820.0', valley = '1000.0', datum = far.z_datum_msl.toFixed(1); // metres above sea level, like msl in the shader
    const mat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.95, metalness: 0 });
    mat.onBeforeCompile = sh => {
      sh.uniforms.uTime = timeU;
      sh.vertexShader = sh.vertexShader.replace('#include <common>', V_PARS + '\nattribute float aRelief; varying float vRelief; attribute float aSun; varying float vSun;').replace('#include <begin_vertex>', V_BODY + '\nvRelief = aRelief; vSun = aSun;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>\nuniform float uTime; varying float vRelief; varying float vSun;\n${NOISE_GLSL}`)
        .replace('#include <lights_fragment_end>', `#include <lights_fragment_end>\n reflectedLight.directDiffuse *= vSun; reflectedLight.directSpecular *= vSun;`)
        .replace('#include <fog_pars_fragment>', FOG_PARS)
        .replace('#include <fog_fragment>', FOG_FRAG)
        .replace('#include <color_fragment>', `#include <color_fragment>
          vec3 P = vWPos2; vec3 N = normalize(vWNrm);
          float msl = P.z + ${datum};
          float slope = 1.0 - N.z;
          float n1 = fbm(P.xy * 0.0035), n2 = fbm(P.xy * 0.018), n3 = vnoise(P.xy * 0.07), n4 = vnoise(P.xy * 0.22);
          // zones: a ragged treeline, forest in patches with clearings, fields on the flat valley floor
          float treeLine = ${treeLine} + (n1 - 0.5) * 220.0;
          float forest = (1.0 - smoothstep(treeLine - 50.0, treeLine + 50.0, msl)) * (1.0 - smoothstep(0.42, 0.66, slope));
          forest *= smoothstep(0.40, 0.60, n2 * 0.55 + n1 * 0.45 + 0.12);
          float valley = (1.0 - smoothstep(${valley} - 30.0, ${valley} + 90.0, msl)) * (1.0 - smoothstep(0.06, 0.18, slope));
          float steep = slope + (n3 - 0.5) * 0.16 + (n4 - 0.5) * 0.06;
          // surfaces
          vec3 snow = vec3(0.90, 0.93, 0.97);
          vec3 rock = mix(vec3(0.26, 0.26, 0.29), vec3(0.44, 0.40, 0.37), n2) * (0.84 + 0.16 * sin(msl * 0.08 + n2 * 7.0 + n3 * 2.0));
          vec3 spruce = mix(vec3(0.13, 0.22, 0.19), vec3(0.24, 0.34, 0.28), n4);
          spruce = mix(spruce, snow, 0.12 + 0.18 * n3);
          vec3 field = mix(vec3(0.86, 0.89, 0.93), vec3(0.70, 0.76, 0.82), smoothstep(0.42, 0.58, vnoise(P.xy * 0.011)) * 0.8);
          vec3 col = mix(rock, snow, smoothstep(0.60, 0.40, steep));
          col = mix(col, spruce, forest);
          col = mix(col, field, valley);
          col *= 0.95 + 0.10 * n3;                                   // wind-packed snow and scree
          col *= 0.90 + 0.12 * vRelief;                              // valleys sit in their own shade, ridges stand out
          float cs = smoothstep(0.38, 0.72, fbm((P.xy + vec2(uTime * 7.0, uTime * 2.5)) * 0.0008));
          col *= 1.0 - 0.30 * cs;                                    // cloud shadows drifting across
          diffuseColor.rgb = col;`)
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          // sky light on the shaded snow: faces turned from the sun go blue, not grey
          float nl2 = dot(normalize(vWNrm), ${SUNV});
          totalEmissiveRadiance += vec3(0.02, 0.05, 0.10) * (1.0 - smoothstep(-0.3, 0.5, nl2)) * smoothstep(0.6, 0.35, slope);`);
    };
    mat.customProgramCacheKey = () => 'far-v3';
    const m = new THREE.Mesh(geo, mat);
    m.receiveShadow = false; m.frustumCulled = false;
    this.scene.add(m);
  }

  private buildRidges() {
    const c = this.course, N = 320, R = 1900;
    const cx = (c.x0 + c.xMax) / 2, cy = (c.y0 + c.yMax) / 2, zb = c.meta.z_min - 500;
    const pos: number[] = [], idx: number[] = [];
    for (let i = 0; i <= N; i++) {
      const a = (i / N) * Math.PI * 2;
      const h = c.meta.z_max + 420 + 230 * Math.sin(3 * a + 0.4) + 150 * Math.sin(7 * a + 2.1) + 90 * Math.sin(13 * a) + 60 * Math.sin(29 * a + 1.3);
      pos.push(cx + R * Math.cos(a), cy + R * Math.sin(a), h, cx + R * Math.cos(a), cy + R * Math.sin(a), zb);
      if (i < N) { const k = i * 2; idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: '#b7cbe3', side: THREE.DoubleSide }));
    m.frustumCulled = false;
    this.scene.add(m);
  }

  private buildTerrain() {
    const c = this.course, nrow = c.nrow, ncol = c.ncol, cell = c.cell;
    const n = nrow * ncol;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), piste = new Float32Array(n);
    const dist = new Float32Array(n).fill(1e9);
    const R = 34, rc = Math.ceil(R / cell);
    for (const p of c.meta.centerline) {
      const pc = (p[0] - c.x0) / cell, pr = (p[1] - c.y0) / cell;
      const r0 = Math.max(0, Math.floor(pr - rc)), r1 = Math.min(nrow - 1, Math.ceil(pr + rc));
      const c0 = Math.max(0, Math.floor(pc - rc)), c1 = Math.min(ncol - 1, Math.ceil(pc + rc));
      for (let r = r0; r <= r1; r++) for (let cc = c0; cc <= c1; cc++) {
        const d = Math.hypot((cc - pc) * cell, (r - pr) * cell);
        const k = r * ncol + cc;
        if (d < dist[k]) dist[k] = d;
      }
    }
    this.pisteDist = dist;
    const snowP = new THREE.Color('#eef3f8'), snowO = new THREE.Color('#dbe4ef');
    for (let r = 0; r < nrow; r++) for (let cc = 0; cc < ncol; cc++) {
      const k = r * ncol + cc;
      const x = c.x0 + cc * cell, y = c.y0 + r * cell, z = c.at(r, cc);
      pos[k * 3] = x; pos[k * 3 + 1] = y; pos[k * 3 + 2] = z;
      const p = THREE.MathUtils.smoothstep(dist[k], 26, 12);
      piste[k] = p;
      const tone = snowO.clone().lerp(snowP, p);
      const nse = (hash2(cc * 0.37, r * 0.53) - 0.5) * 0.05 * (1 - p);
      col[k * 3] = tone.r + nse; col[k * 3 + 1] = tone.g + nse; col[k * 3 + 2] = tone.b + nse * 0.5;
    }
    const idx = new Uint32Array((nrow - 1) * (ncol - 1) * 6);
    let q = 0;
    for (let r = 0; r < nrow - 1; r++) for (let cc = 0; cc < ncol - 1; cc++) {
      const a = r * ncol + cc, b = a + 1, d = a + ncol, e = d + 1;
      idx[q++] = a; idx[q++] = b; idx[q++] = e; idx[q++] = a; idx[q++] = e; idx[q++] = d;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setAttribute('aPiste', new THREE.BufferAttribute(piste, 1));
    const sunA = new Float32Array(n);
    for (let k = 0; k < n; k++) sunA[k] = this.sunVis(pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2] + 6); // + the coarse DEM's slop
    geo.setAttribute('aSun', new THREE.BufferAttribute(sunA, 1));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.0 });
    const dir = Math.atan2(445, 685);
    mat.onBeforeCompile = sh => {
      sh.uniforms.uDir = { value: new THREE.Vector2(Math.cos(dir), Math.sin(dir)) };
      sh.uniforms.uSun = { value: SUN_DIR.clone() };
      sh.uniforms.uSparkle = { value: HIDE.has('sparkle') ? 0 : 1 };
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', V_PARS + '\nattribute float aPiste; varying float vPiste; attribute float aSun; varying float vSun;')
        .replace('#include <begin_vertex>', V_BODY + '\nvPiste = aPiste; vSun = aSun;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>\nuniform vec2 uDir; uniform vec3 uSun; uniform float uSparkle; varying float vPiste; varying float vSun;\n${NOISE_GLSL}`)
        .replace('#include <lights_fragment_end>', `#include <lights_fragment_end>\n reflectedLight.directDiffuse *= vSun; reflectedLight.directSpecular *= vSun;`)
        .replace('#include <fog_pars_fragment>', FOG_PARS)
        .replace('#include <fog_fragment>', FOG_FRAG)
        // wrap lighting: snow scatters light past the terminator, so the shade side is never black
        .replace('#include <lights_physical_pars_fragment>', ShaderChunk.lights_physical_pars_fragment.replace('reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F );', 'reflectedLight.directDiffuse += saturate( ( dot( geometryNormal, directLight.direction ) + 0.3 ) / 1.3 ) * directLight.color * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F );'))
        .replace('#include <color_fragment>', `#include <color_fragment>
          // groomer corduroy along the course on the piste, wind texture off it
          float along = dot(vWPos2.xy, vec2(-uDir.y, uDir.x));
          float cord = sin(along * 12.566) * 0.03 * vPiste;
          float wind = (vnoise(vWPos2.xy * 0.35) - 0.5) * 0.10 * (1.0 - vPiste) + (vnoise(vWPos2.xy * 2.3) - 0.5) * 0.03;
          // off the piste the wind leaves sastrugi, and rock breaks through where it is steep
          float sas = sin(dot(vWPos2.xy, vec2(0.6, 0.8)) * 1.3 + vnoise(vWPos2.xy * 0.5) * 6.0) * 0.02 * (1.0 - vPiste);
          float wsl = 1.0 - normalize(vWNrm).z;
          float rk = smoothstep(0.40, 0.55, wsl + (vnoise(vWPos2.xy * 0.09) - 0.5) * 0.18) * (1.0 - vPiste);
          vec3 rockC = mix(vec3(0.30, 0.30, 0.33), vec3(0.45, 0.42, 0.40), vnoise(vWPos2.xy * 0.6));
          diffuseColor.rgb = mix(diffuseColor.rgb + cord + wind + sas, rockC, rk * 0.9);`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
          // micro bumps so the snow is not glass-flat
          float b1 = vnoise(vWPos2.xy * 3.0), b2 = vnoise(vWPos2.xy * 3.0 + vec2(0.07, 0.0)), b3 = vnoise(vWPos2.xy * 3.0 + vec2(0.0, 0.07));
          normal = normalize(normal + vec3((b2 - b1), (b3 - b1), 0.0) * 1.2 * (1.0 - 0.6 * vPiste));`)
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          vec3 nrm = normalize(vWNrm);
          float nl = dot(nrm, uSun);
          // snow in shade goes blue (sky light), snow facing the sun warms
          totalEmissiveRadiance += vec3(0.02, 0.05, 0.10) * (1.0 - smoothstep(-0.25, 0.55, nl));
          totalEmissiveRadiance += vec3(0.03, 0.02, 0.0) * smoothstep(0.5, 1.0, nl);
          // glitter: every few centimetres an ice crystal with its own facet normal; the ones that happen to
          // mirror the sun into the eye flash, and they change as you move, like real snow in sunshine
          vec3 vd = normalize(cameraPosition - vWPos2);
          float camD = length(cameraPosition - vWPos2);
          vec3 hv = normalize(vd + uSun);
          float glit = 0.0;
          for (int L = 0; L < 2; L++) {
            float sc = L == 0 ? 22.0 : 9.0;
            vec2 cp = vWPos2.xy * sc; vec2 ci = floor(cp);
            vec3 jn = normalize(nrm + (vec3(h21(ci + 3.1), h21(ci + 7.7), h21(ci + 1.3)) - 0.5) * 1.3);
            float pt = smoothstep(0.32, 0.0, length(fract(cp) - 0.5));
            glit += pow(max(dot(jn, hv), 0.0), 900.0) * pt * step(0.45, h21(ci + 11.0)) * (L == 0 ? smoothstep(22.0, 4.0, camD) : smoothstep(55.0, 12.0, camD) * 0.6);
          }
          float lit = smoothstep(-0.05, 0.25, nl) * vSun;
          totalEmissiveRadiance += mix(vec3(1.0), vec3(0.8, 0.92, 1.25), h21(floor(vWPos2.xy * 22.0) + 5.0)) * glit * 5.0 * lit * uSparkle;
          // sheen at grazing angles: a blue-white lift where the snow turns away from you
          totalEmissiveRadiance += vec3(0.035, 0.05, 0.08) * pow(1.0 - max(dot(nrm, vd), 0.0), 4.0);`);
    };
    mat.customProgramCacheKey = () => 'snow-v6';
    this.terrain = new THREE.Mesh(geo, mat);
    this.terrain.receiveShadow = true;
    this.scene.add(this.terrain);
    // Skirt: hide the seam to the far terrain by dropping a wall around the near grid.
    const skirt = new THREE.BufferGeometry();
    const sp: number[] = [], si: number[] = [];
    const ring: [number, number, number][] = [];
    for (let cc = 0; cc < ncol; cc++) ring.push([c.x0 + cc * cell, c.y0, c.at(0, cc)]);
    for (let r = 1; r < nrow; r++) ring.push([c.xMax, c.y0 + r * cell, c.at(r, ncol - 1)]);
    for (let cc = ncol - 2; cc >= 0; cc--) ring.push([c.x0 + cc * cell, c.yMax, c.at(nrow - 1, cc)]);
    for (let r = nrow - 2; r >= 1; r--) ring.push([c.x0, c.y0 + r * cell, c.at(r, 0)]);
    ring.forEach(([x, y, z]) => { sp.push(x, y, z, x, y, z - 40); });
    for (let i = 0; i < ring.length; i++) { const a = i * 2, b = ((i + 1) % ring.length) * 2; si.push(a, a + 1, b, a + 1, b + 1, b); }
    skirt.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3)); skirt.setIndex(si); skirt.computeVertexNormals();
    const skirtMesh = new THREE.Mesh(skirt, new THREE.MeshStandardMaterial({ color: '#dfe7f0', roughness: 0.9, side: THREE.DoubleSide }));
    this.scene.add(skirtMesh);
  }

  /** A spruce forest that thickens away from the piste: three tree shapes (slender, full, young and
   *  heavily loaded), each an instanced mesh, swaying a little; the ones near the course cast shadows. */
  private buildTrees() {
    const c = this.course, nrow = c.nrow, ncol = c.ncol, cell = c.cell;
    const spots: { x: number; y: number; z: number; h: number; w: number; r: number; v: number; near: boolean; tint: number }[] = [];
    for (let r = 1; r < nrow - 1; r += 2) for (let cc = 1; cc < ncol - 1; cc += 2) {
      const k = r * ncol + cc;
      if (this.pisteDist[k] < 27) continue;
      const hh = hash2(cc, r);
      // clumps: the forest comes in stands with glades between them
      const stand = 0.55 + 0.9 * hash2(Math.floor(cc / 9) * 7.1, Math.floor(r / 9) * 3.3);
      const density = (0.11 * THREE.MathUtils.smoothstep(this.pisteDist[k], 27, 70) + 0.035) * stand;
      if (hh > density) continue;
      const x = c.x0 + cc * cell + (hash2(r, cc) - 0.5) * 3.5, y = c.y0 + r * cell + (hash2(cc + 9, r + 3) - 0.5) * 3.5;
      if (c.slopeDeg(x, y) > 42) continue;
      const big = THREE.MathUtils.smoothstep(this.pisteDist[k], 27, 90);
      const h = 5 + hash2(cc * 3, r * 7) * 6 + big * 6 * hash2(r * 5, cc * 2);
      spots.push({ x, y, z: c.sample(x, y).h, h, w: h * (0.8 + 0.35 * hash2(cc * 13, r * 17)), r: hash2(cc * 11, r * 5) * Math.PI * 2, v: Math.floor(hash2(cc * 19, r * 23) * 3), near: this.pisteDist[k] < 55, tint: hash2(cc * 29, r * 31) });
    }
    const geos = [spruceGeometry(11, { tiers: 9, width: 0.3, seg: 12 }), spruceGeometry(23, { tiers: 7, width: 0.4, seg: 12 }), spruceGeometry(37, { tiers: 6, width: 0.38, seg: 10, snow: 1.3 })];
    const mat = snowyMaterial('tree', { detail: 9, sway: 0.03 });
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3(), col = new THREE.Color();
    for (let v = 0; v < 3; v++) for (const near of [true, false]) {
      const mine = spots.filter(t => t.v === v && t.near === near);
      if (!mine.length) continue;
      const im = new THREE.InstancedMesh(geos[v], mat, mine.length);
      mine.forEach((t, i) => {
        q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), t.r);
        s.set(t.w, t.w, t.h); p.set(t.x, t.y, t.z - 0.25);
        m.compose(p, q, s); im.setMatrixAt(i, m);
        col.setRGB(0.85 + 0.3 * t.tint, 0.9 + 0.2 * t.tint, 0.85 + 0.25 * (1 - t.tint)); im.setColorAt(i, col);
        const idx = this.treeSpots.push({ x: t.x, y: t.y, z: t.z, h: t.h, w: t.w, r: t.r, mesh: im, i }) - 1;
        if (near) this.addObstacle({ x: t.x, y: t.y, r: 0.35 + t.h * 0.055, kind: 'tree', idx });
      });
      im.castShadow = near && !this.lowfx; im.receiveShadow = true;
      this.scene.add(im);
    }
  }

  /** Frost-shattered boulders off the piste, more of them where it is steep, snow on their tops. */
  private buildRocks() {
    const c = this.course, nrow = c.nrow, ncol = c.ncol, cell = c.cell;
    const geos = [rockGeometry(3), rockGeometry(8), rockGeometry(15)];
    const mat = snowyMaterial('rock', { detail: 3.5 });
    const lists: { x: number; y: number; z: number; s: number; r: number }[][] = [[], [], []];
    for (let r = 2; r < nrow - 2; r += 3) for (let cc = 2; cc < ncol - 2; cc += 3) {
      const k = r * ncol + cc, d = this.pisteDist[k];
      if (d < 19 || d > 140) continue;
      const x = c.x0 + cc * cell + (hash2(r * 3, cc) - 0.5) * 4, y = c.y0 + r * cell + (hash2(cc, r * 3) - 0.5) * 4;
      const steep = THREE.MathUtils.smoothstep(c.slopeDeg(x, y), 24, 40);
      if (hash2(cc * 7.3, r * 1.9) > 0.012 + 0.08 * steep) continue;
      const sz = 0.5 + hash2(cc * 2.2, r * 4.4) * (1.0 + 1.4 * steep);
      lists[Math.floor(hash2(r * 9, cc * 4) * 3)].push({ x, y, z: c.sample(x, y).h, s: sz, r: hash2(cc, r * 11) * 6.28 });
    }
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    lists.forEach((list, v) => {
      if (!list.length) return;
      const im = new THREE.InstancedMesh(geos[v], mat, list.length);
      list.forEach((o, i) => {
        q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), o.r); s.set(o.s * 1.3, o.s, o.s); p.set(o.x, o.y, o.z - o.s * 0.25);
        m.compose(p, q, s); im.setMatrixAt(i, m);
        this.addObstacle({ x: o.x, y: o.y, r: o.s * 0.95, kind: 'rock', idx: i });
      });
      im.castShadow = !this.lowfx; im.receiveShadow = true;
      this.scene.add(im);
    });
  }

  /** Snow lances along both edges, leaning out over the piste like the ones on a World Cup course. */
  private buildLances() {
    const c = this.course, cl = c.meta.centerline;
    const geo = snowLanceGeometry();
    const spots: { x: number; y: number; z: number; yaw: number }[] = [];
    let k = 0;
    for (let i = c.startIndex + 20; i < c.finishIndex - 20; i += 26) {
      const h = c.headingAt(i), side = k++ % 2 ? 1 : -1, nx = Math.sin(h) * side, ny = -Math.cos(h) * side;
      const x = cl[i][0] + nx * 17, y = cl[i][1] + ny * 17;
      if (!c.inside(x, y, 4)) continue;
      spots.push({ x, y, z: c.sample(x, y).h, yaw: Math.atan2(-ny, -nx) }); // lean toward the piste
    }
    if (!spots.length) return;
    const im = new THREE.InstancedMesh(geo, snowyMaterial('lance', { detail: 2 }), spots.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion();
    spots.forEach((o, i) => {
      q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), o.yaw);
      m.compose(new THREE.Vector3(o.x, o.y, o.z - 0.1), q, new THREE.Vector3(1, 1, 1)); im.setMatrixAt(i, m);
      this.addObstacle({ x: o.x, y: o.y, r: 0.35, kind: 'lance', idx: i });
    });
    im.castShadow = !this.lowfx;
    this.scene.add(im);
  }

  /** The way down, unmistakably: blue edge ribbons, chevrons flowing down the line, a beacon on the next gate. */
  private buildGuides() {
    const c = this.course, cl = c.meta.centerline;
    // edge ribbons at +-13.5 m from the race start to the finish
    for (const side of [-1, 1]) {
      const pos: number[] = [], idx: number[] = []; let v = 0;
      for (let i = c.startIndex; i <= c.finishIndex; i++) {
        const h = c.headingAt(i), nx = Math.sin(h) * side, ny = -Math.cos(h) * side;
        for (const off of [13.2, 13.8]) {
          const x = cl[i][0] + nx * off, y = cl[i][1] + ny * off;
          const smp = c.sample(x, y);
          pos.push(x + smp.n[0] * 0.04, y + smp.n[1] * 0.04, smp.h + smp.n[2] * 0.04);
        }
        if (i > c.startIndex) idx.push(v - 2, v - 1, v, v - 1, v + 1, v);
        v += 2;
      }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setIndex(idx);
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: '#4d8dff', transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 }));
      m.frustumCulled = false; this.scene.add(m);
    }
    // chevrons every 6 m down the middle, flowing with time
    const n = Math.floor((c.finishIndex - c.startIndex) / 3);
    this.chevronMat = new THREE.MeshBasicMaterial({ color: '#9cc3ff', transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 });
    this.chevronMat.onBeforeCompile = sh => {
      sh.uniforms.uTime = timeU;
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vPhase;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvPhase = float(gl_InstanceID);');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uTime; varying float vPhase;').replace('#include <dithering_fragment>', '#include <dithering_fragment>\ngl_FragColor.a *= 0.55 + 0.45 * sin(uTime * 2.5 - vPhase * 0.9);');
    };
    this.chevrons = new THREE.InstancedMesh(chevronGeometry(), this.chevronMat, Math.max(1, n));
    const m4 = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3(1, 1, 1), up = new THREE.Vector3(0, 0, 1);
    let k = 0;
    for (let i = c.startIndex + 3; i < c.finishIndex - 2 && k < n; i += 3) {
      const h = c.headingAt(i), smp = c.sample(cl[i][0], cl[i][1]);
      const nrm = new THREE.Vector3(smp.n[0], smp.n[1], smp.n[2]);
      const fwd = new THREE.Vector3(Math.cos(h), Math.sin(h), 0); fwd.addScaledVector(nrm, -fwd.dot(nrm)).normalize();
      const left = new THREE.Vector3().crossVectors(nrm, fwd);
      q.setFromRotationMatrix(new THREE.Matrix4().makeBasis(fwd, left, nrm));
      p.set(cl[i][0], cl[i][1], smp.h).addScaledVector(nrm, 0.05);
      m4.compose(p, q, sc); this.chevrons.setMatrixAt(k++, m4);
    }
    this.chevrons.count = k; this.chevrons.frustumCulled = false; this.scene.add(this.chevrons);
    void up;
    // beacon: a soft light column and a bobbing arrow over the next gate; a translucent doorway between its panels
    this.beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.42, 6, 16, 1, true).translate(0, 3, 0).rotateX(Math.PI / 2), new THREE.MeshBasicMaterial({ color: '#ff3b4e', transparent: true, opacity: 0.2, alphaMap: fadeTexture(), depthWrite: false, side: THREE.DoubleSide }));
    this.beaconArrow = new THREE.Mesh(new THREE.ConeGeometry(0.4, 0.8, 4).rotateX(Math.PI), new THREE.MeshBasicMaterial({ color: '#ff3b4e', transparent: true, opacity: 0.9 }));
    this.doorway = new THREE.Mesh(doorwayGeometry(), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.42, depthWrite: false, side: THREE.DoubleSide }));
    this.beacon.frustumCulled = this.beaconArrow.frustumCulled = this.doorway.frustumCulled = false;
    this.scene.add(this.beacon, this.beaconArrow, this.doorway);
  }

  private buildMarkers() {
    const c = this.course, cl = c.meta.centerline;
    const n = Math.floor(cl.length / 10);
    const geo = new THREE.CylinderGeometry(0.035, 0.035, 1.7, 6); geo.translate(0, 0.85, 0); geo.rotateX(Math.PI / 2);
    const im = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color: '#ff7a1a', roughness: 0.6 }), n * 2);
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1);
    let k = 0;
    for (let i = 0; i < cl.length - 1 && k < n * 2; i += 10) {
      const h = c.headingAt(i), nx = Math.sin(h), ny = -Math.cos(h);
      for (const side of [-1, 1]) {
        const x = cl[i][0] + nx * side * 15, y = cl[i][1] + ny * side * 15;
        p.set(x, y, c.sample(x, y).h); m.compose(p, q, s); im.setMatrixAt(k++, m);
      }
    }
    im.count = k; im.castShadow = !this.lowfx;
    this.scene.add(im);
  }

  /** Orange safety nets on the outside of turns. */
  private buildFences() {
    const c = this.course, cl = c.meta.centerline;
    const tex = netTexture();
    const mat = new THREE.MeshStandardMaterial({ map: tex, transparent: true, alphaTest: 0.3, side: THREE.DoubleSide, roughness: 0.85, color: '#ff5a0a' });
    const pos: number[] = [], uv: number[] = [], idx: number[] = [];
    const posts: THREE.Vector3[] = [];
    let v = 0;
    const H = 1.3, OFF = 13;
    for (let i = 8; i < cl.length - 8; i += 3) {
      const h0 = c.headingAt(i - 6), h1 = c.headingAt(i + 6);
      const turn = Math.atan2(Math.sin(h1 - h0), Math.cos(h1 - h0)); // + = turning left
      if (Math.abs(turn) < 0.10) continue;
      const side = turn > 0 ? 1 : -1; // net on the outside: right of a left turn
      const h = c.headingAt(i), nx = Math.sin(h) * side, ny = -Math.cos(h) * side;
      const a = cl[i], b = cl[Math.min(cl.length - 1, i + 3)];
      const ax = a[0] + nx * OFF, ay = a[1] + ny * OFF, bx = b[0] + nx * OFF, by = b[1] + ny * OFF;
      const az = c.sample(ax, ay).h, bz = c.sample(bx, by).h;
      const fi = this.fences.push({ ax, ay, bx, by }) - 1;
      for (const [fx, fy] of [[ax, ay], [bx, by], [(ax + bx) / 2, (ay + by) / 2]]) { const key = `${Math.floor(fx / 10)},${Math.floor(fy / 10)}`; const arr = this.fenceHash.get(key) ?? this.fenceHash.set(key, []).get(key)!; if (!arr.includes(fi)) arr.push(fi); }
      pos.push(ax, ay, az - 0.1, bx, by, bz - 0.1, bx, by, bz + H, ax, ay, az + H);
      const L = Math.hypot(bx - ax, by - ay) / 3;
      uv.push(0, 0, L, 0, L, 1, 0, 1);
      posts.push(new THREE.Vector3(ax, ay, az));
      idx.push(v, v + 1, v + 2, v, v + 2, v + 3); v += 4;
    }
    if (!pos.length) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(idx); geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mat); mesh.castShadow = false;
    this.scene.add(mesh);
    // posts with a yellow pad at the top, like real B-net
    const postGeo = new THREE.CylinderGeometry(0.035, 0.04, H + 0.25, 6).rotateX(Math.PI / 2).translate(0, 0, (H + 0.25) / 2 - 0.1);
    const padGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.18, 8).rotateX(Math.PI / 2).translate(0, 0, H + 0.1);
    const pm = new THREE.InstancedMesh(postGeo, new THREE.MeshStandardMaterial({ color: '#39404b', roughness: 0.6, metalness: 0.4 }), posts.length);
    const pad = new THREE.InstancedMesh(padGeo, new THREE.MeshStandardMaterial({ color: '#ffcf33', roughness: 0.7 }), posts.length);
    const m4 = new THREE.Matrix4();
    posts.forEach((v, i) => { m4.makeTranslation(v.x, v.y, v.z); pm.setMatrixAt(i, m4); pad.setMatrixAt(i, m4); });
    pm.castShadow = !this.lowfx;
    this.scene.add(pm, pad);
  }

  /** Sponsor-style banners along the right of the piste. */
  private buildBanners() {
    const c = this.course, cl = c.meta.centerline;
    const texts = ['HAZARD INTELLIGENCE', 'MHACKS 2026', 'SPACETIMEDB', 'PHOTON', 'KITZBÜHEL · STREIF', 'EVERY RUN GROWS THE DATASET'];
    const colors = ['#0b1220', '#1b4fd8', '#0b1220', '#101418', '#b3121f', '#0b1220'];
    const poleGeo = new THREE.CylinderGeometry(0.03, 0.03, 1.5, 6); poleGeo.translate(0, 0.75, 0); poleGeo.rotateX(Math.PI / 2);
    const poleMat = new THREE.MeshStandardMaterial({ color: '#2a2f3a' });
    let k = 0;
    for (let i = c.startIndex + 15; i < c.finishIndex - 10; i += 28) {
      const h = c.headingAt(i), nx = Math.sin(h), ny = -Math.cos(h);
      const side = k % 3 === 2 ? -1 : 1;
      const x = cl[i][0] + nx * side * 11, y = cl[i][1] + ny * side * 11;
      const z = c.sample(x, y).h;
      const t = texts[k % texts.length], col = colors[k % colors.length];
      const tex = textTexture(t, col, '#ffffff', 1024, 192);
      const bmat = new THREE.MeshStandardMaterial({ map: tex, side: THREE.DoubleSide, roughness: 0.7 }); flutter(bmat, 0.03);
      const banner = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 0.8, 8, 2), bmat);
      const along = new THREE.Vector3(Math.cos(h), Math.sin(h), 0), nrm = new THREE.Vector3(-nx * side, -ny * side, 0);
      banner.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(along, new THREE.Vector3(0, 0, 1), nrm));
      banner.position.set(x, y, z + 1.0); banner.castShadow = !this.lowfx;
      this.scene.add(banner);
      for (const d of [-2.1, 2.1]) {
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.set(x + along.x * d, y + along.y * d, c.sample(x + along.x * d, y + along.y * d).h);
        this.scene.add(pole);
      }
      k++;
    }
  }

  /** Giant slalom gates: poles in the gate's colour on black hinges, stretched panels with a white band
   *  and the race name, fluttering in the wind and whipping when you brush through. */
  private buildGates() {
    const c = this.course;
    const poleGeo = new THREE.CylinderGeometry(0.025, 0.03, 1, 8); poleGeo.translate(0, 0.5, 0); poleGeo.rotateX(Math.PI / 2);
    const hingeGeo = new THREE.CylinderGeometry(0.045, 0.05, 0.16, 8); hingeGeo.translate(0, 0.08, 0); hingeGeo.rotateX(Math.PI / 2);
    const poleMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.35, metalness: 0.05 });
    const poles = new THREE.InstancedMesh(poleGeo, poleMat, c.gates.length * 4);
    const hinges = new THREE.InstancedMesh(hingeGeo, new THREE.MeshStandardMaterial({ color: '#15181d', roughness: 0.5 }), c.gates.length * 4);
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3(), col = new THREE.Color();
    let k = 0;
    const ph = c.meta.gates.params.pole_height ?? 1.3;
    const panelTex = { red: gatePanelTexture('#e8192c'), blue: gatePanelTexture('#1d56e8') };
    for (const g of c.gates) {
      const colour = g.color === 'red' ? '#ff2d3f' : '#2f6bff';
      const mats: THREE.MeshStandardMaterial[] = [];
      const glow: THREE.Mesh[] = [];
      const kick = { value: 0 };
      for (const pl of g.poles) {
        p.set(pl[0], pl[1], pl[2]); s.set(1, 1, ph); m.compose(p, q, s); poles.setMatrixAt(k, m); poles.setColorAt(k, col.set(colour));
        s.set(1, 1, 1); m.compose(p, q, s); hinges.setMatrixAt(k, m); k++;
      }
      for (const pn of g.panels) {
        const mat = new THREE.MeshStandardMaterial({ map: panelTex[g.color], roughness: 0.55, side: THREE.DoubleSide, transparent: true, opacity: 1, emissive: new THREE.Color(colour), emissiveIntensity: 0 });
        mat.onBeforeCompile = sh => {
          sh.uniforms.uTime = timeU; sh.uniforms.uKick = kick;
          sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uTime; uniform float uKick;')
            .replace('#include <begin_vertex>', `#include <begin_vertex>
              float fl = clamp(uv.x, 0.0, 1.0);
              float amp = 0.03 + 0.16 * uKick;
              transformed.z += sin(uTime * (9.0 + 14.0 * uKick) + position.x * 5.0 + position.y * 3.0) * amp * fl;
              transformed.y += cos(uTime * 7.0 + position.x * 4.0) * 0.012 * fl;`);
        };
        mat.customProgramCacheKey = () => 'gate-panel';
        mats.push(mat);
        const geo = new THREE.PlaneGeometry(pn.width * 2.4, pn.height * 1.5, 8, 3);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(pn.center[0], pn.center[1], pn.center[2] + pn.height * 0.5);
        const nrm = new THREE.Vector3(pn.normal[0], pn.normal[1], 0).normalize();
        const across = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), nrm).normalize();
        mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(across, new THREE.Vector3(0, 0, 1), nrm));
        mesh.castShadow = !this.lowfx;
        this.scene.add(mesh);
        glow.push(mesh);
      }
      this.gateGroups.push({ g, mats, glow, kick });
    }
    poles.count = hinges.count = k; poles.castShadow = !this.lowfx;
    this.scene.add(poles, hinges);
  }

  /** Hazard Intelligence on the snow: the stretches where people's hearts ran high (pooled from everyone's
   *  Presage vitals) get a translucent band across the piste and a warning sign at their entry. */
  setHazards(bins: { s0: number; s1: number; rise: number; n: number }[]) {
    for (const o of this.hazardObjs) { this.scene.remove(o); (o as THREE.Mesh).geometry?.dispose(); }
    this.hazardObjs = [];
    const c = this.course, cl = c.meta.centerline;
    for (const b of bins) {
      if (b.rise < 0.06 || b.n < 2) continue;
      const hot = b.rise >= 0.12, col = hot ? '#ff3b4e' : '#ffb340';
      // a strip across the whole piste that hugs the snow: 9 vertices across (a flat strip sank under the crown)
      const pos: number[] = [], uv: number[] = [], idx: number[] = [];
      const ACROSS = 9;
      let rows = 0, first = -1;
      for (let i = 0; i < cl.length; i++) {
        if (cl[i][3] < b.s0 || cl[i][3] > b.s1) continue;
        if (first < 0) first = i;
        const h = c.headingAt(i), nx = Math.sin(h), ny = -Math.cos(h);
        for (let j = 0; j < ACROSS; j++) {
          const u = j / (ACROSS - 1), off = -13 + 26 * u;
          const x = cl[i][0] + nx * off, y = cl[i][1] + ny * off, smp = c.sample(x, y);
          pos.push(x + smp.n[0] * 0.06, y + smp.n[1] * 0.06, smp.h + smp.n[2] * 0.06);
          uv.push(u, (cl[i][3] - b.s0) / (b.s1 - b.s0));
        }
        if (rows) for (let j = 0; j < ACROSS - 1; j++) {
          const a0 = (rows - 1) * ACROSS + j, b0 = rows * ACROSS + j;
          idx.push(a0, a0 + 1, b0, a0 + 1, b0 + 1, b0);
        }
        rows++;
      }
      if (rows < 2) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); geo.setIndex(idx);
      const mat = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
        uniforms: { uColor: { value: new THREE.Color(col) }, uTime: timeU, uK: { value: Math.min(1, b.rise / 0.2) } },
        vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        fragmentShader: `uniform vec3 uColor; uniform float uTime, uK; varying vec2 vUv;
          void main(){
            float edge = smoothstep(0.0, 0.12, vUv.x) * smoothstep(0.0, 0.12, 1.0 - vUv.x);
            float ends = smoothstep(0.0, 0.15, vUv.y) * smoothstep(0.0, 0.15, 1.0 - vUv.y);
            float stripes = 0.55 + 0.45 * step(0.5, fract(vUv.y * 6.0 - uTime * 0.6));
            float a = (0.1 + 0.09 * uK) * ends * (0.55 + 0.45 * stripes) * (0.6 + 1.4 * (1.0 - edge)) * (0.85 + 0.15 * sin(uTime * 3.0));
            gl_FragColor = vec4(uColor, a);
          }`,
      });
      const band = new THREE.Mesh(geo, mat); band.frustumCulled = false; band.renderOrder = 2;
      this.scene.add(band); this.hazardObjs.push(band);
      // the sign at the entry, on the right edge, facing up the hill
      const h = c.headingAt(first), nx = Math.sin(h), ny = -Math.cos(h);
      const sx = cl[first][0] + nx * 14.5, sy = cl[first][1] + ny * 14.5, sz = c.sample(sx, sy).h;
      const sign = new THREE.Group();
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 2.2, 6).rotateX(Math.PI / 2).translate(0, 0, 1.1), new THREE.MeshStandardMaterial({ color: '#39404b', roughness: 0.6 }));
      const plate = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.05), new THREE.MeshBasicMaterial({ map: hazardSignTexture(b.rise, hot), transparent: true, side: THREE.DoubleSide }));
      plate.position.z = 2.3;
      plate.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(Math.sin(h), -Math.cos(h), 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-Math.cos(h), -Math.sin(h), 0)));
      sign.add(post, plate); sign.position.set(sx, sy, sz);
      this.scene.add(sign); this.hazardObjs.push(sign);
    }
  }
  private hazardObjs: THREE.Object3D[] = [];

  /** A gate was made: its panels whip. */
  kickGate(i: number) { const gg = this.gateGroups[i]; if (gg) gg.kick.value = 1; }

  /** `assist` gates from `next` on are under the magnet: they glow violet. */
  updateGates(next: number, missed: boolean[], assist = 0) {
    this.gateGroups.forEach((gg, i) => {
      const passed = i < next;
      const o = passed ? (missed[i] ? 0.25 : 0.45) : 1;
      const magnet = i >= next && i < next + assist;
      for (const m of gg.mats) {
        m.opacity = o;
        m.emissive.set(magnet ? '#a45cff' : gg.g.color === 'red' ? '#ff2d3f' : '#2f6bff');
        m.emissiveIntensity = magnet ? 0.9 : i === next ? 0.6 : i === next + 1 ? 0.2 : 0;
      }
    });
    const g = this.course.gates[next];
    if (!g || !this.beacon) { if (this.beacon) this.beacon.visible = this.beaconArrow.visible = this.doorway.visible = false; return; }
    const col = assist > 0 ? '#b07cff' : g.color === 'red' ? '#ff3b4e' : '#3d7bff';
    (this.beacon.material as THREE.MeshBasicMaterial).color.set(col); (this.beaconArrow.material as THREE.MeshBasicMaterial).color.set(col);
    const cx = (g.turn_pole[0] + g.outer_pole[0]) / 2, cy = (g.turn_pole[1] + g.outer_pole[1]) / 2, cz = (g.turn_pole[2] + g.outer_pole[2]) / 2;
    this.beacon.position.set(cx, cy, cz); this.beaconArrow.position.set(cx, cy, cz + 3.2);
    // doorway between the inner edges of the two panels, upright, facing the panel normal; wider under the magnet
    const w = Math.hypot(g.outer_pole[0] - g.turn_pole[0], g.outer_pole[1] - g.turn_pole[1]) - 1.1;
    const nrm = new THREE.Vector3(g.panel_normal[0], g.panel_normal[1], 0).normalize();
    const across = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), nrm).normalize();
    this.doorway.scale.set(Math.max(0.5, w) * (assist > 0 ? 2.2 : 1), 1.5, 1);
    this.doorway.position.set(cx, cy, cz + 0.8);
    this.doorway.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(across, new THREE.Vector3(0, 0, 1), nrm));
    (this.doorway.material as THREE.MeshBasicMaterial).color.set(col);
    this.beacon.visible = this.beaconArrow.visible = this.doorway.visible = true;
  }

  /** Start house: a plank hut with a shingle roof beside the line, and a START banner readable from both sides. */
  private buildStart() {
    const c = this.course, i = c.startIndex, p = c.meta.centerline[i], h = c.headingAt(i);
    const along = new THREE.Vector3(Math.cos(h), Math.sin(h), 0), right = new THREE.Vector3(along.y, -along.x, 0);
    const hutPos = new THREE.Vector3(p[0], p[1], 0).addScaledVector(right, 6.0).addScaledVector(along, -4);
    hutPos.z = c.sample(hutPos.x, hutPos.y).h;
    const hut = new THREE.Group();
    const plank = plankTexture(); plank.repeat.set(2, 1.5);
    const walls = new THREE.Mesh(new THREE.BoxGeometry(2.8, 2.4, 2.0), new THREE.MeshStandardMaterial({ map: plank, roughness: 0.9 }));
    walls.position.z = 1.0; hut.add(walls);
    const shingle = shingleTexture(); shingle.repeat.set(3, 2);
    const roof = new THREE.Mesh(new THREE.ConeGeometry(2.6, 1.2, 4), new THREE.MeshStandardMaterial({ map: shingle, roughness: 0.9 }));
    roof.rotation.x = Math.PI / 2; roof.rotation.y = Math.PI / 4; roof.position.z = 2.6; hut.add(roof);
    const snowCap = new THREE.Mesh(new THREE.ConeGeometry(2.0, 0.5, 4), new THREE.MeshStandardMaterial({ color: '#eef3f8', roughness: 0.95, flatShading: true }));
    snowCap.rotation.x = Math.PI / 2; snowCap.rotation.y = Math.PI / 4; snowCap.position.z = 3.05; hut.add(snowCap);
    const door = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 1.6), new THREE.MeshStandardMaterial({ color: '#2a1d14', roughness: 0.9 }));
    door.position.set(-1.41, 0, 0.8); door.rotation.y = -Math.PI / 2; hut.add(door);
    const win = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.5), new THREE.MeshStandardMaterial({ color: '#9fd0ff', emissive: '#4a78a8', emissiveIntensity: 0.6, roughness: 0.3 }));
    win.position.set(-1.41, -0.8, 1.3); win.rotation.y = -Math.PI / 2; hut.add(win);
    const sign = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.4), new THREE.MeshBasicMaterial({ map: textTexture('TIMING', '#1b2233', '#ffffff', 512, 128) }));
    sign.position.set(-1.42, 0, 1.85); sign.rotation.y = -Math.PI / 2; hut.add(sign);
    hut.position.copy(hutPos); hut.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), h);
    hut.traverse(o => { (o as THREE.Mesh).castShadow = !this.lowfx; });
    this.scene.add(hut);
    this.addObstacle({ x: hutPos.x, y: hutPos.y, r: 1.9, kind: 'hut', idx: 0 });
    // banner across the start, text on both faces, fluttering
    const bp = new THREE.Vector3(p[0], p[1], 0).addScaledVector(along, -1.5);
    bp.z = c.sample(bp.x, bp.y).h + 3.3;
    const tex = textTexture('START', '#0b1220', '#37e6a8', 1024, 192);
    for (const face of [1, -1]) {
      const mat = new THREE.MeshStandardMaterial({ map: tex, side: THREE.FrontSide, roughness: 0.7 });
      flutter(mat, 0.05);
      const banner = new THREE.Mesh(new THREE.PlaneGeometry(7.5, 1.3, 12, 2), mat);
      banner.position.copy(bp).addScaledVector(along, face * 0.01);
      // the plane faces +z; map width to the right vector (flipped per face so the text reads correctly), height up, normal along +-course
      const normal = along.clone().multiplyScalar(face);
      const across = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), normal).normalize();
      banner.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(across, new THREE.Vector3(0, 0, 1), normal));
      this.scene.add(banner);
    }
    const poleGeo = new THREE.CylinderGeometry(0.06, 0.06, 4, 8); poleGeo.translate(0, 2, 0); poleGeo.rotateX(Math.PI / 2);
    for (const d of [-3.75, 3.75]) {
      const pole = new THREE.Mesh(poleGeo, new THREE.MeshStandardMaterial({ color: '#1b2233' }));
      const pp = new THREE.Vector3(p[0], p[1], 0).addScaledVector(along, -1.5).addScaledVector(right, d);
      pp.z = c.sample(pp.x, pp.y).h; pole.position.copy(pp); this.scene.add(pole);
    }
  }

  /** Finish arch, banner, and a crowd along the last stretch. */
  /** Finish: an inflatable arch over the line with banners front and back, national flags waving behind
   *  the crowd, and the crowd itself along the last stretch. */
  private buildFinish() {
    const f = this.course.finish;
    const [a, b] = f.poles;
    const mid = new THREE.Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.min(a[2], b[2]));
    const along = new THREE.Vector3(b[0] - a[0], b[1] - a[1], 0).normalize();
    const fn = new THREE.Vector3(f.line_normal[0], f.line_normal[1], 0).normalize();
    const up = new THREE.Vector3(0, 0, 1);
    for (const pl of [a, b]) this.addObstacle({ x: pl[0], y: pl[1], r: 0.7, kind: 'pillar', idx: 0 });
    const half = Math.hypot(b[0] - a[0], b[1] - a[1]) / 2 + 0.2;
    const archMat = new THREE.MeshPhysicalMaterial({ map: archTexture(), roughness: 0.45, clearcoat: 0.6, clearcoatRoughness: 0.35 });
    const arch = new THREE.Mesh(new THREE.TorusGeometry(half, 0.6, 18, 64, Math.PI), archMat);
    arch.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(along, up, fn));
    arch.scale.set(1, 1.35, 1);
    arch.position.copy(mid).addScaledVector(up, -0.3);
    arch.castShadow = !this.lowfx;
    this.scene.add(arch);
    // banners hung under the top of the arch, readable from both sides
    const top = half * 1.35 - 0.3;
    for (const [face, text] of [[1, 'FINISH'], [-1, 'HAZARD INTELLIGENCE · KITZBÜHEL']] as const) {
      const tex = textTexture(text, '#0b1220', '#ffffff', 1024, 192);
      const banner = new THREE.Mesh(new THREE.PlaneGeometry(half * 1.45, half * 0.27), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.7 }));
      const n = fn.clone().multiplyScalar(face);
      banner.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3().crossVectors(up, n).normalize(), up, n));
      banner.position.copy(mid).addScaledVector(up, top - 0.55).addScaledVector(n, 0.05);
      this.scene.add(banner);
    }
    this.buildFlags(mid, along, fn);
    this.crowd = new Crowd(this, this.course);
  }

  /** A row of flags on tall poles either side of the finish, behind the crowd. */
  private buildFlags(mid: THREE.Vector3, along: THREE.Vector3, fn: THREE.Vector3) {
    const geo = flagGeometry();
    const poleGeo = new THREE.CylinderGeometry(0.035, 0.05, 6, 6).rotateX(Math.PI / 2).translate(0, 0, 3);
    const poleMat = new THREE.MeshStandardMaterial({ color: '#d9dee6', roughness: 0.3, metalness: 0.6 });
    const kinds = ['at', 'ch', 'de', 'it', 'fr', 'no', 'us', 'ca', 'se', 'gt'];
    let k = 0;
    for (const side of [-1, 1]) for (let j = 0; j < 5; j++) {
      const pos = mid.clone().addScaledVector(along, side * (17 + j * 0.2)).addScaledVector(fn, 4 + j * 5.5);
      if (!this.course.inside(pos.x, pos.y, 2)) continue;
      pos.z = this.course.sample(pos.x, pos.y).h;
      const pole = new THREE.Mesh(poleGeo, poleMat); pole.position.copy(pos); pole.castShadow = !this.lowfx; this.scene.add(pole);
      const mat = new THREE.MeshStandardMaterial({ map: flagTexture(kinds[k++ % kinds.length]), side: THREE.DoubleSide, roughness: 0.75 });
      flutter(mat, 0.12);
      const flag = new THREE.Mesh(geo, mat);
      flag.position.copy(pos).add(new THREE.Vector3(0, 0, 5.3));
      flag.rotation.z = Math.atan2(fn.y, fn.x) + Math.PI * 0.85 + (k % 3) * 0.1; // all blowing the same way
      flag.castShadow = !this.lowfx;
      this.scene.add(flag);
    }
  }

  update(dt: number, camPos: THREE.Vector3, timeScale = 1) {
    timeU.value += dt;
    if (this.beaconArrow?.visible) { this.beaconArrow.position.z += Math.sin(timeU.value * 4) * 0.012; this.beaconArrow.rotation.z += dt * 1.5; }
    if (this.treeShake.size) {
      const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), q2 = new THREE.Quaternion(), sc = new THREE.Vector3();
      const touched = new Set<THREE.InstancedMesh>();
      for (const [idx, k] of [...this.treeShake]) {
        const t = this.treeSpots[idx]; if (!t) { this.treeShake.delete(idx); continue; }
        const nk = k - dt * 1.4; const wob = Math.sin(timeU.value * 22) * 0.07 * Math.max(0, nk);
        q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), wob).multiply(q2.setFromAxisAngle(new THREE.Vector3(0, 0, 1), t.r));
        sc.set(t.w, t.w, t.h); p.set(t.x, t.y, t.z - 0.25);
        m.compose(p, q, sc); t.mesh.setMatrixAt(t.i, m); touched.add(t.mesh);
        if (nk <= 0) this.treeShake.delete(idx); else this.treeShake.set(idx, nk);
      }
      for (const mesh of touched) mesh.instanceMatrix.needsUpdate = true;
    }
    for (const gg of this.gateGroups) if (gg.kick.value > 0) gg.kick.value = Math.max(0, gg.kick.value - dt * 1.6);
    this.spray.update(dt);
    this.flakes.update(dt, camPos);
    this.confetti.update(dt);
    this.burst.update(dt);
    this.crowd?.update(dt);
    this.lift?.update(dt * timeScale);
  }
}

// ------------------------------------------------------------------------------------------ textures

function noiseCanvas(w: number, h: number, paint: (g: CanvasRenderingContext2D, rnd: () => number) => void) {
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const g = cv.getContext('2d')!;
  let seed = 7; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  paint(g, rnd);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.anisotropy = 4;
  return tex;
}
/** Wooden planks with grain and dark seams. */
function plankTexture() {
  return noiseCanvas(256, 256, (g, rnd) => {
    g.fillStyle = '#6b4a2f'; g.fillRect(0, 0, 256, 256);
    for (let y = 0; y < 256; y += 32) {
      g.fillStyle = `hsl(${24 + rnd() * 8}, ${38 + rnd() * 10}%, ${26 + rnd() * 8}%)`; g.fillRect(0, y, 256, 31);
      for (let k = 0; k < 60; k++) { g.strokeStyle = `rgba(0,0,0,${0.08 + rnd() * 0.12})`; g.lineWidth = 1; const yy = y + rnd() * 31; g.beginPath(); g.moveTo(0, yy); g.bezierCurveTo(80, yy + rnd() * 3 - 1.5, 160, yy - rnd() * 3 + 1.5, 256, yy); g.stroke(); }
      g.fillStyle = 'rgba(0,0,0,.45)'; g.fillRect(0, y + 31, 256, 1);
      g.fillStyle = 'rgba(0,0,0,.35)'; g.beginPath(); g.arc(20 + rnd() * 20, y + 16, 2, 0, 6.28); g.fill(); g.beginPath(); g.arc(216 + rnd() * 20, y + 16, 2, 0, 6.28); g.fill();
    }
  });
}
/** Roof shingles. */
function shingleTexture() {
  return noiseCanvas(256, 256, (g, rnd) => {
    g.fillStyle = '#5a6675'; g.fillRect(0, 0, 256, 256);
    for (let row = 0; row < 8; row++) for (let i = -1; i < 9; i++) {
      const x = i * 32 + (row % 2) * 16, y = row * 32;
      g.fillStyle = `hsl(${212 + rnd() * 10}, ${12 + rnd() * 8}%, ${34 + rnd() * 12}%)`; g.fillRect(x + 1, y + 1, 30, 30);
      g.fillStyle = 'rgba(0,0,0,.35)'; g.fillRect(x, y + 30, 32, 2);
    }
  });
}
/** Galvanised steel lattice for lift towers. */
function steelTexture() {
  return noiseCanvas(128, 256, (g, rnd) => {
    g.fillStyle = '#7b8694'; g.fillRect(0, 0, 128, 256);
    g.strokeStyle = 'rgba(30,36,44,.55)'; g.lineWidth = 6;
    for (let y = 0; y < 256; y += 64) { g.beginPath(); g.moveTo(8, y); g.lineTo(120, y + 64); g.moveTo(120, y); g.lineTo(8, y + 64); g.stroke(); }
    g.lineWidth = 10; g.beginPath(); g.moveTo(8, 0); g.lineTo(8, 256); g.moveTo(120, 0); g.lineTo(120, 256); g.stroke();
    for (let k = 0; k < 40; k++) { g.fillStyle = `rgba(255,255,255,${0.1 + rnd() * 0.2})`; g.fillRect(rnd() * 128, rnd() * 256, 3, 3); }
  });
}
/** Cloth flutter for banners and panels: a vertex ripple that grows toward the free edge (+x). */
function flutter(mat: THREE.Material, amp = 0.03) {
  mat.onBeforeCompile = sh => {
    sh.uniforms.uTime = timeU;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uTime;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>\nfloat fl = clamp(uv.x, 0.0, 1.0); transformed.z += sin(uTime * 6.0 + position.x * 2.2 + position.y * 1.5) * ${amp.toFixed(3)} * fl; transformed.y += cos(uTime * 4.5 + position.x * 1.7) * ${(amp * 0.4).toFixed(3)} * fl;`);
  };
  mat.customProgramCacheKey = () => 'flutter' + amp;
}
/** Vertical alpha ramp: opaque at v = 0, clear at v = 1 (the top of a cylinder). */
function fadeTexture() {
  const cv = document.createElement('canvas'); cv.width = 2; cv.height = 64;
  const g = cv.getContext('2d')!;
  const gr = g.createLinearGradient(0, 0, 0, 64); gr.addColorStop(0, '#000'); gr.addColorStop(1, '#fff');
  g.fillStyle = gr; g.fillRect(0, 0, 2, 64);
  return new THREE.CanvasTexture(cv);
}
/** An open door frame, unit size, standing on y = -0.5: two uprights and a lintel. */
function doorwayGeometry() {
  const sh = new THREE.Shape();
  sh.moveTo(-0.5, -0.5); sh.lineTo(-0.5, 0.5); sh.lineTo(0.5, 0.5); sh.lineTo(0.5, -0.5); sh.lineTo(0.45, -0.5); sh.lineTo(0.45, 0.4); sh.lineTo(-0.45, 0.4); sh.lineTo(-0.45, -0.5); sh.closePath();
  return new THREE.ShapeGeometry(sh);
}
/** A chevron pointing +x, lying flat. */
function chevronGeometry(w = 2.2, l = 1.6, t = 0.45) {
  const sh = new THREE.Shape();
  sh.moveTo(-l / 2, -w / 2); sh.lineTo(-l / 2 + t, -w / 2); sh.lineTo(l / 2, 0); sh.lineTo(-l / 2 + t, w / 2); sh.lineTo(-l / 2, w / 2); sh.lineTo(l / 2 - t, 0); sh.closePath();
  return new THREE.ShapeGeometry(sh);
}

function textTexture(text: string, bg: string, fg: string, w = 1024, h = 192) {
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d')!;
  ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = fg; ctx.font = `900 ${Math.floor(h * 0.62)}px system-ui,sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  let size = Math.floor(h * 0.62);
  while (ctx.measureText(text).width > w * 0.92 && size > 20) { size -= 4; ctx.font = `900 ${size}px system-ui,sans-serif`; }
  ctx.fillText(text, w / 2, h / 2 + h * 0.02);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  return tex;
}

/** A GS panel: the gate colour with a white band across the middle carrying the race name. */
function gatePanelTexture(color: string) {
  const cv = document.createElement('canvas'); cv.width = 256; cv.height = 192;
  const g = cv.getContext('2d')!;
  g.fillStyle = color; g.fillRect(0, 0, 256, 192);
  const gr = g.createLinearGradient(0, 0, 0, 192); gr.addColorStop(0, 'rgba(255,255,255,.10)'); gr.addColorStop(1, 'rgba(0,0,0,.18)');
  g.fillStyle = gr; g.fillRect(0, 0, 256, 192);
  g.fillStyle = '#ffffff'; g.fillRect(0, 70, 256, 52);
  g.fillStyle = color; g.textAlign = 'center'; g.textBaseline = 'middle';
  fitText(g, 'HAZARD INTELLIGENCE', 128, 97, 236, 30);
  g.fillStyle = 'rgba(255,255,255,.55)'; g.fillRect(0, 0, 256, 4); g.fillRect(0, 188, 256, 4);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  return tex;
}

/** Draw text centred at (x, y), shrinking the font until it fits `maxW`. */
export function fitText(g: CanvasRenderingContext2D, text: string, x: number, y: number, maxW: number, size: number, weight = 900, style = '') {
  do { g.font = `${style} ${weight} ${size}px system-ui,sans-serif`; size -= 1; } while (g.measureText(text).width > maxW && size > 8);
  g.fillText(text, x, y);
}

/** A warning sign: the triangle, HAZARD, and how far above resting people's hearts ran here. */
function hazardSignTexture(rise: number, hot: boolean) {
  const cv = document.createElement('canvas'); cv.width = 384; cv.height = 270;
  const g = cv.getContext('2d')!;
  g.fillStyle = 'rgba(11,18,32,.92)'; g.beginPath(); g.roundRect(4, 4, 376, 262, 22); g.fill();
  g.strokeStyle = hot ? '#ff3b4e' : '#ffb340'; g.lineWidth = 8; g.stroke();
  g.fillStyle = hot ? '#ff3b4e' : '#ffb340';
  g.beginPath(); g.moveTo(70, 34); g.lineTo(120, 120); g.lineTo(20, 120); g.closePath(); g.fill();
  g.fillStyle = '#0b1220'; g.font = '900 64px system-ui,sans-serif'; g.textAlign = 'center'; g.fillText('!', 70, 112);
  g.fillStyle = '#ffffff'; g.textAlign = 'left'; g.font = '900 52px system-ui,sans-serif'; g.fillText('HAZARD', 140, 96);
  g.font = '800 58px system-ui,sans-serif'; g.fillStyle = hot ? '#ff6b7a' : '#ffc76a'; g.fillText(`♥ +${Math.round(rise * 100)}%`, 28, 196);
  g.font = '600 26px system-ui,sans-serif'; g.fillStyle = 'rgba(255,255,255,.75)'; g.fillText('hearts race here', 30, 240);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

function netTexture() {
  const cv = document.createElement('canvas'); cv.width = 256; cv.height = 128;
  const g = cv.getContext('2d')!;
  g.clearRect(0, 0, 256, 128);
  g.strokeStyle = '#ffffff'; g.lineWidth = 2;
  // diamond mesh
  for (let x = -128; x <= 256 + 128; x += 12) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 128, 128); g.stroke(); g.beginPath(); g.moveTo(x, 128); g.lineTo(x + 128, 0); g.stroke(); }
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, 256, 12); g.fillRect(0, 120, 256, 8);
  const tex = new THREE.CanvasTexture(cv); tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.anisotropy = 4;
  return tex;
}

function flareTexture(size: number, inner: number, color: string) {
  const cv = document.createElement('canvas'); cv.width = cv.height = size;
  const g = cv.getContext('2d')!;
  const r = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  r.addColorStop(0, color); r.addColorStop(inner, color + 'aa'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ------------------------------------------------------------------------------------------ dressing

/** A chairlift up the right side of the piste: pylons, sagging cables, chairs going up and down. */
class Lift {
  towers: THREE.Vector3[] = [];
  chairs: THREE.InstancedMesh;
  pathLen = 0;
  seg: { a: THREE.Vector3; b: THREE.Vector3; len: number }[] = [];
  t = 0;
  constructor(world: World, course: Course) {
    const cl = course.meta.centerline;
    const towerGeo = new THREE.CylinderGeometry(0.22, 0.3, 1, 8); towerGeo.translate(0, 0.5, 0); towerGeo.rotateX(Math.PI / 2);
    const steel = steelTexture(); steel.repeat.set(1, 4);
    const towerMat = new THREE.MeshStandardMaterial({ map: steel, roughness: 0.85, metalness: 0.05 });
    const barGeo = new THREE.BoxGeometry(0.2, 4.2, 0.25);
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    const n = Math.floor(cl.length / 34);
    const towers = new THREE.InstancedMesh(towerGeo, towerMat, n), bars = new THREE.InstancedMesh(barGeo, towerMat, n);
    let k = 0;
    for (let i = 10; i < cl.length - 10 && k < n; i += 34) {
      const h = course.headingAt(i), nx = Math.sin(h), ny = -Math.cos(h);
      const x = cl[i][0] + nx * 42, y = cl[i][1] + ny * 42;
      if (!course.inside(x, y, 6)) continue;
      const z = course.sample(x, y).h, H = 11 + (k % 2) * 2;
      p.set(x, y, z); s.set(1, 1, H); q.identity(); m.compose(p, q, s); towers.setMatrixAt(k, m);
      p.set(x, y, z + H); s.set(1, 1, 1); q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), h); m.compose(p, q, s); bars.setMatrixAt(k, m);
      this.towers.push(new THREE.Vector3(x, y, z + H));
      world.addObstacle({ x, y, r: 0.6, kind: 'tower', idx: k });
      k++;
    }
    towers.count = k; bars.count = k; towers.castShadow = !world.lowfx;
    world.scene.add(towers, bars);
    // two cables (up and down), sagging between towers
    const pts: THREE.Vector3[][] = [[], []];
    for (let i = 0; i < this.towers.length - 1; i++) {
      const a = this.towers[i], b = this.towers[i + 1];
      const h = Math.atan2(b.y - a.y, b.x - a.x), nx = Math.sin(h), ny = -Math.cos(h);
      for (let j = 0; j <= 8; j++) {
        const u = j / 8, sag = 4 * 1.6 * u * (1 - u);
        for (const [c, side] of [[0, 1.9], [1, -1.9]] as const) pts[c].push(new THREE.Vector3(a.x + (b.x - a.x) * u + nx * side, a.y + (b.y - a.y) * u + ny * side, a.z + (b.z - a.z) * u - sag));
      }
    }
    const cableMat = new THREE.LineBasicMaterial({ color: '#2a2f3a' });
    for (const c of pts) if (c.length > 1) world.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(c), cableMat));
    // the loop the chairs ride: up the first cable, down the second
    const loop = [...pts[0], ...pts[1].slice().reverse()];
    for (let i = 0; i < loop.length; i++) { const a = loop[i], b = loop[(i + 1) % loop.length]; const len = a.distanceTo(b); this.seg.push({ a, b, len }); this.pathLen += len; }
    const chairGeo = new THREE.BoxGeometry(1.4, 0.5, 0.5).translate(0, 0, -2.6);
    const hanger = new THREE.CylinderGeometry(0.03, 0.03, 2.6, 5).translate(0, -1.3, 0).rotateX(Math.PI / 2);
    const chairMat = new THREE.MeshStandardMaterial({ color: '#2b3340', roughness: 0.8, metalness: 0.05 });
    this.chairs = new THREE.InstancedMesh(mergeGeos([chairGeo, hanger]), chairMat, 22);
    this.chairs.castShadow = false;
    world.scene.add(this.chairs);
    this.update(0);
  }
  update(dt: number) {
    if (!this.pathLen) return;
    this.t = (this.t + dt * 2.2) % this.pathLen;
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < this.chairs.count; i++) {
      let d = (this.t + (i / this.chairs.count) * this.pathLen) % this.pathLen;
      let j = 0;
      while (j < this.seg.length && d > this.seg[j].len) { d -= this.seg[j].len; j++; }
      const sg = this.seg[Math.min(j, this.seg.length - 1)];
      p.lerpVectors(sg.a, sg.b, sg.len ? d / sg.len : 0);
      q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.atan2(sg.b.y - sg.a.y, sg.b.x - sg.a.x) + Math.PI / 2);
      m.compose(p, q, s); this.chairs.setMatrixAt(i, m);
    }
    this.chairs.instanceMatrix.needsUpdate = true;
  }
}

function mergeGeos(geos: THREE.BufferGeometry[]) {
  // simple merge for non-indexed/indexed geometries with position + normal
  const pos: number[] = [], nrm: number[] = [];
  for (const g of geos) {
    const ng = g.index ? g.toNonIndexed() : g;
    pos.push(...Array.from(ng.attributes.position.array as Float32Array));
    nrm.push(...Array.from(ng.attributes.normal.array as Float32Array));
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  return out;
}

/** Spectators along the last 70 m: instanced people in ski jackets and beanies who face the piste, bounce
 *  on their toes and throw their arms up when something happens. */
class Crowd {
  people: { x: number; y: number; z: number; yaw: number; s: number; ph: number }[] = [];
  parts: THREE.InstancedMesh[] = [];
  arms: THREE.InstancedMesh;
  excite = 0;
  private m = new THREE.Matrix4(); private a = new THREE.Matrix4(); private q = new THREE.Quaternion(); private qa = new THREE.Quaternion();
  private p = new THREE.Vector3(); private sc = new THREE.Vector3(); private e = new THREE.Euler(); private sh = new THREE.Vector3(); private one = new THREE.Vector3(1, 1, 1); private zAxis = new THREE.Vector3(0, 0, 1);
  constructor(world: World, course: Course) {
    const cl = course.meta.centerline;
    const i1 = course.finishIndex;
    for (let i = Math.max(0, i1 - 36); i <= i1 + 3; i += 1) {
      const h = course.headingAt(i), nx = Math.sin(h), ny = -Math.cos(h);
      for (const side of [-1, 1]) {
        const count = 1 + Math.floor(hash2(i, side) * 3);
        for (let k = 0; k < count; k++) {
          const off = 9 + hash2(i * 3 + k, side * 7) * 6;
          const x = cl[i][0] + nx * side * off + (hash2(i, k) - 0.5) * 2, y = cl[i][1] + ny * side * off + (hash2(k, i) - 0.5) * 2;
          if (!course.inside(x, y)) continue;
          // face the middle of the piste, give or take
          const yaw = Math.atan2(cl[i][1] - y, cl[i][0] - x) + (hash2(k * 5, i) - 0.5) * 0.6;
          this.people.push({ x, y, z: course.sample(x, y).h, yaw, s: 0.92 + hash2(k, i * 5) * 0.16, ph: hash2(i, k) * 6.28 });
        }
      }
    }
    const n = this.people.length;
    const g = personGeometries();
    const mk = (geo: THREE.BufferGeometry, count: number, rough = 0.8) => { const im = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: rough }), count); im.castShadow = !world.lowfx; im.frustumCulled = false; world.scene.add(im); return im; };
    const body = mk(g.body, n, 0.6), legs = mk(g.legs, n), head = mk(g.head, n, 0.7), hat = mk(g.hat, n, 0.9);
    this.arms = mk(g.arm, n * 2, 0.6);
    this.parts = [body, legs, head, hat];
    const jackets = ['#e8192c', '#1d56e8', '#ffb340', '#18b884', '#f3f6fb', '#8a5cff', '#111827', '#ff6a00', '#0ea5e9'];
    const skins = ['#f1c7a5', '#d9a07a', '#a8704a', '#6b4429', '#f5d6bf'];
    const hats = ['#ffffff', '#e8192c', '#111827', '#ffcf33', '#1d56e8', '#18b884'];
    const pants = ['#1b2233', '#2a2f3a', '#3b2f2a', '#11151c'];
    const c = new THREE.Color();
    this.people.forEach((_pp, i) => {
      const r = rng(i * 7 + 3);
      c.set(jackets[Math.floor(r() * jackets.length)]);
      body.setColorAt(i, c); this.arms.setColorAt(i * 2, c); this.arms.setColorAt(i * 2 + 1, c);
      legs.setColorAt(i, c.set(pants[Math.floor(r() * pants.length)]));
      head.setColorAt(i, c.set(skins[Math.floor(r() * skins.length)]));
      hat.setColorAt(i, c.set(hats[Math.floor(r() * hats.length)]));
    });
    this.update(0);
  }
  cheer(strength = 1) { this.excite = Math.min(1.5, this.excite + strength); }
  update(dt: number) {
    this.excite = Math.max(0, this.excite - dt * 0.35);
    const t = timeU.value, ex = Math.min(1, this.excite);
    const { m, a, q, qa, p, sc, e, sh, one, zAxis } = this;
    for (let i = 0; i < this.people.length; i++) {
      const pp = this.people[i];
      const bounce = (0.02 + 0.16 * ex) * Math.max(0, Math.sin(t * (5 + ex * 5) + pp.ph));
      q.setFromAxisAngle(zAxis, pp.yaw); p.set(pp.x, pp.y, pp.z + bounce); sc.setScalar(pp.s);
      m.compose(p, q, sc);
      for (const part of this.parts) part.setMatrixAt(i, m);
      // arms: hanging and swaying a little, up and waving when the crowd is excited
      for (const side of [-1, 1]) {
        const wave = Math.sin(t * 7 + pp.ph * 2 + side) * 0.35 * ex;
        const raise = 0.18 + (2.5 + wave) * ex * (0.6 + 0.4 * Math.sin(pp.ph * 3 + side));
        e.set(-side * raise, 0, 0); qa.setFromEuler(e);
        a.compose(sh.set(0, side * 0.25, 1.24), qa, one);
        this.arms.setMatrixAt(i * 2 + (side > 0 ? 1 : 0), a.premultiply(m));
      }
    }
    for (const part of this.parts) part.instanceMatrix.needsUpdate = true;
    this.arms.instanceMatrix.needsUpdate = true;
  }
}

/** The inflatable arch: race red with white seams. */
function archTexture() {
  const cv = document.createElement('canvas'); cv.width = 1024; cv.height = 128;
  const g = cv.getContext('2d')!;
  g.fillStyle = '#e8192c'; g.fillRect(0, 0, 1024, 128);
  for (let x = 0; x < 1024; x += 64) { g.fillStyle = 'rgba(255,255,255,.85)'; g.fillRect(x, 0, 3, 128); }
  const gr = g.createLinearGradient(0, 0, 0, 128); gr.addColorStop(0, 'rgba(0,0,0,.15)'); gr.addColorStop(0.5, 'rgba(255,255,255,.08)'); gr.addColorStop(1, 'rgba(0,0,0,.15)');
  g.fillStyle = gr; g.fillRect(0, 0, 1024, 128);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

/** Simple national flags (and one for the project). */
function flagTexture(kind: string) {
  const cv = document.createElement('canvas'); cv.width = 192; cv.height = 128;
  const g = cv.getContext('2d')!;
  const h3 = (a: string, b: string, c: string) => { g.fillStyle = a; g.fillRect(0, 0, 192, 43); g.fillStyle = b; g.fillRect(0, 43, 192, 42); g.fillStyle = c; g.fillRect(0, 85, 192, 43); };
  const v3 = (a: string, b: string, c: string) => { g.fillStyle = a; g.fillRect(0, 0, 64, 128); g.fillStyle = b; g.fillRect(64, 0, 64, 128); g.fillStyle = c; g.fillRect(128, 0, 64, 128); };
  const nordic = (bg: string, cross: string, edge: string | null) => { g.fillStyle = bg; g.fillRect(0, 0, 192, 128); if (edge) { g.fillStyle = edge; g.fillRect(52, 0, 32, 128); g.fillRect(0, 48, 192, 32); } g.fillStyle = cross; g.fillRect(58, 0, 20, 128); g.fillRect(0, 54, 192, 20); };
  switch (kind) {
    case 'at': h3('#ed2939', '#ffffff', '#ed2939'); break;
    case 'de': h3('#111111', '#dd0000', '#ffce00'); break;
    case 'it': v3('#009246', '#ffffff', '#ce2b37'); break;
    case 'fr': v3('#0055a4', '#ffffff', '#ef4135'); break;
    case 'ca': v3('#d52b1e', '#ffffff', '#d52b1e'); g.fillStyle = '#d52b1e'; g.beginPath(); g.moveTo(96, 30); g.lineTo(116, 70); g.lineTo(104, 68); g.lineTo(106, 98); g.lineTo(86, 98); g.lineTo(88, 68); g.lineTo(76, 70); g.closePath(); g.fill(); break;
    case 'ch': g.fillStyle = '#d52b1e'; g.fillRect(0, 0, 192, 128); g.fillStyle = '#fff'; g.fillRect(84, 30, 24, 68); g.fillRect(62, 52, 68, 24); break;
    case 'no': nordic('#ba0c2f', '#00205b', '#ffffff'); break;
    case 'se': nordic('#006aa7', '#fecc00', null); break;
    case 'us': for (let i = 0; i < 13; i++) { g.fillStyle = i % 2 ? '#fff' : '#b22234'; g.fillRect(0, i * 128 / 13, 192, 128 / 13 + 1); } g.fillStyle = '#3c3b6e'; g.fillRect(0, 0, 80, 69); break;
    default: g.fillStyle = '#0b1220'; g.fillRect(0, 0, 192, 128); g.fillStyle = '#37e6a8'; g.font = '900 34px system-ui,sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('HAZARD', 96, 48); g.font = '900 26px system-ui,sans-serif'; g.fillText('INTELLIGENCE', 96, 84);
  }
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ------------------------------------------------------------------------------------------ effects

export class Trail {
  static N = 900;
  geo = new THREE.BufferGeometry();
  pos = new Float32Array(Trail.N * 2 * 3);
  count = 0;
  mesh: THREE.Mesh;
  mat: THREE.MeshBasicMaterial;
  private glow = 0;
  constructor(scene: THREE.Scene) {
    const idx: number[] = [], uv = new Float32Array(Trail.N * 2 * 2);
    for (let i = 0; i < Trail.N - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    for (let i = 0; i < Trail.N; i++) { uv[i * 4] = 0; uv[i * 4 + 2] = 1; }
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    this.geo.setIndex(idx);
    this.geo.setDrawRange(0, 0);
    this.mat = new THREE.MeshBasicMaterial({ map: grooveTexture(), color: '#ffffff', transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }
  reset() { this.count = 0; this.geo.setDrawRange(0, 0); }
  /** 0..1: the groove lights up in a colour (a powerup is running). HDR so it blooms. */
  setGlow(k: number, color: THREE.Color) {
    this.glow += (k - this.glow) * 0.1;
    this.mat.color.setRGB(1, 1, 1).lerp(color, this.glow);
  }
  push(p: THREE.Vector3, side: THREE.Vector3, n: THREE.Vector3, half = 0.05) {
    if (this.count >= Trail.N - 1) { this.pos.copyWithin(0, 6 * 150); this.count -= 150; }
    const i = this.count * 6;
    this.pos[i] = p.x - side.x * half + n.x * 0.03; this.pos[i + 1] = p.y - side.y * half + n.y * 0.03; this.pos[i + 2] = p.z - side.z * half + n.z * 0.03;
    this.pos[i + 3] = p.x + side.x * half + n.x * 0.03; this.pos[i + 4] = p.y + side.y * half + n.y * 0.03; this.pos[i + 5] = p.z + side.z * half + n.z * 0.03;
    this.count++;
    (this.geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    this.geo.setDrawRange(0, Math.max(0, (this.count - 1) * 6));
  }
}

/** Across a carved track: a bright lip of thrown-up snow, the shaded groove, the far lip. */
function grooveTexture() {
  const cv = document.createElement('canvas'); cv.width = 64; cv.height = 4;
  const g = cv.getContext('2d')!;
  const gr = g.createLinearGradient(0, 0, 64, 0);
  gr.addColorStop(0, 'rgba(220,230,244,0)'); gr.addColorStop(0.14, 'rgba(232,240,250,.55)'); gr.addColorStop(0.3, 'rgba(118,140,178,.7)');
  gr.addColorStop(0.62, 'rgba(132,154,192,.6)'); gr.addColorStop(0.84, 'rgba(228,236,248,.5)'); gr.addColorStop(1, 'rgba(220,230,244,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 4);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}

/** Powder: soft particles that billow out, grow and fade, bluish where they are thin. */
export class Spray {
  static N = 2600;
  pos = new Float32Array(Spray.N * 3); vel = new Float32Array(Spray.N * 3); life = new Float32Array(Spray.N); max = new Float32Array(Spray.N);
  aLife = new Float32Array(Spray.N); aSize = new Float32Array(Spray.N);
  head = 0;
  points: THREE.Points;
  private scale = { value: 400 };
  constructor(scene: THREE.Scene) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geo.setAttribute('aLife', new THREE.BufferAttribute(this.aLife, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(this.aSize, 1));
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, fog: false,
      uniforms: { uScale: this.scale },
      vertexShader: `attribute float aLife; attribute float aSize; uniform float uScale; varying float vLife;
        void main(){ vLife = aLife; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
          gl_PointSize = aLife <= 0.0 ? 0.0 : min(aSize * (0.5 + 1.1 * (1.0 - aLife)) * uScale / max(0.5, -mv.z), 160.0); }`,
      fragmentShader: `varying float vLife;
        void main(){ vec2 d = gl_PointCoord - 0.5; float r = length(d) * 2.0; if (r > 1.0) discard;
          float a = (1.0 - r * r) * pow(vLife, 0.8) * 0.75;
          vec3 c = mix(vec3(0.80, 0.87, 0.97), vec3(1.0), smoothstep(1.0, 0.2, r));
          gl_FragColor = vec4(c * 1.05, a); }`,
    });
    this.points = new THREE.Points(geo, mat);
    this.points.frustumCulled = false;
    const v = new THREE.Vector2();
    this.points.onBeforeRender = (renderer, _s, camera) => { renderer.getDrawingBufferSize(v); this.scale.value = v.y / (2 * Math.tan(((camera as THREE.PerspectiveCamera).fov ?? 60) * Math.PI / 360)); };
    for (let i = 0; i < Spray.N; i++) this.pos[i * 3 + 2] = -1e4;
    scene.add(this.points);
  }
  static texture() {
    const c = document.createElement('canvas'); c.width = c.height = 32;
    const g = c.getContext('2d')!; const r = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.45, 'rgba(255,255,255,.6)'); r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, 32, 32);
    return new THREE.CanvasTexture(c);
  }
  emit(p: THREE.Vector3, v: THREE.Vector3, n: number, spread = 1.5, size = 0.22) {
    for (let k = 0; k < n; k++) {
      const i = this.head; this.head = (this.head + 1) % Spray.N;
      this.pos[i * 3] = p.x; this.pos[i * 3 + 1] = p.y; this.pos[i * 3 + 2] = p.z;
      this.vel[i * 3] = v.x + (Math.random() - 0.5) * spread; this.vel[i * 3 + 1] = v.y + (Math.random() - 0.5) * spread; this.vel[i * 3 + 2] = v.z + Math.random() * spread * 1.2;
      this.life[i] = this.max[i] = 0.45 + Math.random() * 0.7;
      this.aSize[i] = size * (0.6 + Math.random() * 0.9);
    }
  }
  update(dt: number) {
    for (let i = 0; i < Spray.N; i++) {
      if (this.life[i] <= 0) { this.aLife[i] = 0; continue; }
      this.life[i] -= dt;
      if (this.life[i] <= 0) { this.pos[i * 3 + 2] = -1e4; this.aLife[i] = 0; continue; }
      this.aLife[i] = this.life[i] / this.max[i];
      this.vel[i * 3 + 2] -= 9.81 * dt * 0.45;
      const drag = Math.exp(-dt * 2.2);
      this.vel[i * 3] *= drag; this.vel[i * 3 + 1] *= drag; this.vel[i * 3 + 2] *= Math.exp(-dt * 0.8);
      this.pos[i * 3] += this.vel[i * 3] * dt; this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt; this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
    }
    const g = this.points.geometry;
    (g.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (g.attributes.aLife as THREE.BufferAttribute).needsUpdate = true;
    (g.attributes.aSize as THREE.BufferAttribute).needsUpdate = true;
  }
}

export class Flakes {
  pos: Float32Array; off: Float32Array; n: number;
  points: THREE.Points;
  constructor(scene: THREE.Scene, n = 700) {
    this.n = n; this.pos = new Float32Array(n * 3); this.off = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { this.off[i * 3] = (Math.random() - 0.5) * 60; this.off[i * 3 + 1] = (Math.random() - 0.5) * 60; this.off[i * 3 + 2] = Math.random() * 30; }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.points = new THREE.Points(geo, new THREE.PointsMaterial({ map: Spray.texture(), size: 0.22, transparent: true, opacity: 0.55, depthWrite: false, color: '#ffffff' }));
    this.points.frustumCulled = false;
    scene.add(this.points);
  }
  update(dt: number, cam: THREE.Vector3) {
    for (let i = 0; i < this.n; i++) {
      this.off[i * 3 + 2] -= dt * 1.4; this.off[i * 3] += Math.sin(this.off[i * 3 + 2] * 0.7 + i) * dt * 0.6;
      if (this.off[i * 3 + 2] < -6) this.off[i * 3 + 2] = 24;
      this.pos[i * 3] = cam.x + this.off[i * 3]; this.pos[i * 3 + 1] = cam.y + this.off[i * 3 + 1]; this.pos[i * 3 + 2] = cam.z + this.off[i * 3 + 2];
    }
    (this.points.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  }
}

/** Confetti: small coloured quads that flutter down. */
export class Confetti {
  static N = 420;
  mesh: THREE.InstancedMesh;
  pos = new Float32Array(Confetti.N * 3); vel = new Float32Array(Confetti.N * 3); life = new Float32Array(Confetti.N); spin = new Float32Array(Confetti.N);
  head = 0;
  constructor(scene: THREE.Scene) {
    this.mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.14, 0.09), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, vertexColors: false }), Confetti.N);
    const palette = ['#ff3b4e', '#3d7bff', '#ffb340', '#37e6a8', '#ffffff', '#8a5cff'];
    for (let i = 0; i < Confetti.N; i++) this.mesh.setColorAt(i, new THREE.Color(palette[i % palette.length]));
    this.mesh.frustumCulled = false; this.mesh.count = 0;
    scene.add(this.mesh);
  }
  burst(p: THREE.Vector3, n = 200, up = 7) {
    for (let k = 0; k < n; k++) {
      const i = this.head; this.head = (this.head + 1) % Confetti.N;
      this.pos[i * 3] = p.x + (Math.random() - 0.5) * 3; this.pos[i * 3 + 1] = p.y + (Math.random() - 0.5) * 3; this.pos[i * 3 + 2] = p.z;
      this.vel[i * 3] = (Math.random() - 0.5) * 5; this.vel[i * 3 + 1] = (Math.random() - 0.5) * 5; this.vel[i * 3 + 2] = up * (0.5 + Math.random());
      this.life[i] = 3 + Math.random() * 2; this.spin[i] = Math.random() * 6.28;
    }
    this.mesh.count = Confetti.N;
  }
  update(dt: number) {
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3(1, 1, 1);
    let any = false;
    for (let i = 0; i < Confetti.N; i++) {
      if (this.life[i] <= 0) { p.set(0, 0, -1e4); m.compose(p, q, s); this.mesh.setMatrixAt(i, m); continue; }
      any = true;
      this.life[i] -= dt;
      this.vel[i * 3 + 2] -= 9.81 * dt * 0.25; this.vel[i * 3 + 2] = Math.max(this.vel[i * 3 + 2], -1.6);
      this.vel[i * 3] += Math.sin(this.spin[i] + this.life[i] * 5) * dt * 2;
      this.pos[i * 3] += this.vel[i * 3] * dt; this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt; this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      p.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
      e.set(this.life[i] * 4 + this.spin[i], this.life[i] * 3, this.spin[i]); q.setFromEuler(e);
      m.compose(p, q, s); this.mesh.setMatrixAt(i, m);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (!any) this.mesh.count = 0;
  }
}

/** A short burst of coloured sparks (gate hits). */
export class Burst {
  static N = 400;
  mesh: THREE.InstancedMesh;
  pos = new Float32Array(Burst.N * 3); vel = new Float32Array(Burst.N * 3); life = new Float32Array(Burst.N);
  head = 0; color = new THREE.Color();
  constructor(scene: THREE.Scene) {
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.08, 0.08, 0.08), new THREE.MeshBasicMaterial(), Burst.N);
    this.mesh.frustumCulled = false; this.mesh.count = 0;
    scene.add(this.mesh);
  }
  emit(p: THREE.Vector3, color: string, n = 30, speed = 6) {
    this.color.set(color);
    for (let k = 0; k < n; k++) {
      const i = this.head; this.head = (this.head + 1) % Burst.N;
      this.pos[i * 3] = p.x; this.pos[i * 3 + 1] = p.y; this.pos[i * 3 + 2] = p.z;
      const a = Math.random() * 6.28, b = Math.random() * 1.2 + 0.3;
      this.vel[i * 3] = Math.cos(a) * Math.cos(b) * speed; this.vel[i * 3 + 1] = Math.sin(a) * Math.cos(b) * speed; this.vel[i * 3 + 2] = Math.sin(b) * speed;
      this.life[i] = 0.5 + Math.random() * 0.4;
      this.mesh.setColorAt(i, this.color);
    }
    this.mesh.count = Burst.N; if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
  update(dt: number) {
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    let any = false;
    for (let i = 0; i < Burst.N; i++) {
      if (this.life[i] <= 0) { p.set(0, 0, -1e4); s.set(1, 1, 1); m.compose(p, q, s); this.mesh.setMatrixAt(i, m); continue; }
      any = true;
      this.life[i] -= dt;
      this.vel[i * 3 + 2] -= 9.81 * dt;
      this.pos[i * 3] += this.vel[i * 3] * dt; this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt; this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      p.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
      const k = Math.max(0.1, this.life[i] * 2); s.set(k, k, k);
      m.compose(p, q, s); this.mesh.setMatrixAt(i, m);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (!any) this.mesh.count = 0;
  }
}
