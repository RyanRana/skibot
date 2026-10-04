// The mountain. Real terrain twice: the piste at 2 m cells as a snow mesh, and the Alps around it at
// 60 m cells out to 10 km for the horizon. A dawn sky with a low sun and lens flare, snow that shades
// blue in the shade and sparkles toward the eye, trees, a chairlift, safety nets on the turns, race
// banners, a start house, a finish arch with a crowd, confetti, gate bursts, ski trails and spray.
// World is z-up in metres, the same frame as the course data.
import * as THREE from 'three';
import { ShaderChunk } from 'three';
import { Lensflare, LensflareElement } from 'three/examples/jsm/objects/Lensflare.js';
import type { Course, Gate } from './course.ts';

export const FOG = new THREE.Color('#e3ebf4');
const SUN_AZ = -48 * Math.PI / 180;   // from the south-east (CCW from +x east, so negative = south of east)
const SUN_EL = 17 * Math.PI / 180;
export const SUN_DIR = new THREE.Vector3(Math.cos(SUN_EL) * Math.cos(SUN_AZ), Math.cos(SUN_EL) * Math.sin(SUN_AZ), Math.sin(SUN_EL)).normalize();

const hash2 = (x: number, y: number) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
const timeU = { value: 0 };

export interface FarTerrain { z: Float32Array; x0: number; y0: number; cell: number; nrow: number; ncol: number; z_min: number; z_max: number; z_datum_msl: number }

// Shared fog chunk: distance fog plus a valley haze that thickens below the course.
const FOG_PARS = `#include <fog_pars_fragment>\nvarying vec3 vWPos2;`;
const FOG_FRAG = `
  #ifdef USE_FOG
    float fd = vFogDepth;
    float ff = 1.0 - exp( - fogDensity * fogDensity * fd * fd );
    float haze = smoothstep(-200.0, -700.0, vWPos2.z) * 0.28 * smoothstep(150.0, 3000.0, fd);
    ff = clamp(ff + haze, 0.0, 1.0);
    gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, ff );
  #endif`;
const V_PARS = `#include <common>\nvarying vec3 vWPos2;`;
const V_BODY = `#include <begin_vertex>\nvWPos2 = (modelMatrix * vec4(position, 1.0)).xyz;`;

export class World {
  scene = new THREE.Scene();
  sun: THREE.DirectionalLight;
  terrain!: THREE.Mesh;
  gateGroups: { g: Gate; mats: THREE.MeshStandardMaterial[]; glow: THREE.Mesh[] }[] = [];
  trailL: Trail; trailR: Trail;
  spray: Spray;
  flakes: Flakes;
  confetti: Confetti;
  burst: Burst;
  crowd: Crowd | null = null;
  lift: Lift | null = null;
  pisteDist!: Float32Array;
  lowfx: boolean;

  constructor(public course: Course, far: FarTerrain | null, lowfx = false) {
    this.lowfx = lowfx;
    const sc = this.scene;
    sc.fog = new THREE.FogExp2(FOG.getHex(), 0.00014);
    sc.background = FOG;
    sc.add(new THREE.HemisphereLight('#9fbfee', '#d9e1ea', 0.6));
    this.sun = new THREE.DirectionalLight('#fff1dc', 3.1);
    this.sun.castShadow = !lowfx;
    this.sun.shadow.mapSize.set(lowfx ? 1024 : 2048, lowfx ? 1024 : 2048);
    const sc2 = this.sun.shadow.camera as THREE.OrthographicCamera;
    sc2.left = -50; sc2.right = 50; sc2.top = 50; sc2.bottom = -50; sc2.near = 1; sc2.far = 600;
    this.sun.shadow.bias = -0.0005; this.sun.shadow.normalBias = 0.04; this.sun.shadow.radius = 3;
    sc.add(this.sun); sc.add(this.sun.target);
    const fill = new THREE.DirectionalLight('#9fb8e0', 0.35); // cool bounce from the opposite side
    fill.position.set(-SUN_DIR.x * 100, -SUN_DIR.y * 100, 60); sc.add(fill);

    this.buildSky();
    if (far) this.buildFar(far); else this.buildRidges();
    this.buildTerrain();
    this.buildTrees();
    this.buildMarkers();
    this.buildFences();
    this.buildBanners();
    this.buildGates();
    this.buildStart();
    this.buildFinish();
    this.lift = new Lift(this, course);
    this.trailL = new Trail(sc); this.trailR = new Trail(sc);
    this.spray = new Spray(sc);
    this.flakes = new Flakes(sc, lowfx ? 250 : 700);
    this.confetti = new Confetti(sc);
    this.burst = new Burst(sc);
  }

  /** Keep the sun and its shadow box over the skier. */
  follow(x: number, y: number, z: number) {
    this.sun.position.set(x + SUN_DIR.x * 180, y + SUN_DIR.y * 180, z + SUN_DIR.z * 180);
    this.sun.target.position.set(x, y, z);
    this.sun.target.updateMatrixWorld();
  }

  private buildSky() {
    const geo = new THREE.SphereGeometry(9000, 48, 24);
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { sunDir: { value: SUN_DIR.clone() }, uTime: timeU },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform vec3 sunDir; uniform float uTime; varying vec3 vDir;
        float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
        float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
          return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }
        void main(){
          vec3 d = normalize(vDir);
          float h = clamp(d.z, -0.1, 1.0);
          vec3 zenith = vec3(0.16, 0.40, 0.80), mid = vec3(0.56, 0.74, 0.93), horizon = vec3(0.98, 0.86, 0.74), haze = vec3(0.93, 0.94, 0.96);
          vec3 c = h < 0.0 ? haze : h < 0.16 ? mix(horizon, mid, smoothstep(0.0, 0.16, h)) : mix(mid, zenith, smoothstep(0.16, 0.8, h));
          // warm glow around the sun, strongest near the horizon
          float s = max(dot(d, sunDir), 0.0);
          float hor = 1.0 - smoothstep(0.0, 0.35, h);
          c += vec3(1.0, 0.72, 0.42) * (pow(s, 4.0) * 0.16 + pow(s, 24.0) * 0.28) * (0.6 + 0.4 * hor);
          c += vec3(1.0, 0.95, 0.85) * pow(s, 400.0) * 2.5; // the disc
          // thin high cloud bands
          vec2 uv = d.xy / max(d.z + 0.15, 0.05);
          float cl = vnoise(uv * 1.3 + vec2(uTime * 0.004, 0.0)) * vnoise(uv * 3.1 - vec2(0.0, uTime * 0.002));
          cl = smoothstep(0.42, 0.75, cl) * smoothstep(0.02, 0.25, h) * (1.0 - smoothstep(0.55, 0.95, h));
          c = mix(c, vec3(0.99, 0.95, 0.93), cl * 0.55);
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    const sky = new THREE.Mesh(geo, mat);
    sky.frustumCulled = false;
    this.scene.add(sky);
    // Sun with a lens flare.
    const flareLight = new THREE.PointLight('#fff2e0', 0, 0);
    flareLight.position.copy(SUN_DIR).multiplyScalar(8000);
    const tex0 = flareTexture(256, 0.5, '#fff6e6'), tex1 = flareTexture(64, 0.25, '#ffd9a8');
    const flare = new Lensflare();
    flare.addElement(new LensflareElement(tex0, 260, 0, new THREE.Color('#fff1dc')));
    flare.addElement(new LensflareElement(tex1, 60, 0.45));
    flare.addElement(new LensflareElement(tex1, 90, 0.6));
    flare.addElement(new LensflareElement(tex1, 40, 0.9));
    flareLight.add(flare);
    this.scene.add(flareLight);
  }

  /** The real mountains around the course, coloured by altitude and slope. */
  private buildFar(far: FarTerrain) {
    const { nrow, ncol, cell, x0, y0 } = far;
    const n = nrow * ncol;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const c = this.course;
    const snowLine = 1850 - far.z_datum_msl, treeLine = 1750 - far.z_datum_msl; // MSL -> course frame
    // January: snow to the valley floor. Forest below the treeline is snow-dusted, not green.
    const forest = new THREE.Color('#8ea39b'), forestFar = new THREE.Color('#aebfd4'), rock = new THREE.Color('#9aa4b1'), snow = new THREE.Color('#f2f5fa'), meadow = new THREE.Color('#e3e9f1');
    const valley = new THREE.Color('#d7dfe9'), valleyLine = 950 - far.z_datum_msl;
    for (let r = 0; r < nrow; r++) for (let cc = 0; cc < ncol; cc++) {
      const k = r * ncol + cc;
      const x = x0 + cc * cell, y = y0 + r * cell, z = far.z[k] - 4; // a little under the near terrain
      pos[k * 3] = x; pos[k * 3 + 1] = y; pos[k * 3 + 2] = z;
      // slope from neighbours
      const zr = far.z[r * ncol + Math.min(ncol - 1, cc + 1)], zu = far.z[Math.min(nrow - 1, r + 1) * ncol + cc];
      const slope = Math.atan(Math.hypot(zr - far.z[k], zu - far.z[k]) / cell);
      const hgt = far.z[k];
      const tone = new THREE.Color();
      if (hgt > snowLine) tone.copy(snow).lerp(rock, THREE.MathUtils.smoothstep(slope, 0.55, 0.9));
      else if (hgt > treeLine) tone.copy(meadow).lerp(snow, THREE.MathUtils.smoothstep(hgt, treeLine, snowLine)).lerp(rock, THREE.MathUtils.smoothstep(slope, 0.6, 0.95));
      else tone.copy(forest).lerp(meadow, 0.35 * hash2(cc, r) + 0.25 * THREE.MathUtils.smoothstep(slope, 0.45, 0.0)).lerp(rock, THREE.MathUtils.smoothstep(slope, 0.7, 1.0));
      if (hgt < valleyLine) tone.lerp(valley, THREE.MathUtils.smoothstep(hgt, valleyLine, valleyLine - 120));
      // low valleys read bluer and lighter with distance haze baked in a little
      tone.lerp(forestFar, THREE.MathUtils.smoothstep(hgt, 200, -600) * 0.25);
      const nse = (hash2(cc * 0.7, r * 1.3) - 0.5) * 0.05;
      col[k * 3] = tone.r + nse; col[k * 3 + 1] = tone.g + nse; col[k * 3 + 2] = tone.b + nse;
    }
    // hole where the near terrain sits (shrunk by one cell so there is no gap)
    const hx0 = c.x0 + 40, hx1 = c.xMax - 40, hy0 = c.y0 + 40, hy1 = c.yMax - 40;
    const idx: number[] = [];
    for (let r = 0; r < nrow - 1; r++) for (let cc = 0; cc < ncol - 1; cc++) {
      const x = x0 + cc * cell, y = y0 + r * cell;
      if (x >= hx0 && x + cell <= hx1 && y >= hy0 && y + cell <= hy1) continue;
      const a = r * ncol + cc, b = a + 1, d = a + ncol, e = d + 1;
      idx.push(a, b, e, a, e, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, flatShading: true });
    mat.onBeforeCompile = sh => {
      sh.vertexShader = sh.vertexShader.replace('#include <common>', V_PARS).replace('#include <begin_vertex>', V_BODY);
      sh.fragmentShader = sh.fragmentShader.replace('#include <fog_pars_fragment>', FOG_PARS).replace('#include <fog_fragment>', FOG_FRAG);
    };
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
    const snowP = new THREE.Color('#f8fafd'), snowO = new THREE.Color('#e4ecf6');
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
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.0 });
    const dir = Math.atan2(445, 685);
    mat.onBeforeCompile = sh => {
      sh.uniforms.uDir = { value: new THREE.Vector2(Math.cos(dir), Math.sin(dir)) };
      sh.uniforms.uSun = { value: SUN_DIR.clone() };
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', V_PARS + '\nattribute float aPiste; varying float vPiste;')
        .replace('#include <begin_vertex>', V_BODY + '\nvPiste = aPiste;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nuniform vec2 uDir; uniform vec3 uSun; varying float vPiste;\nfloat h21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }\nfloat vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f); return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }')
        .replace('#include <fog_pars_fragment>', FOG_PARS)
        .replace('#include <fog_fragment>', FOG_FRAG)
        // wrap lighting: snow scatters light past the terminator, so the shade side is never black
        .replace('#include <lights_physical_pars_fragment>', ShaderChunk.lights_physical_pars_fragment.replace('float dotNL = saturate( dot( geometryNormal, directLight.direction ) );', 'float dotNL = saturate( ( dot( geometryNormal, directLight.direction ) + 0.3 ) / 1.3 );'))
        .replace('#include <color_fragment>', `#include <color_fragment>
          // groomer corduroy along the course on the piste, wind texture off it
          float along = dot(vWPos2.xy, vec2(-uDir.y, uDir.x));
          float cord = sin(along * 12.566) * 0.03 * vPiste;
          float wind = (vnoise(vWPos2.xy * 0.35) - 0.5) * 0.08 * (1.0 - vPiste) + (vnoise(vWPos2.xy * 2.3) - 0.5) * 0.03;
          diffuseColor.rgb += cord + wind;`)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
          // micro bumps so the snow is not glass-flat
          float b1 = vnoise(vWPos2.xy * 3.0), b2 = vnoise(vWPos2.xy * 3.0 + vec2(0.07, 0.0)), b3 = vnoise(vWPos2.xy * 3.0 + vec2(0.0, 0.07));
          normal = normalize(normal + vec3((b2 - b1), (b3 - b1), 0.0) * 1.2 * (1.0 - 0.6 * vPiste));`)
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          vec3 nrm = normalize(vNormal);
          float nl = dot(nrm, uSun);
          // snow in shade goes blue (sky light), snow facing the sun warms
          totalEmissiveRadiance += vec3(0.03, 0.07, 0.14) * (1.0 - smoothstep(-0.25, 0.55, nl));
          totalEmissiveRadiance += vec3(0.05, 0.03, 0.0) * smoothstep(0.5, 1.0, nl);
          // sparkle: sparse facets that light up when they face the eye and the sun
          vec3 vd = normalize(cameraPosition - vWPos2);
          float sp = step(0.991, h21(floor(vWPos2.xy * 24.0))) * pow(max(dot(nrm, normalize(vd + uSun)), 0.0), 48.0);
          totalEmissiveRadiance += vec3(sp) * 1.4;`);
    };
    mat.customProgramCacheKey = () => 'snow-v2';
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

  private buildTrees() {
    const c = this.course, nrow = c.nrow, ncol = c.ncol, cell = c.cell;
    const spots: { x: number; y: number; z: number; s: number; r: number }[] = [];
    for (let r = 1; r < nrow - 1; r += 2) for (let cc = 1; cc < ncol - 1; cc += 2) {
      const k = r * ncol + cc;
      if (this.pisteDist[k] < 28) continue;
      const h = hash2(cc, r);
      const density = 0.10 * THREE.MathUtils.smoothstep(this.pisteDist[k], 28, 60) + 0.03;
      if (h > density) continue;
      const x = c.x0 + cc * cell + (hash2(r, cc) - 0.5) * 3.5, y = c.y0 + r * cell + (hash2(cc + 9, r + 3) - 0.5) * 3.5;
      if (c.slopeDeg(x, y) > 42) continue;
      spots.push({ x, y, z: c.sample(x, y).h, s: 3.5 + hash2(cc * 3, r * 7) * 5, r: hash2(cc * 11, r * 5) * Math.PI * 2 });
    }
    // three tiers of foliage, each with a snow cap, plus a trunk
    const tiers = [[1.0, 0.0, 1.0], [0.78, 0.33, 0.8], [0.55, 0.6, 0.6]]; // radius scale, base height, height scale
    const greenMat = new THREE.MeshStandardMaterial({ color: '#2d5a45', roughness: 0.9, flatShading: true });
    const green2 = new THREE.MeshStandardMaterial({ color: '#365f4a', roughness: 0.9, flatShading: true });
    const whiteMat = new THREE.MeshStandardMaterial({ color: '#f1f5fa', roughness: 0.9, flatShading: true });
    const trunkMat = new THREE.MeshStandardMaterial({ color: '#4a3526', roughness: 1 });
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    const meshes: THREE.InstancedMesh[] = [];
    tiers.forEach(([rs, hb, hs], ti) => {
      const cone = new THREE.ConeGeometry(1, 1, 7); cone.translate(0, 0.5, 0); cone.rotateX(Math.PI / 2);
      const cap = new THREE.ConeGeometry(0.62, 0.42, 7); cap.translate(0, 0.86, 0); cap.rotateX(Math.PI / 2);
      const g = new THREE.InstancedMesh(cone, ti % 2 ? green2 : greenMat, spots.length);
      const w = new THREE.InstancedMesh(cap, whiteMat, spots.length);
      spots.forEach((t, i) => {
        q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), t.r);
        s.set(t.s * 0.42 * rs, t.s * 0.42 * rs, t.s * hs * 0.75); p.set(t.x, t.y, t.z + 0.5 + t.s * hb * 0.75);
        m.compose(p, q, s); g.setMatrixAt(i, m); w.setMatrixAt(i, m);
      });
      meshes.push(g, w);
    });
    const trunk = new THREE.CylinderGeometry(0.12, 0.18, 1, 5); trunk.translate(0, 0.5, 0); trunk.rotateX(Math.PI / 2);
    const brown = new THREE.InstancedMesh(trunk, trunkMat, spots.length);
    spots.forEach((t, i) => { s.set(1, 1, 1.2); p.set(t.x, t.y, t.z - 0.1); q.identity(); m.compose(p, q, s); brown.setMatrixAt(i, m); });
    meshes.push(brown);
    for (const im of meshes) { im.castShadow = !this.lowfx; im.receiveShadow = true; this.scene.add(im); }
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
    const mat = new THREE.MeshStandardMaterial({ map: tex, transparent: true, alphaTest: 0.3, side: THREE.DoubleSide, roughness: 0.9, color: '#ff6a00' });
    const pos: number[] = [], uv: number[] = [], idx: number[] = [];
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
      pos.push(ax, ay, az, bx, by, bz, bx, by, bz + H, ax, ay, az + H);
      uv.push(0, 0, 1, 0, 1, 1, 0, 1);
      idx.push(v, v + 1, v + 2, v, v + 2, v + 3); v += 4;
    }
    if (!pos.length) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(idx); geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mat); mesh.castShadow = false;
    this.scene.add(mesh);
  }

  /** Sponsor-style banners along the right of the piste. */
  private buildBanners() {
    const c = this.course, cl = c.meta.centerline;
    const texts = ['GROUND TRUTH', 'MHACKS 2026', 'SPACETIMEDB', 'KITZBÜHEL · STREIF', 'EVERY RUN GROWS THE DATASET'];
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
      const banner = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 0.8), new THREE.MeshStandardMaterial({ map: tex, side: THREE.DoubleSide, roughness: 0.7 }));
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

  private buildGates() {
    const c = this.course;
    const poleGeo = new THREE.CylinderGeometry(0.03, 0.03, 1, 8); poleGeo.translate(0, 0.5, 0); poleGeo.rotateX(Math.PI / 2);
    const poleMat = new THREE.MeshStandardMaterial({ color: '#f2f4f7', roughness: 0.5 });
    const poles = new THREE.InstancedMesh(poleGeo, poleMat, c.gates.length * 4);
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    let k = 0;
    const ph = c.meta.gates.params.pole_height ?? 1.3;
    for (const g of c.gates) {
      const colour = g.color === 'red' ? '#ff2d3f' : '#2f6bff';
      const mats: THREE.MeshStandardMaterial[] = [];
      const glow: THREE.Mesh[] = [];
      for (const pl of g.poles) { p.set(pl[0], pl[1], pl[2]); s.set(1, 1, ph); m.compose(p, q, s); poles.setMatrixAt(k++, m); }
      for (const pn of g.panels) {
        const mat = new THREE.MeshStandardMaterial({ color: colour, roughness: 0.6, side: THREE.DoubleSide, transparent: true, opacity: 1, emissive: new THREE.Color(colour), emissiveIntensity: 0 });
        // flutter
        mat.onBeforeCompile = sh => {
          sh.uniforms.uTime = timeU;
          sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uTime;')
            .replace('#include <begin_vertex>', '#include <begin_vertex>\nfloat fl = (position.x + 0.66) / 1.32; transformed.z += sin(uTime * 9.0 + position.x * 5.0 + position.y * 3.0) * 0.03 * fl; transformed.y += cos(uTime * 7.0 + position.x * 4.0) * 0.012 * fl;');
        };
        mats.push(mat);
        const geo = new THREE.PlaneGeometry(pn.width * 2.4, pn.height * 1.5, 6, 2);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(pn.center[0], pn.center[1], pn.center[2] + pn.height * 0.5);
        const nrm = new THREE.Vector3(pn.normal[0], pn.normal[1], 0).normalize();
        const across = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), nrm).normalize();
        mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(across, new THREE.Vector3(0, 0, 1), nrm));
        mesh.castShadow = !this.lowfx;
        this.scene.add(mesh);
        glow.push(mesh);
        const band = new THREE.Mesh(new THREE.PlaneGeometry(pn.width * 2.4, pn.height * 0.35), new THREE.MeshStandardMaterial({ color: '#ffffff', side: THREE.DoubleSide, transparent: true }));
        band.position.set(0, 0, 0.004); mesh.add(band); mats.push(band.material as THREE.MeshStandardMaterial);
      }
      this.gateGroups.push({ g, mats, glow });
    }
    poles.count = k; poles.castShadow = !this.lowfx;
    this.scene.add(poles);
  }

  updateGates(next: number, missed: boolean[]) {
    this.gateGroups.forEach((gg, i) => {
      const passed = i < next;
      const o = passed ? (missed[i] ? 0.25 : 0.45) : 1;
      for (const m of gg.mats) m.opacity = o;
      for (const m of gg.glow) (m.material as THREE.MeshStandardMaterial).emissiveIntensity = i === next ? 0.6 : i === next + 1 ? 0.2 : 0;
    });
  }

  /** Start house: a hut beside the line and a START banner over it. */
  private buildStart() {
    const c = this.course, i = c.startIndex, p = c.meta.centerline[i], h = c.headingAt(i);
    const along = new THREE.Vector3(Math.cos(h), Math.sin(h), 0), right = new THREE.Vector3(along.y, -along.x, 0);
    const hutPos = new THREE.Vector3(p[0], p[1], 0).addScaledVector(right, 5.5).addScaledVector(along, -4);
    hutPos.z = c.sample(hutPos.x, hutPos.y).h;
    const hut = new THREE.Group();
    const walls = new THREE.Mesh(new THREE.BoxGeometry(2.6, 2.2, 1.9), new THREE.MeshStandardMaterial({ color: '#5a4030', roughness: 0.9 }));
    walls.position.z = 0.95; hut.add(walls);
    const roof = new THREE.Mesh(new THREE.ConeGeometry(2.3, 1.1, 4), new THREE.MeshStandardMaterial({ color: '#eef2f7', roughness: 0.9, flatShading: true }));
    roof.rotation.x = Math.PI / 2; roof.rotation.y = Math.PI / 4; roof.position.z = 2.4; hut.add(roof);
    hut.position.copy(hutPos); hut.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), h);
    hut.traverse(o => { (o as THREE.Mesh).castShadow = !this.lowfx; });
    this.scene.add(hut);
    // banner across the start
    const tex = textTexture('START', '#0b1220', '#37e6a8', 1024, 192);
    const banner = new THREE.Mesh(new THREE.PlaneGeometry(7.5, 1.3), new THREE.MeshStandardMaterial({ map: tex, side: THREE.DoubleSide }));
    const bp = new THREE.Vector3(p[0], p[1], 0).addScaledVector(along, -1.5);
    bp.z = c.sample(bp.x, bp.y).h + 3.2;
    banner.position.copy(bp);
    banner.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, new THREE.Vector3(0, 0, 1), along));
    this.scene.add(banner);
    const poleGeo = new THREE.CylinderGeometry(0.06, 0.06, 4, 8); poleGeo.translate(0, 2, 0); poleGeo.rotateX(Math.PI / 2);
    for (const d of [-3.75, 3.75]) {
      const pole = new THREE.Mesh(poleGeo, new THREE.MeshStandardMaterial({ color: '#1b2233' }));
      const pp = new THREE.Vector3(p[0], p[1], 0).addScaledVector(along, -1.5).addScaledVector(right, d);
      pp.z = c.sample(pp.x, pp.y).h; pole.position.copy(pp); this.scene.add(pole);
    }
  }

  /** Finish arch, banner, and a crowd along the last stretch. */
  private buildFinish() {
    const f = this.course.finish;
    const [a, b] = f.poles;
    const mid = new THREE.Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
    const along = new THREE.Vector3(b[0] - a[0], b[1] - a[1], 0).normalize();
    const fn = new THREE.Vector3(f.line_normal[0], f.line_normal[1], 0).normalize();
    const pillarMat = new THREE.MeshStandardMaterial({ color: '#ff2d3f', roughness: 0.5 });
    for (const pl of [a, b]) {
      const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.6, 6, 12).translate(0, 3, 0).rotateX(Math.PI / 2), pillarMat);
      pillar.position.set(pl[0], pl[1], pl[2]); pillar.castShadow = !this.lowfx; this.scene.add(pillar);
    }
    const w = Math.hypot(b[0] - a[0], b[1] - a[1]) + 1.2;
    const beam = new THREE.Mesh(new THREE.BoxGeometry(w, 1.6, 1.2), new THREE.MeshStandardMaterial({ map: textTexture('FINISH', '#ff2d3f', '#ffffff', 1024, 256), roughness: 0.6 }));
    beam.position.copy(mid).add(new THREE.Vector3(0, 0, 6.2));
    beam.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(along, fn, new THREE.Vector3(0, 0, 1)));
    beam.castShadow = !this.lowfx;
    this.scene.add(beam);
    const sub = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.9, 0.7), new THREE.MeshBasicMaterial({ map: textTexture('GROUND TRUTH · KITZBÜHEL', '#0b1220', '#ffffff', 1024, 128), side: THREE.DoubleSide }));
    sub.position.copy(mid).add(new THREE.Vector3(0, 0, 5.1)).addScaledVector(fn, 0.7);
    sub.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(along, new THREE.Vector3(0, 0, 1), fn));
    this.scene.add(sub);
    this.crowd = new Crowd(this, this.course);
  }

  update(dt: number, camPos: THREE.Vector3, timeScale = 1) {
    timeU.value += dt;
    this.spray.update(dt);
    this.flakes.update(dt, camPos);
    this.confetti.update(dt);
    this.burst.update(dt);
    this.crowd?.update(dt);
    this.lift?.update(dt * timeScale);
  }
}

// ------------------------------------------------------------------------------------------ textures

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

function netTexture() {
  const cv = document.createElement('canvas'); cv.width = 128; cv.height = 64;
  const g = cv.getContext('2d')!;
  g.clearRect(0, 0, 128, 64);
  g.strokeStyle = '#ffffff'; g.lineWidth = 3;
  for (let x = 0; x <= 128; x += 16) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 64); g.stroke(); }
  for (let y = 0; y <= 64; y += 16) { g.beginPath(); g.moveTo(0, y); g.lineTo(128, y); g.stroke(); }
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, 128, 5); g.fillRect(0, 59, 128, 5);
  const tex = new THREE.CanvasTexture(cv); tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(2, 1);
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
    const towerMat = new THREE.MeshStandardMaterial({ color: '#6b7380', roughness: 0.6, metalness: 0.5 });
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
    const chairMat = new THREE.MeshStandardMaterial({ color: '#2b3340', roughness: 0.6, metalness: 0.4 });
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

/** Spectators along the last 70 m: billboard sprites that bounce when something happens. */
class Crowd {
  sprites: THREE.Sprite[] = [];
  phase: number[] = [];
  excite = 0;
  constructor(world: World, course: Course) {
    const cl = course.meta.centerline;
    const texs = ['#ff3b4e', '#3d7bff', '#ffb340', '#37e6a8', '#f3f6fb', '#8a5cff'].map(c => personTexture(c));
    const i1 = course.finishIndex;
    for (let i = Math.max(0, i1 - 36); i <= i1 + 3; i += 1) {
      const h = course.headingAt(i), nx = Math.sin(h), ny = -Math.cos(h);
      for (const side of [-1, 1]) {
        const count = 1 + Math.floor(hash2(i, side) * 3);
        for (let k = 0; k < count; k++) {
          const off = 9 + hash2(i * 3 + k, side * 7) * 6;
          const x = cl[i][0] + nx * side * off + (hash2(i, k) - 0.5) * 2, y = cl[i][1] + ny * side * off + (hash2(k, i) - 0.5) * 2;
          if (!course.inside(x, y)) continue;
          const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: texs[Math.floor(hash2(i * 7, k * 13) * texs.length)], transparent: true, alphaTest: 0.2 }));
          const sc = 1.15 + hash2(k, i * 5) * 0.25;
          sp.scale.set(sc * 0.55, sc, 1);
          sp.position.set(x, y, course.sample(x, y).h + sc * 0.5);
          sp.center.set(0.5, 0.5);
          world.scene.add(sp); this.sprites.push(sp); this.phase.push(hash2(i, k) * 6.28);
        }
      }
    }
  }
  cheer(strength = 1) { this.excite = Math.min(1.5, this.excite + strength); }
  update(dt: number) {
    this.excite = Math.max(0, this.excite - dt * 0.35);
    const t = timeU.value;
    for (let i = 0; i < this.sprites.length; i++) {
      const s = this.sprites[i];
      const bounce = (0.04 + 0.35 * this.excite) * Math.max(0, Math.sin(t * (6 + this.excite * 4) + this.phase[i]));
      s.position.z += (bounce - (s.userData.b ?? 0)); s.userData.b = bounce;
    }
  }
}

function personTexture(color: string) {
  const cv = document.createElement('canvas'); cv.width = 64; cv.height = 128;
  const g = cv.getContext('2d')!;
  g.fillStyle = '#2b2b33'; g.beginPath(); g.arc(32, 22, 12, 0, Math.PI * 2); g.fill(); // head (hat)
  g.fillStyle = color; g.beginPath(); g.roundRect(14, 36, 36, 50, 10); g.fill(); // jacket
  g.fillStyle = '#1b2233'; g.fillRect(18, 84, 12, 40); g.fillRect(34, 84, 12, 40); // legs
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ------------------------------------------------------------------------------------------ effects

export class Trail {
  static N = 900;
  geo = new THREE.BufferGeometry();
  pos = new Float32Array(Trail.N * 2 * 3);
  count = 0;
  mesh: THREE.Mesh;
  constructor(scene: THREE.Scene) {
    const idx: number[] = [];
    for (let i = 0; i < Trail.N - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setIndex(idx);
    this.geo.setDrawRange(0, 0);
    this.mesh = new THREE.Mesh(this.geo, new THREE.MeshBasicMaterial({ color: '#a9c0dd', transparent: true, opacity: 0.6, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, side: THREE.DoubleSide }));
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }
  reset() { this.count = 0; this.geo.setDrawRange(0, 0); }
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

export class Spray {
  static N = 2200;
  pos = new Float32Array(Spray.N * 3); vel = new Float32Array(Spray.N * 3); life = new Float32Array(Spray.N); size = new Float32Array(Spray.N);
  head = 0;
  points: THREE.Points;
  constructor(scene: THREE.Scene) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.points = new THREE.Points(geo, new THREE.PointsMaterial({ map: Spray.texture(), size: 0.7, transparent: true, opacity: 0.9, depthWrite: false, sizeAttenuation: true, color: '#ffffff' }));
    this.points.frustumCulled = false;
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
  emit(p: THREE.Vector3, v: THREE.Vector3, n: number, spread = 1.5) {
    for (let k = 0; k < n; k++) {
      const i = this.head; this.head = (this.head + 1) % Spray.N;
      this.pos[i * 3] = p.x; this.pos[i * 3 + 1] = p.y; this.pos[i * 3 + 2] = p.z;
      this.vel[i * 3] = v.x + (Math.random() - 0.5) * spread; this.vel[i * 3 + 1] = v.y + (Math.random() - 0.5) * spread; this.vel[i * 3 + 2] = v.z + Math.random() * spread * 1.2;
      this.life[i] = 0.45 + Math.random() * 0.6;
    }
  }
  update(dt: number) {
    for (let i = 0; i < Spray.N; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] -= dt;
      if (this.life[i] <= 0) { this.pos[i * 3 + 2] = -1e4; continue; }
      this.vel[i * 3 + 2] -= 9.81 * dt * 0.55;
      this.vel[i * 3] *= 0.97; this.vel[i * 3 + 1] *= 0.97;
      this.pos[i * 3] += this.vel[i * 3] * dt; this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt; this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
    }
    (this.points.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
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
