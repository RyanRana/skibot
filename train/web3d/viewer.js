// G1 Ski Live 3D: the server streams body poses (about 1 KB per frame at 50 Hz); everything is drawn here.
// World frame is MuJoCo's: x east (downhill on the training tiles), y north, z up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const coarse = matchMedia('(pointer: coarse)').matches;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ---------------------------------------------------------------------------------------------------------------
// Renderer, scene, sky, lights

const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: params.has('shot') });
renderer.setPixelRatio(Math.min(devicePixelRatio, coarse ? 1.5 : 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const HORIZON = new THREE.Color('#d0dff2');
const ZENITH = new THREE.Color('#3a78d4');
scene.fog = new THREE.Fog(HORIZON, 70, 560);
const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.08, 4000);
camera.position.set(-6, 0, 4);

const sunDir = new THREE.Vector3(0.35, -0.55, 0.62).normalize();
const sun = new THREE.DirectionalLight(0xfff1de, 3.1);
sun.castShadow = true;
sun.shadow.mapSize.set(coarse ? 1024 : 2048, coarse ? 1024 : 2048);
Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 220 });
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.025;
scene.add(sun, sun.target);
const hemi = new THREE.HemisphereLight(0xb4ccf5, 0xe9e6df, 0.75);
scene.add(hemi);

const NOISE_GLSL = /* glsl */`
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int k = 0; k < 4; k++){ s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }
`;

// Sky: zenith-to-horizon gradient, sun glow and a ring of hazy snow-capped ranges on the horizon.
const skyMat = new THREE.ShaderMaterial({
  uniforms: { uSun: { value: sunDir }, uZen: { value: ZENITH }, uHor: { value: HORIZON } },
  vertexShader: /* glsl */`varying vec3 vDir; void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    uniform vec3 uSun; uniform vec3 uZen; uniform vec3 uHor; varying vec3 vDir;
    ${NOISE_GLSL}
    float ridge(float az, float seed, float f){ return vnoise(vec2(az * f, seed)) ; }
    void main(){
      vec3 d = normalize(vDir);
      float h = d.z;
      vec3 col = mix(uHor, uZen, pow(smoothstep(-0.03, 0.42, h), 0.55));
      float az = atan(d.y, d.x) / 6.2831853 + 0.5;
      // two layers of distant ranges: far (paler) then nearer (bluer, with snow caps)
      float r1 = 0.030 + 0.050 * ridge(az, 1.0, 9.0) + 0.018 * ridge(az, 2.0, 31.0) + 0.006 * ridge(az, 3.0, 97.0);
      float r2 = 0.012 + 0.032 * ridge(az, 4.0, 14.0) + 0.012 * ridge(az, 5.0, 47.0) + 0.004 * ridge(az, 6.0, 131.0);
      vec3 far = mix(uHor, vec3(0.62, 0.71, 0.86), 0.5);
      vec3 cap = vec3(0.96, 0.975, 1.0);
      if (h < r1) col = mix(far, cap, smoothstep(r1 - 0.022, r1 - 0.003, h) * 0.75);
      if (h < r2) col = mix(mix(uHor, vec3(0.52, 0.62, 0.80), 0.55), cap, smoothstep(r2 - 0.018, r2 - 0.002, h) * 0.8);
      col = mix(col, uHor, smoothstep(0.03, -0.02, h));
      float s = max(dot(d, uSun), 0.0);
      col += vec3(1.0, 0.93, 0.80) * (pow(s, 900.0) * 6.0 + pow(s, 24.0) * 0.20 + pow(s, 4.0) * 0.06);
      gl_FragColor = vec4(col, 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`,
  side: THREE.BackSide, depthWrite: false, fog: false,
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(3000, 48, 24), skyMat);
sky.frustumCulled = false;
sky.renderOrder = -10;
scene.add(sky);

// Image-based light for the robot's metal (sky above, bright snow below).
{
  const pm = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  const envSky = new THREE.Mesh(new THREE.SphereGeometry(10, 32, 16), skyMat.clone());
  envSky.material.uniforms = { uSun: { value: sunDir }, uZen: { value: ZENITH }, uHor: { value: HORIZON } };
  envScene.add(envSky);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(9, 32), new THREE.MeshBasicMaterial({ color: 0xe9eef6 }));
  floor.position.z = -1.2;
  envScene.add(floor);
  scene.environment = pm.fromScene(envScene, 0.02).texture;
  pm.dispose();
}

// ---------------------------------------------------------------------------------------------------------------
// Snow: soft white-blue, drifts and grain, bluish on steep faces, rock on cliffs, sparkle near the camera.

const snowU = { uKind: { value: 0 }, uFall: { value: new THREE.Vector2(1, 0) }, uLocalFall: { value: 0 } };
const SNOW_KIND = { groomed: 0, hardpack: 0, powder: 1, soft: 1, ice: 2, slush: 3 };
function snowMaterial() {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.88, metalness: 0.0, envMapIntensity: 0.3 });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uSun = { value: sunDir };
    Object.assign(sh.uniforms, snowU);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNrm;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWNrm = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNrm;\nuniform vec3 uSun;\nuniform float uKind;\nuniform vec2 uFall;\nuniform float uLocalFall;\n' + NOISE_GLSL)
      .replace('#include <color_fragment>', /* glsl */`#include <color_fragment>
        {
          float dist = length(cameraPosition - vWPos);
          float steep = 1.0 - clamp(normalize(vWNrm).z, 0.0, 1.0);
          float big = fbm(vWPos.xy * 0.03);
          float mid = fbm(vWPos.xy * 0.29 + 7.0);
          float fine = vnoise(vWPos.xy * 3.3);
          float nearF = smoothstep(90.0, 15.0, dist), fineF = smoothstep(25.0, 4.0, dist);
          vec3 c = vec3(0.955, 0.972, 1.0);
          c *= 0.935 + 0.085 * big + (0.045 * mid - 0.022) * nearF + (0.03 * fine - 0.015) * fineF;
          c = mix(c, vec3(0.80, 0.86, 0.97), smoothstep(0.07, 0.34, steep) * 0.6);
          // groomer corduroy along the fall line (groomed and hardpack), seams between groomer passes
          vec2 lf = -normalize(vWNrm).xy;
          vec2 fd = (uLocalFall > 0.5 && length(lf) > 0.06) ? normalize(lf) : uFall;
          float ph = dot(vWPos.xy, vec2(-fd.y, fd.x));
          float cordF = (uKind < 0.5 ? 1.0 : 0.0) * smoothstep(16.0, 3.0, dist);
          c *= 1.0 - cordF * (0.03 * (0.5 + 0.5 * sin(ph * 6.2831853 / 0.075)) + 0.025 * smoothstep(0.9, 1.0, abs(fract(ph / 4.2) * 2.0 - 1.0)));
          if (uKind > 1.5 && uKind < 2.5) c = mix(c, vec3(0.80, 0.88, 0.98), 0.35 + 0.25 * big);   // ice: blue sheen
          if (uKind > 2.5) c *= vec3(0.93, 0.935, 0.95) * (0.96 + 0.08 * mid);                       // slush: grey, wet
          if (uKind > 0.5 && uKind < 1.5) c = mix(c, vec3(1.0), 0.25);                                 // powder: brighter
          float rock = smoothstep(0.42, 0.62, steep + 0.12 * (big - 0.5));
          c = mix(c, vec3(0.42, 0.43, 0.46) * (0.8 + 0.4 * mid), rock * 0.9);
          diffuseColor.rgb *= c;
        }`)
      .replace('#include <normal_fragment_maps>', /* glsl */`#include <normal_fragment_maps>
        {
          float bdist = length(cameraPosition - vWPos);
          float bf = smoothstep(70.0, 6.0, bdist);
          if (bf > 0.001) {
            vec2 bp = vWPos.xy * 0.8;
            float e = 0.12;
            float n0 = vnoise(bp) + 0.5 * vnoise(bp * 2.7 + 3.1);
            float nx = vnoise(bp + vec2(e, 0.0)) + 0.5 * vnoise((bp + vec2(e, 0.0)) * 2.7 + 3.1);
            float ny = vnoise(bp + vec2(0.0, e)) + 0.5 * vnoise((bp + vec2(0.0, e)) * 2.7 + 3.1);
            vec2 g = vec2(nx - n0, ny - n0) / e * 0.075 * bf;
            normal = normalize(normal + (viewMatrix * vec4(-g.x, -g.y, 0.0, 0.0)).xyz);
          }
        }`)
      .replace('#include <opaque_fragment>', /* glsl */`
        {
          float dist = length(cameraPosition - vWPos);
          vec3 V = normalize(cameraPosition - vWPos);
          vec2 cell = floor(vWPos.xy * 34.0);
          float h = hash12(cell);
          vec3 fn = normalize(vec3(hash12(cell + 11.3) - 0.5, hash12(cell + 27.1) - 0.5, 0.9));
          float g = pow(max(dot(reflect(-uSun, fn), V), 0.0), 160.0) * step(uKind > 0.5 && uKind < 1.5 ? 0.82 : 0.9, h);
          g *= smoothstep(26.0, 3.0, dist) * clamp(dot(normalize(vWNrm), uSun) * 3.0, 0.0, 1.0);
          outgoingLight += vec3(1.0, 0.97, 0.9) * g * 2.4;
        }
        #include <opaque_fragment>`);
  };
  return m;
}
const snowMat = snowMaterial();

// ---------------------------------------------------------------------------------------------------------------
// Terrain: a detailed patch (the real heightfield) inside a coarse mountain valley that shares its border.

let world = null; // { near, far, trees, treeGrid, kind, fall, meshes: Group }
const worldGroup = new THREE.Group();
scene.add(worldGroup);

function gridHeight(g, x, y) {
  // Same two triangles per cell as the mesh (split along (r,c)-(r+1,c+1)), so tracks and the camera sit on it.
  const fx = (x - g.x0) / g.cell, fy = (y - g.y0) / g.cell;
  const c = clamp(Math.floor(fx), 0, g.ncol - 2), r = clamp(Math.floor(fy), 0, g.nrow - 2);
  const u = clamp(fx - c, 0, 1), v = clamp(fy - r, 0, 1);
  const z = g.z, i = r * g.ncol + c;
  const z00 = z[i], z10 = z[i + 1], z01 = z[i + g.ncol], z11 = z[i + g.ncol + 1];
  return u >= v ? z00 + (z10 - z00) * u + (z11 - z10) * v : z00 + (z11 - z01) * u + (z01 - z00) * v;
}
function inGrid(g, x, y) {
  return x >= g.x0 && y >= g.y0 && x <= g.x0 + (g.ncol - 1) * g.cell && y <= g.y0 + (g.nrow - 1) * g.cell;
}
function heightAt(x, y) {
  if (!world) return 0;
  return inGrid(world.near, x, y) ? gridHeight(world.near, x, y) : gridHeight(world.far, x, y);
}

function gridMesh(g, ox, oy, hole) {
  const { nrow, ncol, cell, z } = g;
  const pos = new Float32Array(nrow * ncol * 3), nrm = new Float32Array(nrow * ncol * 3);
  for (let r = 0; r < nrow; r++) {
    for (let c = 0; c < ncol; c++) {
      const i = r * ncol + c, j = 3 * i;
      pos[j] = g.x0 + c * cell - ox; pos[j + 1] = g.y0 + r * cell - oy; pos[j + 2] = z[i];
      const zl = z[r * ncol + Math.max(c - 1, 0)], zr = z[r * ncol + Math.min(c + 1, ncol - 1)];
      const zd = z[Math.max(r - 1, 0) * ncol + c], zu = z[Math.min(r + 1, nrow - 1) * ncol + c];
      const dx = (zr - zl) / ((Math.min(c + 1, ncol - 1) - Math.max(c - 1, 0)) * cell);
      const dy = (zu - zd) / ((Math.min(r + 1, nrow - 1) - Math.max(r - 1, 0)) * cell);
      const l = Math.hypot(dx, dy, 1);
      nrm[j] = -dx / l; nrm[j + 1] = -dy / l; nrm[j + 2] = 1 / l;
    }
  }
  const idx = [];
  for (let r = 0; r < nrow - 1; r++) {
    for (let c = 0; c < ncol - 1; c++) {
      if (hole && c >= hole.c0 && c < hole.c1 && r >= hole.r0 && r < hole.r1) continue;
      const a = r * ncol + c, b = a + 1, cc = a + ncol + 1, d = a + ncol;
      idx.push(a, b, cc, a, cc, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  geo.setIndex(nrow * ncol > 65535 ? new THREE.Uint32BufferAttribute(idx, 1) : new THREE.Uint16BufferAttribute(idx, 1));
  geo.computeBoundingSphere();
  const mesh = new THREE.Mesh(geo, snowMat);
  mesh.position.set(ox, oy, 0);
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------------------------------------------------------------------------------------------------------
// 8-bit pines (port of skisim.draw.voxel_tree_boxes): stepped green boxes, a trunk, snow on the upper steps.

const GREENS = [[0.06, 0.25, 0.12], [0.09, 0.33, 0.15], [0.13, 0.42, 0.18], [0.19, 0.50, 0.22]];
const TRUNK = [0.40, 0.25, 0.13];
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function voxelTree(x, y, z, r, h, solid, snow) {
  const rnd = mulberry32(Math.floor(Math.abs(x * 73.856 + y * 19.349) * 1000) % 2147483647);
  const yaw = rnd() * Math.PI / 2, shade = rnd() < 0.5 ? 0 : 1;
  const n = clamp(Math.round(h / 1.1), 3, 6);
  const trunkH = 0.17 * h, tipH = 0.08 * h, t = (h - trunkH - tipH) / n;
  const w0 = 1.45 * r, step = (w0 - 0.28 * w0) / Math.max(n - 1, 1);
  let tw = 0.15 * r + 0.06;
  solid.push([tw, tw, trunkH / 2 + 0.12, x, y, z + trunkH / 2 - 0.06, yaw, TRUNK]);
  const snowFrom = Math.max(1, Math.floor(n / 2) - 1);
  for (let k = 0; k < n; k++) {
    const w = w0 - k * step, zb = z + trunkH + k * t;
    solid.push([w, w, t / 2, x, y, zb + t / 2, yaw, GREENS[clamp(Math.round(k / Math.max(n - 1, 1) * 2) + shade, 0, 3)]]);
    if (k >= snowFrom) { const s = Math.max(0.06, 0.2 * t); snow.push([w + 0.025, w + 0.025, s / 2, x, y, zb + t - 0.3 * s + s / 2, yaw, null]); }
  }
  tw = Math.max(0.12, 0.45 * (w0 - (n - 1) * step));
  snow.push([tw, tw, tipH / 2 + 0.05, x, y, z + h - tipH / 2, yaw, null]);
}
const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const treeSolidMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
const treeSnowMat = new THREE.MeshLambertMaterial({ color: 0xf2f6ff, emissive: 0x4a5160 });
function instanced(boxes, mat, shadow, ox, oy) {
  const m = new THREE.InstancedMesh(boxGeo, mat, Math.max(boxes.length, 1));
  m.count = boxes.length;
  const M = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(), zAx = new THREE.Vector3(0, 0, 1);
  const col = new THREE.Color();
  boxes.forEach((b, i) => {
    q.setFromAxisAngle(zAx, b[6]);
    M.compose(p.set(b[3] - ox, b[4] - oy, b[5]), q, s.set(2 * b[0], 2 * b[1], 2 * b[2]));
    m.setMatrixAt(i, M);
    if (b[7]) m.setColorAt(i, col.setRGB(b[7][0], b[7][1], b[7][2], THREE.SRGBColorSpace));
  });
  m.position.set(ox, oy, 0);
  m.castShadow = shadow; m.receiveShadow = true;
  m.computeBoundingSphere();
  return m;
}

// ---------------------------------------------------------------------------------------------------------------
// Messages

function parseEnvelope(buf) {
  const dv = new DataView(buf);
  const n = dv.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  return { meta, base: 8 + n, buf };
}

function buildWorld(env) {
  const { meta, base, buf } = env;
  const grid = (g) => ({ ...g, z: new Float32Array(buf, base + g.offset, g.nrow * g.ncol) });
  const near = grid(meta.near), far = grid(meta.far);
  const k = meta.k;
  // Stitch near's border to far's coarser edge so the two meshes meet without cracks.
  const zn = new Float32Array(near.z);
  const lerpEdge = (get, set, n) => { for (let i = 0; i < n; i++) { if (i % k) { const a = i - (i % k), b = a + k; const t = (i - a) / k; set(i, get(a) * (1 - t) + get(b) * t); } } };
  const W = near.ncol, H = near.nrow;
  lerpEdge((i) => zn[i], (i, v) => { zn[i] = v; }, W);
  lerpEdge((i) => zn[(H - 1) * W + i], (i, v) => { zn[(H - 1) * W + i] = v; }, W);
  lerpEdge((i) => zn[i * W], (i, v) => { zn[i * W] = v; }, H);
  lerpEdge((i) => zn[i * W + W - 1], (i, v) => { zn[i * W + W - 1] = v; }, H);
  near.z = zn;
  const c0 = Math.round((near.x0 - far.x0) / far.cell), r0 = Math.round((near.y0 - far.y0) / far.cell);
  const hole = { c0, r0, c1: c0 + (near.ncol - 1) / k, r1: r0 + (near.nrow - 1) / k };
  const T = new Float32Array(buf, base + meta.trees.offset, meta.trees.count * meta.trees.stride);
  const trees = [];
  for (let i = 0; i < meta.trees.count; i++) { const o = i * 6; trees.push({ x: T[o], y: T[o + 1], z: T[o + 2], r: T[o + 3], h: T[o + 4], real: T[o + 5] > 0.5 }); }

  disposeGroup(worldGroup);
  const ox = Math.round(near.x0 + (near.ncol - 1) * near.cell / 2), oy = Math.round(near.y0 + (near.nrow - 1) * near.cell / 2);
  world = { near, far, trees, kind: meta.kind, fall: meta.fall, name: meta.name, meta, ox, oy };
  worldGroup.add(gridMesh(near, ox, oy, null));
  worldGroup.add(gridMesh(far, ox, oy, hole));
  const real = { solid: [], snow: [] }, decor = { solid: [], snow: [] };
  for (const t of trees) voxelTree(t.x, t.y, t.z, t.r, t.h, (t.real ? real : decor).solid, (t.real ? real : decor).snow);
  worldGroup.add(instanced(real.solid, treeSolidMat, true, ox, oy), instanced(real.snow, treeSnowMat, true, ox, oy));
  worldGroup.add(instanced(decor.solid, treeSolidMat, false, ox, oy), instanced(decor.snow, treeSnowMat, false, ox, oy));
  // Spatial hash of canopies for keeping the camera out of trees.
  world.treeGrid = new Map();
  for (const t of trees) {
    const key = Math.floor(t.x / 8) * 100003 + Math.floor(t.y / 8);
    if (!world.treeGrid.has(key)) world.treeGrid.set(key, []);
    world.treeGrid.get(key).push(t);
  }
  // Sun from behind-left of a skier facing down the fall line, 38 degrees up: lit robot, relief on the snow.
  const fa = Math.atan2(meta.fall[1], meta.fall[0]) + 2.3, el = 0.66;
  sunDir.set(Math.cos(el) * Math.cos(fa), Math.cos(el) * Math.sin(fa), Math.sin(el)).normalize();
  if (meta.kind === 'course' && !camChosen) setChase(true);
  snowU.uKind.value = SNOW_KIND[meta.snow] ?? 0;
  snowU.uFall.value.set(meta.fall[0], meta.fall[1]);
  snowU.uLocalFall.value = meta.kind === 'course' ? 1 : 0;
  snowMat.roughness = meta.snow === 'ice' ? 0.55 : 0.88;
  tracks.forEach((tr) => tr.clear());
  camSnap = true;
  console.log(`world ${meta.name}: near ${near.nrow}x${near.ncol}, far ${far.nrow}x${far.ncol}, ${trees.length} trees`);
}

function disposeGroup(g) {
  for (const c of [...g.children]) {
    g.remove(c);
    c.traverse?.((o) => { if (o.geometry && o.geometry !== boxGeo && o.geometry !== poleGeo && o.geometry !== panelGeo) o.geometry.dispose(); });
    if (c.dispose) c.dispose();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Gates: red and blue poles in pairs with a panel between each pair; passed gates fade.

const gateGroup = new THREE.Group();
scene.add(gateGroup);
const poleGeo = new THREE.CylinderGeometry(0.022, 0.022, 1.31, 10).rotateX(Math.PI / 2).translate(0, 0, 0.655);
const panelGeo = new THREE.BoxGeometry(1, 0.012, 0.37);
const gateMats = [0xd7262e, 0x2554d8].map((c) => [
  new THREE.MeshStandardMaterial({ color: c, roughness: 0.5 }),
  new THREE.MeshStandardMaterial({ color: c, roughness: 0.5, transparent: true, opacity: 0.28, depthWrite: false }),
]);
let gates = [];
function buildGates(list) {
  disposeGroup(gateGroup);
  gates = list.map((g) => {
    const grp = new THREE.Group();
    const mat = gateMats[g.c][0];
    for (const p of g.p) {
      const m = new THREE.Mesh(poleGeo, mat);
      m.position.set(p[0], p[1], p[2] - 0.05);
      m.castShadow = true;
      grp.add(m);
    }
    for (const [i, j] of g.panels || []) {
      const a = g.p[i], b = g.p[j];
      const m = new THREE.Mesh(panelGeo, mat);
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      m.scale.set(L, 1, 1);
      m.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2 + 0.92);
      m.rotation.z = Math.atan2(b[1] - a[1], b[0] - a[0]);
      m.castShadow = true;
      grp.add(m);
    }
    gateGroup.add(grp);
    const n = g.p.length;
    const cx = g.p.reduce((a, q) => a + q[0], 0) / n, cy = g.p.reduce((a, q) => a + q[1], 0) / n, cz = g.p.reduce((a, q) => a + q[2], 0) / n;
    const half = Math.max(...g.p.map((q) => Math.hypot(q[0] - cx, q[1] - cy)));
    return { grp, c: g.c, passed: false, center: new THREE.Vector3(cx, cy, cz + 0.65), half };
  });
}
function hideNearGates() {
  for (const g of gates) g.grp.visible = camera.position.distanceTo(g.center) > g.half + 1.6;
}
function markGates(next) {
  gates.forEach((g, k) => {
    const passed = k < next;
    if (passed !== g.passed) {
      g.passed = passed;
      g.grp.traverse((o) => { if (o.isMesh) { o.material = gateMats[g.c][passed ? 1 : 0]; o.castShadow = !passed; } });
    }
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Ski tracks: one ribbon per ski, laid on the snow from the ski soles while the ski is loaded.

class Track {
  constructor(max = 5000) {
    this.max = max;
    this.pos = new Float32Array(max * 2 * 3);
    this.idx = new Uint32Array((max - 1) * 6);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    const up = new Float32Array(max * 2 * 3); for (let i = 2; i < up.length; i += 3) up[i] = 1;
    this.geo.setAttribute('normal', new THREE.BufferAttribute(up, 3));
    this.geo.setIndex(new THREE.BufferAttribute(this.idx, 1).setUsage(THREE.DynamicDrawUsage));
    this.mesh = new THREE.Mesh(this.geo, trackMat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    this.clear();
  }
  clear() { this.n = 0; this.ni = 0; this.last = null; this.broken = true; this.geo.setDrawRange(0, 0); this.dirty = null; this.full = true; }
  cut() { this.broken = true; this.last = null; }
  add(x, y) {
    if (!world) return;
    if (this.last) {
      const dx = x - this.last[0], dy = y - this.last[1], L = Math.hypot(dx, dy);
      if (L < 0.12) return;
      if (L > 2.5) this.broken = true;
    }
    if (this.n >= this.max) this.compact();
    const prev = this.last;
    let ux = 1, uy = 0;
    if (prev) { const L = Math.hypot(x - prev[0], y - prev[1]); ux = (x - prev[0]) / L; uy = (y - prev[1]) / L; }
    const w = 0.032, i = this.n, j = 6 * i;
    const z = heightAt(x, y) + 0.012;
    this.pos[j] = x - uy * w; this.pos[j + 1] = y + ux * w; this.pos[j + 2] = z;
    this.pos[j + 3] = x + uy * w; this.pos[j + 4] = y - ux * w; this.pos[j + 5] = z;
    const d = this.dirty || (this.dirty = { p0: j, p1: j + 6, i0: this.ni, i1: this.ni });
    d.p1 = j + 6;
    if (!this.broken && i > 0) {
      const a = 2 * (i - 1), b = 2 * i;
      this.idx.set([a, a + 1, b + 1, a, b + 1, b], this.ni);
      this.ni += 6;
      d.i1 = this.ni;
    }
    this.broken = false;
    this.last = [x, y];
    this.n++;
  }
  flush() { // upload only what changed since the last frame
    const pa = this.geo.attributes.position, ia = this.geo.index;
    if (this.full) {
      pa.clearUpdateRanges(); ia.clearUpdateRanges();
      pa.needsUpdate = true; ia.needsUpdate = true;
    } else if (this.dirty) {
      const d = this.dirty;
      pa.addUpdateRange(d.p0, d.p1 - d.p0); pa.needsUpdate = true;
      if (d.i1 > d.i0) { ia.addUpdateRange(d.i0, d.i1 - d.i0); ia.needsUpdate = true; }
    }
    this.full = false; this.dirty = null;
    this.geo.setDrawRange(0, this.ni);
  }
  compact() { // keep the newest half
    const h = this.max >> 1;
    this.pos.copyWithin(0, h * 6, this.max * 6);
    const off = 2 * h, keep = [];
    for (let q = 0; q < this.ni; q += 6) if (this.idx[q] >= off) for (let r = 0; r < 6; r++) keep.push(this.idx[q + r] - off);
    this.idx.fill(0); this.idx.set(keep); this.ni = keep.length; this.n -= h;
    this.full = true; this.dirty = null;
  }
}
const trackMat = new THREE.MeshLambertMaterial({ color: 0xa9bad6, transparent: true, opacity: 0.32, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
const tracks = [new Track(), new Track()];
tracks.forEach((t) => scene.add(t.mesh));

// ---------------------------------------------------------------------------------------------------------------
// Robot: one group per body, meshes from /model.bin, posed every frame.

let robot = null; // { group, bodies: [Group], pelvis }
const ROBOT_MATS = {
  silver: new THREE.MeshStandardMaterial({ color: 0xc3c6cc, metalness: 0.6, roughness: 0.34 }),
  black: new THREE.MeshStandardMaterial({ color: 0x1c1e22, metalness: 0.3, roughness: 0.5 }),
  ski_red: new THREE.MeshStandardMaterial({ color: 0xd3211b, metalness: 0.15, roughness: 0.28 }),
};
async function loadModel() {
  const res = await fetch('/model.bin');
  const buf = await res.arrayBuffer();
  const { meta, base } = parseEnvelope(buf);
  const geoms = {};
  for (const [id, m] of Object.entries(meta.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, base + m.v, m.nv * 3), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array(buf, base + m.f, m.nf * 3), 1));
    g.computeVertexNormals();
    geoms[id] = g;
  }
  const group = new THREE.Group();
  const bodies = meta.bodies.map((b) => {
    const bg = new THREE.Group();
    bg.name = b.name;
    for (const e of b.geoms) {
      let geo;
      const s = e.size;
      if (e.type === 'mesh') geo = geoms[e.mesh];
      else if (e.type === 'box') geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      else if (e.type === 'capsule') geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 4, 12).rotateX(Math.PI / 2);
      else if (e.type === 'cylinder') geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 16).rotateX(Math.PI / 2);
      else if (e.type === 'sphere') geo = new THREE.SphereGeometry(s[0], 16, 12);
      else if (e.type === 'ellipsoid') geo = new THREE.SphereGeometry(1, 16, 12).scale(s[0], s[1], s[2]);
      else continue;
      const mat = ROBOT_MATS[e.mat] || new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(e.rgba[0], e.rgba[1], e.rgba[2], THREE.SRGBColorSpace), roughness: 0.5 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.fromArray(e.pos);
      mesh.quaternion.set(e.quat[1], e.quat[2], e.quat[3], e.quat[0]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      bg.add(mesh);
    }
    group.add(bg);
    return bg;
  });
  group.visible = false;
  scene.add(group);
  robot = { group, bodies, pelvis: bodies[meta.bodies.findIndex((b) => b.name === 'pelvis')] };
  // Ski poles (visual only, as in skisim.draw.draw_ski_poles): grip in each hand, shaft trailing down and back.
  robot.poles = ['left', 'right'].map((side) => {
    const hand = bodies[meta.bodies.findIndex((b) => b.name === `${side}_wrist_yaw_link`)];
    const pole = new THREE.Group();
    const part = (geo, color, y) => { const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.3 })); m.position.y = y; m.castShadow = true; pole.add(m); };
    part(new THREE.CylinderGeometry(0.009, 0.009, POLE_LEN, 8), 0x2a2d31, POLE_LEN / 2 - 0.05);
    part(new THREE.CylinderGeometry(0.016, 0.016, 0.12, 10), 0x0d0e10, 0);
    part(new THREE.CylinderGeometry(0.04, 0.04, 0.008, 16), 0x15171a, POLE_LEN - 0.12);
    group.add(pole);
    return { hand, pole, sgn: side === 'left' ? 1 : -1 };
  }).filter((p) => p.hand);
  labels = meta.labels || labels;
  console.log(`robot: ${bodies.length} bodies, ${Object.keys(meta.meshes).length} meshes`);
}

// ---------------------------------------------------------------------------------------------------------------
// Frame stream: a short buffer, rendered a few frames behind the newest with interpolation.

let labels = ['Start'];
const frames = [];
let clock = null; // stream seconds being shown
let rate = 1; // stream seconds per wall second, measured
let gap = 0.02; // smoothed wall gap between frames
let lastRecv = 0;
const F_GATES = 1, F_CRASH = 2, F_REC = 4, F_REPLAY = 8, F_COURSE = 16;

function parseFrame(buf) {
  const dv = new DataView(buf);
  const nb = dv.getUint8(3);
  return {
    flags: dv.getUint8(1), label: dv.getUint8(2), seq: dv.getUint32(4, true), t: dv.getFloat64(8, true), srv: dv.getFloat64(16, true),
    disc: dv.getUint32(24, true), gateNext: dv.getUint16(28, true), hits: dv.getUint16(30, true), crossed: dv.getUint16(32, true),
    run: dv.getUint16(34, true), speed: dv.getFloat32(36, true), fade: dv.getFloat32(40, true), heading: dv.getFloat32(44, true),
    pose: new Float32Array(buf, 48, nb * 7), ski: new Float32Array(buf, 48 + nb * 28, 8), recv: performance.now(), nb,
  };
}

const stat = { n: 0, bytes: 0, t0: performance.now(), hz: 0, kb: 0, net: null, rtt: null, offset: null };
function onFrame(buf) {
  const f = parseFrame(buf);
  if (frames.length) {
    const p = frames[frames.length - 1];
    if (f.t <= p.t) frames.length = 0; // stream restarted
    else {
      const wall = (f.recv - p.recv) / 1000;
      gap += (wall - gap) * 0.08;
      if (f.disc === p.disc && wall > 0) rate += (clamp((f.t - p.t) / Math.max(gap, 1e-3), 0.05, 4) - rate) * 0.05;
    }
  }
  frames.push(f);
  if (frames.length > 200) frames.splice(0, frames.length - 200);
  stat.n++; stat.bytes += buf.byteLength;
  if (stat.offset !== null) stat.net = Date.now() + stat.offset - f.srv;
  lastRecv = f.recv;
}

const q0 = new THREE.Quaternion(), q1 = new THREE.Quaternion();
const POLE_LEN = 0.85, POLE_X = new THREE.Vector3(), POLE_G = new THREE.Vector3(), POLE_U = new THREE.Vector3(), POLE_Y = new THREE.Vector3(0, 1, 0);
let shown = null; // the interpolated state we rendered
function sampleFrames(dt) {
  if (!frames.length) return null;
  const newest = frames[frames.length - 1];
  const lagTarget = clamp(0.035 + 1.6 * gap * rate, 0.05, 0.3);
  if (clock === null || clock > newest.t || newest.t - clock > 0.6) clock = newest.t - lagTarget;
  else {
    const err = (newest.t - clock) - lagTarget;
    clock += dt * rate * (1 + clamp(err * 3, -0.6, 0.6));
    clock = Math.min(clock, newest.t);
  }
  let i = frames.length - 1;
  while (i > 0 && frames[i - 1].t > clock) i--;
  const b = frames[i], a = i > 0 ? frames[i - 1] : b;
  let alpha = a === b ? 1 : clamp((clock - a.t) / (b.t - a.t), 0, 1);
  if (a.disc !== b.disc) alpha = 1; // respawn or seek: no smear across it
  // drop frames we will never need again
  if (i > 3) frames.splice(0, i - 3);
  return { a, b, alpha, lag: newest.t - clock };
}

function applyPose(s) {
  if (!robot) return;
  const { a, b, alpha } = s;
  const n = Math.min(robot.bodies.length, b.nb);
  for (let k = 0; k < n; k++) {
    const o = 7 * k, g = robot.bodies[k];
    const pa = a.pose, pb = b.pose;
    g.position.set(pa[o] + (pb[o] - pa[o]) * alpha, pa[o + 1] + (pb[o + 1] - pa[o + 1]) * alpha, pa[o + 2] + (pb[o + 2] - pa[o + 2]) * alpha);
    q0.set(pa[o + 4], pa[o + 5], pa[o + 6], pa[o + 3]);
    q1.set(pb[o + 4], pb[o + 5], pb[o + 6], pb[o + 3]);
    g.quaternion.slerpQuaternions(q0, q1, alpha);
  }
  if (robot.poles && robot.pelvis) {
    const fwd = POLE_X.set(1, 0, 0).applyQuaternion(robot.pelvis.quaternion).setZ(0).normalize();
    for (const p of robot.poles) {
      const grip = POLE_G.set(0.08, 0, 0).applyQuaternion(p.hand.quaternion).add(p.hand.position);
      const u = POLE_U.set(-0.55 * fwd.x - p.sgn * 0.12 * fwd.y, -0.55 * fwd.y + p.sgn * 0.12 * fwd.x, -0.82).normalize();
      p.pole.position.copy(grip);
      p.pole.quaternion.setFromUnitVectors(POLE_Y, u);
    }
  }
  robot.group.visible = true;
}

// ---------------------------------------------------------------------------------------------------------------
// Camera: orbit around the skier (target follows smoothly, user pan kept as an offset) or an automatic chase cam.
// Never below the snow, never inside a tree, and the skier stays in sight over bumps.

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.minDistance = 1.0;
controls.maxDistance = 160;
controls.maxPolarAngle = Math.PI * 0.62;
controls.zoomSpeed = 0.9;
controls.target.set(0, 0, 0);
let chase = params.get('cam') === 'chase';
let camChosen = params.has('cam'); // the user picked a camera: keep it
let camSnap = true;
let chaseDist = 5.2, chaseHeight = 1.9;
const panOffset = new THREE.Vector3();
const travel = new THREE.Vector3(1, 0, 0);
const prevPelvis = new THREE.Vector3();
const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();
let lastDisc = -1;
// The clearance push is applied only for drawing: the orbit state keeps the user's own offset, so lifts over bumps
// never accumulate. lift rises at once (never below the snow) and eases back down.
const desired = new THREE.Vector3();
const pushed = new THREE.Vector3();
let lift = 0, pull = 1, chasePull = 1;
let restore = false;

function treePull(pos, look) {
  // Fraction of the way from the skier to the camera before the sight line enters a canopy (1 = clear).
  if (!world) return 1;
  const dx = pos.x - look.x, dy = pos.y - look.y, dz = pos.z - look.z, L2 = dx * dx + dy * dy;
  if (L2 < 1e-6) return 1;
  let tmin = 1;
  const x0 = Math.floor((Math.min(pos.x, look.x) - 3) / 8), x1 = Math.floor((Math.max(pos.x, look.x) + 3) / 8);
  const y0 = Math.floor((Math.min(pos.y, look.y) - 3) / 8), y1 = Math.floor((Math.max(pos.y, look.y) + 3) / 8);
  for (let cx = x0; cx <= x1; cx++) for (let cy = y0; cy <= y1; cy++) {
    for (const t of world.treeGrid.get(cx * 100003 + cy) || []) {
      const R = 2.1 * t.r + 0.35, fx = look.x - t.x, fy = look.y - t.y;
      const c = fx * fx + fy * fy - R * R;
      if (c < 0) continue; // the skier is beside this trunk: handled by the push-out below
      const b = fx * dx + fy * dy, disc = b * b - L2 * c;
      if (disc < 0) continue;
      const th = (-b - Math.sqrt(disc)) / L2;
      if (th < 0 || th >= tmin || look.z + th * dz > t.z + t.h + 0.4) continue; // behind us, farther, or over the top
      tmin = th;
    }
  }
  return tmin;
}

function keepCameraClear(pos, look) {
  if (!world) return;
  // out of tree canopies (a cylinder a bit wider than the widest voxel layer)
  for (let pass = 0; pass < 2; pass++) { // twice: a push out of one canopy can land in a neighbour's
  const cx = Math.floor(pos.x / 8), cy = Math.floor(pos.y / 8);
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
    const list = world.treeGrid.get((cx + i) * 100003 + (cy + j));
    if (!list) continue;
    for (const t of list) {
      if (pos.z > t.z + t.h + 0.35) continue;
      const R = 2.1 * t.r + 0.35, dx = pos.x - t.x, dy = pos.y - t.y, d = Math.hypot(dx, dy);
      if (d < R) { const s = R / Math.max(d, 1e-3); pos.x = t.x + dx * s; pos.y = t.y + dy * s; if (d < 1e-3) pos.x += R; }
    }
  }
  }
  { // still inside overlapping canopies: go over the top instead
    const cx = Math.floor(pos.x / 8), cy = Math.floor(pos.y / 8);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
      for (const t of world.treeGrid.get((cx + i) * 100003 + (cy + j)) || []) {
        if (Math.hypot(pos.x - t.x, pos.y - t.y) < 2.1 * t.r + 0.35 && pos.z < t.z + t.h + 0.4) pos.z = t.z + t.h + 0.4;
      }
    }
  }
  const clear = 0.65;
  const h = heightAt(pos.x, pos.y);
  if (pos.z < h + clear) pos.z = h + clear;
  for (const f of [0.25, 0.5, 0.75]) { // line of sight to the skier
    const px = pos.x + (look.x - pos.x) * f, py = pos.y + (look.y - pos.y) * f, pz = pos.z + (look.z - pos.z) * f;
    const need = heightAt(px, py) + 0.25 + 0.4 * (1 - f);
    if (pz < need) pos.z += (need - pz) / (1 - f);
  }
}

function updateCamera(dt, s) {
  if (!robot || !robot.pelvis) return;
  const p = robot.pelvis.position;
  const disc = s ? s.b.disc : lastDisc;
  if (disc !== lastDisc) { camSnap = true; lastDisc = disc; }
  // travel direction (smoothed) for the chase cam
  tmp.subVectors(p, prevPelvis); tmp.z = 0;
  const L = tmp.length();
  if (!camSnap && L > 0.02 && L < 3) travel.lerp(tmp.divideScalar(L), 1 - Math.exp(-dt * 2.5)).normalize();
  else if (camSnap && world) {
    const q = robot.pelvis.quaternion; tmp2.set(1, 0, 0).applyQuaternion(q); tmp2.z = 0;
    if (tmp2.lengthSq() > 1e-4) travel.copy(tmp2.normalize()); else travel.set(world.fall[0], world.fall[1], 0);
  }
  prevPelvis.copy(p);
  const follow = tmp.copy(p).add(panOffset); follow.z -= 0.15;
  if (chase) {
    const want = tmp2.copy(p).addScaledVector(travel, -chaseDist); want.z += chaseHeight;
    const look = new THREE.Vector3().copy(p).addScaledVector(travel, 1.2); look.z += 0.1;
    if (camSnap) camera.position.copy(want); else camera.position.lerp(want, 1 - Math.exp(-dt * 3.5));
    const tp = treePull(camera.position, look);
    chasePull = Math.min(tp, chasePull + dt * 0.6);
    if (chasePull < 0.999) camera.position.lerpVectors(look, camera.position, Math.max(chasePull - 0.04, 0.12));
    keepCameraClear(camera.position, look);
    controls.target.copy(look);
    camera.lookAt(look);
    restore = false;
  } else {
    if (restore) camera.position.copy(desired); // back to the user's orbit before the controls move it
    if (camSnap) {
      if (restore) camera.position.copy(desired);
      restore = false;
      lift = 0; pull = 1;
      const off = tmp2.subVectors(camera.position, controls.target);
      if (off.lengthSq() < 0.01 || !Number.isFinite(off.x) || firstView) {
        off.copy(travel).multiplyScalar(-6.5).addScaledVector(new THREE.Vector3(-travel.y, travel.x, 0), 2.4);
        off.z = Math.max(heightAt(follow.x + off.x, follow.y + off.y) + 2.4 - follow.z, 1.2);
        firstView = false;
      }
      controls.target.copy(follow);
      camera.position.copy(follow).add(off);
    } else {
      const k = 1 - Math.exp(-dt * 9);
      const d = tmp2.subVectors(follow, controls.target).multiplyScalar(k);
      controls.target.add(d);
      camera.position.add(d);
    }
    const before = controls.target.clone();
    controls.update();
    panOffset.add(before.sub(controls.target).negate()); // whatever the user panned this frame
    desired.copy(camera.position);
    pushed.copy(desired);
    const tp = treePull(pushed, controls.target);
    pull = Math.min(tp, pull + dt * 0.6);
    if (pull < 0.999) pushed.lerpVectors(controls.target, desired, Math.max(pull - 0.04, 0.12));
    const baseZ = pushed.z; // after the pull-in; the lift below is measured from here
    keepCameraClear(pushed, controls.target);
    lift = Math.max(pushed.z - baseZ, lift - dt * 1.5);
    camera.position.set(pushed.x, pushed.y, baseZ + lift);
    keepCameraClear(camera.position, controls.target); // the eased lift may still be short after an xy push
    camera.lookAt(controls.target);
    restore = true;
  }
  camSnap = false;
  // the sun's shadow box travels with the skier
  sun.target.position.copy(p);
  sun.position.copy(p).addScaledVector(sunDir, 80);
  sun.target.updateMatrixWorld();
}
let firstView = true;
canvas.addEventListener('dblclick', () => { panOffset.set(0, 0, 0); });
canvas.addEventListener('wheel', (e) => { if (chase) { chaseDist = clamp(chaseDist * (e.deltaY > 0 ? 1.1 : 0.9), 2, 40); e.preventDefault(); } }, { passive: false });
function setChase(on) {
  chase = on;
  controls.enabled = !on;
  $('cam').setAttribute('aria-pressed', on);
  if (!on && robot) { controls.target.copy(robot.pelvis.position); panOffset.set(0, 0, 0); restore = false; lift = 0; }
}

// ---------------------------------------------------------------------------------------------------------------
// Websocket

let ws = null, hello = null, hud = {};
let ready = { model: false, world: false };
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { setLoad('Connected', 'waiting for the simulation'); ping(); };
  ws.onclose = () => { setLoad('Reconnecting', 'the simulator is not reachable'); $('loading').classList.remove('done'); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') return onText(JSON.parse(ev.data));
    const kind = new DataView(ev.data).getUint8(0);
    if (kind === 1) onFrame(ev.data);
    else if (kind === 2) { buildWorld(parseEnvelope(ev.data)); ready.world = true; checkReady(); }
  };
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
function ping() { send({ cmd: 'ping', t: Date.now() }); }
setInterval(ping, 2000);
const pings = [];
function onText(d) {
  if (d.type === 'hello') { hello = d; fillRecordings(d.recordings || []); if (d.source === 'starting') setLoad('Starting the simulation', 'building the env and loading the newest checkpoint'); }
  else if (d.type === 'gates') { buildGates(d.gates); markGates(d.next || 0); }
  else if (d.type === 'hud') { hud = d; updateHudSlow(); }
  else if (d.type === 'recordings') fillRecordings(d.files);
  else if (d.type === 'pong') {
    const now = Date.now(), rtt = now - d.t;
    pings.push({ rtt, off: d.srv - (d.t + now) / 2 });
    if (pings.length > 10) pings.shift();
    const best = pings.reduce((m, p) => (p.rtt < m.rtt ? p : m), pings[0]);
    stat.rtt = rtt; stat.offset = best.off;
  }
}
function setLoad(a, b) { $('loadmsg').textContent = a; $('loadsub').textContent = b || ''; }
function checkReady() { if (ready.model && ready.world) $('loading').classList.add('done'); }

// ---------------------------------------------------------------------------------------------------------------
// HUD and controls

let fps = 60, fpsN = 0, fpsT = performance.now();
let lastMode = null;
const domCache = new Map();
function setText(id, v) { if (domCache.get(id) !== v) { domCache.set(id, v); $(id).textContent = v; } }
function setAttr(id, a, v) { const key = id + '@' + a; if (domCache.get(key) !== v) { domCache.set(key, v); if (a === 'opacity') $(id).style.opacity = v; else $(id).setAttribute(a, v); } }
let lastLabel = -1;
function updateHudFast(s) {
  const f = s.b;
  const speed = s.a.speed + (f.speed - s.a.speed) * s.alpha;
  setText('kmh', String(Math.round(speed * 3.6)));
  if (f.label !== lastLabel) {
    lastLabel = f.label;
    const name = labels[f.label] || '';
    $('label').textContent = name;
    $('move').classList.toggle('crash', name === 'Crash');
  }
  const gatesOn = f.flags & (F_GATES | F_COURSE);
  setText('gates', gatesOn ? `${f.hits} of ${f.crossed}${f.flags & F_COURSE && gates.length ? ` / ${gates.length}` : ''}` : 'free steer');
  markGates(f.gateNext);
  setAttr('fade', 'opacity', (s.a.fade + (f.fade - s.a.fade) * s.alpha).toFixed(2));
  setAttr('rec', 'aria-pressed', String(!!(f.flags & F_REC)));
}
function updateHudSlow() {
  const h = hud;
  $('policy').textContent = h.iter != null ? `iter ${h.iter}${h.pending ? ` (next ${h.pending})` : ''}` : (h.policy || '--');
  $('runinfo').textContent = h.mode === 'replay' ? `frame ${h.frame} / ${h.frames}` : `#${h.run ?? '-'} · ${h.distance ?? 0} m · falls ${h.falls ?? 0}`;
  $('where').textContent = h.where || '';
  const live = h.mode !== 'replay';
  $('replaybar').classList.toggle('on', !live);
  for (const id of ['mode', 'prev', 'next', 'reset']) $(id).style.display = live && h.mode !== 'course' ? '' : 'none';
  $('reset').style.display = live ? '' : 'none';
  $('rec').style.display = live ? '' : 'none';
  $('pad').style.display = live ? '' : 'none';
  $('live').style.display = hello && hello.live ? '' : 'none';
  if (h.mode !== lastMode) { lastMode = h.mode; padLabels(); }
  if (live && h.mode !== 'course') {
    const gm = h.mode === 'gates';
    if (cmd.gates !== gm && performance.now() - lastPad > 1500) { cmd.gates = gm; padLabels(); }
  }
  if (!live) {
    $('pause').setAttribute('aria-pressed', !!h.paused); $('pause').textContent = h.paused ? 'Play' : 'Pause';
    if (!seeking && h.frames) $('seek').value = Math.round(1000 * h.frame / Math.max(h.frames - 1, 1));
  }
  if (h.recording) $('rec').textContent = `Rec ${Math.round(h.rec_s)}s`; else $('rec').textContent = 'Rec';
}
function updatePerf(s) {
  const now = performance.now();
  fpsN++;
  if (now - fpsT > 500) {
    fps = fpsN * 1000 / (now - fpsT); fpsN = 0; fpsT = now;
    const el = (now - stat.t0) / 1000; stat.hz = stat.n / el; stat.kb = stat.bytes / Math.max(stat.n, 1) / 1024; stat.n = 0; stat.bytes = 0; stat.t0 = now;
    $('fps').textContent = `${fps.toFixed(0)} fps · ${stat.hz.toFixed(0)} Hz`;
    const parts = [`${stat.kb.toFixed(2)} KB/frame`];
    if (stat.net != null) parts.push(`net ${Math.max(0, stat.net).toFixed(0)} ms`);
    if (stat.rtt != null) parts.push(`rtt ${stat.rtt} ms`);
    if (s) parts.push(`buffer ${(s.lag * 1000).toFixed(0)} ms`);
    if (hud.rt) parts.push(`sim ${hud.rt}x`);
    if (hud.step_ms) parts.push(`step ${hud.step_ms} ms`);
    if (hud.send_ms != null) parts.push(`send ${hud.send_ms} ms`);
    $('perf').textContent = parts.join(' · ');
    window.__stats = { fps, hz: stat.hz, kb: stat.kb, net: stat.net, rtt: stat.rtt, lag: s ? s.lag : null, rate, gap, hud };
  }
}

// Steering pad: gates mode -> left/right course curviness (next run), up/down speed. Free -> steer and speed.
const cmd = { gates: true, heading: 0, padX: 0, speed: 9, amp: 3 };
function steerSign() { // +heading turns toward +y; map "pad right" to "screen right" from wherever the camera looks
  const r = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  return Math.abs(r.y) > 0.25 ? Math.sign(r.y) : -1;
}
let lastSign = -1;
setInterval(() => { const sgn = steerSign(); if (sgn !== lastSign) { lastSign = sgn; if (!cmd.gates && cmd.padX) { cmd.heading = cmd.padX * sgn; sendSteer(); } } }, 250);
let lastPad = 0;
const pad = $('pad'), dot = $('dot');
function padLabels() {
  const course = hud.mode === 'course';
  $('padlab').textContent = course ? 'up/down: speed' : cmd.gates ? 'curviness (next run) / speed' : 'steer / speed';
  $('mode').textContent = cmd.gates ? 'Gates' : 'Free steer';
  $('mode').setAttribute('aria-pressed', cmd.gates);
  placeDot();
}
function placeDot() {
  const r = pad.getBoundingClientRect();
  const fx = hud.mode === 'course' ? 0.5 : cmd.gates ? (cmd.amp - 1.5) / 3.5 : (cmd.padX + 1) / 2;
  dot.style.left = `${fx * r.width}px`;
  dot.style.top = `${(1 - (cmd.speed - 3) / 11) * r.height}px`;
}
function padMove(e) {
  const r = pad.getBoundingClientRect();
  const x = clamp((e.clientX - r.left) / r.width, 0, 1), y = clamp((e.clientY - r.top) / r.height, 0, 1);
  if (cmd.gates) cmd.amp = +(1.5 + x * 3.5).toFixed(2); else { cmd.padX = +((x * 2 - 1)).toFixed(3); cmd.heading = cmd.padX * steerSign(); }
  cmd.speed = +(3 + (1 - y) * 11).toFixed(2);
  lastPad = performance.now();
  placeDot();
  sendSteer();
}
let steerTimer = null;
function sendSteer() { if (steerTimer) return; steerTimer = setTimeout(() => { steerTimer = null; send({ cmd: 'steer', heading: cmd.heading, speed: cmd.speed, amp: cmd.amp }); }, 30); }
pad.addEventListener('pointerdown', (e) => { pad.setPointerCapture(e.pointerId); padMove(e); e.preventDefault(); });
pad.addEventListener('pointermove', (e) => { if (e.buttons || e.pointerType === 'touch') padMove(e); });
pad.addEventListener('pointerup', () => { if (!cmd.gates) { cmd.padX = 0; cmd.heading = 0; placeDot(); sendSteer(); } });

$('cam').onclick = () => { camChosen = true; setChase(!chase); };
$('mode').onclick = () => { cmd.gates = !cmd.gates; lastPad = performance.now(); padLabels(); send({ cmd: 'mode', gates: cmd.gates }); };
$('prev').onclick = () => send({ cmd: 'tile', d: -1 });
$('next').onclick = () => send({ cmd: 'tile', d: 1 });
$('reset').onclick = () => send({ cmd: 'reset' });
$('rec').onclick = () => send({ cmd: 'record' });
$('live').onclick = () => send({ cmd: 'replay', file: null });
$('pause').onclick = () => send({ cmd: 'replay_ctl', pause: !hud.paused });
$('rspeed').onchange = (e) => send({ cmd: 'replay_ctl', speed: +e.target.value });
let seeking = false;
$('seek').addEventListener('pointerdown', () => { seeking = true; });
$('seek').addEventListener('change', (e) => { seeking = false; send({ cmd: 'replay_ctl', seek: +e.target.value / 1000 }); });
$('recs').addEventListener('focus', () => send({ cmd: 'list' }));
$('recs').onchange = (e) => { if (e.target.value) send({ cmd: 'replay', file: e.target.value }); e.target.value = ''; };
function fillRecordings(files) {
  const sel = $('recs');
  sel.innerHTML = '<option value="">Replays</option>' + files.map((f) => `<option value="${f.name}">${f.name} (${f.mb} MB)</option>`).join('');
}
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key;
  if (k === 'c' || k === 'C') { camChosen = true; setChase(!chase); }
  else if (k === 'g' || k === 'G') $('mode').click();
  else if (k === 'r' || k === 'R') send({ cmd: 'reset' });
  else if (k === '[') send({ cmd: 'tile', d: -1 });
  else if (k === ']') send({ cmd: 'tile', d: 1 });
  else if (k === ' ' && hud.mode === 'replay') send({ cmd: 'replay_ctl', pause: !hud.paused });
  else if (k.startsWith('Arrow')) {
    if (k === 'ArrowLeft' || k === 'ArrowRight') {
      const d = k === 'ArrowLeft' ? -1 : 1; // screen left / right
      if (cmd.gates) cmd.amp = clamp(cmd.amp + 0.25 * d, 1.5, 5); else { cmd.padX = clamp(cmd.padX + 0.25 * d, -1, 1); cmd.heading = cmd.padX * steerSign(); }
    } else cmd.speed = clamp(cmd.speed + (k === 'ArrowUp' ? 1 : -1), 3, 14);
    lastPad = performance.now(); placeDot(); sendSteer(); e.preventDefault();
  }
});

// ---------------------------------------------------------------------------------------------------------------
// Loop

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  placeDot();
}
addEventListener('resize', resize);

let lastT = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
  const s = sampleFrames(dt);
  if (s) {
    applyPose(s);
    if (s.b.disc !== shown?.disc) { tracks.forEach((t) => t.cut()); }
    for (let k = 0; k < 2; k++) {
      const o = 4 * k, A = s.a.ski, B = s.b.ski, al = s.alpha;
      const load = A[o + 3] + (B[o + 3] - A[o + 3]) * al;
      if (load > 20) tracks[k].add(A[o] + (B[o] - A[o]) * al, A[o + 1] + (B[o + 1] - A[o + 1]) * al);
      else tracks[k].cut();
    }
    updateHudFast(s);
    shown = { disc: s.b.disc };
  }
  tracks.forEach((t) => t.flush());
  updateCamera(dt, s);
  hideNearGates();
  sky.position.copy(camera.position);
  renderer.render(scene, camera);
  updatePerf(s);
}

setLoad('Loading the robot', 'meshes from /model.bin');
loadModel().then(() => { ready.model = true; checkReady(); }).catch((e) => setLoad('Could not load the robot model', String(e)));
connect();
padLabels();
setChase(chase);
requestAnimationFrame(frame);
window.__viewer = {
  scene, camera, controls, frames, heightAt, keepCameraClear,
  get world() { return world; }, get robot() { return robot; },
  setOrbit(dist, azDeg, elDeg) { // for scripted views: distance, azimuth from +x, elevation, around the skier
    const t = controls.target, a = azDeg * Math.PI / 180, e = elDeg * Math.PI / 180;
    camera.position.set(t.x + dist * Math.cos(e) * Math.cos(a), t.y + dist * Math.cos(e) * Math.sin(a), t.z + dist * Math.sin(e));
    restore = false; lift = 0; setChase(false);
  },
};
