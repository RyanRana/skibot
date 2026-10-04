// Post-processing: a soft bloom on sun and sparkle, and a speed pass that pulls the edges of the frame
// into a radial blur with a touch of chromatic aberration and vignette as you get faster. MSAA is kept
// through the composer with a multisampled render target, and OutputPass applies tone mapping. After it, in
// display space, a grade: cool shadows, warm highlights, a little more colour, a gentle S-curve and fine
// grain, so the frame reads like a winter broadcast instead of a flat render.
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

const SpeedShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uSpeed: { value: 0 },      // 0..1
    uFlash: { value: 0 },      // 0..1 white flash (gate hit)
    uFlashColor: { value: new THREE.Color('#ffffff') },
    uDim: { value: 0 },        // 0..1 darken + desaturate (wipeout, finish freeze)
    uAspect: { value: 16 / 9 },
    uTime: { value: 0 },
    uBeat: { value: 0 },       // 0..1 heartbeat thump (Presage)
    uTunnel: { value: 0 },     // 0..1 how hard the heart is racing
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uSpeed, uFlash, uDim, uAspect, uTime, uBeat, uTunnel;
    float hash(float n){ return fract(sin(n) * 43758.5453); } uniform vec3 uFlashColor; varying vec2 vUv;
    void main(){
      vec2 c = vec2(0.5, 0.56);
      vec2 d = vUv - c; d.x *= uAspect;
      float r = length(d);
      float edge = smoothstep(0.25, 0.95, r);
      float amt = uSpeed * uSpeed * edge;
      vec2 dir = (vUv - c) * 0.045 * amt;
      // radial blur toward the centre, more at the edges and at speed
      vec4 col = vec4(0.0);
      const int N = 7;
      for (int i = 0; i < N; i++) { float t = float(i) / float(N - 1); col += texture2D(tDiffuse, vUv - dir * t); }
      col /= float(N);
      // chromatic aberration at speed
      vec2 ca = (vUv - c) * 0.006 * uSpeed * edge;
      col.r = texture2D(tDiffuse, vUv - dir * 0.5 + ca).r * 0.5 + col.r * 0.5;
      col.b = texture2D(tDiffuse, vUv - dir * 0.5 - ca).b * 0.5 + col.b * 0.5;
      // speed lines streaking in from the edges
      float ang = atan(d.y, d.x);
      float lines = step(0.982, hash(floor(ang * 160.0) + floor(uTime * 22.0))) * smoothstep(0.4, 0.95, r) * uSpeed * uSpeed;
      col.rgb += lines * 0.45;
      // vignette
      float vig = 1.0 - smoothstep(0.55, 1.25, r) * (0.35 + 0.35 * uSpeed);
      col.rgb *= vig;
      // your heartbeat at the edges of the frame: a thump on every beat, a red tunnel when it races
      float edgeB = smoothstep(0.45, 1.15, r);
      col.rgb *= 1.0 - edgeB * (0.05 + 0.25 * uTunnel) * uBeat - edgeB * 0.22 * uTunnel;
      col.rgb += vec3(0.20, 0.0, 0.03) * edgeB * uBeat * uTunnel;
      // dim / desaturate
      float l = dot(col.rgb, vec3(0.299, 0.587, 0.114));
      col.rgb = mix(col.rgb, vec3(l) * 0.6, uDim);
      // flash
      col.rgb = mix(col.rgb, uFlashColor, uFlash * 0.75);
      gl_FragColor = col;
    }`,
};

const GradeShader = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null }, uTime: { value: 0 }, uAmount: { value: 1 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uAmount; varying vec2 vUv;
    float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec3 c0 = texture2D(tDiffuse, vUv).rgb, c = c0;
      float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = mix(vec3(l), c, 1.1);                                              // a little more colour
      c += vec3(-0.012, 0.004, 0.034) * (1.0 - smoothstep(0.0, 0.55, l));    // cool shadows
      c *= mix(vec3(1.0), vec3(1.035, 1.0, 0.955), smoothstep(0.55, 1.0, l)); // warm highlights
      c = mix(c, c * c * (3.0 - 2.0 * c), 0.12);                             // gentle S-curve
      c += (h(vUv * 1024.0 + fract(uTime) * 61.0) - 0.5) * 0.018;            // fine grain
      gl_FragColor = vec4(mix(c0, clamp(c, 0.0, 1.0), uAmount), 1.0);
    }`,
};

export class Post {
  composer: EffectComposer;
  speed: ShaderPass;
  bloom: UnrealBloomPass;
  grade: ShaderPass;
  private flash = 0; private flashColor = new THREE.Color('#ffffff');
  dim = 0;
  beat = 0; tunnel = 0;
  constructor(public renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, lowfx: boolean) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const target = new THREE.WebGLRenderTarget(size.x, size.y, { samples: lowfx ? 0 : 4, type: THREE.HalfFloatType });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x / 2, size.y / 2), lowfx ? 0.1 : 0.18, 0.4, 2.05 /* above any sunlit white; only emitters bloom */);
    if (!new URLSearchParams(location.search).has('nobloom')) this.composer.addPass(this.bloom);
    this.speed = new ShaderPass(SpeedShader);
    this.composer.addPass(this.speed);
    this.composer.addPass(new OutputPass());
    this.grade = new ShaderPass(GradeShader);
    if (!new URLSearchParams(location.search).has('nograde')) this.composer.addPass(this.grade);
    this.setSize(innerWidth, innerHeight);
  }
  setSize(w: number, h: number) {
    this.composer.setSize(w, h);
    this.speed.uniforms.uAspect.value = w / h;
    this.bloom.setSize(w / 2, h / 2);
  }
  /** One-frame flash of colour, fades over ~250 ms. */
  hit(color: string | THREE.Color = '#ffffff', strength = 1) { this.flash = Math.max(this.flash, strength); this.flashColor.set(color as any); }
  render(dt: number, speed01: number) {
    this.flash = Math.max(0, this.flash - dt * 4);
    this.speed.uniforms.uSpeed.value = speed01;
    this.speed.uniforms.uFlash.value = this.flash;
    this.speed.uniforms.uFlashColor.value.copy(this.flashColor);
    this.speed.uniforms.uDim.value = this.dim;
    this.speed.uniforms.uBeat.value = this.beat;
    this.speed.uniforms.uTunnel.value = this.tunnel;
    this.speed.uniforms.uTime.value += dt;
    this.grade.uniforms.uTime.value += dt;
    this.composer.render(dt);
  }
}
