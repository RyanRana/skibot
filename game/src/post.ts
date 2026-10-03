// Post-processing: a soft bloom on sun and sparkle, and a speed pass that pulls the edges of the frame
// into a radial blur with a touch of chromatic aberration and vignette as you get faster. MSAA is kept
// through the composer with a multisampled render target, and OutputPass applies tone mapping.
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
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uSpeed, uFlash, uDim, uAspect, uTime;
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
      // dim / desaturate
      float l = dot(col.rgb, vec3(0.299, 0.587, 0.114));
      col.rgb = mix(col.rgb, vec3(l) * 0.6, uDim);
      // flash
      col.rgb = mix(col.rgb, uFlashColor, uFlash * 0.75);
      gl_FragColor = col;
    }`,
};

export class Post {
  composer: EffectComposer;
  speed: ShaderPass;
  bloom: UnrealBloomPass;
  private flash = 0; private flashColor = new THREE.Color('#ffffff');
  dim = 0;
  constructor(public renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, lowfx: boolean) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const target = new THREE.WebGLRenderTarget(size.x, size.y, { samples: lowfx ? 0 : 4, type: THREE.HalfFloatType });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x / 2, size.y / 2), lowfx ? 0.16 : 0.26, 0.5, 0.86);
    this.composer.addPass(this.bloom);
    this.speed = new ShaderPass(SpeedShader);
    this.composer.addPass(this.speed);
    this.composer.addPass(new OutputPass());
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
    this.speed.uniforms.uTime.value += dt;
    this.composer.render(dt);
  }
}
