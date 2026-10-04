// Procedural sound and a race announcer. Wind rises with speed, an edge hisses when it scrubs snow,
// gates thwack, cowbells ring from the crowd, the start beeps like a real race, and the finish gets a
// fanfare. The announcer uses the browser's speech synthesis (no API key needed).
export class Audio {
  ctx: AudioContext | null = null;
  private wind!: GainNode; private hiss!: GainNode; private windFilter!: BiquadFilterNode;
  private noise!: AudioBufferSourceNode; private noiseBuf!: AudioBuffer;
  private master!: GainNode;
  voiceOn = true;
  private speaking = 0;

  start() {
    if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.ctx = ctx;
    this.master = ctx.createGain(); this.master.gain.value = 0.9; this.master.connect(ctx.destination);
    const len = ctx.sampleRate * 2, buf = ctx.createBuffer(1, len, ctx.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    this.noiseBuf = buf;
    this.noise = ctx.createBufferSource(); this.noise.buffer = buf; this.noise.loop = true;
    this.windFilter = ctx.createBiquadFilter(); this.windFilter.type = 'lowpass'; this.windFilter.frequency.value = 400; this.windFilter.Q.value = 0.7;
    this.wind = ctx.createGain(); this.wind.gain.value = 0;
    this.noise.connect(this.windFilter).connect(this.wind).connect(this.master);
    const hf = ctx.createBiquadFilter(); hf.type = 'bandpass'; hf.frequency.value = 2600; hf.Q.value = 0.8;
    this.hiss = ctx.createGain(); this.hiss.gain.value = 0;
    this.noise.connect(hf).connect(this.hiss).connect(this.master);
    this.noise.start();
  }
  update(speed: number, slip: number, air: boolean, timeScale = 1) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const w = Math.min(1, speed / 24);
    this.wind.gain.setTargetAtTime((0.02 + w * w * 0.32 * (air ? 1.3 : 1)) * timeScale, t, 0.08);
    this.windFilter.frequency.setTargetAtTime((300 + w * 1400) * (0.6 + 0.4 * timeScale), t, 0.1);
    this.hiss.gain.setTargetAtTime(air ? 0 : Math.min(0.25, slip * 0.06 + Math.min(1, speed / 20) * 0.03) * timeScale, t, 0.05);
  }
  thwack() { this.blip(180, 0.08, 0.5, 'square'); this.burst(0.06, 0.25, 1800); }
  miss() { this.blip(120, 0.25, 0.25, 'sawtooth'); }
  chime() { [523, 659, 784, 1046].forEach((f, i) => setTimeout(() => this.blip(f, 0.35, 0.25, 'sine'), i * 110)); }
  /** Race start: three low beeps and a high one. */
  startBeep(final: boolean) { this.blip(final ? 1320 : 880, final ? 0.5 : 0.12, 0.35, 'square'); }
  /** A cowbell: inharmonic metallic partials with a fast decay. */
  cowbell(vol = 0.5) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    for (const [f, g] of [[587, 1], [845, 0.8], [1200, 0.3]] as const) {
      const o = this.ctx.createOscillator(), gn = this.ctx.createGain(), bp = this.ctx.createBiquadFilter();
      o.type = 'square'; o.frequency.value = f * (0.98 + Math.random() * 0.04);
      bp.type = 'bandpass'; bp.frequency.value = f; bp.Q.value = 6;
      gn.gain.setValueAtTime(vol * g * 0.25, t); gn.gain.exponentialRampToValueAtTime(0.001, t + 0.45);
      o.connect(bp).connect(gn).connect(this.master); o.start(t); o.stop(t + 0.5);
    }
  }
  /** Crowd: a swell of filtered noise with cowbells on top. */
  crowd(duration = 2.2, vol = 0.5) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource(); src.buffer = this.noiseBuf; src.loop = true;
    const bp = this.ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 900; bp.Q.value = 0.5;
    const g = this.ctx.createGain(); g.gain.setValueAtTime(0.001, t); g.gain.exponentialRampToValueAtTime(vol * 0.35, t + 0.35); g.gain.exponentialRampToValueAtTime(0.001, t + duration);
    src.connect(bp).connect(g).connect(this.master); src.start(t); src.stop(t + duration + 0.1);
    for (let i = 0; i < 4; i++) setTimeout(() => this.cowbell(0.3 + Math.random() * 0.3), 120 + Math.random() * duration * 600);
  }
  fanfare() { [[523, 0], [659, 120], [784, 240], [1046, 360], [1046, 600], [1318, 720]].forEach(([f, d]) => setTimeout(() => this.blip(f, 0.5, 0.3, 'triangle'), d)); this.crowd(3.5, 0.7); }
  wipeout() { this.burst(0.3, 0.6, 500); this.blip(90, 0.4, 0.3, 'sawtooth'); }
  land(hard: number) { this.burst(0.08 + hard * 0.1, 0.2 + hard * 0.3, 900); }
  private burst(dur: number, gain: number, freq: number) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource(); src.buffer = this.noiseBuf;
    const f = this.ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = freq;
    const g = this.ctx.createGain(); g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(f).connect(g).connect(this.master); src.start(t); src.stop(t + dur + 0.05);
  }
  private blip(freq: number, dur: number, gain: number, type: OscillatorType) {
    if (!this.ctx) return;
    const o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.type = type; o.frequency.value = freq; g.gain.value = gain;
    g.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + dur);
    o.connect(g).connect(this.master); o.start(); o.stop(this.ctx.currentTime + dur);
  }

  /** The announcer. Short lines only; drops lines while one is still being spoken unless forced. */
  say(text: string, force = false) {
    if (!this.voiceOn || !('speechSynthesis' in window)) return;
    const now = performance.now();
    if (!force && now < this.speaking) return;
    try {
      if (force) speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.08; u.pitch = 1.0; u.volume = 0.9;
      const voices = speechSynthesis.getVoices();
      const pick = voices.find(v => /en[-_](GB|US)/i.test(v.lang) && /Daniel|Samantha|Google UK English Male|Alex|Aaron/i.test(v.name)) || voices.find(v => /^en/i.test(v.lang));
      if (pick) u.voice = pick;
      this.speaking = now + 400 + text.length * 55;
      speechSynthesis.speak(u);
    } catch {}
  }
}
