// The race commentator, voiced by ElevenLabs through /api/tts (the dev server locally, a Vercel function live;
// the key never reaches the page). Lines have a priority: a big
// moment (the finish, a crash) cuts in, routine calls wait their turn, and chatter that has gone stale by the
// time the voice is free is dropped, so the commentary always matches what is on screen. Audio is fetched as
// soon as a line is asked for, so it is usually ready by the time its turn comes. Without a key it falls back
// to the browser's speech synthesis.

export const P = { chatter: 0, call: 1, big: 2, must: 3 } as const;
type Line = { text: string; pri: number; at: number; audio: Promise<AudioBuffer | null> };

export class Commentator {
  enabled = false;
  voice = '';
  private queue: Line[] = [];
  private playing: { src: AudioBufferSourceNode; pri: number } | null = null;
  private lastSpoke = 0;
  private gen = 0;              // bumped on every interruption, so a line still loading knows it was cut

  constructor(private ctx: () => AudioContext | null, private out: () => AudioNode | null, private fallback: (text: string, force: boolean) => void) {
    fetch('/api/tts').then(r => r.ok ? r.json() : null).then(j => { this.enabled = !!j?.enabled; this.voice = j?.voice ?? ''; }).catch(() => {});
  }

  /** Seconds since the commentator last finished or started a line (for idle chatter). */
  get quietFor() { return this.playing ? 0 : (performance.now() - this.lastSpoke) / 1000; }

  say(text: string, pri: number = P.call) {
    const ctx = this.ctx();
    if (!this.enabled || !ctx) { this.lastSpoke = performance.now(); this.fallback(text, pri >= P.big); return; }
    // routine lines are not worth queueing behind each other forever
    if (pri <= P.chatter && (this.playing || this.queue.length)) return;
    const line: Line = { text, pri, at: performance.now(), audio: this.fetch(text, ctx) };
    if (this.playing && pri >= P.big && pri > this.playing.pri) { this.queue = this.queue.filter(l => l.pri >= pri); this.queue.unshift(line); this.stop(); }
    else { this.queue.push(line); this.queue.sort((a, b) => b.pri - a.pri); this.queue = this.queue.slice(0, 3); }
    if (!this.playing) this.next();
  }

  private fetch(text: string, ctx: AudioContext) {
    return fetch(`/api/tts?t=${encodeURIComponent(text)}`)
      .then(r => r.ok ? r.arrayBuffer() : null)
      .then(b => b ? ctx.decodeAudioData(b) : null)
      .catch(() => null);
  }

  private stop() { this.gen++; try { this.playing?.src?.stop(); } catch {} this.playing = null; }

  private async next() {
    const line = this.queue.shift();
    if (!line) return;
    // a call that is more than a few seconds old no longer matches the race
    const maxAge = line.pri >= P.big ? 8000 : line.pri >= P.call ? 4000 : 2500;
    if (performance.now() - line.at > maxAge) { this.next(); return; }
    this.playing = { src: null as unknown as AudioBufferSourceNode, pri: line.pri };
    const gen = this.gen;
    const buf = await line.audio;
    if (gen !== this.gen) return;   // interrupted while loading: the interrupting line has its own turn
    const ctx = this.ctx(), out = this.out();
    if (!buf || !ctx || !out) { this.playing = null; if (!buf) this.fallback(line.text, false); this.next(); return; }
    if (performance.now() - line.at > maxAge + 1500) { this.playing = null; this.next(); return; }
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(out);
    this.playing = { src, pri: line.pri };
    this.lastSpoke = performance.now();
    src.onended = () => { if (this.playing?.src === src) { this.playing = null; this.lastSpoke = performance.now(); this.next(); } };
    src.start();
  }
}

export const pick = <T>(a: readonly T[]) => a[Math.floor(Math.random() * a.length)];
