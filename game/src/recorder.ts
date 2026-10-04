// `?watch`: record a play session for review. Every frame a compact line of what the tracker read and what
// the game did with it (30 Hz), the full landmarks and the robot's joint targets at 10 Hz, events (phase
// changes, gates, crashes, airs, the finish), and twice a second a side-by-side picture: the camera with the
// skeleton on the left, the game on the right. Posted to the dev server, which writes game/.rec/<session>/.
import type { BodySignals } from './pose.ts';

export interface Snapshot {
  phase: string; raceT: number;
  sig: BodySignals;
  inp: { lean: number; crouch: number; push?: number; boost?: number; jump: boolean; pole?: number };
  state: { speed: number; edge: number; air: boolean; crashed: number; lateral: number; progress: number };
  gates: { made: number; next: number; missed: number };
  pose: Record<string, number>;
  vit?: { hr: number; br: number; arousal: number; mode: string };
}

const r3 = (x: number) => Math.round(x * 1000) / 1000;

export class Recorder {
  session = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  private lines: string[] = [];
  private lastFlush = 0; private lastFrame = 0; private lastLm = 0; private lastRow = 0; private n = 0;
  private prev: { phase: string; made: number; missed: number; crashed: boolean; air: boolean; airT: number } | null = null;
  private comp = Object.assign(document.createElement('canvas'), { width: 640, height: 240 });
  private busy = false;
  constructor(private cam: HTMLCanvasElement, private gl: HTMLCanvasElement) {
    this.event('session', { ua: navigator.userAgent, w: innerWidth, h: innerHeight });
    addEventListener('pagehide', () => this.flush(true));
  }

  event(ev: string, data: Record<string, unknown> = {}) { this.lines.push(JSON.stringify({ ev, t: r3(performance.now() / 1000), ...data })); }

  /** Call once per frame, right after the frame is rendered (the WebGL canvas is only readable then). */
  tick(now: number, s: Snapshot) {
    const p = this.prev;
    if (!p || p.phase !== s.phase) this.event('phase', { phase: s.phase });
    if (p && s.gates.made > p.made) this.event('gate', { k: s.gates.next, raceT: r3(s.raceT) });
    if (p && s.gates.missed > p.missed) this.event('miss', { k: s.gates.next, raceT: r3(s.raceT), lean: r3(s.inp.lean), lateral: r3(s.state.lateral) });
    if (p && s.state.crashed > 0 && !p.crashed) this.event('crash', { raceT: r3(s.raceT), speed: r3(s.state.speed), edge: r3(s.state.edge), lean: r3(s.inp.lean) });
    if (p && s.state.air && !p.air) this.event('air', { raceT: r3(s.raceT), speed: r3(s.state.speed) });
    if (p && !s.state.air && p.air) this.event('land', { raceT: r3(s.raceT), air: r3(s.raceT - p.airT) });
    if (s.sig.jump) this.event('hop', { tier: s.sig.tier, crouch: r3(s.sig.crouch) });
    if (s.sig.pole) this.event('pole', { n: s.sig.pole, tier: s.sig.tier });
    if (s.phase === 'finished' && p?.phase !== 'finished') this.event('finish', { raceT: r3(s.raceT), made: s.gates.made, missed: s.gates.missed });
    this.prev = { phase: s.phase, made: s.gates.made, missed: s.gates.missed, crashed: s.state.crashed > 0, air: s.state.air, airT: s.state.air && !(p?.air) ? s.raceT : p?.airT ?? 0 };

    if (now - this.lastRow >= 33) {
      this.lastRow = now;
      const g = s.sig;
      this.lines.push(JSON.stringify({
        t: r3(now / 1000), ph: s.phase, rt: r3(s.raceT),
        tier: g.tier, pr: g.present ? 1 : 0, cal: g.calibrated ? 1 : 0, fps: g.fps,
        lean: r3(g.lean), raw: r3(g.rawLean), cr: r3(g.crouch), kL: r3(g.kneeL), kR: r3(g.kneeR), tp: r3(g.torsoPitch),
        up: g.handsUp ? 1 : 0, aL: g.armL ? 1 : 0, aR: g.armR ? 1 : 0, av: g.armsVisible ? 1 : 0,
        iL: r3(s.inp.lean), iC: r3(s.inp.crouch), iB: r3(s.inp.boost ?? 0),
        v: r3(s.state.speed), e: r3(s.state.edge), air: s.state.air ? 1 : 0, cr2: r3(s.state.crashed), lat: r3(s.state.lateral), pg: r3(s.state.progress),
        gm: s.gates.made, gn: s.gates.next,
        ...(s.vit ? { hr: r3(s.vit.hr), br: r3(s.vit.br), ar: r3(s.vit.arousal), vm: s.vit.mode } : {}),
      }));
    }
    if (now - this.lastLm >= 100 && s.sig.image && s.sig.world) {
      this.lastLm = now;
      this.lines.push(JSON.stringify({
        t: r3(now / 1000), lm: s.sig.image.map(l => [r3(l.x), r3(l.y), r3(l.z), r3(l.visibility ?? 1)]),
        w: s.sig.world.map(l => [r3(l.x), r3(l.y), r3(l.z)]),
        q: Object.fromEntries(Object.entries(s.pose).map(([k, v]) => [k.replace(/_joint$/, ''), r3(v)])),
        dbg: s.sig.debug,
      }));
    }
    if (now - this.lastFrame >= 500 && !this.busy) { this.lastFrame = now; this.frame(now, s); }
    if (now - this.lastFlush >= 1000) this.flush();
  }

  private frame(now: number, s: Snapshot) {
    const g = this.comp.getContext('2d')!;
    g.fillStyle = '#000'; g.fillRect(0, 0, 640, 240);
    // the camera as the player sees it (mirrored), skeleton included
    g.save(); g.translate(320, 0); g.scale(-1, 1); g.drawImage(this.cam, 0, 0, 320, 240); g.restore();
    // the game, centre-cropped to 4:3
    const W = this.gl.width, H = this.gl.height, cw = Math.min(W, H * 4 / 3), ch = cw * 3 / 4;
    g.drawImage(this.gl, (W - cw) / 2, (H - ch) / 2, cw, ch, 320, 0, 320, 240);
    g.fillStyle = 'rgba(0,0,0,.6)'; g.fillRect(0, 216, 640, 24);
    g.fillStyle = '#fff'; g.font = '12px ui-monospace,monospace';
    const q = s.sig;
    g.fillText(`${(now / 1000).toFixed(1)}s ${s.phase} ${s.raceT.toFixed(1)} | ${q.tier}${q.calibrated ? '' : '(uncal)'} lean ${q.lean.toFixed(2)} raw ${q.rawLean.toFixed(2)} crouch ${q.crouch.toFixed(2)} knee ${q.kneeL.toFixed(1)}/${q.kneeR.toFixed(1)} | v ${(s.state.speed * 3.6).toFixed(0)} g ${s.gates.made}/${s.gates.next}`, 6, 232);
    const n = this.n++;
    this.busy = true;
    this.comp.toBlob(b => {
      this.busy = false;
      if (b) fetch(`/__rec/frame?s=${this.session}&n=${n}`, { method: 'POST', body: b }).catch(() => {});
    }, 'image/jpeg', 0.72);
  }

  flush(final = false) {
    this.lastFlush = performance.now();
    if (!this.lines.length) return;
    const body = this.lines.join('\n') + '\n'; this.lines = [];
    const url = `/__rec/log?s=${this.session}`;
    if (final && navigator.sendBeacon) navigator.sendBeacon(url, body);
    else fetch(url, { method: 'POST', body, keepalive: body.length < 60000 }).catch(() => {});
  }
}
