// The player's vitals while they ski, from Presage (contactless heart rate and breathing from the same camera
// that reads their pose). Readings arrive every few seconds; this module keeps the latest values, learns a
// resting baseline while the player stands at the start, and turns the difference into a 0..1 "arousal" the
// game reacts to: a heartbeat on the robot's chest and on the screen, help when you are flustered and a
// reward for staying cool.

export interface VitalsReading { hr?: number | null; br?: number | null; hrConf?: number | null; brConf?: number | null }

export class Vitals {
  hr = 0; br = 0;            // latest, smoothed (beats / breaths per minute), 0 = not measured yet
  baseHr = 0; baseBr = 0;    // resting baseline, learned at the start
  status = 'off';            // off, connecting, measuring, live, or an error to show
  hint = '';                 // what the player should do for a reading (face the camera, hold still...)
  lastAt = 0;                // performance.now() of the last reading
  history: { t: number; hr: number; br: number }[] = [];
  private beatPhase = 0;

  /** Feed a reading from any source. */
  push(r: VitalsReading, calm: boolean) {
    const now = performance.now();
    const okHr = !!r.hr && r.hr > 35 && r.hr < 220 && (r.hrConf ?? 1) > 0.2;
    const okBr = !!r.br && r.br > 4 && r.br < 60 && (r.brConf ?? 1) > 0.2;
    if (okHr) this.hr = this.hr ? this.hr + (r.hr! - this.hr) * 0.6 : r.hr!;
    if (okBr) this.br = this.br ? this.br + (r.br! - this.br) * 0.6 : r.br!;
    if (!okHr && !okBr) return;            // nothing new and trustworthy: the last good value stands
    this.lastAt = now;
    this.history.push({ t: now, hr: this.hr, br: this.br });
    // resting baseline: learned while standing at the start; it can only drift down a little during a run
    if (this.hr) this.baseHr = !this.baseHr ? this.hr : calm ? this.baseHr + (this.hr - this.baseHr) * 0.3 : Math.min(this.baseHr, this.baseHr + (this.hr - this.baseHr) * 0.05);
    if (this.br) this.baseBr = !this.baseBr ? this.br : calm ? this.baseBr + (this.br - this.baseBr) * 0.3 : this.baseBr;
    this.status = 'live';
  }

  /** Live while readings keep coming; skiing moves too much for a fresh one, so the last good value holds for a while. */
  get live() { return this.status === 'live' && performance.now() - this.lastAt < 90000; }

  /** 0 at your resting rate, 1 at +35 % (a racing heart). */
  get arousal() { return this.live && this.baseHr ? Math.max(0, Math.min(1, (this.hr / this.baseHr - 1) / 0.35)) : 0; }

  /** Advance the heartbeat clock; returns 0..1, a sharp thump on each beat (lub) with a smaller second (dub). */
  beat(dt: number) {
    if (!this.live || !this.hr) return 0;
    this.beatPhase = (this.beatPhase + dt * this.hr / 60) % 1;
    const p = this.beatPhase;
    return Math.exp(-p * 18) + 0.55 * Math.exp(-Math.abs(p - 0.28) * 22);
  }

  /** For the result card: averages and peak over a time window. */
  summary(fromT: number) {
    const h = this.history.filter(x => x.t >= fromT);
    if (!h.length) return null;
    const hrs = h.map(x => x.hr).filter(Boolean), brs = h.map(x => x.br).filter(Boolean);
    const avg = (a: number[]) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
    return { avgHr: avg(hrs), peakHr: hrs.length ? Math.max(...hrs) : 0, avgBr: avg(brs), baseHr: this.baseHr };
  }
}
