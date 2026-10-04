// SpacetimeDB: one shared mountain. Connect, keep the identity across reloads, stream our skier at
// ~8 Hz, and surface everyone else's rows to the game. Works offline: every call is a no-op until
// the connection is up, and the game never waits on it.
import { DbConnection, tables } from './module_bindings/index.ts';
import type { Skier, Run, Player, DatasetStats, Feed, Challenge, PoseSample, Cheer, TraceChunk, VitalChunk, VitalSample } from './module_bindings/types.ts';
import type { Identity } from 'spacetimedb';

export type { Skier, Run, Player, DatasetStats, Feed, Challenge, PoseSample, Cheer, TraceChunk, VitalChunk, VitalSample };

/** One stretch of the course in the hazard map: how far above their resting rate people's hearts ran there. */
export interface HazardBin { s0: number; s1: number; rise: number; n: number; runs: number }

/** Pool everyone's vitals into course bins (by metres along the course): the mean pulse rise over resting. */
export function hazardMap(chunks: VitalChunk[], binM = 25): HazardBin[] {
  const bins = new Map<number, { sum: number; n: number; runs: Set<string> }>();
  for (const c of chunks) for (const v of c.samples) {
    if (!(v.rise > -0.5 && v.rise < 1.5)) continue;
    const k = Math.floor(v.s / binM);
    const b = bins.get(k) ?? bins.set(k, { sum: 0, n: 0, runs: new Set() }).get(k)!;
    b.sum += v.rise; b.n++; b.runs.add(c.runKey);
  }
  return [...bins.entries()].sort((a, b) => a[0] - b[0]).map(([k, b]) => ({ s0: k * binM, s1: (k + 1) * binM, rise: b.sum / b.n, n: b.n, runs: b.runs.size }));
}

export const STDB_URI = (import.meta.env.VITE_STDB_URI as string | undefined) || 'ws://127.0.0.1:3000';
export const STDB_DB = (import.meta.env.VITE_STDB_DB as string | undefined) || 'ground-truth';
const TOKEN_KEY = `stdb-token:${STDB_URI}/${STDB_DB}`;

export interface NetEvents {
  status?: (s: string) => void;
  skier?: (kind: 'insert' | 'update' | 'delete', row: Skier) => void;
  cheer?: (c: Cheer) => void;
  changed?: () => void; // any table changed: refresh the HUD lists
}

export class Net {
  conn: DbConnection | null = null;
  identity: Identity | null = null;
  ready = false;
  status = 'offline';
  private lastState = 0;
  private ev: NetEvents;
  private pending: (() => void)[] = [];
  private joinArgs: { name: string; joinCode: string; course: string } | null = null;
  private traceSubs = new Map<string, { handle: { unsubscribe(): void } | null; cb: (chunks: TraceChunk[]) => void }>();

  constructor(ev: NetEvents = {}) { this.ev = ev; }

  connect() {
    this.set('connecting');
    let token: string | undefined;
    try { token = localStorage.getItem(TOKEN_KEY) || undefined; } catch {}
    try {
      this.conn = DbConnection.builder()
        .withUri(STDB_URI)
        .withDatabaseName(STDB_DB)
        .withToken(token)
        .withConfirmedReads(false)
        .onConnect((conn, identity, tok) => {
          this.identity = identity;
          try { localStorage.setItem(TOKEN_KEY, tok); } catch {}
          this.set('syncing');
          conn.subscriptionBuilder()
            .onApplied(() => {
              this.ready = true; this.set('live');
              if (this.joinArgs) this.call(() => this.conn!.reducers.join(this.joinArgs!));
              const q = this.pending; this.pending = []; q.forEach(f => f());
              this.ev.changed?.();
            })
            .onError((_c: unknown, e?: unknown) => { console.error('subscription', e); this.set('subscription error'); })
            .subscribe([tables.player, tables.skier, tables.run, tables.datasetStats, tables.feed, tables.challenge, tables.cheer]);
          // The hazard map's vitals on their own subscription: if a database has not been republished with
          // vital_chunk yet, only the hazard map goes quiet, not the leaderboard and the live skiers.
          conn.subscriptionBuilder()
            .onApplied(() => this.ev.changed?.())
            .onError((_c: unknown, e?: unknown) => console.warn('vitals subscription (republish the module for the hazard map)', e))
            .subscribe([tables.vitalChunk]);
          conn.db.skier.onInsert((_c, r) => this.ev.skier?.('insert', r));
          conn.db.skier.onUpdate((_c, _o, r) => this.ev.skier?.('update', r));
          conn.db.skier.onDelete((_c, r) => this.ev.skier?.('delete', r));
          conn.db.cheer.onInsert((_c, r) => this.ev.cheer?.(r));
          const ping = () => this.ev.changed?.();
          conn.db.run.onInsert(ping); conn.db.run.onUpdate(ping);
          conn.db.player.onInsert(ping); conn.db.player.onUpdate(ping); conn.db.player.onDelete(ping);
          conn.db.datasetStats.onInsert(ping); conn.db.datasetStats.onUpdate(ping);
          conn.db.feed.onInsert(ping); conn.db.feed.onDelete(ping);
          conn.db.vitalChunk.onInsert(ping);
          conn.db.challenge.onInsert(ping); conn.db.challenge.onUpdate(ping);
          conn.db.traceChunk.onInsert((_c, r) => this.traceChanged(r.runKey));
        })
        .onConnectError((_c, e) => { console.warn('spacetimedb connect error', e); this.set('offline (no SpacetimeDB)'); })
        .onDisconnect(() => { this.ready = false; this.set('disconnected'); })
        .build();
    } catch (e) {
      console.warn('spacetimedb', e); this.set('offline');
    }
  }

  private set(s: string) { this.status = s; this.ev.status?.(s); }

  get me(): Identity | null { return this.identity; }
  isMe(id: Identity) { return !!this.identity && id.isEqual(this.identity); }

  // ---- reads (from the local cache) ----
  skiers(): Skier[] { return this.conn ? [...this.conn.db.skier.iter()] : []; }
  players(): Player[] { return this.conn ? [...this.conn.db.player.iter()] : []; }
  myPlayer(): Player | null { return this.conn && this.identity ? (this.conn.db.player.identity.find(this.identity) ?? null) : null; }
  runs(course?: string): Run[] {
    if (!this.conn) return [];
    const rows = [...this.conn.db.run.iter()].filter(r => r.finished && (!course || r.course === course));
    return rows.sort((a, b) => a.timeMs - b.timeMs);
  }
  /** The fastest finished run on this course, with gates to compare against. */
  bestRun(course: string, gatesTotal: number): Run | null {
    return this.runs(course).find(r => r.gatesTotal === gatesTotal && r.splits.length > 0) ?? null;
  }
  /** Everyone's vitals on this course (the hazard map is built from these). */
  vitals(course?: string): VitalChunk[] { return this.conn ? [...this.conn.db.vitalChunk.iter()].filter(c => !course || c.course === course) : []; }
  stats(): DatasetStats | null { return this.conn ? (this.conn.db.datasetStats.id.find(0) ?? null) : null; }
  feed(): Feed[] { return this.conn ? [...this.conn.db.feed.iter()].sort((a, b) => Number(b.id - a.id)) : []; }
  challengesFor(name: string): Challenge[] {
    return this.conn ? [...this.conn.db.challenge.iter()].filter(c => c.status === 'open' && c.toName.toLowerCase() === name.toLowerCase()) : [];
  }

  /** Follow one run's trace (for the ghost). Calls back with all chunks seen so far whenever they change. */
  watchTrace(runKey: string, cb: (chunks: TraceChunk[]) => void) {
    if (!this.conn || !this.ready) return;
    const prev = this.traceSubs.get(runKey);
    if (prev) { prev.cb = cb; this.traceChanged(runKey); return; }
    const entry = { handle: null as { unsubscribe(): void } | null, cb };
    this.traceSubs.set(runKey, entry);
    try {
      entry.handle = this.conn.subscriptionBuilder()
        .onApplied(() => this.traceChanged(runKey))
        .subscribe(tables.traceChunk.where(r => r.runKey.eq(runKey)));
    } catch (e) { console.warn('trace subscription', e); }
  }
  unwatchTrace(runKey: string) {
    const e = this.traceSubs.get(runKey);
    if (e) { try { e.handle?.unsubscribe(); } catch {} this.traceSubs.delete(runKey); }
  }
  private traceChanged(runKey: string) {
    const e = this.traceSubs.get(runKey);
    if (!e || !this.conn) return;
    e.cb([...this.conn.db.traceChunk.runKey.filter(runKey)].sort((a, b) => a.seq - b.seq));
  }

  // ---- writes ----
  private call(fn: () => Promise<unknown> | void, queue = false) {
    if (!this.conn || !this.ready) { if (queue && this.pending.length < 64) this.pending.push(() => this.call(fn)); return; }
    try { const p = fn(); if (p && typeof (p as Promise<unknown>).catch === 'function') (p as Promise<unknown>).catch(e => console.warn('reducer', e)); }
    catch (e) { console.warn('reducer', e); }
  }
  join(name: string, joinCode: string, course: string) { this.joinArgs = { name, joinCode, course }; this.call(() => this.conn!.reducers.join({ name, joinCode, course })); }
  /** Throttled to `hz`. */
  setState(s: { x: number; y: number; z: number; heading: number; speed: number; lean: number; crouch: number; air: boolean; phase: number; progress: number }, hz = 8) {
    const now = performance.now();
    if (now - this.lastState < 1000 / hz) return;
    this.lastState = now;
    this.call(() => this.conn!.reducers.setState(s));
  }
  startRun(key: string, course: string, gatesTotal: number) { this.call(() => this.conn!.reducers.startRun({ key, course, gatesTotal }), true); }
  pushTrace(runKey: string, seq: number, samples: PoseSample[]) { this.call(() => this.conn!.reducers.pushTrace({ runKey, seq, samples }), true); }
  finishRun(key: string, timeMs: number, gatesHit: number, maxSpeed: number, distanceM: number, splits: number[]) { this.call(() => this.conn!.reducers.finishRun({ key, timeMs, gatesHit, maxSpeed, distanceM, splits }), true); }
  pushVitals(runKey: string, restHr: number, samples: VitalSample[]) { this.call(() => this.conn!.reducers.pushVitals({ runKey, restHr, samples }), true); }
  pushPhoto(runKey: string, jpeg: string) { this.call(() => this.conn!.reducers.pushPhoto({ runKey, jpeg }), true); }
  sendCheer(fromName: string, toName: string, kind: string) { this.call(() => this.conn!.reducers.sendCheer({ fromName, toName, kind })); }
}

export const fmtTime = (ms: number) => {
  const m = Math.floor(ms / 60000), s = ((ms % 60000) / 1000).toFixed(2).padStart(5, '0');
  return `${m}:${s}`;
};
export const fmtDelta = (ms: number) => `${ms < 0 ? '−' : '+'}${(Math.abs(ms) / 1000).toFixed(2)}`;
export const fmtInt = (n: number | bigint) => Number(n).toLocaleString('en-US');
export const colorOf = (c: number) => '#' + (c & 0xffffff).toString(16).padStart(6, '0');
