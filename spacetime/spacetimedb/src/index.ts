// Ground Truth: one shared mountain. Every skier who opens the game is on the same real piste at the
// same time, every finished run grows the motion + terrain dataset, and the iMessage agent reads the
// outbox to text people their results.
import { schema, table, t, SenderError, type InferSchema, type ReducerCtx } from 'spacetimedb/server';
import { ScheduleAt } from 'spacetimedb';

// One motion sample registered to the ground under it: where the body was, how it was leaning and
// crouching, and the slope of the real terrain at that point. This is the record Ground Truth collects.
const PoseSample = t.object('PoseSample', {
  tMs: t.u32(),
  x: t.f32(),
  y: t.f32(),
  z: t.f32(),
  speed: t.f32(),
  heading: t.f32(),
  lean: t.f32(),
  crouch: t.f32(),
  kneeL: t.f32(),
  kneeR: t.f32(),
  slopeDeg: t.f32(),
});

const player = table(
  { name: 'player', public: true },
  {
    identity: t.identity().primaryKey(),
    name: t.string(),
    color: t.u32(),
    joinCode: t.string().index('btree'),
    online: t.bool(),
    runs: t.u32(),
    bestTimeMs: t.u32(),
    cheers: t.u32(),
    lastSeen: t.timestamp(),
  }
);

// Live state of every skier on the mountain, streamed by the client at about 10 Hz.
const skier = table(
  { name: 'skier', public: true },
  {
    identity: t.identity().primaryKey(),
    name: t.string(),
    color: t.u32(),
    course: t.string(),
    x: t.f32(),
    y: t.f32(),
    z: t.f32(),
    heading: t.f32(),
    speed: t.f32(),
    lean: t.f32(),
    crouch: t.f32(),
    air: t.bool(),
    phase: t.u8(), // 0 lobby, 1 calibrating, 2 racing, 3 finished
    progress: t.f32(), // 0..1 down the course
    updatedAt: t.timestamp(),
  }
);

const run = table(
  { name: 'run', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(),
    identity: t.identity().index('btree'),
    name: t.string(),
    course: t.string(),
    finished: t.bool().index('btree'),
    timeMs: t.u32(),
    gatesHit: t.u16(),
    gatesTotal: t.u16(),
    maxSpeed: t.f32(),
    distanceM: t.f32(),
    samples: t.u32(),
    splits: t.array(t.u32()), // elapsed ms at each gate, 0 for a missed gate; the ghost and the TV splits read this
    startedAt: t.timestamp(),
    finishedAt: t.timestamp(),
  }
);

// The dataset itself, in chunks so a run is a handful of rows instead of thousands.
const traceChunk = table(
  { name: 'trace_chunk', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    runKey: t.string().index('btree'),
    seq: t.u32(),
    samples: t.array(PoseSample),
  }
);

const datasetStats = table(
  { name: 'dataset_stats', public: true },
  {
    id: t.u8().primaryKey(),
    samples: t.u64(),
    runs: t.u64(),
    meters: t.f64(),
    skiers: t.u32(),
  }
);

// Ticker for the mountain board: who joined, who cleared the course, who beat whom.
const feed = table(
  { name: 'feed', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    kind: t.string(),
    text: t.string(),
    at: t.timestamp(),
  }
);

// Challenges come in over iMessage ("challenge Ryan") and resolve when the target finishes a run.
const challenge = table(
  { name: 'challenge', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    fromName: t.string(),
    toName: t.string().index('btree'),
    course: t.string(),
    targetTimeMs: t.u32(),
    status: t.string(), // open, beaten, failed
    createdAt: t.timestamp(),
  }
);

// Messages for the iMessage agent to deliver. Keyed by join code, never by phone number.
const outbox = table(
  { name: 'outbox', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    joinCode: t.string(),
    kind: t.string(),
    body: t.string(),
    ref: t.string(), // run key for results, so the agent can attach the finish photo
    sent: t.bool().index('btree'),
    createdAt: t.timestamp(),
  }
);

// Spectators on the board tap to cheer; every screen gets it once and nobody stores it.
const cheer = table(
  { name: 'cheer', public: true, event: true },
  {
    fromName: t.string(),
    toName: t.string(),
    kind: t.string(), // bell, clap, fire
    at: t.timestamp(),
  }
);

// The finish-line photo of a run (JPEG, base64), texted by the agent with the result.
const runPhoto = table(
  { name: 'run_photo', public: true },
  {
    runKey: t.string().primaryKey(),
    jpeg: t.string(),
    at: t.timestamp(),
  }
);

const tickTimer = table(
  { name: 'tick_timer' },
  {
    scheduledId: t.u64().primaryKey().autoInc(),
    scheduledAt: t.scheduleAt(),
  }
);

const spacetimedb = schema({ player, skier, run, traceChunk, datasetStats, feed, challenge, outbox, cheer, runPhoto, tickTimer });
export default spacetimedb;

type Ctx = ReducerCtx<InferSchema<typeof spacetimedb>>;

const STALE_MICROS = 10_000_000n;
const FEED_MAX = 60;

function stats(ctx: Ctx) {
  return ctx.db.datasetStats.id.find(0) ?? ctx.db.datasetStats.insert({ id: 0, samples: 0n, runs: 0n, meters: 0, skiers: 0 });
}

function post(ctx: Ctx, kind: string, text: string) {
  ctx.db.feed.insert({ id: 0n, kind, text, at: ctx.timestamp });
  const rows = [...ctx.db.feed.iter()].sort((a, b) => Number(a.id - b.id));
  for (let i = 0; i < rows.length - FEED_MAX; i++) ctx.db.feed.id.delete(rows[i].id);
}

function fmtTime(ms: number) {
  const m = Math.floor(ms / 60000);
  const s = ((ms % 60000) / 1000).toFixed(2).padStart(5, '0');
  return `${m}:${s}`;
}

function text(ctx: Ctx, joinCode: string, kind: string, body: string, ref = '') {
  if (!joinCode) return;
  ctx.db.outbox.insert({ id: 0n, joinCode, kind, body, ref, sent: false, createdAt: ctx.timestamp });
}

export const init = spacetimedb.init(ctx => {
  stats(ctx);
  ctx.db.tickTimer.insert({ scheduledId: 0n, scheduledAt: ScheduleAt.interval(2_000_000n) });
});

export const onConnect = spacetimedb.clientConnected(_ctx => {});

export const onDisconnect = spacetimedb.clientDisconnected(ctx => {
  const p = ctx.db.player.identity.find(ctx.sender);
  if (p) ctx.db.player.identity.update({ ...p, online: false, lastSeen: ctx.timestamp });
  ctx.db.skier.identity.delete(ctx.sender);
});

// Called by the game when someone enters their name. A join code ties the browser to a phone that
// texted the iMessage agent; the agent is the only thing that knows which phone that is.
export const join = spacetimedb.reducer(
  { name: t.string(), joinCode: t.string(), course: t.string() },
  (ctx, { name, joinCode, course }) => {
    const clean = name.trim().slice(0, 24) || 'Skier';
    const code = joinCode.trim().toUpperCase().slice(0, 8);
    const existing = ctx.db.player.identity.find(ctx.sender);
    const color = existing?.color ?? ctx.random.integerInRange(0, 0xffffff);
    if (existing) {
      ctx.db.player.identity.update({ ...existing, name: clean, joinCode: code || existing.joinCode, online: true, lastSeen: ctx.timestamp });
    } else {
      ctx.db.player.insert({ identity: ctx.sender, name: clean, color, joinCode: code, online: true, runs: 0, bestTimeMs: 0, cheers: 0, lastSeen: ctx.timestamp });
      const s = stats(ctx);
      ctx.db.datasetStats.id.update({ ...s, skiers: s.skiers + 1 });
      post(ctx, 'join', `${clean} is on the mountain`);
    }
    const row = { identity: ctx.sender, name: clean, color, course, x: 0, y: 0, z: 0, heading: 0, speed: 0, lean: 0, crouch: 0, air: false, phase: 0, progress: 0, updatedAt: ctx.timestamp };
    if (ctx.db.skier.identity.find(ctx.sender)) ctx.db.skier.identity.update(row);
    else ctx.db.skier.insert(row);
    if (code) text(ctx, code, 'joined', `${clean}, you're on the ${course}. Lean to turn, crouch to go faster. I'll text your time when you finish.`);
  }
);

// Live position, about 10 times a second while skiing.
export const setState = spacetimedb.reducer(
  { x: t.f32(), y: t.f32(), z: t.f32(), heading: t.f32(), speed: t.f32(), lean: t.f32(), crouch: t.f32(), air: t.bool(), phase: t.u8(), progress: t.f32() },
  (ctx, s) => {
    const row = ctx.db.skier.identity.find(ctx.sender);
    if (!row) return;
    ctx.db.skier.identity.update({ ...row, ...s, updatedAt: ctx.timestamp });
  }
);

export const startRun = spacetimedb.reducer(
  { key: t.string(), course: t.string(), gatesTotal: t.u16() },
  (ctx, { key, course, gatesTotal }) => {
    const p = ctx.db.player.identity.find(ctx.sender);
    if (!p) throw new SenderError('join first');
    if (ctx.db.run.key.find(key)) return;
    ctx.db.run.insert({ id: 0n, key, identity: ctx.sender, name: p.name, course, finished: false, timeMs: 0, gatesHit: 0, gatesTotal, maxSpeed: 0, distanceM: 0, samples: 0, splits: [], startedAt: ctx.timestamp, finishedAt: ctx.timestamp });
    post(ctx, 'start', `${p.name} is in the start gate`);
  }
);

// A chunk of the dataset: body motion registered to the ground under it.
export const pushTrace = spacetimedb.reducer(
  { runKey: t.string(), seq: t.u32(), samples: t.array(PoseSample) },
  (ctx, { runKey, seq, samples }) => {
    const r = ctx.db.run.key.find(runKey);
    if (!r || !r.identity.equals(ctx.sender)) throw new SenderError('not your run');
    if (samples.length === 0 || samples.length > 400) return;
    ctx.db.traceChunk.insert({ id: 0n, runKey, seq, samples });
    ctx.db.run.id.update({ ...r, samples: r.samples + samples.length });
    const s = stats(ctx);
    ctx.db.datasetStats.id.update({ ...s, samples: s.samples + BigInt(samples.length) });
  }
);

export const finishRun = spacetimedb.reducer(
  { key: t.string(), timeMs: t.u32(), gatesHit: t.u16(), maxSpeed: t.f32(), distanceM: t.f32(), splits: t.array(t.u32()) },
  (ctx, { key, timeMs, gatesHit, maxSpeed, distanceM, splits }) => {
    const r = ctx.db.run.key.find(key);
    if (!r || !r.identity.equals(ctx.sender)) throw new SenderError('not your run');
    if (r.finished) return;
    ctx.db.run.id.update({ ...r, finished: true, timeMs, gatesHit, maxSpeed, distanceM, splits, finishedAt: ctx.timestamp });
    const p = ctx.db.player.identity.find(ctx.sender);
    if (!p) return;
    const best = p.bestTimeMs === 0 || timeMs < p.bestTimeMs ? timeMs : p.bestTimeMs;
    ctx.db.player.identity.update({ ...p, runs: p.runs + 1, bestTimeMs: best, lastSeen: ctx.timestamp });
    const s = stats(ctx);
    ctx.db.datasetStats.id.update({ ...s, runs: s.runs + 1n, meters: s.meters + distanceM });

    // Rank among finished runs on this course.
    const finished = [...ctx.db.run.finished.filter(true)].filter(x => x.course === r.course).sort((a, b) => a.timeMs - b.timeMs);
    const rank = finished.findIndex(x => x.id === r.id) + 1;
    post(ctx, 'finish', `${p.name} finished the ${r.course} in ${fmtTime(timeMs)} (#${rank}), ${gatesHit}/${r.gatesTotal} gates, ${Math.round(maxSpeed * 3.6)} km/h`);
    text(ctx, p.joinCode, 'result',
      `${p.name}: ${r.course} in ${fmtTime(timeMs)}, #${rank} of ${finished.length}. ${gatesHit}/${r.gatesTotal} gates, top speed ${Math.round(maxSpeed * 3.6)} km/h. ` +
      `Your run added ${r.samples} motion samples on real terrain to Ground Truth (now ${s.samples} samples from ${s.skiers} skiers). ` +
      `Reply CHALLENGE <name> to send someone your time.`, key);

    // Resolve challenges aimed at this player.
    for (const c of ctx.db.challenge.toName.filter(p.name)) {
      if (c.status !== 'open' || c.course !== r.course) continue;
      const beaten = timeMs < c.targetTimeMs;
      ctx.db.challenge.id.update({ ...c, status: beaten ? 'beaten' : 'failed' });
      post(ctx, 'challenge', beaten ? `${p.name} beat ${c.fromName}'s challenge by ${fmtTime(c.targetTimeMs - timeMs)}` : `${p.name} missed ${c.fromName}'s time by ${fmtTime(timeMs - c.targetTimeMs)}`);
      for (const from of ctx.db.player.iter()) {
        if (from.name === c.fromName) text(ctx, from.joinCode, 'challenge', beaten ? `${p.name} just beat your ${fmtTime(c.targetTimeMs)} on the ${r.course} with ${fmtTime(timeMs)}. Your move.` : `${p.name} tried your ${fmtTime(c.targetTimeMs)} on the ${r.course} and got ${fmtTime(timeMs)}. Still yours.`);
      }
    }
  }
);

// Sent by the iMessage agent on behalf of a phone that texted "challenge <name>".
export const createChallenge = spacetimedb.reducer(
  { fromName: t.string(), toName: t.string(), course: t.string(), targetTimeMs: t.u32() },
  (ctx, { fromName, toName, course, targetTimeMs }) => {
    const to = toName.trim();
    ctx.db.challenge.insert({ id: 0n, fromName, toName: to, course, targetTimeMs, status: 'open', createdAt: ctx.timestamp });
    post(ctx, 'challenge', `${fromName} challenged ${to}: beat ${fmtTime(targetTimeMs)} on the ${course}`);
    for (const p of ctx.db.player.iter()) {
      if (p.name.toLowerCase() === to.toLowerCase()) text(ctx, p.joinCode, 'challenge', `${fromName} challenged you: beat ${fmtTime(targetTimeMs)} on the ${course}. Step onto the snow when you're ready.`);
    }
  }
);

// The agent can also link a code to a name that is already playing (text "join ABCD" after the fact).
export const linkCode = spacetimedb.reducer(
  { name: t.string(), joinCode: t.string() },
  (ctx, { name, joinCode }) => {
    for (const p of ctx.db.player.iter()) {
      if (p.name.toLowerCase() === name.trim().toLowerCase()) ctx.db.player.identity.update({ ...p, joinCode: joinCode.toUpperCase() });
    }
  }
);

// A spectator cheers from the board. Event rows reach every client once.
export const sendCheer = spacetimedb.reducer(
  { fromName: t.string(), toName: t.string(), kind: t.string() },
  (ctx, { fromName, toName, kind }) => {
    const k = ['bell', 'clap', 'fire'].includes(kind) ? kind : 'bell';
    ctx.db.cheer.insert({ fromName: fromName.trim().slice(0, 24) || 'Someone', toName: toName.trim().slice(0, 24), kind: k, at: ctx.timestamp });
    for (const p of ctx.db.player.iter()) if (p.name === toName.trim()) ctx.db.player.identity.update({ ...p, cheers: p.cheers + 1 });
  }
);

// The game uploads a small JPEG of the finish; the agent texts it with the result.
export const pushPhoto = spacetimedb.reducer(
  { runKey: t.string(), jpeg: t.string() },
  (ctx, { runKey, jpeg }) => {
    const r = ctx.db.run.key.find(runKey);
    if (!r || !r.identity.equals(ctx.sender)) throw new SenderError('not your run');
    if (jpeg.length > 400_000) throw new SenderError('photo too large');
    const existing = ctx.db.runPhoto.runKey.find(runKey);
    if (existing) ctx.db.runPhoto.runKey.update({ ...existing, jpeg, at: ctx.timestamp });
    else ctx.db.runPhoto.insert({ runKey, jpeg, at: ctx.timestamp });
  }
);

export const markSent = spacetimedb.reducer({ id: t.u64() }, (ctx, { id }) => {
  const m = ctx.db.outbox.id.find(id);
  if (m) ctx.db.outbox.id.update({ ...m, sent: true });
});

// Every 2 s: drop skiers that stopped streaming (closed the tab without a clean disconnect).
export const tick = spacetimedb.reducer({ onSchedule: tickTimer }, { timer: tickTimer.rowType }, (ctx, _args) => {
  const cutoff = ctx.timestamp.microsSinceUnixEpoch - STALE_MICROS;
  for (const s of ctx.db.skier.iter()) {
    if (s.updatedAt.microsSinceUnixEpoch < cutoff) {
      ctx.db.skier.identity.delete(s.identity);
      const p = ctx.db.player.identity.find(s.identity);
      if (p && p.online) ctx.db.player.identity.update({ ...p, online: false });
    }
  }
});
