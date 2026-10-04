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

// ---------------------------------------------------------------------------------------------- hikes
// Real hikes recorded by the ground truth iPhone app (ground/ in this repo). The phone server registers each
// hike against the terrain and writes it here; the iMessage agent texts the summary from the outbox.

// One second of a hike: where the hiker was, how fast, the slope under them, and how their body moved
// (steps per minute, vertical bounce, hardest step impact). Only the route past the first and last 200 m is
// stored, so no one's home shows up.
const HikeSample = t.object('HikeSample', {
  tS: t.u32(), // seconds since the hike started
  lat: t.f64(),
  lon: t.f64(),
  altM: t.f32(),
  speed: t.f32(),
  slopeDeg: t.f32(),
  cadence: t.f32(),
  bounce: t.f32(),
  impact: t.f32(),
});

const hike = table(
  { name: 'hike', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(), // session id from the phone server
    joinCode: t.string().index('btree'),
    placement: t.string(), // pocket, hand, backpack, hip belt
    startedS: t.f64(),
    endedS: t.f64(),
    distanceM: t.f32(),
    climbM: t.f32(),
    trail: t.string(),
    sacScale: t.string(),
    surface: t.string(),
    motionSamples: t.u32(), // raw 100 Hz samples on the phone server; the rows here are per second
    rateHz: t.f32(),
    gaps: t.u32(),
    gapS: t.f32(),
    cadenceSpm: t.f32(),
    notes: t.array(t.string()), // what the hiker told the agent about the trail
    at: t.timestamp(),
  }
);

const hikeChunk = table(
  { name: 'hike_chunk', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    hikeKey: t.string().index('btree'),
    seq: t.u32(),
    samples: t.array(HikeSample),
  }
);

const hikeStats = table(
  { name: 'hike_stats', public: true },
  {
    id: t.u8().primaryKey(),
    hikes: t.u64(),
    meters: t.f64(),
    seconds: t.f64(),
    motionSamples: t.u64(),
    hikers: t.u32(),
  }
);

// The one identity allowed to write hikes: the phone server, which claims it once with claim_hike_writer.
const hikeWriter = table(
  { name: 'hike_writer' },
  {
    id: t.u8().primaryKey(),
    writer: t.identity(),
  }
);

// ------------------------------------------------------------------------------------------ captures
// One shape for every raw stream Ground Truth collects, for skiing and hiking alike:
//   source 'phone'  IMU / GPS / barometer from the iPhone app
//   source 'game'   G1 joint angles + root pose from the browser game
//   source 'video'  joints estimated from a ski or hike video (the video file itself stays in object storage, see uri)
// A capture declares its channels once; its chunks are flat f32 frames (frame-major, channels.length values per
// frame), so a minute of 100 Hz x 30 channel data is one ~720 KB row with no per-sample field names or tags.
// Regular streams set rateHz and leave tMs empty (frame i is at t0Ms + i*1000/rateHz); irregular ones send tMs.
// Nothing reads these tables yet: they collect only, outside the game's and the agent's paths.
const capture = table(
  { name: 'capture', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(),
    owner: t.identity(), // only this identity can add to or close the capture
    source: t.string().index('btree'), // phone, game, video
    activity: t.string().index('btree'), // ski, hike
    joinCode: t.string().index('btree'),
    ref: t.string(), // run key, hike key or video id this capture belongs to
    uri: t.string(), // where the raw video or file lives, if anywhere
    channels: t.array(t.string()), // e.g. accX..gyroZ, lat, lon, or left_knee_joint..
    rateHz: t.f32(), // 0 = irregular, tMs on every chunk
    meta: t.string(), // JSON: device, placement, course, camera, model version
    frames: t.u64(),
    chunks: t.u32(),
    closed: t.bool(),
    startedAt: t.timestamp(),
    endedAt: t.timestamp(),
  }
);

const captureChunk = table(
  { name: 'capture_chunk', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    captureKey: t.string().index('btree'),
    seq: t.u32(),
    t0Ms: t.u64(), // ms since the capture started
    tMs: t.array(t.u32()), // per-frame offsets from t0Ms, only for irregular streams
    data: t.array(t.f32()),
  }
);

const captureStats = table(
  { name: 'capture_stats', public: true },
  {
    key: t.string().primaryKey(), // source:activity, e.g. game:ski
    captures: t.u64(),
    frames: t.u64(),
    values: t.u64(),
  }
);

const spacetimedb = schema({ player, skier, run, traceChunk, datasetStats, feed, challenge, outbox, cheer, runPhoto, tickTimer, hike, hikeChunk, hikeStats, hikeWriter, capture, captureChunk, captureStats });
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

// ---------------------------------------------------------------------------------------------- hikes

function hikeTotals(ctx: Ctx) {
  return ctx.db.hikeStats.id.find(0) ?? ctx.db.hikeStats.insert({ id: 0, hikes: 0n, meters: 0, seconds: 0, motionSamples: 0n, hikers: 0 });
}

function requireHikeWriter(ctx: Ctx) {
  const w = ctx.db.hikeWriter.id.find(0);
  if (!w || !w.writer.equals(ctx.sender)) throw new SenderError('only the phone server writes hikes');
}

// The phone server calls this once at startup. The first caller becomes the hike writer; anyone else is refused.
export const claimHikeWriter = spacetimedb.reducer({}, ctx => {
  const w = ctx.db.hikeWriter.id.find(0);
  if (!w) ctx.db.hikeWriter.insert({ id: 0, writer: ctx.sender });
  else if (!w.writer.equals(ctx.sender)) throw new SenderError('hike writer already claimed by another identity');
});

// A registered hike. Recording it again (the server re-registers after a fix) updates the row and does not
// count it twice; only the first time queues the summary text and posts to the feed.
export const recordHike = spacetimedb.reducer(
  {
    key: t.string(), joinCode: t.string(), placement: t.string(), startedS: t.f64(), endedS: t.f64(),
    distanceM: t.f32(), climbM: t.f32(), trail: t.string(), sacScale: t.string(), surface: t.string(),
    motionSamples: t.u32(), rateHz: t.f32(), gaps: t.u32(), gapS: t.f32(), cadenceSpm: t.f32(), message: t.string(),
  },
  (ctx, a) => {
    requireHikeWriter(ctx);
    const { message, ...fields } = a;
    const code = fields.joinCode.trim().toUpperCase();
    const existing = ctx.db.hike.key.find(fields.key);
    if (existing) {
      ctx.db.hike.id.update({ ...existing, ...fields, joinCode: code, at: ctx.timestamp });
      return;
    }
    const newHiker = !!code && [...ctx.db.hike.joinCode.filter(code)].length === 0;
    ctx.db.hike.insert({ id: 0n, ...fields, joinCode: code, notes: [], at: ctx.timestamp });
    const s = hikeTotals(ctx);
    ctx.db.hikeStats.id.update({
      ...s, hikes: s.hikes + 1n, meters: s.meters + fields.distanceM, seconds: s.seconds + (fields.endedS - fields.startedS),
      motionSamples: s.motionSamples + BigInt(fields.motionSamples), hikers: s.hikers + (newHiker ? 1 : 0),
    });
    const where = fields.trail ? ` on ${fields.trail}` : '';
    post(ctx, 'hike', `a hiker added ${(fields.distanceM / 1000).toFixed(1)} km${where} to Ground Truth`);
    if (message) text(ctx, code, 'hike', message, fields.key);
  }
);

// Per-second features of a hike, about a minute per chunk. Sending the same seq again replaces it.
export const pushHikeChunk = spacetimedb.reducer(
  { hikeKey: t.string(), seq: t.u32(), samples: t.array(HikeSample) },
  (ctx, { hikeKey, seq, samples }) => {
    requireHikeWriter(ctx);
    if (!ctx.db.hike.key.find(hikeKey)) throw new SenderError('record the hike first');
    if (samples.length === 0 || samples.length > 600) throw new SenderError('1 to 600 samples per chunk');
    for (const c of ctx.db.hikeChunk.hikeKey.filter(hikeKey)) if (c.seq === seq) ctx.db.hikeChunk.id.delete(c.id);
    ctx.db.hikeChunk.insert({ id: 0n, hikeKey, seq, samples });
  }
);

// Something the hiker told the agent about the trail ("slipped on wet roots near the creek").
export const labelHike = spacetimedb.reducer({ key: t.string(), note: t.string() }, (ctx, { key, note }) => {
  requireHikeWriter(ctx);
  const h = ctx.db.hike.key.find(key);
  if (!h) throw new SenderError('no such hike');
  const clean = note.trim().slice(0, 500);
  if (clean) ctx.db.hike.id.update({ ...h, notes: [...h.notes, clean].slice(-20) });
});

// "delete my data": every hike, chunk and hike text tied to a join code.
export const deleteHikes = spacetimedb.reducer({ joinCode: t.string() }, (ctx, { joinCode }) => {
  requireHikeWriter(ctx);
  const code = joinCode.trim().toUpperCase();
  if (!code) return;
  const s = hikeTotals(ctx);
  let hikes = 0n, meters = 0, seconds = 0, samples = 0n;
  for (const h of [...ctx.db.hike.joinCode.filter(code)]) {
    for (const c of [...ctx.db.hikeChunk.hikeKey.filter(h.key)]) ctx.db.hikeChunk.id.delete(c.id);
    hikes += 1n; meters += h.distanceM; seconds += h.endedS - h.startedS; samples += BigInt(h.motionSamples);
    ctx.db.hike.id.delete(h.id);
  }
  for (const m of [...ctx.db.outbox.iter()]) if (m.joinCode === code && m.kind === 'hike') ctx.db.outbox.id.delete(m.id);
  if (hikes > 0n) {
    ctx.db.hikeStats.id.update({
      ...s, hikes: s.hikes - hikes, meters: Math.max(0, s.meters - meters), seconds: Math.max(0, s.seconds - seconds),
      motionSamples: s.motionSamples - samples, hikers: Math.max(0, s.hikers - 1),
    });
  }
});

// ---------------------------------------------------------------------------------------------- captures

const SOURCES = ['phone', 'game', 'video'];
const ACTIVITIES = ['ski', 'hike'];
const CHUNK_VALUES_MAX = 262_144; // 1 MB of f32 per chunk

function bumpCaptureStats(ctx: Ctx, key: string, captures: bigint, frames: bigint, values: bigint) {
  const s = ctx.db.captureStats.key.find(key);
  if (s) ctx.db.captureStats.key.update({ ...s, captures: s.captures + captures, frames: s.frames + frames, values: s.values + values });
  else ctx.db.captureStats.insert({ key, captures, frames, values });
}

// Starts a capture. Opening the same key again by its owner is a no-op, so clients can retry freely.
export const openCapture = spacetimedb.reducer(
  { key: t.string(), source: t.string(), activity: t.string(), joinCode: t.string(), ref: t.string(), uri: t.string(), channels: t.array(t.string()), rateHz: t.f32(), meta: t.string() },
  (ctx, a) => {
    if (!SOURCES.includes(a.source)) throw new SenderError(`source must be one of ${SOURCES.join(', ')}`);
    if (!ACTIVITIES.includes(a.activity)) throw new SenderError(`activity must be one of ${ACTIVITIES.join(', ')}`);
    if (a.channels.length === 0 || a.channels.length > 512) throw new SenderError('1 to 512 channels');
    if (!a.key || a.key.length > 64) throw new SenderError('key is 1 to 64 characters');
    if (ctx.db.capture.key.find(a.key)) return;
    ctx.db.capture.insert({ id: 0n, ...a, owner: ctx.sender, joinCode: a.joinCode.trim().toUpperCase(), meta: a.meta.slice(0, 8000), frames: 0n, chunks: 0, closed: false, startedAt: ctx.timestamp, endedAt: ctx.timestamp });
    bumpCaptureStats(ctx, `${a.source}:${a.activity}`, 1n, 0n, 0n);
  }
);

// A block of frames. Sending the same seq again replaces it (and the counts follow).
export const pushCaptureChunk = spacetimedb.reducer(
  { captureKey: t.string(), seq: t.u32(), t0Ms: t.u64(), tMs: t.array(t.u32()), data: t.array(t.f32()) },
  (ctx, { captureKey, seq, t0Ms, tMs, data }) => {
    const c = ctx.db.capture.key.find(captureKey);
    if (!c) throw new SenderError('open the capture first');
    if (!c.owner.equals(ctx.sender)) throw new SenderError('not your capture');
    if (c.closed) throw new SenderError('capture is closed');
    const width = c.channels.length;
    if (data.length === 0 || data.length > CHUNK_VALUES_MAX || data.length % width !== 0) throw new SenderError(`data must be whole frames of ${width} values, at most ${CHUNK_VALUES_MAX}`);
    const n = data.length / width;
    if (c.rateHz <= 0 && tMs.length !== n) throw new SenderError('irregular capture: one tMs per frame');
    if (c.rateHz > 0 && tMs.length !== 0) throw new SenderError('regular capture: leave tMs empty');
    let frames = BigInt(n), chunks = 1;
    for (const old of [...ctx.db.captureChunk.captureKey.filter(captureKey)]) {
      if (old.seq !== seq) continue;
      frames -= BigInt(old.data.length / width); chunks -= 1;
      ctx.db.captureChunk.id.delete(old.id);
    }
    ctx.db.captureChunk.insert({ id: 0n, captureKey, seq, t0Ms, tMs, data });
    ctx.db.capture.id.update({ ...c, frames: c.frames + frames, chunks: c.chunks + chunks, endedAt: ctx.timestamp });
    bumpCaptureStats(ctx, `${c.source}:${c.activity}`, 0n, frames, frames * BigInt(width));
  }
);

export const closeCapture = spacetimedb.reducer({ key: t.string() }, (ctx, { key }) => {
  const c = ctx.db.capture.key.find(key);
  if (c && !c.closed && c.owner.equals(ctx.sender)) ctx.db.capture.id.update({ ...c, closed: true, endedAt: ctx.timestamp });
});
