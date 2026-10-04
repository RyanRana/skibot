// Ground Truth: one shared mountain. Every skier who opens the game is on the same real piste at the
// same time, every finished run grows the motion + terrain dataset, and the iMessage agent reads the
// outbox to text people their results.
import { schema, table, t, SenderError, type InferSchema, type ReducerCtx, type ViewCtx, type AnonymousViewCtx } from 'spacetimedb/server';
import { ScheduleAt, Timestamp } from 'spacetimedb';

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

// -------------------------------------------------------------------------------------------- tenants
// Organisations own everything outside the live game. Hazard Intelligence is the one public tenant: our maps, video,
// GoPro telemetry and training inputs are open data. Every other organisation (a resort, a rescue team, a lab, a
// partner app) is private. Tenant rows sit in private tables: the public_* views serve public tenants' rows to anyone,
// and the my_* views serve each caller the rows of the organisations they belong to. The game, hike and outbox tables
// above stay as they are, because the game, the agent and the phone server read and write them directly.

const account = table(
  { name: 'account' },
  {
    identity: t.identity().primaryKey(),
    name: t.string(),
    kind: t.string(), // person, or service for a server such as the loader or the policy API
    activeTenant: t.string(), // where this identity's new data lands
    createdAt: t.timestamp(),
    lastSeen: t.timestamp(),
  }
);

const tenant = table(
  { name: 'tenant' },
  {
    id: t.string().primaryKey(), // slug: hazard-intelligence, vail-ski-patrol
    name: t.string(),
    kind: t.string(), // team, resort, rescue, lab, partner
    visibility: t.string(), // public: anyone reads its rows; private: members only
    createdBy: t.identity(),
    createdAt: t.timestamp(),
  }
);

// owner and admin manage the organisation; member and service read and write its data; viewer only reads.
const member = table(
  { name: 'member' },
  {
    id: t.u64().primaryKey().autoInc(),
    tenantId: t.string().index('btree'),
    identity: t.identity().index('btree'),
    role: t.string(),
    addedBy: t.identity(),
    createdAt: t.timestamp(),
  }
);

const invite = table(
  { name: 'invite' },
  {
    code: t.string().primaryKey(),
    tenantId: t.string().index('btree'),
    role: t.string(),
    maxUses: t.u32(),
    uses: t.u32(),
    createdBy: t.identity(),
    createdAt: t.timestamp(),
  }
);

// The one identity that runs the platform: it creates the public tenant. Claimed once, right after publishing.
const platformAdmin = table(
  { name: 'platform_admin' },
  {
    id: t.u8().primaryKey(),
    admin: t.identity(),
  }
);

// ----------------------------------------------------------------------------------------- maps and terrain
// Every row below carries the tenant that owns it. Coordinates are in each course's local frame: x east, y north,
// z up, metres, around (originLat, originLon); heights are relative to zDatumMsl.

const resort = table(
  { name: 'resort' },
  {
    id: t.string().primaryKey(), // kitzbuhel
    tenant: t.string().index('btree'),
    name: t.string(),
    country: t.string(),
    lat: t.f64(),
    lon: t.f64(),
    osm: t.string(), // relation/85657
    source: t.string(),
    meta: t.string(), // JSON
    updatedAt: t.timestamp(),
  }
);

// A point on a course centreline, s metres along it.
const LinePoint = t.object('LinePoint', { x: t.f32(), y: t.f32(), z: t.f32(), s: t.f32() });

// A slalom gate: pass between the turn pole and the outer pole (pole bases on the snow).
const Gate = t.object('Gate', {
  s: t.f32(),
  side: t.string(),
  color: t.string(),
  turn: t.string(),
  turnX: t.f32(), turnY: t.f32(), turnZ: t.f32(),
  outerX: t.f32(), outerY: t.f32(), outerZ: t.f32(),
});

// A ski run, a hiking route or a climbing wall: the line a person or robot follows and where it starts.
const course = table(
  { name: 'course' },
  {
    id: t.string().primaryKey(), // kitzbuhel-streif
    tenant: t.string().index('btree'),
    resortId: t.string().index('btree'), // empty for a trail
    name: t.string(),
    activity: t.string().index('btree'), // ski, hike, climb
    difficulty: t.string(),
    lengthM: t.f32(),
    dropM: t.f32(),
    meanSlopeDeg: t.f32(),
    maxSlopeDeg: t.f32(),
    originLat: t.f64(),
    originLon: t.f64(),
    zDatumMsl: t.f32(),
    line: t.array(LinePoint), // the course itself
    fullRun: t.array(LinePoint), // the whole run the course was cut from
    gates: t.array(Gate),
    startPose: t.string(), // JSON: where and how the robot starts
    scene: t.string(), // MuJoCo scene for the G1 on this course
    stats: t.string(), // JSON: build stats, OSM ids, sources
    source: t.string(),
    license: t.string(),
    updatedAt: t.timestamp(),
  }
);

// A grid on a regular lattice: ny rows of nx values, row-major, cell metres apart, starting at (x0, y0). The layer
// says what the values are: ground elevation, canopy height, far-field terrain, or a mosaic of training tiles.
// The values live in terrain_chunk rows of whole rows, so one course is a handful of rows of about 100 KB.
const terrain = table(
  { name: 'terrain' },
  {
    id: t.string().primaryKey(), // kitzbuhel-streif, bird-hills/canopy, ski-tiles-v4
    tenant: t.string().index('btree'),
    courseId: t.string().index('btree'), // empty for a training mosaic
    layer: t.string(), // elevation, canopy, far, tiles
    nx: t.u32(),
    ny: t.u32(),
    cellM: t.f32(),
    x0: t.f32(),
    y0: t.f32(),
    zMin: t.f32(),
    zMax: t.f32(),
    rows: t.u32(), // rows received so far; complete when rows == ny
    source: t.string(),
    license: t.string(),
    meta: t.string(), // JSON: projection, DEM, smoothing
    updatedAt: t.timestamp(),
  }
);

const terrainChunk = table(
  { name: 'terrain_chunk' },
  {
    id: t.u64().primaryKey().autoInc(),
    terrainId: t.string().index('btree'),
    tenant: t.string().index('btree'),
    row0: t.u32(),
    heights: t.array(t.f32()), // whole rows: length is a multiple of nx
  }
);

// ground.py: every trail or road around a place, each with an elevation profile, like OpenSkiMap's runs.
const trailNetwork = table(
  { name: 'trail_network' },
  {
    id: t.string().primaryKey(), // annarbor
    tenant: t.string().index('btree'),
    name: t.string(),
    place: t.string(),
    kind: t.string(), // trails, walk, roads, all
    centerLat: t.f64(),
    centerLon: t.f64(),
    radiusM: t.f32(),
    ways: t.u32(),
    lengthKm: t.f32(),
    areas: t.u32(),
    sources: t.string(), // JSON: OSM and elevation licences
    meta: t.string(), // JSON: the rest of meta.json
    updatedAt: t.timestamp(),
  }
);

const LatLon = t.object('LatLon', { lat: t.f64(), lon: t.f64() });

const trail = table(
  { name: 'trail' },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(), // network/way/123
    network: t.string().index('btree'),
    tenant: t.string().index('btree'),
    osm: t.string(), // way/4687992
    name: t.string(),
    highway: t.string(),
    surface: t.string(),
    sacScale: t.string(),
    area: t.string(), // the park or reserve it runs through
    lengthM: t.f32(),
    climbM: t.f32(),
    descentM: t.f32(),
    maxGradeDeg: t.f32(),
    path: t.array(LatLon),
    profileStepM: t.f32(),
    profile: t.array(t.f32()), // elevation every profileStepM along the way, metres MSL
    tags: t.string(), // JSON: every other OSM tag and statistic
  }
);

// Parks, nature reserves and protected areas holding a network's trails.
const area = table(
  { name: 'area' },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(), // network/way/27785567
    network: t.string().index('btree'),
    tenant: t.string().index('btree'),
    osm: t.string(),
    name: t.string(),
    kind: t.string(), // park, nature_reserve, protected_area
    ways: t.u32(),
    lengthM: t.f32(),
    geometry: t.string(), // GeoJSON MultiPolygon
    stats: t.string(), // JSON
  }
);

// ------------------------------------------------------------------------------------------- recordings
// One shape for every stream an organisation records: phone IMU and GPS, game joint angles, joints from video, GoPro
// telemetry, robot rollouts in simulation. A recording declares its channels and their units once; its chunks are
// flat f32 frames (frame-major, channels.length values per frame). Regular streams set rateHz and leave tMs empty
// (frame i is at t0Ms + i * 1000 / rateHz); irregular ones send one tMs per frame. The capture tables above are the
// first version of this, kept as they were so publishing never disconnects a client; nothing writes them.
const recording = table(
  { name: 'recording' },
  {
    key: t.string().primaryKey(), // video/fXVIpXY3rI4, gopro/GX010060ap
    tenant: t.string().index('btree'),
    source: t.string().index('btree'), // phone, game, video, gopro, sim
    activity: t.string().index('btree'), // ski, hike, climb
    ref: t.string(), // the course, hike, run or video it belongs to
    uri: t.string(), // where the raw file lives (a video URL), if anywhere
    channels: t.array(t.string()),
    units: t.array(t.string()), // one per channel: rad, deg, deg/s, m, m/s, m/s2, bool
    rateHz: t.f32(), // 0 = irregular
    frames: t.u64(),
    chunks: t.u32(),
    title: t.string(),
    license: t.string(),
    meta: t.string(), // JSON: device, model version, author, anything else
    writer: t.identity(),
    closed: t.bool(),
    startedAt: t.timestamp(),
    updatedAt: t.timestamp(),
  }
);

const recordingChunk = table(
  { name: 'recording_chunk' },
  {
    id: t.u64().primaryKey().autoInc(),
    recordingKey: t.string().index('btree'),
    tenant: t.string().index('btree'),
    seq: t.u32(),
    t0Ms: t.u64(),
    tMs: t.array(t.u32()),
    data: t.array(t.f32()),
  }
);


// Points on a course or a terrain: trees and bushes along a trail, the holds on a climbing wall. Each point is
// columns.length values (x, y, crown radius, height...), stored flat in point_chunk rows.
const pointSet = table(
  { name: 'point_set' },
  {
    id: t.string().primaryKey(), // bird-hills/trees
    tenant: t.string().index('btree'),
    courseId: t.string().index('btree'),
    kind: t.string(), // trees, holds
    columns: t.array(t.string()),
    units: t.array(t.string()),
    count: t.u32(), // points received so far
    source: t.string(),
    license: t.string(),
    meta: t.string(), // JSON
    updatedAt: t.timestamp(),
  }
);

const pointChunk = table(
  { name: 'point_chunk' },
  {
    id: t.u64().primaryKey().autoInc(),
    setId: t.string().index('btree'),
    tenant: t.string().index('btree'),
    seq: t.u32(),
    data: t.array(t.f32()), // whole points
  }
);

// --------------------------------------------------------------------------------------- training inputs
// Time spans inside a recording: a GoPro descent, a turn, a climbing move.
const segment = table(
  { name: 'segment' },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(),
    recordingKey: t.string().index('btree'),
    tenant: t.string().index('btree'),
    kind: t.string(), // descent, turn, move
    t0Ms: t.u64(),
    t1Ms: t.u64(),
    label: t.string(),
    meta: t.string(), // JSON
  }
);

// A robot pose measured from people: the G1 joint targets (radians) for one phase of a skill.
const referencePose = table(
  { name: 'reference_pose' },
  {
    id: t.string().primaryKey(), // ski/neutral
    tenant: t.string().index('btree'),
    activity: t.string(),
    phase: t.string(),
    robot: t.string(), // unitree_g1_29dof
    joints: t.array(t.string()),
    values: t.array(t.f32()),
    frames: t.u32(), // video frames averaged
    human: t.string(), // JSON: the human joint angles behind it, degrees
    sources: t.string(), // JSON: which videos, how many frames each
    updatedAt: t.timestamp(),
  }
);

// What a model trains on: named lists of courses, terrains, captures and reference poses.
const trainingSet = table(
  { name: 'training_set' },
  {
    id: t.string().primaryKey(), // g1-ski
    tenant: t.string().index('btree'),
    name: t.string(),
    activity: t.string(),
    robot: t.string(),
    description: t.string(),
    courses: t.array(t.string()),
    terrains: t.array(t.string()),
    recordings: t.array(t.string()),
    referencePoses: t.array(t.string()),
    meta: t.string(), // JSON: curriculum, tiles, anything the trainer reads
    updatedAt: t.timestamp(),
  }
);

// ------------------------------------------------------------------------------------------------ models
// The policy registry. Weights and the ONNX runtime stay on Modal; this is everything about a policy that is not code.
const policy = table(
  { name: 'policy' },
  {
    id: t.string().primaryKey(), // g1-ski
    tenant: t.string().index('btree'),
    name: t.string(),
    robot: t.string(),
    skill: t.string(),
    obsDim: t.u32(),
    actionDim: t.u32(),
    controlHz: t.f32(),
    runtime: t.string(), // modal
    endpoint: t.string(), // where the runtime serves it
    trainingSet: t.string(),
    benchmark: t.string(), // JSON
    spec: t.string(), // JSON: the full meta.json (architecture, every observation slice, action layout, provenance)
    updatedAt: t.timestamp(),
  }
);

// One training run: what it trained on, how far it got and how it scored.
const trainRun = table(
  { name: 'train_run' },
  {
    id: t.string().primaryKey(), // g1-hike/20261003-201557
    tenant: t.string().index('btree'),
    policyId: t.string().index('btree'),
    trainingSet: t.string(),
    hardware: t.string(),
    status: t.string(), // running, done, stopped
    iterations: t.u32(),
    checkpoint: t.string(), // where the weights are, e.g. a Modal volume path
    metrics: t.string(), // JSON
    benchmark: t.string(), // JSON
    notes: t.string(),
    startedAt: t.timestamp(),
    updatedAt: t.timestamp(),
  }
);


// How a policy checkpoint did: on the fixed benchmark, riding a course, or walking a trail stretch.
const evaluation = table(
  { name: 'evaluation' },
  {
    id: t.string().primaryKey(), // benchmark/v2-1325, ride/<checkpoint>/<course>
    tenant: t.string().index('btree'),
    policyId: t.string().index('btree'),
    runId: t.string().index('btree'),
    checkpoint: t.string(),
    kind: t.string(), // benchmark, ride, hike
    courseId: t.string(),
    summary: t.string(), // JSON: the headline numbers
    detail: t.string(), // JSON: per tile, per gate, crash events
    at: t.timestamp(),
  }
);


// --------------------------------------------------------------------------------------------- app users
// The people using the Ground Truth app and its iMessage agent, keyed by their iMessage handle: a phone number in
// E.164 (+19195550123) or an email. Phone numbers, messages and raw phone recordings are personal, so these rows
// belong to the private ground-truth-app tenant and only its owners, admins and services (the phone server, the
// agent) read them, through the my_app_* views. deleteAppUser removes a person and everything recorded for them.

const appUser = table(
  { name: 'app_user' },
  {
    handle: t.string().primaryKey(),
    tenant: t.string().index('btree'),
    code: t.string().index('btree'), // the join code shared by their game runs and hikes
    name: t.string(),
    optedOut: t.bool(), // texted STOP: nothing goes out until START
    pendingLabel: t.string(), // the session the agent asked about and is waiting to hear back on
    pendingDelete: t.bool(),
    invites: t.u32(),
    notes: t.array(t.string()), // what the agent remembers about them
    consentVersion: t.string(),
    consentedAt: t.timestamp(),
    firstSeen: t.timestamp(),
    lastSeen: t.timestamp(),
  }
);

// Every text, both ways: what people sent the agent, what the agent and the server sent them.
const appMessage = table(
  { name: 'app_message' },
  {
    id: t.u64().primaryKey().autoInc(),
    key: t.string().unique(), // the sender's own id, so a retry never duplicates
    handle: t.string().index('btree'),
    tenant: t.string().index('btree'),
    direction: t.string(), // in, out
    kind: t.string(), // chat, welcome, nudge, summary, result
    text: t.string(),
    session: t.string(),
    status: t.string(), // out: pending, sent, failed
    error: t.string(),
    at: t.timestamp(),
  }
);

// The /g/<token> links that open the app. The token is a secret: whoever has it can upload a session.
const appInvite = table(
  { name: 'app_invite' },
  {
    token: t.string().primaryKey(),
    tenant: t.string().index('btree'),
    handle: t.string().index('btree'),
    code: t.string(),
    link: t.string(),
    createdAt: t.timestamp(),
  }
);

// One recording session from the phone: consent, upload, registration. Its raw streams are recordings
// phone/<id>/imu, phone/<id>/gps and phone/<id>/baro; the registered hike is the hike row with key <id>.
const appSession = table(
  { name: 'app_session' },
  {
    id: t.string().primaryKey(),
    tenant: t.string().index('btree'),
    handle: t.string().index('btree'),
    code: t.string(),
    token: t.string(),
    status: t.string(), // consented, uploading, registering, done, failed
    consent: t.string(), // JSON: version, placement, device, when, from where
    meta: t.string(), // JSON: the phone's meta.json (device, gaps, pedometer, files)
    summary: t.string(), // JSON: what ground/register.py measured
    labels: t.array(t.string()),
    error: t.string(),
    createdAt: t.timestamp(),
    updatedAt: t.timestamp(),
  }
);

// ---------------------------------------------------------------------------------------- platform and API
// Servers that act for every organisation: the policy API logs calls for whichever organisation owns the key.
const platformService = table(
  { name: 'platform_service' },
  {
    identity: t.identity().primaryKey(),
    name: t.string(),
    kind: t.string(), // api
    addedAt: t.timestamp(),
  }
);

// Model API keys. Only the SHA-256 of a key is stored; the key itself is shown once, to whoever made it.
const apiKey = table(
  { name: 'api_key' },
  {
    hash: t.string().primaryKey(),
    tenant: t.string().index('btree'),
    prefix: t.string(), // the first characters, to tell keys apart
    label: t.string(),
    createdBy: t.identity(),
    createdAt: t.timestamp(),
    lastUsedAt: t.timestamp(),
    calls: t.u64(),
    revoked: t.bool(),
  }
);

// Every Model API call: which organisation (by key), which policy, which route, how it went.
const apiCall = table(
  { name: 'api_call' },
  {
    id: t.u64().primaryKey().autoInc(),
    tenant: t.string().index('btree'), // the key's organisation; calls without a key count for the public tenant
    keyPrefix: t.string(),
    policyId: t.string().index('btree'),
    route: t.string(), // list, spec, onnx, weights, infer, ws, missions
    status: t.u16(),
    ms: t.f32(),
    at: t.timestamp(),
  }
);

const spacetimedb = schema({
  player, skier, run, traceChunk, datasetStats, feed, challenge, outbox, cheer, runPhoto, tickTimer, hike, hikeChunk, hikeStats,
  hikeWriter, capture, captureChunk, captureStats, account, tenant, member, invite, platformAdmin, resort, course, terrain,
  terrainChunk, trailNetwork, trail, area, recording, recordingChunk, pointSet, pointChunk, segment, referencePose, trainingSet, policy, trainRun, evaluation,
  appUser, appMessage, appInvite, appSession, platformService, apiKey, apiCall,
});
export default spacetimedb;

type Ctx = ReducerCtx<InferSchema<typeof spacetimedb>>;
type VCtx = ViewCtx<InferSchema<typeof spacetimedb>>;
type AVCtx = AnonymousViewCtx<InferSchema<typeof spacetimedb>>;

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
  dropHikes(ctx, joinCode);
});

function dropHikes(ctx: Ctx, joinCode: string) {
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
}

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

// ---------------------------------------------------------------------------------------------- tenants

const HI = 'hazard-intelligence';
const TENANT_KINDS = ['team', 'resort', 'rescue', 'lab', 'partner'];
const ROLES = ['owner', 'admin', 'member', 'service', 'viewer'];
const MANAGE = ['owner', 'admin'];
const WRITE = ['owner', 'admin', 'member', 'service'];
const SLUG = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no 0/O or 1/I, so a code survives being read aloud

function membership(ctx: Ctx, tenantId: string, who = ctx.sender) {
  for (const m of ctx.db.member.identity.filter(who)) if (m.tenantId === tenantId) return m;
  return undefined;
}

function requireRole(ctx: Ctx, tenantId: string, roles: string[]) {
  if (!ctx.db.tenant.id.find(tenantId)) throw new SenderError(`no organisation ${tenantId}`);
  const m = membership(ctx, tenantId);
  if (!m || !roles.includes(m.role)) throw new SenderError(`needs ${roles.join(' or ')} in ${tenantId}`);
  return m;
}

function upsertAccount(ctx: Ctx, fields: { name?: string; kind?: string; activeTenant?: string }) {
  const a = ctx.db.account.identity.find(ctx.sender);
  if (a) {
    ctx.db.account.identity.update({ ...a, ...Object.fromEntries(Object.entries(fields).filter(([, v]) => v !== undefined)), lastSeen: ctx.timestamp });
  } else {
    ctx.db.account.insert({ identity: ctx.sender, name: fields.name ?? '', kind: fields.kind ?? 'person', activeTenant: fields.activeTenant ?? '', createdAt: ctx.timestamp, lastSeen: ctx.timestamp });
  }
}

function addMember(ctx: Ctx, tenantId: string, who: typeof ctx.sender, role: string) {
  const m = membership(ctx, tenantId, who);
  if (m) {
    // An invite never demotes: an admin redeeming a member code stays an admin.
    if (ROLES.indexOf(role) < ROLES.indexOf(m.role)) ctx.db.member.id.update({ ...m, role });
    return;
  }
  ctx.db.member.insert({ id: 0n, tenantId, identity: who, role, addedBy: ctx.sender, createdAt: ctx.timestamp });
}

function owners(ctx: Ctx, tenantId: string) {
  return [...ctx.db.member.tenantId.filter(tenantId)].filter(m => m.role === 'owner').length;
}

// The publisher calls this once after the first publish. It makes the caller the platform admin and the owner of the
// public Hazard Intelligence tenant, which every loader and server then writes under.
export const claimPlatformAdmin = spacetimedb.reducer({}, ctx => {
  const a = ctx.db.platformAdmin.id.find(0);
  if (a && !a.admin.equals(ctx.sender)) throw new SenderError('platform admin already claimed by another identity');
  if (!a) ctx.db.platformAdmin.insert({ id: 0, admin: ctx.sender });
  if (!ctx.db.tenant.id.find(HI)) {
    ctx.db.tenant.insert({ id: HI, name: 'Hazard Intelligence', kind: 'team', visibility: 'public', createdBy: ctx.sender, createdAt: ctx.timestamp });
  }
  addMember(ctx, HI, ctx.sender, 'owner');
  upsertAccount(ctx, { activeTenant: HI });
});

// Anyone signed in can start an organisation and becomes its owner. New organisations are private unless asked.
export const createTenant = spacetimedb.reducer(
  { id: t.string(), name: t.string(), kind: t.string(), visibility: t.string() },
  (ctx, { id, name, kind, visibility }) => {
    const slug = id.trim().toLowerCase();
    if (!SLUG.test(slug) || slug.length < 3 || slug.length > 48) throw new SenderError('id is 3 to 48 lowercase letters, digits and dashes');
    if (ctx.db.tenant.id.find(slug)) throw new SenderError(`${slug} is taken`);
    if (!TENANT_KINDS.includes(kind)) throw new SenderError(`kind must be one of ${TENANT_KINDS.join(', ')}`);
    if (visibility !== 'public' && visibility !== 'private') throw new SenderError('visibility is public or private');
    ctx.db.tenant.insert({ id: slug, name: name.trim().slice(0, 80) || slug, kind, visibility, createdBy: ctx.sender, createdAt: ctx.timestamp });
    addMember(ctx, slug, ctx.sender, 'owner');
    upsertAccount(ctx, { activeTenant: slug });
  }
);

export const updateTenant = spacetimedb.reducer(
  { id: t.string(), name: t.string(), kind: t.string(), visibility: t.string() },
  (ctx, { id, name, kind, visibility }) => {
    requireRole(ctx, id, MANAGE);
    if (!TENANT_KINDS.includes(kind)) throw new SenderError(`kind must be one of ${TENANT_KINDS.join(', ')}`);
    if (visibility !== 'public' && visibility !== 'private') throw new SenderError('visibility is public or private');
    const tn = ctx.db.tenant.id.find(id)!;
    ctx.db.tenant.id.update({ ...tn, name: name.trim().slice(0, 80) || tn.name, kind, visibility });
  }
);

// An invite code joins whoever redeems it to the organisation with the given role (never owner).
export const createInvite = spacetimedb.reducer(
  { tenantId: t.string(), role: t.string(), maxUses: t.u32() },
  (ctx, { tenantId, role, maxUses }) => {
    const me = requireRole(ctx, tenantId, MANAGE);
    if (!ROLES.includes(role) || role === 'owner') throw new SenderError('role is admin, member, service or viewer');
    if (role === 'admin' && me.role !== 'owner') throw new SenderError('only an owner invites admins');
    let code = '';
    do {
      code = '';
      for (let i = 0; i < 8; i++) code += CODE_CHARS[ctx.random.integerInRange(0, CODE_CHARS.length - 1)];
    } while (ctx.db.invite.code.find(code));
    ctx.db.invite.insert({ code, tenantId, role, maxUses: Math.max(1, Math.min(maxUses, 10_000)), uses: 0, createdBy: ctx.sender, createdAt: ctx.timestamp });
  }
);

export const revokeInvite = spacetimedb.reducer({ code: t.string() }, (ctx, { code }) => {
  const inv = ctx.db.invite.code.find(code.trim().toUpperCase());
  if (!inv) return;
  requireRole(ctx, inv.tenantId, MANAGE);
  ctx.db.invite.code.delete(inv.code);
});

export const redeemInvite = spacetimedb.reducer({ code: t.string(), name: t.string() }, (ctx, { code, name }) => {
  const inv = ctx.db.invite.code.find(code.trim().toUpperCase());
  if (!inv || inv.uses >= inv.maxUses) throw new SenderError('that invite code is not valid');
  addMember(ctx, inv.tenantId, ctx.sender, inv.role);
  ctx.db.invite.code.update({ ...inv, uses: inv.uses + 1 });
  upsertAccount(ctx, { name: name.trim().slice(0, 48) || undefined, kind: inv.role === 'service' ? 'service' : undefined, activeTenant: inv.tenantId });
});

// Owners and admins can change roles; nobody can remove or demote the last owner.
export const setMemberRole = spacetimedb.reducer(
  { tenantId: t.string(), identity: t.identity(), role: t.string() },
  (ctx, { tenantId, identity, role }) => {
    const me = requireRole(ctx, tenantId, MANAGE);
    if (!ROLES.includes(role)) throw new SenderError(`role must be one of ${ROLES.join(', ')}`);
    if ((role === 'owner' || role === 'admin') && me.role !== 'owner') throw new SenderError('only an owner grants owner or admin');
    const m = membership(ctx, tenantId, identity);
    if (!m) throw new SenderError('not a member');
    if (m.role === 'owner' && role !== 'owner' && owners(ctx, tenantId) <= 1) throw new SenderError('an organisation needs an owner');
    ctx.db.member.id.update({ ...m, role });
  }
);

// Remove someone (owners and admins), or leave (anyone, for themselves).
export const removeMember = spacetimedb.reducer({ tenantId: t.string(), identity: t.identity() }, (ctx, { tenantId, identity }) => {
  if (!identity.equals(ctx.sender)) requireRole(ctx, tenantId, MANAGE);
  const m = membership(ctx, tenantId, identity);
  if (!m) return;
  if (m.role === 'owner' && owners(ctx, tenantId) <= 1) throw new SenderError('an organisation needs an owner');
  ctx.db.member.id.delete(m.id);
});

export const setActiveTenant = spacetimedb.reducer({ tenantId: t.string() }, (ctx, { tenantId }) => {
  if (!membership(ctx, tenantId)) throw new SenderError(`not a member of ${tenantId}`);
  upsertAccount(ctx, { activeTenant: tenantId });
});

export const setProfile = spacetimedb.reducer({ name: t.string() }, (ctx, { name }) => {
  upsertAccount(ctx, { name: name.trim().slice(0, 48) });
});

const TenantRole = t.row('TenantRole', {
  id: t.string(),
  name: t.string(),
  kind: t.string(),
  visibility: t.string(),
  role: t.string(),
});

export const myAccount = spacetimedb.view({ name: 'my_account', public: true }, t.option(account.rowType), ctx => {
  return ctx.db.account.identity.find(ctx.sender) ?? undefined;
});

export const myTenant = spacetimedb.view({ name: 'my_tenant', public: true }, t.array(TenantRole), ctx => {
  const out = [];
  for (const m of ctx.db.member.identity.filter(ctx.sender)) {
    const tn = ctx.db.tenant.id.find(m.tenantId);
    if (tn) out.push({ id: tn.id, name: tn.name, kind: tn.kind, visibility: tn.visibility, role: m.role });
  }
  return out;
});

export const publicTenant = spacetimedb.anonymousView({ name: 'public_tenant', public: true }, t.array(tenant.rowType), ctx => {
  return [...ctx.db.tenant.iter()].filter(tn => tn.visibility === 'public');
});

// Owners and admins see their organisations' members and live invite codes.
function managed(ctx: VCtx) {
  return [...ctx.db.member.identity.filter(ctx.sender)].filter(m => MANAGE.includes(m.role)).map(m => m.tenantId);
}

export const myMember = spacetimedb.view({ name: 'my_member', public: true }, t.array(member.rowType), ctx => {
  return managed(ctx).flatMap(id => [...ctx.db.member.tenantId.filter(id)]);
});

export const myInvite = spacetimedb.view({ name: 'my_invite', public: true }, t.array(invite.rowType), ctx => {
  return managed(ctx).flatMap(id => [...ctx.db.invite.tenantId.filter(id)]);
});

// ------------------------------------------------------------------------------------------- tenant data
// Loaders and servers write maps, training inputs and the policy registry through these reducers. Each one names the
// tenant it writes for and needs a member, service, admin or owner role there; writing a row again replaces it.

function requireWriter(ctx: Ctx, tenantId: string) {
  return requireRole(ctx, tenantId, WRITE);
}

function own(ctx: Ctx, tenantId: string, existing: { tenant: string } | null | undefined, what: string) {
  requireWriter(ctx, tenantId);
  if (existing && existing.tenant !== tenantId) throw new SenderError(`${what} belongs to ${existing.tenant}`);
}

const CHUNK_F32_MAX = 262_144; // 1 MB of f32 per row, the same bound as capture chunks

export const upsertResort = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), name: t.string(), country: t.string(), lat: t.f64(), lon: t.f64(), osm: t.string(), source: t.string(), meta: t.string() },
  (ctx, a) => {
    const old = ctx.db.resort.id.find(a.id);
    own(ctx, a.tenant, old, `resort ${a.id}`);
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.resort.id.update(row);
    else ctx.db.resort.insert(row);
  }
);

export const upsertCourse = spacetimedb.reducer(
  {
    id: t.string(), tenant: t.string(), resortId: t.string(), name: t.string(), activity: t.string(), difficulty: t.string(),
    lengthM: t.f32(), dropM: t.f32(), meanSlopeDeg: t.f32(), maxSlopeDeg: t.f32(), originLat: t.f64(), originLon: t.f64(),
    zDatumMsl: t.f32(), line: t.array(LinePoint), fullRun: t.array(LinePoint), gates: t.array(Gate), startPose: t.string(),
    scene: t.string(), stats: t.string(), source: t.string(), license: t.string(),
  },
  (ctx, a) => {
    const old = ctx.db.course.id.find(a.id);
    own(ctx, a.tenant, old, `course ${a.id}`);
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.course.id.update(row);
    else ctx.db.course.insert(row);
  }
);

// Declares a heightfield and clears any heights it had; pushTerrainRows then fills it.
export const putTerrain = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), courseId: t.string(), layer: t.string(), nx: t.u32(), ny: t.u32(), cellM: t.f32(), x0: t.f32(), y0: t.f32(), zMin: t.f32(), zMax: t.f32(), source: t.string(), license: t.string(), meta: t.string() },
  (ctx, a) => {
    const old = ctx.db.terrain.id.find(a.id);
    if (!a.layer) throw new SenderError('name the layer: elevation, canopy, far or tiles');
    own(ctx, a.tenant, old, `terrain ${a.id}`);
    if (a.nx === 0 || a.ny === 0) throw new SenderError('nx and ny must be positive');
    for (const c of [...ctx.db.terrainChunk.terrainId.filter(a.id)]) ctx.db.terrainChunk.id.delete(c.id);
    const row = { ...a, rows: 0, updatedAt: ctx.timestamp };
    if (old) ctx.db.terrain.id.update(row);
    else ctx.db.terrain.insert(row);
  }
);

// Whole rows of heights starting at row0. Sending the same row0 again replaces that block.
export const pushTerrainRows = spacetimedb.reducer(
  { terrainId: t.string(), row0: t.u32(), heights: t.array(t.f32()) },
  (ctx, { terrainId, row0, heights }) => {
    const tr = ctx.db.terrain.id.find(terrainId);
    if (!tr) throw new SenderError('put the terrain first');
    requireWriter(ctx, tr.tenant);
    if (heights.length === 0 || heights.length > CHUNK_F32_MAX || heights.length % tr.nx !== 0) throw new SenderError(`heights must be whole rows of ${tr.nx}, at most ${CHUNK_F32_MAX} values`);
    const n = heights.length / tr.nx;
    if (row0 + n > tr.ny) throw new SenderError(`rows ${row0}..${row0 + n} run past ny ${tr.ny}`);
    let rows = tr.rows + n;
    for (const c of [...ctx.db.terrainChunk.terrainId.filter(terrainId)]) {
      if (c.row0 !== row0) continue;
      rows -= c.heights.length / tr.nx;
      ctx.db.terrainChunk.id.delete(c.id);
    }
    ctx.db.terrainChunk.insert({ id: 0n, terrainId, tenant: tr.tenant, row0, heights });
    ctx.db.terrain.id.update({ ...tr, rows, updatedAt: ctx.timestamp });
  }
);

export const upsertTrailNetwork = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), name: t.string(), place: t.string(), kind: t.string(), centerLat: t.f64(), centerLon: t.f64(), radiusM: t.f32(), ways: t.u32(), lengthKm: t.f32(), areas: t.u32(), sources: t.string(), meta: t.string() },
  (ctx, a) => {
    const old = ctx.db.trailNetwork.id.find(a.id);
    own(ctx, a.tenant, old, `trail network ${a.id}`);
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.trailNetwork.id.update(row);
    else ctx.db.trailNetwork.insert(row);
  }
);

const TrailIn = t.object('TrailIn', {
  osm: t.string(), name: t.string(), highway: t.string(), surface: t.string(), sacScale: t.string(), area: t.string(),
  lengthM: t.f32(), climbM: t.f32(), descentM: t.f32(), maxGradeDeg: t.f32(), path: t.array(LatLon), profileStepM: t.f32(),
  profile: t.array(t.f32()), tags: t.string(),
});

// A batch of ways for one network, keyed network/osm id so a reload replaces rather than duplicates.
export const pushTrails = spacetimedb.reducer({ networkId: t.string(), trails: t.array(TrailIn) }, (ctx, { networkId, trails }) => {
  const net = ctx.db.trailNetwork.id.find(networkId);
  if (!net) throw new SenderError('upsert the trail network first');
  requireWriter(ctx, net.tenant);
  for (const w of trails) {
    const key = `${networkId}/${w.osm}`;
    const old = ctx.db.trail.key.find(key);
    const row = { ...w, id: old?.id ?? 0n, key, network: networkId, tenant: net.tenant };
    if (old) ctx.db.trail.id.update(row);
    else ctx.db.trail.insert(row);
  }
});

const AreaIn = t.object('AreaIn', {
  osm: t.string(), name: t.string(), kind: t.string(), ways: t.u32(), lengthM: t.f32(), geometry: t.string(), stats: t.string(),
});

export const pushAreas = spacetimedb.reducer({ networkId: t.string(), areas: t.array(AreaIn) }, (ctx, { networkId, areas }) => {
  const net = ctx.db.trailNetwork.id.find(networkId);
  if (!net) throw new SenderError('upsert the trail network first');
  requireWriter(ctx, net.tenant);
  for (const w of areas) {
    const key = `${networkId}/${w.osm}`;
    const old = ctx.db.area.key.find(key);
    const row = { ...w, id: old?.id ?? 0n, key, network: networkId, tenant: net.tenant };
    if (old) ctx.db.area.id.update(row);
    else ctx.db.area.insert(row);
  }
});

const RECORDING_SOURCES = ['phone', 'game', 'video', 'gopro', 'sim'];
const RECORDING_ACTIVITIES = ['ski', 'hike', 'climb'];

// Declares a recording (or replaces its description) and clears its frames; pushRecordingChunk then fills it.
export const putRecording = spacetimedb.reducer(
  { key: t.string(), tenant: t.string(), source: t.string(), activity: t.string(), ref: t.string(), uri: t.string(), channels: t.array(t.string()), units: t.array(t.string()), rateHz: t.f32(), title: t.string(), license: t.string(), meta: t.string() },
  (ctx, a) => {
    const old = ctx.db.recording.key.find(a.key);
    own(ctx, a.tenant, old, `recording ${a.key}`);
    if (!a.key || a.key.length > 96) throw new SenderError('key is 1 to 96 characters');
    if (!RECORDING_SOURCES.includes(a.source)) throw new SenderError(`source must be one of ${RECORDING_SOURCES.join(', ')}`);
    if (!RECORDING_ACTIVITIES.includes(a.activity)) throw new SenderError(`activity must be one of ${RECORDING_ACTIVITIES.join(', ')}`);
    if (a.channels.length === 0 || a.channels.length > 512) throw new SenderError('1 to 512 channels');
    if (a.units.length !== a.channels.length) throw new SenderError('one unit per channel');
    for (const c of [...ctx.db.recordingChunk.recordingKey.filter(a.key)]) ctx.db.recordingChunk.id.delete(c.id);
    const row = { ...a, frames: 0n, chunks: 0, writer: ctx.sender, closed: false, startedAt: old?.startedAt ?? ctx.timestamp, updatedAt: ctx.timestamp };
    if (old) ctx.db.recording.key.update(row);
    else ctx.db.recording.insert(row);
  }
);

// A block of whole frames. Sending the same seq again replaces it, and the counts follow.
export const pushRecordingChunk = spacetimedb.reducer(
  { recordingKey: t.string(), seq: t.u32(), t0Ms: t.u64(), tMs: t.array(t.u32()), data: t.array(t.f32()) },
  (ctx, { recordingKey, seq, t0Ms, tMs, data }) => {
    const r = ctx.db.recording.key.find(recordingKey);
    if (!r) throw new SenderError('put the recording first');
    requireWriter(ctx, r.tenant);
    if (r.closed) throw new SenderError('recording is closed');
    const width = r.channels.length;
    if (data.length === 0 || data.length > CHUNK_F32_MAX || data.length % width !== 0) throw new SenderError(`data must be whole frames of ${width} values, at most ${CHUNK_F32_MAX}`);
    const n = data.length / width;
    if (r.rateHz <= 0 && tMs.length !== n) throw new SenderError('irregular recording: one tMs per frame');
    if (r.rateHz > 0 && tMs.length !== 0) throw new SenderError('regular recording: leave tMs empty');
    let frames = BigInt(n), chunks = 1;
    for (const old of [...ctx.db.recordingChunk.recordingKey.filter(recordingKey)]) {
      if (old.seq !== seq) continue;
      frames -= BigInt(old.data.length / width); chunks -= 1;
      ctx.db.recordingChunk.id.delete(old.id);
    }
    ctx.db.recordingChunk.insert({ id: 0n, recordingKey, tenant: r.tenant, seq, t0Ms, tMs, data });
    ctx.db.recording.key.update({ ...r, frames: r.frames + frames, chunks: r.chunks + chunks, updatedAt: ctx.timestamp });
  }
);

export const closeRecording = spacetimedb.reducer({ key: t.string() }, (ctx, { key }) => {
  const r = ctx.db.recording.key.find(key);
  if (!r) return;
  requireWriter(ctx, r.tenant);
  if (!r.closed) ctx.db.recording.key.update({ ...r, closed: true, updatedAt: ctx.timestamp });
});

const SegmentIn = t.object('SegmentIn', { key: t.string(), kind: t.string(), t0Ms: t.u64(), t1Ms: t.u64(), label: t.string(), meta: t.string() });

export const pushSegments = spacetimedb.reducer({ recordingKey: t.string(), segments: t.array(SegmentIn) }, (ctx, { recordingKey, segments }) => {
  const r = ctx.db.recording.key.find(recordingKey);
  if (!r) throw new SenderError('put the recording first');
  requireWriter(ctx, r.tenant);
  for (const sg of segments) {
    const key = `${recordingKey}/${sg.key}`;
    const old = ctx.db.segment.key.find(key);
    const row = { ...sg, id: old?.id ?? 0n, key, recordingKey, tenant: r.tenant };
    if (old) ctx.db.segment.id.update(row);
    else ctx.db.segment.insert(row);
  }
});

export const upsertReferencePose = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), activity: t.string(), phase: t.string(), robot: t.string(), joints: t.array(t.string()), values: t.array(t.f32()), frames: t.u32(), human: t.string(), sources: t.string() },
  (ctx, a) => {
    const old = ctx.db.referencePose.id.find(a.id);
    own(ctx, a.tenant, old, `reference pose ${a.id}`);
    if (a.joints.length !== a.values.length) throw new SenderError('one value per joint');
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.referencePose.id.update(row);
    else ctx.db.referencePose.insert(row);
  }
);

export const upsertTrainingSet = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), name: t.string(), activity: t.string(), robot: t.string(), description: t.string(), courses: t.array(t.string()), terrains: t.array(t.string()), recordings: t.array(t.string()), referencePoses: t.array(t.string()), meta: t.string() },
  (ctx, a) => {
    const old = ctx.db.trainingSet.id.find(a.id);
    own(ctx, a.tenant, old, `training set ${a.id}`);
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.trainingSet.id.update(row);
    else ctx.db.trainingSet.insert(row);
  }
);

export const upsertPolicy = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), name: t.string(), robot: t.string(), skill: t.string(), obsDim: t.u32(), actionDim: t.u32(), controlHz: t.f32(), runtime: t.string(), endpoint: t.string(), trainingSet: t.string(), benchmark: t.string(), spec: t.string() },
  (ctx, a) => {
    const old = ctx.db.policy.id.find(a.id);
    own(ctx, a.tenant, old, `policy ${a.id}`);
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.policy.id.update(row);
    else ctx.db.policy.insert(row);
  }
);

export const upsertTrainRun = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), policyId: t.string(), trainingSet: t.string(), hardware: t.string(), status: t.string(), iterations: t.u32(), checkpoint: t.string(), metrics: t.string(), benchmark: t.string(), notes: t.string(), startedAt: t.timestamp() },
  (ctx, a) => {
    const old = ctx.db.trainRun.id.find(a.id);
    own(ctx, a.tenant, old, `train run ${a.id}`);
    const row = { ...a, updatedAt: ctx.timestamp };
    if (old) ctx.db.trainRun.id.update(row);
    else ctx.db.trainRun.insert(row);
  }
);

// Declares a point set and clears its points; pushPoints then fills it.
export const putPointSet = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), courseId: t.string(), kind: t.string(), columns: t.array(t.string()), units: t.array(t.string()), source: t.string(), license: t.string(), meta: t.string() },
  (ctx, a) => {
    const old = ctx.db.pointSet.id.find(a.id);
    own(ctx, a.tenant, old, `point set ${a.id}`);
    if (a.columns.length === 0 || a.units.length !== a.columns.length) throw new SenderError('one unit per column');
    for (const c of [...ctx.db.pointChunk.setId.filter(a.id)]) ctx.db.pointChunk.id.delete(c.id);
    const row = { ...a, count: 0, updatedAt: ctx.timestamp };
    if (old) ctx.db.pointSet.id.update(row);
    else ctx.db.pointSet.insert(row);
  }
);

export const pushPoints = spacetimedb.reducer({ setId: t.string(), seq: t.u32(), data: t.array(t.f32()) }, (ctx, { setId, seq, data }) => {
  const ps = ctx.db.pointSet.id.find(setId);
  if (!ps) throw new SenderError('put the point set first');
  requireWriter(ctx, ps.tenant);
  const w = ps.columns.length;
  if (data.length === 0 || data.length > CHUNK_F32_MAX || data.length % w !== 0) throw new SenderError(`data must be whole points of ${w} values, at most ${CHUNK_F32_MAX}`);
  let count = ps.count + data.length / w;
  for (const c of [...ctx.db.pointChunk.setId.filter(setId)]) {
    if (c.seq !== seq) continue;
    count -= c.data.length / w;
    ctx.db.pointChunk.id.delete(c.id);
  }
  ctx.db.pointChunk.insert({ id: 0n, setId, tenant: ps.tenant, seq, data });
  ctx.db.pointSet.id.update({ ...ps, count, updatedAt: ctx.timestamp });
});

export const upsertEvaluation = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), policyId: t.string(), runId: t.string(), checkpoint: t.string(), kind: t.string(), courseId: t.string(), summary: t.string(), detail: t.string(), at: t.timestamp() },
  (ctx, a) => {
    const old = ctx.db.evaluation.id.find(a.id);
    own(ctx, a.tenant, old, `evaluation ${a.id}`);
    if (old) ctx.db.evaluation.id.update(a);
    else ctx.db.evaluation.insert(a);
  }
);

// Removes one item and everything that hangs off it (a course's terrain layers and point sets, a network's trails
// and areas, a recording's chunks and segments). Any writer of the owning organisation can do it.
export const deleteItem = spacetimedb.reducer({ kind: t.string(), id: t.string() }, (ctx, { kind, id }) => {
  const dropTerrain = (tid: string) => {
    for (const c of [...ctx.db.terrainChunk.terrainId.filter(tid)]) ctx.db.terrainChunk.id.delete(c.id);
    ctx.db.terrain.id.delete(tid);
  };
  const dropPoints = (sid: string) => {
    for (const c of [...ctx.db.pointChunk.setId.filter(sid)]) ctx.db.pointChunk.id.delete(c.id);
    ctx.db.pointSet.id.delete(sid);
  };
  const check = (row: { tenant: string } | null) => {
    if (!row) throw new SenderError(`no ${kind} ${id}`);
    requireWriter(ctx, row.tenant);
  };
  switch (kind) {
    case 'resort': check(ctx.db.resort.id.find(id)); ctx.db.resort.id.delete(id); break;
    case 'course':
      check(ctx.db.course.id.find(id));
      for (const tr of [...ctx.db.terrain.courseId.filter(id)]) dropTerrain(tr.id);
      for (const ps of [...ctx.db.pointSet.courseId.filter(id)]) dropPoints(ps.id);
      ctx.db.course.id.delete(id);
      break;
    case 'terrain': check(ctx.db.terrain.id.find(id)); dropTerrain(id); break;
    case 'point_set': check(ctx.db.pointSet.id.find(id)); dropPoints(id); break;
    case 'trail_network':
      check(ctx.db.trailNetwork.id.find(id));
      for (const w of [...ctx.db.trail.network.filter(id)]) ctx.db.trail.id.delete(w.id);
      for (const w of [...ctx.db.area.network.filter(id)]) ctx.db.area.id.delete(w.id);
      ctx.db.trailNetwork.id.delete(id);
      break;
    case 'recording': check(ctx.db.recording.key.find(id)); dropRecording(ctx, id); break;
    case 'reference_pose': check(ctx.db.referencePose.id.find(id)); ctx.db.referencePose.id.delete(id); break;
    case 'training_set': check(ctx.db.trainingSet.id.find(id)); ctx.db.trainingSet.id.delete(id); break;
    case 'policy': check(ctx.db.policy.id.find(id)); ctx.db.policy.id.delete(id); break;
    case 'train_run': check(ctx.db.trainRun.id.find(id)); ctx.db.trainRun.id.delete(id); break;
    case 'evaluation': check(ctx.db.evaluation.id.find(id)); ctx.db.evaluation.id.delete(id); break;
    default: throw new SenderError(`cannot delete a ${kind}`);
  }
});

// ------------------------------------------------------------------------------------------------ views
// public_<table> serves the rows of public tenants, the same for every caller. my_<table> serves the rows of every
// organisation the caller belongs to, whatever their role. Both are semijoins on indexed columns (tenant.id or
// member.tenant_id against the row's tenant), so SpacetimeDB keeps them up to date incrementally instead of rerunning
// them over every chunk when one row changes.

export const publicResortView = spacetimedb.anonymousView({ name: 'public_resort', public: true }, t.array(resort.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.resort, (tn, r) => tn.id.eq(r.tenant))
);

export const myResortView = spacetimedb.view({ name: 'my_resort', public: true }, t.array(resort.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.resort, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicCourseView = spacetimedb.anonymousView({ name: 'public_course', public: true }, t.array(course.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.course, (tn, r) => tn.id.eq(r.tenant))
);

export const myCourseView = spacetimedb.view({ name: 'my_course', public: true }, t.array(course.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.course, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicTerrainView = spacetimedb.anonymousView({ name: 'public_terrain', public: true }, t.array(terrain.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.terrain, (tn, r) => tn.id.eq(r.tenant))
);

export const myTerrainView = spacetimedb.view({ name: 'my_terrain', public: true }, t.array(terrain.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.terrain, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicTerrainChunkView = spacetimedb.anonymousView({ name: 'public_terrain_chunk', public: true }, t.array(terrainChunk.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.terrainChunk, (tn, r) => tn.id.eq(r.tenant))
);

export const myTerrainChunkView = spacetimedb.view({ name: 'my_terrain_chunk', public: true }, t.array(terrainChunk.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.terrainChunk, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicTrailNetworkView = spacetimedb.anonymousView({ name: 'public_trail_network', public: true }, t.array(trailNetwork.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.trailNetwork, (tn, r) => tn.id.eq(r.tenant))
);

export const myTrailNetworkView = spacetimedb.view({ name: 'my_trail_network', public: true }, t.array(trailNetwork.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.trailNetwork, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicTrailView = spacetimedb.anonymousView({ name: 'public_trail', public: true }, t.array(trail.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.trail, (tn, r) => tn.id.eq(r.tenant))
);

export const myTrailView = spacetimedb.view({ name: 'my_trail', public: true }, t.array(trail.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.trail, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicAreaView = spacetimedb.anonymousView({ name: 'public_area', public: true }, t.array(area.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.area, (tn, r) => tn.id.eq(r.tenant))
);

export const myAreaView = spacetimedb.view({ name: 'my_area', public: true }, t.array(area.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.area, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicRecordingView = spacetimedb.anonymousView({ name: 'public_recording', public: true }, t.array(recording.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.recording, (tn, r) => tn.id.eq(r.tenant))
);

export const myRecordingView = spacetimedb.view({ name: 'my_recording', public: true }, t.array(recording.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.recording, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicRecordingChunkView = spacetimedb.anonymousView({ name: 'public_recording_chunk', public: true }, t.array(recordingChunk.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.recordingChunk, (tn, r) => tn.id.eq(r.tenant))
);

export const myRecordingChunkView = spacetimedb.view({ name: 'my_recording_chunk', public: true }, t.array(recordingChunk.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.recordingChunk, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicSegmentView = spacetimedb.anonymousView({ name: 'public_segment', public: true }, t.array(segment.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.segment, (tn, r) => tn.id.eq(r.tenant))
);

export const mySegmentView = spacetimedb.view({ name: 'my_segment', public: true }, t.array(segment.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.segment, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicReferencePoseView = spacetimedb.anonymousView({ name: 'public_reference_pose', public: true }, t.array(referencePose.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.referencePose, (tn, r) => tn.id.eq(r.tenant))
);

export const myReferencePoseView = spacetimedb.view({ name: 'my_reference_pose', public: true }, t.array(referencePose.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.referencePose, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicTrainingSetView = spacetimedb.anonymousView({ name: 'public_training_set', public: true }, t.array(trainingSet.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.trainingSet, (tn, r) => tn.id.eq(r.tenant))
);

export const myTrainingSetView = spacetimedb.view({ name: 'my_training_set', public: true }, t.array(trainingSet.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.trainingSet, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicPolicyView = spacetimedb.anonymousView({ name: 'public_policy', public: true }, t.array(policy.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.policy, (tn, r) => tn.id.eq(r.tenant))
);

export const myPolicyView = spacetimedb.view({ name: 'my_policy', public: true }, t.array(policy.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.policy, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicTrainRunView = spacetimedb.anonymousView({ name: 'public_train_run', public: true }, t.array(trainRun.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.trainRun, (tn, r) => tn.id.eq(r.tenant))
);

export const myTrainRunView = spacetimedb.view({ name: 'my_train_run', public: true }, t.array(trainRun.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.trainRun, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicPointSetView = spacetimedb.anonymousView({ name: 'public_point_set', public: true }, t.array(pointSet.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.pointSet, (tn, r) => tn.id.eq(r.tenant))
);

export const myPointSetView = spacetimedb.view({ name: 'my_point_set', public: true }, t.array(pointSet.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.pointSet, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicPointChunkView = spacetimedb.anonymousView({ name: 'public_point_chunk', public: true }, t.array(pointChunk.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.pointChunk, (tn, r) => tn.id.eq(r.tenant))
);

export const myPointChunkView = spacetimedb.view({ name: 'my_point_chunk', public: true }, t.array(pointChunk.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.pointChunk, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicEvaluationView = spacetimedb.anonymousView({ name: 'public_evaluation', public: true }, t.array(evaluation.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.evaluation, (tn, r) => tn.id.eq(r.tenant))
);

export const myEvaluationView = spacetimedb.view({ name: 'my_evaluation', public: true }, t.array(evaluation.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.evaluation, (m, r) => m.tenantId.eq(r.tenant))
);

function dropRecording(ctx: Ctx, key: string) {
  for (const c of [...ctx.db.recordingChunk.recordingKey.filter(key)]) ctx.db.recordingChunk.id.delete(c.id);
  for (const sg of [...ctx.db.segment.recordingKey.filter(key)]) ctx.db.segment.id.delete(sg.id);
  ctx.db.recording.key.delete(key);
}

// ---------------------------------------------------------------------------------------------- app users
// The phone server and the agent write these as services of the app's tenant. Times they already know (when someone
// first texted, when a message went out) come in as unix milliseconds; 0 means now.

const APP_ROLES = ['owner', 'admin', 'service'];
const ZERO_TIME = new Timestamp(0n);

function atMs(ctx: Ctx, ms: bigint) {
  return ms > 0n ? new Timestamp(ms * 1000n) : ctx.timestamp;
}

// A person the server or agent mentions before the agent has synced them gets a bare row, so every message, invite
// and session has someone to belong to.
function ensureAppUser(ctx: Ctx, tenantId: string, handle: string, code: string) {
  const h = handle.trim();
  if (!h) return;
  const c = code.trim().toUpperCase();
  const u = ctx.db.appUser.handle.find(h);
  if (u) {
    if (u.tenant === tenantId && c && !u.code) ctx.db.appUser.handle.update({ ...u, code: c });
    return;
  }
  ctx.db.appUser.insert({ handle: h, tenant: tenantId, code: c, name: '', optedOut: false, pendingLabel: '', pendingDelete: false, invites: 0, notes: [], consentVersion: '', consentedAt: ZERO_TIME, firstSeen: ctx.timestamp, lastSeen: ctx.timestamp });
}

export const upsertAppUser = spacetimedb.reducer(
  { handle: t.string(), tenant: t.string(), code: t.string(), name: t.string(), optedOut: t.bool(), pendingLabel: t.string(), pendingDelete: t.bool(), invites: t.u32(), notes: t.array(t.string()), firstSeenMs: t.u64() },
  (ctx, a) => {
    requireRole(ctx, a.tenant, APP_ROLES);
    const handle = a.handle.trim();
    if (!handle) throw new SenderError('handle required');
    const old = ctx.db.appUser.handle.find(handle);
    if (old && old.tenant !== a.tenant) throw new SenderError(`${handle} belongs to ${old.tenant}`);
    const first = atMs(ctx, a.firstSeenMs);
    const row = {
      handle, tenant: a.tenant, code: a.code.trim().toUpperCase(), name: a.name.trim().slice(0, 80), optedOut: a.optedOut,
      pendingLabel: a.pendingLabel, pendingDelete: a.pendingDelete, invites: a.invites, notes: a.notes.slice(-200),
      consentVersion: old?.consentVersion ?? '', consentedAt: old?.consentedAt ?? ZERO_TIME,
      firstSeen: old && old.firstSeen.microsSinceUnixEpoch < first.microsSinceUnixEpoch ? old.firstSeen : first, lastSeen: ctx.timestamp,
    };
    if (old) ctx.db.appUser.handle.update(row);
    else ctx.db.appUser.insert(row);
  }
);

// One text. Sending the same key again updates it (a pending message that went out, or failed).
export const upsertAppMessage = spacetimedb.reducer(
  { key: t.string(), handle: t.string(), tenant: t.string(), direction: t.string(), kind: t.string(), text: t.string(), session: t.string(), status: t.string(), error: t.string(), atMs: t.u64() },
  (ctx, a) => {
    requireRole(ctx, a.tenant, APP_ROLES);
    if (!a.key || a.key.length > 96) throw new SenderError('key is 1 to 96 characters');
    if (a.direction !== 'in' && a.direction !== 'out') throw new SenderError('direction is in or out');
    const old = ctx.db.appMessage.key.find(a.key);
    if (old && old.tenant !== a.tenant) throw new SenderError(`message ${a.key} belongs to ${old.tenant}`);
    ensureAppUser(ctx, a.tenant, a.handle, '');
    const { atMs: ms, ...fields } = a;
    const row = { ...fields, id: old?.id ?? 0n, text: a.text.slice(0, 8000), error: a.error.slice(0, 500), at: old?.at ?? atMs(ctx, ms) };
    if (old) ctx.db.appMessage.id.update(row);
    else ctx.db.appMessage.insert(row);
  }
);

export const upsertAppInvite = spacetimedb.reducer(
  { token: t.string(), tenant: t.string(), handle: t.string(), code: t.string(), link: t.string(), createdAtMs: t.u64() },
  (ctx, a) => {
    requireRole(ctx, a.tenant, APP_ROLES);
    if (!a.token) throw new SenderError('token required');
    const old = ctx.db.appInvite.token.find(a.token);
    if (old && old.tenant !== a.tenant) throw new SenderError('invite belongs to another organisation');
    ensureAppUser(ctx, a.tenant, a.handle, a.code);
    const row = { token: a.token, tenant: a.tenant, handle: a.handle.trim(), code: a.code.trim().toUpperCase(), link: a.link, createdAt: old?.createdAt ?? atMs(ctx, a.createdAtMs) };
    if (old) ctx.db.appInvite.token.update(row);
    else ctx.db.appInvite.insert(row);
  }
);

// A session as the phone server sees it. Consent with a version also stamps the person's consent.
export const upsertAppSession = spacetimedb.reducer(
  { id: t.string(), tenant: t.string(), handle: t.string(), code: t.string(), token: t.string(), status: t.string(), consent: t.string(), meta: t.string(), summary: t.string(), labels: t.array(t.string()), error: t.string(), createdAtMs: t.u64() },
  (ctx, a) => {
    requireRole(ctx, a.tenant, APP_ROLES);
    if (!a.id) throw new SenderError('id required');
    const old = ctx.db.appSession.id.find(a.id);
    if (old && old.tenant !== a.tenant) throw new SenderError(`session ${a.id} belongs to ${old.tenant}`);
    ensureAppUser(ctx, a.tenant, a.handle, a.code);
    const { createdAtMs, ...fields } = a;
    const row = { ...fields, handle: a.handle.trim(), code: a.code.trim().toUpperCase(), labels: a.labels.slice(-50), error: a.error.slice(0, 2000), createdAt: old?.createdAt ?? atMs(ctx, createdAtMs), updatedAt: ctx.timestamp };
    if (old) ctx.db.appSession.id.update(row);
    else ctx.db.appSession.insert(row);
    let version = '';
    try { version = String(JSON.parse(a.consent || '{}').consent_version ?? ''); } catch { version = ''; }
    const u = ctx.db.appUser.handle.find(row.handle);
    if (version && u && u.tenant === a.tenant && u.consentVersion !== version) {
      ctx.db.appUser.handle.update({ ...u, consentVersion: version, consentedAt: ctx.timestamp });
    }
  }
);

// "delete my data": the person, every message, invite and session, the raw phone recordings and their hikes, in one
// transaction, so either all of it goes or none does.
export const deleteAppUser = spacetimedb.reducer({ tenant: t.string(), handle: t.string() }, (ctx, { tenant: tenantId, handle }) => {
  requireRole(ctx, tenantId, APP_ROLES);
  const h = handle.trim();
  const u = ctx.db.appUser.handle.find(h);
  if (u && u.tenant !== tenantId) throw new SenderError(`${h} belongs to ${u.tenant}`);
  const codes = new Set<string>(u?.code ? [u.code] : []);
  for (const se of [...ctx.db.appSession.handle.filter(h)].filter(x => x.tenant === tenantId)) {
    for (const kind of ['imu', 'gps', 'baro']) if (ctx.db.recording.key.find(`phone/${se.id}/${kind}`)) dropRecording(ctx, `phone/${se.id}/${kind}`);
    const hk = ctx.db.hike.key.find(se.id);
    if (hk && hk.joinCode) codes.add(hk.joinCode);
    else if (hk) {
      for (const c of [...ctx.db.hikeChunk.hikeKey.filter(hk.key)]) ctx.db.hikeChunk.id.delete(c.id);
      const st = hikeTotals(ctx);
      ctx.db.hikeStats.id.update({ ...st, hikes: st.hikes - 1n, meters: Math.max(0, st.meters - hk.distanceM), seconds: Math.max(0, st.seconds - (hk.endedS - hk.startedS)), motionSamples: st.motionSamples - BigInt(hk.motionSamples) });
      ctx.db.hike.id.delete(hk.id);
    }
    if (se.code) codes.add(se.code);
    ctx.db.appSession.id.delete(se.id);
  }
  for (const inv of [...ctx.db.appInvite.handle.filter(h)].filter(x => x.tenant === tenantId)) {
    if (inv.code) codes.add(inv.code);
    ctx.db.appInvite.token.delete(inv.token);
  }
  for (const m of [...ctx.db.appMessage.handle.filter(h)].filter(x => x.tenant === tenantId)) ctx.db.appMessage.id.delete(m.id);
  for (const c of codes) dropHikes(ctx, c);
  if (u) ctx.db.appUser.handle.delete(h);
});

// ---------------------------------------------------------------------------------------- platform and API

function isPlatformAdmin(ctx: Ctx) {
  const a = ctx.db.platformAdmin.id.find(0);
  return !!a && a.admin.equals(ctx.sender);
}

export const addPlatformService = spacetimedb.reducer({ identity: t.identity(), name: t.string(), kind: t.string() }, (ctx, a) => {
  if (!isPlatformAdmin(ctx)) throw new SenderError('platform admin only');
  const row = { identity: a.identity, name: a.name.slice(0, 80), kind: a.kind, addedAt: ctx.timestamp };
  if (ctx.db.platformService.identity.find(a.identity)) ctx.db.platformService.identity.update(row);
  else ctx.db.platformService.insert(row);
});

export const removePlatformService = spacetimedb.reducer({ identity: t.identity() }, (ctx, { identity }) => {
  if (!isPlatformAdmin(ctx)) throw new SenderError('platform admin only');
  ctx.db.platformService.identity.delete(identity);
});

// Adds an identity (a server, the agent) to an organisation directly, without an invite code.
export const addMemberIdentity = spacetimedb.reducer({ tenantId: t.string(), identity: t.identity(), role: t.string() }, (ctx, { tenantId, identity, role }) => {
  const me = requireRole(ctx, tenantId, MANAGE);
  if (!ROLES.includes(role)) throw new SenderError(`role must be one of ${ROLES.join(', ')}`);
  if ((role === 'owner' || role === 'admin') && me.role !== 'owner') throw new SenderError('only an owner grants owner or admin');
  addMember(ctx, tenantId, identity, role);
});

const HEX64 = /^[0-9a-f]{64}$/;

// The key is made and shown on the caller's machine (tools/api_key.py); only its SHA-256 comes here.
export const addApiKey = spacetimedb.reducer({ tenantId: t.string(), hash: t.string(), prefix: t.string(), label: t.string() }, (ctx, a) => {
  requireRole(ctx, a.tenantId, MANAGE);
  if (!HEX64.test(a.hash)) throw new SenderError('hash is the lowercase hex SHA-256 of the key');
  if (ctx.db.apiKey.hash.find(a.hash)) throw new SenderError('that key is already registered');
  ctx.db.apiKey.insert({ hash: a.hash, tenant: a.tenantId, prefix: a.prefix.slice(0, 12), label: a.label.slice(0, 80), createdBy: ctx.sender, createdAt: ctx.timestamp, lastUsedAt: ZERO_TIME, calls: 0n, revoked: false });
});

export const revokeApiKey = spacetimedb.reducer({ hash: t.string() }, (ctx, { hash }) => {
  const k = ctx.db.apiKey.hash.find(hash);
  if (!k) throw new SenderError('no such key');
  requireRole(ctx, k.tenant, MANAGE);
  ctx.db.apiKey.hash.update({ ...k, revoked: true });
});

const ApiCallIn = t.object('ApiCallIn', { keyHash: t.string(), policyId: t.string(), route: t.string(), status: t.u16(), ms: t.f32(), atMs: t.u64() });

// The policy API sends its calls in batches. A call with a key counts for the key's organisation, one without for the
// public tenant.
export const logApiCalls = spacetimedb.reducer({ calls: t.array(ApiCallIn) }, (ctx, { calls }) => {
  if (!ctx.db.platformService.identity.find(ctx.sender) && !isPlatformAdmin(ctx)) throw new SenderError('platform services only');
  if (calls.length > 1000) throw new SenderError('at most 1000 calls per batch');
  for (const c of calls) {
    const at = atMs(ctx, c.atMs);
    const k = c.keyHash ? ctx.db.apiKey.hash.find(c.keyHash) : null;
    if (k) ctx.db.apiKey.hash.update({ ...k, calls: k.calls + 1n, lastUsedAt: k.lastUsedAt.microsSinceUnixEpoch > at.microsSinceUnixEpoch ? k.lastUsedAt : at });
    ctx.db.apiCall.insert({ id: 0n, tenant: k?.tenant ?? HI, keyPrefix: k?.prefix ?? '', policyId: c.policyId.slice(0, 64), route: c.route.slice(0, 32), status: c.status, ms: c.ms, at });
  }
});

// ------------------------------------------------------------------------------------ app and API views

function appTenantIds(ctx: VCtx) {
  return [...ctx.db.member.identity.filter(ctx.sender)].filter(m => APP_ROLES.includes(m.role)).map(m => m.tenantId);
}

export const myAppUserView = spacetimedb.view({ name: 'my_app_user', public: true }, t.array(appUser.rowType), ctx => {
  return appTenantIds(ctx).flatMap(id => [...ctx.db.appUser.tenant.filter(id)]);
});

export const myAppMessageView = spacetimedb.view({ name: 'my_app_message', public: true }, t.array(appMessage.rowType), ctx => {
  return appTenantIds(ctx).flatMap(id => [...ctx.db.appMessage.tenant.filter(id)]);
});

export const myAppInviteView = spacetimedb.view({ name: 'my_app_invite', public: true }, t.array(appInvite.rowType), ctx => {
  return appTenantIds(ctx).flatMap(id => [...ctx.db.appInvite.tenant.filter(id)]);
});

export const myAppSessionView = spacetimedb.view({ name: 'my_app_session', public: true }, t.array(appSession.rowType), ctx => {
  return appTenantIds(ctx).flatMap(id => [...ctx.db.appSession.tenant.filter(id)]);
});

// Owners and admins see their organisations' keys (hashes and prefixes only).
export const myApiKeyView = spacetimedb.view({ name: 'my_api_key', public: true }, t.array(apiKey.rowType), ctx => {
  return managed(ctx).flatMap(id => [...ctx.db.apiKey.tenant.filter(id)]);
});

export const myApiCallView = spacetimedb.view({ name: 'my_api_call', public: true }, t.array(apiCall.rowType), ctx =>
  ctx.from.member.where(m => m.identity.eq(ctx.sender)).rightSemijoin(ctx.from.apiCall, (m, r) => m.tenantId.eq(r.tenant))
);

export const publicApiCallView = spacetimedb.anonymousView({ name: 'public_api_call', public: true }, t.array(apiCall.rowType), ctx =>
  ctx.from.tenant.where(tn => tn.visibility.eq('public')).rightSemijoin(ctx.from.apiCall, (tn, r) => tn.id.eq(r.tenant))
);

// The policy API checks keys through this: platform services see every key's hash, organisation and state.
export const serviceApiKeyView = spacetimedb.view({ name: 'service_api_key', public: true }, t.array(apiKey.rowType), ctx => {
  return ctx.db.platformService.identity.find(ctx.sender) ? [...ctx.db.apiKey.iter()] : [];
});
