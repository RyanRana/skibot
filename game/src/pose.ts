// Your body as the controller. MediaPipe Pose Landmarker on the webcam, turned into three signals a
// skier understands: lean (left/right), crouch (0..1) and a hop. Works in tiers, from whatever the
// camera can see: head and shoulders alone, shoulders and hips, or the whole body.
import { FilesetResolver, PoseLandmarker, type NormalizedLandmark, type Landmark } from '@mediapipe/tasks-vision';
import { LM, EDGES } from './landmarks.ts';
export { LM, EDGES };

const SHOULDERS = [LM.lsh, LM.rsh];
const HIPS = [LM.lhip, LM.rhip];
const LEGS = [LM.lkn, LM.rkn, LM.lan, LM.ran];

export type Tier = 'none' | 'shoulders' | 'torso' | 'legs';

export interface BodySignals {
  present: boolean;       // at least the shoulders are tracked
  tier: Tier;
  legs: boolean;          // knees and ankles tracked too
  calibrated: boolean;
  lean: number;           // -1..1, + = skier's right
  rawLean: number;        // before the range scaling and curve (the tutorial reads this)
  crouch: number;         // 0..1
  jump: boolean;          // true for one frame
  handsUp: boolean;       // both arms raised
  armL: boolean; armR: boolean;
  armsVisible: boolean;   // elbows and wrists tracked, so the robot can copy the arms
  pole: number;           // pole plants this frame: 0, 1 (one arm) or 2 (double pole)
  handsTogether: boolean;
  kneeL: number;          // radians of flexion (0 straight), 0 when legs are not visible
  kneeR: number;
  torsoPitch: number;
  world?: Landmark[];
  image?: NormalizedLandmark[];
  fps: number;
  debug: string;
}

const params = new URLSearchParams(location.search);
const MODEL_KIND = params.get('pose') === 'lite' ? 'lite' : params.get('pose') === 'heavy' ? 'heavy' : 'full';
const MODEL = `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${MODEL_KIND}/float16/latest/pose_landmarker_${MODEL_KIND}.task`;
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const CAM_KEY = 'gt-camera';

const ang = (a: Landmark, b: Landmark, c: Landmark) => {
  const ux = a.x - b.x, uy = a.y - b.y, uz = a.z - b.z, vx = c.x - b.x, vy = c.y - b.y, vz = c.z - b.z;
  const d = (ux * vx + uy * vy + uz * vz) / (Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz) + 1e-9);
  return Math.acos(Math.max(-1, Math.min(1, d)));
};
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
// The learned lean range. Recorded players lean hard (raw 0.5-0.9 in turns); a cap of 0.45 put them at full
// lock a quarter of the time, so steering felt on/off. Full lock now sits at 80 % of your own biggest lean.
const LEAN_MIN = 0.12, LEAN_MAX = 0.8;

/** One Euro filter (Casiez et al. 2012): low jitter at rest, low lag in motion. Timestamps in ms. */
export class OneEuro {
  x: number | null = null; dx = 0; t = 0;
  constructor(public minCutoff = 1, public beta = 0, public dCutoff = 1) {}
  static a(cutoff: number, dt: number) { return 1 / (1 + 1 / (2 * Math.PI * cutoff * dt)); }
  filter(x: number, tMs: number) {
    if (this.x === null) { this.x = x; this.t = tMs; return x; }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000); this.t = tMs;
    this.dx += OneEuro.a(this.dCutoff, dt) * ((x - this.x) / dt - this.dx);
    this.x += OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (x - this.x);
    return this.x;
  }
  reset() { this.x = null; this.dx = 0; }
}
/** Expo response: flat near zero, full at the edges, no jump on leaving a dead zone. */
const expo = (x: number) => Math.sign(x) * (0.6 * Math.abs(x) ** 3 + 0.4 * Math.abs(x));

/** Cameras the browser can see (labels appear after the first permission grant). */
export async function listCameras(): Promise<{ id: string; label: string }[]> {
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    return devs.filter(d => d.kind === 'videoinput').map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` }));
  } catch { return []; }
}
export function preferredCamera(): string { return params.get('camera') || (() => { try { return localStorage.getItem(CAM_KEY) || ''; } catch { return ''; } })(); }
export function rememberCamera(id: string) { try { localStorage.setItem(CAM_KEY, id); } catch {} }

export class BodyTracker {
  video: HTMLVideoElement;
  landmarker: PoseLandmarker | null = null;
  signals: BodySignals = { present: false, tier: 'none', legs: false, calibrated: false, lean: 0, rawLean: 0, crouch: 0, jump: false, handsUp: false, armL: false, armR: false, armsVisible: false, pole: 0, handsTogether: false, kneeL: 0, kneeR: 0, torsoPitch: 0, fps: 0, debug: '' };
  status = 'camera off';
  delegate = 'GPU';
  cameraLabel = '';
  private stream: MediaStream | null = null;
  // calibration: a neutral standing pose in distance-free units (everything over shoulder width)
  private cal = { shY: 0, midY: 0, torso: 0, legspan: 0, tier: 'none' as Tier, n: 0, acc: [0, 0, 0, 0], stableSince: 0 };
  private leanFilt = new OneEuro(1.2, 4, 1); private crouchFilt = new OneEuro(1.0, 2, 1);
  private leanPeak = 0.3;          // the player's own comfortable lean, learned (in lean units before the curve)
  private lastPeakT = 0;
  private scale = 0;               // slow distance scale (image units), immune to leans and turns
  private standSh = 0;             // standing shoulder height in scale units (baseline tracker)
  private standHip = 0;            // standing hip height in scale units
  private lastY = 0; private lastYT = 0; private riseV = 0;
  private turnedState = false;
  private tierCand: Tier = 'none'; private tierDwell = 0;
  private lastSh = { x: 0, y: 0, w: 0 }; private held = 0;
  private lumaCanvas: HTMLCanvasElement | null = null; private luma = 128; private lumaFrame = 0;
  /** Telemetry to the dev console (forwarded to the Vite terminal) for debugging moves. */
  telemetry = new URLSearchParams(location.search).has('telemetry') || import.meta.env.DEV;
  private lastTele = 0; private lastRise = 0;
  private lastJump = 0; private lastHopDbg = 0; private vetoApex = -1;
  /** Set by the game: during the countdown and the race the standing baselines hold through a tuck; before
   *  that they simply follow the player (who walks about, types a name, raises their arms). */
  raceMode = false;
  private hopHist: { t: number; c: number; h: number; tl: number; wl: number; wr: number; up: boolean }[] = [];
  private wristHist: { t: number; l: number; r: number }[] = [];
  private lastPole = [0, 0];
  private lastDouble = 0;
  private frames = 0; private fpsT = 0;
  private lastVideoT = -1;

  constructor(video: HTMLVideoElement) { this.video = video; }

  async start(deviceId = preferredCamera()) {
    this.status = 'starting camera';
    if (!navigator.mediaDevices?.getUserMedia) { this.status = 'camera needs localhost or https'; throw new Error('getUserMedia unavailable (open the game at http://localhost:5173, not the IP)'); }
    await this.openCamera(deviceId);
    if (!this.landmarker) {
      this.status = `loading pose model (${MODEL_KIND})`;
      const fileset = await FilesetResolver.forVisionTasks(WASM);
      const opts = { runningMode: 'VIDEO' as const, numPoses: 1, minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4, minTrackingConfidence: 0.4 };
      try {
        this.landmarker = await PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' }, ...opts });
      } catch (e) {
        console.warn('pose: GPU delegate failed, using CPU', e);
        this.delegate = 'CPU';
        this.landmarker = await PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: MODEL, delegate: 'CPU' }, ...opts });
      }
    }
    this.status = 'step into frame';
  }

  /** Switch cameras (by deviceId or a substring of the label). */
  async openCamera(pick = '') {
    let deviceId = '';
    if (pick) { const cams = await listCameras(); const hit = cams.find(c => c.id === pick) ?? cams.find(c => c.label.toLowerCase().includes(pick.toLowerCase())); deviceId = hit ? hit.id : ''; }
    const video: MediaTrackConstraints = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
    if (deviceId) video.deviceId = { exact: deviceId }; else video.facingMode = 'user';
    const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false }).catch(async e => { if (deviceId) return navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false }); throw e; });
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = stream;
    this.cameraLabel = stream.getVideoTracks()[0]?.label ?? '';
    this.video.srcObject = stream;
    await this.video.play();
    this.lastVideoT = -1;
    this.recalibrate();
  }

  recalibrate() { this.cal = { shY: 0, midY: 0, torso: 0, legspan: 0, tier: 'none', n: 0, acc: [0, 0, 0, 0], stableSince: 0 }; this.signals.calibrated = false; this.leanFilt.reset(); this.crouchFilt.reset(); this.standSh = this.standHip = 0; this.scale = 0; }
  /** Set the full-lean range from a measured comfortable lean (the tutorial does this). */
  learnLeanPeak(absRaw: number) { this.leanPeak = clamp(absRaw * 0.8, LEAN_MIN, LEAN_MAX); this.lastPeakT = this.video.currentTime * 1000; }
  /** The learned lean range (debug / tutorial display). */
  get leanRange() { return this.leanPeak; }

  /** Call every animation frame. Returns the latest signals (jump is edge-triggered). */
  update(now: number): BodySignals {
    const s = this.signals;
    s.jump = false; s.pole = 0;
    if (!this.landmarker || this.video.readyState < 2) return s;
    if (this.video.currentTime === this.lastVideoT) return s;
    this.lastVideoT = this.video.currentTime;
    let res;
    try { res = this.landmarker.detectForVideo(this.video, now); } catch (e) { console.warn('pose detect', e); return s; }
    this.frames++;
    if (now - this.fpsT > 1000) { s.fps = this.frames; this.frames = 0; this.fpsT = now; }
    const lm = res.landmarks[0], w = res.worldLandmarks[0];
    if (!lm || !w) { this.decay(); this.status = 'no one in frame'; return s; }
    s.image = lm; s.world = w;
    const vis = (ids: readonly number[]) => Math.min(...ids.map(i => lm[i].visibility ?? 1));
    const inFrame = (ids: readonly number[]) => ids.every(i => lm[i].y > -0.05 && lm[i].y < 1.05 && lm[i].x > -0.05 && lm[i].x < 1.05);
    const shVis = vis(SHOULDERS), hipVis = vis(HIPS), legVis = vis(LEGS);
    // Tier with hysteresis and a short dwell: MediaPipe low-passes visibility (~300 ms), so it hovers.
    const enter = 0.5, exit = 0.22;
    const cur = s.tier;
    const wantTier: Tier = shVis < (cur === 'none' ? enter : exit) || !inFrame(SHOULDERS) ? 'none'
      : (hipVis < (cur === 'shoulders' || cur === 'none' ? enter : exit) || !inFrame(HIPS)) ? 'shoulders'
      : (legVis < (cur === 'legs' ? exit : enter) || !inFrame(LEGS)) ? 'torso' : 'legs';
    if (wantTier !== this.tierCand) { this.tierCand = wantTier; this.tierDwell = 0; }
    const tier: Tier = (++this.tierDwell >= 12 || cur === 'none' || wantTier === 'none') ? wantTier : cur;
    if (tier === 'none') { this.decay(); this.status = 'step into frame and face the camera'; return s; }

    const sh = mid(lm[LM.lsh], lm[LM.rsh]), hip = mid(lm[LM.lhip], lm[LM.rhip]), nose = lm[LM.nose];
    const shoulderW = Math.hypot(lm[LM.lsh].x - lm[LM.rsh].x, lm[LM.lsh].y - lm[LM.rsh].y) + 1e-6;
    const torso = tier !== 'shoulders' ? Math.hypot(sh.x - hip.x, sh.y - hip.y) : shoulderW * 1.6;
    const tMs = this.video.currentTime * 1000;

    // Reject implausible jumps (the detector re-centred after losing track): hold the last values for up to 3 frames.
    const L = this.lastSh;
    const jumped = L.w > 0 && (Math.hypot(sh.x - L.x, sh.y - L.y) > 0.3 * torso || Math.abs(shoulderW - L.w) / L.w > 0.25);
    if (jumped && this.held < 3) { this.held++; return s; }
    this.held = 0; this.lastSh = { x: sh.x, y: sh.y, w: shoulderW };
    s.present = true; s.tier = tier; s.legs = tier === 'legs';

    // A slow distance scale: leans and turns change the apparent shoulder width within a frame, walking
    // changes the distance over seconds. Vertical signals are measured against this, never raw width.
    const scaleRaw = tier === 'shoulders' ? shoulderW * 1.6 : Math.max(torso, shoulderW * 1.2);
    const dtS = this.lastYT ? Math.min(0.1, (tMs - this.lastYT) / 1000) : 0.016;
    if (!this.scale) this.scale = scaleRaw; else this.scale += (scaleRaw - this.scale) * (1 - Math.exp(-dtS / 1.5));
    const scale = this.scale;
    // Facing away: the shoulders foreshorten against the slow scale. 2D only (depth from MediaPipe is not reliable).
    const ratio = shoulderW / (scale / 1.6);
    // (a hard sideways lean also foreshortens the shoulders, to ~0.55, so the bar for "turned away" sits below it)
    this.turnedState = this.turnedState ? ratio < 0.6 : ratio < 0.45;
    const turned = this.turnedState;

    // Lean. Raw frame: the subject's right is at smaller x. Tilting right drops the right shoulder.
    const roll = Math.atan2(lm[LM.rsh].y - lm[LM.lsh].y, lm[LM.lsh].x - lm[LM.rsh].x);
    const leanRoll = Math.sin(roll) * 1.3;                    // shoulder line tilt
    const leanHead = (sh.x - nose.x) / shoulderW * 0.5;       // head carried to the side
    let raw: number;
    if (tier === 'shoulders') raw = 0.8 * leanRoll + 0.6 * leanHead;
    else {
      const leanTorso = (hip.x - sh.x) / torso;                // sin of the trunk's lateral flexion, yaw-stable
      raw = 1.0 * leanTorso + 0.35 * leanRoll + 0.2 * leanHead;
      if (tier === 'legs') { const an = mid(lm[LM.lan], lm[LM.ran]); raw += 0.25 * (an.x - hip.x) / torso; }
    }
    s.rawLean = raw;
    if (!turned) {
      // The player's own range: a peak that decays with a 30 s half-life, so a big lean is full lock and
      // a timid player still steers. The tutorial's lean steps set it directly.
      const decay = Math.pow(0.5, dtS / 30);
      if (Math.abs(raw) > 0.05) this.leanPeak = Math.max(this.leanPeak * decay, Math.abs(raw) * 0.8);
      this.leanPeak = clamp(this.leanPeak, LEAN_MIN, LEAN_MAX);
      let lean = clamp(raw / this.leanPeak, -1, 1);
      lean = this.leanFilt.filter(lean, tMs);
      lean += 0.5 * this.leanFilt.dx * 0.033;                 // a frame of prediction, capped
      s.lean = Math.abs(lean) < 0.03 ? 0 : expo(clamp(lean, -1, 1));
    } // turned: hold the last lean

    // Knees when the legs are in frame.
    if (tier === 'legs') { s.kneeL = Math.PI - ang(w[LM.lhip], w[LM.lkn], w[LM.lan]); s.kneeR = Math.PI - ang(w[LM.rhip], w[LM.rkn], w[LM.ran]); }
    else { s.kneeL = s.kneeR = 0; }
    const wsh = mid3(w[LM.lsh], w[LM.rsh]), whip = mid3(w[LM.lhip], w[LM.rhip]);
    s.torsoPitch = Math.atan2(-(wsh.z - whip.z), -(wsh.y - whip.y));

    // Arms up, from the direction of each arm. MediaPipe still predicts elbows and wrists that have left
    // the top of the frame, so no visibility gate here: an arm counts as up when its elbow is clearly
    // above its shoulder or its wrist is above the nose.
    const armUp = (shI: number, elI: number, wrI: number) => lm[elI].y < lm[shI].y - 0.2 * shoulderW || lm[wrI].y < nose.y - 0.1 * shoulderW;
    s.armL = armUp(LM.lsh, LM.lel, LM.lwr); s.armR = armUp(LM.rsh, LM.rel, LM.rwr);
    s.handsUp = s.armL && s.armR;
    s.handsTogether = Math.hypot(lm[LM.lwr].x - lm[LM.rwr].x, lm[LM.lwr].y - lm[LM.rwr].y) < shoulderW * 0.5 && lm[LM.lwr].y < sh.y + shoulderW;

    // Vertical signals in scale units (image y over the slow scale, down positive).
    const ySh = sh.y / scale, yHip = hip.y / scale;
    // Standing baselines: adopt quickly when the body is taller than the baseline (you can only be
    // standing taller, never crouching taller), slowly otherwise and only while steady, so a long
    // crouch never pulls the baseline down and a step back re-levels within a couple of seconds.
    const steady = Math.abs(this.riseV) < 0.6;
    // Recorded: in a held tuck the baseline crept down to meet the body (6 s time constant), so the tuck faded
    // from 1.0 to 0.5 while the player was still low, and standing up afterwards read as a hop. Now a lower
    // baseline is only learned while the body reads as standing, and slowly.
    // Recorded again: the start gesture (both arms up) lifts the shoulders, and a baseline taken then read a
    // full crouch once the arms came down. So: never learn with an arm up; before the race just follow the
    // player; in the race hold the baseline through a tuck.
    const crouching = s.calibrated && s.crouch > 0.25;
    const armsUp = s.armL || s.armR;
    const track = (base: number, y: number) => {
      if (!base) return y;
      if (!steady || armsUp) return base;             // never learn from a move or with an arm in the air
      if (!this.raceMode) return base + (y - base) * (1 - Math.exp(-dtS / 1.2));
      if (y > base && crouching) return base;          // never learn "standing" from a crouch
      const tau = y < base ? 0.8 : 20.0;              // quick to accept standing taller, slow to accept lower
      return base + (y - base) * (1 - Math.exp(-dtS / tau));
    };
    this.standSh = track(this.standSh, ySh);
    if (tier !== 'shoulders') {
      // first sight of the hips (or after a reset): assume they sit as far below standing as the shoulders do
      if (!this.standHip) this.standHip = yHip - (ySh - this.standSh);
      this.standHip = track(this.standHip, yHip);
    }
    // Readiness: hold roughly upright for 0.8 s once, so the first readings are not taken mid-move.
    const c = this.cal;
    if (!s.calibrated) {
      const upright = Math.abs(raw) < 0.4 && (tier !== 'legs' || Math.min(s.kneeL, s.kneeR) < 0.7);
      if (upright) {
        if (!c.stableSince) c.stableSince = now;
        c.n++;
        this.status = `hold still ${'●'.repeat(Math.min(4, Math.ceil((now - c.stableSince) / 200)))}`;
        if (now - c.stableSince > 800 && c.n > 6) { s.calibrated = true; c.n = 0; c.stableSince = 0; c.tier = tier; }
      } else { c.stableSince = 0; c.n = 0; this.status = 'stand up straight for a second'; }
    } else this.status = turned ? 'face the camera' : shoulderW < 0.08 ? 'step closer' : this.luma < 55 ? 'tracking · more light would help' : tier === 'legs' ? 'tracking · full body' : tier === 'torso' ? 'tracking · upper body' : 'tracking · head and shoulders';
    // scene brightness, sampled now and then from a tiny canvas
    if (++this.lumaFrame % 30 === 0) {
      try {
        this.lumaCanvas ??= Object.assign(document.createElement('canvas'), { width: 32, height: 18 });
        const g = this.lumaCanvas.getContext('2d', { willReadFrequently: true })!;
        g.drawImage(this.video, 0, 0, 32, 18);
        const d = g.getImageData(0, 0, 32, 18).data; let sum = 0;
        for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        this.luma = sum / (d.length / 4);
      } catch {}
    }

    // Crouch: how far the body sits below its standing baseline. Hips when visible (they only drop when
    // the knees bend; shoulders also drop when you bow forward), shoulders otherwise, knees and leg span
    // when the legs are in frame.
    const legspan = tier === 'legs' ? (mid(lm[LM.lan], lm[LM.ran]).y - hip.y) / scale : 0;
    if (s.calibrated) {
      const tiltDrop = (1 - Math.cos(roll)) * 1.0; // a tilted shoulder line lowers its midpoint without any crouch
      let crouch = tier !== 'shoulders' && this.standHip ? (yHip - this.standHip) / 0.42 : (ySh - this.standSh - tiltDrop) / 0.55;
      // a hard sideways lean drops the hip midpoint a little in the image without any knee bend (recorded:
      // 0.2-0.4 of "crouch" at full lean), so take that back out
      if (tier !== 'shoulders') crouch -= 0.45 * Math.max(0, Math.abs(raw) - 0.12);
      if (tier === 'legs') {
        if (!c.legspan || legspan > c.legspan) c.legspan = legspan; // the longest leg span seen is standing
        crouch = Math.max(crouch, (1 - legspan / c.legspan) / 0.32, (Math.max(s.kneeL, s.kneeR) - 0.35) / 1.1);
      }
      crouch = this.crouchFilt.filter(clamp(crouch, 0, 1), tMs);
      s.crouch = crouch < 0.03 ? 0 : clamp(crouch, 0, 1);
    }

    // Hop: the whole body goes up and comes back down within half a second. Measured on the body centre
    // (shoulders and hips, or shoulders and head when the hips are out of frame) against its own recent
    // path, not against a standing baseline, so it works wherever the player stands. Both the shoulders and
    // the hips must rise (raising the arms lifts only the shoulders), the distance must hold (stepping toward
    // the camera also moves the body up), and it has to fall back (standing up from a tuck does not).
    const rise = this.lastYT ? clamp((this.lastY - ySh) / Math.max(0.008, dtS), -10, 10) : 0; // scale units per second upward
    this.riseV += (rise - this.riseV) * 0.5;
    this.lastY = ySh; this.lastYT = tMs;
    // (head and shoulders are always in view, so the centre is measured on them; switching to the hips when
    // they flicker in and out of frame made false hops. The hips only confirm, when they are seen.)
    const hips = tier !== 'shoulders';
    const yC = (ySh + nose.y / scale) / 2;
    const H = this.hopHist;
    // wrist heights over the shoulders in shoulder widths (+ = above): raising the arms sends these up fast
    const wl = (lm[LM.lsh].y - lm[LM.lwr].y) / shoulderW, wr = (lm[LM.rsh].y - lm[LM.rwr].y) / shoulderW;
    H.push({ t: now, c: yC, h: hips ? yHip : NaN, tl: hips ? torso : NaN, wl, wr, up: s.armL || s.armR || lm[LM.lwr].y < nose.y || lm[LM.rwr].y < nose.y });
    while (H.length && now - H[0].t > 900) H.shift();
    let ai = 0;
    for (let i = 1; i < H.length; i++) if (H[i].c < H[ai].c) ai = i;
    const apex = H[ai], age = now - apex.t;
    let hopRise = 0;
    if (age > 50 && age < 450 && now - this.lastJump > 700) {
      // the take-off point: the lowest the body sat in the half second before the apex
      let low = ai;
      for (let i = 0; i < ai; i++) if (apex.t - H[i].t < 500 && H[i].c > H[low].c) low = i;
      hopRise = H[low].c - apex.c;
      let hipBefore = NaN;
      for (let i = 0; i < ai; i++) if (apex.t - H[i].t < 500 && !isNaN(H[i].h)) hipBefore = isNaN(hipBefore) ? H[i].h : Math.max(hipBefore, H[i].h);
      const fall = yC - apex.c;
      const hipRise = isNaN(apex.h) || isNaN(hipBefore) ? NaN : hipBefore - apex.h;
      // Recorded hops: the head and shoulders rose 0.09-0.22 torso lengths and the hips just as much, while
      // the shoulder width swung 12-44 % (arms, shoulders rolling), so width cannot gate them. A hop is:
      // the hips rising with the head and shoulders when the hips are seen (a shrug lifts only the
      // shoulders), or a clear rise of the head and shoulders when they are all the camera sees; and the same
      // torso length before the take-off and after the landing (walking toward or away changes it).
      // Raising the arms lifts the shoulders and head and stretches the torso, which can pass for a hop (it
      // did). In a hop the wrists travel with the body; when the arms go up they rise fast over the shoulders.
      // Labelled test (4 arm raises, then 4 hops): the wrists rose 1.0-1.6 shoulder widths over the shoulders
      // in the arm raises and 0.2-0.7 in the hops. A big jump (the hips well up) counts whatever the arms do.
      let upSeen = false, wristRise = 0;
      for (let i = low; i < H.length; i++) { upSeen ||= H[i].up; wristRise = Math.max(wristRise, H[i].wl - H[low].wl, H[i].wr - H[low].wr); }
      const bigJump = isNaN(hipRise) ? hopRise >= 0.5 : hipRise >= 0.3;
      const armsRose = apex.t === this.vetoApex || ((upSeen || wristRise > 0.85) && !bigJump);
      if (armsRose) this.vetoApex = apex.t;     // once an arm raise, always an arm raise
      // (second round of real hops: some had the hips out of frame at 0.13-0.26, some hips right at 0.07; the
      // arm check now carries the shrug and arm-raise cases, so these can be a little more generous)
      const strong = !armsRose && (isNaN(hipRise) ? hopRise >= 0.1 : hopRise >= 0.07 && hipRise >= Math.max(0.06, hopRise * 0.35));
      const tlBefore = H[low].tl;
      const sameDistance = isNaN(tlBefore) || !hips || Math.max(tlBefore, torso) / Math.min(tlBefore, torso) < 1.2;
      if (this.telemetry && hopRise > 0.06 && now - this.lastHopDbg > 250) {
        this.lastHopDbg = now;
        console.warn(`[gt] hop? rise=${hopRise.toFixed(2)} hips=${isNaN(hipRise) ? 'n/a' : hipRise.toFixed(2)} fall=${fall.toFixed(3)}/0.018 torso=${isNaN(tlBefore) || !hips ? 'n/a' : (Math.max(tlBefore, torso) / Math.min(tlBefore, torso)).toFixed(2)}/1.20 age=${age.toFixed(0)} tier=${tier} wrist+${wristRise.toFixed(1)}sw${upSeen ? ' UP' : ''}${armsRose ? ' ARMS' : ''} -> ${strong && fall > 0.018 && sameDistance ? 'HOP' : 'no'}`);
      }
      if (strong && fall > 0.018 && sameDistance) {
        s.jump = true; this.lastJump = now; H.length = 0;
        if (this.telemetry) console.warn(`[gt] JUMP rise=${hopRise.toFixed(2)} hips=${isNaN(hipRise) ? 'n/a' : hipRise.toFixed(2)} fall=${fall.toFixed(2)} apexAge=${age.toFixed(0)}ms tier=${tier}`);
      }
    }
    // mid-hop the arms swing down hard: that is not a pole plant
    const hopping = now - this.lastJump < 400 || hopRise > 0.08;
    const aboveStanding = this.standSh - ySh;             // + = shoulders higher than when standing (telemetry)
    if (this.telemetry && s.pole) console.warn(`[gt] POLE x${s.pole} tier=${tier}`);
    if (this.telemetry && now - this.lastTele > 250) {
      this.lastTele = now;
      console.warn(`[gt] t=${(now / 1000).toFixed(1)} ${tier}${s.calibrated ? '' : ' (uncal)'} lean=${s.lean.toFixed(2)} raw=${raw.toFixed(2)} range=${this.leanPeak.toFixed(2)} crouch=${s.crouch.toFixed(2)} ySh=${ySh.toFixed(2)} std=${this.standSh.toFixed(2)} hip=${yHip.toFixed(2)}/${this.standHip.toFixed(2)} above=${aboveStanding.toFixed(2)} rise=${this.riseV.toFixed(2)} scale=${scale.toFixed(3)} ratio=${ratio.toFixed(2)} arms=${s.armL ? 'L' : '-'}${s.armR ? 'R' : '-'} vis=${shVis.toFixed(2)}/${hipVis.toFixed(2)}/${legVis.toFixed(2)} sw=${shoulderW.toFixed(3)} fps=${s.fps}${turned ? ' TURNED' : ''} c=${yC.toFixed(2)} range=${(Math.max(...H.map(e => e.c)) - Math.min(...H.map(e => e.c))).toFixed(2)}`);
    }
    // Pole plants: a wrist swinging down fast, ending below the shoulder line. One arm is a plant, both
    // arms together is a double pole. Measured in shoulder widths per second so distance cancels.
    const wristVis = (lm[LM.lwr].visibility ?? 1) > 0.3 && (lm[LM.rwr].visibility ?? 1) > 0.3;
    s.armsVisible = wristVis && (lm[LM.lel].visibility ?? 1) > 0.3 && (lm[LM.rel].visibility ?? 1) > 0.3;
    this.wristHist.push({ t: now, l: lm[LM.lwr].y / shoulderW, r: lm[LM.rwr].y / shoulderW });
    while (this.wristHist.length && now - this.wristHist[0].t > 160) this.wristHist.shift();
    if (wristVis && this.wristHist.length > 2 && !hopping) {
      const a = this.wristHist[0], b = this.wristHist[this.wristHist.length - 1], dtW = (b.t - a.t) / 1000 + 1e-6;
      const vl = (b.l - a.l) / dtW, vr = (b.r - a.r) / dtW; // + = moving down the frame
      const plant = (v: number, wrY: number, shY: number, i: number) => { if (v > 4.0 && wrY > shY && now - this.lastPole[i] > 350) { this.lastPole[i] = now; return 1; } return 0; };
      const n = plant(vl, lm[LM.lwr].y, lm[LM.lsh].y, 0) + plant(vr, lm[LM.rwr].y, lm[LM.rsh].y, 1);
      if (n === 2 || (n === 1 && now - this.lastDouble < 150)) { s.pole = 2; this.lastDouble = 0; }
      else if (n === 1) { s.pole = 1; this.lastDouble = now; }
    }
    s.debug = `${tier} lean ${s.lean.toFixed(2)} (range ${this.leanPeak.toFixed(2)}${turned ? ', turned' : ''}) crouch ${s.crouch.toFixed(2)} luma ${this.luma.toFixed(0)} arms ${s.armL ? 'L' : '-'}${s.armR ? 'R' : '-'} vis sh ${shVis.toFixed(2)} hip ${hipVis.toFixed(2)} legs ${legVis.toFixed(2)} ${this.delegate}`;
    return s;
  }

  private decay() {
    const s = this.signals;
    s.present = false; s.tier = 'none'; s.legs = false; s.lean *= 0.8; s.crouch *= 0.9; if (Math.abs(s.lean) < 0.02) { s.lean = 0; this.leanFilt.reset(); } if (s.crouch < 0.02) { s.crouch = 0; this.crouchFilt.reset(); } this.held = 0; this.lastSh.w = 0; s.handsUp = false; s.armL = s.armR = false; s.armsVisible = false; s.pole = 0; s.handsTogether = false;
  }

  /** Mirror-view skeleton for the picture-in-picture, with a framing guide. */
  draw(ctx: CanvasRenderingContext2D) {
    const W = ctx.canvas.width, H = ctx.canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (this.video.readyState >= 2) ctx.drawImage(this.video, 0, 0, W, H);
    const lm = this.signals.image;
    if (!lm) return;
    const present = this.signals.present;
    ctx.lineWidth = 3; ctx.lineCap = 'round';
    ctx.strokeStyle = !present ? 'rgba(255,255,255,.35)' : this.signals.calibrated ? 'rgba(55,230,168,.95)' : 'rgba(255,179,64,.95)';
    for (const [a, b] of EDGES) {
      if ((lm[a].visibility ?? 1) < 0.3 || (lm[b].visibility ?? 1) < 0.3) continue;
      ctx.beginPath(); ctx.moveTo(lm[a].x * W, lm[a].y * H); ctx.lineTo(lm[b].x * W, lm[b].y * H); ctx.stroke();
    }
    ctx.fillStyle = '#fff';
    for (const i of [...SHOULDERS, ...HIPS, ...LEGS]) { if ((lm[i].visibility ?? 1) < 0.3) continue; ctx.beginPath(); ctx.arc(lm[i].x * W, lm[i].y * H, 4, 0, Math.PI * 2); ctx.fill(); }
    if (present) {
      // shoulder line (lean)
      ctx.strokeStyle = 'rgba(61,123,255,.9)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lm[LM.lsh].x * W, lm[LM.lsh].y * H); ctx.lineTo(lm[LM.rsh].x * W, lm[LM.rsh].y * H); ctx.stroke();
      // arm-up indicators: one bar per arm along the top edge (the canvas is mirrored, so left arm on the right)
      const s = this.signals;
      ctx.fillStyle = s.armR ? 'rgba(55,230,168,.95)' : 'rgba(255,255,255,.2)'; ctx.fillRect(4, 4, W / 2 - 6, 6);
      ctx.fillStyle = s.armL ? 'rgba(55,230,168,.95)' : 'rgba(255,255,255,.2)'; ctx.fillRect(W / 2 + 2, 4, W / 2 - 6, 6);
    }
  }
}

function mid(a: NormalizedLandmark, b: NormalizedLandmark) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
function mid3(a: Landmark, b: Landmark) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 }; }
