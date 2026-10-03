// Your body as the controller. MediaPipe Pose Landmarker on the webcam, turned into three signals a
// skier understands: lean (left/right), crouch (0..1) and a hop. Also the joint angles the G1 copies.
import { FilesetResolver, PoseLandmarker, type NormalizedLandmark, type Landmark } from '@mediapipe/tasks-vision';

import { LM, EDGES } from './landmarks.ts';
export { LM, EDGES };
const KEY = [LM.lsh, LM.rsh, LM.lhip, LM.rhip, LM.lkn, LM.rkn, LM.lan, LM.ran];

export interface BodySignals {
  present: boolean;       // whole body tracked
  calibrated: boolean;
  lean: number;           // -1..1, + = skier's right
  crouch: number;         // 0..1
  jump: boolean;          // true for one frame
  handsUp: boolean;       // both wrists above the nose
  handsTogether: boolean; // wrists close: the calibration gesture
  kneeL: number;          // radians of flexion (0 straight)
  kneeR: number;
  torsoPitch: number;     // forward lean of the torso, radians
  world?: Landmark[];     // 33 metric landmarks, hips at the origin
  image?: NormalizedLandmark[];
  fps: number;
}

const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';

const ang = (a: Landmark, b: Landmark, c: Landmark) => {
  const ux = a.x - b.x, uy = a.y - b.y, uz = a.z - b.z, vx = c.x - b.x, vy = c.y - b.y, vz = c.z - b.z;
  const d = (ux * vx + uy * vy + uz * vz) / (Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz) + 1e-9);
  return Math.acos(Math.max(-1, Math.min(1, d)));
};

export class BodyTracker {
  video: HTMLVideoElement;
  landmarker: PoseLandmarker | null = null;
  signals: BodySignals = { present: false, calibrated: false, lean: 0, crouch: 0, jump: false, handsUp: false, handsTogether: false, kneeL: 0, kneeR: 0, torsoPitch: 0, fps: 0 };
  status = 'camera off';
  // calibration: neutral standing pose
  private cal = { legspan: 0, hipY: 0, n: 0, acc: [0, 0, 0] as number[], stableSince: 0 };
  private leanF = 0; private crouchF = 0;
  private hipHist: { t: number; y: number }[] = [];
  private lastJump = 0;
  private frames = 0; private fpsT = 0;
  private lastVideoT = -1;

  constructor(video: HTMLVideoElement) { this.video = video; }

  async start() {
    this.status = 'starting camera';
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
    this.video.srcObject = stream;
    await this.video.play();
    this.status = 'loading pose model';
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    this.landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO', numPoses: 1, minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    this.status = 'step into frame';
  }

  recalibrate() { this.cal = { legspan: 0, hipY: 0, n: 0, acc: [0, 0, 0], stableSince: 0 }; this.signals.calibrated = false; }

  /** Call every animation frame. Returns the latest signals (jump is edge-triggered). */
  update(now: number): BodySignals {
    const s = this.signals;
    s.jump = false;
    if (!this.landmarker || this.video.readyState < 2) return s;
    if (this.video.currentTime === this.lastVideoT) return s;
    this.lastVideoT = this.video.currentTime;
    const res = this.landmarker.detectForVideo(this.video, now);
    this.frames++;
    if (now - this.fpsT > 1000) { s.fps = this.frames; this.frames = 0; this.fpsT = now; }
    const lm = res.landmarks[0], w = res.worldLandmarks[0];
    if (!lm || !w) { this.decay(); this.status = 'step into frame'; return s; }
    s.image = lm; s.world = w;
    let vis = 1, ymin = 1, ymax = 0;
    for (const i of KEY) { vis = Math.min(vis, lm[i].visibility ?? 1); ymin = Math.min(ymin, lm[i].y); ymax = Math.max(ymax, lm[i].y); }
    if (vis < 0.45 || ymax - ymin < 0.12) { this.decay(); this.status = 'step back: whole body in frame'; return s; }
    s.present = true;

    const sh = mid(lm[LM.lsh], lm[LM.rsh]), hip = mid(lm[LM.lhip], lm[LM.rhip]), an = mid(lm[LM.lan], lm[LM.ran]);
    const shoulderW = Math.hypot(lm[LM.lsh].x - lm[LM.rsh].x, lm[LM.lsh].y - lm[LM.rsh].y) + 1e-6;
    const torso = Math.hypot(sh.x - hip.x, sh.y - hip.y) + 1e-6;
    const legspan = (an.y - hip.y) / shoulderW; // scale free: legs in shoulder widths

    // Lean. The image is not mirrored here: the user's left is +x. Leaning to their right moves the
    // shoulders to smaller x than the hips, so right lean = hips.x - shoulders.x.
    const leanTorso = (hip.x - sh.x) / shoulderW;
    const leanHips = (an.x - hip.x) / shoulderW; // hips pushed inside the turn, feet outside
    let lean = 0.75 * leanTorso + 0.35 * leanHips;
    const dead = 0.08;
    lean = Math.abs(lean) < dead ? 0 : (lean - Math.sign(lean) * dead) / (0.55 - dead);
    lean = Math.max(-1, Math.min(1, lean));
    this.leanF += (lean - this.leanF) * 0.35;
    s.lean = this.leanF;

    // Knees from metric 3D landmarks.
    s.kneeL = Math.PI - ang(w[LM.lhip], w[LM.lkn], w[LM.lan]);
    s.kneeR = Math.PI - ang(w[LM.rhip], w[LM.rkn], w[LM.ran]);
    const wsh = mid3(w[LM.lsh], w[LM.rsh]), whip = mid3(w[LM.lhip], w[LM.rhip]);
    s.torsoPitch = Math.atan2(-(wsh.z - whip.z), -(wsh.y - whip.y)); // + = chest forward (toward camera)

    // Calibration: hold still standing for ~1.2 s. Hands together also forces it.
    const c = this.cal;
    s.handsTogether = Math.hypot(lm[LM.lwr].x - lm[LM.rwr].x, lm[LM.lwr].y - lm[LM.rwr].y) < shoulderW * 0.5 && lm[LM.lwr].y < hip.y;
    s.handsUp = lm[LM.lwr].y < lm[LM.nose].y && lm[LM.rwr].y < lm[LM.nose].y;
    if (!s.calibrated || s.handsTogether) {
      const knee = Math.min(s.kneeL, s.kneeR);
      const standing = knee < 0.5 && Math.abs(lean) < 0.25;
      if (standing) {
        if (!c.stableSince) c.stableSince = now;
        c.acc[0] += legspan; c.acc[1] += hip.y; c.n++;
        if (now - c.stableSince > 1200 && c.n > 10) {
          c.legspan = c.acc[0] / c.n; c.hipY = c.acc[1] / c.n; s.calibrated = true; c.acc = [0, 0, 0]; c.n = 0; c.stableSince = 0;
        }
        this.status = s.calibrated ? 'tracking' : 'stand still…';
      } else { c.stableSince = 0; c.acc = [0, 0, 0]; c.n = 0; this.status = 'stand up straight to calibrate'; }
    } else this.status = 'tracking';

    // Crouch: legs shorter than standing (in shoulder widths) and knees bent.
    if (s.calibrated) {
      const byLegs = 1 - legspan / c.legspan;             // 0 standing, ~0.35 deep tuck
      const byKnees = (Math.max(s.kneeL, s.kneeR) - 0.35) / 1.1; // 20 deg..85 deg
      let crouch = Math.max(byLegs / 0.32, byKnees);
      crouch = Math.max(0, Math.min(1, crouch));
      this.crouchF += (crouch - this.crouchF) * 0.25;
      s.crouch = this.crouchF;
    }

    // Jump: hips shoot upward (image y decreases) faster than a squat-stand.
    this.hipHist.push({ t: now, y: hip.y / torso });
    while (this.hipHist.length && now - this.hipHist[0].t > 220) this.hipHist.shift();
    if (this.hipHist.length > 2) {
      const a = this.hipHist[0], b = this.hipHist[this.hipHist.length - 1];
      const v = (a.y - b.y) / ((b.t - a.t) / 1000 + 1e-6); // torso lengths per second upward
      if (v > 1.6 && now - this.lastJump > 900) { s.jump = true; this.lastJump = now; }
    }
    return s;
  }

  private decay() {
    const s = this.signals;
    s.present = false; this.leanF *= 0.8; this.crouchF *= 0.9; s.lean = this.leanF; s.crouch = this.crouchF; s.handsUp = false; s.handsTogether = false;
  }

  /** Mirror-view skeleton for the picture-in-picture. */
  draw(ctx: CanvasRenderingContext2D) {
    const W = ctx.canvas.width, H = ctx.canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (this.video.readyState >= 2) ctx.drawImage(this.video, 0, 0, W, H);
    const lm = this.signals.image;
    if (!lm || !this.signals.present) return;
    ctx.lineWidth = 3; ctx.lineCap = 'round';
    ctx.strokeStyle = this.signals.calibrated ? 'rgba(55,230,168,.95)' : 'rgba(255,179,64,.95)';
    for (const [a, b] of EDGES) { ctx.beginPath(); ctx.moveTo(lm[a].x * W, lm[a].y * H); ctx.lineTo(lm[b].x * W, lm[b].y * H); ctx.stroke(); }
    ctx.fillStyle = '#fff';
    for (const i of KEY) { ctx.beginPath(); ctx.arc(lm[i].x * W, lm[i].y * H, 4, 0, Math.PI * 2); ctx.fill(); }
  }
}

function mid(a: NormalizedLandmark, b: NormalizedLandmark) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
function mid3(a: Landmark, b: Landmark) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 }; }
