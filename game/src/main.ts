// Ground Truth: Ski. Your body steers a Unitree G1 down the real Streif; everyone on the mountain is
// live through SpacetimeDB; every run becomes motion-on-terrain data. This file is the game loop:
// phases, camera, feel (hit-stop, shake, slow-mo), TV splits, the leader's ghost, cheers and the finish.
import * as THREE from 'three';
import '@fontsource-variable/saira/standard.css';
import '@fontsource-variable/saira/standard-italic.css';
import { loadCourse, type Course } from './course.ts';
import { initialState, step, GateTracker, type SkierState, type Input } from './physics.ts';
import { World, type FarTerrain } from './world.ts';
import { Post } from './post.ts';
import { loadG1Assets, G1, retarget, retargetArms, stanceFrom, SKI_STANCE } from './g1.ts';
import { BodyTracker, listCameras, preferredCamera, rememberCamera } from './pose.ts';
import { Net, fmtTime, fmtDelta, fmtInt, colorOf, hazardMap, type Skier, type PoseSample, type Run, type TraceChunk, type HazardBin, type VitalSample } from './net.ts';
import { Hud } from './hud.ts';
import { Audio } from './audio.ts';
import { Powerups, POWER_COLOR, shieldBubble, magnetAura, type Pickup } from './powerups.ts';
import { Recorder } from './recorder.ts';
import { Vitals } from './vitals.ts';
import { startPresage, presageEndpoint, PRESAGE_HINT, type PresageMessage } from './presage.ts';
import { Commentator, P, pick } from './commentator.ts';

const COURSE_SLUG = 'kitzbuhel-streif';
const params = new URLSearchParams(location.search);
const MISS_PENALTY_MS = 1500;
const STREAK_WORD = 'STREIF';
const CRYSTAL_MS = 100;
type Phase = 'lobby' | 'intro' | 'ready' | 'countdown' | 'racing' | 'finished';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const ease = (t: number) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

async function main() {
  const hud = new Hud();
  const status = (s: string) => { $('status').textContent = s; };
  const lowfx = params.has('lowfx');
  const noCam = params.has('nocam');
  const autopilot = params.has('autopilot');
  const fast = Math.max(1, Math.min(8, Number(params.get('fast') || 1)));

  // ---- renderer ----
  const canvas = $('gl') as HTMLCanvasElement;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(lowfx ? 1 : Math.min(devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = !lowfx; renderer.shadowMap.type = THREE.PCFShadowMap;
  // ACES by default (punchy snow); ?tone=agx for cleaner whites and stabler hues, ?tone=neutral to compare
  renderer.toneMapping = params.get('tone') === 'agx' ? THREE.AgXToneMapping : params.get('tone') === 'neutral' ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = Number(params.get('exposure')) || 1.08;
  const camera = new THREE.PerspectiveCamera(62, innerWidth / innerHeight, 0.3, 30000);
  camera.up.set(0, 0, 1);

  status('loading the Streif…');
  const lenParam = params.get('length');
  const raceLength = lenParam === 'full' ? 1e9 : Number(lenParam) || 650;
  const [course, g1Assets, far] = await Promise.all([loadCourse(COURSE_SLUG, '/courses', 90, raceLength), loadG1Assets(), loadFar(COURSE_SLUG)]);
  const courseName = shortName(course);
  const dropM = course.meta.centerline[course.startIndex][2] - course.meta.centerline[course.finishIndex][2];
  const startAlt = course.meta.z_datum_msl + course.meta.centerline[course.startIndex][2];
  hud.course(`${course.meta.resort} · start ${startAlt.toFixed(0)} m`, courseName, `${course.raceLength.toFixed(0)} m · ${dropM.toFixed(0)} m drop · ${course.gates.length} gates`);
  const world = new World(course, far, lowfx);
  G1.env = world.makeEnv(renderer);
  const post = new Post(renderer, world.scene, camera, lowfx);
  const powerups = new Powerups(world.scene, course, G1.env);
  const bubble = shieldBubble(); world.scene.add(bubble.mesh);
  const aura = magnetAura(); world.scene.add(aura.group);
  const resize = () => { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); post.setSize(innerWidth, innerHeight); };
  addEventListener('resize', resize); resize();

  const me = new G1(g1Assets.def, g1Assets.meshes);
  world.scene.add(me.root);
  me.apply(SKI_STANCE, 1);
  const ghost = new G1(g1Assets.def, g1Assets.meshes, new THREE.Color('#ffc34d'), true);
  ghost.root.visible = false; world.scene.add(ghost.root);
  const ghostLabel = makeLabel('leader', '#ffc34d'); ghostLabel.visible = false; world.scene.add(ghostLabel);

  // ---- state ----
  let phase: Phase = 'lobby';
  let state: SkierState = initialState(course);
  let gates = new GateTracker(course);
  let name = '', joinCode = '', myColor = '#ff3b4e';
  let raceT = 0, maxSpeed = 0, runKey = '', traceSeq = 0, lastTraceT = 0, samples = 0, finishedAt = 0, runStartWall = 0;
  let splits: number[] = [], streak = 0, bestStreak = 0, airStart = 0;
  let chunk: PoseSample[] = [];
  let countdownEnd = 0, countdownBeeps = 0;
  let introStart = 0; const INTRO_S = 6.5;
  let handsSince = 0;
  let timeScale = 1, freeze = 0, trauma = 0, slowmoUntil = 0, orbitStart = 0, photoDue = 0, photoTaken = false;
  let cheersThisRun = 0, crowdMeter = 0;
  // powerups: seconds of slipstream left, a shield held, crystals and the time they took off, misses the
  // shield forgave, and the tuck charge (hold a straight tuck to charge, stand up to fire)
  let boostT = 0, shield = false, crystals = 0, bonusMs = 0, forgiven = 0;
  let tuckHold = 0, tuckLevel = 0, tuckBoostT = 0, tuckFired = 0, sparkCd = 0;
  const crystalTotal = powerups.pickups.filter(p => p.kind === 'crystal').length;
  /** Everything the clock adds or takes away: missed gates (minus forgiven) and crystals. */
  const adjMs = () => (gates.missed.filter(Boolean).length - forgiven) * MISS_PENALTY_MS - bonusMs;
  // vitals (Presage): a resting baseline from the start gate, then Ice veins when you stay cool (crystals x2)
  // and Steady when your heart races (the gates widen a little)
  const vitals = new Vitals();
  let vitalMode: '' | 'ice' | 'steady' = '', saidBreathe = false;
  const resetPowers = () => { boostT = 0; shield = false; crystals = 0; bonusMs = 0; forgiven = 0; tuckHold = 0; tuckLevel = 0; tuckBoostT = 0; tuckFired = 0; powerups.reset(); };
  let camMode: 'chase' | 'front' | 'side' = (params.get('cam') as any) || 'chase';
  let best: Run | null = null; let ghostSamples: PoseSample[] = []; let ghostKey = '';
  let autoRestart = 0;
  // Mimic-the-ghost tutorial: the gold G1 demonstrates, the player copies, each step confirms the tracking.
  const TUT = [
    { title: 'LEAN LEFT', sub: 'shoulders over your left hip, like the gold skier', test: () => inp.lean < -0.5, pose: { lean: -1, crouch: 0.15 } },
    { title: 'LEAN RIGHT', sub: 'and over to the right', test: () => inp.lean > 0.5, pose: { lean: 1, crouch: 0.15 } },
    { title: 'GET LOW', sub: 'bend your knees into a tuck', test: () => inp.crouch > 0.6, pose: { lean: 0, crouch: 1 } },
    { title: 'PLANT YOUR POLES', sub: 'swing both arms down like a push', test: () => (inp.pole ?? 0) > 0, pose: { lean: 0, crouch: 0.3 }, event: true },
    { title: 'HOP', sub: 'a quick little jump, both feet off the floor', test: () => inp.jump, pose: { lean: 0, crouch: 0.5 }, event: true },
  ];
  const tutorialWanted = !autopilot && params.get('tutorial') !== '0';
  let tut = { on: false, step: 0, holdSince: 0, doneOnce: false, flash: 0 };
  let tutLeanMax = 0;
  const gauge = makeGauge(); world.scene.add(gauge.group);
  const keys = new Set<string>();
  let keyJump = false, keyPole = false, hopAt = 0;

  const net = new Net({
    status: s => status(`SpacetimeDB: ${s}`),
    skier: (kind, row) => { if (!net.isMe(row.identity)) ghostEvent(kind, row); },
    cheer: c => onCheer(c.fromName, c.toName, c.kind),
    changed: () => refreshLists(),
  });
  net.connect();

  const audio = new Audio();
  // the race commentator (ElevenLabs via the dev server, the browser's voice without a key)
  const commentator = new Commentator(() => audio.ctx, () => audio.out, (t, f) => audio.speak(t, f));
  audio.speaker = (text, force) => commentator.say(text, force ? P.big : P.call);
  const say = (text: string, pri: number = P.call) => { if (audio.voiceOn) commentator.say(text, pri); };
  let saidSpeed = 0, saidIce = false, saidRest = false, chatterN = 0, lastChatter = 0, ghostGap: number | null = null;
  /** The commentator fills a quiet booth: speed milestones, then every ~10 s speed, gates left, the gap to the leader or your pulse. */
  function chatter(now: number) {
    if (phase !== 'racing' || !audio.voiceOn) return;
    const kmh = state.speed * 3.6;
    if (kmh > 50 && saidSpeed < 50) { saidSpeed = 50; say(pick(['Fifty kilometres an hour!', 'Over fifty now, and building!'])); return; }
    if (kmh > 65 && saidSpeed < 65) { saidSpeed = 65; say(pick(['Sixty-five, flat out!', 'Look at the speed, sixty-five!'])); return; }
    if (raceT < 6 || now - lastChatter < 10000 || commentator.quietFor < 6) return;
    lastChatter = now;
    const left = gates.total - gates.next;
    const opts = [`${Math.round(kmh)} kilometres an hour.`, `${left} gates to go.`];
    if (ghostGap !== null && best) opts.push(ghostGap > 0 ? `${Math.abs(ghostGap).toFixed(0)} metres ahead of ${best.name}'s line.` : `${Math.abs(ghostGap).toFixed(0)} metres behind ${best.name}.`);
    if (vitals.live) opts.push(`Heart rate ${Math.round(vitals.hr)}.`);
    say(opts[chatterN++ % opts.length], P.chatter);
  }
  const video = $('cam') as HTMLVideoElement;
  const body = new BodyTracker(video);
  const camCtx = ($('cam-canvas') as HTMLCanvasElement).getContext('2d')!;
  // on the live site the camera stream goes to the Presage relay to read the pulse: say so on the start screen
  if (presageEndpoint()?.remote && !params.has('nopresage')) $('pulse-note').hidden = false;
  const rec = params.has('watch') ? new Recorder($('cam-canvas') as HTMLCanvasElement, canvas) : null;
  const attractCtx = ($('attract-canvas') as HTMLCanvasElement).getContext('2d')!;
  let camStarted = false;
  const camSelect = $('camera') as HTMLSelectElement;
  async function fillCameras() {
    const cams = await listCameras();
    const want = preferredCamera();
    camSelect.innerHTML = '<option value="">default</option>' + cams.map(c => `<option value="${c.id}">${c.label.replace(/</g, '')}</option>`).join('');
    const hit = cams.find(c => c.id === want || (want && c.label.toLowerCase().includes(want.toLowerCase())));
    if (hit) camSelect.value = hit.id;
  }
  camSelect.addEventListener('change', async () => {
    rememberCamera(camSelect.value);
    if (camStarted) { try { await body.openCamera(camSelect.value); hud.toast(`camera: ${body.cameraLabel || 'default'}`); } catch (e) { console.warn(e); hud.toast('could not open that camera'); } }
  });
  async function startCamera() {
    if (camStarted || noCam) return; camStarted = true;
    try { await body.start(); await fillCameras(); hud.camLabel(`camera: ${body.cameraLabel || 'on'}`); const ep = presageEndpoint(); if (ep && !params.has('nopresage')) { vitals.status = 'connecting'; startPresage(video, onPresage, ep); } }
    catch (err) { console.warn(err); hud.camLabel(`no camera: ${String((err as Error)?.message ?? err).slice(0, 60)}`); }
  }
  if (!noCam) startCamera(); else hud.camLabel('keyboard mode');

  // ---- other skiers ----
  const others = new Map<string, { g1: G1; label: THREE.Sprite; row: Skier; pos: THREE.Vector3; heading: number }>();
  function ghostEvent(kind: 'insert' | 'update' | 'delete', row: Skier) {
    const id = row.identity.toHexString();
    if (kind === 'delete') { const g = others.get(id); if (g) { world.scene.remove(g.g1.root); world.scene.remove(g.label); others.delete(id); } return; }
    let g = others.get(id);
    if (!g) {
      const g1 = new G1(g1Assets.def, g1Assets.meshes, new THREE.Color(colorOf(row.color)), true);
      g1.setBib(row.name, colorOf(row.color), (row.color % 89) + 10);
      const label = makeLabel(row.name, colorOf(row.color));
      world.scene.add(g1.root); world.scene.add(label);
      g = { g1, label, row, pos: new THREE.Vector3(row.x, row.y, row.z), heading: row.heading };
      others.set(id, g);
    }
    g.row = row;
  }

  // Hazard Intelligence: everyone's vitals pooled by stretch of course; rebuilt when new vitals arrive
  let hazards: HazardBin[] = [], hazardCount = -1;
  const hazardsSaid = new Set<number>();
  let vitalBuf: VitalSample[] = [], lastVitalT = 0, lastVitalAt = 0, vitalsMapped = 0;
  function flushVitals(final = false) {
    if (vitalBuf.length && (vitalBuf.length >= 10 || final) && vitals.baseHr) { net.pushVitals(runKey, vitals.baseHr, vitalBuf); vitalBuf = []; }
  }
  function refreshLists() {
    const runs = net.runs(courseName);
    const chunks = net.vitals(courseName);
    if (chunks.length !== hazardCount) {
      hazardCount = chunks.length; hazards = hazardMap(chunks); world.setHazards(hazards);
      hud.hearts(chunks.reduce((n, c) => n + c.samples.length, 0), hazards.filter(b => b.rise >= 0.06 && b.n >= 2).length);
    }
    hud.leaderboard(runs, net.me);
    hud.dataset(net.stats(), net.skiers().length);
    hud.feed(net.feed());
    const p = net.myPlayer();
    if (p && colorOf(p.color) !== myColor) { myColor = colorOf(p.color); me.setBib(name, myColor, (p.color % 89) + 10); }
  }

  function onCheer(from: string, to: string, kind: string) {
    const forMe = !to || to.toLowerCase() === name.toLowerCase();
    crowdMeter = Math.min(1, crowdMeter + (forMe ? 0.25 : 0.1));
    world.crowd?.cheer(forMe ? 1 : 0.4);
    if (forMe) { cheersThisRun++; hud.cheer(from, kind, crowdMeter); if (kind === 'fire') audio.crowd(1.6, 0.6); else if (kind === 'clap') audio.crowd(1.2, 0.35); else audio.cowbell(0.7); }
    else hud.crowdMeter(crowdMeter);
  }

  // ---- join ----
  function join(n: string, code: string) {
    name = n.trim().slice(0, 24) || 'Skier';
    joinCode = code.trim().toUpperCase();
    audio.start();
    $('start').classList.add('hidden');
    net.join(name, joinCode, courseName);
    me.setBib(name, myColor, 1);
    startCamera();
    beginIntro();
    const ch = net.challengesFor(name);
    if (ch.length) setTimeout(() => hud.toast(`Challenge from ${ch[0].fromName}: beat ${fmtTime(ch[0].targetTimeMs)}`, 5000, 'warn'), INTRO_S * 1000);
  }
  $('join-form').addEventListener('submit', e => { e.preventDefault(); join(($('name') as HTMLInputElement).value, params.get('code') || ''); });
  if (params.get('name')) ($('name') as HTMLInputElement).value = params.get('name')!;
  if (params.has('auto')) setTimeout(() => ($('join-form') as HTMLFormElement).requestSubmit(), 300);

  addEventListener('keydown', e => {
    keys.add(e.key);
    if (e.key === ' ') { keyJump = true; e.preventDefault(); }
    if (e.key.toLowerCase() === 'p') keyPole = true;
    if (e.key === 'Enter') { if (phase === 'intro') endIntro(); else if (phase === 'ready') { tut.on = false; startCountdown(); } else if (phase === 'finished') resetRun(); }
    if (e.key.toLowerCase() === 't' && phase === 'ready') { tut.on = true; tut.step = 0; tut.holdSince = 0; }
    if (e.key.toLowerCase() === 'r' && (phase === 'racing' || phase === 'finished')) resetRun();
    if (e.key.toLowerCase() === 'c') camMode = camMode === 'chase' ? 'front' : camMode === 'front' ? 'side' : 'chase';
    if (e.key.toLowerCase() === 'k') body.recalibrate();
    if (e.key.toLowerCase() === 'v') { audio.voiceOn = !audio.voiceOn; hud.toast(`announcer ${audio.voiceOn ? 'on' : 'off'}`); }
    if (e.key.toLowerCase() === 'b') onCheer('Keyboard', name, 'bell');
  });
  addEventListener('keyup', e => keys.delete(e.key));
  canvas.addEventListener('pointerdown', () => { if (phase === 'intro') endIntro(); });

  // ---- phases ----
  function beginIntro() {
    phase = 'intro'; introStart = performance.now();
    say(pick([`Welcome to Kitzbühel and the mighty ${courseName}. ${name} is in the start house.`, `Here we are at the top of the ${courseName}. ${name} is up next.`, `The Hahnenkamm, Kitzbühel. ${name}, the ${courseName} is yours.`]), P.must);
    camInit = false;
  }
  function endIntro() { goReady(); }
  function goReady() {
    phase = 'ready'; state = initialState(course); gates = new GateTracker(course);
    world.trailL.reset(); world.trailR.reset(); world.updateGates(0, gates.missed);
    hud.clearSplits(); hud.streak(0, STREAK_WORD); hud.ghost('', ''); hud.result(null);
    streak = 0; cheersThisRun = 0; post.dim = 0; timeScale = 1; resetPowers();
    tut.on = tutorialWanted && (!tut.doneOnce || params.get('tutorial') === '1'); tut.step = 0; tut.holdSince = 0; tutLeanMax = 0;
    if (!tut.on) hud.msg('RAISE BOTH HANDS', 'hold them up for a second, or press Enter', 0);
    best = net.bestRun(courseName, gates.total);
    if (best && best.key !== ghostKey) { if (ghostKey) net.unwatchTrace(ghostKey); ghostKey = best.key; ghostSamples = []; net.watchTrace(ghostKey, chunks => { ghostSamples = flattenTrace(chunks); }); }
    ghost.root.visible = ghostLabel.visible = false;
    autoRestart = performance.now() + 4000;
  }
  function startCountdown() { tut.on = false; tut.doneOnce = true; phase = 'countdown'; countdownEnd = performance.now() + 3300; countdownBeeps = 0; audio.start(); hud.hideMsg(); } // (no recalibration here: the arms are still up)
  function beginRace() {
    phase = 'racing';
    state = initialState(course); gates = new GateTracker(course);
    raceT = 0; maxSpeed = 0; traceSeq = 0; samples = 0; chunk = []; lastTraceT = 0; splits = []; streak = 0; bestStreak = 0; photoTaken = false;
    resetPowers(); world.updateGates(0, gates.missed, 0); saidBreathe = false; saidIce = false; saidSpeed = 0;
    vitalBuf = []; lastVitalT = 0; vitalsMapped = 0; hazardsSaid.clear();
    runStartWall = performance.now();
    runKey = `${name}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    world.trailL.reset(); world.trailR.reset();
    net.startRun(runKey, courseName, gates.total);
    hud.hideMsg(); hud.toast('GO', 900, 'good');
    say(pick([`And ${name} is away!`, `${name} pushes out of the start!`, 'Here we go!', `Go, ${name}!`]), P.big);
    audio.startBeep(true);
    post.hit('#37e6a8', 0.5);
  }
  function resetRun() { if (ghostKey && phase === 'finished') { /* keep watching */ } goReady(); }
  function flushTrace(final = false) {
    if (chunk.length && (chunk.length >= 50 || final)) { net.pushTrace(runKey, traceSeq++, chunk); chunk = []; }
  }
  function finishRace() {
    phase = 'finished'; finishedAt = performance.now();
    flushTrace(true); flushVitals(true);
    const timeMs = Math.max(1000, Math.round(raceT * 1000) + adjMs());
    net.finishRun(runKey, timeMs, gates.made, maxSpeed, course.raceLength, splits);
    const runs = net.runs(courseName);
    const faster = runs.filter(r => r.timeMs < timeMs).length;
    const leader = runs[0];
    const deltaMs = leader ? timeMs - leader.timeMs : null;
    const myBest = runs.filter(r => net.me && r.identity.isEqual(net.me))[0];
    // feel: freeze, then slow motion with an orbit, confetti and the crowd
    freeze = 0.15; slowmoUntil = performance.now() + 150 + 1200; orbitStart = performance.now(); photoDue = performance.now() + 900;
    trauma = Math.max(trauma, 0.5);
    const f = course.finish; const fm = new THREE.Vector3((f.poles[0][0] + f.poles[1][0]) / 2, (f.poles[0][1] + f.poles[1][1]) / 2, (f.poles[0][2] + f.poles[1][2]) / 2 + 5);
    world.confetti.burst(fm, 300, 6);
    world.crowd?.cheer(1.5);
    audio.fanfare();
    post.hit('#ffffff', 0.9);
    hud.result({ heart: vitals.summary(runStartWall) ? { ...vitals.summary(runStartWall)!, mapped: vitalsMapped } : null, timeMs, rank: faster + 1, of: runs.length + (runs.some(r => r.key === runKey) ? 0 : 1), deltaMs: deltaMs === null || deltaMs <= 0 ? (leader && leader.key !== runKey ? deltaMs : null) : deltaMs, gates: `${gates.made}/${gates.total}`, kmh: Math.round(maxSpeed * 3.6), samples, cheers: cheersThisRun, best: !!myBest && timeMs < myBest.timeMs });
    const place = ['first', 'second', 'third'][faster] ?? `${faster + 1}th`;
    const heart = vitals.summary(runStartWall);
    setTimeout(() => say(`${pick([`${name} crosses the line!`, `And that's the finish for ${name}!`, `${name} is home!`])} ${spokenTime(timeMs)}, ${place} place.${deltaMs !== null && deltaMs > 0 ? ` ${(deltaMs / 1000).toFixed(1)} seconds off the lead.` : faster === 0 ? ' A new fastest time!' : ''}${heart && heart.peakHr ? ` Heart rate peaked at ${Math.round(heart.peakHr)}.` : ''}`, P.must), 500);
    autoRestart = performance.now() + 16000;
  }

  // ---- camera ----
  const camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), goal = new THREE.Vector3(), goalLook = new THREE.Vector3();
  let camInit = false;
  const fwd = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3(0, 0, 1), pos = new THREE.Vector3();
  const tmpN = new THREE.Vector3(), tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();
  const shake = new THREE.Vector3();

  // ---- loop ----
  let lastNow = performance.now();
  let frame = 0;
  let poseTarget: Record<string, number> = { ...SKI_STANCE };
  const inp: Input = { lean: 0, crouch: 0, jump: false, push: 0, pole: 0 };
  let sig0 = body.signals;
  const photoCanvas = document.createElement('canvas'); photoCanvas.width = 640; photoCanvas.height = 360;

  function tick() {
    requestAnimationFrame(tick);
    const now = performance.now();
    const wallDt = Math.min((now - lastNow) / 1000, 0.05); lastNow = now;
    frame++;
    crowdMeter = Math.max(0, crowdMeter - wallDt * 0.04);
    if (document.body.dataset.phase !== phase) document.body.dataset.phase = phase;

    // body (the standing baselines hold through a tuck only once the race is on, or in the tutorial's tuck step)
    body.raceMode = phase === 'countdown' || phase === 'racing' || (phase === 'ready' && tut.on && TUT[tut.step]?.title === 'GET LOW');
    const sig = body.update(now); sig0 = sig;
    if (frame % 2 === 0) { body.draw(camCtx); if (phase === 'lobby') body.draw(attractCtx); }
    if (!(phase === 'lobby' && handsSince && now - handsSince > 100)) hud.camLabel(body.landmarker ? `${body.status}${sig.fps ? ` · ${sig.fps} fps` : ''}${params.has('debug') ? ` · ${sig.debug}` : ''}` : (noCam ? 'keyboard mode' : body.status));

    // input: body first, keys override
    const kl = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
    const kc = keys.has('ArrowDown') ? 1 : 0;
    const useBody = sig.present;
    inp.lean = kl !== 0 ? kl : useBody ? sig.lean : 0;
    inp.crouch = kc ? 1 : useBody ? sig.crouch : 0;
    inp.jump = keyJump || (useBody && sig.jump); keyJump = false;
    // accelerate: ArrowUp on the keyboard; with the body, a deep crouch pumps you forward
    inp.push = keys.has('ArrowUp') ? 1 : useBody ? Math.max(0, (sig.crouch - 0.45) / 0.55) : 0;
    inp.pole = keyPole ? 1 : useBody ? sig.pole : 0; keyPole = false;
    if (inp.pole && (phase === 'racing' || phase === 'finished')) { audio.pole(inp.pole === 2); if (phase === 'racing') { trauma = Math.max(trauma, 0.08); } }
    if (autopilot && (phase === 'racing' || phase === 'finished')) autopilotInput();
    hud.meters(inp.lean, inp.crouch);
    if (phase === 'racing' && !autopilot) tuckCharge(wallDt * timeScale * fast); // the attract autopilot skis clean, no tuck boosts
    inp.boost = phase === 'racing' ? (boostT > 0 ? 0.6 : 0) + (tuckBoostT > 0 ? 0.15 * tuckFired : 0) : 0;

    // hands up: start (held ~1.2 s) from lobby / ready / finished, skip the intro
    if (sig.handsUp) { if (!handsSince) handsSince = now; } else handsSince = 0;
    const held = handsSince ? now - handsSince : 0;
    if (phase === 'lobby') {
      if (held > 1200) { join(($('name') as HTMLInputElement).value || `Skier ${Math.floor(Math.random() * 90 + 10)}`, params.get('code') || ''); handsSince = 0; }
      else if (held > 100) hud.camLabel(`arms up, hold ${'●'.repeat(Math.ceil(held / 300))}`);
    }
    if (phase === 'intro' && held > 600) endIntro();
    if (phase === 'ready') {
      if (held > 1200 || (autopilot && now > autoRestart)) startCountdown();
      else if (tut.on) tutorialTick(now);
      else if (held > 150) hud.msg('RAISE BOTH HANDS', `hold… ${'●'.repeat(Math.ceil(held / 300))}`, 0);
      else if (sig.present && (sig.armL !== sig.armR)) hud.msg('RAISE BOTH HANDS', `${sig.armL ? 'left arm up' : 'left arm down'} · ${sig.armR ? 'right arm up' : 'right arm down'}`, 0);
    }
    // the gauge under the skier: what the game reads, where the player is looking
    updateGauge(now, wallDt);
    if (phase === 'finished' && (held > 1200 || (autopilot && now > autoRestart))) resetRun();

    // phases
    if (phase === 'intro' && now - introStart > INTRO_S * 1000) endIntro();
    if (phase === 'countdown') {
      const left = (countdownEnd - now) / 1000;
      const n = Math.ceil(left);
      if (left <= 0) beginRace();
      else { hud.msg(String(Math.min(3, n)), 'get low', 0); if (3 - n + 1 > countdownBeeps && n <= 3) { countdownBeeps = 3 - n + 1; audio.startBeep(false); } }
    }
    // time: hit-stop and slow motion
    if (freeze > 0) { freeze -= wallDt; timeScale = 0; }
    else if (phase === 'finished' && now < slowmoUntil) timeScale = 0.3;
    else timeScale = 1;
    const dt = wallDt * timeScale;

    if ((phase === 'racing' || phase === 'finished') && dt > 0) {
      const wasAir = state.air, wasCrashed = state.crashed > 0, vzBefore = state.vz;
      // sub-step so the integration never takes more than 1/60 s, whatever the frame rate
      const simS = dt * fast, nSteps = Math.max(1, Math.ceil(simS / (1 / 60))), h = simS / nSteps;
      for (let k = 0; k < nSteps; k++) { if (autopilot && k) autopilotInput(); step(state, inp, h, course); if (phase === 'racing') raceT += h; }
      boostT = Math.max(0, boostT - simS); tuckBoostT = Math.max(0, tuckBoostT - simS);
      // the shield takes a wipeout from over-edging or a hard landing
      if (state.crashed > 0 && !wasCrashed && shield && phase === 'racing') { state.crashed = 0; useShield('saved you'); }
      if (phase === 'racing') {
        const res = gates.update(state.x, state.y, state.sAlong);
        if (res === 'hit') onGateHit(gates.next - 1);
        if (res === 'miss') onGateMiss(gates.next - 1);
        maxSpeed = Math.max(maxSpeed, state.speed);
        hud.clock(raceT * 1000 + adjMs());
        for (const pk of powerups.collect(state.x, state.y, state.z)) onPickup(pk);
        // dataset: 20 Hz of body + ground
        if (raceT - lastTraceT >= 0.05) {
          lastTraceT = raceT;
          chunk.push({ tMs: Math.round(raceT * 1000), x: state.x, y: state.y, z: state.z, speed: state.speed, heading: state.heading, lean: inp.lean, crouch: inp.crouch, kneeL: sig.kneeL, kneeR: sig.kneeR, slopeDeg: state.slopeDeg });
          samples++; flushTrace();
        }
        // Hazard Intelligence: a fresh, confident reading about once a second, registered to the course
        if (raceT - lastVitalT >= 1 && vitals.live && vitals.baseHr && vitals.lastAt > lastVitalAt) {
          lastVitalT = raceT; lastVitalAt = vitals.lastAt;
          vitalBuf.push({ tMs: Math.round(raceT * 1000), x: state.x, y: state.y, s: state.sAlong, hr: vitals.hr, br: vitals.br, rise: vitals.hr / vitals.baseHr - 1 });
          vitalsMapped++; flushVitals();
        }
        // entering a stretch where hearts race
        const hz = hazards.findIndex(b => state.sAlong >= b.s0 && state.sAlong < b.s1 && b.rise >= 0.06 && b.n >= 2);
        if (hz >= 0 && !hazardsSaid.has(hz)) {
          hazardsSaid.add(hz);
          const pct = Math.round(hazards[hz].rise * 100);
          hud.toast(`HAZARD ZONE · hearts +${pct}%`, 1800, hazards[hz].rise >= 0.12 ? 'bad' : 'warn');
          say(pick([`Hazard zone. Hearts race here, ${pct} percent over resting.`, `This is where the pulse climbs, plus ${pct} percent. Careful now.`, `Hazard Intelligence says hearts jump ${pct} percent on this stretch.`]));
        }
        if (gates.crossedFinish(state.x, state.y) && state.progress > 0.9) finishRace();
      }
      // obstacles: trees, the hut, lift towers and finish pillars are a wipeout; the nets catch and bounce you
      if (state.crashed === 0) {
        for (const o of world.nearObstacles(state.x, state.y)) {
          const dx = state.x - o.x, dy = state.y - o.y, d = Math.hypot(dx, dy), minD = o.r + 0.35;
          if (d < minD && shield && phase === 'racing') {
            // the shield bursts through: a shower of snow, a shaken tree, most of the speed kept
            const hit = new THREE.Vector3(o.x, o.y, state.z + 0.9);
            world.burst.emit(hit, POWER_COLOR.shield, 50, 6); world.spray.emit(hit, new THREE.Vector3(0, 0, 2), 50, 3, 0.3);
            if (o.kind === 'tree') world.hitTree(o.idx);
            state.vx *= 0.85; state.vy *= 0.85;
            useShield(o.kind === 'tree' ? 'smashed through a tree' : 'saved you');
            break;
          }
          if (d < minD) {
            const nx = d > 1e-3 ? dx / d : 1, ny = d > 1e-3 ? dy / d : 0;
            state.x = o.x + nx * (minD + 0.05); state.y = o.y + ny * (minD + 0.05);
            const vn = state.vx * nx + state.vy * ny;
            state.vx = (state.vx - 1.6 * vn * nx) * 0.3; state.vy = (state.vy - 1.6 * vn * ny) * 0.3;
            state.crashed = 1.0; state.speed = Math.hypot(state.vx, state.vy);
            const hit = new THREE.Vector3(o.x + nx * o.r, o.y + ny * o.r, state.z + 0.9);
            world.burst.emit(hit, '#ffffff', 40, 6); world.burst.emit(hit, o.kind === 'tree' ? '#3f8a62' : '#9aa4b1', 18, 5);
            world.spray.emit(hit, new THREE.Vector3(nx * 3, ny * 3, 2.5), 40, 3);
            if (o.kind === 'tree') world.hitTree(o.idx);
            trauma = 1; post.hit('#ff3b4e', 0.6); audio.wipeout(); streak = 0; hud.streak(0, STREAK_WORD);
            hud.toast(`Oof, ${({ tree: 'you hit a tree', hut: 'you hit the timing hut', tower: 'you hit a lift tower', rock: 'you hit a rock', lance: 'you hit a snow gun', pillar: 'you hit the finish arch' } as Record<string, string>)[o.kind] ?? 'wipeout'}`, 1600, 'bad');
            say(o.kind === 'tree' ? pick([`Straight into the trees! ${name} is down.`, 'Into the forest, that hurt!']) : pick([`Oh, ${name} is down!`, 'A big crash there!', 'Ooh, that will leave a mark.']), P.big);
            break;
          }
        }
        for (const f of world.nearFences(state.x, state.y)) {
          const ex = f.bx - f.ax, ey = f.by - f.ay, L2 = ex * ex + ey * ey || 1;
          const u = clamp01(((state.x - f.ax) * ex + (state.y - f.ay) * ey) / L2);
          const px = f.ax + ex * u, py = f.ay + ey * u;
          const dx = state.x - px, dy = state.y - py, d = Math.hypot(dx, dy);
          if (d < 0.5) {
            const nx = d > 1e-3 ? dx / d : -ey / Math.sqrt(L2), ny = d > 1e-3 ? dy / d : ex / Math.sqrt(L2);
            const vn = state.vx * nx + state.vy * ny;
            if (vn < 0) { state.vx = (state.vx - 1.5 * vn * nx) * 0.55; state.vy = (state.vy - 1.5 * vn * ny) * 0.55; }
            state.x = px + nx * 0.55; state.y = py + ny * 0.55;
            state.heading = Math.atan2(state.vy, state.vx); state.speed = Math.hypot(state.vx, state.vy); state.edge *= 0.3;
            world.burst.emit(new THREE.Vector3(px, py, state.z + 0.7), '#ff7a1a', 24, 4);
            trauma = Math.max(trauma, 0.5); post.hit('#ff7a1a', 0.3); audio.land(0.8); hud.toast('Into the net!', 1400, 'bad'); say(pick(['Into the safety net!', 'Caught by the net!']));
            break;
          }
        }
      }
      // air and landings
      if (state.air && !wasAir) airStart = raceT;
      if (!state.air && wasAir) { const t = raceT - airStart; const hard = clamp01(-vzBefore / 8); trauma = Math.max(trauma, 0.25 + hard * 0.5); audio.land(hard); if (t > 0.45) { hud.toast(`AIR ${t.toFixed(2)} s`, 1400, 'warn'); if (t > 0.8) say(pick(['Big air!', 'Huge jump!', `Look at that air, ${t.toFixed(1)} seconds!`])); } }
      if (state.crashed > 0 && !wasCrashed) { hud.toast('Wipeout · back on your feet', 1400, 'bad'); audio.wipeout(); trauma = 1; streak = 0; hud.streak(0, STREAK_WORD); say(pick([`Oh, ${name} is down! Back up quickly.`, 'Wipeout! Too much edge there.', 'Down goes the robot!']), P.big); }
    } else if (phase !== 'finished' && phase !== 'racing') { hud.clock(0); state.speed = 0; } // (racing with dt = 0 is a hit-stop: leave the clock and speed alone)
    hud.speed(state.speed);
    if (frame % 6 === 0) hud.gates(gates);
    if (frame % 3 === 0) {
      if (phase === 'racing' && !gates.done) {
        // point at the next gate that is still ahead of you; never ask the player to turn around
        let gi = gates.next, g = course.gates[gi], ang = 0, gx = 0, gy = 0;
        for (let tries = 0; tries < 2 && g; tries++) {
          gx = (g.turn_pole[0] + g.outer_pole[0]) / 2; gy = (g.turn_pole[1] + g.outer_pole[1]) / 2;
          const want = Math.atan2(gy - state.y, gx - state.x);
          ang = Math.atan2(Math.sin(want - state.heading), Math.cos(want - state.heading));
          if (Math.abs(ang) < 1.9) break;
          g = course.gates[++gi];
        }
        if (g) hud.gateArrow(-ang * 180 / Math.PI, Math.hypot(gx - state.x, gy - state.y), Math.abs(ang) > 0.45, g.color); else hud.gateArrow(0, 0, false, 'red');
      } else hud.gateArrow(0, 0, false, 'red');
    }
    audio.update(state.speed, state.slip, state.air, timeScale);

    // network
    net.setState({ x: state.x, y: state.y, z: state.z, heading: state.heading, speed: state.speed, lean: inp.lean, crouch: inp.crouch, air: state.air, phase: phase === 'racing' ? 2 : phase === 'finished' ? 3 : phase === 'lobby' ? 0 : 1, progress: state.progress }, phase === 'racing' ? 8 : 2);

    // pose the G1
    if (useBody && sig.legs && sig.world) { const t = retarget(sig.world); t.waist_roll_joint = 0.35 * sig.lean; poseTarget = t; }
    else { poseTarget = stanceFrom(inp.lean, inp.crouch); if (useBody && sig.armsVisible && sig.world) Object.assign(poseTarget, retargetArms(sig.world)); }
    if (state.air) { poseTarget.left_knee_joint = Math.max(poseTarget.left_knee_joint ?? 1, 1.4); poseTarget.right_knee_joint = Math.max(poseTarget.right_knee_joint ?? 1, 1.4); }
    // a hop off the race (start gate, tutorial): the robot hops on the spot so you can see it was read
    if (inp.jump && phase !== 'racing' && phase !== 'finished') hopAt = now;
    if (inp.jump && (phase === 'racing' || phase === 'ready')) hud.toast('HOP', 600, 'good');
    const hopU = hopAt ? (now - hopAt) / 480 : 1;
    if (hopU < 1) { poseTarget.left_knee_joint = poseTarget.right_knee_joint = 1.5; poseTarget.left_hip_pitch_joint = poseTarget.right_hip_pitch_joint = -1.2; }
    me.apply(poseTarget, 1 - Math.exp(-wallDt * 12));
    placeSkier(me.root, me, state.x, state.y, state.z, state.heading, state.normal, state.edge, inp.crouch, state.air, state.crashed);
    if (hopU < 1) { me.root.position.z += 0.38 * Math.sin(Math.PI * hopU); me.root.updateMatrixWorld(true); me.updatePoles(); }

    // trails and spray
    fwd.set(Math.cos(state.heading), Math.sin(state.heading), 0);
    right.set(fwd.y, -fwd.x, 0);
    tmpN.set(state.normal[0], state.normal[1], state.normal[2]);
    if (!state.air && state.speed > 0.5 && (phase === 'racing' || phase === 'finished') && dt > 0) {
      for (const [ski, side] of [[world.trailL, -1], [world.trailR, 1]] as const) {
        pos.set(state.x, state.y, state.z).addScaledVector(right, side * 0.14);
        pos.z = course.sample(pos.x, pos.y).h;
        ski.push(pos, right, tmpN, 0.045);
      }
      const work = Math.abs(Math.sin(state.edge)) * state.speed;
      if (work > 2) {
        const n = Math.min(18, Math.floor(work * 0.8 * timeScale + 1));
        pos.set(state.x, state.y, state.z).addScaledVector(right, Math.sign(state.edge) * 0.3).addScaledVector(fwd, -0.4);
        tmpA.copy(fwd).multiplyScalar(-state.speed * 0.25).addScaledVector(right, Math.sign(state.edge) * work * 0.4).addScaledVector(up, 1.4 + work * 0.14);
        world.spray.emit(pos, tmpA, n, 1.3 + work * 0.12);
      }
    }

    // the leader's ghost
    updateGhost();

    // others
    for (const g of others.values()) {
      const r = g.row;
      tmpA.set(r.x, r.y, r.z);
      g.pos.lerp(tmpA, 1 - Math.exp(-wallDt * 5));
      g.heading += Math.atan2(Math.sin(r.heading - g.heading), Math.cos(r.heading - g.heading)) * (1 - Math.exp(-wallDt * 6));
      const smp = course.sample(g.pos.x, g.pos.y);
      g.g1.apply(stanceFrom(r.lean, r.crouch), 1 - Math.exp(-wallDt * 8));
      placeSkier(g.g1.root, g.g1, g.pos.x, g.pos.y, r.air ? g.pos.z : smp.h, g.heading, smp.n, r.lean * 0.9, r.crouch, r.air, 0);
      g.label.position.set(g.pos.x, g.pos.y, smp.h + 1.9);
      g.g1.root.visible = r.phase !== 0;
      g.g1.fade(clamp01((g.g1.root.position.distanceTo(camera.position) - 2.5) / 4));
      g.label.visible = r.phase !== 0 && g.label.position.distanceTo(camera.position) > 7; // a name tag right at the lens would fill the screen
    }

    updatePowerFx(wallDt);
    updateVitals(wallDt);
    chatter(now);
    powerups.update(wallDt, state.x, state.y, state.z);
    // camera
    updateCamera(now, wallDt);
    world.follow(state.x, state.y, state.z);
    world.update(wallDt, camera.position, timeScale);
    const speed01 = clamp01((state.speed - 6) / 22);
    post.render(wallDt, phase === 'racing' ? Math.min(1, speed01 + (boostT > 0 ? 0.35 : tuckBoostT > 0 ? 0.2 : 0)) : speed01 * 0.4);
    if (phase === 'finished' && !photoTaken && now > photoDue) takePhoto();
    rec?.tick(now, { phase, raceT, sig, inp, state, gates: { made: gates.made, next: gates.next, missed: gates.missed.filter(Boolean).length }, pose: poseTarget, vit: { hr: vitals.hr, br: vitals.br, arousal: vitals.arousal, mode: vitalMode } });
  }

  function onGateHit(k: number) {
    const g = course.gates[k];
    const col = g.color === 'red' ? '#ff3b4e' : '#3d7bff';
    freeze = 0.04; trauma = Math.max(trauma, 0.22);
    tmpA.set((g.turn_pole[0] + g.outer_pole[0]) / 2, (g.turn_pole[1] + g.outer_pole[1]) / 2, (g.turn_pole[2] + g.outer_pole[2]) / 2 + 0.9);
    world.burst.emit(tmpA, col, 28, 5);
    post.hit(col, 0.28);
    audio.thwack(); if (crowdMeter > 0.3 && Math.random() < crowdMeter) audio.cowbell(0.4);
    world.kickGate(k);
    world.updateGates(gates.next, gates.missed, gates.assist);
    const ms = Math.round(raceT * 1000);
    splits[k] = ms;
    streak++; bestStreak = Math.max(bestStreak, streak);
    hud.streak(streak % (STREAK_WORD.length + 1), STREAK_WORD);
    if (streak % STREAK_WORD.length === 0) { // the word is complete: a push and a shout
      state.vx += Math.cos(state.heading) * 2.2; state.vy += Math.sin(state.heading) * 2.2;
      hud.toast(`${STREAK_WORD}! ${streak} clean gates · boost`, 1600, 'good'); audio.chime(); say(pick([`${streak} clean gates! That's the Streif bonus!`, 'Beautiful rhythm, six in a row!', 'Streif! The bonus kicks in!'])); post.hit('#37e6a8', 0.5);
    }
    const ref = best && best.key !== runKey ? best.splits[k] : 0;
    const delta = ref ? ms - ref : null;
    hud.split(k + 1, ms, delta);
    if ((k + 1) % 6 === 0 && delta !== null && best) say(delta < 0 ? pick([`Gate ${k + 1}, ${Math.abs(delta / 1000).toFixed(1)} up on ${best.name}!`, `Green split at gate ${k + 1}! ${Math.abs(delta / 1000).toFixed(1)} ahead.`]) : pick([`Gate ${k + 1}, ${(delta / 1000).toFixed(1)} down on ${best.name}.`, `Red split, ${(delta / 1000).toFixed(1)} behind at gate ${k + 1}.`]));
  }
  function onGateMiss(k: number) {
    splits[k] = 0;
    if (shield) { forgiven++; useShield(`forgave gate ${k + 1}`); world.updateGates(gates.next, gates.missed, gates.assist); return; }
    streak = 0; hud.streak(0, STREAK_WORD);
    audio.miss(); hud.toast(`Missed gate ${k + 1} · +${MISS_PENALTY_MS / 1000}s`, 1600, 'bad');
    say(pick([`Missed gate ${k + 1}, that's a penalty.`, `Gate ${k + 1} missed. One and a half seconds.`]));
    world.updateGates(gates.next, gates.missed, gates.assist);
    post.hit('#ff3b4e', 0.18);
    $('clock').classList.remove('flash-red'); void $('clock').offsetWidth; $('clock').classList.add('flash-red');
  }

  function onPickup(pk: Pickup) {
    const col = POWER_COLOR[pk.kind];
    tmpA.set(pk.x, pk.y, pk.z + (pk.kind === 'ring' ? 1.9 : 0));
    world.burst.emit(tmpA, col, pk.kind === 'crystal' ? 14 : 36, pk.kind === 'crystal' ? 3 : 6);
    if (pk.kind === 'crystal') {
      const ms = CRYSTAL_MS * (vitalMode === 'ice' ? 2 : 1);
      crystals++; bonusMs += ms; post.hit(col, 0.12);
      hud.pop(`−${(ms / 1000).toFixed(2)}`, col);
      if (crystals % 5 === 0) say(`${crystals} crystals${vitalMode === 'ice' ? ', and with ice in the veins they count double' : ''}.`, P.chatter);
      return;
    }
    post.hit(col, 0.4); trauma = Math.max(trauma, 0.2);
    if (pk.kind === 'ring') {
      boostT = 2.0; state.vx += Math.cos(state.heading) * 0.8; state.vy += Math.sin(state.heading) * 0.8;
      hud.toast('SLIPSTREAM', 1200, 'cyan');
      say(pick(['Into the slipstream!', 'Through the ring, slipstream!', 'Slipstream, and the speed builds!']));
    } else if (pk.kind === 'shield') {
      shield = true; hud.toast('SHIELD · forgives one crash or miss', 1800, 'ice');
      say(pick(['Shield on board.', `${name} picks up the shield.`]));
    } else if (pk.kind === 'magnet') {
      gates.assist = 3; world.updateGates(gates.next, gates.missed, gates.assist);
      hud.toast('MAGNET · next 3 gates are wide', 1800, 'violet');
      say(pick(['Magnet! The gates open up.', 'The magnet, wide gates ahead.']));
    }
  }
  function useShield(what: string) {
    shield = false; trauma = Math.max(trauma, 0.4); post.hit(POWER_COLOR.shield, 0.55);
    tmpA.copy(me.root.position).add(new THREE.Vector3(0, 0, 0.8));
    world.burst.emit(tmpA, POWER_COLOR.shield, 60, 7);
    hud.toast(`Shield ${what}`, 1600, 'ice');
    say(pick([`The shield saves ${name}!`, 'Saved by the shield!']), P.big);
  }
  /** Tuck charge: hold a low, straight tuck and it charges in three steps; stand up to fire it. */
  function tuckCharge(dt: number) {
    if (dt <= 0) return;
    const tucked = inp.crouch > 0.7 && Math.abs(inp.lean) < 0.3 && !state.air && state.crashed === 0 && state.speed > 6;
    if (tucked) {
      tuckHold += dt;
      const lvl = Math.min(3, Math.floor(tuckHold / 0.7));
      if (lvl > tuckLevel) { tuckLevel = lvl; post.hit(['#4da3ff', '#ffb340', '#ff4fd8'][lvl - 1], 0.12); }
    } else if (tuckLevel > 0 && inp.crouch < 0.5 && Math.abs(inp.lean) < 0.6) {
      tuckFired = tuckLevel; tuckBoostT = 1.2; tuckHold = 0; tuckLevel = 0;
      hud.toast(`TUCK BOOST ${'▮'.repeat(tuckFired)}`, 1100, 'warn');
      if (tuckFired >= 2) say(pick(['Out of the tuck, and look at that acceleration!', 'Stands up from the tuck, here comes the speed!']));
      post.hit('#ffb340', 0.25);
    } else if (!tucked) { tuckHold = Math.max(0, tuckHold - dt * 2); tuckLevel = Math.min(tuckLevel, Math.floor(tuckHold / 0.7)); }
    // sparks off the ski tails while charged, in the charge colour
    sparkCd -= dt;
    if (tuckLevel > 0 && sparkCd <= 0) {
      sparkCd = 0.06;
      for (const side of [-1, 1]) {
        tmpB.set(state.x, state.y, state.z + 0.05).addScaledVector(fwd, -0.5).addScaledVector(right, side * 0.14);
        world.burst.emit(tmpB, ['#4da3ff', '#ffb340', '#ff4fd8'][tuckLevel - 1], 2, 1.5);
      }
    }
  }
  /** Presage readings: keep pulse and breathing only when the SDK is confident (stable, or 60 % and up). */
  function onPresage(m: PresageMessage) {
    const calm = phase !== 'racing' && phase !== 'finished';
    if (m.t === 'vitals') {
      const ok = (x: { c: number; s: boolean } | null) => !!x && (x.s || x.c >= 60) && x.c > 0;
      vitals.push({ hr: ok(m.hr) ? m.hr!.v : null, hrConf: m.hr ? m.hr.c / 100 : 0, br: ok(m.br) ? m.br!.v : null, brConf: m.br ? m.br.c / 100 : 0 }, calm);
      if (vitals.status !== 'live') vitals.status = 'measuring your pulse';
    } else if (m.t === 'valid') vitals.hint = m.code === 0 ? '' : PRESAGE_HINT[m.code] ?? m.hint;
    else if (m.t === 'started') vitals.status = 'measuring your pulse';
    else if (m.t === 'error') { vitals.status = `pulse: ${m.message.slice(0, 48)}`; console.warn('[presage]', m); }
    else if (m.t === 'closed' && vitals.status !== 'live' && !vitals.status.startsWith('pulse:')) vitals.status = 'pulse offline';
  }

  function updateVitals(wallDt: number) {
    const racing = phase === 'racing';
    const a = vitals.arousal;
    if (racing && vitals.live) {
      if (vitalMode !== 'steady' && a > 0.6) { vitalMode = 'steady'; if (!saidBreathe) { saidBreathe = true; say(pick([`Heart rate ${Math.round(vitals.hr)}. Breathe, ${name}.`, `The heart is racing, ${Math.round(vitals.hr)} beats a minute. Stay calm.`])); } }
      else if (vitalMode === 'steady' && a < 0.45) vitalMode = '';
      if (vitalMode !== 'steady') vitalMode = raceT > 4 && (vitalMode === 'ice' ? a < 0.25 : a < 0.15) ? 'ice' : '';
      if (vitalMode === 'ice' && !saidIce) { saidIce = true; say(`Ice in the veins! ${Math.round(vitals.hr)} beats a minute at ${Math.round(state.speed * 3.6)} kilometres an hour.`); }
    } else vitalMode = '';
    if (phase === 'ready' && vitals.live && vitals.baseHr && !saidRest) { saidRest = true; say(`Resting heart rate for ${name}, ${Math.round(vitals.baseHr)}.`, P.chatter); }
    gates.ease += ((vitalMode === 'steady' ? 1 : 0) - gates.ease) * (1 - Math.exp(-wallDt * 2));
    const beat = vitals.beat(wallDt);
    me.heart(beat, a);
    post.beat = beat * (racing ? 1 : 0.4);
    post.tunnel += ((racing ? a : 0) - post.tunnel) * (1 - Math.exp(-wallDt * 1.5));
    if (frame % 6 === 0) hud.vitals(phase === 'lobby' || vitals.status === 'off' ? null : { live: vitals.live, status: vitals.hint && !vitals.live ? `${vitals.status} · ${vitals.hint}` : vitals.status, hr: vitals.hr, br: vitals.br, baseHr: vitals.baseHr, arousal: a, mode: vitalMode });
  }

  function updatePowerFx(wallDt: number) {
    const racing = phase === 'racing';
    // the bubble rides on the robot, the aura on the snow under it
    bubble.mesh.visible = racing && shield;
    if (bubble.mesh.visible) { bubble.mesh.position.copy(me.root.position).addScaledVector(up, 0.45); bubble.mat.uniforms.uTime.value += wallDt; bubble.mat.uniforms.uAlpha.value = 0.75 + 0.2 * Math.sin(performance.now() / 180); }
    aura.group.visible = racing && gates.assist > 0;
    if (aura.group.visible) {
      aura.group.position.set(state.x, state.y, state.z + 0.06);
      aura.group.quaternion.setFromUnitVectors(up, tmpN.set(state.normal[0], state.normal[1], state.normal[2]));
      aura.inner.rotation.z += wallDt * 2; aura.outer.rotation.z -= wallDt * 1.2;
    }
    const glowing = racing && (boostT > 0 || tuckBoostT > 0);
    const glowCol = boostT > 0 ? new THREE.Color(POWER_COLOR.ring).multiplyScalar(2.2) : tuckBoostT > 0 ? new THREE.Color(POWER_COLOR.tuck).multiplyScalar(2.0) : new THREE.Color(POWER_COLOR.magnet).multiplyScalar(1.6);
    const gk = glowing ? 1 : racing && gates.assist > 0 ? 0.6 : 0;
    world.trailL.setGlow(gk, glowCol); world.trailR.setGlow(gk, glowCol);
    if (frame % 4 === 0) hud.powers({
      racing,
      boost: boostT > 0 ? boostT / 2 : 0,
      tuckBoost: tuckBoostT > 0 ? tuckBoostT / 1.2 : 0,
      shield: racing && shield,
      magnet: racing ? gates.assist : 0,
      tuck: tuckLevel,
      tuckCharge: Math.min(1, tuckHold / 2.1),
      crystals, crystalTotal,
    });
  }

  function autopilotInput() {
    const cl = course.meta.centerline;
    const look = cl[Math.min(cl.length - 1, state.clIndex + 10)];
    let tx = look[0], ty = look[1];
    if (Math.abs(state.lateral) > 12) { const b = cl[Math.min(cl.length - 1, state.clIndex + 5)]; tx = b[0]; ty = b[1]; }
    else if (!gates.done) { const g = course.gates[gates.next]; if (g.s - state.sAlong < 45) { tx = (g.turn_pole[0] + g.outer_pole[0]) / 2; ty = (g.turn_pole[1] + g.outer_pole[1]) / 2; } }
    let want = Math.atan2(ty - state.y, tx - state.x);
    if (state.speed < 1.5 && !state.air) { const [gx, gy] = course.grad(state.x, state.y); const fall = Math.atan2(-gy, -gx); if (Math.cos(fall - want) < 0.1) want = fall; inp.jump = true; }
    const err = Math.atan2(Math.sin(want - state.heading), Math.cos(want - state.heading));
    inp.lean = Math.max(-1, Math.min(1, -err * 2.5 * Math.min(1, 12 / Math.max(6, state.speed))));
    inp.crouch = Math.abs(inp.lean) < 0.25 ? 0.9 : 0.3;
    inp.push = Math.max(0, (inp.crouch - 0.45) / 0.55);
  }

  function updateGhost() {
    if (phase === 'ready' && tut.on) { demoGhost(); return; }
    if (phase !== 'racing' || ghostSamples.length < 2) { ghostGap = null; ghost.root.visible = ghostLabel.visible = false; if (phase !== 'racing') hud.ghost('', ''); return; }
    const t = raceT * 1000;
    let lo = 0, hi = ghostSamples.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (ghostSamples[m].tMs < t) lo = m + 1; else hi = m; }
    const b = ghostSamples[Math.max(1, lo)], a = ghostSamples[Math.max(0, lo - 1)];
    const u = b.tMs > a.tMs ? clamp01((t - a.tMs) / (b.tMs - a.tMs)) : 1;
    const gx = a.x + (b.x - a.x) * u, gy = a.y + (b.y - a.y) * u;
    const smp = course.sample(gx, gy);
    const gz = Math.max(smp.h, a.z + (b.z - a.z) * u);
    const gh = a.heading + Math.atan2(Math.sin(b.heading - a.heading), Math.cos(b.heading - a.heading)) * u;
    ghost.apply(stanceFrom(a.lean, a.crouch), 0.3);
    placeSkier(ghost.root, ghost, gx, gy, gz, gh, smp.n, a.lean * 0.9, a.crouch, gz > smp.h + 0.2, 0);
    ghost.root.visible = t <= ghostSamples[ghostSamples.length - 1].tMs + 500;
    ghost.fade(clamp01((ghost.root.position.distanceTo(camera.position) - 2.5) / 4));
    ghostLabel.position.set(gx, gy, smp.h + 1.9);
    ghostLabel.visible = ghost.root.visible && ghostLabel.position.distanceTo(camera.position) > 7;
    // along-course gap
    const gs = course.nearest(gx, gy, Math.max(0, state.clIndex - 30), 80).s;
    const gap = state.sAlong - gs;
    ghostGap = gap;
    if (Math.abs(gap) < 0.5) hud.ghost(`level with <b>${best?.name ?? 'leader'}</b>`, '');
    else hud.ghost(`${gap > 0 ? '▲' : '▼'} <b>${Math.abs(gap).toFixed(1)} m</b> ${gap > 0 ? 'ahead of' : 'behind'} ${best?.name ?? 'leader'}`, gap > 0 ? 'ahead' : 'behind');
  }

  function tutorialTick(now: number) {
    const step = TUT[tut.step];
    if (!step) { // all done: the usual start prompt
      hud.msg('RAISE BOTH HANDS', `${TUT.map(s => '✓ ' + s.title.toLowerCase()).join(' · ')} · hold both arms up to start`, 0);
      return;
    }
    if (body.landmarker && !sig0.present && !keys.size) { hud.msg(step.title, 'step into frame, face the camera', 0); tut.holdSince = 0; return; }
    const ok = step.test();
    if (ok) { if (!tut.holdSince) tut.holdSince = now; } else if (!step.event) tut.holdSince = 0;
    const need = step.event ? 0 : 500;
    const progress = tut.holdSince ? Math.min(1, need ? (now - tut.holdSince) / need : 1) : 0;
    const list = TUT.map((s, i) => (i < tut.step ? '✓ ' : i === tut.step ? '▶ ' : '○ ') + s.title.toLowerCase()).join(' · ');
    hud.msg(step.title, `${step.sub}${progress > 0 ? ' ' + '●'.repeat(Math.ceil(progress * 3)) : ''}\n${list}`, 0);
    if (step.title.startsWith('LEAN') && sig0.present) tutLeanMax = Math.max(tutLeanMax, Math.abs(sig0.rawLean));
    if (tut.holdSince && now - tut.holdSince >= need) {
      if (tut.step === 1 && tutLeanMax > 0.08) body.learnLeanPeak(tutLeanMax); // after both lean steps
      tut.step++; tut.holdSince = 0; audio.chime(); post.hit('#37e6a8', 0.3); world.burst.emit(new THREE.Vector3(state.x, state.y, state.z + 1), '#37e6a8', 20, 3);
      if (tut.step >= TUT.length) { audio.say('Nice. Raise both hands when you are ready.'); }
      else audio.say(TUT[tut.step].title.toLowerCase());
    }
  }

  function demoGhost() {
    const step = TUT[Math.min(tut.step, TUT.length - 1)];
    const r = new THREE.Vector3(Math.sin(state.heading), -Math.cos(state.heading), 0);
    const gx = state.x + r.x * 2.4, gy = state.y + r.y * 2.4;
    const smp = course.sample(gx, gy);
    const tt = performance.now() / 1000;
    let pose = stanceFrom(step.pose.lean, step.pose.crouch);
    if (tut.step >= TUT.length) { pose = stanceFrom(0, 0.1); pose.left_shoulder_pitch_joint = pose.right_shoulder_pitch_joint = -2.6; pose.left_elbow_joint = pose.right_elbow_joint = 0.3; }
    else if (step.title.startsWith('PLANT')) { const sw = Math.sin(tt * 6); pose.left_shoulder_pitch_joint = pose.right_shoulder_pitch_joint = -1.0 + 0.9 * sw; pose.left_elbow_joint = pose.right_elbow_joint = 0.8 - 0.4 * sw; }
    ghost.apply(pose, 0.25); ghost.fade(1);
    placeSkier(ghost.root, ghost, gx, gy, smp.h, state.heading, smp.n, step.pose.lean * 0.9, step.pose.crouch, false, 0);
    if (step.title === 'HOP' && tut.step < TUT.length) { const u = (tt % 1.4) / 0.5; if (u < 1) { ghost.root.position.z += 0.4 * Math.sin(Math.PI * u); ghost.root.updateMatrixWorld(true); ghost.updatePoles(); } }
    ghost.root.visible = true; ghostLabel.position.set(gx, gy, smp.h + 1.9); ghostLabel.visible = ghostLabel.position.distanceTo(camera.position) > 7;
    hud.ghost('', '');
  }

  function updateGauge(now: number, wallDt: number) {
    const show = phase === 'ready' || phase === 'countdown' || (phase === 'racing' && raceT < 8);
    gauge.group.visible = show || gauge.fade > 0.02;
    gauge.fade += ((show ? 1 : 0) - gauge.fade) * (1 - Math.exp(-wallDt * 3));
    if (!gauge.group.visible) return;
    const n = new THREE.Vector3(state.normal[0], state.normal[1], state.normal[2]).lerp(new THREE.Vector3(0, 0, 1), 0.3).normalize();
    const f = new THREE.Vector3(Math.cos(state.heading), Math.sin(state.heading), 0); f.addScaledVector(n, -f.dot(n)).normalize();
    const l = new THREE.Vector3().crossVectors(n, f);
    gauge.group.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(f, l, n));
    gauge.group.position.set(state.x, state.y, state.z).addScaledVector(n, 0.04);
    gauge.needle.rotation.z = -inp.lean * 1.0;
    const col = !body.landmarker ? '#37e6a8' : !sig0.present ? '#8a94a6' : sig0.calibrated ? '#37e6a8' : '#ffb340';
    (gauge.ring.material as THREE.MeshBasicMaterial).color.set(col); (gauge.needle.material as THREE.MeshBasicMaterial).color.set(col);
    (gauge.ring.material as THREE.MeshBasicMaterial).opacity = 0.55 * gauge.fade; (gauge.needle.material as THREE.MeshBasicMaterial).opacity = 0.9 * gauge.fade;
    gauge.tuck.scale.set(1, Math.max(0.01, inp.crouch), 1); (gauge.tuck.material as THREE.MeshBasicMaterial).opacity = 0.8 * gauge.fade;
  }

  function updateCamera(now: number, wallDt: number) {
    const sp = clamp01(state.speed / 26);
    trauma = Math.max(0, trauma - wallDt * 1.6);
    const sh = trauma * trauma;
    shake.set(Math.sin(now * 0.031) * sh * 0.25, Math.sin(now * 0.047 + 1) * sh * 0.2, Math.sin(now * 0.053 + 2) * sh * 0.18);
    fwd.set(Math.cos(state.heading), Math.sin(state.heading), 0);
    right.set(fwd.y, -fwd.x, 0);
    let rate = 4.5, lookRate = 7;
    if (phase === 'lobby') {
      // the start screen: a slow orbit round the robot in the start gate, which copies whoever is in front of the camera
      const sway = Math.sin(now * 0.00013);
      goal.set(state.x, state.y, state.z).addScaledVector(fwd, -4.2 + sway * 0.8).addScaledVector(right, -2.6 + sway * 1.2).addScaledVector(up, 2.1 + Math.sin(now * 0.0002) * 0.3);
      goalLook.set(state.x, state.y, state.z).addScaledVector(fwd, 14).addScaledVector(right, -3.2).addScaledVector(up, -1.5);
      rate = 2; lookRate = 3;
    } else if (phase === 'intro') {
      const u = ease(clamp01((now - introStart) / (INTRO_S * 1000)));
      const cl = course.meta.centerline;
      const a = cl[course.finishIndex], b = cl[course.startIndex];
      const mid = cl[Math.floor((course.startIndex + course.finishIndex) / 2)];
      // sweep from high above the finish up the course to behind the start
      goal.set(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, 0);
      const h = course.headingAt(course.startIndex);
      goal.addScaledVector(new THREE.Vector3(Math.cos(h), Math.sin(h), 0), -40 * (1 - u) - 9 * u);
      goal.z = course.sample(goal.x, goal.y).h + 70 * (1 - u) + 4 * u;
      goalLook.set(mid[0] + (b[0] - mid[0]) * u, mid[1] + (b[1] - mid[1]) * u, mid[2] + (b[2] - mid[2]) * u + 1);
      rate = 3; lookRate = 3;
      if (!camInit) { camPos.copy(goal); camLook.copy(goalLook); camInit = true; }
    } else if (phase === 'finished') {
      const u = (now - orbitStart) / 1000;
      const ang = state.heading + Math.PI + 0.35 + u * 0.35;
      const r = 6.5 + Math.sin(u * 0.5) * 1.0;
      goal.set(state.x + Math.cos(ang) * r, state.y + Math.sin(ang) * r, state.z + 2.2 + Math.sin(u * 0.4) * 0.6);
      goalLook.set(state.x, state.y, state.z + 0.8);
      rate = 2.5; lookRate = 4;
    } else if ((phase === 'ready' || phase === 'countdown') && camMode === 'chase') {
      // getting ready: face the skier so the player sees the robot copy them
      goal.set(state.x, state.y, state.z).addScaledVector(fwd, 3.6).addScaledVector(right, -1.6).addScaledVector(up, 1.25);
      goalLook.set(state.x, state.y, state.z).addScaledVector(up, 0.7);
      rate = 3;
    } else if (camMode === 'chase') {
      goal.set(state.x, state.y, state.z).addScaledVector(fwd, -3.9 - sp * 2.4).addScaledVector(right, 0.6).addScaledVector(up, 1.45 + sp * 0.7);
      goalLook.set(state.x, state.y, state.z).addScaledVector(fwd, 6).addScaledVector(up, 0.7);
    } else if (camMode === 'front') {
      goal.set(state.x, state.y, state.z).addScaledVector(fwd, 5.2).addScaledVector(right, -1.1).addScaledVector(up, 1.45);
      goalLook.set(state.x, state.y, state.z).addScaledVector(up, 0.8);
    } else {
      goal.set(state.x, state.y, state.z).addScaledVector(right, 6.5).addScaledVector(fwd, 1.5).addScaledVector(up, 1.5);
      goalLook.set(state.x, state.y, state.z).addScaledVector(up, 0.7);
    }
    goal.z = Math.max(goal.z, course.sample(goal.x, goal.y).h + 0.7);
    if (!camInit) { camPos.copy(goal); camLook.copy(goalLook); camInit = true; }
    camPos.lerp(goal, 1 - Math.exp(-wallDt * rate));
    camLook.lerp(goalLook, 1 - Math.exp(-wallDt * lookRate));
    camera.position.copy(camPos).add(shake);
    camera.lookAt(camLook);
    const fovGoal = phase === 'intro' ? 58 : 60 + 25 * sp * sp + (phase === 'racing' && (boostT > 0 || tuckBoostT > 0) ? 8 : 0);
    camera.fov += (fovGoal - camera.fov) * (1 - Math.exp(-wallDt * 6)); camera.updateProjectionMatrix();
    camera.rotateZ(-state.edge * 0.07 + shake.z * 0.3);
  }

  function takePhoto() {
    photoTaken = true;
    try {
      const g = photoCanvas.getContext('2d')!;
      g.drawImage(renderer.domElement, 0, 0, photoCanvas.width, photoCanvas.height);
      g.fillStyle = 'rgba(8,14,28,.6)'; g.fillRect(0, 300, 640, 60);
      g.fillStyle = '#fff'; g.font = '900 28px system-ui'; g.fillText(fmtTime(Math.max(1000, Math.round(raceT * 1000) + adjMs())), 16, 340);
      g.font = '700 16px system-ui'; g.fillStyle = '#37e6a8'; g.fillText(`${name} · ${courseName} · Hazard Intelligence`, 180, 338);
      const data = photoCanvas.toDataURL('image/jpeg', 0.62);
      net.pushPhoto(runKey, data.split(',')[1] ?? '');
    } catch (e) { console.warn('photo', e); }
  }

  /** Put a G1 on the snow: yaw from heading, up from the terrain normal, roll into the turn, skis on the surface. */
  function placeSkier(root: THREE.Object3D, g1: G1, x: number, y: number, z: number, heading: number, normal: readonly number[], edge: number, crouch: number, air: boolean, crashed: number) {
    tmpN.set(normal[0], normal[1], normal[2]).lerp(up, 0.45).normalize();
    fwd.set(Math.cos(heading), Math.sin(heading), 0);
    fwd.addScaledVector(tmpN, -fwd.dot(tmpN)).normalize();
    tmpB.crossVectors(tmpN, fwd);
    const m = new THREE.Matrix4().makeBasis(fwd, tmpB, tmpN);
    root.quaternion.setFromRotationMatrix(m);
    root.rotateX(-edge * 0.55);
    root.rotateY(0.08 + 0.12 * crouch + (air ? -0.15 : 0));
    if (crashed > 0) { const k = Math.min(1, (1.0 - crashed) * 2.5); root.rotateY(1.2 * k); root.rotateX(0.6 * k); }
    root.position.set(x, y, z);
    root.updateMatrixWorld(true);
    let lowest = Infinity;
    for (const ski of g1.skis) { ski.getWorldPosition(tmpA); const d = tmpA.sub(root.position).dot(tmpN) - 0.01; lowest = Math.min(lowest, d); }
    if (isFinite(lowest)) root.position.addScaledVector(tmpN, -lowest + (crashed > 0 ? -0.25 : 0));
    root.updateMatrixWorld(true);
    g1.updatePoles();
  }

  status('ready');
  Object.defineProperty(window, '__gt', { value: { get phase() { return phase; }, get state() { return state; }, get gates() { return { next: gates.next, made: gates.made, total: gates.total }; }, get net() { return net.status; }, get raceT() { return raceT; }, get obstacles() { return world.obstacles; }, get fences() { return world.fences; }, get course() { return course; }, get inp() { return inp; }, get world() { return world; }, get powers() { return { boostT, shield, crystals, bonusMs, forgiven, assist: gates.assist, tuckLevel, tuckBoostT }; } } });
  tick();
}

async function loadFar(slug: string): Promise<FarTerrain | null> {
  try {
    const [meta, bin] = await Promise.all([fetch(`/courses/${slug}/far.json`).then(r => r.ok ? r.json() : null), fetch(`/courses/${slug}/far.bin`).then(r => r.ok ? r.arrayBuffer() : null)]);
    if (!meta || !bin) return null;
    return { ...meta, z: new Float32Array(bin) };
  } catch { return null; }
}

function flattenTrace(chunks: TraceChunk[]): PoseSample[] {
  const out: PoseSample[] = [];
  for (const c of chunks) for (const s of c.samples) out.push(s);
  out.sort((a, b) => a.tMs - b.tMs);
  return out;
}

function spokenTime(ms: number) {
  const m = Math.floor(ms / 60000), s = (ms % 60000) / 1000;
  return m ? `${m} minute${m > 1 ? 's' : ''} ${s.toFixed(1)}` : `${s.toFixed(1)} seconds`;
}

function shortName(course: Course) {
  const r = course.meta.run || course.meta.slug;
  return r.split(/[+(]/)[0].replace(/-?Rennstrecke|Familienabfahrt/g, '').trim() || course.meta.slug;
}

/** A ring on the snow under the skier with a needle for lean and a bar for tuck. */
function makeGauge() {
  const group = new THREE.Group();
  const ringMat = new THREE.MeshBasicMaterial({ color: '#37e6a8', transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.8, 0.9, 48, 1, Math.PI * 0.15, Math.PI * 0.7), ringMat); // an arc ahead of the skier
  ring.rotation.z = -Math.PI / 2 + Math.PI * 0.0; // arc centred on +x (forward)
  const needle = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.07, 0.01).translate(0.45, 0, 0), new THREE.MeshBasicMaterial({ color: '#37e6a8', transparent: true, opacity: 0.9, depthWrite: false }));
  const tuck = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.6, 0.01).translate(0, 0.3, 0), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.8, depthWrite: false }));
  tuck.position.set(-0.6, -0.3, 0);
  group.add(ring, needle, tuck);
  group.renderOrder = 5;
  return { group, ring, needle, tuck, fade: 0 };
}

function makeLabel(text: string, color: string) {
  const c = document.createElement('canvas'); c.width = 256; c.height = 64;
  const g = c.getContext('2d')!;
  g.fillStyle = 'rgba(8,14,28,.72)'; roundRect(g, 4, 8, 248, 48, 14); g.fill();
  g.fillStyle = color; g.beginPath(); g.arc(30, 32, 10, 0, Math.PI * 2); g.fill();
  g.fillStyle = '#fff'; g.font = '700 28px system-ui,sans-serif'; g.textBaseline = 'middle'; g.fillText(text.slice(0, 14), 50, 33);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  s.scale.set(2.2, 0.55, 1);
  return s;
}
function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

main().catch(e => { console.error(e); const s = document.getElementById('status'); if (s) s.textContent = `error: ${e?.message ?? e}`; });
