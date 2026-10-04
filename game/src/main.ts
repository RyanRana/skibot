// Ground Truth: Ski. Your body steers a Unitree G1 down the real Streif; everyone on the mountain is
// live through SpacetimeDB; every run becomes motion-on-terrain data. This file is the game loop:
// phases, camera, feel (hit-stop, shake, slow-mo), TV splits, the leader's ghost, cheers and the finish.
import * as THREE from 'three';
import { loadCourse, type Course } from './course.ts';
import { initialState, step, GateTracker, type SkierState, type Input } from './physics.ts';
import { World, type FarTerrain } from './world.ts';
import { Post } from './post.ts';
import { loadG1Assets, G1, retarget, stanceFrom, SKI_STANCE } from './g1.ts';
import { BodyTracker } from './pose.ts';
import { Net, fmtTime, fmtDelta, fmtInt, colorOf, type Skier, type PoseSample, type Run, type TraceChunk } from './net.ts';
import { Hud } from './hud.ts';
import { Audio } from './audio.ts';

const COURSE_SLUG = 'kitzbuhel-streif';
const params = new URLSearchParams(location.search);
const MISS_PENALTY_MS = 1500;
const STREAK_WORD = 'STREIF';
type Phase = 'lobby' | 'intro' | 'ready' | 'countdown' | 'racing' | 'finished';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const ease = (t: number) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

async function main() {
  const hud = new Hud();
  const status = (s: string) => { $('status').textContent = s; };
  const agentNumber = import.meta.env.VITE_AGENT_NUMBER as string | undefined;
  if (agentNumber) $('agent-number').textContent = agentNumber;
  const lowfx = params.has('lowfx');
  const noCam = params.has('nocam');
  const autopilot = params.has('autopilot');
  const fast = Math.max(1, Math.min(8, Number(params.get('fast') || 1)));

  // ---- renderer ----
  const canvas = $('gl') as HTMLCanvasElement;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(lowfx ? 1 : Math.min(devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = !lowfx; renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.95;
  const camera = new THREE.PerspectiveCamera(62, innerWidth / innerHeight, 0.3, 30000);
  camera.up.set(0, 0, 1);

  status('loading the Streif…');
  const lenParam = params.get('length');
  const raceLength = lenParam === 'full' ? 1e9 : Number(lenParam) || 650;
  const [course, g1Assets, far] = await Promise.all([loadCourse(COURSE_SLUG, '/courses', 90, raceLength), loadG1Assets(), loadFar(COURSE_SLUG)]);
  const courseName = shortName(course);
  const dropM = course.meta.centerline[course.startIndex][2] - course.meta.centerline[course.finishIndex][2];
  const startAlt = course.meta.z_datum_msl + course.meta.centerline[course.startIndex][2];
  hud.course(course.meta.resort, courseName, `${course.raceLength.toFixed(0)} m · ${dropM.toFixed(0)} m drop · ${course.gates.length} gates · from ${startAlt.toFixed(0)} m`);
  const world = new World(course, far, lowfx);
  const post = new Post(renderer, world.scene, camera, lowfx);
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
  let camMode: 'chase' | 'front' | 'side' = (params.get('cam') as any) || 'chase';
  let best: Run | null = null; let ghostSamples: PoseSample[] = []; let ghostKey = '';
  let autoRestart = 0;
  const keys = new Set<string>();
  let keyJump = false;

  const net = new Net({
    status: s => status(`SpacetimeDB: ${s}`),
    skier: (kind, row) => { if (!net.isMe(row.identity)) ghostEvent(kind, row); },
    cheer: c => onCheer(c.fromName, c.toName, c.kind),
    changed: () => refreshLists(),
  });
  net.connect();

  const audio = new Audio();
  const video = $('cam') as HTMLVideoElement;
  const body = new BodyTracker(video);
  const camCtx = ($('cam-canvas') as HTMLCanvasElement).getContext('2d')!;
  const attractCtx = ($('attract-canvas') as HTMLCanvasElement).getContext('2d')!;
  let camStarted = false;
  async function startCamera() {
    if (camStarted || noCam) return; camStarted = true;
    try { await body.start(); hud.camLabel('camera on'); } catch (err) { console.warn(err); hud.camLabel('no camera: keyboard'); }
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

  function refreshLists() {
    const runs = net.runs(courseName);
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
    if (ch.length) setTimeout(() => hud.toast(`Challenge from ${ch[0].fromName}: beat ${fmtTime(ch[0].targetTimeMs)}`, 5000), INTRO_S * 1000);
  }
  $('join-form').addEventListener('submit', e => { e.preventDefault(); join(($('name') as HTMLInputElement).value, ($('code') as HTMLInputElement).value); });
  if (params.get('name')) ($('name') as HTMLInputElement).value = params.get('name')!;
  if (params.get('code')) ($('code') as HTMLInputElement).value = params.get('code')!;
  if (params.has('auto')) setTimeout(() => ($('join-form') as HTMLFormElement).requestSubmit(), 300);

  addEventListener('keydown', e => {
    keys.add(e.key);
    if (e.key === ' ') { keyJump = true; e.preventDefault(); }
    if (e.key === 'Enter') { if (phase === 'intro') endIntro(); else if (phase === 'ready') startCountdown(); else if (phase === 'finished') resetRun(); }
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
    hud.intro(true, courseName, `${course.meta.resort} · the real Hahnenkamm`, `${course.raceLength.toFixed(0)} m · ${dropM.toFixed(0)} m drop · ${course.gates.length} gates · start at ${startAlt.toFixed(0)} m`);
    audio.say(`Welcome to the ${courseName}. ${name}, to the start.`, true);
    camInit = false;
  }
  function endIntro() { hud.intro(false); goReady(); }
  function goReady() {
    phase = 'ready'; state = initialState(course); gates = new GateTracker(course);
    world.trailL.reset(); world.trailR.reset(); world.updateGates(0, gates.missed);
    hud.clearSplits(); hud.streak(0, STREAK_WORD); hud.ghost('', ''); hud.result(null);
    streak = 0; cheersThisRun = 0; post.dim = 0; timeScale = 1;
    hud.msg('RAISE BOTH HANDS', 'hold them up for a second, or press Enter', 0);
    best = net.bestRun(courseName, gates.total);
    if (best && best.key !== ghostKey) { if (ghostKey) net.unwatchTrace(ghostKey); ghostKey = best.key; ghostSamples = []; net.watchTrace(ghostKey, chunks => { ghostSamples = flattenTrace(chunks); }); }
    ghost.root.visible = ghostLabel.visible = false;
    autoRestart = performance.now() + 4000;
  }
  function startCountdown() { phase = 'countdown'; countdownEnd = performance.now() + 3300; countdownBeeps = 0; audio.start(); hud.hideMsg(); }
  function beginRace() {
    phase = 'racing';
    state = initialState(course); gates = new GateTracker(course);
    raceT = 0; maxSpeed = 0; traceSeq = 0; samples = 0; chunk = []; lastTraceT = 0; splits = []; streak = 0; bestStreak = 0; photoTaken = false;
    runStartWall = performance.now();
    runKey = `${name}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
    world.trailL.reset(); world.trailR.reset();
    net.startRun(runKey, courseName, gates.total);
    hud.msg('GO', '', 600);
    audio.startBeep(true);
    post.hit('#37e6a8', 0.5);
  }
  function resetRun() { if (ghostKey && phase === 'finished') { /* keep watching */ } goReady(); }
  function flushTrace(final = false) {
    if (chunk.length && (chunk.length >= 50 || final)) { net.pushTrace(runKey, traceSeq++, chunk); chunk = []; }
  }
  function finishRace() {
    phase = 'finished'; finishedAt = performance.now();
    flushTrace(true);
    const penalty = gates.missed.filter(Boolean).length * MISS_PENALTY_MS;
    const timeMs = Math.round(raceT * 1000) + penalty;
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
    hud.result({ timeMs, rank: faster + 1, of: runs.length + (runs.some(r => r.key === runKey) ? 0 : 1), deltaMs: deltaMs === null || deltaMs <= 0 ? (leader && leader.key !== runKey ? deltaMs : null) : deltaMs, gates: `${gates.made}/${gates.total}`, kmh: Math.round(maxSpeed * 3.6), samples, cheers: cheersThisRun, best: !!myBest && timeMs < myBest.timeMs });
    const place = ['first', 'second', 'third'][faster] ?? `${faster + 1}th`;
    setTimeout(() => audio.say(`${name}. ${spokenTime(timeMs)}. ${place} place.${deltaMs !== null && deltaMs > 0 ? ` ${(deltaMs / 1000).toFixed(1)} seconds off the lead.` : ''}`, true), 600);
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
  const inp: Input = { lean: 0, crouch: 0, jump: false };
  const photoCanvas = document.createElement('canvas'); photoCanvas.width = 640; photoCanvas.height = 360;

  function tick() {
    requestAnimationFrame(tick);
    const now = performance.now();
    const wallDt = Math.min((now - lastNow) / 1000, 0.05); lastNow = now;
    frame++;
    crowdMeter = Math.max(0, crowdMeter - wallDt * 0.04);

    // body
    const sig = body.update(now);
    if (frame % 2 === 0) { body.draw(camCtx); if (phase === 'lobby') body.draw(attractCtx); }
    hud.camLabel(body.landmarker ? `${body.status}${sig.fps ? ` · ${sig.fps} fps` : ''}` : (noCam ? 'keyboard mode' : body.status));

    // input: body first, keys override
    const kl = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
    const kc = keys.has('ArrowDown') ? 1 : 0;
    const useBody = sig.present && sig.calibrated;
    inp.lean = kl !== 0 ? kl : useBody ? sig.lean : 0;
    inp.crouch = kc ? 1 : useBody ? sig.crouch : 0;
    inp.jump = keyJump || (useBody && sig.jump); keyJump = false;
    if (autopilot && (phase === 'racing' || phase === 'finished')) autopilotInput();
    hud.meters(inp.lean, inp.crouch);

    // hands up: start (held ~1.2 s) from lobby / ready / finished, skip the intro
    if (sig.handsUp) { if (!handsSince) handsSince = now; } else handsSince = 0;
    const held = handsSince ? now - handsSince : 0;
    if (phase === 'lobby' && held > 1800) { join(($('name') as HTMLInputElement).value || `Skier ${Math.floor(Math.random() * 90 + 10)}`, ($('code') as HTMLInputElement).value); handsSince = 0; }
    if (phase === 'intro' && held > 600) endIntro();
    if (phase === 'ready') {
      if (held > 1200 || (autopilot && now > autoRestart)) startCountdown();
      else if (held > 150) hud.msg('RAISE BOTH HANDS', `hold… ${'●'.repeat(Math.ceil(held / 300))}`, 0);
    }
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
      if (phase === 'racing') {
        const res = gates.update(state.x, state.y, state.sAlong);
        if (res === 'hit') onGateHit(gates.next - 1);
        if (res === 'miss') onGateMiss(gates.next - 1);
        maxSpeed = Math.max(maxSpeed, state.speed);
        hud.clock(raceT * 1000 + gates.missed.filter(Boolean).length * MISS_PENALTY_MS);
        // dataset: 20 Hz of body + ground
        if (raceT - lastTraceT >= 0.05) {
          lastTraceT = raceT;
          chunk.push({ tMs: Math.round(raceT * 1000), x: state.x, y: state.y, z: state.z, speed: state.speed, heading: state.heading, lean: inp.lean, crouch: inp.crouch, kneeL: sig.kneeL, kneeR: sig.kneeR, slopeDeg: state.slopeDeg });
          samples++; flushTrace();
        }
        if (gates.crossedFinish(state.x, state.y) && state.progress > 0.9) finishRace();
      }
      // air and landings
      if (state.air && !wasAir) airStart = raceT;
      if (!state.air && wasAir) { const t = raceT - airStart; const hard = clamp01(-vzBefore / 8); trauma = Math.max(trauma, 0.25 + hard * 0.5); audio.land(hard); if (t > 0.45) { hud.toast(`AIR ${t.toFixed(2)} s`); if (t > 1.0) audio.say('big air'); } }
      if (state.crashed > 0 && !wasCrashed) { hud.msg('OOF', 'back on your feet', 900); audio.wipeout(); trauma = 1; streak = 0; hud.streak(0, STREAK_WORD); }
    } else if (phase !== 'finished') { hud.clock(0); state.speed = 0; }
    hud.speed(state.speed);
    if (frame % 6 === 0) hud.gates(gates);
    audio.update(state.speed, state.slip, state.air, timeScale);

    // network
    net.setState({ x: state.x, y: state.y, z: state.z, heading: state.heading, speed: state.speed, lean: inp.lean, crouch: inp.crouch, air: state.air, phase: phase === 'racing' ? 2 : phase === 'finished' ? 3 : phase === 'lobby' ? 0 : 1, progress: state.progress }, phase === 'racing' ? 8 : 2);

    // pose the G1
    if (useBody && sig.world) { const t = retarget(sig.world); t.waist_roll_joint = 0.35 * sig.lean; poseTarget = t; }
    else poseTarget = stanceFrom(inp.lean, inp.crouch);
    if (state.air) { poseTarget.left_knee_joint = Math.max(poseTarget.left_knee_joint ?? 1, 1.4); poseTarget.right_knee_joint = Math.max(poseTarget.right_knee_joint ?? 1, 1.4); }
    me.apply(poseTarget, 1 - Math.exp(-wallDt * 12));
    placeSkier(me.root, me, state.x, state.y, state.z, state.heading, state.normal, state.edge, inp.crouch, state.air, state.crashed);

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
      g.g1.root.visible = g.label.visible = r.phase !== 0;
    }

    // camera
    updateCamera(now, wallDt);
    world.follow(state.x, state.y, state.z);
    world.update(wallDt, camera.position, timeScale);
    const speed01 = clamp01((state.speed - 6) / 22);
    post.render(wallDt, phase === 'racing' ? speed01 : speed01 * 0.4);
    if (phase === 'finished' && !photoTaken && now > photoDue) takePhoto();
  }

  function onGateHit(k: number) {
    const g = course.gates[k];
    const col = g.color === 'red' ? '#ff3b4e' : '#3d7bff';
    freeze = 0.04; trauma = Math.max(trauma, 0.22);
    tmpA.set((g.turn_pole[0] + g.outer_pole[0]) / 2, (g.turn_pole[1] + g.outer_pole[1]) / 2, (g.turn_pole[2] + g.outer_pole[2]) / 2 + 0.9);
    world.burst.emit(tmpA, col, 28, 5);
    post.hit(col, 0.28);
    audio.thwack(); if (crowdMeter > 0.3 && Math.random() < crowdMeter) audio.cowbell(0.4);
    world.updateGates(gates.next, gates.missed);
    const ms = Math.round(raceT * 1000);
    splits[k] = ms;
    streak++; bestStreak = Math.max(bestStreak, streak);
    hud.streak(streak % (STREAK_WORD.length + 1), STREAK_WORD);
    if (streak % STREAK_WORD.length === 0) { // the word is complete: a push and a shout
      state.vx += Math.cos(state.heading) * 2.2; state.vy += Math.sin(state.heading) * 2.2;
      hud.msg(STREAK_WORD + '!', `${streak} clean gates · boost`, 1100); audio.chime(); audio.say('Streif! Boost.'); post.hit('#37e6a8', 0.5);
    }
    const ref = best && best.key !== runKey ? best.splits[k] : 0;
    const delta = ref ? ms - ref : null;
    hud.split(k + 1, ms, delta, best && delta !== null ? best.name : '');
    if ((k + 1) % 6 === 0 && delta !== null) audio.say(`Gate ${k + 1}. ${Math.abs(delta / 1000).toFixed(1)} ${delta < 0 ? 'up' : 'down'}.`);
  }
  function onGateMiss(k: number) {
    splits[k] = 0; streak = 0; hud.streak(0, STREAK_WORD);
    audio.miss(); hud.toast(`Missed gate ${k + 1} · +${MISS_PENALTY_MS / 1000}s`);
    world.updateGates(gates.next, gates.missed);
    post.hit('#ff3b4e', 0.18);
    $('clock').classList.remove('flash-red'); void $('clock').offsetWidth; $('clock').classList.add('flash-red');
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
  }

  function updateGhost() {
    if (phase !== 'racing' || ghostSamples.length < 2) { ghost.root.visible = ghostLabel.visible = false; if (phase !== 'racing') hud.ghost('', ''); return; }
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
    ghost.root.visible = ghostLabel.visible = t <= ghostSamples[ghostSamples.length - 1].tMs + 500;
    ghostLabel.position.set(gx, gy, smp.h + 1.9);
    // along-course gap
    const gs = course.nearest(gx, gy, Math.max(0, state.clIndex - 30), 80).s;
    const gap = state.sAlong - gs;
    if (Math.abs(gap) < 0.5) hud.ghost(`level with <b>${best?.name ?? 'leader'}</b>`, '');
    else hud.ghost(`${gap > 0 ? '▲' : '▼'} <b>${Math.abs(gap).toFixed(1)} m</b> ${gap > 0 ? 'ahead of' : 'behind'} ${best?.name ?? 'leader'}`, gap > 0 ? 'ahead' : 'behind');
  }

  function updateCamera(now: number, wallDt: number) {
    const sp = clamp01(state.speed / 26);
    trauma = Math.max(0, trauma - wallDt * 1.6);
    const sh = trauma * trauma;
    shake.set(Math.sin(now * 0.031) * sh * 0.25, Math.sin(now * 0.047 + 1) * sh * 0.2, Math.sin(now * 0.053 + 2) * sh * 0.18);
    fwd.set(Math.cos(state.heading), Math.sin(state.heading), 0);
    right.set(fwd.y, -fwd.x, 0);
    let rate = 4.5, lookRate = 7;
    if (phase === 'intro') {
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
    const fovGoal = phase === 'intro' ? 58 : 60 + 25 * sp * sp;
    camera.fov += (fovGoal - camera.fov) * (1 - Math.exp(-wallDt * 6)); camera.updateProjectionMatrix();
    camera.rotateZ(-state.edge * 0.07 + shake.z * 0.3);
  }

  function takePhoto() {
    photoTaken = true;
    try {
      const g = photoCanvas.getContext('2d')!;
      g.drawImage(renderer.domElement, 0, 0, photoCanvas.width, photoCanvas.height);
      g.fillStyle = 'rgba(8,14,28,.6)'; g.fillRect(0, 300, 640, 60);
      g.fillStyle = '#fff'; g.font = '900 28px system-ui'; g.fillText(fmtTime(Math.round(raceT * 1000) + gates.missed.filter(Boolean).length * MISS_PENALTY_MS), 16, 340);
      g.font = '700 16px system-ui'; g.fillStyle = '#37e6a8'; g.fillText(`${name} · ${courseName} · Ground Truth`, 180, 338);
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
  Object.defineProperty(window, '__gt', { value: { get phase() { return phase; }, get state() { return state; }, get gates() { return { next: gates.next, made: gates.made, total: gates.total }; }, get net() { return net.status; }, get raceT() { return raceT; } } });
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
