// The HUD is plain DOM over the canvas, laid out like a race broadcast that gets out of the way: during a
// run only the course tag and splits (top-left), the timing cluster with its powerup chips and callouts
// (top-centre), cheers (right edge) and a small picture of your body (bottom-left) stay up; the live
// dataset, the leaderboard and the mountain feed come back between runs. The middle of the frame is kept
// for start prompts, the countdown and the gate arrow.
import type { GateTracker } from './physics.ts';
import { fmtTime, fmtDelta, fmtInt, type Run, type Feed, type DatasetStats } from './net.ts';
import type { Identity } from 'spacetimedb';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
export const esc = (s: string) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));

export class Hud {
  private gateEls: HTMLElement[] = [];
  private msgTimer = 0; private cheerTimers: number[] = [];
  private lastGates = '';
  splitRows: { gate: number; delta: number | null; t: number }[] = [];

  course(resort: string, name: string, stats: string) { $('course-resort').textContent = resort; $('course-name').textContent = name; $('course-stats').textContent = stats; }
  clock(ms: number) { $('clock').textContent = fmtTime(Math.max(0, ms)); }
  speed(ms: number) {
    const kmh = Math.round(ms * 3.6);
    $('speed').textContent = String(kmh);
    const ring = $('speedo');
    ring.style.setProperty('--p', Math.min(1, kmh / 90).toFixed(3));
    ring.classList.toggle('hot', kmh >= 60);
  }

  gates(t: GateTracker) {
    const host = $('gates');
    if (this.gateEls.length !== t.total) {
      host.innerHTML = ''; this.gateEls = [];
      for (let i = 0; i < t.total; i++) { const el = document.createElement('i'); host.appendChild(el); this.gateEls.push(el); }
    }
    const key = `${t.next}:${t.made}:${t.missed.filter(Boolean).length}`;
    if (key === this.lastGates) return;
    this.lastGates = key;
    $('gate-made').textContent = String(t.made); $('gate-total').textContent = String(t.total);
    for (let i = 0; i < t.total; i++) {
      const el = this.gateEls[i];
      el.className = t.hit[i] ? (i % 2 === 0 ? 'red' : 'blue') : t.missed[i] ? 'miss' : '';
      if (i === t.next) el.classList.add('next');
    }
  }

  /** Streak of consecutive clean gates spelled out; full = bonus. */
  streak(n: number, word = 'STREIF') {
    const el = $('streak');
    const k = Math.min(word.length, n);
    el.innerHTML = word.split('').map((ch, i) => `<i class="${i < k ? 'on' : ''}">${ch}</i>`).join('');
    el.classList.toggle('full', k >= word.length);
  }

  dataset(s: DatasetStats | null, online: number) {
    $('ds-samples').textContent = s ? fmtInt(s.samples) : '0';
    $('ds-runs').textContent = s ? fmtInt(s.runs) : '0';
    $('ds-skiers').textContent = s ? fmtInt(s.skiers) : '0';
    $('ds-online').textContent = String(online);
  }

  /** Fastest run per skier (runs arrive fastest first), six rows, you in mint. */
  leaderboard(runs: Run[], me: Identity | null) {
    const ol = $('leaderboard');
    const seen = new Set<string>(), top: Run[] = [];
    for (const r of runs) { const k = r.identity.toHexString(), dup = `${r.name}|${r.timeMs}`; if (seen.has(k) || seen.has(dup)) continue; seen.add(k); seen.add(dup); top.push(r); if (top.length >= 6) break; }
    ol.innerHTML = top.length
      ? top.map((r, i) => `<li class="${me && r.identity.isEqual(me) ? 'me' : ''}"><span class="rank">${i + 1}</span><span class="name">${esc(r.name)}</span><span class="t">${fmtTime(r.timeMs)} · ${r.gatesHit}/${r.gatesTotal}</span></li>`).join('')
      : '<li class="empty">No runs yet today. Be the first down.</li>';
  }

  /** Hazard Intelligence: heart readings pooled so far and how many stretches of the course are flagged. */
  hearts(readings: number, zones: number) { $('ds-hearts').textContent = readings ? `♥ ${fmtInt(readings)} heart readings · ${zones} hazard zone${zones === 1 ? '' : 's'}` : '♥ no heart readings yet'; }

  feed(rows: Feed[]) { $('feed').innerHTML = rows.slice(0, 3).map(r => `<div class="${r.kind}">${esc(r.text)}</div>`).join(''); }

  meters(lean: number, crouch: number) {
    const lb = $('lean-bar');
    lb.style.width = `${Math.abs(lean) * 50}%`;
    lb.style.left = lean < 0 ? `${50 - Math.abs(lean) * 50}%` : '50%';
    lb.classList.toggle('neg', lean < 0);
    $('crouch-bar').style.width = `${crouch * 100}%`;
  }

  camLabel(s: string) { $('cam-label').textContent = s; const a = document.getElementById('attract-label'); if (a) a.textContent = s; }

  msg(text: string, sub = '', ms = 1800) {
    const el = $('msg');
    el.innerHTML = `${esc(text)}${sub ? `<small>${esc(sub)}</small>` : ''}`;
    el.classList.add('show');
    clearTimeout(this.msgTimer);
    if (ms > 0) this.msgTimer = window.setTimeout(() => el.classList.remove('show'), ms);
  }
  hideMsg() { $('msg').classList.remove('show'); }

  /** A small callout under the timing cluster; the newest is on top and at most three show at once. */
  toast(text: string, ms = 1400, kind: 'good' | 'bad' | 'warn' | 'cyan' | 'ice' | 'violet' | '' = '') {
    const host = $('callouts');
    const div = document.createElement('div');
    div.className = kind; div.textContent = text;
    host.prepend(div);
    while (host.children.length > 3) host.lastElementChild!.remove();
    window.setTimeout(() => { div.classList.add('out'); window.setTimeout(() => div.remove(), 250); }, ms);
  }

  /** A time bonus floats up off the clock. */
  pop(text: string, color: string) {
    const el = $('pop'); el.textContent = text; el.style.color = color;
    el.classList.remove('show'); void el.offsetWidth; el.classList.add('show');
  }

  /** Active powerups as chips under the clock (a ring drains as each runs out), and the crystal tally. */
  powers(o: { racing: boolean; boost: number; tuckBoost: number; shield: boolean; magnet: number; tuck: number; tuckCharge: number; crystals: number; crystalTotal: number }) {
    const chips: string[] = [];
    const chip = (c: string, p: number, icon: string, label: string, extra = '') => chips.push(`<div class="chip" style="--c:${c};--p:${p.toFixed(3)}"><i>${icon}</i>${label}${extra}</div>`);
    if (o.racing) {
      if (o.boost > 0) chip('var(--cyan)', o.boost, ICON.ring, 'Slipstream');
      if (o.tuckBoost > 0) chip('var(--amber)', o.tuckBoost, ICON.tuck, 'Tuck boost');
      else if (o.tuck > 0 || o.tuckCharge > 0.15) chip(['var(--amber)', '#4da3ff', 'var(--amber)', '#ff4fd8'][o.tuck], o.tuckCharge, ICON.tuck, 'Tuck', `<span class="pips">${[1, 2, 3].map(k => `<u class="${k <= o.tuck ? 'on' : ''}"></u>`).join('')}</span>`);
      if (o.shield) chip('var(--ice)', 1, ICON.shield, 'Shield');
      if (o.magnet > 0) chip('var(--violet)', o.magnet / 3, ICON.magnet, 'Magnet', ` <b>${o.magnet}</b>`);
      if (o.crystalTotal) chips.push(`<div class="chip tally">◆ <b>${o.crystals}</b>/${o.crystalTotal}</div>`);
    }
    const html = chips.join('');
    if (html !== this.lastPowers) { $('powers').innerHTML = html; this.lastPowers = html; }
  }
  private lastPowers = '';

  /** Heart rate (a heart beating at your real rate), breathing, and what the game is doing about it. */
  vitals(o: { live: boolean; status: string; hr: number; br: number; baseHr: number; arousal: number; mode: '' | 'ice' | 'steady' } | null) {
    let html = '';
    if (o) {
      const c = o.arousal < 0.3 ? 'var(--teal)' : o.arousal < 0.65 ? 'var(--amber)' : 'var(--red)';
      if (o.live && o.hr) {
        html = `<div class="vital" style="--c:${c};--bt:${(60 / o.hr).toFixed(2)}s"><span class="heart">♥</span><b>${Math.round(o.hr)}</b>bpm${o.baseHr ? ` <small style="opacity:.6">rest ${Math.round(o.baseHr)}</small>` : ''}</div>`;
        if (o.br) html += `<div class="vital" style="--c:${c}">breath <b>${Math.round(o.br)}</b>/min</div>`;
        if (o.mode === 'ice') html += `<div class="vital mode" style="--c:var(--cyan)">Ice veins · crystals ×2</div>`;
        if (o.mode === 'steady') html += `<div class="vital mode" style="--c:var(--violet)">Steady · wider gates</div>`;
      } else html = `<div class="vital" style="--c:var(--dim)"><span class="heart" style="animation:none">♥</span>${esc(o.status)}</div>`;
    }
    if (html !== this.lastVitals) { $('vitals').innerHTML = html; this.lastVitals = html; }
  }
  private lastVitals = '';

  /** TV split: the newest row lands on top of the stack under the course card. */
  split(gate: number, elapsedMs: number, deltaMs: number | null) {
    this.splitRows.unshift({ gate, delta: deltaMs, t: elapsedMs });
    this.splitRows = this.splitRows.slice(0, 5);
    $('splits').innerHTML = this.splitRows.map((r, i) => `<div class="${r.delta === null ? '' : r.delta < 0 ? 'neg' : 'pos'}${i === 0 ? ' new' : ''}"><span>Gate ${r.gate}</span><span>${r.delta === null ? fmtTime(r.t) : fmtDelta(r.delta)}</span></div>`).join('');
  }
  clearSplits() { this.splitRows = []; $('splits').innerHTML = ''; }

  ghost(text: string, state: 'ahead' | 'behind' | '') { const el = $('ghost'); el.innerHTML = text; el.className = 'ghostline ' + state; }

  /** Arrow to the next gate when it is off to the side. `deg` is clockwise from straight ahead. */
  gateArrow(deg: number, dist: number, show: boolean, color: 'red' | 'blue') {
    const el = $('gate-arrow');
    el.classList.toggle('show', show);
    if (!show) return;
    el.className = `gate-arrow show ${color}`;
    (el.querySelector('i') as HTMLElement).style.transform = `rotate(${deg.toFixed(0)}deg)`;
    (el.querySelector('span') as HTMLElement).textContent = `next gate · ${dist.toFixed(0)} m`;
  }

  /** A cheer from the crowd, named. */
  cheer(from: string, kind: string, meter: number) {
    const host = $('cheer');
    const icon = kind === 'fire' ? '🔥' : kind === 'clap' ? '👏' : '🔔';
    const div = document.createElement('div');
    div.textContent = `${icon} ${from} ${kind === 'fire' ? 'is on fire for you' : kind === 'clap' ? 'applauds' : 'rings the cowbell'}`;
    host.prepend(div);
    while (host.querySelectorAll('div:not(.meter)').length > 3) host.querySelectorAll('div:not(.meter)')[host.querySelectorAll('div:not(.meter)').length - 1].remove();
    this.cheerTimers.push(window.setTimeout(() => div.remove(), 3500));
    this.crowdMeter(meter);
  }
  crowdMeter(v: number) {
    const host = $('cheer');
    let m = host.querySelector('.meter') as HTMLElement | null;
    if (!m) { m = document.createElement('div'); m.className = 'meter'; m.innerHTML = 'crowd <i><b></b></i>'; host.appendChild(m); }
    (m.querySelector('b') as HTMLElement).style.width = `${Math.min(100, v * 100)}%`;
  }

  result(o: { timeMs: number; rank: number; of: number; deltaMs: number | null; gates: string; kmh: number; samples: number; cheers: number; best: boolean; heart?: { avgHr: number; peakHr: number; baseHr: number; avgBr: number; mapped?: number } | null } | null) {
    const el = $('result');
    if (!o) { el.classList.remove('show'); return; }
    el.innerHTML = `<div class="time">${fmtTime(o.timeMs)}</div>
      <div class="rank ${o.deltaMs === null ? '' : o.deltaMs <= 0 ? 'neg' : 'pos'}">#${o.rank} of ${o.of}${o.deltaMs === null ? '' : o.deltaMs <= 0 ? ' · new fastest time' : ` · ${fmtDelta(o.deltaMs)} to the leader`}${o.best ? ' · personal best' : ''}</div>
      <div class="row"><span><b>${esc(o.gates)}</b>gates</span><span><b>${o.kmh}</b>km/h top</span><span><b>+${fmtInt(o.samples)}</b>samples to Hazard Intelligence</span><span><b>${o.cheers}</b>cheers</span></div>
      ${o.heart && o.heart.avgHr ? `<div class="row"><span><b>♥ ${Math.round(o.heart.avgHr)}</b>avg bpm</span><span><b>${Math.round(o.heart.peakHr)}</b>peak bpm</span><span><b>${Math.round(o.heart.baseHr)}</b>resting</span>${o.heart.avgBr ? `<span><b>${Math.round(o.heart.avgBr)}</b>breaths/min</span>` : ''}${o.heart.mapped ? `<span><b>+${o.heart.mapped}</b>heart readings mapped</span>` : ''}</div>` : ''}
      <div class="fine">Raise both hands or press Enter to go again</div>`;
    el.classList.add('show');
  }

}

const ICON = {
  ring: '<svg viewBox="0 0 24 24"><path d="M4 5l7 7-7 7h4l7-7-7-7zm7 0l7 7-7 7h4l7-7-7-7z"/></svg>',
  shield: '<svg viewBox="0 0 24 24"><path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5z"/></svg>',
  magnet: '<svg viewBox="0 0 24 24"><path d="M4 3h5v9a3 3 0 006 0V3h5v9a8 8 0 01-16 0zm0 0v4h5V3zm11 0v4h5V3z"/></svg>',
  tuck: '<svg viewBox="0 0 24 24"><path d="M13 2L4 14h7l-1 8 9-12h-7z"/></svg>',
};
