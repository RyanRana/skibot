// The HUD is plain DOM over the canvas: a clock and speed in the middle, the course on the left,
// the growing dataset on the right, your body in a mirror at the bottom-left, TV splits, a streak
// of clean gates, cheers from the crowd, and the result card.
import type { GateTracker } from './physics.ts';
import { fmtTime, fmtDelta, fmtInt, type Run, type Feed, type DatasetStats } from './net.ts';
import type { Identity } from 'spacetimedb';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
export const esc = (s: string) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));

export class Hud {
  private gateEls: HTMLElement[] = [];
  private msgTimer = 0; private toastTimer = 0; private popTimer = 0; private cheerTimers: number[] = [];
  private lastGates = '';
  splitRows: { gate: number; delta: number | null; t: number }[] = [];

  course(resort: string, name: string, stats: string) { $('course-resort').textContent = resort; $('course-name').textContent = name; $('course-stats').textContent = stats; }
  clock(ms: number) { $('clock').textContent = fmtTime(Math.max(0, ms)); }
  speed(ms: number) { $('speed').textContent = String(Math.round(ms * 3.6)); }

  gates(t: GateTracker) {
    const host = $('gates');
    if (this.gateEls.length !== t.total) {
      host.innerHTML = ''; this.gateEls = [];
      for (let i = 0; i < t.total; i++) { const el = document.createElement('i'); host.appendChild(el); this.gateEls.push(el); }
    }
    const key = `${t.next}:${t.made}:${t.missed.filter(Boolean).length}`;
    if (key === this.lastGates) return;
    this.lastGates = key;
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

  leaderboard(runs: Run[], me: Identity | null) {
    const ol = $('leaderboard');
    const top = runs.slice(0, 6);
    ol.innerHTML = top.length ? top.map(r => `<li class="${me && r.identity.isEqual(me) ? 'me' : ''}"><span>${esc(r.name)}</span><span>${fmtTime(r.timeMs)} · ${r.gatesHit}/${r.gatesTotal}</span></li>`).join('') : '<li><span>No runs yet</span><span>be first</span></li>';
  }

  feed(rows: Feed[]) { $('feed').innerHTML = rows.slice(0, 3).map(r => `<div class="${r.kind}">${esc(r.text)}</div>`).join(''); }

  meters(lean: number, crouch: number) {
    const lb = $('lean-bar');
    lb.style.width = `${Math.abs(lean) * 50}%`;
    lb.style.left = lean < 0 ? `${50 - Math.abs(lean) * 50}%` : '50%';
    lb.classList.toggle('neg', lean < 0);
    $('crouch-bar').style.width = `${crouch * 100}%`;
  }

  camLabel(s: string) { $('cam-label').textContent = s; const a = document.getElementById('attract-label'); if (a) a.textContent = s; }
  hint(s: string) { $('hint').textContent = s; }

  msg(text: string, sub = '', ms = 1800) {
    const el = $('msg');
    el.innerHTML = `${esc(text)}${sub ? `<small>${esc(sub)}</small>` : ''}`;
    el.classList.add('show');
    clearTimeout(this.msgTimer);
    if (ms > 0) this.msgTimer = window.setTimeout(() => el.classList.remove('show'), ms);
  }
  hideMsg() { $('msg').classList.remove('show'); }

  toast(text: string, ms = 1400) {
    const el = $('toast'); el.textContent = text; el.classList.add('show');
    clearTimeout(this.toastTimer); this.toastTimer = window.setTimeout(() => el.classList.remove('show'), ms);
  }

  /** TV split: big centred delta, then it slides into the stack on the left. */
  split(gate: number, elapsedMs: number, deltaMs: number | null, label = '') {
    const pop = $('split-pop');
    pop.className = 'split-pop show ' + (deltaMs === null ? 'plain' : deltaMs < 0 ? 'neg' : 'pos');
    pop.innerHTML = deltaMs === null ? `${fmtTime(elapsedMs)}<small>gate ${gate}${label ? ' · ' + esc(label) : ''}</small>` : `${fmtDelta(deltaMs)}<small>gate ${gate} · vs leader${label ? ' · ' + esc(label) : ''}</small>`;
    clearTimeout(this.popTimer);
    this.popTimer = window.setTimeout(() => pop.classList.remove('show'), 1100);
    this.splitRows.unshift({ gate, delta: deltaMs, t: elapsedMs });
    this.splitRows = this.splitRows.slice(0, 5);
    $('splits').innerHTML = this.splitRows.map(r => `<div class="${r.delta === null ? '' : r.delta < 0 ? 'neg' : 'pos'}"><span>Gate ${r.gate}</span><span>${r.delta === null ? fmtTime(r.t) : fmtDelta(r.delta)}</span></div>`).join('');
  }
  clearSplits() { this.splitRows = []; $('splits').innerHTML = ''; $('split-pop').classList.remove('show'); }

  ghost(text: string, state: 'ahead' | 'behind' | '') { const el = $('ghost'); el.innerHTML = text; el.className = 'ghostline ' + state; }

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

  result(o: { timeMs: number; rank: number; of: number; deltaMs: number | null; gates: string; kmh: number; samples: number; cheers: number; best: boolean } | null) {
    const el = $('result');
    if (!o) { el.classList.remove('show'); return; }
    el.innerHTML = `<div class="time">${fmtTime(o.timeMs)}</div>
      <div class="rank ${o.deltaMs === null ? '' : o.deltaMs <= 0 ? 'neg' : 'pos'}">#${o.rank} of ${o.of}${o.deltaMs === null ? '' : o.deltaMs <= 0 ? ' · new fastest time' : ` · ${fmtDelta(o.deltaMs)} to the leader`}${o.best ? ' · personal best' : ''}</div>
      <div class="row"><span><b>${esc(o.gates)}</b>gates</span><span><b>${o.kmh}</b>km/h top</span><span><b>+${fmtInt(o.samples)}</b>samples to Ground Truth</span><span><b>${o.cheers}</b>cheers</span></div>
      <div class="fine">Raise both hands or press Enter to go again</div>`;
    el.classList.add('show');
  }

  intro(show: boolean, title = '', eyebrow = '', sub = '') {
    const el = $('intro');
    if (show) el.innerHTML = `<div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><div class="sub">${esc(sub)}</div>`;
    el.classList.toggle('show', show);
  }
}
