// The mountain board: a live map of the real piste with every skier on it, the leaderboard, the
// dataset counter and how to join. Meant for a second screen or a phone at the table.
import QRCode from 'qrcode';
import { loadCourse, type Course } from './course.ts';
import { Net, fmtTime, fmtInt, colorOf, type Skier } from './net.ts';
import { esc } from './hud.ts';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const COURSE_SLUG = 'kitzbuhel-streif';

async function main() {
  const course = await loadCourse(COURSE_SLUG);
  const courseName = (course.meta.run || '').split(/[+(]/)[0].replace(/-?Rennstrecke|Familienabfahrt/g, '').trim() || course.meta.slug;
  $('b-resort').textContent = course.meta.resort; $('b-course').textContent = courseName;
  const dropM = course.meta.centerline[course.startIndex][2] - course.meta.centerline[course.finishIndex][2];
  $('b-stats').textContent = `${course.raceLength.toFixed(0)} m race on a ${course.meta.stats.length_m.toFixed(0)} m course · ${dropM.toFixed(0)} m drop · ${course.gates.length} gates · OpenStreetMap + AWS Terrain Tiles`;
  const gameUrl = `${location.origin}/`;
  QRCode.toCanvas($('qr') as HTMLCanvasElement, gameUrl, { width: 160, margin: 1, color: { dark: '#f3f6fb', light: '#00000000' } }).catch(() => {});

  const canvas = $('map') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const relief = buildRelief(course);
  const trails = new Map<string, { x: number; y: number }[]>();
  let skiers: Skier[] = [];

  const net = new Net({
    skier: (kind, row) => {
      const id = row.identity.toHexString();
      if (kind === 'delete') { trails.delete(id); return; }
      const t = trails.get(id) ?? []; trails.set(id, t);
      if (row.phase === 2) { t.push({ x: row.x, y: row.y }); if (t.length > 400) t.shift(); }
      if (row.phase !== 2 && row.phase !== 3) t.length = 0;
    },
    changed: refresh,
    cheer: c => { const d = document.createElement('div'); d.className = 'cheer'; d.textContent = `${c.kind === 'fire' ? '🔥' : c.kind === 'clap' ? '👏' : '🔔'} ${c.fromName} → ${c.toName || 'everyone'}`; $('b-feed').prepend(d); setTimeout(() => d.remove(), 12000); },
  });
  net.connect();

  // cheer controls
  const nameInput = $('cheer-name') as HTMLInputElement;
  try { nameInput.value = localStorage.getItem('gt-cheer-name') || ''; } catch {}
  for (const btn of document.querySelectorAll<HTMLButtonElement>('.cheer-buttons button')) {
    btn.addEventListener('click', () => {
      const from = nameInput.value.trim() || 'A fan';
      try { localStorage.setItem('gt-cheer-name', nameInput.value); } catch {}
      net.sendCheer(from, ($('cheer-target') as HTMLSelectElement).value, btn.dataset.kind || 'bell');
      btn.style.transform = 'scale(.92)'; setTimeout(() => (btn.style.transform = ''), 120);
      if (navigator.vibrate) navigator.vibrate(30);
    });
  }

  function refresh() {
    const s = net.stats();
    $('b-samples').textContent = s ? fmtInt(s.samples) : '0';
    $('b-runs').textContent = s ? fmtInt(s.runs) : '0';
    $('b-meters').textContent = s ? fmtInt(Math.round(s.meters)) : '0';
    $('b-skiers').textContent = s ? fmtInt(s.skiers) : '0';
    const runs = net.runs(courseName).slice(0, 8);
    $('b-board').innerHTML = runs.length ? runs.map(r => `<li><span>${esc(r.name)}</span><span>${fmtTime(r.timeMs)} · ${r.gatesHit}/${r.gatesTotal}</span></li>`).join('') : '<li><span>No runs yet</span><span></span></li>';
    $('b-feed').innerHTML = net.feed().slice(0, 8).map(f => `<div class="${f.kind}">${esc(f.text)}</div>`).join('');
  }

  // map transform: fit the terrain bounds, rotated so the course runs top to bottom
  function fit() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) { canvas.width = w * devicePixelRatio; canvas.height = h * devicePixelRatio; }
  }
  const PAD = 30;
  function toScreen(x: number, y: number, w: number, h: number) {
    const sx = (w - 2 * PAD) / (course.xMax - course.x0), sy = (h - 2 * PAD) / (course.yMax - course.y0);
    const s = Math.min(sx, sy);
    const ox = (w - s * (course.xMax - course.x0)) / 2, oy = (h - s * (course.yMax - course.y0)) / 2;
    return [ox + (x - course.x0) * s, h - (oy + (y - course.y0) * s)];
  }

  function draw() {
    requestAnimationFrame(draw);
    fit();
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    // relief
    const [x0, y0] = toScreen(course.x0, course.yMax, w, h), [x1, y1] = toScreen(course.xMax, course.y0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(relief, x0, y0, x1 - x0, y1 - y0);
    // piste corridor
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const cl = course.meta.centerline;
    const scale = (x1 - x0) / (course.xMax - course.x0);
    ctx.beginPath();
    cl.forEach((p, i) => { const [sx, sy] = toScreen(p[0], p[1], w, h); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
    ctx.strokeStyle = 'rgba(255,255,255,.22)'; ctx.lineWidth = 44 * scale; ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1.2; ctx.stroke();
    // gates
    for (const g of course.gates) {
      const [ax, ay] = toScreen(g.turn_pole[0], g.turn_pole[1], w, h), [bx, by] = toScreen(g.outer_pole[0], g.outer_pole[1], w, h);
      ctx.strokeStyle = g.color === 'red' ? '#ff3b4e' : '#3d7bff'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    const f = course.finish; const [fa, fb] = [toScreen(f.poles[0][0], f.poles[0][1], w, h), toScreen(f.poles[1][0], f.poles[1][1], w, h)];
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(fa[0], fa[1]); ctx.lineTo(fb[0], fb[1]); ctx.stroke();
    // start
    const sp = course.meta.centerline[course.startIndex];
    const [stx, sty] = toScreen(sp[0], sp[1], w, h);
    ctx.fillStyle = '#37e6a8'; ctx.font = '600 11px system-ui'; ctx.fillText('START', stx + 8, sty + 4);
    // trails + skiers
    skiers = net.skiers();
    for (const s of skiers) {
      const t = trails.get(s.identity.toHexString());
      const col = colorOf(s.color);
      if (t && t.length > 1) {
        ctx.beginPath(); t.forEach((p, i) => { const [sx, sy] = toScreen(p.x, p.y, w, h); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.globalAlpha = 0.8; ctx.stroke(); ctx.globalAlpha = 1;
      }
      if (s.phase === 0) continue;
      const [sx, sy] = toScreen(s.x, s.y, w, h);
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = '700 12px system-ui'; ctx.fillText(`${s.name}${s.phase === 2 ? ` · ${Math.round(s.speed * 3.6)} km/h` : ''}`, sx + 10, sy - 8);
    }
    const sel = $('cheer-target') as HTMLSelectElement;
    const names = skiers.filter(s => s.phase !== 0).map(s => s.name);
    const have = [...sel.options].slice(1).map(o => o.value);
    if (names.join('|') !== have.join('|')) { const cur = sel.value; sel.innerHTML = '<option value="">Everyone on the course</option>' + names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join(''); sel.value = names.includes(cur) ? cur : ''; }
    $('b-live').innerHTML = skiers.length ? skiers.map(s => `<li><span><i class="dot" style="background:${colorOf(s.color)}"></i>${esc(s.name)}</span><span>${s.phase === 2 ? `${Math.round(s.progress * 100)}% · ${Math.round(s.speed * 3.6)} km/h` : s.phase === 3 ? 'finished' : 'at the start'}</span></li>`).join('') : '<li><span>Nobody yet</span><span>scan to ski</span></li>';
  }
  draw();
}

/** Hillshade of the terrain as an offscreen canvas (north up). */
function buildRelief(course: Course) {
  const { nrow, ncol } = course;
  const c = document.createElement('canvas'); c.width = ncol; c.height = nrow;
  const g = c.getContext('2d')!; const img = g.createImageData(ncol, nrow);
  const lx = 0.6, ly = -0.5, lz = 0.62;
  for (let r = 0; r < nrow; r++) for (let cc = 0; cc < ncol; cc++) {
    const x = course.x0 + cc * course.cell, y = course.y0 + r * course.cell;
    const s = course.sample(x, y);
    const sh = Math.max(0, s.n[0] * lx + s.n[1] * ly + s.n[2] * lz);
    const slope = 1 - s.n[2];
    const k = ((nrow - 1 - r) * ncol + cc) * 4; // row 0 is south: flip so north is up
    img.data[k] = 150 + 95 * sh - 60 * slope; img.data[k + 1] = 170 + 80 * sh - 50 * slope; img.data[k + 2] = 200 + 55 * sh - 30 * slope; img.data[k + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  // contour lines every 10 m
  const g2 = c; const cx = g2.getContext('2d')!;
  cx.strokeStyle = 'rgba(255,255,255,.18)'; cx.lineWidth = 0.6;
  for (let r = 1; r < nrow; r++) for (let cc = 1; cc < ncol; cc++) {
    const z = course.at(r, cc), zl = course.at(r, cc - 1), zd = course.at(r - 1, cc);
    if (Math.floor(z / 10) !== Math.floor(zl / 10) || Math.floor(z / 10) !== Math.floor(zd / 10)) { cx.fillStyle = 'rgba(255,255,255,.22)'; cx.fillRect(cc, nrow - 1 - r, 1, 1); }
  }
  return c;
}

main().catch(e => console.error(e));
