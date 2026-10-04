// Presage relay: contactless heart rate and breathing for the live Hazard Intelligence game.
//
// The game in a visitor's browser opens a WebSocket to /presage, sends {t:'start', w, h, fmt} and then one
// binary message per camera frame: an 8-byte little-endian float64 capture time in microseconds followed by
// the frame (fmt 'jpeg' over the internet, 'rgba' raw pixels on a local network). This server decodes the
// frames, runs the Presage SmartSpectra SDK on them and sends back {t:'vitals', hr, br}, {t:'valid', code,
// hint} and {t:'error', ...}. The Presage key is PRESAGE_API_KEY in this server's environment and nowhere
// else: the site only knows this server's address.
//
// Protecting the key's credits on a public site: only the origins in ALLOWED_ORIGINS may connect (browsers
// always send Origin on a WebSocket), one measurement at a time (the SDK's state is process-wide), each
// session capped at MAX_SESSION_SECONDS, and at most SESSIONS_PER_HOUR sessions per visitor address.
import http from 'node:http';
import { createRequire } from 'node:module';
import { WebSocketServer } from 'ws';
import jpeg from 'jpeg-js';

const require = createRequire(import.meta.url);
const ss = require('@smartspectra/node-sdk');

const PORT = Number(process.env.PORT || 8787);
const KEY = process.env.PRESAGE_API_KEY || '';
const ALLOWED = (process.env.ALLOWED_ORIGINS || '').split(',').map(s => s.trim().replace(/\/$/, '')).filter(Boolean);
const MAX_SECONDS = Number(process.env.MAX_SESSION_SECONDS || 300);
const PER_HOUR = Number(process.env.SESSIONS_PER_HOUR || 8);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** Exact origins from ALLOWED_ORIGINS; with none set, only this machine (http://localhost:*). */
function originOk(origin) {
  if (!origin) return false;
  if (ALLOWED.length) return ALLOWED.includes(origin.replace(/\/$/, ''));
  try { const u = new URL(origin); return u.hostname === 'localhost' || u.hostname === '127.0.0.1'; } catch { return false; }
}

const visits = new Map(); // address -> session start times in the last hour
function tooMany(addr) {
  const now = Date.now(), recent = (visits.get(addr) || []).filter(t => now - t < 3600_000);
  if (recent.length >= PER_HOUR) { visits.set(addr, recent); return true; }
  recent.push(now); visits.set(addr, recent);
  if (visits.size > 10000) visits.clear();
  return false;
}

let active = null;                    // { sdk, ws }
let dying = Promise.resolve();        // the previous session's teardown

const server = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'content-type': 'application/json', 'access-control-allow-origin': '*' });
    res.end(JSON.stringify({ ok: true, keyed: !!KEY, busy: !!active }));
    return;
  }
  res.writeHead(404); res.end();
});

const wss = new WebSocketServer({ noServer: true, maxPayload: 4_000_000 });
server.on('upgrade', (req, sock, head) => {
  const path = new URL(req.url || '/', 'http://relay').pathname;
  const origin = req.headers.origin || '';
  if (path !== '/presage' || !originOk(origin)) {
    log('refused', path, origin || '(no origin)');
    sock.write('HTTP/1.1 403 Forbidden\r\n\r\n'); sock.destroy(); return;
  }
  wss.handleUpgrade(req, sock, head, ws => wss.emit('connection', ws, req));
});

wss.on('connection', async (ws, req) => {
  const send = o => { if (ws.readyState === 1) ws.send(JSON.stringify(o)); };
  const addr = String(req.headers['x-forwarded-for'] || req.socket.remoteAddress || '').split(',')[0].trim();
  if (!KEY) { send({ t: 'error', message: 'pulse server has no key' }); ws.close(); return; }
  if (active) { send({ t: 'error', message: 'pulse busy: someone else is being measured' }); ws.close(); return; }
  if (tooMany(addr)) { send({ t: 'error', message: 'pulse: too many sessions, try later' }); ws.close(); return; }
  await dying;
  if (active || ws.readyState !== 1) { if (ws.readyState === 1) { send({ t: 'error', message: 'pulse busy' }); ws.close(); } return; }

  const sdk = new ss.SmartSpectraSDK({ apiKey: KEY, requestedMetrics: [...ss.breathingMetrics, ...ss.cardioMetrics] });
  active = { sdk, ws };
  log('session start', addr);
  const latest = a => { const x = a?.length ? a[a.length - 1] : null; return x ? { v: x.value, c: x.confidence, s: !!x.stable } : null; };
  let frames = 0, lastLog = 0, lastHint = '';
  sdk.on('metrics', b => {
    const m = ss.decodeMetrics(b);
    const hr = latest(m.cardio?.pulseRate), br = latest(m.breathing?.rate);
    send({ t: 'vitals', hr, br });
    if (Date.now() - lastLog > 5000) { lastLog = Date.now(); log(`pulse ${hr ? hr.v.toFixed(0) + ' (' + hr.c.toFixed(0) + ')' : '-'} breathing ${br ? br.v.toFixed(1) + ' (' + br.c.toFixed(0) + ')' : '-'} frames ${frames}`); }
  });
  sdk.on('validationStatus', (code, _ts, hint) => { send({ t: 'valid', code, hint }); if (hint !== lastHint) { lastHint = hint; log(`status ${code}: ${hint || 'ok'}`); } });
  sdk.on('processingStatus', status => send({ t: 'status', status }));
  sdk.on('error', (code, message, retryable) => { send({ t: 'error', code, message, retryable }); log(`error ${code}: ${message}`); });

  const cap = setTimeout(() => { send({ t: 'error', message: 'pulse session over' }); ws.close(); }, MAX_SECONDS * 1000);
  let w = 0, h = 0, fmt = 'jpeg', started = false, lastTs = 0;
  ws.on('message', (d, binary) => {
    if (!binary) {
      let msg; try { msg = JSON.parse(String(d)); } catch { return; }
      if (msg.t === 'start' && !started) {
        w = Math.min(1920, Math.max(16, msg.w | 0)); h = Math.min(1080, Math.max(16, msg.h | 0)); fmt = msg.fmt === 'rgba' ? 'rgba' : 'jpeg';
        try { sdk.useCustomInput(ss.FrameTransform.kNone); sdk.start(); started = true; send({ t: 'started' }); log(`measuring ${w}x${h} ${fmt}`); }
        catch (e) { send({ t: 'error', code: e?.code, message: e?.message ?? String(e) }); }
      }
      return;
    }
    if (!started || d.length < 16) return;
    const ts = d.readDoubleLE(0);
    if (!(ts > lastTs)) return;
    let px, fw = w, fh = h;
    try {
      if (fmt === 'jpeg') { const img = jpeg.decode(d.subarray(8), { useTArray: true, formatAsRGBA: true, maxResolutionInMP: 4 }); px = img.data; fw = img.width; fh = img.height; }
      else { if (d.length !== 8 + w * h * 4) return; px = d.subarray(8); }
      sdk.sendFrame(px, fw, fh, fw * 4, ss.PixelFormat.kRGBA, ts);
      lastTs = ts; frames++;
    } catch { /* a bad frame or a timestamp hiccup: the next one carries on */ }
  });
  ws.on('close', () => {
    clearTimeout(cap);
    if (active?.ws !== ws) return;
    active = null;
    dying = sdk.destroy().catch(() => {});
    log('session end', `${frames} frames`);
  });
});

server.listen(PORT, () => log(`presage relay on :${PORT}, ${KEY ? 'key set' : 'NO KEY'}, origins: ${ALLOWED.length ? ALLOWED.join(' ') : 'localhost only'}`));
