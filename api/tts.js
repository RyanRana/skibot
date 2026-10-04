// Vercel function: the race commentator's voice. The game asks GET /api/tts?t=<line> and gets MP3 back; the
// ElevenLabs key lives only here, as the ELEVENLABS_API_KEY environment variable of the Vercel project (never
// in the repo, never in the page). GET /api/tts with no text says whether the voice is on.
//
// Guarding the key's credits on a public site: only same-site requests (the browser's Origin/Referer must be
// this host), short lines of plain text, a per-visitor rate limit, and the credit cap on the key itself.
// Answers are cached by Vercel's CDN for a year, keyed on the line, so a line said once is free after that.
// Optional: ELEVENLABS_VOICE_ID to pick the voice; otherwise a commentator-style voice from the account.

const MODEL = 'eleven_flash_v2_5';
const PREFERRED = ['Daniel', 'George', 'Brian', 'Charlie', 'Liam', 'Will', 'Eric', 'Chris', 'Roger', 'Adam'];
const TEXT_OK = /^[\p{L}\p{N}\s.,!?'’:;()\-–—%&]+$/u;
const LIMIT = 40, WINDOW_MS = 60_000;            // lines per visitor per minute, per function instance
const hits = new Map();
let voice = null;

async function pickVoice(key) {
  if (voice) return voice;
  if (process.env.ELEVENLABS_VOICE_ID) return (voice = { id: process.env.ELEVENLABS_VOICE_ID, name: 'custom' });
  const r = await fetch('https://api.elevenlabs.io/v1/voices', { headers: { 'xi-api-key': key } });
  if (!r.ok) return null;
  const vs = (await r.json()).voices ?? [];
  for (const want of PREFERRED) {
    const v = vs.find(x => x.name.split(/[\s-]/)[0].toLowerCase() === want.toLowerCase());
    if (v) return (voice = { id: v.voice_id, name: v.name });
  }
  return vs[0] ? (voice = { id: vs[0].voice_id, name: vs[0].name }) : null;
}

function sameSite(req) {
  const host = req.headers.host || '';
  const from = req.headers.origin || req.headers.referer || '';
  if (!from) return false;
  try { return new URL(from).host === host; } catch { return false; }
}

function limited(req) {
  const ip = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() || 'unknown';
  const now = Date.now();
  const h = (hits.get(ip) || []).filter(t => now - t < WINDOW_MS);
  h.push(now); hits.set(ip, h);
  if (hits.size > 5000) hits.clear();
  return h.length > LIMIT;
}

module.exports = async (req, res) => {
  const key = process.env.ELEVENLABS_API_KEY;
  if (req.method !== 'GET') { res.statusCode = 405; return res.end(); }
  if (!sameSite(req)) { res.statusCode = 403; return res.end('same-site only'); }
  const text = (new URL(req.url, 'http://x').searchParams.get('t') || '').trim();
  if (!text) {
    const v = key ? await pickVoice(key).catch(() => null) : null;
    res.setHeader('content-type', 'application/json'); res.setHeader('cache-control', 'no-store');
    return res.end(JSON.stringify({ enabled: !!v, voice: v ? v.name : null }));
  }
  if (!key) { res.statusCode = 503; return res.end('voice off'); }
  if (text.length > 300 || !TEXT_OK.test(text)) { res.statusCode = 400; return res.end('bad line'); }
  if (limited(req)) { res.statusCode = 429; return res.end('slow down'); }
  try {
    const v = await pickVoice(key);
    if (!v) { res.statusCode = 503; return res.end('no voice'); }
    const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${v.id}?output_format=mp3_44100_128`, {
      method: 'POST',
      headers: { 'xi-api-key': key, 'content-type': 'application/json', accept: 'audio/mpeg' },
      body: JSON.stringify({ text, model_id: MODEL, voice_settings: { stability: 0.32, similarity_boost: 0.8, style: 0.55, use_speaker_boost: true } }),
    });
    if (!r.ok) { res.statusCode = 502; return res.end('voice error'); }
    const audio = Buffer.from(await r.arrayBuffer());
    res.setHeader('content-type', 'audio/mpeg');
    res.setHeader('cache-control', 'public, max-age=86400, s-maxage=31536000, immutable');
    return res.end(audio);
  } catch {
    res.statusCode = 502; return res.end('voice error');
  }
};
