// ElevenLabs text-to-speech for the race commentator, in the dev server. The browser posts a line of
// commentary to /__tts and gets MP3 back; the key (ELEVENLABS_API_KEY in game/.env.local, gitignored) stays
// here. Every line is cached on disk (game/.tts-cache/, gitignored), so a line said once costs nothing the
// next time. Uses the low-latency Flash model; the voice is ELEVENLABS_VOICE_ID or a commentator-style voice
// picked from the account. Loopback only.
import type { Plugin } from 'vite';
import { createHash } from 'node:crypto';
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);
const MODEL = 'eleven_flash_v2_5';
const PREFERRED = ['Daniel', 'George', 'Brian', 'Charlie', 'Liam', 'Will', 'Eric', 'Chris', 'Roger', 'Adam'];

export function elevenlabs(apiKey: string | undefined, voiceId: string | undefined, root: string): Plugin {
  const cacheDir = resolve(root, '.tts-cache');
  let voice: Promise<{ id: string; name: string } | null> | null = null;
  const pickVoice = () => voice ??= (async () => {
    if (!apiKey) return null;
    if (voiceId) return { id: voiceId, name: voiceId };
    try {
      const r = await fetch('https://api.elevenlabs.io/v1/voices', { headers: { 'xi-api-key': apiKey } });
      if (!r.ok) return null;
      const vs = ((await r.json()) as { voices: { voice_id: string; name: string }[] }).voices ?? [];
      for (const want of PREFERRED) { const v = vs.find(x => x.name.split(/[\s-]/)[0].toLowerCase() === want.toLowerCase()); if (v) return { id: v.voice_id, name: v.name }; }
      return vs[0] ? { id: vs[0].voice_id, name: vs[0].name } : null;
    } catch { return null; }
  })();

  return {
    name: 'elevenlabs',
    apply: 'serve',
    configureServer(server) {
      const log = (s: string) => server.config.logger.info(`[tts] ${s}`, { timestamp: true });
      server.middlewares.use('/__tts', async (req, res) => {
        if (!LOOPBACK.has(req.socket.remoteAddress ?? '')) { res.statusCode = 403; res.end(); return; }
        const v = await pickVoice();
        if (!v && apiKey) voice = null; // a network blip: try again next time
        if (req.method === 'GET') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ enabled: !!v, voice: v?.name ?? null })); return; }
        if (!v) { res.statusCode = 503; res.end('no ElevenLabs key or voice'); return; }
        const chunks: Buffer[] = [];
        for await (const c of req) chunks.push(c as Buffer);
        let text = '';
        try { text = String(JSON.parse(Buffer.concat(chunks).toString()).text ?? '').slice(0, 400).trim(); } catch {}
        if (!text) { res.statusCode = 400; res.end(); return; }
        const file = resolve(cacheDir, createHash('sha1').update(`${v.id}|${MODEL}|${text}`).digest('hex') + '.mp3');
        res.setHeader('content-type', 'audio/mpeg');
        if (existsSync(file)) { res.end(readFileSync(file)); return; }
        try {
          const t0 = Date.now();
          const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${v.id}?output_format=mp3_44100_128`, {
            method: 'POST',
            headers: { 'xi-api-key': apiKey!, 'content-type': 'application/json', accept: 'audio/mpeg' },
            body: JSON.stringify({ text, model_id: MODEL, voice_settings: { stability: 0.32, similarity_boost: 0.8, style: 0.55, use_speaker_boost: true } }),
          });
          if (!r.ok) { const msg = (await r.text()).slice(0, 200); log(`error ${r.status}: ${msg}`); res.statusCode = 502; res.end(msg); return; }
          const audio = Buffer.from(await r.arrayBuffer());
          mkdirSync(cacheDir, { recursive: true }); writeFileSync(file, audio);
          log(`${Date.now() - t0} ms, ${text.length} chars: "${text.slice(0, 60)}"`);
          res.end(audio);
        } catch (e) { log(`failed: ${(e as Error).message}`); res.statusCode = 502; res.end(); }
      });
    },
  };
}
