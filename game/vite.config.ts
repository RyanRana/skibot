import { defineConfig, loadEnv, type Plugin } from 'vite';
import { resolve } from 'node:path';
import { mkdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { presage } from './presage.server.ts';
import { elevenlabs } from './elevenlabs.server.ts';

/** Dev only: `?watch` sessions post their pose log and side-by-side frames here, written to game/.rec/<session>/
 *  so a run can be reviewed afterwards (log.jsonl + frames/NNNNN.jpg). Nothing leaves the machine. */
function recorder(): Plugin {
  const root = resolve(__dirname, '.rec');
  return {
    name: 'watch-recorder',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__rec', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        const chunks: Buffer[] = []; let size = 0;
        req.on('data', (c: Buffer) => { size += c.length; if (size < 4e6) chunks.push(c); });
        req.on('end', () => {
          try {
            const url = new URL(req.url ?? '/', 'http://local');
            const session = (url.searchParams.get('s') || 'session').replace(/[^\w-]/g, '').slice(0, 64);
            const dir = resolve(root, session);
            mkdirSync(resolve(dir, 'frames'), { recursive: true });
            const body = Buffer.concat(chunks);
            if (url.pathname === '/log') appendFileSync(resolve(dir, 'log.jsonl'), body);
            else if (url.pathname === '/frame') writeFileSync(resolve(dir, 'frames', `${String(Number(url.searchParams.get('n')) || 0).padStart(5, '0')}.jpg`), body);
            res.statusCode = 204;
          } catch (e) { console.warn('recorder', e); res.statusCode = 500; }
          res.end();
        });
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  // keys without a VITE_ prefix are read here, on the server, and never reach the client bundle
  const env = loadEnv(mode, __dirname, '');
  return {
  server: { host: true, port: 5173 },
  plugins: [recorder(), presage(env.PRESAGE_API_KEY), elevenlabs(env.ELEVENLABS_API_KEY, env.ELEVENLABS_VOICE_ID, __dirname)],
  build: {
    target: 'es2022',
    rollupOptions: {
      input: { main: resolve(__dirname, 'index.html'), board: resolve(__dirname, 'board.html') },
    },
  },
  };
});
