// Presage SmartSpectra in the dev server. The browser streams the webcam frames it already has over a
// localhost WebSocket (/__presage); the Node SDK measures pulse and breathing on this machine (Presage's
// server only checks the API key) and the readings stream back. The key stays here: PRESAGE_API_KEY in
// game/.env or game/.env.local (gitignored), never in the client bundle. Loopback connections only, one
// session at a time (the SDK's state is process-wide).
import type { Plugin } from 'vite';
import { createRequire } from 'node:module';
import { WebSocketServer, type WebSocket } from 'ws';

const req = createRequire(import.meta.url);
const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

export function presage(apiKey: string | undefined): Plugin {
  return {
    name: 'presage',
    apply: 'serve',
    configureServer(server) {
      const wss = new WebSocketServer({ noServer: true });
      server.httpServer?.on('upgrade', (r, sock, head) => {
        if (r.url !== '/__presage') return;
        if (!LOOPBACK.has(r.socket.remoteAddress ?? '')) { sock.destroy(); return; }
        wss.handleUpgrade(r, sock, head, ws => wss.emit('connection', ws));
      });
      let sdkMod: any = null, decode: any = null;
      let active: { sdk: any; ws: WebSocket } | null = null;
      let dying: Promise<void> = Promise.resolve();
      const log = (s: string) => server.config.logger.info(`[presage] ${s}`, { timestamp: true });

      wss.on('connection', async ws => {
        const send = (o: object) => { if (ws.readyState === 1) ws.send(JSON.stringify(o)); };
        if (!apiKey) { send({ t: 'error', message: 'no PRESAGE_API_KEY in game/.env.local' }); ws.close(); return; }
        // a reload while a session runs: finish the old one first
        if (active) { const old = active; active = null; try { old.ws.close(); } catch {} dying = old.sdk.destroy().catch(() => {}); }
        await dying;
        try { sdkMod ??= req('@smartspectra/node-sdk'); decode ??= sdkMod.decodeMetrics; }
        catch (e) { send({ t: 'error', message: `Presage SDK not installed (${(e as Error).message})` }); ws.close(); return; }
        const { SmartSpectraSDK, PixelFormat, FrameTransform, breathingMetrics, cardioMetrics } = sdkMod;
        // the full bundles, as Presage's own examples request them (pulse alone never produced a pulse rate)
        const sdk = new SmartSpectraSDK({ apiKey, requestedMetrics: [...breathingMetrics, ...cardioMetrics] });
        active = { sdk, ws };
        const latest = (a?: any[]) => { const x = a?.length ? a[a.length - 1] : null; return x ? { v: x.value, c: x.confidence, s: !!x.stable } : null; };
        let lastLog = 0, frames = 0, lastHint = '';
        sdk.on('metrics', (b: Buffer) => {
          const m = decode(b);
          const hr = latest(m.cardio?.pulseRate), br = latest(m.breathing?.rate);
          send({ t: 'vitals', hr, br });
          const now = Date.now();
          if (now - lastLog > 5000) { lastLog = now; log(`[${m.cardio ? `cardio: ${m.cardio.pulseRate?.length ?? 0} pulse` : 'no cardio'}${m.breathing ? `, breathing: ${m.breathing.rate?.length ?? 0} rate` : ''}] pulse ${hr ? `${hr.v.toFixed(0)} (conf ${hr.c.toFixed(0)}${hr.s ? ', stable' : ''})` : '-'}  breathing ${br ? `${br.v.toFixed(1)} (conf ${br.c.toFixed(0)}${br.s ? ', stable' : ''})` : '-'}  frames ${frames}`); }
        });
        sdk.on('validationStatus', (code: number, _ts: number, hint: string) => { send({ t: 'valid', code, hint }); if (hint !== lastHint) { lastHint = hint; log(`status ${code}: ${hint || 'ok'}`); } });
        sdk.on('processingStatus', (status: number) => send({ t: 'status', status }));
        sdk.on('error', (code: number, message: string, retryable: boolean) => { send({ t: 'error', code, message, retryable }); log(`error ${code}: ${message}`); });

        let w = 0, h = 0, started = false, lastTs = 0;
        ws.on('message', (d: Buffer, binary: boolean) => {
          if (!binary) {
            const msg = JSON.parse(String(d));
            if (msg.t === 'start' && !started) {
              w = msg.w; h = msg.h;
              try { sdk.useCustomInput(FrameTransform.kNone); sdk.start(); started = true; send({ t: 'started' }); log(`session started ${w}x${h}`); }
              catch (e: any) { send({ t: 'error', code: e?.code, message: e?.message ?? String(e) }); log(`start failed: ${e?.message ?? e}`); }
            } else if (msg.t === 'size') { w = msg.w; h = msg.h; }
            return;
          }
          if (!started || d.length !== 8 + w * h * 4) return;
          const ts = d.readDoubleLE(0);
          if (ts <= lastTs) return;
          lastTs = ts;
          try { sdk.sendFrame(d.subarray(8), w, h, w * 4, PixelFormat.kRGBA, ts); frames++; } catch { /* a gap or a hiccup: the next frame carries on */ }
        });
        ws.on('close', () => {
          if (active?.ws !== ws) return;
          active = null;
          dying = sdk.destroy().catch(() => {});
          log('session closed');
        });
      });
    },
  };
}
