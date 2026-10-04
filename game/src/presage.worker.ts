// Off the main thread: turn each camera frame into bytes and stream it to the Presage session. Each binary
// message is an 8-byte little-endian float64 capture timestamp (microseconds) followed by the frame: raw
// RGBA to the local dev server, a JPEG (about 30 KB instead of 1.2 MB) to the relay over the internet.
let ws: WebSocket | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let canvas: OffscreenCanvas | null = null;
let W = 0, H = 0, started = false, fmt: 'jpeg' | 'rgba' = 'rgba';
// JPEG encoding is pipelined: up to 3 frames encode at once (one at a time only reached ~15 fps; Presage wants
// 25+), and a promise chain keeps them leaving in capture order
let inflight = 0, chain: Promise<void> = Promise.resolve();

function pack(ts: number, bytes: Uint8Array | Uint8ClampedArray) {
  const out = new Uint8Array(8 + bytes.length);
  new DataView(out.buffer).setFloat64(0, ts, true);
  out.set(bytes, 8);
  return out;
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  if (m.t === 'open') {
    fmt = m.fmt === 'jpeg' ? 'jpeg' : 'rgba';
    ws = new WebSocket(m.url);
    ws.onopen = () => postMessage({ t: 'open' });
    ws.onclose = () => { postMessage({ t: 'closed' }); ws = null; };
    ws.onmessage = ev => postMessage({ t: 'msg', data: ev.data });
    return;
  }
  if (m.t === 'frame') {
    const f = m.frame as VideoFrame;
    // drop frames rather than queue them when the socket is backed up or the last JPEG is still encoding
    const limit = fmt === 'jpeg' ? 1.5e6 : 6e6;
    if (!ws || ws.readyState !== 1 || ws.bufferedAmount > limit || inflight >= 3) { f.close(); return; }
    const w = f.displayWidth, h = f.displayHeight;
    if (w !== W || h !== H) {
      W = w; H = h;
      canvas = new OffscreenCanvas(w, h);
      ctx = canvas.getContext('2d', { willReadFrequently: fmt === 'rgba' });
      ws.send(JSON.stringify({ t: started ? 'size' : 'start', w, h, fmt }));
      started = true;
    }
    ctx!.drawImage(f, 0, 0); f.close();
    if (fmt === 'rgba') { ws.send(pack(m.ts, ctx!.getImageData(0, 0, W, H).data)); return; }
    inflight++;
    const ts = m.ts, encoded = canvas!.convertToBlob({ type: 'image/jpeg', quality: 0.75 }); // snapshots the canvas now
    chain = chain.then(async () => {
      try { const bytes = new Uint8Array(await (await encoded).arrayBuffer()); if (ws && ws.readyState === 1) ws.send(pack(ts, bytes)); }
      catch { /* skip this frame */ }
      finally { inflight--; }
    });
  }
};
