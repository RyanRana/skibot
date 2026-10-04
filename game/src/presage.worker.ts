// Off the main thread: turn each camera frame into raw RGBA and stream it to the dev server's Presage session.
// Each binary message is an 8-byte little-endian float64 capture timestamp (microseconds) followed by the pixels.
let ws: WebSocket | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let W = 0, H = 0, started = false;

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.t === 'open') {
    ws = new WebSocket(m.url);
    ws.onopen = () => postMessage({ t: 'open' });
    ws.onclose = () => { postMessage({ t: 'closed' }); ws = null; };
    ws.onmessage = ev => postMessage({ t: 'msg', data: ev.data });
    return;
  }
  if (m.t === 'frame') {
    const f = m.frame as VideoFrame;
    // drop frames rather than queue them when the socket is backed up
    if (!ws || ws.readyState !== 1 || ws.bufferedAmount > 6e6) { f.close(); return; }
    const w = f.displayWidth, h = f.displayHeight;
    if (w !== W || h !== H) {
      W = w; H = h;
      ctx = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
      ws.send(JSON.stringify({ t: started ? 'size' : 'start', w, h }));
      started = true;
    }
    ctx!.drawImage(f, 0, 0); f.close();
    const px = ctx!.getImageData(0, 0, W, H).data;
    const out = new Uint8Array(8 + px.length);
    new DataView(out.buffer).setFloat64(0, m.ts, true);
    out.set(px, 8);
    ws.send(out);
  }
};
