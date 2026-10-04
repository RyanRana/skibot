// Presage in the game: stream the frames of the webcam the pose tracker already uses to a SmartSpectra
// session and get the player's pulse and breathing back. Locally that is the dev server (presage.server.ts,
// raw pixels); on the live site it is the Presage relay (presage-relay/, a Docker server holding the key),
// reached at the build's VITE_PRESAGE_URL with JPEG frames. The capture clock comes from
// requestVideoFrameCallback; the pixel work happens in a worker.

export type PresageMessage =
  | { t: 'vitals'; hr: { v: number; c: number; s: boolean } | null; br: { v: number; c: number; s: boolean } | null }
  | { t: 'valid'; code: number; hint: string }
  | { t: 'status'; status: number }
  | { t: 'error'; code?: number; message: string; retryable?: boolean }
  | { t: 'started' } | { t: 'open' } | { t: 'closed' };

/** Short words for the SDK's validation codes (what the player should do). */
export const PRESAGE_HINT: Record<number, string> = {
  1: 'face the camera', 2: 'one face only', 3: 'centre your face', 5: 'more light', 6: 'too bright',
  7: 'chest out of view', 11: 'camera too slow', 12: 'hold still', 13: 'step back', 14: 'step closer', 17: 'look at the camera',
};

export interface PresageEndpoint { url: string; fmt: 'jpeg' | 'rgba'; remote: boolean }

/** Where to measure: the live site's relay (VITE_PRESAGE_URL at build time), the dev server's own
 *  /__presage locally, or nowhere (a deployed site without a relay). ?presage=<url> overrides it in dev
 *  only: in a live build a link could otherwise send a visitor's camera to someone else's server. */
export function presageEndpoint(): PresageEndpoint | null {
  const q = import.meta.env.DEV ? new URLSearchParams(location.search).get('presage') : null;
  if (q) return { url: q, fmt: 'jpeg', remote: true };
  const env = import.meta.env.VITE_PRESAGE_URL as string | undefined;
  if (env) return { url: env, fmt: 'jpeg', remote: true };
  if (import.meta.env.DEV) return { url: `ws://${location.host}/__presage`, fmt: 'rgba', remote: false };
  return null;
}

export function startPresage(video: HTMLVideoElement, on: (m: PresageMessage) => void, ep: PresageEndpoint) {
  if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype) || typeof VideoFrame === 'undefined') {
    on({ t: 'error', message: 'this browser cannot stream camera frames' });
    return () => {};
  }
  const worker = new Worker(new URL('./presage.worker.ts', import.meta.url), { type: 'module' });
  let open = false, stopped = false, last = 0;
  worker.onmessage = e => {
    const m = e.data;
    if (m.t === 'open') { open = true; on({ t: 'open' }); }
    else if (m.t === 'closed') { open = false; on({ t: 'closed' }); }
    else if (m.t === 'msg') { try { on(JSON.parse(m.data)); } catch {} }
  };
  worker.postMessage({ t: 'open', url: ep.url, fmt: ep.fmt });
  const pump = (_now: number, meta: VideoFrameCallbackMetadata) => {
    if (stopped) return;
    if (open && video.readyState >= 2) {
      let ts = Math.round((performance.timeOrigin + (meta.captureTime ?? meta.expectedDisplayTime)) * 1000);
      if (ts <= last) ts = last + 1;
      last = ts;
      try { const frame = new VideoFrame(video, { timestamp: ts }); worker.postMessage({ t: 'frame', frame, ts }, [frame]); } catch {}
    }
    video.requestVideoFrameCallback(pump);
  };
  video.requestVideoFrameCallback(pump);
  return () => { stopped = true; worker.terminate(); };
}
