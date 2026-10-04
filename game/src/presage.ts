// Presage in the game: stream the frames of the webcam the pose tracker already uses to the dev server's
// SmartSpectra session (see presage.server.ts) and get the player's pulse and breathing back. The capture
// clock comes from requestVideoFrameCallback; the pixel work happens in a worker.

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

export function startPresage(video: HTMLVideoElement, on: (m: PresageMessage) => void) {
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
  worker.postMessage({ t: 'open', url: `ws://${location.host}/__presage` });
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
