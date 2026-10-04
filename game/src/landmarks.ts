// MediaPipe Pose landmark indices and the skeleton edges we draw. No dependencies, so the retarget can be tested in Node.
export const LM = { nose: 0, lsh: 11, rsh: 12, lel: 13, rel: 14, lwr: 15, rwr: 16, lhip: 23, rhip: 24, lkn: 25, rkn: 26, lan: 27, ran: 28, lhe: 29, rhe: 30, lft: 31, rft: 32 } as const;
export const EDGES: [number, number][] = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32]];
export interface LandmarkLike { x: number; y: number; z: number; visibility?: number }
