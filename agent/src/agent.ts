// Ground Truth's iMessage agent on Photon's Spectrum. Text it SKI for a join code; it texts your time,
// rank and data contribution when you finish, and carries challenges between skiers.
import { Spectrum, type SpectrumInstance } from 'spectrum-ts';
import { imessage } from 'spectrum-ts/providers/imessage';
import { terminal } from 'spectrum-ts/providers/terminal';
import { DbConnection, tables } from './module_bindings/index.ts';
import { makeBrain } from './brain.ts';

const PROJECT_ID = process.env.PHOTON_PROJECT_ID;
const PROJECT_SECRET = process.env.PHOTON_PROJECT_SECRET;
const STDB_URI = process.env.STDB_URI ?? 'ws://127.0.0.1:3000';
const STDB_DB = process.env.STDB_DB ?? 'ground-truth';
const GAME_URL = process.env.GAME_URL ?? 'http://localhost:5173/';
const BOARD_URL = process.env.BOARD_URL ?? new URL('board.html', GAME_URL).toString();
const COURSE = process.env.COURSE_NAME ?? 'Streif';
const STATE_FILE = process.env.AGENT_STATE ?? '.agent-state.json';

const onIMessage = !!(PROJECT_ID && PROJECT_SECRET);
const app: SpectrumInstance<any> = onIMessage
  ? await Spectrum({ projectId: PROJECT_ID!, projectSecret: PROJECT_SECRET!, providers: [imessage.config()] })
  : await Spectrum({ providers: [terminal.config()] });

let brain: ReturnType<typeof makeBrain> | null = null;
const conn = DbConnection.builder()
  .withUri(STDB_URI).withDatabaseName(STDB_DB).withConfirmedReads(false)
  .onConnect((c, id) => {
    console.log('[stdb] connected as', id.toHexString().slice(0, 12));
    c.subscriptionBuilder()
      .onApplied(() => { console.log('[stdb] synced'); brain?.drainOutbox(); })
      .subscribe([tables.player, tables.run, tables.datasetStats, tables.outbox, tables.challenge, tables.skier, tables.runPhoto]);
    c.db.outbox.onInsert(() => brain?.drainOutbox());
  })
  .onConnectError((_c, e) => console.error('[stdb] connect error', e))
  .onDisconnect(() => console.warn('[stdb] disconnected'))
  .build();
brain = makeBrain({ conn, app, imessage: onIMessage, gameUrl: GAME_URL, boardUrl: BOARD_URL, course: COURSE, stateFile: STATE_FILE });

console.log(`[agent] ${onIMessage ? 'iMessage via Photon' : 'terminal mode (set PHOTON_PROJECT_ID and PHOTON_PROJECT_SECRET for iMessage)'} · game ${GAME_URL} · spacetimedb ${STDB_URI}/${STDB_DB}`);
for await (const [space, message] of app.messages) {
  brain.handle(space, message).catch(e => console.error('[agent] handler', e));
}
