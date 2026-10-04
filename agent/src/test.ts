// Scripted conversation against the live SpacetimeDB, with a fake iMessage space. Run: npx tsx src/test.ts
import { DbConnection, tables } from './module_bindings/index.ts';
import { makeBrain } from './brain.ts';
import type { Space, Message } from 'spectrum-ts';

const STDB_URI = process.env.STDB_URI ?? 'ws://127.0.0.1:3000';
const STDB_DB = process.env.STDB_DB ?? 'ground-truth';
const sent: string[] = [];
const show = async (c: any) => { if (typeof c === 'string') return c; const b = c?.build ? await c.build() : c; return b?.markdown ?? b?.text ?? b?.url ?? (b?.type === 'attachment' ? `[attachment ${b.name ?? ''} ${b.mimeType ?? ''}]` : JSON.stringify(b).slice(0, 120)); };
const space = { id: 'any;-;+15550100', __platform: 'imessage', send: async (...c: any[]) => { for (const x of c) sent.push(await show(x)); return undefined; }, responding: async (fn: any) => fn() } as unknown as Space;
const msg = (text: string) => ({ direction: 'inbound', platform: 'imessage', sender: { id: '+15550100' }, space, content: { type: 'text', text }, reply: async (...c: any[]) => { for (const x of c) sent.push(await show(x)); return undefined; } }) as unknown as Message;

const conn = DbConnection.builder().withUri(STDB_URI).withDatabaseName(STDB_DB).withConfirmedReads(false)
  .onConnect((c) => { c.subscriptionBuilder().onApplied(() => run().catch(e => { console.error(e); process.exit(1); })).subscribe([tables.player, tables.run, tables.datasetStats, tables.outbox, tables.challenge, tables.skier, tables.runPhoto]); })
  .onConnectError((_c, e) => { console.error('connect', e); process.exit(1); })
  .build();
const brain = makeBrain({ conn, app: null, imessage: false, gameUrl: 'http://localhost:5173/', boardUrl: 'http://localhost:5173/board.html', course: 'Streif', stateFile: '/tmp/gt-agent-test-state.json', log: s => sent.push(s) });
conn.db.outbox.onInsert(() => brain.drainOutbox());

const say = async (t: string) => { sent.length = 0; await brain.handle(space, msg(t)); console.log(`\n> ${t}`); for (const s of sent) console.log('  <', s.replace(/\n/g, '\n    ')); };
const wait = (ms: number) => new Promise(r => setTimeout(r, ms));

async function run() {
  await say('hi');
  await say('ski');
  const code = brain.codeFor('+15550100')!;
  console.log('\n[test] code', code, '- now the laptop joins with it and skis a run (this connection plays the skier)');
  await conn.reducers.join({ name: 'Tester', joinCode: code, course: 'Streif' });
  const key = `test-${Date.now()}`;
  await conn.reducers.startRun({ key, course: 'Streif', gatesTotal: 60 });
  await conn.reducers.pushTrace({ runKey: key, seq: 0, samples: Array.from({ length: 40 }, (_, i) => ({ tMs: i * 50, x: i, y: 0, z: -i * 0.2, speed: 10, heading: 1.6, lean: 0.2, crouch: 0.5, kneeL: 1, kneeR: 1, slopeDeg: 15 })) });
  await conn.reducers.finishRun({ key, timeMs: 95_430, gatesHit: 52, maxSpeed: 23.4, distanceM: 1200, splits: Array.from({ length: 60 }, (_, i) => (i + 1) * 1500) });
  await conn.reducers.pushPhoto({ runKey: key, jpeg: Buffer.from('not really a jpeg').toString('base64') });
  sent.length = 0; await wait(3200);
  console.log('\n[test] photo row present:', !!conn.db.runPhoto.runKey.find(key));
  console.log('[test] after the run, the outbox delivered:'); for (const s of sent) console.log('  <', s);
  await say('top');
  await say('stats');
  await say('map');
  await say('challenge Ryan');
  await say('me');
  await say('what is this');
  const open = [...conn.db.challenge.iter()].filter(c => c.status === 'open').length;
  const unsent = [...conn.db.outbox.iter()].filter(o => !o.sent).length;
  console.log(`\n[test] open challenges ${open}, unsent outbox ${unsent}`);
  process.exit(0);
}
