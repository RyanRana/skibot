// The agent's brain: join codes, commands and outbox delivery. Independent of how messages arrive, so
// Photon's Spectrum drives it in production and a scripted fake drives it in tests. No phone number
// ever reaches the database: the brain keeps the code -> phone map, SpacetimeDB only sees the code.
import { markdown, richlink, attachment, type Space, type Message } from 'spectrum-ts';
import { imessage } from 'spectrum-ts/providers/imessage';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import type { DbConnection } from './module_bindings/index.ts';

export interface BrainEnv {
  conn: DbConnection;
  app: unknown;             // the Spectrum instance (for proactive sends over iMessage)
  imessage: boolean;        // true when running on Photon, false in terminal/test mode
  gameUrl: string; boardUrl: string; course: string; stateFile: string;
  log?: (s: string) => void;
}

type Link = { sender: string; spaceId: string; platform: string; created: number };

export const fmtTime = (ms: number) => `${Math.floor(ms / 60000)}:${((ms % 60000) / 1000).toFixed(2).padStart(5, '0')}`;
const n = (x: number | bigint) => Number(x).toLocaleString('en-US');

export function makeBrain(env: BrainEnv) {
  const { conn, gameUrl, boardUrl, course } = env;
  const log = env.log ?? ((s: string) => console.log(s));
  const links: Record<string, Link> = existsSync(env.stateFile) ? JSON.parse(readFileSync(env.stateFile, 'utf8')) : {};
  const save = () => writeFileSync(env.stateFile, JSON.stringify(links, null, 1));
  const codeFor = (sender: string) => Object.entries(links).find(([, l]) => l.sender === sender)?.[0];
  const newCode = () => { const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; let c = ''; do { c = Array.from({ length: 4 }, () => A[Math.floor(Math.random() * A.length)]).join(''); } while (links[c]); return c; };
  const spaces = new Map<string, Space>();

  const runs = () => [...conn.db.run.iter()].filter(r => r.finished).sort((a, b) => a.timeMs - b.timeMs);
  const playerByCode = (code: string) => [...conn.db.player.iter()].find(p => p.joinCode === code);
  const playerByName = (name: string) => [...conn.db.player.iter()].find(p => p.name.toLowerCase() === name.toLowerCase());
  const stats = () => conn.db.datasetStats.id.find(0);

  async function spaceFor(link: Link): Promise<Space | undefined> {
    const cached = spaces.get(link.spaceId);
    if (cached) return cached;
    if (!env.imessage) return undefined; // terminal/test: only a live conversation can be messaged
    const im = imessage(env.app as any);
    const space = await im.space.get(link.spaceId).catch(async () => im.space.create(await im.user(link.sender)));
    spaces.set(link.spaceId, space);
    return space;
  }

  // Deliver what the module queued, by join code.
  let draining = false;
  async function drainOutbox() {
    if (draining) return;
    draining = true;
    try {
      for (const m of [...conn.db.outbox.iter()].filter(o => !o.sent).sort((a, b) => Number(a.id - b.id))) {
        const link = links[m.joinCode];
        if (link) {
          try {
            const space = await spaceFor(link);
            if (space) {
              // Results: wait a moment for the finish photo to land, then send it with the text.
              if (m.kind === 'result' && m.ref) {
                const photo = conn.db.runPhoto.runKey.find(m.ref);
                const ageMs = Date.now() - Number(m.createdAt.microsSinceUnixEpoch / 1000n);
                if (!photo && ageMs < 4000) { setTimeout(() => drainOutbox(), 1200); continue; }
                if (photo) { try { await space.send(attachment(Buffer.from(photo.jpeg, 'base64'), { name: 'finish.jpg', mimeType: 'image/jpeg' })); } catch (e) { log(`[agent] photo failed: ${e}`); } }
              }
              await space.send(m.body);
              if (m.kind === 'result') await space.send(richlink(boardUrl));
              log(`[agent] -> ${m.joinCode} (${m.kind}): ${m.body.slice(0, 90)}`);
            } else log(`[agent] no conversation for ${m.joinCode}, dropping ${m.kind}`);
          } catch (e) { log(`[agent] send failed: ${e}`); }
        }
        await conn.reducers.markSent({ id: m.id }).catch(() => {});
      }
    } finally { draining = false; }
  }

  const help = () => markdown([
    `**Ground Truth** · ski the ${course} with your body.`,
    `• **SKI** — get a join code for the laptop`,
    `• **TOP** — today's fastest`,
    `• **CHALLENGE <name>** — send someone your best time`,
    `• **STATS** — how big the dataset is`,
    `• **MAP** — who is on the mountain right now`,
  ].join('\n'));

  async function handle(space: Space, msg: Message) {
    if (msg.direction === 'outbound' || msg.content.type !== 'text') return;
    const sender = msg.sender?.id ?? space.id;
    spaces.set(space.id, space);
    const text = msg.content.text.trim();
    const [cmd = '', ...rest] = text.split(/\s+/);
    const c = cmd.toLowerCase().replace(/[^a-z]/g, '');
    const arg = rest.join(' ').trim();
    await space.responding(async () => {
      if (['ski', 'join', 'play', 'start', 'hi', 'hello', 'hey', 'yo'].includes(c)) {
        let code = codeFor(sender);
        if (!code) { code = newCode(); links[code] = { sender, spaceId: space.id, platform: msg.platform, created: Date.now() }; save(); }
        const url = `${gameUrl}${gameUrl.includes('?') ? '&' : '?'}code=${code}`;
        await msg.reply(markdown(`Your code is **${code}**. Type it on the laptop with your name, stand two metres from the camera, and raise both hands to start.\n\nLean to carve, crouch to tuck, hop to jump. I'll text your time when you cross the line.`));
        await space.send(richlink(url));
      } else if (['top', 'leaderboard', 'board', 'best', 'fastest'].includes(c)) {
        const top = runs().slice(0, 5);
        await msg.reply(top.length
          ? markdown(`**Fastest on the ${course}**\n` + top.map((r, i) => `${i + 1}. ${r.name} — ${fmtTime(r.timeMs)} · ${r.gatesHit}/${r.gatesTotal} gates`).join('\n'))
          : `Nobody has finished the ${course} yet. Text SKI to be first.`);
      } else if (['stats', 'data', 'dataset', 'size'].includes(c)) {
        const s = stats();
        await msg.reply(s
          ? markdown(`**Ground Truth** holds **${n(s.samples)}** motion samples on real terrain, from **${n(s.skiers)}** skiers over **${n(s.runs)}** runs and **${n(Math.round(s.meters))} m** of piste. Every run adds to it.`)
          : 'The mountain is still waking up. Try again in a moment.');
      } else if (['map', 'mountain', 'live', 'who'].includes(c)) {
        const live = [...conn.db.skier.iter()].filter(s => s.phase === 2);
        await msg.reply(live.length ? `${live.length} skiing right now: ${live.map(s => `${s.name} (${Math.round(s.speed * 3.6)} km/h, ${Math.round(s.progress * 100)}% down)`).join(', ')}.` : 'Nobody is on the course right now.');
        await space.send(richlink(boardUrl));
      } else if (['challenge', 'race', 'dare', 'beat'].includes(c)) {
        const code = codeFor(sender);
        const me = code ? playerByCode(code) : undefined;
        if (!me) { await msg.reply('Ski a run first so I know who you are: text SKI, then type the code on the laptop.'); return; }
        const best = runs().filter(r => r.identity.isEqual(me.identity))[0];
        if (!best) { await msg.reply(`You haven't finished the ${course} yet. Finish a run and I'll let you challenge anyone with it.`); return; }
        if (!arg) { await msg.reply('Who? Text CHALLENGE <name>, using the name they skied under.'); return; }
        const target = playerByName(arg);
        await conn.reducers.createChallenge({ fromName: me.name, toName: target?.name ?? arg, course, targetTimeMs: best.timeMs });
        await msg.reply(target
          ? `Done. ${target.name} has to beat your ${fmtTime(best.timeMs)} on the ${course}. I'll text you when they try.`
          : `Challenge posted for "${arg}". When they join under that name they'll see it on the start line.`);
      } else if (['me', 'code', 'mycode'].includes(c)) {
        const code = codeFor(sender);
        const p = code ? playerByCode(code) : undefined;
        await msg.reply(code ? `Your code is ${code}${p ? `, skiing as ${p.name}, best ${p.bestTimeMs ? fmtTime(p.bestTimeMs) : 'no finish yet'}` : ' (not linked to a run yet)'}.` : 'No code yet. Text SKI.');
      } else {
        await msg.reply(help());
      }
    });
  }

  return { handle, drainOutbox, spaces, links, codeFor };
}
