// Ground Truth agent on Photon Spectrum, the one agent for the whole project: hikers get their app link and their
// hike summary (then a question about the trail, filed on that hike); skiers at the table get a join code, their
// time and finish photo, and challenges. Everything the team's SpacetimeDB module queues in its outbox is sent here.
//
//   node agent.mjs                       iMessage through Spectrum Cloud (SPECTRUM_PROJECT_ID / _SECRET)
//   node agent.mjs --terminal            same agent, chat with it in this terminal (no Photon account needed)
//   node agent.mjs nudge +19195550123    ask the server to text someone "heading out? record this one" (the running
//                                        agent sends it; later the app's trailhead geofence does this)
//
// Settings come from the environment or ground/agent/.env: SPECTRUM_PROJECT_ID, SPECTRUM_PROJECT_SECRET,
// XAI_API_KEY (Grok; without it replies are scripted), XAI_MODEL (default grok-4), GT_SERVER (default http://127.0.0.1:8770),
// STDB_URI (ws://127.0.0.1:3000 local, wss://maincloud.spacetimedb.com hosted; unset = no ski game), STDB_DB (ground-truth),
// GAME_URL, BOARD_URL, COURSE_NAME (Streif).
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const envPath = join(dirname(fileURLToPath(import.meta.url)), '.env')
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/)
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '')
  }
}

process.env.SPECTRUM_PROJECT_ID ||= process.env.PHOTON_PROJECT_ID  // the team's agent used these names
process.env.SPECTRUM_PROJECT_SECRET ||= process.env.PHOTON_PROJECT_SECRET
const { Spectrum, attachment } = await import('spectrum-ts')
const { createBrain, grok } = await import('./brain.mjs')
const { groundServer } = await import('./server.mjs')
const { Store, key } = await import('./state.mjs')

const server = groundServer()

if (process.argv[2] === 'nudge') {
  const to = process.argv[3]
  if (!to) { console.error('usage: node agent.mjs nudge +1XXXXXXXXXX'); process.exit(2) }
  const r = await server.trigger(key(to), new Store().code(key(to)))
  console.log(`[agent] queued a nudge to ${key(to)} with ${r.invite.link}. the running agent sends it within a few seconds.`)
  process.exit(0)
}

const useTerminal = process.argv.includes('--terminal')
const llm = grok()
if (!llm) console.warn('[agent] WARNING: no XAI_API_KEY, replies are SCRIPTED, not grok. add it to ground/agent/.env')
else console.log(`[agent] grok model ${process.env.XAI_MODEL || 'grok-4'}`)

let app, im
if (useTerminal) {
  const { terminal } = await import('spectrum-ts/providers/terminal')
  app = await Spectrum({ providers: [terminal.config()] })
} else {
  const { SPECTRUM_PROJECT_ID: projectId, SPECTRUM_PROJECT_SECRET: projectSecret } = process.env
  if (!projectId || !projectSecret) {
    console.error('[agent] SPECTRUM_PROJECT_ID and SPECTRUM_PROJECT_SECRET are not set. get them from your project ' +
      'settings at https://app.photon.codes and put them in ground/agent/.env, or run with --terminal to test locally.')
    process.exit(1)
  }
  const { imessage } = await import('spectrum-ts/providers/imessage')
  app = await Spectrum({ projectId, projectSecret, providers: [imessage.config()] })
  im = imessage(app)
}

const store = new Store()

// ------------------------------------------------------------------ the team's SpacetimeDB (ski game + hikes)
const STDB_URI = process.env.STDB_URI
const GAME_URL = process.env.GAME_URL || 'http://localhost:5173/'
let mountain = null
if (STDB_URI) {
  const { DbConnection, tables } = await import('./module_bindings/index.ts')
  const STDB_DB = process.env.STDB_DB || 'ground-truth'
  const conn = DbConnection.builder().withUri(STDB_URI).withDatabaseName(STDB_DB).withConfirmedReads(false)
    .onConnect((c, id) => {
      console.log(`[stdb] connected to ${STDB_URI}/${STDB_DB} as ${id.toHexString().slice(0, 12)}`)
      c.subscriptionBuilder()
        .onApplied(() => { console.log('[stdb] synced'); drainStdb() })
        .subscribe([tables.player, tables.run, tables.datasetStats, tables.hikeStats, tables.outbox, tables.challenge, tables.skier, tables.runPhoto])
      c.db.outbox.onInsert(() => drainStdb())
    })
    .onConnectError((_c, e) => console.error(`[stdb] CONNECT ERROR to ${STDB_URI}/${STDB_DB}:`, e?.message || e))
    .onDisconnect(() => console.error('[stdb] DISCONNECTED: ski results and hike summaries will not be sent until it reconnects'))
    .build()
  mountain = { conn, gameUrl: GAME_URL, boardUrl: process.env.BOARD_URL || new URL('board.html', GAME_URL).toString(),
    course: process.env.COURSE_NAME || 'Streif' }
} else {
  console.warn('[agent] WARNING: no STDB_URI, so no ski game and hike summaries come only from the local server outbox')
}

const brain = createBrain({ store, server, llm, mountain })

// One conversation at a time per person, everyone else in parallel.
const queues = new Map()
function serial(id, fn) {
  const next = (queues.get(id) || Promise.resolve()).then(fn, fn)
  queues.set(id, next.catch(() => {}))
  return next
}

async function sendTo(handle, parts) {
  if (useTerminal) {
    for (const t of parts) console.log(`\n[outbound to ${handle}] ${typeof t === 'string' ? t : '[finish photo]'}`)
    return
  }
  const space = await im.space.create(handle)
  for (const t of parts) {
    if (typeof t === 'string') await space.send(t)
    else await space.send(attachment(Buffer.from(t.photo, 'base64'), { name: 'finish.jpg', mimeType: 'image/jpeg' }))
      .catch((e) => console.error(`[agent] PHOTO FAILED to ${handle}:`, e.message))
  }
}

// Everything the SpacetimeDB module queued, sent once each, then marked sent in the database.
const delivered = new Set()
let draining = false
async function drainStdb() {
  if (!mountain || draining) return
  draining = true
  const { conn } = mountain
  try {
    for (const m of [...conn.db.outbox.iter()].filter((o) => !o.sent).sort((a, b) => Number(a.id - b.id))) {
      if (delivered.has(m.id)) continue
      let photo = null
      if (m.kind === 'result' && m.ref) {  // the finish photo lands a moment after the result; wait up to 4 s
        photo = conn.db.runPhoto.runKey.find(m.ref)?.jpeg ?? null
        if (!photo && Date.now() - Number(m.createdAt.microsSinceUnixEpoch / 1000n) < 4000) { setTimeout(drainStdb, 1200); continue }
      }
      delivered.add(m.id)
      const handle = store.byCode(m.joinCode)
      if (!handle) {
        console.error(`[agent] NO ONE HAS CODE ${m.joinCode}: ${m.kind} not sent (did they text ski or hike to this agent?)`)
      } else {
        await serial(key(handle), async () => {
          try {
            const out = brain.deliver(m, photo)
            await sendTo(handle, out.parts)
            console.log(`[agent] sent ${m.kind} to ${handle} (${m.joinCode})`)
          } catch (e) {
            console.error(`[agent] SEND FAILED for ${m.kind} to ${handle}:`, e.message)
          }
        })
      }
      await conn.reducers.markSent({ id: m.id }).catch((e) => console.error('[stdb] markSent failed:', e?.message || e))
    }
  } finally {
    draining = false
  }
}

// ------------------------------------------------------------------ outbox: summaries and nudges from the server
let pollFailing = false
setInterval(async () => {
  let box
  try {
    box = await server.outbox()
    if (pollFailing) console.log('[agent] ground server reachable again')
    pollFailing = false
  } catch (e) {
    if (!pollFailing) console.error('[agent] OUTBOX POLL FAILED:', e.message)
    pollFailing = true
    return
  }
  for (const msg of box) {
    await serial(key(msg.to), async () => {
      try {
        await sendTo(key(msg.to), brain.outbound(msg))
        await server.markSent(msg.id)
        console.log(`[agent] sent ${msg.kind || 'summary'} to ${key(msg.to)}`)
      } catch (e) {
        console.error(`[agent] SEND FAILED to ${key(msg.to)}:`, e.message)
        await server.markFailed(msg.id, e.message).catch((err) => console.error('[agent] could not mark failed:', err.message))
      }
    })
  }
}, 3000)

// ------------------------------------------------------------------ inbound
console.log(useTerminal ? '[agent] terminal mode: type a message to talk to ground truth.' : '[agent] listening on imessage.')
for await (const [space, message] of app.messages) {
  if (message.direction !== 'inbound' || message.content.type !== 'text') continue
  const from = useTerminal ? 'terminal' : message.sender?.id
  if (!from) { console.error('[agent] message with no sender, skipped'); continue }
  serial(key(from), async () => {
    try {
      const replies = await space.responding(() => brain.handle(from, message.content.text))
      for (const r of replies) await space.send(r)
    } catch (e) {
      console.error(`[agent] REPLY FAILED for ${key(from)}:`, e)
      await space.send("sorry, something broke on our end. try again in a minute?").catch(() => {})
    }
  })
}
