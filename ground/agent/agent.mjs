// Ground Truth agent on Photon Spectrum: talks with hikers over iMessage, sends their app link, sends the summary
// when a hike is registered, asks about the trail and files the answer on that hike.
//
//   node agent.mjs                       iMessage through Spectrum Cloud (SPECTRUM_PROJECT_ID / _SECRET)
//   node agent.mjs --terminal            same agent, chat with it in this terminal (no Photon account needed)
//   node agent.mjs nudge +19195550123    ask the server to text someone "heading out? record this one" (the running
//                                        agent sends it; later the app's trailhead geofence does this)
//
// Settings come from the environment or ground/agent/.env: SPECTRUM_PROJECT_ID, SPECTRUM_PROJECT_SECRET,
// XAI_API_KEY (Grok; without it replies are scripted), XAI_MODEL (default grok-4), GT_SERVER (default http://127.0.0.1:8770).
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

const { Spectrum } = await import('spectrum-ts')
const { createBrain, grok } = await import('./brain.mjs')
const { groundServer } = await import('./server.mjs')
const { Store, key } = await import('./state.mjs')

const server = groundServer()

if (process.argv[2] === 'nudge') {
  const to = process.argv[3]
  if (!to) { console.error('usage: node agent.mjs nudge +1XXXXXXXXXX'); process.exit(2) }
  const r = await server.trigger(key(to))
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
const brain = createBrain({ store, server, llm })

// One conversation at a time per person, everyone else in parallel.
const queues = new Map()
function serial(id, fn) {
  const next = (queues.get(id) || Promise.resolve()).then(fn, fn)
  queues.set(id, next.catch(() => {}))
  return next
}

async function sendTo(handle, texts) {
  if (useTerminal) {
    for (const t of texts) console.log(`\n[outbound to ${handle}] ${t}`)
    return
  }
  const space = await im.space.create(handle)
  for (const t of texts) await space.send(t)
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
