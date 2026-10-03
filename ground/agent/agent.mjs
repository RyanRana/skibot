// Ground Truth Photon agent: texts hikers the opt-in link and sends their hike summary when it is ready.
// Runs on a Mac signed into Messages, with Full Disk Access for the terminal (Photon's kit reads chat.db).
//
//   node agent.mjs                    watch for "hike" texts and send whatever the server's outbox holds
//   node agent.mjs invite +19195550123  text one person the opt-in link now
//
// GT_SERVER defaults to http://127.0.0.1:8770 (ground/server.py).
import { IMessageSDK } from '@photon-ai/imessage-kit'

const SERVER = process.env.GT_SERVER || 'http://127.0.0.1:8770'
const TRIGGER = /\b(hike|hiking|join|ground ?truth|opt ?in)\b/i

const inviteText = (link) =>
  `hey, it's ground truth. want this hike to help train rescue robots? ` +
  `your phone records motion + gps while you walk, nothing else, and you can delete it anytime.\n\n` +
  `opt in here: ${link}`

async function api(path, body) {
  const r = await fetch(SERVER + path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const j = await r.json().catch(() => ({ error: `non-json reply, HTTP ${r.status}` }))
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${j.error || ''}`)
  return j
}

async function invite(sdk, to) {
  const inv = await api('/api/invite', { phone: to })
  await sdk.send({ to, text: inviteText(inv.link) })
  console.log(`[agent] invited ${to} -> ${inv.link}`)
}

async function drainOutbox(sdk) {
  for (const m of await api('/api/outbox')) {
    try {
      await sdk.send({ to: m.to, text: m.text })
      await api(`/api/outbox/${m.id}/sent`, {})
      console.log(`[agent] sent summary to ${m.to}`)
    } catch (e) {
      await api(`/api/outbox/${m.id}/failed`, { error: String(e.message || e) })
      console.error(`[agent] SEND FAILED to ${m.to}:`, e)
    }
  }
}

const sdk = new IMessageSDK()

if (process.argv[2] === 'invite') {
  const to = process.argv[3]
  if (!to) { console.error('usage: node agent.mjs invite +1XXXXXXXXXX'); process.exit(2) }
  await invite(sdk, to)
  await sdk.close()
  process.exit(0)
}

await sdk.startWatching({
  onDirectMessage: async (msg) => {
    if (!msg.text || !TRIGGER.test(msg.text) || !msg.participant) return
    try { await invite(sdk, msg.participant) } catch (e) { console.error('[agent] INVITE FAILED:', e) }
  },
  onError: (e) => console.error('[agent] WATCHER ERROR:', e),
})
console.log(`[agent] watching Messages for "hike", outbox at ${SERVER}`)

let failing = false
setInterval(async () => {
  try {
    await drainOutbox(sdk)
    if (failing) console.log('[agent] server reachable again')
    failing = false
  } catch (e) {
    if (!failing) console.error('[agent] OUTBOX POLL FAILED (is ground/server.py running?):', e.message)
    failing = true
  }
}, 3000)
