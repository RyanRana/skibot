// End-to-end checks of the agent brain against the running ground server (ground/run.sh), with a throwaway
// memory file and a fake phone number whose data is deleted at the end. Grok is replaced by a stand-in that
// exercises the tool loop the same way, so this needs no API keys.  node test.mjs
import assert from 'node:assert/strict'
import { rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createBrain } from './brain.mjs'
import { groundServer } from './server.mjs'
import { Store } from './state.mjs'

const PHONE = '+15550001234'
const server = groundServer()
const statePath = join(tmpdir(), `gt-agent-test-${process.pid}.json`)
const store = new Store(statePath)
let passed = 0
const ok = (name) => { passed++; console.log(`  ok  ${name}`) }

async function consentedSession(phone) {
  const inv = await server.invite(phone)
  const sid = crypto.randomUUID()
  const r = await fetch(`${server.base}/api/consent`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: inv.token, session_id: sid, consent_version: 'gt-consent-1' }) })
  assert.equal(r.status, 200)
  return sid
}

try {
  // ---------------------------------------------------------- scripted (no grok)
  const s = createBrain({ store, server, llm: null })
  let out = await s.handle(PHONE, 'hi')
  assert.match(out[0], /ground truth/); assert.match(out[0], /reply yes/)
  ok('first text gets the intro')
  out = await s.handle(PHONE, 'yes!')
  assert.match(out[0], /\/g\/[A-Za-z0-9_-]+/)
  ok('yes gets a real invite link')
  out = await s.handle(PHONE, 'how did my hike go')
  assert.match(out[0], /no hikes yet/)
  ok('no invented hikes')

  // ---------------------------------------------------------- outbound summary, follow-up, label
  const sid = await consentedSession(PHONE)
  const texts = s.outbound({ to: PHONE, kind: 'summary', session: sid, text: 'ground truth: 2.10 km on deer track trail' })
  assert.equal(texts.length, 2); assert.match(texts[1], /\?/)
  ok('summary is followed by a question about the trail')
  out = await s.handle(PHONE, 'slipped twice on wet roots near the creek')
  const hikes = await server.hikes(PHONE)
  assert.deepEqual(hikes.find((h) => h.session === sid).labels, ['slipped twice on wet roots near the creek'])
  ok('the answer is saved as a label on that hike')

  // ---------------------------------------------------------- grok tool loop (stand-in model)
  const calls = []
  const fakeGrok = async (messages, tools) => {
    calls.push(messages.at(-1))
    assert.ok(tools.find((t) => t.function.name === 'send_invite'))
    const last = messages.at(-1)
    if (last.role === 'user') return { content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'send_invite', arguments: '{}' } }] }
    return { content: "here you go, tap it when you're at the trailhead" }  // forgets to include the link
  }
  const g = createBrain({ store, server, llm: fakeGrok })
  out = await g.handle(PHONE, 'can i get the link again')
  assert.equal(calls[1].role, 'tool'); assert.match(JSON.parse(calls[1].content).link, /\/g\//)
  assert.match(out.join('\n'), /\/g\/[A-Za-z0-9_-]+/)
  ok('tool result flows back to the model, and the link is added if the model drops it')
  assert.ok(store.get(PHONE).history.length >= 6)
  ok('conversation memory persists across turns')

  // ---------------------------------------------------------- nudge (trailhead trigger)
  const t = await server.trigger(PHONE)
  const box = await server.outbox()
  assert.ok(box.find((m) => m.id === t.message.id && m.kind === 'nudge' && m.text.includes(t.invite.link)))
  ok('trigger queues a nudge with a fresh link')

  // ---------------------------------------------------------- stop / delete
  out = await s.handle(PHONE, 'STOP')
  assert.throws(() => s.outbound({ to: PHONE, kind: 'nudge', text: 'x' }), /opted out/)
  ok('stop blocks every outbound text')
  out = await s.handle(PHONE, 'start')
  out = await s.handle(PHONE, 'please delete my data')
  assert.match(out[0], /reply yes/)
  out = await s.handle(PHONE, 'actually no')
  assert.match(out[0], /nothing deleted/)
  assert.equal((await server.hikes(PHONE)).length, 1)
  ok('delete needs an explicit yes')
  await s.handle(PHONE, 'delete my data')
  out = await s.handle(PHONE, 'yes')
  assert.match(out[0], /deleted 1 hike/)
  assert.equal((await server.hikes(PHONE)).length, 0)
  assert.equal(store.get(PHONE).notes.length, 0)
  ok('yes deletes their hikes, invites and messages')

  console.log(`\n${passed} passed`)
} finally {
  await server.deleteUser(PHONE).catch(() => {})
  rmSync(statePath, { force: true })
  rmSync(statePath + '.tmp', { force: true })
}
