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
  assert.match(out[0], /ground truth/i); assert.match(out[0], /reply yes/i)
  ok('first text gets the intro')
  out = await s.handle(PHONE, 'yes!')
  assert.match(out[0], /\/g\/[A-Za-z0-9_-]+/i)
  ok('yes gets a real invite link')
  out = await s.handle(PHONE, 'how did my hike go')
  assert.match(out[0], /no hikes yet/i)
  ok('no invented hikes')

  // ---------------------------------------------------------- outbound summary, follow-up, label
  const sid = await consentedSession(PHONE)
  const texts = s.outbound({ to: PHONE, kind: 'summary', session: sid, text: 'ground truth: 2.10 km on deer track trail' })
  assert.equal(texts.length, 2); assert.match(texts[1], /\?/i)
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
  assert.equal(calls[1].role, 'tool'); assert.match(JSON.parse(calls[1].content).link, /\/g\//i)
  assert.match(out.join('\n'), /\/g\/[A-Za-z0-9_-]+/i)
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
  assert.match(out[0], /reply yes/i)
  out = await s.handle(PHONE, 'actually no')
  assert.match(out[0], /nothing deleted/i)
  assert.equal((await server.hikes(PHONE)).length, 1)
  ok('delete needs an explicit yes')
  await s.handle(PHONE, 'delete my data')
  out = await s.handle(PHONE, 'yes')
  assert.match(out[0], /deleted 1 hike/i)
  assert.equal((await server.hikes(PHONE)).length, 0)
  assert.equal(store.get(PHONE).notes.length, 0)
  ok('yes deletes their hikes, invites and messages')

  // ---------------------------------------------------------- ski game commands + SpacetimeDB outbox delivery
  out = await s.handle(PHONE, 'ski')
  assert.match(out[0], /isn't connected/i)
  ok('game commands say so loudly when there is no spacetimedb')
  const me = { isEqual: (x) => x === me }, other = { isEqual: (x) => x === other }
  const fakeConn = {
    db: {
      run: { iter: () => [{ finished: true, timeMs: 61230, name: 'Ryan', gatesHit: 20, gatesTotal: 22, identity: other }].values() },
      player: { iter: () => [].values() }, skier: { iter: () => [].values() },
      datasetStats: { id: { find: () => ({ samples: 5459n, skiers: 26, runs: 6n }) } },
      hikeStats: { id: { find: () => ({ hikes: 2n, meters: 7100, motionSamples: 720000n, hikers: 1 }) } },
    },
    reducers: {},
  }
  const m = createBrain({ store, server, llm: null, mountain: { conn: fakeConn, gameUrl: 'http://game/', boardUrl: 'http://game/board.html', course: 'Streif' } })
  out = await m.handle(PHONE, 'SKI')
  const code = store.get(PHONE).code
  assert.match(code, /^[A-Z2-9]{4}$/i); assert.match(out[0], new RegExp(code)); assert.equal(out[1], `http://game/?code=${code}`)
  ok('ski gives this person their join code and the game link')
  out = await m.handle(PHONE, 'top')
  assert.match(out[0], /1\. Ryan — 1:01\.23/i)
  out = await m.handle(PHONE, 'stats')
  assert.match(out[0], /5,459 ski motion samples/i); assert.match(out[0], /2 real hikes \(7\.1 km/i)
  ok('top and stats read the live database')
  out = await m.handle(PHONE, "i'm going to ski tomorrow with my friends")
  assert.doesNotMatch(out.join(' '), /your code is/i)
  ok('a sentence that mentions ski is not a command')
  const inv2 = await server.invite(PHONE, code)
  assert.equal(inv2.code, code)
  ok('invites carry the join code to the server')

  assert.equal(m.deliver({ joinCode: 'ZZZZ', kind: 'result', body: 'x' }), null)
  ok('outbox rows for unknown codes are not sent to anyone')
  let d = m.deliver({ joinCode: code.toLowerCase(), kind: 'result', body: 'Yashu: Streif in 1:05.10', ref: 'run1' }, 'AAAA')
  assert.equal(d.handle, PHONE); assert.deepEqual(d.parts[0], { photo: 'AAAA' }); assert.equal(d.parts.at(-1), 'http://game/board.html')
  ok('ski results go out with the finish photo and the board link')
  const hikeSid = await consentedSession(PHONE)
  d = m.deliver({ joinCode: code, kind: 'hike', body: 'ground truth: 5.20 km', ref: hikeSid })
  assert.equal(d.parts.length, 2); assert.equal(store.get(PHONE).pendingLabel.session, hikeSid)
  await m.handle(PHONE, 'muddy switchbacks after the bridge')
  assert.deepEqual((await server.hikes(PHONE)).find((h) => h.session === hikeSid).labels, ['muddy switchbacks after the bridge'])
  ok('hike summaries from the database ask the trail question, and the answer lands on that hike')

  console.log(`\n${passed} passed`)
} finally {
  await server.deleteUser(PHONE).catch(() => {})
  rmSync(statePath, { force: true })
  rmSync(statePath + '.tmp', { force: true })
}
