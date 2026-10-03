// The conversation: who this person is to us, what they asked, what we say back. Safety-critical turns (stop,
// delete, confirming a delete) are plain code, never left to the model. Everything else goes to Grok with a few
// tools; with no XAI_API_KEY the agent runs scripted replies instead and says so at startup.

const STOP = /^\s*(stop|unsubscribe|quit|stopall|cancel)\s*[.!]*\s*$/i
const START = /^\s*(start|unstop|resume)\s*[.!]*\s*$/i
const DELETE = /\b(delete|erase|remove|wipe)\b[^?]*\b(data|everything|hikes?|recordings?|info|me)\b|^\s*delete\s*$/i
const CONFIRM = /^\s*(yes|y|yep|yeah|confirm|delete( it| everything)?|do it)\s*[.!]*\s*$/i
const YES = /^\s*(y|ya|yes|yeah|yep|yup|sure|ok|okay|i'?m in|im in|let'?s go|down|do it|sounds good|send it)\b/i
const LABEL_TTL_MS = 24 * 3600 * 1000

const INTRO = "hey! this is ground truth. we turn hikes into training data for rescue robots: your phone records how you " +
  "move on the trail while you hike, and nothing else. want in? reply yes and i'll send the app link."

const FOLLOWUPS = [
  "quick one: anything tricky out there? mud, loose rock, a slip, a stream crossing? whatever you tell me gets attached to this hike.",
  "how was the ground today? any slick spots, scrambles or places you slowed down? i'll attach it to the hike.",
  "anything worth noting about the trail? wet roots, scree, a sketchy descent? it helps the robots know what they're looking at.",
]

const FACTS = `what ground truth is:
- hikers record a hike with the ground truth iphone app. it captures motion (accelerometer + gyroscope, 100 times a second), gps, altitude from the barometer, and steps.
- that data is matched to the exact trail and terrain, then used to train humanoid robots in simulation to move on real ground, so they can one day do search and rescue on trails.
- recording only happens during a hike the person starts and ends in the app. it keeps going with the phone locked in a pocket, but stops if they swipe the app closed.
- never recorded: contacts, photos, audio, messages.
- the first and last 200 m of every route are trimmed before anything is shown on a map, so no one's home shows up.
- battery use is about like a fitness app.
- after they end a hike, it uploads and a summary text arrives within a few minutes.
- they can delete everything anytime by texting "delete my data", and stop texts with "stop".`

const TOOLS = [
  { type: 'function', function: {
    name: 'send_invite',
    description: 'Create this person\'s personal link that opens the ground truth app to record a hike. Use when they want to record, join, or ask for the link/app.',
    parameters: { type: 'object', properties: {}, additionalProperties: false } } },
  { type: 'function', function: {
    name: 'get_my_hikes',
    description: 'Their recorded hikes, newest first: date, duration, distance, climb, trail, motion samples, gaps, notes they gave. The only source for any number you state about their hikes.',
    parameters: { type: 'object', properties: {}, additionalProperties: false } } },
  { type: 'function', function: {
    name: 'remember',
    description: 'Save a short lasting fact about this person for future conversations (their name, home trails, how they carry their phone, preferences). Not for hike observations.',
    parameters: { type: 'object', properties: { note: { type: 'string' } }, required: ['note'], additionalProperties: false } } },
  { type: 'function', function: {
    name: 'add_hike_note',
    description: 'Attach what they said about trail conditions or events (mud, a slip, scree, a stream crossing, where they slowed) to their most recent hike.',
    parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'], additionalProperties: false } } },
]

function systemPrompt(p, extra) {
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
  return `you are ground truth, texting someone over imessage. today is ${today}.

${FACTS}

how you text:
- like a friendly person texting, all lowercase, usually 1-2 short sentences. no markdown, no lists, no headers. emojis rarely.
- warm and direct, a little playful about hiking. never pushy: if they're not interested, that's fine.
- you can make small talk about hiking, trails, gear or robots, but keep it short and don't pretend to know their area.
- if they mention an injury, being lost or an emergency, tell them to call 911 first, before anything else.

rules:
- never state a number about their hikes (distance, time, climb, samples) unless it came from get_my_hikes in this conversation.
- to give them the app link, call send_invite. never make up a link.
- if they tell you something lasting about themselves, call remember. if they describe trail conditions or what happened on a hike, call add_hike_note.
- don't claim to do things you can't: you can't see their location, start a recording for them, or text anyone else.
- deleting data and stopping texts are handled outside of you. if asked, tell them to text "delete my data" or "stop".

what you know about this person: ${p.notes.length ? p.notes.join('; ') : 'nothing yet'}.${p.history.length ? '' : ' this is your first conversation with them.'}${extra ? '\n\n' + extra : ''}`
}

function hikeLine(h) {
  if (h.status !== 'done') return `a hike from ${day(h.started)} is still ${h.status || 'processing'}`
  const parts = [h.distance_m != null && `${(h.distance_m / 1000).toFixed(2)} km`, h.trail && `on ${h.trail.toLowerCase()}`,
    h.climb_m != null && `+${Math.round(h.climb_m)} m`, h.duration_s && dur(h.duration_s)].filter(Boolean)
  return `${day(h.started)}: ${parts.join(', ')}`
}
const day = (t) => (t ? new Date(t * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toLowerCase() : 'recently')
const dur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h${String(Math.floor(s / 60) % 60).padStart(2, '0')}m` : `${Math.max(1, Math.round(s / 60))} min`)

export function createBrain({ store, server, llm, log = console }) {
  // ---------------------------------------------------------------- inbound
  async function handle(handle, raw) {
    const p = store.get(handle)
    const text = (raw || '').trim()
    if (!text) return []
    store.remember(handle, 'user', text)
    const out = await route(p, handle, text)
    for (const r of out) store.remember(handle, 'assistant', r)
    return out
  }

  async function route(p, handle, text) {
    if (STOP.test(text)) {
      p.optedOut = true
      p.pendingDelete = false
      store.save()
      return ["got it, you won't get any more texts from us unless you text first. text start to turn them back on."]
    }
    if (START.test(text) && p.optedOut) {
      p.optedOut = false
      store.save()
      return ["you're back on. text me anytime you head out on a trail."]
    }
    if (p.pendingDelete) {
      p.pendingDelete = false
      store.save()
      if (!CONFIRM.test(text)) return ['ok, nothing deleted.']
      const r = await server.deleteUser(handle)
      p.notes = []
      p.pendingLabel = null
      p.history = []
      store.save()
      const n = r.deleted.hikes
      return [`done. deleted ${n} hike${n === 1 ? '' : 's'} and everything tied to your number on our side.`]
    }
    if (DELETE.test(text)) {
      p.pendingDelete = true
      store.save()
      return ['just to be sure: this permanently deletes all your recorded hikes and anything you told me. reply yes to delete.']
    }

    let extra = ''
    if (p.pendingLabel && Date.now() - p.pendingLabel.at < LABEL_TTL_MS && !/\?\s*$/.test(text)) {
      await server.label(p.pendingLabel.session, text)
      extra = 'their last message answered your question about the hike and is already saved as a note on it. thank them briefly; don\'t call add_hike_note for it.'
      p.pendingLabel = null
      store.save()
    }
    return llm ? converse(p, handle, text, extra) : scripted(p, handle, text, extra)
  }

  async function converse(p, handle, text, extra) {
    const messages = [{ role: 'system', content: systemPrompt(p, extra) },
      ...p.history.slice(0, -1).map(({ role, content }) => ({ role, content })),
      { role: 'user', content: text }]
    let link = null
    for (let step = 0; step < 5; step++) {
      const msg = await llm(messages, TOOLS)
      if (!msg.tool_calls?.length) {
        let reply = (msg.content || '').trim()
        if (!reply) throw new Error('grok returned an empty reply')
        if (link && !reply.includes(link)) reply += `\n\n${link}`
        return split(reply)
      }
      messages.push({ role: 'assistant', content: msg.content || '', tool_calls: msg.tool_calls })
      for (const call of msg.tool_calls) {
        let args = {}
        try { args = JSON.parse(call.function.arguments || '{}') } catch { /* tool gets {} */ }
        let result
        try {
          result = await tool(p, handle, call.function.name, args)
          if (call.function.name === 'send_invite') link = result.link
        } catch (e) {
          log.error(`[agent] TOOL ${call.function.name} FAILED:`, e.message)
          result = { error: e.message }
        }
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) })
      }
    }
    throw new Error('grok kept calling tools without replying')
  }

  async function tool(p, handle, name, args) {
    if (name === 'send_invite') {
      const inv = await server.invite(handle)
      p.invites += 1
      store.save()
      return { link: inv.link, note: 'include this exact link in your reply' }
    }
    if (name === 'get_my_hikes') {
      const hikes = (await server.hikes(handle)).slice(0, 5)
      return hikes.length ? hikes.map((h) => ({ ...h, date: day(h.started), session: undefined, started: undefined }))
        : { hikes: [], note: 'they have not recorded a hike yet' }
    }
    if (name === 'remember') {
      const note = String(args.note || '').trim().slice(0, 200)
      if (note && !p.notes.includes(note)) p.notes = [...p.notes, note].slice(-20)
      store.save()
      return { ok: true }
    }
    if (name === 'add_hike_note') {
      const [latest] = await server.hikes(handle)
      if (!latest) return { error: 'no hikes recorded yet' }
      await server.label(latest.session, String(args.text || '').slice(0, 2000))
      return { ok: true, hike: day(latest.started) }
    }
    return { error: `unknown tool ${name}` }
  }

  async function scripted(p, handle, text, extra) {
    if (extra) return ['thanks, added that to your hike.']
    if (/\b(how did|how was|last|latest|stats|went)\b/i.test(text)) {
      const hikes = await server.hikes(handle)
      return [hikes.length ? `your latest: ${hikeLine(hikes[0])}.` : "no hikes yet. reply hike when you're heading out and i'll send the link."]
    }
    if (YES.test(text) || /\b(hike|record|link|app|join|start)\b/i.test(text)) {
      const inv = await server.invite(handle)
      p.invites += 1
      store.save()
      return [`here's your link. it opens the app, then hit start when you're at the trailhead:\n\n${inv.link}`]
    }
    if (p.history.length <= 1) return [INTRO]
    return ['i can send your app link (text hike), tell you about your last hike (text how did it go), or delete your data (text delete my data).']
  }

  // ---------------------------------------------------------------- outbound (from the server's outbox)
  function outbound(msg) {
    const p = store.get(msg.to)
    if (p.optedOut) throw new Error('opted out (texted stop)')
    const texts = [msg.text]
    if (msg.kind === 'summary' && msg.session) {
      texts.push(FOLLOWUPS[Math.floor(Math.random() * FOLLOWUPS.length)])
      p.pendingLabel = { session: msg.session, at: Date.now() }
    }
    for (const t of texts) store.remember(msg.to, 'assistant', t)
    return texts
  }

  return { handle, outbound }
}

/** Long replies become up to three bubbles, split on blank lines, like a person texting. */
function split(reply) {
  const parts = reply.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean)
  return parts.length <= 3 ? parts : [...parts.slice(0, 2), parts.slice(2).join('\n\n')]
}

/** Grok over xAI's OpenAI-compatible chat API. */
export function grok({ apiKey = process.env.XAI_API_KEY, model = process.env.XAI_MODEL || 'grok-4' } = {}) {
  if (!apiKey) return null
  return async function chat(messages, tools) {
    const r = await fetch('https://api.x.ai/v1/chat/completions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, tools, tool_choice: 'auto', temperature: 0.7 }),
    })
    if (!r.ok) throw new Error(`grok HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`)
    return (await r.json()).choices[0].message
  }
}
