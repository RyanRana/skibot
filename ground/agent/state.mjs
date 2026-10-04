// Per-person memory for the agent, kept across restarts in out/ground/agent_state.json, keyed by the person's
// iMessage handle (phone or email): what they've agreed to, notes about them, and the recent conversation.
import { mkdirSync, readFileSync, renameSync, writeFileSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
export const STATE_PATH = process.env.GT_AGENT_STATE || join(ROOT, 'out', 'ground', 'agent_state.json')
const HISTORY_TURNS = 24

export function key(handle) {
  if (!handle) return ''
  if (handle.includes('@')) return handle.trim().toLowerCase()
  const d = handle.replace(/\D/g, '')
  return d.length === 10 ? '+1' + d : '+' + d
}

export class Store {
  constructor(path = STATE_PATH) {
    this.path = path
    this.people = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {}
  }

  get(handle) {
    const k = key(handle)
    this.people[k] ??= {
      handle, firstSeen: Date.now(), optedOut: false, notes: [], history: [],
      pendingLabel: null, pendingDelete: false, invites: 0,
    }
    return this.people[k]
  }

  remember(handle, role, content) {
    const p = this.get(handle)
    p.history.push({ role, content, at: Date.now() })
    p.history = p.history.slice(-HISTORY_TURNS)
    this.save()
  }

  save() {
    mkdirSync(dirname(this.path), { recursive: true })
    const tmp = this.path + '.tmp'
    writeFileSync(tmp, JSON.stringify(this.people, null, 1))
    renameSync(tmp, this.path)
  }
}

// One join code per person, shared by the ski game (typed on the laptop) and their hikes. SpacetimeDB only ever
// sees the code; this file is the only place that knows which phone it belongs to.
const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'

Store.prototype.code = function code(handle) {
  const p = this.get(handle)
  if (!p.code) {
    const taken = new Set(Object.values(this.people).map((x) => x.code).filter(Boolean))
    let c
    do { c = Array.from({ length: 4 }, () => CODE_ALPHABET[Math.floor(Math.random() * CODE_ALPHABET.length)]).join('') } while (taken.has(c))
    p.code = c
    this.save()
  }
  return p.code
}

Store.prototype.byCode = function byCode(code) {
  const c = String(code || '').trim().toUpperCase()
  if (!c) return null
  const hit = Object.values(this.people).find((p) => p.code === c)
  return hit ? hit.handle : null
}
