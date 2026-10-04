// The agent's people and messages in SpacetimeDB, as a service of the app's organisation (GT_TENANT, default
// ground-truth-app). The identity joins with GT_SERVICE_INVITE once, or an owner adds it with
// python tools/stdb_admin.py add-service <identity>. agent_state.json stays the agent's working copy.
import { messageKey, STATE_PATH } from './state.mjs'

export const APP_TENANT = process.env.GT_TENANT || 'ground-truth-app'
const APP_ROLES = ['owner', 'admin', 'service']

// c: a connected DbConnection subscribed to tables.myTenant. Resolves to the role it writes as, or null.
export async function startAppSync(c, id, store, tenant = APP_TENANT) {
  if (store.onPerson) return null
  const role = () => [...c.db.myTenant.iter()].find((t) => t.id === tenant)?.role
  if (!role() && process.env.GT_SERVICE_INVITE) {
    await c.reducers.redeemInvite({ code: process.env.GT_SERVICE_INVITE, name: 'ground truth agent' })
      .catch((e) => console.error('[stdb] GT_SERVICE_INVITE refused:', e?.message || e))
    for (let i = 0; i < 30 && !role(); i++) await new Promise((r) => setTimeout(r, 100))
  }
  const r = role()
  if (!APP_ROLES.includes(r)) {
    console.warn(`[agent] WARNING: 0x${id.toHexString()} is not a service of ${tenant} in SpacetimeDB, so people and messages ` +
      `stay in ${STATE_PATH}. Fix: GT_SERVICE_INVITE=<code>, or python tools/stdb_admin.py add-service 0x${id.toHexString()}`)
    return null
  }
  const fail = (what) => (e) => console.error(`[stdb] ${what} failed:`, e?.message || e)
  store.onPerson = (k, p) => c.reducers.upsertAppUser({
    handle: k, tenant, code: p.code || '', name: p.name || '', optedOut: !!p.optedOut, pendingLabel: p.pendingLabel || '',
    pendingDelete: !!p.pendingDelete, invites: p.invites || 0, notes: (p.notes || []).map((n) => (typeof n === 'string' ? n : JSON.stringify(n))),
    firstSeenMs: BigInt(p.firstSeen || 0),
  }).catch(fail(`person ${k}`))
  store.onMessage = (k, m, kind = 'chat') => c.reducers.upsertAppMessage({
    key: messageKey(k, m), handle: k, tenant, direction: m.role === 'user' ? 'in' : 'out', kind, text: String(m.content),
    session: '', status: m.role === 'user' ? '' : 'sent', error: '', atMs: BigInt(m.at || 0),
  }).catch(fail(`message for ${k}`))
  store.onForget = (k) => c.reducers.deleteAppUser({ tenant, handle: k }).catch(fail(`delete for ${k}`))
  store.flush() // everyone already in agent_state.json, then their recent conversations
  for (const [k, p] of Object.entries(store.people)) if (!p.forgotten) for (const m of p.history || []) store.onMessage(k, m)
  console.log(`[stdb] writing people and messages as ${r} of ${tenant}`)
  return r
}
