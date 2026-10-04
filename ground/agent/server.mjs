// Calls to ground/server.py. Every non-2xx throws with the server's own error message.
export function groundServer(base = process.env.GT_SERVER || 'http://127.0.0.1:8770') {
  async function call(path, body) {
    let r
    try {
      r = await fetch(base + path, body === undefined ? {} : {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
    } catch (e) {
      throw new Error(`ground server unreachable at ${base} (is ground/run.sh running?): ${e.cause?.code || e.message}`)
    }
    const j = await r.json().catch(() => ({ error: `non-json reply, HTTP ${r.status}` }))
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${j.error || ''}`)
    return j
  }
  return {
    base,
    invite: (phone, code) => call('/api/invite', { phone, code }),
    trigger: (phone, code, text) => call('/api/trigger', { phone, code, text }),
    hikes: async (phone) => (await call(`/api/user/${encodeURIComponent(phone)}`)).hikes,
    label: (session, text) => call('/api/label', { session, text, source: 'imessage' }),
    deleteUser: (phone) => call('/api/delete', { phone }),
    outbox: () => call('/api/outbox'),
    markSent: (id) => call(`/api/outbox/${id}/sent`, {}),
    markFailed: (id, error) => call(`/api/outbox/${id}/failed`, { error }),
  }
}
