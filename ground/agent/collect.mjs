// Ground Truth collector: a read-only listener on the team's SpacetimeDB that keeps everything the game and the
// board produce, including what the module itself throws away (cheers are event rows nobody stores, the skier table
// holds only the latest 10 Hz state, the feed keeps 60 lines). It calls no reducers and nothing reads its output yet;
// it only writes files, so it can be started and stopped without touching the game, the board or the agent.
//
//   npm run collect                      STDB_URI / STDB_DB from the environment or ground/agent/.env
//
// Output, one JSON object per line, appended: out/ground/collect/<table>.jsonl
//   {"at": ISO time we saw it, "op": "snapshot" | "insert" | "update" | "delete", "row": {...}}
// A snapshot of every table is written each time the collector (re)connects, so files are self-contained.
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const envPath = join(here, '.env')
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/)
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '')
  }
}

const STDB_URI = process.env.STDB_URI || 'ws://127.0.0.1:3000'
const STDB_DB = process.env.STDB_DB || 'ground-truth'
const OUT = resolve(process.env.COLLECT_DIR || join(here, '..', '..', 'out', 'ground', 'collect'))
mkdirSync(OUT, { recursive: true })

const { DbConnection, tables } = await import('./module_bindings/index.ts')

// Every public table. The game writes player, skier, run, trace_chunk, run_photo; the board writes cheer;
// the module writes feed, dataset_stats, challenge, outbox; the phone server writes the hike tables; phone, game
// joint and video streams for skiing and hiking go into capture / capture_chunk.
const TABLES = ['player', 'skier', 'run', 'traceChunk', 'runPhoto', 'cheer', 'feed', 'datasetStats',
  'challenge', 'outbox', 'hike', 'hikeChunk', 'hikeStats', 'capture', 'captureChunk', 'captureStats']
  .filter(t => t in tables) // capture tables appear once the bindings are regenerated

// SpacetimeDB values to plain JSON: u64 as strings, identities as hex, timestamps as ISO.
function plain(v) {
  if (typeof v === 'bigint') return v.toString()
  if (v === null || typeof v !== 'object') return v
  if (typeof v.toHexString === 'function') return v.toHexString()
  if (typeof v.microsSinceUnixEpoch === 'bigint') return new Date(Number(v.microsSinceUnixEpoch / 1000n)).toISOString()
  if (Array.isArray(v)) return v.map(plain)
  const o = {}
  for (const [k, x] of Object.entries(v)) o[k] = plain(x)
  return o
}

const counts = Object.fromEntries(TABLES.map(t => [t, 0]))
function write(table, op, row) {
  appendFileSync(join(OUT, `${table}.jsonl`), JSON.stringify({ at: new Date().toISOString(), op, row: plain(row) }) + '\n')
  counts[table]++
}

let synced = false
DbConnection.builder().withUri(STDB_URI).withDatabaseName(STDB_DB).withConfirmedReads(false)
  .onConnect((c, id) => {
    console.log(`[collect] connected to ${STDB_URI}/${STDB_DB} as ${id.toHexString().slice(0, 12)}, writing ${OUT}`)
    for (const t of TABLES) {
      const h = c.db[t]
      h.onInsert((_ctx, row) => { if (synced) write(t, 'insert', row) })
      if (h.onUpdate) h.onUpdate((_ctx, _old, row) => write(t, 'update', row))
      if (h.onDelete) h.onDelete((_ctx, row) => write(t, 'delete', row))
    }
    c.subscriptionBuilder()
      .onApplied(() => {
        for (const t of TABLES) for (const row of c.db[t].iter?.() ?? []) write(t, 'snapshot', row)
        synced = true
        console.log('[collect] synced; snapshot written')
      })
      .onError((_ctx, e) => console.error('[collect] subscription error:', e))
      .subscribe(TABLES.map(t => tables[t]))
  })
  .onConnectError((_ctx, e) => { console.error(`[collect] could not connect to ${STDB_URI}/${STDB_DB}:`, e?.message ?? e); process.exit(1) })
  .onDisconnect(() => { console.error('[collect] disconnected'); process.exit(1) })
  .build()

setInterval(() => {
  const busy = Object.entries(counts).filter(([, n]) => n).map(([t, n]) => `${t} ${n}`).join(', ')
  if (busy) console.log(`[collect] rows so far: ${busy}`)
}, 60_000)
