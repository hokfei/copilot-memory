import { statSync } from "node:fs";
import { embedDocuments } from "./embed.mjs";
import { parseTags } from "./store.mjs";

// Embed any items lacking a vector and persist the results, so recall and
// forget rank every candidate on real scores instead of zero vectors.
// Mutates items in place. Returns the number filled.
export async function fillMissingVectors(db, items) {
  const missing = items.filter((m) => !m.vector);
  if (missing.length === 0) return 0;
  const fresh = await embedDocuments(missing.map((m) => m.text));
  fresh.forEach((v, i) => {
    missing[i].vector = Float32Array.from(v);
    db.prepare("INSERT OR REPLACE INTO memory_vectors (id, vector) VALUES (?, ?)")
      .run(missing[i].id, Buffer.from(new Float32Array(v).buffer));
  });
  return missing.length;
}

// Under WAL, commits land in the -wal sidecar (the main file only moves on
// checkpoint), so the freshness stamp must cover both files.
function dbStamp(dbPath) {
  let stamp = null;
  for (const p of [dbPath, dbPath + "-wal"]) {
    try {
      const m = statSync(p).mtimeMs;
      stamp = stamp === null ? m : Math.max(stamp, m);
    } catch {
      // Sidecar may not exist yet; ignore.
    }
  }
  return stamp;
}

function decodeRow(r) {
  return {
    id: r.id,
    text: r.text,
    scope: r.scope,
    tags: parseTags(r.tags),
    pinned: (r.pinned | 0) === 1,
    kind: r.kind === "event" ? "event" : "state",
    detail: typeof r.detail === "string" ? r.detail : "",
    confirmed: r.confirmed | 0,
    source_session: r.source_session,
    created_at: r.created_at,
    updated_at: r.updated_at,
    vector: r.vec
      ? new Float32Array(r.vec.buffer, r.vec.byteOffset, r.vec.byteLength / 4)
      : null,
  };
}

// Single-JOIN loader shared by recall_memory and forget_memory: one query
// instead of N+1 per-row vector selects, vectors decoded once, results
// cached in memory keyed by the db files' freshness stamp (so parallel
// writers are picked up on the next call after their commit lands).
const cache = new Map(); // dbPath -> { stamp, items }

export async function loadScoredMemories(db, dbPath, scope = null) {
  const stamp = dbStamp(dbPath);
  let entry = cache.get(dbPath);
  if (!entry || entry.stamp !== stamp) {
    const rows = db
      .prepare(
        "SELECT m.*, v.vector AS vec FROM memories m LEFT JOIN memory_vectors v ON m.id = v.id WHERE m.superseded_by IS NULL ORDER BY m.created_at DESC"
      )
      .all();
    const items = rows.map(decodeRow);
    await fillMissingVectors(db, items);
    entry = { stamp: dbStamp(dbPath), items };
    cache.set(dbPath, entry);
  }
  if (!scope) return entry.items;
  return entry.items.filter((m) => m.scope === scope);
}

export function dropVectorCache(dbPath = null) {
  if (dbPath) cache.delete(dbPath);
  else cache.clear();
}
