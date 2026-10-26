import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { memoryDbPath } from "./paths.mjs";
import { classifyKind } from "./quality.mjs";
import { findNearDuplicate } from "./similarity.mjs";

export const GLOBAL_SCOPE = "global";
export const DEFAULT_SCOPE_CAP = 500;

export function normalizeScope(scope) {
  const s = (scope || "global").trim().toLowerCase();
  if (s === "global" || s === "") return "global";
  return s;
}

// Canonical form for content addressing: case/whitespace-insensitive.
export function normalizeText(text) {
  return (text || "").trim().toLowerCase().replace(/\s+/g, " ");
}

// Content-addressed id: re-remembering the same text refreshes instead of
// duplicating (single-key dedupe across scopes — latest scope/tags win).
export function memoryIdFor(text) {
  return createHash("sha1").update(normalizeText(text), "utf8").digest("hex");
}

export function isValidScope(scope) {
  if (scope === "global") return true;
  return scope.startsWith("project:") && scope.slice("project:".length).trim().length > 0;
}

export function openDb(dbPath = memoryDbPath) {
  const db = new DatabaseSync(dbPath);
  // Sessions run in parallel: WAL lets readers proceed during a writer's
  // commit, and busy_timeout turns an instant SQLITE_BUSY into a short wait.
  db.exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
  db.exec(`
    CREATE TABLE IF NOT EXISTS memories (
      id TEXT PRIMARY KEY,
      text TEXT NOT NULL,
      scope TEXT NOT NULL DEFAULT 'global',
      tags TEXT NOT NULL DEFAULT '[]',
      source_session TEXT,
      pinned INTEGER NOT NULL DEFAULT 0,
      kind TEXT NOT NULL DEFAULT 'state',
      detail TEXT NOT NULL DEFAULT '',
      superseded_by TEXT,
      supersedes TEXT,
      confirmed INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
    CREATE TABLE IF NOT EXISTS memory_vectors (
      id TEXT PRIMARY KEY,
      vector BLOB NOT NULL
    );
  `);
  // Migrate DBs created before pinning / kind-detail existed.
  try {
    const cols = db.prepare("PRAGMA table_info(memories)").all();
    const names = new Set(cols.map((c) => c.name));
    if (!names.has("pinned")) {
      db.exec("ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0");
    }
    if (!names.has("kind")) {
      db.exec("ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'state'");
    }
    if (!names.has("detail")) {
      db.exec("ALTER TABLE memories ADD COLUMN detail TEXT NOT NULL DEFAULT ''");
    }
    if (!names.has("superseded_by")) {
      db.exec("ALTER TABLE memories ADD COLUMN superseded_by TEXT");
    }
    if (!names.has("supersedes")) {
      db.exec("ALTER TABLE memories ADD COLUMN supersedes TEXT");
    }
    if (!names.has("confirmed")) {
      db.exec("ALTER TABLE memories ADD COLUMN confirmed INTEGER NOT NULL DEFAULT 0");
    }
  } catch {
    // Fresh table already has the columns; nothing to do.
  }
  return db;
}

export function parseTags(tagsJson) {
  try {
    const v = JSON.parse(tagsJson);
    return Array.isArray(v) ? v.filter((t) => typeof t === "string") : [];
  } catch {
    return [];
  }
}

function rowToMemory(row) {
  return {
    id: row.id,
    text: row.text,
    scope: row.scope,
    tags: parseTags(row.tags),
    source_session: row.source_session,
    pinned: (row.pinned | 0) === 1,
    kind: row.kind === "event" ? "event" : "state",
    detail: typeof row.detail === "string" ? row.detail : "",
    superseded_by: typeof row.superseded_by === "string" ? row.superseded_by : null,
    supersedes: typeof row.supersedes === "string" ? row.supersedes : null,
    confirmed: row.confirmed | 0,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

// Upsert by content hash. Returns { id, created }. Refreshing keeps the
// original created_at and (unless pin is given) the existing pin state,
// and bumps the confirmed counter (the same fact seen again).
// Kind is ALWAYS classified structurally from the text (callers cannot set
// it); detail is replaced when given, kept otherwise.
export function upsertMemory(
  db,
  { text, scope = "global", tags = [], sourceSession = null, pin = null, detail = null }
) {
  const now = new Date().toISOString();
  const id = memoryIdFor(text);
  const kind = classifyKind(text);
  const existing = db.prepare("SELECT * FROM memories WHERE id = ?").get(id);
  if (!existing) {
    db.prepare(
      "INSERT INTO memories (id, text, scope, tags, source_session, pinned, kind, detail, confirmed, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)"
    ).run(
      id,
      text.trim(),
      normalizeScope(scope),
      JSON.stringify(tags),
      sourceSession,
      pin ? 1 : 0,
      kind,
      detail ?? "",
      now,
      now
    );
    return { id, created: true };
  }
  const pinned = pin === null || pin === undefined ? existing.pinned | 0 : pin ? 1 : 0;
  db.prepare(
    "UPDATE memories SET text = ?, scope = ?, tags = ?, source_session = COALESCE(?, source_session), pinned = ?, kind = ?, detail = COALESCE(?, detail), confirmed = confirmed + 1, updated_at = ? WHERE id = ?"
  ).run(text.trim(), normalizeScope(scope), JSON.stringify(tags), sourceSession, pinned, kind, detail, now, id);
  return { id, created: false };
}

// Retire near-duplicates of a newly created row: among ACTIVE rows in the
// same scope, the best IDF-Jaccard match at or above threshold is marked
// superseded_by the new id (kept for audit, hidden from recall/list/caps).
// Pin carries forward to the new row; the new row records supersedes.
// Returns { supersededId, score } or null. Token logic only, no embedding.
export function applySupersession(db, newId, text, scope) {
  const now = new Date().toISOString();
  const scoped = normalizeScope(scope);
  const candidates = db
    .prepare("SELECT id, text, pinned FROM memories WHERE scope = ? AND superseded_by IS NULL AND id != ?")
    .all(scoped, newId);
  const hit = findNearDuplicate(text, candidates);
  if (!hit) return null;
  const old = candidates.find((c) => c.id === hit.id);
  if (old && (old.pinned | 0) === 1) setPinned(db, newId, true);
  db.prepare("UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?").run(newId, now, hit.id);
  db.prepare("UPDATE memories SET supersedes = ? WHERE id = ?").run(hit.id, newId);
  return { supersededId: hit.id, score: hit.score };
}

// Kept for tests and one-off inserts; prefer upsertMemory in handlers.
export function insertMemory(db, { text, scope = "global", tags = [], sourceSession = null, detail = null }) {
  return upsertMemory(db, { text, scope, tags, sourceSession, detail }).id;
}

export function insertVector(db, id, vector) {
  const buf = Buffer.from(new Float32Array(vector).buffer);
  db.prepare("INSERT OR REPLACE INTO memory_vectors (id, vector) VALUES (?, ?)").run(id, buf);
}

export function setPinned(db, id, pinned) {
  const res = db.prepare("UPDATE memories SET pinned = ?, updated_at = ? WHERE id = ?").run(
    pinned ? 1 : 0,
    new Date().toISOString(),
    id
  );
  return res.changes > 0;
}

export function countScopeMemories(db, scope) {
  return db
    .prepare("SELECT COUNT(*) AS n FROM memories WHERE scope = ? AND superseded_by IS NULL")
    .get(normalizeScope(scope)).n;
}

// Drop oldest unpinned ACTIVE entries over the cap (pinned are never
// evicted; superseded rows are already invisible and don't count).
// Returns the number evicted.
export function enforceScopeCap(db, scope, cap = DEFAULT_SCOPE_CAP) {
  const over =
    countScopeMemories(db, scope) - Math.max(1, Math.floor(cap) || DEFAULT_SCOPE_CAP);
  if (over <= 0) return 0;
  const victims = db
    .prepare(
      "SELECT id FROM memories WHERE scope = ? AND pinned = 0 AND superseded_by IS NULL ORDER BY created_at ASC, rowid ASC LIMIT ?"
    )
    .all(normalizeScope(scope), over);
  for (const v of victims) deleteMemory(db, v.id);
  return victims.length;
}

export function getMemory(db, id) {
  const row = db.prepare("SELECT * FROM memories WHERE id = ?").get(id);
  return row ? rowToMemory(row) : null;
}

export function deleteMemory(db, id) {
  db.prepare("DELETE FROM memory_vectors WHERE id = ?").run(id);
  const res = db.prepare("DELETE FROM memories WHERE id = ?").run(id);
  return res.changes > 0;
}

// Update a row's metadata in place (text/id/vector untouched). Used by
// consolidate for absorb-only merges: the survivor gains the union tags,
// an absorbed session when it has none, and pin if any absorbed row is pinned.
export function updateMemoryMeta(db, id, { tags, sourceSession = null, pin = false } = {}) {
  const now = new Date().toISOString();
  db.prepare(
    "UPDATE memories SET tags = ?, source_session = COALESCE(?, source_session), pinned = ?, updated_at = ? WHERE id = ?"
  ).run(JSON.stringify(tags ?? []), sourceSession ?? null, pin ? 1 : 0, now, id);
}

export function listMemories(db, { scope = null, tag = null, limit = 100 } = {}) {
  // Superseded rows are audit history: hidden from browse, recall, and caps.
  const conds = ["superseded_by IS NULL"];
  const params = [];
  if (scope) {
    conds.push("scope = ?");
    params.push(normalizeScope(scope));
  }
  let sql = "SELECT * FROM memories WHERE " + conds.join(" AND ");
  sql += " ORDER BY created_at DESC LIMIT ?";
  params.push(Math.max(1, Math.min(500, limit | 0 || 100)));
  const rows = db.prepare(sql).all(...params).map(rowToMemory);
  if (tag) return rows.filter((m) => m.tags.includes(tag));
  return rows;
}

export function getVector(db, id) {
  const row = db.prepare("SELECT vector FROM memory_vectors WHERE id = ?").get(id);
  if (!row) return null;
  const buf = row.vector;
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function getAllVectors(db, scope = null) {
  let sql = "SELECT m.id, m.scope, m.pinned, m.kind, m.detail, v.vector FROM memories m JOIN memory_vectors v ON m.id = v.id WHERE m.superseded_by IS NULL";
  const params = [];
  if (scope) {
    sql += " AND m.scope = ?";
    params.push(normalizeScope(scope));
  }
  const out = [];
  for (const row of db.prepare(sql).all(...params)) {
    const buf = row.vector;
    out.push({
      id: row.id,
      scope: row.scope,
      pinned: (row.pinned | 0) === 1,
      kind: row.kind === "event" ? "event" : "state",
      detail: typeof row.detail === "string" ? row.detail : "",
      vector: new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4),
    });
  }
  return out;
}

export function countMemories(db) {
  return db.prepare("SELECT COUNT(*) AS n FROM memories WHERE superseded_by IS NULL").get().n;
}
