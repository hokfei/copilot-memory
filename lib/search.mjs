export const DEFAULT_LIMIT = 5;
export const MAX_LIMIT = 15;
const PER_SCOPE_CAP = 2;

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// Stored vectors must match the live query vector's dims; after an
// embedding-model change they don't, and dot products would be garbage.
// Returns an explanation string, or null when every vector matches.
export function dimsMismatch(memories, queryVec) {
  const bad = memories.find((m) => !m.vector || m.vector.length !== queryVec.length);
  if (!bad) return null;
  const got = bad.vector ? `${bad.vector.length} dims` : "no vector";
  return `Stored vectors (${got}) disagree with the model (${queryVec.length} dims).`;
}

// memories: [{ id, text, scope, tags, pinned, ... , vector: Float32Array }]
// Pinned entries always rank before unpinned ones (then by score).
export function searchMemories(memories, queryVec, limit = DEFAULT_LIMIT) {
  limit = Math.max(1, Math.min(MAX_LIMIT, Math.floor(limit) || DEFAULT_LIMIT));
  const scored = memories.map((m) => ({ m, score: dotProduct(m.vector, queryVec) }));
  scored.sort(
    (a, b) => Number(b.m.pinned || false) - Number(a.m.pinned || false) || b.score - a.score
  );
  const candidates = scored.slice(0, Math.max(limit, limit * 4));

  const picked = [];
  const perScope = new Map();
  for (const c of candidates) {
    if (picked.length >= limit) break;
    const used = perScope.get(c.m.scope) || 0;
    if (used >= PER_SCOPE_CAP) continue;
    perScope.set(c.m.scope, used + 1);
    picked.push(c);
  }
  if (picked.length < limit) {
    const taken = new Set(picked.map((p) => p.m.id));
    for (const c of candidates) {
      if (picked.length >= limit) break;
      if (taken.has(c.m.id)) continue;
      picked.push(c);
    }
  }
  return picked.map(({ m, score }) => ({
    id: m.id,
    score,
    text: m.text,
    scope: m.scope,
    tags: m.tags,
    pinned: !!m.pinned,
    source_session: m.source_session,
    created_at: m.created_at,
    updated_at: m.updated_at,
  }));
}
