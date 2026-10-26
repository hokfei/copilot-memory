// Near-duplicate detection for memory supersession (exact-hash dedupe lets
// paraphrases like "uses Postgres 14" and "we run Postgres 14 in prod" live
// side by side until they contradict each other). Dependency-free.
//
// Similarity is IDF-weighted Jaccard over tokens:
//
//   tokens(s)    = lowercase, non-alphanumerics to spaces, words longer than 3
//   idf(w)       = ln((N+1)/(df(w)+1)) + 1, measured over the ACTIVE rows
//                  themselves, so boilerplate shared by many rows
//                  ("windows", "scheduled", "task") counts little while
//                  distinctive words decide
//   similarity   = sum idf of shared tokens / sum idf of UNION (not min, so a
//                  short fact cannot be swallowed by a long one containing it)
//
// Threshold 0.45 was calibrated on real data: unrelated pairs peaked at
// 0.278, a direct contradiction (Thursdays vs Tuesdays) scored 0.600, a
// close paraphrase 0.491. A wrong merge destroys a fact silently while a
// missed duplicate only costs redundancy, so the threshold stays
// conservative. Pure synonym restatements are missed; that is accepted —
// the embedding-based consolidate pass can catch those offline.

export const SIMILARITY_THRESHOLD = 0.45;

export function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(" ")
    .filter((w) => w.length > 3);
}

function idf(weight, df, n) {
  return Math.log((n + 1) / ((df.get(weight) || 0) + 1)) + 1;
}

// candidates: [{ id, text }]. Returns { id, score } of the best match at or
// above the threshold, or null. Ties break toward the first candidate.
export function findNearDuplicate(text, candidates, threshold = SIMILARITY_THRESHOLD) {
  const query = new Set(tokenize(text));
  const rows = (candidates || []).map((c) => ({ id: c.id, tokens: new Set(tokenize(c.text)) }));
  if (query.size === 0 || rows.length === 0) return null;
  const df = new Map();
  for (const r of rows) {
    for (const w of r.tokens) df.set(w, (df.get(w) || 0) + 1);
  }
  const n = rows.length;
  let best = null;
  for (const r of rows) {
    const union = new Set([...query, ...r.tokens]);
    let shared = 0;
    let total = 0;
    for (const w of union) {
      const v = idf(w, df, n);
      total += v;
      if (query.has(w) && r.tokens.has(w)) shared += v;
    }
    const score = total === 0 ? 0 : shared / total;
    if (score >= threshold && (!best || score > best.score)) {
      best = { id: r.id, score };
    }
  }
  return best;
}
