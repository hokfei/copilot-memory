import { embedDocuments } from "./embed.mjs";

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
