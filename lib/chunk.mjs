import { createHash } from "node:crypto";

export function sha1(text) {
  return createHash("sha1").update(text, "utf8").digest("hex");
}

export function chunkText(text, size = 1500, overlap = 200) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + size, text.length);
    if (end < text.length) {
      const snap = text.lastIndexOf(" ", end);
      if (snap > start + size - 200) end = snap;
    }
    chunks.push(text.slice(start, end));
    if (end >= text.length) break;
    start = Math.max(end - overlap, start + 1);
  }
  return chunks;
}

export function chunkKey(sessionId, sourceType, sourceId, index) {
  return `${sessionId}:${sourceType}:${sourceId}:${index}`;
}
