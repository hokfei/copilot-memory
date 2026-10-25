import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { sessionStoreDb, dataDb, sessionStateDir } from "./paths.mjs";

const MIN_CHARS = 40;

function openReadOnly(path) {
  try {
    if (!existsSync(path)) return null;
    return new DatabaseSync(path, { readOnly: true });
  } catch {
    return null;
  }
}

function tableExists(db, name) {
  try {
    const row = db
      .prepare("SELECT 1 AS ok FROM sqlite_master WHERE name = ?")
      .get(name);
    return !!row;
  } catch {
    return false;
  }
}

function dataDbTitles() {
  const titles = new Map();
  const db = openReadOnly(dataDb);
  if (!db) return titles;
  try {
    if (!tableExists(db, "sessions")) return titles;
    for (const row of db
      .prepare("SELECT id, title, created_at FROM sessions")
      .all()) {
      titles.set(row.id, {
        title: row.title || "Untitled session",
        date: row.created_at || null,
      });
    }
  } finally {
    db.close();
  }
  return titles;
}

function messageText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((b) => (b && typeof b.text === "string" ? b.text : ""))
      .join("\n");
  }
  return "";
}

function readEventsDocs(titles) {
  const docs = [];
  let dirs = [];
  try {
    dirs = readdirSync(sessionStateDir, { withFileTypes: true });
  } catch {
    return docs;
  }
  for (const dir of dirs) {
    if (!dir.isDirectory() || dir.name.startsWith(".")) continue;
    const sessionId = dir.name;
    const eventsPath = join(sessionStateDir, sessionId, "events.jsonl");
    if (!existsSync(eventsPath)) continue;
    const info = titles.get(sessionId) || {};
    let fallbackDate = null;
    try {
      fallbackDate = statSync(eventsPath).mtime.toISOString();
    } catch {
      fallbackDate = null;
    }
    const title = info.title || "Untitled session";
    const date = info.date || fallbackDate;
    let raw;
    try {
      raw = readFileSync(eventsPath, "utf8");
    } catch {
      continue;
    }
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }
      let sourceType = null;
      if (event.type === "user.message") sourceType = "user-message";
      else if (event.type === "assistant.message")
        sourceType = "assistant-message";
      if (!sourceType) continue;
      const text = messageText(event.data && event.data.content).trim();
      if (text.length < MIN_CHARS) continue;
      const sourceId =
        (event.data && event.data.messageId) || event.id || "unknown";
      docs.push({ sessionId, title, date, sourceType, sourceId, text });
    }
  }
  return docs;
}

function readSessionStoreDocs() {
  const docs = [];
  const db = openReadOnly(sessionStoreDb);
  if (!db) return docs;
  try {
    const summaries = new Map();
    if (tableExists(db, "sessions")) {
      for (const row of db
        .prepare("SELECT id, summary, created_at FROM sessions")
        .all()) {
        summaries.set(row.id, {
          title: row.summary || "Untitled session",
          date: row.created_at || null,
        });
        const text = (row.summary || "").trim();
        if (text.length >= MIN_CHARS) {
          docs.push({
            sessionId: row.id,
            title: text.slice(0, 120),
            date: row.created_at || null,
            sourceType: "summary",
            sourceId: "summary",
            text,
          });
        }
      }
    }
    if (tableExists(db, "turns")) {
      for (const row of db
        .prepare(
          "SELECT session_id, turn_index, user_message, assistant_response, timestamp FROM turns"
        )
        .all()) {
        const info = summaries.get(row.session_id) || {};
        const title = info.title || "Untitled session";
        const date = row.timestamp || info.date || null;
        for (const [sourceType, text] of [
          ["user-message", row.user_message],
          ["assistant-message", row.assistant_response],
        ]) {
          const clean = (text || "").trim();
          if (clean.length < MIN_CHARS) continue;
          docs.push({
            sessionId: row.session_id,
            title,
            date,
            sourceType,
            sourceId: `turn-${row.turn_index}`,
            text: clean,
          });
        }
      }
    }
  } finally {
    db.close();
  }
  return docs;
}

export function readAllDocs() {
  const titles = dataDbTitles();
  const docs = [...readSessionStoreDocs(), ...readEventsDocs(titles)];
  const seen = new Set();
  const unique = [];
  for (const doc of docs) {
    const digest = createHash("sha1").update(doc.text, "utf8").digest("hex");
    const dedupeKey = `${doc.sessionId}\n${digest}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    unique.push(doc);
  }
  return unique;
}
