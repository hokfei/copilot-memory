// Offline auto-capture: mine durable facts from session checkpoints.
//
// The miner reads CHECKPOINT prose from ~/.copilot/session-store.db (already
// summarised by the cloud model — extraction from clean text, not raw
// transcripts), asks the local GGUF chat model for durable reusable facts as
// JSON, runs every candidate through the full gate in lib/miner.mjs, then
// either stages survivors for approval (default) or stores them through the
// same supersession path as remember (behind --auto-import).
//
// On-device: no network, no AI credits. Uses SESSION_MEMORY_CHAT_MODEL.
//
// Usage:
//   node scripts/mine.mjs [--limit N] [--auto-import] [--min-confidence X]
//   node scripts/mine.mjs --import [mined-staging.json]
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import {
  extRoot,
  sessionStoreDb,
  minedStatePath,
  minedStagingPath,
} from "../lib/paths.mjs";
import {
  readCheckpoints,
  extractionPrompt,
  parseExtraction,
  minerGate,
  MIN_CONFIDENCE,
} from "../lib/miner.mjs";
import { repoScope } from "../lib/preload.mjs";
import {
  openDb,
  normalizeScope,
  upsertMemory,
  insertVector,
  applySupersession,
  enforceScopeCap,
  DEFAULT_SCOPE_CAP,
} from "../lib/store.mjs";
import { embedDocuments, disposeEmbedder } from "../lib/embed.mjs";
import { runChat, disposeChat } from "../lib/synthesize.mjs";

const SYSTEM_PROMPT =
  "You extract durable, reusable facts from session checkpoints. Reply with exactly one JSON array and no other text.";

function scopeForSession(sessionId) {
  try {
    if (!existsSync(sessionStoreDb)) return "global";
    const db = new DatabaseSync(sessionStoreDb, { readOnly: true });
    try {
      const row = db
        .prepare("SELECT repository, cwd FROM sessions WHERE id = ?")
        .get(sessionId);
      const repo = (row && (row.repository || "").trim()) || "";
      if (repo) return normalizeScope(repoScope(repo));
      const cwd = (row && (row.cwd || "").trim()) || "";
      if (cwd) {
        const base = cwd.split(/[/\\]/).filter(Boolean).pop() || "";
        if (base) return normalizeScope(repoScope(base));
      }
    } finally {
      db.close();
    }
  } catch {
    // Sessions table absent or unreadable — global scope.
  }
  return "global";
}

function loadState() {
  try {
    const raw = JSON.parse(readFileSync(minedStatePath, "utf8"));
    if (Array.isArray(raw.processed)) return new Set(raw.processed);
  } catch {
    // No state yet.
  }
  return new Set();
}

function saveState(processed) {
  writeFileSync(minedStatePath, JSON.stringify({ processed: [...processed] }, null, 1), "utf8");
}

async function runMine({ limit = 25, autoImport = false, minConfidence = 0.85 }) {
  const checkpoints = readCheckpoints(sessionStoreDb);
  const processed = loadState();
  const fresh = checkpoints.filter((c) => !processed.has(`${c.sessionId}:${c.number}`));
  console.log(`checkpoints: ${checkpoints.length} total, ${fresh.length} new`);
  if (fresh.length === 0) return;
  const todo = fresh.slice(0, Math.max(1, limit));
  const staged = [];
  const skipped = {};
  const bump = (reason) => {
    skipped[reason] = (skipped[reason] || 0) + 1;
  };
  for (const cp of todo) {
    if (!cp.text.trim()) {
      processed.add(`${cp.sessionId}:${cp.number}`);
      continue;
    }
    const prompt = extractionPrompt(cp.text);
    let reply;
    try {
      ({ reply } = await runChat(SYSTEM_PROMPT, prompt));
    } catch (err) {
      console.error(`checkpoint ${cp.sessionId}:${cp.number}: model failed (${err.message}), stopping`);
      break;
    }
    const { candidates, errors } = parseExtraction(reply);
    for (const e of errors) bump(`parse: ${e}`);
    for (const cand of candidates) {
      const reason = minerGate(cand, prompt);
      if (reason) {
        bump(reason.split(" (")[0]);
        continue;
      }
      staged.push({
        text: cand.text,
        detail: cand.detail || "",
        tags: cand.tags || [],
        confidence: cand.confidence,
        scope: scopeForSession(cp.sessionId),
        source_session: cp.sessionId,
      });
    }
    processed.add(`${cp.sessionId}:${cp.number}`);
  }
  await disposeChat().catch(() => {});
  saveState(processed);
  if (autoImport) {
    const { imported, gated } = await importStaged(staged, minConfidence);
    console.log(`mined: ${staged.length} survivors, ${imported} auto-imported (>= ${minConfidence}), ${gated} held`);
  } else {
    const prev = existsSync(minedStagingPath)
      ? JSON.parse(readFileSync(minedStagingPath, "utf8"))
      : [];
    writeFileSync(minedStagingPath, JSON.stringify([...prev, ...staged], null, 1), "utf8");
    console.log(`mined: ${staged.length} survivors staged -> ${minedStagingPath} (review, then --import)`);
  }
  if (Object.keys(skipped).length > 0) {
    console.log("skipped: " + Object.entries(skipped).map(([k, v]) => `${v}x ${k}`).join("; "));
  }
}

export async function importStaged(items, minConfidence = 0) {
  const eligible = items.filter((it) => (it.confidence ?? 1) >= minConfidence);
  const gated = items.length - eligible.length;
  if (eligible.length === 0) return { imported: 0, gated };
  const vecs = await embedDocuments(eligible.map((it) => it.text));
  const db = openDb();
  try {
    eligible.forEach((it, i) => {
      const { id } = upsertMemory(db, {
        text: it.text,
        scope: it.scope || "global",
        tags: Array.isArray(it.tags) ? it.tags : [],
        sourceSession: it.source_session || null,
        detail: it.detail || null,
      });
      insertVector(db, id, vecs[i]);
      applySupersession(db, id, it.text, it.scope || "global");
      enforceScopeCap(db, it.scope || "global", DEFAULT_SCOPE_CAP);
    });
  } finally {
    db.close();
  }
  await disposeEmbedder().catch(() => {});
  return { imported: eligible.length, gated };
}

async function runImport(file) {
  if (!existsSync(file)) {
    console.error(`staging file not found: ${file}`);
    process.exitCode = 1;
    return;
  }
  const items = JSON.parse(readFileSync(file, "utf8"));
  if (!Array.isArray(items) || items.length === 0) {
    console.error("nothing to import (empty or invalid staging file)");
    process.exitCode = 1;
    return;
  }
  const { imported } = await importStaged(items, 0);
  console.log(`imported ${imported} memories from ${file}`);
}

async function main() {
  process.chdir(extRoot);
  const args = process.argv.slice(2);
  if (args.includes("--import")) {
    await runImport(args[args.indexOf("--import") + 1] || minedStagingPath);
    return;
  }
  if (args.includes("--help") || args.includes("-h")) {
    console.log("Usage: node scripts/mine.mjs [--limit N] [--auto-import] [--min-confidence X]");
    console.log("       node scripts/mine.mjs --import [mined-staging.json]");
    return;
  }
  const limitIdx = args.indexOf("--limit");
  const limit = limitIdx >= 0 ? parseInt(args[limitIdx + 1], 10) || 25 : 25;
  const confIdx = args.indexOf("--min-confidence");
  const minConfidence = confIdx >= 0 ? parseFloat(args[confIdx + 1]) || MIN_CONFIDENCE : 0.85;
  await runMine({
    limit,
    autoImport: args.includes("--auto-import"),
    minConfidence,
  });
}

main().catch((err) => {
  console.error(`mine failed: ${err.message}`);
  process.exitCode = 1;
});
