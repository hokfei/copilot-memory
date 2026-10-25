// One-time extraction of candidate memories from local session history.
// Writes staging.json for USER APPROVAL — never inserts silently.
//
// Usage:
//   node scripts/backfill.mjs [global|project:<name>] [--limit N]
//   node scripts/backfill.mjs --import [staging.json]
//   node scripts/backfill.mjs --reembed
//
// After reviewing/editing staging.json, import approved items with --import.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { readAllDocs } from "../lib/read.mjs";
import { chunkText, sha1 } from "../lib/chunk.mjs";
import { stagingPath } from "../lib/paths.mjs";
import { openDb, normalizeScope, isValidScope, insertMemory, insertVector } from "../lib/store.mjs";
import { embedDocuments, disposeEmbedder } from "../lib/embed.mjs";

// Heuristic signals that a chunk states something worth remembering.
const SIGNALS = [
  /\b(prefer|preference|always|never|don't|do not|remember|note that|keep in mind)\b/i,
  /\b(decided|decision|chose|convention|standard|policy|rule of thumb)\b/i,
  /\b(uses?|using|configured?|set up|installed?)\b.{0,40}\b(for|as|with)\b/i,
  /\b(project|repo|codebase|workspace)\b.{0,30}\b(structure|layout|named?|called?)\b/i,
];

const SKIP = [
  /^(ok|thanks|thank you|sure|got it|lgtm|\+1)\.?$/i,
  /```/,
  /^\s*import\s+[{("']/m,
  // Never stage credential-adjacent text: memories are long-lived and broadly
  // recalled, so secret-bearing chunks are excluded, not collected.
  /\b(api[_ -]?key|auth[_ -]?token|password|passwd|secret|client[_ -]?secret|connection[_ -]?string|bearer\s+[a-z0-9\-._~+/]+=*)\b/i,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
];

export function scoreChunk(text) {
  if (text.length < 100 || text.length > 1200) return -1;
  if (SKIP.some((re) => re.test(text))) return -1;
  let score = 0;
  for (const re of SIGNALS) if (re.test(text)) score++;
  if (/^(i'll|i will|let me|i'm going to|looking into|checking)\b/i.test(text)) score--;
  return score;
}

function suggestScope(text) {
  if (/\b(project|repo|codebase|service|package)\b/i.test(text)) return "project:unknown";
  return "global";
}

function suggestTags(text) {
  const tags = [];
  if (/\bprefer|taste|like\b/i.test(text)) tags.push("preference");
  if (/\bdecided|decision|chose\b/i.test(text)) tags.push("decision");
  if (/\bdeploy|pipeline|ci\b|github actions/i.test(text)) tags.push("workflow");
  if (/\bproject|repo|codebase\b/i.test(text)) tags.push("project-fact");
  return tags;
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--import")) {
    const file = args[args.indexOf("--import") + 1] || stagingPath;
    await runImport(file);
    return;
  }
  if (args.includes("--reembed")) {
    await runReembed();
    return;
  }
  if (args.includes("--help") || args.includes("-h")) {
    console.log("Usage: node scripts/backfill.mjs [global|project:<name>] [--limit N]");
    console.log("       node scripts/backfill.mjs --import [staging.json]");
    console.log("       node scripts/backfill.mjs --reembed");
    return;
  }

  // Positional scope: skip flags and their values (--limit N) when scanning.
  const takesValue = new Set(["--limit"]);
  const positionals = [];
  for (let i = 0; i < args.length; i++) {
    if (takesValue.has(args[i])) { i++; continue; }
    if (args[i].startsWith("--")) continue;
    positionals.push(args[i]);
  }
  const scopeArg = positionals[0];
  const limitIdx = args.indexOf("--limit");
  const limit = limitIdx >= 0 ? parseInt(args[limitIdx + 1], 10) || 50 : 50;
  const scopeFilter = scopeArg ? normalizeScope(scopeArg) : null;
  if (scopeFilter && !isValidScope(scopeFilter)) {
    console.error(`invalid scope '${scopeArg}'. Use 'global' or 'project:<name>'.`);
    process.exitCode = 1;
    return;
  }

  const docs = readAllDocs();
  console.log(`sources: ${docs.length} documents`);
  const seen = new Set();
  const candidates = [];
  for (const doc of docs) {
    for (const part of chunkText(doc.text)) {
      const text = part.trim();
      const digest = sha1(text);
      if (seen.has(digest)) continue;
      seen.add(digest);
      const score = scoreChunk(text);
      if (score < 2) continue;
      candidates.push({
        text,
        score,
        scope: scopeFilter || suggestScope(text),
        tags: suggestTags(text),
        source_session: doc.sessionId,
      });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  const picked = candidates.slice(0, limit).map(({ score, ...rest }) => rest);
  writeFileSync(stagingPath, JSON.stringify(picked, null, 1), "utf8");
  await disposeEmbedder().catch(() => {});
  console.log(`candidates: ${picked.length} (from ${candidates.length} scored) -> ${stagingPath}`);
  console.log("Review/edit staging.json, then: node scripts/backfill.mjs --import");
}

// Normalize + validate staging items BEFORE embedding (keeps vecs[] aligned).
// Returns { staged, skipped }. Pure: unit-tested, no DB or model needed.
export function prepareImportItems(items) {
  const staged = [];
  let skipped = 0;
  for (const it of items) {
    if (!it || typeof it.text !== "string" || !it.text.trim()) { skipped++; continue; }
    const scope = normalizeScope(it.scope || "global");
    if (!isValidScope(scope)) {
      console.error(`skipping item with invalid scope '${it.scope}'`);
      skipped++;
      continue;
    }
    staged.push({
      text: it.text.trim(),
      scope,
      tags: Array.isArray(it.tags)
        ? it.tags.filter((t) => typeof t === "string" && t.trim()).map((t) => t.trim())
        : [],
      sourceSession: it.source_session || null,
    });
  }
  return { staged, skipped };
}

async function runImport(file) {
  if (!existsSync(file)) {
    console.error(`staging file not found: ${file}`);
    process.exitCode = 1;
    return;
  }
  const raw = readFileSync(file);
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    console.error(
      `staging file looks like UTF-16 (e.g. created by PowerShell 'echo > file' redirection). ` +
        `Re-save ${file} as UTF-8 and retry. Nothing imported.`
    );
    process.exitCode = 1;
    return;
  }
  const items = JSON.parse(raw.toString("utf8"));
  if (!Array.isArray(items) || items.length === 0) {
    console.error("nothing to import (empty or invalid staging file)");
    process.exitCode = 1;
    return;
  }
  const { staged, skipped } = prepareImportItems(items);
  if (staged.length === 0) {
    console.error("nothing to import (no valid items)");
    process.exitCode = 1;
    return;
  }
  console.log(`importing ${staged.length} approved memories from ${file}`);
  const vecs = await embedDocuments(staged.map((it) => it.text), (done, total) => {
    if (done % 25 === 0 || done === total) console.log(`embedded ${done}/${total}`);
  });
  const db = openDb();
  try {
    staged.forEach((it, i) => {
      const id = insertMemory(db, {
        text: it.text,
        scope: it.scope,
        tags: it.tags,
        sourceSession: it.sourceSession,
      });
      insertVector(db, id, vecs[i]);
    });
  } finally {
    db.close();
  }
  await disposeEmbedder();
  console.log(`imported ${staged.length} memories${skipped ? ` (${skipped} skipped)` : ""}`);
}

// Re-embed every stored memory in place. Recovery for an embedding-model
// change, when stored vector dims no longer match the live model.
async function runReembed() {
  const db = openDb();
  let rows;
  try {
    rows = db.prepare("SELECT id, text FROM memories").all();
  } finally {
    db.close();
  }
  if (rows.length === 0) {
    console.log("no memories to re-embed");
    return;
  }
  const vecs = await embedDocuments(rows.map((r) => r.text), (done, total) => {
    if (done % 25 === 0 || done === total) console.log(`embedded ${done}/${total}`);
  });
  const db2 = openDb();
  try {
    rows.forEach((r, i) => insertVector(db2, r.id, vecs[i]));
  } finally {
    db2.close();
  }
  await disposeEmbedder();
  console.log(`re-embedded ${rows.length} memories`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((err) => {
    console.error(`backfill failed: ${err.message}`);
    process.exitCode = 1;
  });
}
