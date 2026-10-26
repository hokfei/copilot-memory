import { DatabaseSync } from "node:sqlite";
import { existsSync } from "node:fs";
import { looksSecret } from "./quality.mjs";

// Offline auto-capture: extraction from checkpoint prose (already summarised
// by the cloud model), never comprehension of raw transcripts. Pure helpers
// live here so they are unit-testable without the chat model; orchestration
// (model calls, embedding, sqlite writes) lives in scripts/mine.mjs.

export const MINED_MIN_LEN = 25;
export const MINED_MAX_LEN = 200;
export const MIN_CONFIDENCE = 0.7;

// Good examples shown to the extractor. ALSO the prompt-echo reference:
// the small local model copies these verbatim into the store, so any mined
// fact matching them is rejected.
export const EXTRACTION_EXAMPLES = [
  "Production deploys run on Thursdays with a staging gate",
  "The webapp prefers TypeScript strict mode",
  "Staging database resets nightly at 03:00 UTC",
];

// Durable = still true in six months. Everything else is rejected downstream.
export function extractionPrompt(checkpointText) {
  return (
    `Extract durable, reusable facts from this session checkpoint. A fact is durable ` +
    `only if it will still be true in six months (conventions, environments, decisions, ` +
    `ownership, recurring pitfalls). Do NOT extract progress, task status, review ` +
    `status, branch names, credentials, or anything readable from the repo.\n\n` +
    `Reply with exactly one JSON array and no other text. Each item: ` +
    `{"text": "headline, max 200 chars", "detail": "reasons or evidence (optional)", ` +
    `"tags": ["preference"|"decision"|"workflow"|"project-fact"], "confidence": 0.0-1.0}.\n\n` +
    `Good examples:\n- ${EXTRACTION_EXAMPLES.join("\n- ")}\n\n` +
    `Checkpoint:\n${checkpointText}`
  );
}

// Summarised prose columns, newest last. Returns [] when the store is absent.
export function readCheckpoints(sessionStorePath) {
  if (!sessionStorePath || !existsSync(sessionStorePath)) return [];
  let db = null;
  try {
    db = new DatabaseSync(sessionStorePath, { readOnly: true });
  } catch {
    return [];
  }
  try {
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'")
      .all();
    if (tables.length === 0) return [];
    const rows = db
      .prepare(
        `SELECT session_id, checkpoint_number, title, overview, history, work_done, technical_details
         FROM checkpoints ORDER BY session_id, checkpoint_number`
      )
      .all();
    return rows.map((r) => ({
      sessionId: r.session_id,
      number: r.checkpoint_number,
      title: r.title || "",
      text: [r.overview, r.history, r.work_done, r.technical_details]
        .filter((s) => typeof s === "string" && s.trim())
        .join("\n\n"),
    }));
  } catch {
    return [];
  } finally {
    try {
      db.close();
    } catch {
      // Ignore close errors on a read-only handle.
    }
  }
}

// Strict parse: exactly one JSON array of candidate objects. Anything else
// is an error, never a partial import.
export function parseExtraction(reply) {
  const errors = [];
  const fenced = (reply || "").match(/```(?:json)?\s*([\s\S]*?)```/);
  const raw = (fenced ? fenced[1] : reply || "").trim();
  const start = raw.indexOf("[");
  const end = raw.lastIndexOf("]");
  if (start < 0 || end <= start) {
    return { candidates: [], errors: ["no JSON array in model reply"] };
  }
  let arr;
  try {
    arr = JSON.parse(raw.slice(start, end + 1));
  } catch {
    return { candidates: [], errors: ["model reply is not valid JSON"] };
  }
  if (!Array.isArray(arr)) return { candidates: [], errors: ["model reply is not an array"] };
  const candidates = [];
  arr.forEach((it, i) => {
    if (!it || typeof it !== "object" || typeof it.text !== "string" || !it.text.trim()) {
      errors.push(`item ${i}: missing text`);
      return;
    }
    candidates.push({
      text: it.text.trim(),
      detail: typeof it.detail === "string" ? it.detail.trim() : "",
      tags: Array.isArray(it.tags) ? it.tags.filter((t) => typeof t === "string") : [],
      confidence: typeof it.confidence === "number" ? it.confidence : null,
    });
  });
  return { candidates, errors };
}

const TRANSIENT_RES = [
  /\b(is|are|was|were|has been|have been) (being|currently|now) \w+/i,
  /\b(is|are|was|were) (complete|completed|finished|done|shipped|ready|uncommitted|blocked)\b/i,
  /\b(in progress|so far|for now|at present|currently|recently|today|yesterday|this week|next step)\b/i,
  /\b(session|sibling session|checkpoint|worktree)\b/i,
  /\bbranch\b/i,
  /\ball tests pass\b/i,
  /\b(implementing|adding|fixing|refactoring|migrating|investigating|debugging) /i,
];
const NARRATIVE_RE =
  /^(the user|this session|we|it|they)\b.*\b(worked|did|tried|discussed|continued)\b/i;
const DANGLING_RE =
  /^(the|this|that|its?)\s+(project|system|automation|extension|app|application|tool|engine|environment|service|script|bot|plan|work|feature|change|approach|implementation|repo|repository|codebase)\b/i;

function normalize(s) {
  return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}

// Full miner gate. Returns null when the candidate survives, else the reason.
// Order: cheap structural checks first, model confidence last (it came back
// uniformly high even on narrative, so it is never trusted alone).
export function minerGate(cand, instructionText) {
  const text = (cand && cand.text ? cand.text : "").trim();
  if (text.length < MINED_MIN_LEN) return `too short (${text.length} < ${MINED_MIN_LEN})`;
  if (text.length > MINED_MAX_LEN) return `too long (${text.length} > ${MINED_MAX_LEN})`;
  if (typeof cand.confidence !== "number" || cand.confidence < MIN_CONFIDENCE) {
    return `confidence below ${MIN_CONFIDENCE}`;
  }
  if (looksSecret(text) || (cand.detail && looksSecret(cand.detail))) return "secret-looking";
  for (const re of TRANSIENT_RES) {
    if (re.test(text)) return "transient (progress/status/time-bound)";
  }
  if (NARRATIVE_RE.test(text)) return "narrative (session storytelling)";
  if (DANGLING_RE.test(text)) return "dangling subject (names nothing outside its session)";
  const norm = normalize(text);
  if (norm && normalize(instructionText).includes(norm)) return "prompt echo (copied from extraction instructions)";
  return null;
}
