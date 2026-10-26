import { joinSession } from "@github/copilot-sdk/extension";
import { basename } from "node:path";
import { embedQuery, embedDocuments } from "./lib/embed.mjs";
import { loadScoredMemories } from "./lib/vectors.mjs";
import { looksSecret, headlineTooLong, HEADLINE_CAP } from "./lib/quality.mjs";
import { resolveRepo, repoScope, buildPreload, auditHook, joinWithFallback, MAX_INJECT_CHARS } from "./lib/preload.mjs";
import { memoryDbPath, hookAuditPath } from "./lib/paths.mjs";
import {
  openDb,
  normalizeScope,
  isValidScope,
  upsertMemory,
  applySupersession,
  memoryIdFor,
  getMemory,
  insertVector,
  deleteMemory,
  updateMemoryMeta,
  setPinned,
  enforceScopeCap,
  DEFAULT_SCOPE_CAP,
  listMemories,
  countMemories,
} from "./lib/store.mjs";
import {
  searchMemories,
  dimsMismatch,
  DEFAULT_LIMIT,
  MAX_LIMIT,
} from "./lib/search.mjs";

function formatMemory(m) {
  const tags = m.tags.length > 0 ? ` [${m.tags.join(", ")}]` : "";
  const pin = m.pinned ? " 📌" : "";
  const kind = m.kind === "event" ? " [event]" : "";
  const conf = (m.confirmed | 0) > 1 ? ` (×${m.confirmed})` : "";
  const detail = m.detail ? `\nDetails: ${m.detail}` : "";
  return `(${m.id}) [${m.scope}]${tags}${pin}${kind}${conf}\n${m.text}${detail}`;
}

const session = await joinWithFallback(joinSession, {
  tools: [
    {
      name: "remember_memory",
      description:
        "Save a durable fact to long-term memory — only what will still be true in six months " +
        "(conventions, environments, decisions, ownership, recurring pitfalls). Do NOT store " +
        "progress/task/review status, branch names, credentials, or anything readable from the repo. " +
        "Headline `text` is capped at 300 chars; put reasons, examples, and evidence in `detail` " +
        "(searchable and returned by recall, never preloaded). Past changes are auto-classified " +
        "as events and stay searchable but out of the preload. Secrets are rejected. " +
        "Scope defaults to global; use project:<name> for project-specific memories. " +
        "Re-remembering the same text refreshes the existing entry instead of duplicating it.",
      parameters: {
        type: "object",
        properties: {
          text: { type: "string", description: "Headline fact, max 300 chars. Reasons/evidence go in detail." },
          detail: {
            type: "string",
            description: "Optional longer context: reasons, examples, evidence. Searchable and returned by recall_memory, never preloaded.",
          },
          scope: {
            type: "string",
            description: "Memory scope: 'global' or 'project:<name>'. Default 'global'.",
          },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Optional tags for filtering (e.g. ['preference', 'workflow']).",
          },
          pin: {
            type: "boolean",
            description: "Pin this memory so it ranks first in recall and is never auto-evicted.",
          },
        },
        required: ["text"],
      },
      handler: async (args) => {
        const text = (args.text || "").trim();
        if (!text) return "Give a non-empty text to remember.";
        const detail = typeof args.detail === "string" ? args.detail.trim() : "";
        if (looksSecret(text) || (detail && looksSecret(detail))) {
          return "Refused: text looks like it contains a secret (password, token, key, credential, or credentialed connection string). Never store secrets in memory.";
        }
        if (headlineTooLong(text)) {
          return `Refused: headline is ${text.length} chars (cap ${HEADLINE_CAP}). Shorten the headline and move reasons/examples/evidence into detail.`;
        }
        const scope = normalizeScope(args.scope);
        if (!isValidScope(scope)) {
          return `Invalid scope '${args.scope}'. Use 'global' or 'project:<name>'.`;
        }
        const tags = Array.isArray(args.tags)
          ? args.tags.filter((t) => typeof t === "string" && t.trim()).map((t) => t.trim())
          : [];
        const vec = (await embedDocuments([text]))[0];
        const db = openDb();
        try {
          const { id, created } = upsertMemory(db, {
            text,
            scope,
            tags,
            pin: args.pin ?? null,
            detail: detail || null,
          });
          insertVector(db, id, vec);
          if (!created) {
            const conf = getMemory(db, id)?.confirmed ?? 1;
            return `Confirmed existing memory (${id}) [${scope}] — recorded ${conf} time${conf === 1 ? "" : "s"}; no duplicate created.`;
          }
          const sup = applySupersession(db, id, text, scope);
          const supNote = sup
            ? ` Superseded near-duplicate (${sup.supersededId}, similarity ${sup.score.toFixed(2)}).`
            : "";
          const evicted = enforceScopeCap(db, scope, DEFAULT_SCOPE_CAP);
          const note = evicted > 0 ? ` (evicted ${evicted} oldest unpinned entr${evicted === 1 ? "y" : "ies"} over the ${DEFAULT_SCOPE_CAP}/scope cap)` : "";
          return `Remembered (${id}) [${scope}]${args.pin ? " 📌" : ""}.${supNote}${note}`;
        } finally {
          db.close();
        }
      },
    },
    {
      name: "recall_memory",
      description:
        "Search long-term memories by meaning. Searches memories only — past session history stays in the session-search extension.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "What to look for in memory." },
          scope: {
            type: "string",
            description: "Optional scope filter: 'global' or 'project:<name>'.",
          },
          limit: {
            type: "integer",
            description: `Max results (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT}).`,
            minimum: 1,
            maximum: MAX_LIMIT,
          },
        },
        required: ["query"],
      },
      handler: async (args) => {
        const query = (args.query || "").trim();
        if (!query) return "Give a non-empty query to recall memories.";
        const db = openDb();
        try {
          if (countMemories(db) === 0) return "No memories stored yet. Use remember_memory to save some.";
          const scopeFilter = args.scope ? normalizeScope(args.scope) : null;
          const withVecs = await loadScoredMemories(db, memoryDbPath, scopeFilter);
          if (withVecs.length === 0) {
            return `No memories in scope '${scopeFilter}'.`;
          }
          const queryVec = await embedQuery(query);
          const mismatch = dimsMismatch(withVecs, queryVec);
          if (mismatch) {
            return `${mismatch} The embedding model changed — re-embed the store (see the memory extension README under Recovery), then try again.`;
          }
          const results = searchMemories(withVecs, queryVec, args.limit ?? DEFAULT_LIMIT);
          if (results.length === 0) return "No matching memories found.";
          return results
            .map(
              (r, i) =>
                `[${i + 1}] ${formatMemory(r)} (score ${r.score.toFixed(3)})`
            )
            .join("\n\n---\n\n");
        } finally {
          db.close();
        }
      },
    },
    {
      name: "forget_memory",
      description:
        "Delete a memory by id, or by query (deletes the single top match, only after confirmation).",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "Memory id to delete." },
          query: {
            type: "string",
            description: "Alternatively, text to match the single top memory.",
          },
          confirm: {
            type: "boolean",
            description: "Required true when deleting by query: confirms the echoed match.",
          },
        },
      },
      handler: async (args) => {
        const db = openDb();
        try {
          if (args.id) {
            const ok = deleteMemory(db, args.id);
            return ok ? `Forgot memory (${args.id}).` : `No memory found with id '${args.id}'.`;
          }
          const query = (args.query || "").trim();
          if (!query) return "Give an id or a query to forget a memory.";
          if (countMemories(db) === 0) return "No memories stored yet.";
          const withVecs = await loadScoredMemories(db, memoryDbPath);
          const queryVec = await embedQuery(query);
          const mismatch = dimsMismatch(withVecs, queryVec);
          if (mismatch) {
            return `${mismatch} The embedding model changed — re-embed the store (see the memory extension README under Recovery), then try again.`;
          }
          const [top] = searchMemories(withVecs, queryVec, 1);
          if (!top) return "No matching memories found.";
          if (args.confirm !== true) {
            return `About to forget this memory:\n${formatMemory(top)}\nCall forget_memory again with id '${top.id}' (or query plus confirm:true) to delete it.`;
          }
          deleteMemory(db, top.id);
          return `Forgot memory (${top.id}).`;
        } finally {
          db.close();
        }
      },
    },
    {
      name: "list_memories",
      description: "Browse stored memories, optionally filtered by scope or tag. No embedding needed.",
      parameters: {
        type: "object",
        properties: {
          scope: {
            type: "string",
            description: "Optional scope filter: 'global' or 'project:<name>'.",
          },
          tag: { type: "string", description: "Optional tag filter." },
          limit: {
            type: "integer",
            description: "Max results (default 100, max 500).",
            minimum: 1,
            maximum: 500,
          },
        },
      },
      handler: async (args) => {
        const db = openDb();
        try {
          const mems = listMemories(db, {
            scope: args.scope || null,
            tag: args.tag || null,
            limit: args.limit ?? 100,
          });
          if (mems.length === 0) return "No memories found.";
          return mems.map((m, i) => `[${i + 1}] ${formatMemory(m)}`).join("\n\n");
        } finally {
          db.close();
        }
      },
    },
    {
      name: "consolidate_memories",
      description:
        "On-demand cleanup of memories in a scope: merges near-duplicates, flags contradictions. Uses the local chat model; may take a minute.",
      parameters: {
        type: "object",
        properties: {
          scope: {
            type: "string",
            description: "Scope to consolidate: 'global' or 'project:<name>'. Default 'global'.",
          },
        },
      },
      handler: async (args) => {
        const scope = normalizeScope(args.scope);
        if (!isValidScope(scope)) {
          return `Invalid scope '${args.scope}'. Use 'global' or 'project:<name>'.`;
        }
        const db = openDb();
        let mems;
        try {
          mems = listMemories(db, { scope, limit: 200 });
        } finally {
          db.close();
        }
        if (mems.length < 2) {
          return `Only ${mems.length} memor${mems.length === 1 ? "y" : "ies"} in scope '${scope}' — nothing to consolidate.`;
        }
        const { runChat, extractJson } = await import("./lib/synthesize.mjs");
        const systemPrompt =
          "You curate a personal memory store. Reply with exactly one JSON object and no other text.";
        const userPrompt = `Given these memories (ids in parentheses), identify:
- "merges": groups of near-duplicates to combine into one memory. Each has "keep_id", "merge_ids" (absorbed and deleted), and "merged_text" (the combined wording).
- "contradictions": pairs that disagree, as {"a_id","b_id","note"}.
- "standalone": ids that are fine as-is.
Reply schema: {"merges":[{"keep_id":"...","merge_ids":["..."],"merged_text":"..."}],"contradictions":[{"a_id":"...","b_id":"...","note":"..."}],"standalone":["..."]}`;
        let reply;
        try {
          ({ reply } = await runChat(
            systemPrompt,
            userPrompt,
            mems.map((m) => `(${m.id}) ${m.text}`)
          ));
        } catch (err) {
          return `Consolidation failed to run the local model: ${err.message}`;
        }
        let plan;
        try {
          plan = extractJson(reply);
        } catch {
          return `The model did not return usable JSON, so nothing was changed. Raw reply:\n${reply}`;
        }
        const byId = new Map(mems.map((m) => [m.id, m]));
        const applied = [];
        const warnings = [];
        // The plan is untrusted LLM JSON: validate shape BEFORE opening the
        // write DB, pre-embed every rewrite, then apply in one transaction.
        if (!plan || typeof plan !== "object" || Array.isArray(plan)) {
          return `The model did not return a usable plan, so nothing was changed. Raw reply:\n${reply}`;
        }
        if (plan.merges !== undefined && !Array.isArray(plan.merges)) {
          warnings.push("ignoring malformed merges list");
        }
        const merges = Array.isArray(plan.merges) ? plan.merges : [];
        const ops = [];
        const claimed = new Set();
        const seenTexts = new Set();
        for (const merge of merges) {
          if (!merge || typeof merge !== "object" || Array.isArray(merge)) {
            warnings.push("skip merge: malformed entry (not an object)");
            continue;
          }
          if (typeof merge.keep_id !== "string" || !byId.has(merge.keep_id)) {
            warnings.push(`skip merge: unknown keep_id ${JSON.stringify(merge.keep_id)}`);
            continue;
          }
          if (claimed.has(merge.keep_id)) {
            warnings.push(`skip merge: keep_id ${merge.keep_id} already claimed by an earlier merge`);
            continue;
          }
          if (merge.merge_ids !== undefined && !Array.isArray(merge.merge_ids)) {
            warnings.push(`merge into ${merge.keep_id}: ignoring malformed merge_ids`);
          }
          if (merge.merged_text !== undefined && typeof merge.merged_text !== "string") {
            warnings.push(`merge into ${merge.keep_id}: ignoring non-string merged_text`);
          }
          const mergeIds = Array.isArray(merge.merge_ids) ? merge.merge_ids : [];
          const absorbed = mergeIds.filter(
            (id) => typeof id === "string" && byId.has(id) && id !== merge.keep_id && !claimed.has(id)
          );
          const text = typeof merge.merged_text === "string" ? merge.merged_text.trim() : "";
          if (absorbed.length === 0 && !text) {
            warnings.push(`skip merge into ${merge.keep_id}: nothing to absorb or rewrite`);
            continue;
          }
          if (text && seenTexts.has(memoryIdFor(text))) {
            warnings.push(`skip merge into ${merge.keep_id}: duplicate merged_text`);
            continue;
          }
          let vec = null;
          if (text) {
            try {
              [vec] = await embedDocuments([text]);
            } catch (err) {
              warnings.push(`skip merge into ${merge.keep_id}: embed failed (${err.message})`);
              continue;
            }
            seenTexts.add(memoryIdFor(text));
          }
          claimed.add(merge.keep_id);
          absorbed.forEach((id) => claimed.add(id));
          const keep = byId.get(merge.keep_id);
          const tags = [...new Set([...keep.tags, ...absorbed.flatMap((id) => byId.get(id)?.tags ?? [])])];
          const absorbedSession = absorbed.map((id) => byId.get(id)?.source_session).find(Boolean) ?? null;
          const pin = keep.pinned || absorbed.some((id) => byId.get(id)?.pinned);
          ops.push({ keep, absorbed, text, vec, tags, absorbedSession, pin });
        }
        const remap = new Map(); // old keep id -> new content-hash id
        const db2 = openDb();
        try {
          db2.exec("BEGIN");
          try {
            for (const op of ops) {
              if (op.text) {
                const dupe = getMemory(db2, memoryIdFor(op.text));
                if (dupe && dupe.id !== op.keep.id && !op.absorbed.includes(dupe.id)) {
                  warnings.push(`skip merge into ${op.keep.id}: merged_text duplicates memory (${dupe.id})`);
                  continue;
                }
              }
              for (const id of op.absorbed) {
                deleteMemory(db2, id);
                applied.push(`merged ${id} into ${op.keep.id}`);
              }
              let survivor = op.keep.id;
              if (op.text) {
                deleteMemory(db2, op.keep.id);
                const { id: newId } = upsertMemory(db2, {
                  text: op.text,
                  scope: op.keep.scope,
                  tags: op.tags,
                  sourceSession: op.keep.source_session ?? op.absorbedSession,
                  pin: op.pin,
                });
                insertVector(db2, newId, op.vec);
                remap.set(op.keep.id, newId);
                survivor = newId;
                applied.push(`rewrote ${op.keep.id} as ${newId}`);
              } else {
                updateMemoryMeta(db2, op.keep.id, {
                  tags: op.tags,
                  sourceSession: op.absorbedSession,
                  pin: op.pin,
                });
              }
              for (const id of op.absorbed) remap.set(id, survivor);
            }
            db2.exec("COMMIT");
          } catch (err) {
            db2.exec("ROLLBACK");
            return `Consolidation failed mid-write (${err.message}); rolled back, nothing was changed.`;
          }
        } finally {
          db2.close();
        }
        if (plan.contradictions !== undefined && !Array.isArray(plan.contradictions)) {
          warnings.push("ignoring malformed contradictions list");
        }
        const contradictions = Array.isArray(plan.contradictions) ? plan.contradictions : [];
        const lines = [
          `Consolidated ${mems.length} memories in scope '${scope}':`,
          ...applied.map((a) => `- ${a}`),
        ];
        for (const c of contradictions) {
          if (!c || typeof c !== "object") {
            warnings.push("ignoring malformed contradiction entry");
            continue;
          }
          const a = typeof c.a_id === "string" ? (remap.get(c.a_id) ?? c.a_id) : "?";
          const b = typeof c.b_id === "string" ? (remap.get(c.b_id) ?? c.b_id) : "?";
          const note = typeof c.note === "string" && c.note ? c.note : "flagged";
          lines.push(`- CONTRADICTION ${a} vs ${b}: ${note}`);
        }
        for (const w of warnings) lines.push(`- warning: ${w}`);
        if (applied.length === 0 && contradictions.length === 0) {
          lines.push("- no duplicates found; everything stands alone.");
        }
        return lines.join("\n");
      },
    },
    {
      name: "bootstrap_instructions",
      description:
        "Draft a muse-instructions.md for a project from its project-scoped memories plus top session-history hits. Returns draft text only — never writes into repos.",
      parameters: {
        type: "object",
        properties: {
          project: {
            type: "string",
            description: "Project name or repo (matches the project:<name> memory scope).",
          },
        },
        required: ["project"],
      },
      handler: async (args) => {
        const project = (args.project || "").trim();
        if (!project) return "Give a project name to draft instructions for.";
        const scope = normalizeScope(`project:${project}`);
        const db = openDb();
        let mems;
        try {
          mems = listMemories(db, { scope, limit: 200 });
        } finally {
          db.close();
        }

        let sessionHits = [];
        try {
          // Read the session-search index directly (absolute paths, no
          // cross-extension import — the extension sandbox may block those).
          const { readFileSync: rf, existsSync: ex } = await import("node:fs");
          const { join: jp } = await import("node:path");
          const { copilotDir: cdir } = await import("./lib/paths.mjs");
          const ssBin = jp(cdir, "extensions", "session-search", "index.bin");
          const ssMeta = jp(cdir, "extensions", "session-search", "index.meta.json");
          if (ex(ssBin) && ex(ssMeta)) {
            const meta = JSON.parse(rf(ssMeta, "utf8"));
            const buf = rf(ssBin);
            const vectors = new Float32Array(
              buf.buffer, buf.byteOffset, buf.byteLength / 4
            );
            // Same length guard as loadIndex: a torn index yields NaN scores
            // and garbage hits, so treat disagreement as zero session hits.
            if (!Array.isArray(meta.chunks) || vectors.length !== meta.chunks.length * meta.dims) {
              return `No project memories in scope '${scope}' and the session index is unreadable right now — nothing to draft from. Save project facts with remember first.`;
            }
            const queryVec = await embedQuery(
              `coding conventions, decisions, and setup for the ${project} project`
            );
            if (queryVec.length === meta.dims) {
              const scored = [];
              for (let row = 0; row < meta.chunks.length; row++) {
                let sum = 0;
                const off = row * meta.dims;
                for (let i = 0; i < meta.dims; i++) sum += vectors[off + i] * queryVec[i];
                scored.push({ row, score: sum });
              }
              scored.sort((a, b) => b.score - a.score);
              sessionHits = scored
                .slice(0, 8)
                .map((s) => (meta.chunks[s.row].text || "").slice(0, 600))
                .filter(Boolean);
            }
          }
        } catch {
          sessionHits = [];
        }

        if (mems.length === 0 && sessionHits.length === 0) {
          return `No project memories in scope '${scope}' and no session history found — nothing to draft from. Save project facts with remember first.`;
        }

        const { runChat } = await import("./lib/synthesize.mjs");
        const systemPrompt =
          "You write concise, practical muse-instructions.md drafts for software projects. Reply with only the draft markdown, no preamble.";
        const userPrompt =
          `Draft a muse-instructions.md for the "${project}" project from the material below. ` +
          `Keep it short: conventions, key decisions, workflow notes. Mark anything uncertain with "(verify)".`;
        const items = [
          ...mems.map((m) => `MEMORY: ${m.text}`),
          ...sessionHits.map((s) => `SESSION: ${s.slice(0, 600)}`),
        ];
        try {
          const { reply } = await runChat(systemPrompt, userPrompt, items);
          return (
            `Draft muse-instructions.md for "${project}" (from ${mems.length} memories, ` +
            `${sessionHits.length} session hits — review before using, nothing was written to any repo):\n\n${reply}`
          );
        } catch (err) {
          return `Draft failed to run the local model: ${err.message}`;
        }
      },
    },
    {
      name: "pin_memory",
      description:
        "Pin a memory so it ranks first in recall and is never auto-evicted (or unpin it).",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "Memory id to pin or unpin." },
          pinned: {
            type: "boolean",
            description: "True to pin, false to unpin. Default true.",
          },
        },
        required: ["id"],
      },
      handler: async (args) => {
        if (!(args.id || "").trim()) return "Give a memory id to pin.";
        const db = openDb();
        try {
          const ok = setPinned(db, args.id, args.pinned ?? true);
          if (!ok) return `No memory found with id '${args.id}'.`;
          return args.pinned === false
            ? `Unpinned memory (${args.id}).`
            : `Pinned memory (${args.id}) — ranks first in recall, exempt from eviction.`;
        } finally {
          db.close();
        }
      },
    },
  ],
  },
  { onSessionStart }
);

await session.log("memory extension ready", { ephemeral: true });

// Session-start preload: capped known-facts block, sqlite only, never throws.
// A slow or throwing hook would surface on every session start, so the body
// is defensive and node-llama-cpp is never loaded on this path.
async function onSessionStart(input) {
  const started = Date.now();
  const cwd = (input && input.workingDirectory) || process.cwd();
  let repo = basename(cwd);
  let injected = 0;
  let withheld = 0;
  let chars = 0;
  try {
    repo = resolveRepo(cwd).repo;
    const db = openDb();
    try {
      const built = buildPreload(db, repo, MAX_INJECT_CHARS);
      injected = built.injected;
      withheld = built.withheld;
      chars = built.text ? built.text.length : 0;
      if (built.text) return { additionalContext: built.text };
      return;
    } finally {
      db.close();
    }
  } catch {
    return;
  } finally {
    auditHook(hookAuditPath, {
      ts: new Date().toISOString(),
      cwd,
      repo,
      injected,
      withheld,
      chars,
      ms: Date.now() - started,
    });
  }
}

// (joinWithFallback lives in lib/preload.mjs, tested with a stub joinFn.)
