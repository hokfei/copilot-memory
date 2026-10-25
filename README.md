# memory

Global (not per-project) long-term memory for the GitHub Copilot app, as a
sibling to the `session-search` extension. Everything runs on your machine:
no network at query time, no API keys. Explicit `remember` first;
auto-capture is a non-goal for v1.

> **Divergence risk:** `lib/chunk.mjs`, `lib/read.mjs`, and the embedder
> pattern in `lib/embed.mjs` were copied from `session-search`. If that
> extension changes its chunking, prefixes, or embedding setup, re-embed
> assumptions here must be re-checked (dims + `QUERY_PREFIX` parity).

## How it works

```
remember(text) ──embed──▶ memory.db (memories + memory_vectors, sqlite)
(Qwen3-Embedding-0.6B, 1024 dims, local GGUF — shared file, never copied)

recall(query) ──embed (same QUERY_PREFIX)──▶ dot product, pinned first ──▶ top limit ──▶ max 2 per scope
```

Memories live in `memory.db` (sqlite via `node:sqlite`):

- `memories(id, text, scope, tags JSON, source_session NULL, pinned, created_at, updated_at)`
- `memory_vectors(id, vector BLOB)` — Float32, same dims as the session-search index

Scope is `global` or `project:<project_id|repo>`.

Three mechanics ported from the old keyword-based `opencode-memory-plugin`
(semantic search here supersedes its keyword matching — only these carry over):

- **Content-hash ids**: `id = sha1(normalize(text))` (case/whitespace-folded).
  Re-remembering the same text refreshes the entry (latest scope/tags win,
  `updated_at` bumped, vector re-embedded) instead of duplicating. The key is
  global — the same text in another scope updates the one entry.
- **Pinning**: pinned entries rank before all unpinned ones in `recall`
  (then by score) and are never auto-evicted. Set via `remember(..., pin:true)`
  or the `pin_memory` tool.
- **Per-scope caps**: each scope holds at most `DEFAULT_SCOPE_CAP` (500)
  entries. New inserts past the cap evict the oldest **unpinned** entries
  (by `created_at`), vectors included.

**Split with session-search:** session history search stays in
`session-search` (`search_sessions`); durable facts the user asked to keep
live here (`recall`). The two never merge indexes.

## Setup

Requires Node 22.5+ (uses `node:sqlite`).

```bash
cd ~/.copilot/extensions/memory

# 1. Install (downloads prebuilt llama.cpp binaries, no compiler needed)
npm install
node ./node_modules/node-llama-cpp/dist/cli/cli.js postinstall

# 2. Restart the Copilot app so it loads the extension
```

The embedding model is shared with session-search and resolved via
`SESSION_MEMORY_EMBED_MODEL` (default:
`../session-search/models/Qwen3-Embedding-0.6B-Q8_0.gguf`). The generative
model for consolidation/bootstrap resolves via `SESSION_MEMORY_CHAT_MODEL`
(default: `~/models/Qwen3.5-4B-Q4_K_M.gguf`).

Performance note: on machines where node-llama-cpp's prebuilt Vulkan binary
fails its startup test (observed on a 4 GB GTX 1650), it falls back to CPU
automatically. Everything still works — embedding a short text takes seconds,
but the Qwen3.5-4B passes (consolidate, bootstrap) take minutes on CPU. The
embedder and chat model are never resident at the same time (mutual
`dispose`), so the 4 GB card is not oversubscribed after the first load.

## Tools

| Tool | Purpose |
| ---- | ------- |
| `remember` | Save a fact/note (`text`, `scope` default global, `tags`, `pin?`). Same text refreshes instead of duplicating; new entries past the 500/scope cap evict oldest unpinned. |
| `recall` | Semantic search over memories (`query`, `scope?`, `limit?`). Pinned entries rank first. |
| `forget` | Delete by `id` directly; query form echoes the top match and needs `confirm:true` (two-step). |
| `pin_memory` | Pin/unpin by `id` (`pinned` default true). Pinned ranks first in recall, exempt from eviction. |
| `list_memories` | Filterable browse (`scope?`, `tag?`, `limit?`), no embedding. |
| `consolidate_memories` | On-demand LLM pass: dedupe, merge near-duplicates, flag contradictions. |
| `bootstrap_instructions` | Draft a muse-instructions.md from project memories + top session hits. Text only — never writes into repos. |

## Backfill (one-time, approval-gated)

```bash
node scripts/backfill.mjs [project:<name>|global] [--limit N]
```

Extracts candidate memories from local session history into `staging.json`
for **user approval**. Never inserts silently — import approved items with
`remember` (or `scripts/backfill.mjs --import staging.json` after editing).
`staging.json` is written as explicit UTF-8; `--import` refuses UTF-16 files
with a clear error (PowerShell `echo ... > file` redirection writes UTF-16 —
build the file with `node` or re-save as UTF-8 instead).

## Tests

```bash
npm test
```

Self-consistency checks against a throwaway DB (insert/get/list/search
round-trips, scope/tag filters, score ordering, per-scope cap, limit clamp,
delete semantics). Real `memory.db` is never touched.

## Layout

```
extension.mjs          entry point, registers the seven tools
lib/paths.mjs          path resolution (relative, no hardcoded usernames)
lib/read.mjs           read-only session store readers (copied from session-search)
lib/chunk.mjs          sliding-window chunking + keying/hashing (copied)
lib/embed.mjs          local embedding, model path from env (copied + repointed)
lib/store.mjs          sqlite memories + vectors (content-hash upsert, pinned, scope caps)
lib/search.mjs         brute-force dot-product search + per-scope cap
lib/vectors.mjs        fill missing vectors (shared by recall, forget)
lib/synthesize.mjs     lazy Qwen3.5-4B chat context, budgeted prompts
scripts/backfill.mjs   one-time history extraction into staging.json
scripts/test-memory.mjs    self-consistency tests (throwaway DB)
memory.db              memories (generated, gitignored)
staging.json           backfill candidates awaiting approval (generated)
```

## Recovery

If recall or forget reports that stored vectors disagree with the model, the
embedding model changed. Re-embed the store in place:

```bash
cd ~/.copilot/extensions/memory
node scripts/backfill.mjs --reembed
```

## License

MIT
