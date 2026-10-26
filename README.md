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

- `memories(id, text, scope, tags JSON, source_session NULL, pinned, kind, detail, superseded_by NULL, supersedes NULL, confirmed, created_at, updated_at)`
- `memory_vectors(id, vector BLOB)` — Float32, same dims as the session-search index

Scope is `global` or `project:<project_id|repo>`.

- **kind/detail**: `kind` is `state` (standing truth) or `event` (a past
  change), always classified structurally from the text — callers cannot set
  it. Rows without a kind read as state, so migration removes nothing from
  recall. Headlines cap at 300 chars; reasons/examples/evidence go in
  `detail`, which is searchable and returned by recall but never preloaded.
  Events stay searchable but out of the session-start preload (they would
  otherwise push out the standing truths that change behaviour); pinned rows
  are always preloadable.

Three mechanics ported from the old keyword-based `opencode-memory-plugin`
(semantic search here supersedes its keyword matching — only these carry over):

- **Content-hash ids**: `id = sha1(normalize(text))` (case/whitespace-folded).
  Re-remembering the same text refreshes the entry (latest scope/tags win,
  `updated_at` bumped, vector re-embedded) instead of duplicating. The key is
  global — the same text in another scope updates the one entry.
- **Pinning**: pinned entries rank before all unpinned ones in `recall`
  (then by score) and are never auto-evicted. Set via `remember(..., pin:true)`
  or the `pin_memory` tool.
- **Near-duplicate supersession** (beyond exact-hash dedupe): on remember,
  the new row is compared against active rows in the SAME scope with
  IDF-weighted Jaccard over tokens (`lib/similarity.mjs`, threshold 0.45).
  On a hit the old row is retired (`superseded_by` = new id — kept for audit,
  hidden from recall/list/preload/caps), pin carries forward, and the new row
  records `supersedes`. Identical normalized text instead bumps a `confirmed`
  counter (shown as `×N`). Unrelated facts, cross-scope paraphrases, and short
  facts contained in long ones do not merge.
- **Per-scope caps**: each scope holds at most `DEFAULT_SCOPE_CAP` (500)
  entries. New inserts past the cap evict the oldest **unpinned** entries
  (by `created_at`), vectors included. Superseded rows don't count.

`recall_memory`/`forget_memory` share one JOIN loader (`lib/vectors.mjs`):
memories + vectors in a single query, decoded once, cached in memory keyed
by the db + `-wal` freshness stamp (WAL commits only touch the sidecar until
checkpoint). `memory.db` runs in WAL mode with a 5s busy timeout so parallel
sessions don't hit `SQLITE_BUSY`.

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
| `remember_memory` | Save a durable fact (`text` headline max 300 chars, `detail?` for reasons/evidence, `scope` default global, `tags`, `pin?`). Only what stays true in six months — never progress status, branch names, credentials, or repo-readable facts. Secrets rejected. Same text refreshes instead of duplicating; new entries past the 500/scope cap evict oldest unpinned. Past changes auto-classify as events (searchable, not preloaded). Renamed from `remember`. |
| `recall_memory` | Semantic search over memories (`query`, `scope?`, `limit?`). Pinned entries rank first; returns headline + detail + kind. Renamed from `recall`. |
| `forget_memory` | Delete by `id` directly; query form echoes the top match and needs `confirm:true` (two-step). Renamed from `forget`. |
| `pin_memory` | Pin/unpin by `id` (`pinned` default true). Pinned ranks first in recall, exempt from eviction. |
| `list_memories` | Filterable browse (`scope?`, `tag?`, `limit?`), no embedding. |
| `consolidate_memories` | On-demand LLM pass: dedupe, merge near-duplicates, flag contradictions. |
| `bootstrap_instructions` | Draft a muse-instructions.md from project memories + top session hits. Text only — never writes into repos. |

## Session-start preload

The extension registers an `onSessionStart` hook that injects a capped block
of known facts, so the agent never has to remember to call recall:

```
Known about <owner/repo>:
- fact (+)
General working context:
- fact
```

- Selection: active (non-superseded) rows where scope is `global` or matches
  the current repo. Pinned first, then repo-scoped before global, then higher
  `confirmed` count, then newest. Events are excluded unless pinned.
- Budget: `MAX_INJECT_CHARS` (3500), header included; a line that doesn't fit
  is skipped so one long fact can't shut out shorter ones behind it. `(+)`
  marks rows with detail — fetch it with `recall_memory`. The footer tells the
  model how many in-scope rows were withheld.
- Repo resolution: walks up from the session working directory to `.git`; in
  app worktrees `.git` is a file (`gitdir: <path>`), followed up to the shared
  config, whose `url = ...` parses to `owner/repo` (scope `project:<owner/repo>`).
  Falls back to the directory basename — facts learned in one worktree apply in
  every checkout of the same repo.
- Safety: the hook body never throws, reads sqlite only, and never loads
  node-llama-cpp on this path (~1ms). Hosts without hook support fall back to
  tools-only registration; transient join failures retry on backoff.
- One line per hook run is appended to `hook-audit.log` (timestamp, cwd, repo,
  injected, withheld, chars) — the only way to know whether a host ran the hook.

## Offline auto-capture (miner)

`scripts/mine.mjs` reads session checkpoint prose from
`~/.copilot/session-store.db` (already summarised by the cloud model —
extraction from clean text, never comprehension of raw transcripts), asks the
local chat model (`SESSION_MEMORY_CHAT_MODEL`) for durable reusable facts as
JSON, runs every candidate through the full gate
(`lib/miner.mjs`: 25–200 chars, confidence ≥ 0.7, secret-looking, transient,
narrative, dangling-subject, prompt-echo), then stages survivors to
`mined-staging.json` for approval (default). `--auto-import` stores survivors
at/above `--min-confidence` (default 0.85) directly through the same
upsert + supersede + cap path as `remember_memory`; `--import` imports a
staging file. `mined.json` tracks processed `session:checkpoint` ids so
nothing is reprocessed. On-device: no network, no AI credits.

```bash
npm run mine -- --limit 25        # stage for approval
node scripts/mine.mjs --import    # import staged
node scripts/mine.mjs --auto-import
npm run schedule-weekly           # OS scheduled task, weekly Sunday 03:00
```

## Evaluation

```bash
npm run eval
```

24 questions with known-answer rows over a throwaway 16-fact store:
12 paraphrased (embeddings should win), 12 exact-name (token overlap often
wins), plus 4 confusable pairs with vocabulary overlap. Reports recall@3
split by kind, per-question ranks, and session-start hook latency.
Current: **23/24 (96%)** — paraphrase 12/12, exact 11/12; the miss is
"Opsgenie on-call" (PagerDuty neighbor outranks it). Hook preload builds in
~1ms over sqlite. An eval measures, it does not gate.

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
lib/preload.mjs         session-start hook: repo resolution, selection, budget, join fallback
lib/store.mjs          sqlite memories + vectors (content-hash upsert, pinned, scope caps)
lib/search.mjs         brute-force dot-product search + per-scope cap
lib/vectors.mjs        fill missing vectors (shared by recall, forget)
lib/synthesize.mjs     lazy Qwen3.5-4B chat context, budgeted prompts
scripts/backfill.mjs   one-time history extraction into staging.json
scripts/mine.mjs       offline checkpoint miner (stage default, --auto-import)
scripts/schedule-weekly.mjs  register weekly OS mining task
scripts/eval-recall.mjs    recall@3 over 24 questions + hook latency (eval, not a gate)
scripts/test-memory.mjs    self-consistency tests (throwaway DB)
memory.db              memories (+ -wal/-shm under WAL, generated, gitignored)
hook-audit.log         one line per session-start hook run (generated, gitignored)
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
