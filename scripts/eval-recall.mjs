// Recall evaluation: ~20 questions with known-answer rows, recall@3.
// Run: node scripts/eval-recall.mjs
// Uses a throwaway DB (never touches memory.db). Reports honestly, including
// where semantic search loses to token overlap. Exit 0 always: an eval
// measures, it does not gate.
import { rmSync } from "node:fs";
import { join } from "node:path";
import { extRoot } from "../lib/paths.mjs";
import { openDb, upsertMemory, insertVector } from "../lib/store.mjs";
import { searchMemories } from "../lib/search.mjs";
import { embedDocuments, embedQuery, disposeEmbedder } from "../lib/embed.mjs";
import { buildPreload, MAX_INJECT_CHARS } from "../lib/preload.mjs";

process.chdir(extRoot);

const FACTS = [
  { text: "Production deploys run on Thursdays with a staging gate", scope: "global" },
  { text: "The webapp prefers TypeScript strict mode", scope: "project:webapp" },
  { text: "Staging database resets nightly at 0300 UTC", scope: "project:infra" },
  { text: "On-call incidents page PagerDuty within five minutes", scope: "global" },
  { text: "The monorepo uses pnpm workspaces for package management", scope: "project:webapp" },
  { text: "API errors return RFC 7807 problem JSON with trace ids", scope: "project:infra" },
  { text: "Feature flags live in LaunchDarkly under the webapp project", scope: "global" },
  { text: "Backups retain thirty daily snapshots in cold storage", scope: "project:webapp" },
  { text: "The billing service runs on Postgres fourteen in production", scope: "project:infra" },
  { text: "Code review requires two approvals before merge", scope: "global" },
  { text: "The status page updates every minute during incidents", scope: "project:webapp" },
  { text: "Secrets live in Vault under the production namespace", scope: "project:infra" },
  // Confusable pairs: same shape, different key detail. This is where
  // recall can honestly fail — disambiguation under vocabulary overlap.
  { text: "Production deploys run on Fridays for the mobile app", scope: "global" },
  { text: "Staging database resets nightly at 0400 UTC for analytics", scope: "project:infra" },
  { text: "The mobile app prefers Kotlin with strict null checks", scope: "project:mobile" },
  { text: "On-call incidents page Opsgenie within fifteen minutes", scope: "global" },
];

// kind: paraphrase (embeddings should win) vs exact (token overlap often wins).
const QUESTIONS = [
  { q: "when do we ship to production", want: 0, kind: "paraphrase" },
  { q: "which language settings does the frontend use", want: 1, kind: "paraphrase" },
  { q: "what time does staging get wiped", want: 2, kind: "paraphrase" },
  { q: "how fast must on-call respond", want: 3, kind: "paraphrase" },
  { q: "what package manager does the monorepo use", want: 4, kind: "paraphrase" },
  { q: "what shape do API failures take", want: 5, kind: "paraphrase" },
  { q: "where are feature toggles configured", want: 6, kind: "paraphrase" },
  { q: "how long are backups kept", want: 7, kind: "paraphrase" },
  { q: "which database does billing use", want: 8, kind: "paraphrase" },
  { q: "how many reviewers before merging", want: 9, kind: "paraphrase" },
  { q: "staging database reset", want: 2, kind: "exact" },
  { q: "PagerDuty on-call", want: 3, kind: "exact" },
  { q: "pnpm workspaces", want: 4, kind: "exact" },
  { q: "LaunchDarkly", want: 6, kind: "exact" },
  { q: "Vault production namespace secrets", want: 11, kind: "exact" },
  { q: "Thursday deploy staging gate", want: 0, kind: "exact" },
  { q: "RFC 7807 problem JSON", want: 5, kind: "exact" },
  { q: "two approvals merge", want: 9, kind: "exact" },
  { q: "status page incident updates", want: 10, kind: "exact" },
  { q: "cold storage snapshots", want: 7, kind: "exact" },
  { q: "when does the mobile app ship", want: 12, kind: "paraphrase" },
  { q: "what time does analytics staging reset", want: 13, kind: "paraphrase" },
  { q: "Friday mobile deploy", want: 12, kind: "exact" },
  { q: "Opsgenie on-call", want: 14, kind: "exact" },
];

async function main() {
  const testDb = join(extRoot, ".test-eval.db");
  rmSync(testDb, { force: true });
  rmSync(testDb + "-wal", { force: true });
  rmSync(testDb + "-shm", { force: true });
  const db = openDb(testDb);
  try {
    const ids = [];
    const vecs = await embedDocuments(FACTS.map((f) => f.text), (done, total) => {
      if (done % 6 === 0 || done === total) console.log(`embedded ${done}/${total}`);
    });
    FACTS.forEach((f, i) => {
      const { id } = upsertMemory(db, { text: f.text, scope: f.scope, tags: [] });
      insertVector(db, id, vecs[i]);
      ids.push(id);
    });
    const withVecs = FACTS.map((f, i) => ({ ...f, id: ids[i], vector: vecs[i] }));

    let hits = 0;
    const byKind = { paraphrase: [0, 0], exact: [0, 0] };
    for (const { q, want, kind } of QUESTIONS) {
      const qv = await embedQuery(q);
      const top = searchMemories(withVecs, qv, 5).slice(0, 3);
      const rank = top.findIndex((r) => r.id === ids[want]);
      const ok = rank >= 0;
      if (ok) hits++;
      byKind[kind][0] += ok ? 1 : 0;
      byKind[kind][1]++;
      console.log(`${ok ? "HIT " : "MISS"} [${kind}] rank=${rank >= 0 ? rank + 1 : "-"} want=${want} :: ${q}`);
    }
    console.log(`\nrecall@3: ${hits}/${QUESTIONS.length} (${((hits / QUESTIONS.length) * 100).toFixed(0)}%)`);
    for (const k of ["paraphrase", "exact"]) {
      const [h, n] = byKind[k];
      console.log(`  ${k}: ${h}/${n} (${((h / n) * 100).toFixed(0)}%)`);
    }

    // Hook latency: realistic preload over the seeded store, plus a scope
    // with no repo rows (global facts still apply everywhere).
    for (const [label, scope] of [["seeded", "project:webapp"], ["no-repo-rows", "project:nothing-here"]]) {
      const t0 = Date.now();
      const built = buildPreload(db, scope, MAX_INJECT_CHARS);
      console.log(`hook latency [${label}]: ${Date.now() - t0}ms (injected=${built.injected}, chars=${built.text ? built.text.length : 0})`);
    }
  } finally {
    db.close();
    rmSync(testDb, { force: true });
    rmSync(testDb + "-wal", { force: true });
    rmSync(testDb + "-shm", { force: true });
  }
  await disposeEmbedder().catch(() => {});
}

main().catch((err) => {
  console.error(`eval failed: ${err.message}`);
  process.exitCode = 1;
});
