// Self-consistency tests for the memory extension.
// Uses a throwaway DB file so real memories are never touched.
import { rmSync } from "node:fs";
import { join } from "node:path";
import { extRoot } from "../lib/paths.mjs";
import {
  openDb,
  insertMemory,
  insertVector,
  upsertMemory,
  setPinned,
  enforceScopeCap,
  normalizeText,
  normalizeScope,
  isValidScope,
  memoryIdFor,
  getMemory,
  getVector,
  deleteMemory,
  listMemories,
  getAllVectors,
  countMemories,
  countScopeMemories,
} from "../lib/store.mjs";
import { searchMemories, dimsMismatch, MAX_LIMIT } from "../lib/search.mjs";
import { scoreChunk, prepareImportItems } from "./backfill.mjs";
import { embedQuery, embedDocuments } from "../lib/embed.mjs";

process.chdir(extRoot);

const failures = [];
function check(name, cond, detail = "") {
  if (cond) console.log(`PASS ${name}`);
  else {
    console.log(`FAIL ${name} ${detail}`);
    failures.push(name);
  }
}

const testDb = join(extRoot, ".test-memory.db");
rmSync(testDb, { force: true });
const db = openDb(testDb);
try {
  check("fresh db is empty", countMemories(db) === 0);

  // Near-duplicate similarity (pure token math, no model needed).
  // NOTE: idf is measured over the candidate set, so single-candidate toy
  // cases underweight shared words; realistic tests use multi-row sets with
  // shared boilerplate, exactly how the remember path calls it.
  {
    const { tokenize, findNearDuplicate, SIMILARITY_THRESHOLD } = await import("../lib/similarity.mjs");
    check("tokenize filters short words", JSON.stringify(tokenize("We run Postgres 14 in prod!")) === JSON.stringify(["postgres", "prod"]));
    check("identical texts score 1", findNearDuplicate("we run postgres in prod", [{ id: "a", text: "we run postgres in prod" }]).score === 1);
    check("disjoint texts score null", findNearDuplicate("penguins huddle in winter", [{ id: "a", text: "we run postgres in prod" }]) === null);
    const boilerplate = [
      { id: "a", text: "production postgres database runs version fourteen with nightly backups" },
      { id: "b", text: "production postgres database stores weekly reports for analytics" },
      { id: "c", text: "production postgres database cleans temp tables every sunday" },
    ];
    const para = findNearDuplicate("production postgres database runs version fourteen with weekly backups", boilerplate);
    check("close paraphrase picks right row", para !== null && para.id === "a", `got=${para ? para.id + "@" + para.score.toFixed(3) : "none"}`);
    const contra = findNearDuplicate(
      "production deploy runs every Tuesday morning with full backup",
      [
        { id: "a", text: "production deploy runs every Thursday morning with full backup" },
        { id: "b", text: "production deploy validates config before starting services" },
        { id: "c", text: "production deploy notifies the team after finishing migrations" },
      ]
    );
    check("contradiction supersedes", contra !== null && contra.id === "a" && contra.score >= SIMILARITY_THRESHOLD, `got=${contra ? contra.id + "@" + contra.score.toFixed(3) : "none"}`);
    const swallow = findNearDuplicate("postgres prod", [
      { id: "a", text: "postgres prod windows scheduled task nightly backup rotation policy quarterly review" },
    ]);
    check("short fact not swallowed by long one", swallow === null, `score=${swallow ? swallow.score.toFixed(3) : "none"}`);
    check("empty query is null", findNearDuplicate("a an of", [{ id: "a", text: "we run postgres in prod" }]) === null);
    check("no candidates is null", findNearDuplicate("we run postgres in prod", []) === null);
  }

  // Preload: repo resolution, selection order, budget honesty (no model).
  {
    const { resolveRepo, repoScope, selectPreloadRows, renderPreload, buildPreload, MAX_INJECT_CHARS } =
      await import("../lib/preload.mjs");
    const { mkdtempSync, mkdirSync, writeFileSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { join: joinP, basename: baseP } = await import("node:path");
    const root = mkdtempSync(joinP(tmpdir(), "preload-"));
    mkdirSync(joinP(root, ".git"));
    writeFileSync(joinP(root, ".git", "config"), '[remote "origin"]\n\turl = git@github.com:hokfei/copilot-memory.git\n');
    mkdirSync(joinP(root, "sub", "dir"), { recursive: true });
    check("resolveRepo finds repo from subdir", resolveRepo(joinP(root, "sub", "dir")).repo === "hokfei/copilot-memory");
    mkdirSync(joinP(root, ".git", "worktrees", "wt"), { recursive: true });
    const wt = mkdtempSync(joinP(tmpdir(), "preload-wt-"));
    writeFileSync(joinP(wt, ".git"), `gitdir: ${joinP(root, ".git", "worktrees", "wt")}\n`);
    check("resolveRepo follows worktree gitdir", resolveRepo(wt).repo === "hokfei/copilot-memory");
    const plain = mkdtempSync(joinP(tmpdir(), "preload-plain-"));
    check("resolveRepo falls back to basename", resolveRepo(plain).repo === baseP(plain));
    check("repoScope lowercases", repoScope("HokFei/Copilot-Memory") === "project:hokfei/copilot-memory");

    const scope = repoScope("hokfei/copilot-memory");
    const p0 = upsertMemory(db, { text: "preload ordinary global standing fact", scope: "global", tags: [] });
    const p1 = upsertMemory(db, { text: "preload pinned global standing truth", scope: "global", tags: [] });
    setPinned(db, p1.id, true);
    const p2 = upsertMemory(db, { text: "preload repo standing fact recorded", scope, tags: [], detail: "recorded for reasons" });
    const p3 = upsertMemory(db, { text: "the old script was replaced with polymer clay", scope, tags: [] });
    const p4 = upsertMemory(db, { text: "the old pipeline was migrated to new runners", scope: "global", tags: [] });
    setPinned(db, p4.id, true);
    const sel = selectPreloadRows(db, scope).map((r) => r.id);
    check("pinned first, repo before global", JSON.stringify(sel.slice(0, 4)) === JSON.stringify([p4.id, p1.id, p2.id, p0.id]), `order=${sel.slice(0, 4)}`);
    check("event excluded unless pinned", !sel.includes(p3.id) && sel.includes(p4.id));
    const full = buildPreload(db, "hokfei/copilot-memory");
    check("full preload fits budget", full.text !== null && full.text.length <= MAX_INJECT_CHARS, `chars=${full.text ? full.text.length : "null"}`);
    check("preload marks detail rows", full.text.includes(" (+)"));
    check("preload leads with background disclaimer", full.text.startsWith("Persistent context recalled"));
    const all = selectPreloadRows(db, scope);
    const tight = buildPreload(db, "hokfei/copilot-memory", 550);
    check("tight budget still fits exactly", tight.text !== null && tight.text.length <= 550, `chars=${tight.text ? tight.text.length : "null"}`);
    check("tight budget accounts every row", tight.injected + tight.withheld === all.length, `injected=${tight.injected} withheld=${tight.withheld} total=${all.length}`);
    const rr = renderPreload("x", [{ text: "x".repeat(200), detail: "" }], [{ text: "ok", detail: "" }], 500);
    check(
      "long fact skipped, shorter behind still picked",
      rr.text !== null && rr.injected === 1 && rr.withheld === 1 && rr.text.includes("- ok") && !rr.text.includes("xxxx")
    );
    const none = buildPreload(db, "hokfei/copilot-memory", 10);
    check("impossible budget returns null", none.text === null && none.injected === 0);
    for (const r of [p0, p1, p2, p3, p4]) deleteMemory(db, r.id);
  }
  {
    const { classifyKind, looksSecret, headlineTooLong, HEADLINE_CAP } = await import("../lib/quality.mjs");
    check("classifies standing truth as state", classifyKind("I prefer TypeScript with strict mode") === "state");
    check("classifies past change as event (was fixed)", classifyKind("The retry bug was fixed in the deploy script") === "event");
    check("classifies past change as event (has been)", classifyKind("The pipeline has been validated end to end") === "event");
    check("classifies past change as event (now returns)", classifyKind("The endpoint now returns JSON errors") === "event");
    check("classifies past change as event (have switched)", classifyKind("We have switched to pnpm") === "event");
    check("self-report does not override structure", classifyKind("STATE FACT: the deploy was migrated to Docker") === "event");
    check("rejects api key", looksSecret("the api key for staging is sk-abc123") === true);
    check("rejects password", looksSecret("my password for the deploy user is hunter2") === true);
    check("rejects bearer token", looksSecret("use bearer abc.def.ghi for auth") === true);
    check("rejects credential connection string", looksSecret("postgres://admin:s3cret@db:5432/app") === true);
    check("rejects PEM block", looksSecret("key: -----BEGIN RSA PRIVATE KEY----- ...") === true);
    check("allows endpoint docs", looksSecret("the api endpoint uses REST conventions") === false);
    check("allows plural token prose", looksSecret("we use OAuth tokens for CI") === false);
    check("headline cap boundary", HEADLINE_CAP === 300 && headlineTooLong("x".repeat(300)) === false);
    check("headline over cap", headlineTooLong("x".repeat(301)) === true);
  }
  {
    const q = new Float32Array([1, 0, 0]);
    check("dims match returns null", dimsMismatch([{ vector: new Float32Array([0, 1, 0]) }], q) === null);
    check("dims mismatch returns message", typeof dimsMismatch([{ vector: new Float32Array([0, 1]) }], q) === "string");
    check("missing vector returns message", typeof dimsMismatch([{ vector: null }], q) === "string");
    const pad = (s) => s + " padding".repeat(20);
    check("skips api key chunk", scoreChunk(pad("the api key for staging is abc123, remember this decision")) === -1);
    check("skips password chunk", scoreChunk(pad("my password for the deploy user was rotated, note that")) === -1);
    check("skips private key chunk", scoreChunk(pad("here is the key -----BEGIN RSA PRIVATE KEY----- ...")) === -1);
    check("skips connection string chunk", scoreChunk(pad("the connection string for prod db, keep in mind")) === -1);
    check("scores legit decision chunk", scoreChunk(pad("we decided to use TypeScript strict mode, remember this convention")) >= 2);
    check("keeps endpoint docs", scoreChunk(pad("the api endpoint uses REST conventions, note that we chose JSON")) >= 1);
    check("global scope valid", isValidScope(normalizeScope("global")) === true);
    check("project scope valid", isValidScope(normalizeScope("project:webapp")) === true);
    check("empty project name invalid", isValidScope(normalizeScope("project:")) === false);
    check("blank project name invalid", isValidScope(normalizeScope("project:  ")) === false);
    check("bare scope invalid", isValidScope(normalizeScope("10")) === false);
    const prep = prepareImportItems([
      { text: "  legit memory ", scope: "project:webapp", tags: ["a", "", 7, " b "], source_session: "s1" },
      { text: "bad scope", scope: "10" },
      { text: "   " },
      null,
    ]);
    check("import keeps valid", prep.staged.length === 1 && prep.staged[0].text === "legit memory");
    check("import normalizes scope", prep.staged[0].scope === "project:webapp");
    check("import filters tags", JSON.stringify(prep.staged[0].tags) === JSON.stringify(["a", "b"]));
    check("import counts skipped", prep.skipped === 3, `skipped=${prep.skipped}`);
  }

  const fixtures = [
    { text: "I prefer TypeScript with strict null checks for all new projects", scope: "global", tags: ["preference"] },
    { text: "The deployment pipeline uses GitHub Actions with a staging gate", scope: "project:webapp", tags: ["workflow"] },
    { text: "Emperor penguins huddle together to survive Antarctic winters", scope: "global", tags: [] },
  ];
  const vecs = await embedDocuments(fixtures.map((f) => f.text));
  check("embed dims consistent", vecs.every((v) => v.length === vecs[0].length), `${vecs.map((v) => v.length)}`);
  const ids = fixtures.map((f, i) => {
    const id = insertMemory(db, f);
    insertVector(db, id, vecs[i]);
    return id;
  });
  check("inserted 3 memories", countMemories(db) === 3);

  const got = getMemory(db, ids[0]);
  check("get round-trip", got && got.text === fixtures[0].text && got.scope === "global" && got.tags[0] === "preference");

  const listed = listMemories(db, { scope: "project:webapp" });
  check("scope filter", listed.length === 1 && listed[0].id === ids[1], `got=${listed.length}`);
  const tagged = listMemories(db, { tag: "workflow" });
  check("tag filter", tagged.length === 1 && tagged[0].id === ids[1], `got=${tagged.length}`);

  const all = getAllVectors(db);
  check("vectors join", all.length === 3 && all.every((r) => r.vector.length === vecs[0].length));

  const withMeta = all.map((r) => ({ ...getMemory(db, r.id), vector: r.vector }));
  const qv = await embedQuery("TypeScript strict null checks preference");
  const top = searchMemories(withMeta, qv, 3);
  check("search returns results", top.length > 0 && top.length <= 3);
  check(
    "self-consistency: related memory ranks first",
    top.length > 0 && top[0].id === ids[0],
    `top=${top.length ? top[0].id : "none"} want=${ids[0]}`
  );
  const desc = top.every((r, i) => i === 0 || top[i - 1].score >= r.score);
  check("scores descend", desc);

  const capped = searchMemories(withMeta, qv, 999);
  check("limit clamps to max", capped.length <= MAX_LIMIT, `got=${capped.length}`);

  // Two-step forget safety: delete by id directly.
  check("delete by id", deleteMemory(db, ids[2]) && getMemory(db, ids[2]) === null);
  check("delete unknown id is false", deleteMemory(db, "no-such-id") === false);
  check("count after delete", countMemories(db) === 2);

  // Recall scope filter path: vectors restricted to one scope.
  const scoped = getAllVectors(db, "global").map((r) => ({ ...getMemory(db, r.id), vector: r.vector }));
  check("scoped vector fetch", scoped.length === 1 && scoped[0].id === ids[0], `got=${scoped.length}`);

  // Content-hash dedupe: same text (different case/spacing) refreshes, no duplicate.
  const dupe = upsertMemory(db, {
    text: "  i PREFER typescript WITH strict null checks for all new PROJECTS ",
    scope: "global",
    tags: ["preference"],
  });
  check("dedupe returns same id", dupe.id === ids[0], `got=${dupe.id} want=${ids[0]}`);
  check("dedupe does not duplicate", countMemories(db) === 2, `count=${countMemories(db)}`);
  check(
    "dedupe refreshes text",
    getMemory(db, ids[0]).text.startsWith("i PREFER"),
    getMemory(db, ids[0]).text.slice(0, 40)
  );
  check(
    "id is sha1 of normalized text",
    ids[0] === memoryIdFor(fixtures[0].text) && normalizeText("  A  B ") === "a b"
  );

  // Pinning: setPinned round-trip + pinned ranks first regardless of score.
  check("setPinned unknown id is false", setPinned(db, "no-such-id", true) === false);
  check("setPinned works", setPinned(db, ids[1], true) === true);
  check("pinned flag round-trips", getMemory(db, ids[1]).pinned === true);
  const ranked = searchMemories(
    [
      { id: "u", text: "u", scope: "global", tags: [], pinned: false, vector: new Float32Array([1, 0]) },
      { id: "p", text: "p", scope: "global", tags: [], pinned: true, vector: new Float32Array([0, 1]) },
    ],
    new Float32Array([1, 0]),
    2
  );
  check(
    "pinned ranks first despite lower score",
    ranked.length === 2 && ranked[0].id === "p" && ranked[0].pinned === true,
    JSON.stringify(ranked.map((r) => r.id))
  );

  // Per-scope cap: oldest unpinned evicted first, pinned exempt.
  check("unpin works", setPinned(db, ids[1], false) === true);
  const capDb = openDb(testDb + ".cap");
  try {
    const mk = (t) => {
      const { id } = upsertMemory(capDb, { text: t, scope: "global", tags: [] });
      insertVector(capDb, id, [1, 0]);
      return id;
    };
    const c1 = mk("cap oldest unpinned entry one");
    const c2 = mk("cap pinned keeper entry two");
    const c3 = mk("cap newest unpinned entry three");
    setPinned(capDb, c2, true);
    // Force distinct created_at ordering (ISO strings sort lexicographically).
    capDb.prepare("UPDATE memories SET created_at = ? WHERE id = ?").run("2020-01-01T00:00:00.000Z", c1);
    capDb.prepare("UPDATE memories SET created_at = ? WHERE id = ?").run("2020-01-02T00:00:00.000Z", c2);
    capDb.prepare("UPDATE memories SET created_at = ? WHERE id = ?").run("2020-01-03T00:00:00.000Z", c3);
    const evicted = enforceScopeCap(capDb, "global", 2);
    check("cap evicts oldest unpinned", evicted === 1, `evicted=${evicted}`);
    check("cap count enforced", countScopeMemories(capDb, "global") === 2);
    check("cap keeps pinned", getMemory(capDb, c2) !== null);
    check("cap drops oldest", getMemory(capDb, c1) === null && getVector(capDb, c1) === null);
    check("cap under limit evicts nothing", enforceScopeCap(capDb, "global", 5) === 0);
  } finally {
    capDb.close();
    rmSync(testDb + ".cap", { force: true });
  }

  // Migration: legacy DB without pinned column gains it via openDb.
  const legacyDb = testDb + ".legacy";
  rmSync(legacyDb, { force: true });
  {
    const { DatabaseSync } = await import("node:sqlite");
    const raw = new DatabaseSync(legacyDb);
    raw.exec(
      "CREATE TABLE memories (id TEXT PRIMARY KEY, text TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'global', tags TEXT NOT NULL DEFAULT '[]', source_session TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL); CREATE TABLE memory_vectors (id TEXT PRIMARY KEY, vector BLOB NOT NULL);"
    );
    raw.close();
  }
  const migDb = openDb(legacyDb);
  try {
    const { id } = upsertMemory(migDb, { text: "legacy migration probe", scope: "global", tags: [] });
    check("legacy db migrates + upserts", getMemory(migDb, id).pinned === false);
  } finally {
    migDb.close();
    rmSync(legacyDb, { force: true });
  }

  // kind/detail: structural auto-classification, round-trip, migration.
  const { loadScoredMemories: loadKind, dropVectorCache: dropKindCache } = await import("../lib/vectors.mjs");
  const evId = upsertMemory(db, { text: "the old endpoint was replaced with REST", scope: "global", tags: [] }).id;
  check("event auto-classified", getMemory(db, evId).kind === "event");
  check("state auto-classified", getMemory(db, ids[0]).kind === "state");
  const detId = upsertMemory(db, { text: "detail round trip probe", scope: "global", tags: [], detail: "because reasons" }).id;
  check("detail stored", getMemory(db, detId).detail === "because reasons");
  upsertMemory(db, { text: "detail round trip probe", scope: "global", tags: [] });
  check("detail kept on refresh", getMemory(db, detId).detail === "because reasons");
  upsertMemory(db, { text: "detail round trip probe", scope: "global", tags: [], detail: "new reasons" });
  check("detail replaced when given", getMemory(db, detId).detail === "new reasons");
  check("detail defaults empty", getMemory(db, ids[0]).detail === "");
  dropKindCache(testDb);
  const withKind = await loadKind(db, testDb);
  const evRow = withKind.find((m) => m.id === evId);
  check(
    "loader carries kind/detail",
    evRow && evRow.kind === "event" && withKind.find((m) => m.id === detId).detail === "new reasons"
  );
  deleteMemory(db, evId);
  deleteMemory(db, detId);
  dropKindCache(testDb);

  // Migration: original schema (no pinned/kind/detail) gains all three.
  const legacyDb2 = testDb + ".legacy2";
  rmSync(legacyDb2, { force: true });
  {
    const { DatabaseSync } = await import("node:sqlite");
    const raw = new DatabaseSync(legacyDb2);
    raw.exec(
      "CREATE TABLE memories (id TEXT PRIMARY KEY, text TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'global', tags TEXT NOT NULL DEFAULT '[]', source_session TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL); CREATE TABLE memory_vectors (id TEXT PRIMARY KEY, vector BLOB NOT NULL);"
    );
    raw.close();
  }
  const migDb2 = openDb(legacyDb2);
  try {
    const { id } = upsertMemory(migDb2, { text: "the cache was migrated to disk", scope: "global", tags: [] });
    const m = getMemory(migDb2, id);
    check("original schema migrates kind/detail", m.kind === "event" && m.detail === "" && m.pinned === false);
  } finally {
    migDb2.close();
    rmSync(legacyDb2, { force: true });
  }

  // Supersession: paraphrase retires, pin carries, scope isolates, repeats confirm.
  dropKindCache(testDb);
  const s1 = upsertMemory(db, { text: "production postgres database runs version fourteen with nightly backups", scope: "global", tags: [] });
  insertVector(db, s1.id, [1, 0]);
  check("supersede setup created", s1.created === true);
  const s2 = upsertMemory(db, { text: "production postgres database runs version fourteen with weekly backups", scope: "global", tags: [] });
  insertVector(db, s2.id, [0, 1]);
  const { applySupersession } = await import("../lib/store.mjs");
  const hit = applySupersession(db, s2.id, "production postgres database runs version fourteen with weekly backups", "global");
  check("paraphrase supersedes", hit !== null && hit.supersededId === s1.id, `got=${hit ? hit.supersededId + "@" + hit.score.toFixed(3) : "none"}`);
  check("old row retired with pointer", getMemory(db, s1.id).superseded_by === s2.id);
  check("new row records supersedes", getMemory(db, s2.id).supersedes === s1.id);
  check("retired hidden from list", listMemories(db, {}).every((m) => m.id !== s1.id));
  check("retired hidden from vectors", getAllVectors(db).every((r) => r.id !== s1.id));
  dropKindCache(testDb);
  check("retired hidden from loader", (await loadKind(db, testDb)).every((m) => m.id !== s1.id));
  const s3 = upsertMemory(db, { text: "emperor penguins migrate across antarctic ice shelves yearly", scope: "global", tags: [] });
  insertVector(db, s3.id, [1, 1]);
  check("unrelated does not supersede", applySupersession(db, s3.id, "emperor penguins migrate across antarctic ice shelves yearly", "global") === null);
  // Same text in another scope refreshes the one global-keyed entry (no duplicate);
  // cross-scope paraphrases must match within their own scope only.
  const o1 = upsertMemory(db, { text: "staging postgres database runs version fourteen with nightly backups", scope: "project:other", tags: [] });
  insertVector(db, o1.id, [1, 0]);
  const o2 = upsertMemory(db, { text: "staging postgres database runs version fourteen with weekly backups", scope: "project:other", tags: [] });
  insertVector(db, o2.id, [0, 1]);
  const hitO = applySupersession(db, o2.id, "staging postgres database runs version fourteen with weekly backups", "project:other");
  check("supersession stays within scope", hitO !== null && hitO.supersededId === o1.id, `got=${hitO ? hitO.supersededId : "none"}`);
  check("global rows untouched by other-scope retire", getMemory(db, s2.id).superseded_by === null);
  setPinned(db, s2.id, true);
  const s5 = upsertMemory(db, { text: "production postgres database runs version fourteen with monthly backups", scope: "global", tags: [] });
  insertVector(db, s5.id, [1, 0]);
  const hit2 = applySupersession(db, s5.id, "production postgres database runs version fourteen with monthly backups", "global");
  check("pin carries to superseding row", hit2 !== null && getMemory(db, s5.id).pinned === true, `got=${hit2 ? hit2.supersededId : "none"}`);
  const before = getMemory(db, s5.id).confirmed | 0;
  upsertMemory(db, { text: "production postgres database runs version fourteen with monthly backups", scope: "global", tags: [] });
  check("identical repeat bumps confirmed", (getMemory(db, s5.id).confirmed | 0) === before + 1);
  for (const id of [s1.id, s2.id, s3.id, o1.id, o2.id, s5.id]) deleteMemory(db, id);
  dropKindCache(testDb);

  // WAL + single-JOIN loader: parity, scope filter, cache, invalidation.
  check("journal mode is WAL", db.prepare("PRAGMA journal_mode").get().journal_mode === "wal");
  const { loadScoredMemories, dropVectorCache } = await import("../lib/vectors.mjs");
  dropVectorCache(testDb);
  const loaded = await loadScoredMemories(db, testDb);
  check("loader returns all rows", loaded.length === countMemories(db), `got=${loaded.length}`);
  check(
    "loader vectors decoded",
    loaded.every((m) => m.vector instanceof Float32Array && m.vector.length === vecs[0].length)
  );
  const loadedScoped = await loadScoredMemories(db, testDb, "project:webapp");
  check("loader scope filter", loadedScoped.length === 1 && loadedScoped[0].id === ids[1], `got=${loadedScoped.length}`);
  const parity = getAllVectors(db).map((r) => ({ ...getMemory(db, r.id), vector: r.vector }));
  const byId = new Map(loaded.map((m) => [m.id, m]));
  check(
    "loader parity with per-row path",
    parity.every((p) => {
      const l = byId.get(p.id);
      return (
        l && l.text === p.text && l.scope === p.scope && l.pinned === p.pinned &&
        l.vector.length === p.vector.length && l.vector.every((v, i) => v === p.vector[i])
      );
    })
  );
  const again = await loadScoredMemories(db, testDb);
  check("cache hit returns same rows", again === loaded);
  // Simulate a parallel session's write through a second connection.
  const db2 = openDb(testDb);
  try {
    const { id: nid } = upsertMemory(db2, { text: "parallel session probe entry", scope: "global", tags: [] });
    insertVector(db2, nid, Array.from(vecs[0]));
  } finally {
    db2.close();
  }
  const after = await loadScoredMemories(db, testDb);
  check("cache invalidates on external write", after.length === loaded.length + 1, `got=${after.length}`);
  check("cache refreshes reference", after !== loaded);
  deleteMemory(db, memoryIdFor("parallel session probe entry"));
  dropVectorCache(testDb);
} finally {
  db.close();
  rmSync(testDb, { force: true });
}

if (failures.length > 0) {
  console.log(`\n${failures.length} failing: ${failures.join(", ")}`);
  process.exitCode = 1;
} else {
  console.log("\nall memory tests passed");
}
process.exit(process.exitCode || 0);
