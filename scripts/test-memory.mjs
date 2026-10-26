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

  // Pure checks (no model needed): dims validation + secret exclusion.
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
