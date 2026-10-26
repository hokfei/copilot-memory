import { homedir } from "node:os";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";

export const extRoot = dirname(dirname(fileURLToPath(import.meta.url)));
export const copilotDir = join(homedir(), ".copilot");
export const sessionStateDir = join(copilotDir, "session-state");
export const sessionStoreDb = join(copilotDir, "session-store.db");
export const dataDb = join(copilotDir, "data.db");

// Shared embedding model — same file as session-search, never copied.
export const embedModelPath =
  process.env.SESSION_MEMORY_EMBED_MODEL ||
  join(
    copilotDir,
    "extensions",
    "session-search",
    "models",
    "Qwen3-Embedding-0.6B-Q8_0.gguf"
  );
export const embedModelName = basename(embedModelPath);

// Generative model for consolidate/bootstrap (lazy-loaded, on-demand only).
export const chatModelPath =
  process.env.SESSION_MEMORY_CHAT_MODEL ||
  join(homedir(), "models", "Qwen3.5-4B-Q4_K_M.gguf");

export const memoryDbPath = join(extRoot, "memory.db");
export const hookAuditPath = join(extRoot, "hook-audit.log");
export const stagingPath = join(extRoot, "staging.json");
