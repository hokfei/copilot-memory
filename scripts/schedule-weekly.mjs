import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { join } from "node:path";
import { extRoot } from "../lib/paths.mjs";

const execFileAsync = promisify(execFile);

// Weekly offline mining is enough: checkpoints accumulate slowly, and each
// pass loads the chat model (minutes on CPU fallback). Run manually with:
//   node scripts/mine.mjs --limit 25          (stage for approval)
//   node scripts/mine.mjs --import             (import staged)
//   node scripts/mine.mjs --auto-import        (store directly, >= 0.85 confidence)

async function main() {
  const script = join(extRoot, "scripts", "mine.mjs");
  if (process.platform === "win32") {
    const taskName = "MemoryWeeklyMine";
    const tr = `"${process.execPath}" "${script}" --limit 25`;
    await execFileAsync("schtasks", [
      "/create",
      "/tn",
      taskName,
      "/tr",
      tr,
      "/sc",
      "weekly",
      "/d",
      "SUN",
      "/st",
      "03:00",
      "/f",
    ]);
    console.log(`scheduled task "${taskName}" created: weekly SUN 03:00 -> ${tr}`);
    console.log("mined.json state guarantees no checkpoint is ever processed twice.");
  } else {
    console.log("Add this cron line (weekly Sunday 03:00):");
    console.log(`0 3 * * 0 "${process.execPath}" "${script}" --limit 25`);
    console.log("mined.json state guarantees no checkpoint is ever processed twice.");
  }
}

main().catch((err) => {
  console.error(`schedule failed: ${err.message}`);
  process.exitCode = 1;
});
