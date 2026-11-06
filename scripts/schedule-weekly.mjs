import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { join } from "node:path";
import { extRoot } from "../lib/paths.mjs";

const execFileAsync = promisify(execFile);

// Offline mining every 3h while the machine is on: mined.json state skips
// processed checkpoints before any model loads, so empty runs cost one
// sqlite read. Each pass with fresh checkpoints loads the chat model.
// Run manually with:
//   node scripts/mine.mjs --limit 25          (stage for approval)
//   node scripts/mine.mjs --import             (import staged)
//   node scripts/mine.mjs --auto-import        (store directly, >= 0.85 confidence)

async function main() {
  const script = join(extRoot, "scripts", "mine.mjs");
  if (process.platform === "win32") {
    const taskName = "MemoryMine";
    const tr = `"${process.execPath}" "${script}" --limit 25`;
    try {
      await execFileAsync("schtasks", ["/delete", "/tn", "MemoryWeeklyMine", "/f"]);
    } catch {
      // Already gone (first install or previously migrated).
    }
    await execFileAsync("schtasks", [
      "/create",
      "/tn",
      taskName,
      "/tr",
      tr,
      "/sc",
      "hourly",
      "/mo",
      "3",
      "/st",
      "00:00",
      "/f",
    ]);
    console.log(`scheduled task "${taskName}" created: every 3h -> ${tr}`);
    console.log("mined.json state guarantees no checkpoint is ever processed twice.");
    console.log("Then apply catch-up settings (needs no admin):");
    console.log(`powershell -Command "Set-ScheduledTask -TaskName '${taskName}' -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 4))"`);
  } else {
    console.log("Add this cron line (every 3 hours):");
    console.log(`0 */3 * * * "${process.execPath}" "${script}" --limit 25`);
    console.log("mined.json state guarantees no checkpoint is ever processed twice.");
  }
}

main().catch((err) => {
  console.error(`schedule failed: ${err.message}`);
  process.exitCode = 1;
});
