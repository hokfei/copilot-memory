// Structural quality checks for stored facts. Dependency-free so both the
// hot remember path and the offline miner share the exact same rules.
//
// A stored fact is injected into future prompts as background truth; a wrong
// or leaked one is worse than a missing one. Manual remember applies only
// the secret guard plus the headline cap — never block explicit user intent
// on style. The full gate (transient/narrative/dangling) is for mined facts.

export const HEADLINE_CAP = 300;

// A past change rather than a standing truth. Structural ON PURPOSE: an
// override of anything the caller or model self-reports (self-reported
// labels proved unreliable), so there is deliberately no `kind` parameter
// anywhere — classification always runs on the text itself.
const EVENT_RE =
  /\b(has|have|had) been \w+|\b(was|were) (added|changed|updated|validated|removed|fixed|implemented|introduced|created|renamed|migrated|replaced|moved|corrected)\b|\bnow (returns|uses|includes|supports|points|reads|writes|sends|contains)\b|\b(has|have) (changed|moved|switched)\b/i;

export function classifyKind(text) {
  return EVENT_RE.test(text || "") ? "event" : "state";
}

// Secret-looking text must never be stored: memories are long-lived and
// broadly recalled. Covers named secrets, connection strings carrying
// credentials, and PEM private-key blocks.
const SECRET_RES = [
  /\b(password|passwd|secret|api[_ -]?key|token|bearer|private[_ -]?key|credential)\b/i,
  /\b[A-Za-z][A-Za-z0-9+.-]*:\/\/[^/\s]*:[^/\s]*@/,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
];

export function looksSecret(text) {
  const s = text || "";
  return SECRET_RES.some((re) => re.test(s));
}

export function headlineTooLong(text) {
  return (text || "").trim().length > HEADLINE_CAP;
}
