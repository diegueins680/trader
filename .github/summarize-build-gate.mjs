// Summarize the API's leaderboard build-gate, sync-phase and DB-read log events.
// Input: log lines on stdin (already filtered on the server to these events).
// Output: one JSON object of counts and millisecond statistics. Only the fixed
// enum labels and numbers these events carry are read; nothing else is echoed.
import { createInterface } from "node:readline";

const nowMs = Number(process.argv[2] ?? Date.now());
const gate = new Map();
const openPhase = new Map(); // attemptId -> sync phase started but not yet finished
const phases = {};
const dbReads = {};
let first = null;
let last = null;
let parsed = 0;

const num = value => (typeof value === "number" && Number.isFinite(value) ? value : null);
const push = (bucket, key, value) => { if (value != null) (bucket[key] ??= []).push(value); };
const stats = values => {
  if (!values?.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const at = q => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return { n: sorted.length, p50: Math.round(at(0.5)), p95: Math.round(at(0.95)), max: Math.round(sorted.at(-1)) };
};

for await (const line of createInterface({ input: process.stdin })) {
  const start = line.indexOf("{");
  if (start < 0) continue;
  let event;
  try { event = JSON.parse(line.slice(start)); } catch { continue; }
  const atMs = num(event?.atMs);
  if (atMs == null || typeof event.attemptId !== "string") continue;
  parsed += 1;
  first = first == null ? atMs : Math.min(first, atMs);
  last = last == null ? atMs : Math.max(last, atMs);
  if (event.event === "top_combos_build_gate" && ["response", "sync"].includes(event.origin)) {
    const entry = gate.get(event.attemptId) ?? { origin: event.origin };
    if (event.phase === "waiting") entry.waitingAtMs = atMs;
    if (event.phase === "acquired") { entry.acquiredAtMs = atMs; entry.waitMs = num(event.waitMs); }
    if (event.phase === "finished") { entry.finishedAtMs = atMs; entry.waitMs = num(event.waitMs); entry.heldMs = num(event.heldMs); }
    gate.set(event.attemptId, entry);
  } else if (event.event === "top_combos_sync_phase" && /^[a-z0-9-]{1,20}$/.test(event.phase)) {
    if (event.state === "started") openPhase.set(event.attemptId, { phase: event.phase, atMs });
    if (event.state === "finished") { openPhase.delete(event.attemptId); push(phases, event.phase, num(event.elapsedMs)); }
  } else if (event.event === "top_combos_db_read" && ["connection-wait", "query"].includes(event.stage)) {
    push(dbReads, event.stage, num(event.elapsedMs));
  }
}

const byOrigin = {};
const open = [];
for (const [attemptId, entry] of gate) {
  const bucket = (byOrigin[entry.origin] ??= { finished: 0, waitMs: [], heldMs: [], startsAtMs: [] });
  if (entry.waitingAtMs != null) bucket.startsAtMs.push(entry.waitingAtMs);
  if (entry.finishedAtMs != null) {
    bucket.finished += 1;
    if (entry.waitMs != null) bucket.waitMs.push(entry.waitMs);
    if (entry.heldMs != null) bucket.heldMs.push(entry.heldMs);
  } else if (entry.waitingAtMs != null || entry.acquiredAtMs != null) {
    // Still queued or still holding the gate when the logs were read.
    const holding = entry.acquiredAtMs != null;
    const sinceMs = holding ? entry.acquiredAtMs : entry.waitingAtMs;
    const phase = openPhase.get(attemptId);
    open.push({
      origin: entry.origin, state: holding ? "holding" : "waiting", ageMs: Math.max(0, nowMs - sinceMs),
      ...(phase ? { inPhase: phase.phase, phaseAgeMs: Math.round(Math.max(0, nowMs - phase.atMs)) } : {}),
    });
  }
}
const gateSummary = Object.fromEntries(Object.entries(byOrigin).map(([origin, bucket]) => {
  const starts = bucket.startsAtMs.sort((a, b) => a - b);
  const gaps = starts.slice(1).map((at, i) => at - starts[i]);
  return [origin, { finished: bucket.finished, waitMs: stats(bucket.waitMs), heldMs: stats(bucket.heldMs), gapBetweenStartsMs: stats(gaps) }];
}));
const statsOf = bucket => Object.fromEntries(Object.entries(bucket).map(([key, values]) => [key, stats(values)]));

console.log(JSON.stringify({
  events: parsed,
  windowMs: first == null ? null : { first, last, spanMs: last - first, sinceLastMs: nowMs - last },
  gate: gateSummary,
  open: open.sort((a, b) => b.ageMs - a.ageMs).slice(0, 10).map(row => ({ ...row, ageMs: Math.round(row.ageMs) })),
  syncPhasesMs: statsOf(phases),
  dbReadsMs: statsOf(dbReads),
}));
