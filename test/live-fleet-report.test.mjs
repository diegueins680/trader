import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildFleetReport, collectFleetSnapshot, graduationReport, parseUtcInstant, renderMarkdown, utcDay } from "../scripts/report-live-fleet.mjs";

const A = "11111111-1111-1111-1111-111111111111";
const B = "22222222-2222-2222-2222-222222222222";
const date = "2026-10-20";
const { startAtMs: start, endAtMs: end } = utcDay(date);
const interval = 900000;
const flags = { live: true, tradeEnabled: true, running: true, halted: false, error: null };

function selector() {
  return {
    mode: "shadow", configuredMode: "shadow", configuration: { maxBots: 5, maxGrossWeight: 0.05 },
    graduation: { enabled: true, startedAtMs: start - 20 * 86400000, reviewedUuids: [A, B], configVersion: "current-config", review: null,
      requirements: { maximumBaselineAgeMs: interval, statusIntervalMs: interval, minimumDailyObservations: 10 } },
  };
}

function fixture() {
  let id = 0;
  const ops = [];
  for (const [comboUuid, symbol] of [[A, "BTCUSDT"], [B, "ETHUSDT"]]) {
    ops.push({ id: ++id, comboUuid, symbol, kind: "bot.status", atMs: start - 1000, equity: 2, result: { ...flags } });
    for (let bucket = 0; bucket < 96; bucket += 1) {
      ops.push({ id: ++id, comboUuid, symbol, kind: "bot.status", atMs: start + bucket * interval + 1000, equity: bucket === 1 ? 2.4 : 2.2, result: { ...flags } });
    }
  }
  const snapshot = {
    schemaVersion: 1, capturedAtMs: end + 1000, botStatus: { portfolioSelector: selector() },
    operations: { enabled: true, ops }, coverage: { complete: true, tenantScoped: true, fromMs: start - interval, toMs: end - 1 },
  };
  return snapshot;
}

function order(snapshot, values) {
  snapshot.operations.ops.push({ id: Math.max(...snapshot.operations.ops.map((row) => row.id)) + 1, atMs: start + 5000,
    comboUuid: A, symbol: "BTCUSDT", kind: "bot.order", result: { event: { order: { mode: "live", ...values } } } });
}

function revenue() {
  const breakdown = { realizedPnl: 10, funding: -1, commission: -2, rebates: 0.5, otherOperating: 0,
    exchangeNet: 7.5, excludedNonOperating: 500, unclassified: 0 };
  return { market: "futures", testnet: false, fetchedAtMs: end + 1000,
    ledger: { asset: "USDT", startAtMs: start, endAtMs: end - 1, incomeMayBeTruncated: false, tradesMayBeTruncated: false,
      unclassifiedIncomeTypes: [],
      breakdown, symbols: [{ symbol: "BTCUSDT", breakdown: { ...breakdown } }] } };
}

test("UTC fleet report rebases each UUID, measures sampled drawdown and charges missing/unhealthy intervals", () => {
  const snapshot = fixture();
  snapshot.operations.ops = snapshot.operations.ops.filter((row) => !(row.comboUuid === B && row.atMs === start + 2 * interval + 1000));
  const first = snapshot.operations.ops.find((row) => row.comboUuid === A && row.atMs === start + 1000);
  snapshot.operations.ops.push({ ...first, id: 9999, result: { ...flags, halted: true } });
  const report = buildFleetReport(snapshot, { date });
  assert.ok(Math.abs(report.workers[0].equityReturn - 0.1) < 1e-12);
  assert.ok(Math.abs(report.workers[0].sampledMaxDrawdown - (1 - 2.2 / 2.4)) < 1e-12);
  assert.equal(report.workers[0].status.healthyIntervals, 95);
  assert.equal(report.workers[1].status.missingIntervals, 1);
  assert.equal(report.workers[1].status.healthyCoverage, 95 / 96);
  assert.equal(report.venue.available, false);
  assert.equal(report.workers[0].venueNetPnl, null);
  assert.equal(report.complete, false);
});

test("fleet report distinguishes primary live submissions, policy denials, invalid requests, non-live events and reversal closes", () => {
  const snapshot = fixture();
  order(snapshot, { sent: true });
  order(snapshot, { sent: false });
  order(snapshot, { sent: false, marketRisk: { admissionOutcome: "policy_denied" }, precedingClose: { mode: "live", sent: true } });
  order(snapshot, { sent: false, marketRisk: { admissionOutcome: "invalid_request" } });
  order(snapshot, { mode: "paper", sent: true });
  const report = buildFleetReport(snapshot, { date });
  assert.deepEqual(report.workers[0].orders, { observed: 5, submitted: 1, notSubmitted: 1, policyDenied: 1, invalidRequests: 1,
    nonLive: 1, unknown: 0, precedingCloseSubmitted: 1, submissionFraction: 0.5 });
  order(snapshot, { sent: "true" });
  assert.equal(buildFleetReport(snapshot, { date }).workers[0].orders.submissionFraction, null);
});

test("missing, stale or invalid baselines and incomplete exports withhold return rather than inventing zero", () => {
  for (const mutate of [
    (s) => { s.operations.ops = s.operations.ops.filter((row) => row.atMs >= start); },
    (s) => { s.operations.ops[0].atMs = start - interval - 1; },
    (s) => { s.operations.ops[0].equity = NaN; },
    (s) => { s.operations.ops.find((row) => row.comboUuid === A && row.atMs > start).equity = -1; },
    (s) => { s.coverage.complete = false; },
    (s) => { s.capturedAtMs = start + 1000; },
  ]) {
    const snapshot = fixture();
    mutate(snapshot);
    const report = buildFleetReport(snapshot, { date });
    assert.equal(report.workers[0].equityReturn, null);
    assert.equal(report.workers[0].sampledMaxDrawdown, null);
    assert.equal(report.complete, false);
  }
  const snapshot = fixture();
  snapshot.coverage.tenantScoped = false;
  assert.throws(() => buildFleetReport(snapshot, { date }), /Tenant-scoped/);
});

test("reports deduplicate exact operation IDs, reject conflicts and contain no raw credentials or errors", () => {
  const snapshot = fixture();
  snapshot.operations.ops.push(structuredClone(snapshot.operations.ops[0]));
  snapshot.operations.ops[1].params = { apiKey: "credential-canary" };
  snapshot.operations.ops[1].result.error = "credential-canary";
  const report = buildFleetReport(snapshot, { date });
  assert.equal(report.operationsCoverage.uniqueOperations, 194);
  assert.doesNotMatch(JSON.stringify(report) + renderMarkdown(report), /credential-canary|apiKey/);
  snapshot.operations.ops.push({ ...snapshot.operations.ops[0], equity: 20 });
  assert.throws(() => buildFleetReport(snapshot, { date }), /Conflicting/);
});

test("venue accounting uses exact-day complete live income, includes fees/funding, and never attributes a symbol to a UUID", () => {
  const snapshot = fixture();
  snapshot.revenue = revenue();
  const report = buildFleetReport(snapshot, { date });
  assert.equal(report.complete, true);
  assert.equal(report.venue.account.exchangeNet, 7.5);
  assert.equal(report.venue.account.excludedNonOperating, 500);
  assert.equal(report.venue.symbols[0].commission, -2);
  assert.equal(report.workers[0].venueNetPnl, null);
  for (const mutate of [
    (r) => { r.testnet = true; }, (r) => { r.ledger.incomeMayBeTruncated = true; },
    (r) => { r.ledger.endAtMs = end; }, (r) => { r.fetchedAtMs = start; },
    (r) => { r.ledger.breakdown.exchangeNet = 8; }, (r) => { r.ledger.breakdown.unclassified = 1; },
    (r) => { r.ledger.breakdown.commission = NaN; },
    (r) => { r.ledger.unclassifiedIncomeTypes = ["UNKNOWN"]; },
    (r) => { r.ledger.symbols[0].breakdown.funding = -2; r.ledger.symbols[0].breakdown.exchangeNet = 6.5; },
  ]) {
    const input = fixture(); input.revenue = revenue(); mutate(input.revenue);
    assert.equal(buildFleetReport(input, { date }).venue.account, null);
  }
  snapshot.revenue.ledger.symbols[0].breakdown.excludedNonOperating = 0;
  snapshot.revenue.ledger.symbols.push({ symbol: "", breakdown: { realizedPnl: 0, funding: 0, commission: 0, rebates: 0, otherOperating: 0,
    exchangeNet: 0, excludedNonOperating: 500, unclassified: 0 } });
  assert.equal(buildFleetReport(snapshot, { date }).venue.available, true);
  assert.match(renderMarkdown(buildFleetReport(snapshot, { date })), /Account \(unassigned\)/);
});

test("window proposals bind a stable fleet identity, retain the five-bot transition cap, and do not apply changes", () => {
  const config = selector();
  config.graduation.review = { reviewedUuids: [A], configVersion: "current-config", decision: "graduated" };
  const proposal = graduationReport(config, end, end + 1000);
  assert.equal(proposal.reviewMatchesCurrentFleet, false);
  assert.equal(proposal.recordedDecision, null);
  assert.equal(proposal.windowProposal.firstFullDayMs, end + 86400000);
  assert.equal(proposal.windowProposal.earliestReviewAtMs, end + 11 * 86400000);
  assert.equal(proposal.transition.enforcingMaxBots, 5);
  assert.equal(proposal.authorizesGraduation, false);
  assert.equal(proposal.windowProposal.applied, false);
  const changed = structuredClone(config); changed.graduation.configVersion = "another-config";
  assert.notEqual(graduationReport(changed, end, end + 1000).windowProposal.proposalId, proposal.windowProposal.proposalId);
  config.graduation.reviewedUuids.reverse();
  assert.equal(graduationReport(config, end).fleetVersion, proposal.fleetVersion);
  config.graduation.reviewedUuids = [A];
  assert.notEqual(graduationReport(config, end).fleetVersion, proposal.fleetVersion);
  assert.throws(() => graduationReport(config, end, end - 1), /future start/);
  assert.equal(graduationReport(config, end, end).windowProposal.firstFullDayMs, end);
  assert.throws(() => parseUtcInstant("2026-02-30T00:00:00Z"));
  assert.throws(() => parseUtcInstant("2026-10-22T24:00:00Z"));
  assert.equal(parseUtcInstant("2026-10-21T00:00:00.000Z"), end);
});

test("bounded GET collection paginates from since=0, freezes the ID ceiling, and withholds completeness at the page limit", async () => {
  const one = { id: 1, atMs: start, kind: "bot.status" };
  const two = { id: 2, atMs: start + 1, kind: "bot.status" };
  const seen = [];
  const fetchImpl = async (url, options) => {
    assert.equal(options.method, "GET"); assert.equal(options.redirect, "error");
    assert.equal(options.headers["X-Tenant-Key"], "private-tenant");
    if (url.includes("/bot/status")) return Response.json({ portfolioSelector: selector() });
    const query = new URL(url).searchParams;
    seen.push(query.get("since"));
    assert.equal(query.get("fromMs"), String(start - interval));
    return Response.json({ enabled: true, latestId: seen.length === 1 ? 2 : 999,
      ops: query.get("since") === "0" ? [one] : [two] });
  };
  const config = { fetchImpl, tenantKey: "private-tenant", token: "secret", now: () => end + 1, pageSize: 1 };
  const snapshot = await collectFleetSnapshot("https://example.com", date, config);
  assert.deepEqual(seen, ["0", "1"]);
  assert.equal(snapshot.coverage.complete, true);
  assert.equal(snapshot.operations.ops.length, 2);
  seen.length = 0;
  const truncated = await collectFleetSnapshot("https://example.com", date, { ...config, maxPages: 1 });
  assert.equal(truncated.coverage.complete, false);
  assert.doesNotMatch(JSON.stringify(snapshot), /private-tenant|secret/);
  await assert.rejects(() => collectFleetSnapshot("https://secret@example.com", date, config));
  assert.throws(() => utcDay("2026-02-30"));
});

test("fleet report CLI handles native and UI revenue exports, distinguishes incomplete evidence, and redacts parse failures", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fleet-report-"));
  const input = join(directory, "snapshot.json");
  const income = join(directory, "income.json");
  const command = fileURLToPath(new URL("../scripts/report-live-fleet.mjs", import.meta.url));
  const run = (extra = []) => spawnSync(process.execPath, [command, "--input", input, "--date", date, ...extra], { encoding: "utf8", timeout: 10000 });
  try {
    await writeFile(input, JSON.stringify(fixture()));
    let result = run(["--format", "markdown"]);
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stdout, /Evidence: incomplete/);
    await writeFile(income, JSON.stringify({ schemaVersion: 1, reportType: "strategy-assurance-revenue", response: revenue() }));
    result = run(["--revenue", income]);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).venue.account.exchangeNet, 7.5);
    await writeFile(input, '{"apiKey":"credential-canary" invalid JSON');
    result = run();
    assert.equal(result.status, 2);
    assert.doesNotMatch(result.stderr + result.stdout, /credential-canary|apiKey/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
