#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const DAY_MS = 86400000;
const MAX_OPERATIONS = 100000;
const UUID = /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i;
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const timestamp = (value) => Number.isSafeInteger(value) && value > 0 && value <= 8640000000000000;
const positive = (value) => finite(value) && value > 0;
const canonicalUuid = (value) => typeof value === "string" && UUID.test(value.trim()) ? value.trim().toLowerCase() : null;
const symbolName = (value) => typeof value === "string" && /^[A-Z0-9_-]{1,32}$/.test(value) ? value : null;
const hash = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");

export function utcDay(date) {
  if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("Invalid UTC day");
  const startAtMs = Date.parse(`${date}T00:00:00.000Z`);
  if (!timestamp(startAtMs) || new Date(startAtMs).toISOString().slice(0, 10) !== date) throw new Error("Invalid UTC day");
  return { startAtMs, endAtMs: startAtMs + DAY_MS };
}

export function parseUtcInstant(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$/.test(value)) throw new Error("Expected UTC proposal time");
  const parsed = Date.parse(value);
  if (!timestamp(parsed) || new Date(parsed).toISOString().replace(".000Z", "Z") !== value.replace(".000Z", "Z")) throw new Error("Invalid UTC proposal time");
  return parsed;
}

function reviewedFleet(selector) {
  const values = selector?.graduation?.reviewedUuids;
  if (!Array.isArray(values) || values.length < 1 || values.length > 256 || values.some((value) => canonicalUuid(value) === null)) {
    throw new Error("Missing or invalid reviewed fleet");
  }
  return [...new Set(values.map(canonicalUuid))].sort();
}

function reportConfig(selector) {
  const requirements = selector?.graduation?.requirements;
  const baselineAgeMs = requirements?.maximumBaselineAgeMs;
  const intervalMs = requirements?.statusIntervalMs;
  if (!Number.isSafeInteger(baselineAgeMs) || baselineAgeMs < 60000 || baselineAgeMs > DAY_MS
    || !Number.isSafeInteger(intervalMs) || intervalMs < 60000 || intervalMs > DAY_MS) throw new Error("Invalid review sampling configuration");
  return { baselineAgeMs, intervalMs };
}

function normalizedOperations(raw) {
  if (!Array.isArray(raw) || raw.length > MAX_OPERATIONS) throw new Error("Invalid or oversized operations export");
  const unique = new Map();
  for (const row of raw) {
    if (!row || !Number.isSafeInteger(row.id) || row.id <= 0 || !timestamp(row.atMs) || typeof row.kind !== "string") {
      throw new Error("Invalid operation identity or timestamp");
    }
    if (unique.has(row.id) && JSON.stringify(unique.get(row.id)) !== JSON.stringify(row)) throw new Error("Conflicting duplicate operation");
    unique.set(row.id, row);
  }
  return [...unique.values()].sort((a, b) => a.atMs - b.atMs || a.id - b.id);
}

function healthyStatus(row) {
  const status = row.result;
  return status?.live === true && status?.tradeEnabled === true && status?.running === true
    && status?.halted === false && status?.error == null;
}

function liveEquity(row) {
  return row?.result?.live === true && row?.result?.tradeEnabled === true && positive(row.equity);
}

function workerDay(uuid, operations, window, config, complete) {
  const rows = operations.filter((row) => canonicalUuid(row.comboUuid) === uuid);
  const statuses = rows.filter((row) => row.kind === "bot.status");
  const within = (row) => row.atMs >= window.startAtMs && row.atMs < window.endAtMs;
  const daily = statuses.filter(within);
  const baseline = statuses.filter((row) => row.atMs >= window.startAtMs - config.baselineAgeMs && row.atMs <= window.startAtMs).at(-1);
  const closing = daily.at(-1);
  const issues = [];
  if (!baseline) issues.push("boundary-baseline-missing");
  else if (!liveEquity(baseline)) issues.push("boundary-baseline-invalid");
  if (!closing || window.endAtMs - closing.atMs > config.baselineAgeMs) issues.push("closing-equity-missing-or-stale");
  else if (!liveEquity(closing)) issues.push("closing-equity-invalid");
  if (daily.some((row) => !liveEquity(row))) issues.push("daily-equity-invalid-or-not-live");
  if (!complete) issues.push("operations-coverage-incomplete");
  let equityReturn = null;
  let sampledMaxDrawdown = null;
  if (issues.length === 0) {
    equityReturn = closing.equity / baseline.equity - 1;
    let peak = baseline.equity;
    sampledMaxDrawdown = 0;
    for (const row of daily) {
      peak = Math.max(peak, row.equity);
      sampledMaxDrawdown = Math.max(sampledMaxDrawdown, 1 - row.equity / peak);
    }
    if (!finite(equityReturn) || !finite(sampledMaxDrawdown)) {
      equityReturn = null;
      sampledMaxDrawdown = null;
      issues.push("equity-arithmetic-non-finite");
    }
  }
  const buckets = new Map();
  for (const row of daily) {
    const bucket = Math.floor((row.atMs - window.startAtMs) / config.intervalMs);
    buckets.set(bucket, (buckets.get(bucket) ?? true) && healthyStatus(row));
  }
  const expectedIntervals = Math.ceil(DAY_MS / config.intervalMs);
  const healthyIntervals = [...buckets.values()].filter(Boolean).length;
  const orders = rows.filter((row) => row.kind === "bot.order" && within(row));
  let policyDenied = 0;
  let invalidRequests = 0;
  let submitted = 0;
  let notSubmitted = 0;
  let nonLive = 0;
  let precedingCloseSubmitted = 0;
  let unknown = 0;
  for (const row of orders) {
    const order = row.result?.event?.order;
    const admission = order?.marketRisk?.admissionOutcome;
    if (order?.precedingClose?.mode === "live" && order.precedingClose.sent === true) precedingCloseSubmitted += 1;
    if (["paper", "dry-run", "test"].includes(order?.mode)) nonLive += 1;
    else if (order?.mode !== "live" || (["policy_denied", "invalid_request"].includes(admission) && order.sent === true)) unknown += 1;
    else if (admission === "policy_denied") policyDenied += 1;
    else if (admission === "invalid_request") invalidRequests += 1;
    else if (order?.sent === true) submitted += 1;
    else if (order?.sent === false) notSubmitted += 1;
    else unknown += 1;
  }
  const symbols = [...new Set(rows.filter(within).map((row) => symbolName(row.symbol)).filter(Boolean))].sort();
  return {
    uuid, symbols, equityReturn, sampledMaxDrawdown,
    equityBasis: "sampled bot ledger; not venue profit or account drawdown",
    venueNetPnl: null,
    status: { expectedIntervals, observedIntervals: buckets.size, healthyIntervals,
      missingIntervals: expectedIntervals - buckets.size, healthyCoverage: complete ? healthyIntervals / expectedIntervals : null },
    orders: { observed: orders.length, submitted, notSubmitted, policyDenied, invalidRequests, nonLive, unknown, precedingCloseSubmitted,
      submissionFraction: complete && unknown === 0 && submitted + notSubmitted > 0 ? submitted / (submitted + notSubmitted) : null },
    issues: [...issues, "venue-pnl-not-attributed-to-uuid"],
  };
}

export function graduationReport(selector, capturedAtMs, proposeStartMs) {
  if (!timestamp(capturedAtMs)) throw new Error("Invalid report capture time");
  const reviewedUuids = reviewedFleet(selector);
  const fleetVersion = `fleet-v1:${hash(reviewedUuids)}`;
  const graduation = selector.graduation;
  const review = graduation.review;
  let matches = false;
  try {
    matches = JSON.stringify(reviewedFleet({ graduation: { reviewedUuids: review?.reviewedUuids } })) === JSON.stringify(reviewedUuids)
      && typeof graduation.configVersion === "string" && graduation.configVersion.length > 0 && review?.configVersion === graduation.configVersion;
  } catch { /* Missing reviews remain pending. */ }
  const result = {
    fleetVersion, reviewedUuids, configuredMode: ["shadow", "canary", "enforce"].includes(selector.configuredMode) ? selector.configuredMode : null,
    effectiveMode: ["shadow", "canary", "enforce"].includes(selector.mode) ? selector.mode : null,
    enabled: graduation.enabled === true, startedAtMs: timestamp(graduation.startedAtMs) ? graduation.startedAtMs : null,
    sourceConfigVersion: typeof graduation.configVersion === "string" && graduation.configVersion.length > 0 ? graduation.configVersion : null,
    reviewMatchesCurrentFleet: matches,
    recordedDecision: matches && ["pending", "graduated"].includes(review?.decision) ? review.decision : null,
    transition: {
      reviewedStrategies: reviewedUuids.length,
      enforcingMaxBots: Number.isSafeInteger(selector.configuration?.maxBots) && selector.configuration.maxBots >= 0 ? selector.configuration.maxBots : null,
      enforcingMaxGrossWeight: finite(selector.configuration?.maxGrossWeight) && selector.configuration.maxGrossWeight >= 0 && selector.configuration.maxGrossWeight <= 1 ? selector.configuration.maxGrossWeight : null,
    },
    windowProposal: null,
    authorizesGraduation: false,
  };
  if (proposeStartMs !== undefined) {
    const minimumDays = graduation.requirements?.minimumDailyObservations;
    const { baselineAgeMs } = reportConfig(selector);
    if (!timestamp(proposeStartMs) || proposeStartMs < capturedAtMs || !Number.isSafeInteger(minimumDays) || minimumDays < 2 || minimumDays > 365) {
      throw new Error("Review proposal must use a future start and valid evidence duration");
    }
    if (result.sourceConfigVersion === null || result.transition.enforcingMaxBots === null || result.transition.enforcingMaxGrossWeight === null) {
      throw new Error("Review proposal requires explicit source configuration and transition limits");
    }
    // Match the server's ceiling-to-UTC-day arithmetic, including exact midnight.
    const firstFullDayMs = Math.ceil(proposeStartMs / DAY_MS) * DAY_MS;
    const earliestReviewAtMs = firstFullDayMs + minimumDays * DAY_MS;
    if (!timestamp(earliestReviewAtMs)) throw new Error("Review proposal date overflow");
    result.windowProposal = {
      schemaVersion: 1, proposalId: `window-v1:${hash([fleetVersion, result.sourceConfigVersion, proposeStartMs, result.transition])}`,
      fleetVersion, sourceConfigVersion: result.sourceConfigVersion, reviewedUuids, proposedStartedAtMs: proposeStartMs,
      transition: result.transition,
      firstFullDayMs, baselineWindow: { fromMs: firstFullDayMs - baselineAgeMs, toMs: firstFullDayMs },
      earliestReviewAtMs,
      proposedEnvironment: { TRADER_PORTFOLIO_AUTO_GRADUATE_STARTED_AT_MS: String(proposeStartMs) },
      applied: false,
    };
  }
  return result;
}

function venueDay(response, window) {
  if (!response) return { available: false, issues: ["venue-ledger-not-supplied"], account: null, symbols: [] };
  const ledger = response.ledger;
  const issues = [];
  if (response.market !== "futures" || response.testnet !== false) issues.push("venue-ledger-not-live-futures");
  if (!timestamp(response.fetchedAtMs) || response.fetchedAtMs < window.endAtMs) issues.push("venue-ledger-fetched-before-day-close");
  if (ledger?.startAtMs !== window.startAtMs || ledger?.endAtMs !== window.endAtMs - 1) issues.push("venue-ledger-window-mismatch");
  if (ledger?.incomeMayBeTruncated !== false || ledger?.tradesMayBeTruncated !== false) issues.push("venue-ledger-coverage-unproven");
  if (!Array.isArray(ledger?.unclassifiedIncomeTypes) || ledger.unclassifiedIncomeTypes.length !== 0) issues.push("venue-ledger-income-classification-unproven");
  if (typeof ledger?.asset !== "string" || !/^[A-Z0-9]{1,16}$/.test(ledger.asset)) issues.push("venue-ledger-asset-invalid");
  const keys = ["realizedPnl", "funding", "commission", "rebates", "otherOperating", "exchangeNet", "excludedNonOperating", "unclassified"];
  const breakdown = (row) => {
    if (!row || keys.some((key) => !finite(row[key]))) throw new Error("Invalid venue breakdown");
    const sum = row.realizedPnl + row.funding + row.commission + row.rebates + row.otherOperating;
    if (!finite(sum) || Math.abs(sum - row.exchangeNet) > 1e-8 * Math.max(1, Math.abs(sum), Math.abs(row.exchangeNet))) throw new Error("Unreconciled venue breakdown");
    if (row.unclassified !== 0) throw new Error("Unclassified venue income");
    return Object.fromEntries(keys.map((key) => [key, row[key]]));
  };
  let account = null;
  let symbols = [];
  try {
    account = breakdown(ledger?.breakdown);
    if (!Array.isArray(ledger?.symbols) || ledger.symbols.length > 1000) throw new Error("Invalid venue symbols");
    symbols = ledger.symbols.map((row) => {
      // Binance transfers and account adjustments may have an empty symbol.
      const symbol = row.symbol === "" ? null : symbolName(row.symbol);
      if (symbol === null && row.symbol !== "") throw new Error("Invalid venue symbol");
      return { symbol, ...breakdown(row.breakdown) };
    }).sort((a, b) => (a.symbol ?? "") < (b.symbol ?? "") ? -1 : (a.symbol ?? "") > (b.symbol ?? "") ? 1 : 0);
    if (new Set(symbols.map((row) => row.symbol)).size !== symbols.length) throw new Error("Duplicate venue symbol");
    for (const key of keys) {
      const sum = symbols.reduce((total, row) => total + row[key], 0);
      if (!finite(sum) || Math.abs(sum - account[key]) > 1e-8 * Math.max(1, Math.abs(sum), Math.abs(account[key]))) throw new Error("Venue symbol totals do not reconcile");
    }
  } catch { issues.push("venue-ledger-invalid-or-unreconciled"); }
  return {
    available: issues.length === 0, asset: /^[A-Z0-9]{1,16}$/.test(ledger?.asset ?? "") ? ledger.asset : null,
    basis: "account/symbol exchange income after signed commissions and funding; excludes unrealized P&L and infrastructure",
    account: issues.length === 0 ? account : null, symbols: issues.length === 0 ? symbols : [], issues,
  };
}

export function buildFleetReport(snapshot, { date, proposeStartMs } = {}) {
  const window = utcDay(date);
  if (snapshot?.schemaVersion !== 1 || !timestamp(snapshot.capturedAtMs)) throw new Error("Invalid fleet snapshot");
  const selector = snapshot.botStatus?.portfolioSelector;
  const fleet = reviewedFleet(selector);
  const config = reportConfig(selector);
  const coverage = snapshot.coverage;
  if (coverage?.tenantScoped !== true || !timestamp(coverage.fromMs) || !timestamp(coverage.toMs)) throw new Error("Tenant-scoped operations coverage is required");
  const complete = coverage.complete === true && coverage.fromMs <= window.startAtMs - config.baselineAgeMs
    && coverage.toMs >= window.endAtMs - 1 && snapshot.capturedAtMs >= window.endAtMs;
  const operations = normalizedOperations(snapshot.operations?.ops);
  const issues = [];
  if (snapshot.operations.enabled !== true) issues.push("operations-persistence-unavailable");
  if (!complete) issues.push("operations-window-incomplete");
  if (snapshot.capturedAtMs < window.endAtMs) issues.push("utc-day-not-complete");
  const scoped = operations.filter((row) => row.atMs >= window.startAtMs - config.baselineAgeMs && row.atMs < window.endAtMs);
  const workers = fleet.map((uuid) => workerDay(uuid, scoped, window, config, complete && snapshot.operations.enabled === true));
  const known = new Set(fleet);
  const unattributedOperations = scoped.filter((row) => !known.has(canonicalUuid(row.comboUuid))).length;
  const venue = venueDay(snapshot.revenue, window);
  return {
    schemaVersion: 1, reportType: "live-fleet-day", date, timezone: "UTC", ...window, capturedAtMs: snapshot.capturedAtMs,
    complete: issues.length === 0 && workers.every((worker) => worker.equityReturn !== null) && venue.available,
    operationsCoverage: { complete: complete && snapshot.operations.enabled === true, uniqueOperations: scoped.length, unattributedOperations },
    graduation: graduationReport(selector, snapshot.capturedAtMs, proposeStartMs), workers, venue,
    issues: [...issues, ...venue.issues],
    accountingNote: "Venue income is reported at account/symbol scope. No per-UUID profit is inferred from a symbol, current ownership, or simulated equity.",
  };
}

async function getJson(url, headers, fetchImpl) {
  const response = await fetchImpl(url, { method: "GET", headers, redirect: "error", signal: AbortSignal.timeout(30000) });
  if (response.status !== 200 || !response.headers.get("content-type")?.includes("application/json")) {
    await response.body?.cancel();
    throw new Error("Fleet report endpoint unavailable");
  }
  const chunks = [];
  let bytes = 0;
  for await (const chunk of response.body ?? []) {
    bytes += chunk.length;
    if (bytes > 16 * 1024 * 1024) throw new Error("Fleet report response too large");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

export async function collectFleetSnapshot(baseUrl, date, {
  fetchImpl = fetch, token = process.env.TRADER_API_TOKEN, tenantKey = process.env.TRADER_REPORT_TENANT_KEY,
  now = Date.now, maxPages = 200, pageSize = 500,
} = {}) {
  const origin = new URL(baseUrl);
  if (!["http:", "https:"].includes(origin.protocol) || origin.username || origin.password || origin.search || origin.hash || origin.pathname !== "/") throw new Error("Expected API origin");
  if (typeof tenantKey !== "string" || !tenantKey.trim()) throw new Error("Explicit tenant context is required");
  if (!Number.isSafeInteger(maxPages) || maxPages < 1 || maxPages > 200 || !Number.isSafeInteger(pageSize) || pageSize < 1 || pageSize > 500) throw new Error("Invalid collection bounds");
  const capturedAtMs = now();
  const window = utcDay(date);
  const headers = { "X-Tenant-Key": tenantKey };
  if (token) headers.Authorization = `Bearer ${token}`;
  const botStatus = await getJson(`${origin.origin}/bot/status?tail=1`, headers, fetchImpl);
  const config = reportConfig(botStatus.portfolioSelector);
  const fromMs = window.startAtMs - config.baselineAgeMs;
  const toMs = window.endAtMs - 1;
  const rows = [];
  let since = 0;
  let ceiling;
  let complete = false;
  for (let page = 0; page < maxPages; page += 1) {
    const url = new URL("/ops", origin);
    url.search = new URLSearchParams({ bot: "true", fromMs: String(fromMs), toMs: String(toMs), since: String(since), limit: String(pageSize) }).toString();
    const body = await getJson(url.href, headers, fetchImpl);
    if (body.enabled !== true || !Number.isSafeInteger(body.latestId) || body.latestId < 0 || !Array.isArray(body.ops) || body.ops.length > pageSize) throw new Error("Invalid operations page");
    ceiling ??= body.latestId;
    let previous = since;
    for (const row of body.ops) {
      if (!Number.isSafeInteger(row.id) || row.id <= previous || !timestamp(row.atMs) || row.atMs < fromMs || row.atMs > toMs) throw new Error("Operations pagination did not advance within the requested window");
      previous = row.id;
      if (row.id <= ceiling) rows.push(row);
    }
    if (body.ops.length < pageSize || previous >= ceiling) { complete = true; break; }
    since = previous;
  }
  return { schemaVersion: 1, capturedAtMs, botStatus, operations: { enabled: true, ops: rows }, coverage: { fromMs, toMs, complete, tenantScoped: true } };
}

export function renderMarkdown(report) {
  const pct = (value) => value === null ? "unavailable" : `${(value * 100).toFixed(2)}%`;
  const lines = [
    `# Fleet daily report — ${report.date} UTC`, "",
    `Evidence: ${report.complete ? "complete within the stated scopes" : "incomplete"}. Fleet version: ${report.graduation.fleetVersion}.`, "",
    "| Strategy UUID | Symbols | Bot ledger return | Sampled bot drawdown | Healthy status coverage | Submitted / not submitted | Policy denials |", "|---|---|---:|---:|---:|---:|---:|",
    ...report.workers.map((worker) => `| ${worker.uuid} | ${worker.symbols.join(", ") || "unknown"} | ${pct(worker.equityReturn)} | ${pct(worker.sampledMaxDrawdown)} | ${pct(worker.status.healthyCoverage)} | ${worker.orders.submitted} / ${worker.orders.notSubmitted} | ${worker.orders.policyDenied} |`), "",
    "Bot ledger returns and sampled drawdowns are model-state observations. Missing status intervals count against coverage. Submission fractions count primary live order flags, exclude explicit policy denials, invalid requests and non-live events, and do not measure fills. Unknown outcomes withhold the fraction.", "",
    report.venue.available ? `Venue account income: ${report.venue.account.exchangeNet} ${report.venue.asset}, including signed commissions ${report.venue.account.commission} and funding ${report.venue.account.funding}.` : "Venue income unavailable: a complete, reconciled live futures revenue export for this exact UTC day is required.", "",
    report.accountingNote, "",
    `Graduation reviews ${report.graduation.transition.reviewedStrategies} strategies; enforcing selection is capped at ${report.graduation.transition.enforcingMaxBots ?? "unknown"} bots and ${pct(report.graduation.transition.enforcingMaxGrossWeight)} gross weight.`,
    `Saved review matches current fleet/configuration: ${report.graduation.reviewMatchesCurrentFleet ? "yes" : "no"}. This report authorizes no graduation.`,
  ];
  if (report.venue.available) lines.push("", `Venue income by symbol (${report.venue.asset}):`, "",
    "| Symbol | Realized P&L | Funding | Signed commissions | Exchange net |", "|---|---:|---:|---:|---:|",
    ...report.venue.symbols.map((row) => `| ${row.symbol ?? "Account (unassigned)"} | ${row.realizedPnl} | ${row.funding} | ${row.commission} | ${row.exchangeNet} |`), "",
    `Excluded non-operating income: ${report.venue.account.excludedNonOperating} ${report.venue.asset}.`);
  if (report.issues.length > 0) lines.push("", `Evidence gaps: ${report.issues.join(", ")}.`);
  lines.push("", "Strategy evidence gaps:", "");
  for (const worker of report.workers) lines.push(`- ${worker.uuid}: ${worker.issues.join(", ")}`);
  if (report.graduation.windowProposal) lines.push("", "Proposed review window (not applied):", "", "```json", JSON.stringify(report.graduation.windowProposal, null, 2), "```");
  return `${lines.join("\n")}\n`;
}

async function readJsonFile(path) {
  if ((await stat(path)).size > 128 * 1024 * 1024) throw new Error("Fleet input too large");
  return JSON.parse(await readFile(path, "utf8"));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const args = process.argv.slice(2);
    if (args.includes("--help")) {
      console.log("Usage: node scripts/report-live-fleet.mjs (--input SNAPSHOT.json | --base-url ORIGIN) --date YYYY-MM-DD [--revenue REVENUE.json] [--propose-start ISO_UTC] [--format json|markdown]\nCollection sends only GET requests; credentials come from TRADER_API_TOKEN and TRADER_REPORT_TENANT_KEY. Outputs a report or an unapplied window proposal; exit 1 indicates incomplete evidence.");
    } else {
      const options = {};
      const allowed = new Set(["--input", "--base-url", "--date", "--revenue", "--propose-start", "--format"]);
      for (let index = 0; index < args.length; index += 2) {
        if (!allowed.has(args[index]) || !args[index + 1] || Object.hasOwn(options, args[index])) throw new Error("Invalid CLI option");
        options[args[index]] = args[index + 1];
      }
      if (Boolean(options["--input"]) === Boolean(options["--base-url"])) throw new Error("Choose one source");
      utcDay(options["--date"]);
      const format = options["--format"] ?? "json";
      if (!["json", "markdown"].includes(format)) throw new Error("Invalid output format");
      const snapshot = options["--input"] ? await readJsonFile(options["--input"]) : await collectFleetSnapshot(options["--base-url"], options["--date"]);
      if (options["--revenue"]) {
        const revenue = await readJsonFile(options["--revenue"]);
        snapshot.revenue = revenue?.reportType === "strategy-assurance-revenue" && revenue.schemaVersion === 1 ? revenue.response : revenue;
      }
      const proposal = options["--propose-start"];
      const report = buildFleetReport(snapshot, { date: options["--date"], proposeStartMs: proposal ? parseUtcInstant(proposal) : undefined });
      console.log(format === "markdown" ? renderMarkdown(report) : JSON.stringify(report, null, 2));
      process.exitCode = report.complete ? 0 : 1;
    }
  } catch {
    console.error("Fleet report failed. Check the input schema, UTC day, tenant scope, and endpoint access; use --help for usage.");
    process.exitCode = 2;
  }
}
