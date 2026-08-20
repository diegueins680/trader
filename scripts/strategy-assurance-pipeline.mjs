#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

export const PIPELINE_STATUSES = [
  "proposal",
  "accepted",
  "paid",
  "in-delivery",
  "delivered",
  "monitoring",
  "lost",
  "cancelled",
  "refunded",
  "closed",
];

const STATUS_SET = new Set(PIPELINE_STATUSES);
const TRANSITIONS = {
  proposal: new Set(["accepted", "lost"]),
  accepted: new Set(["paid", "cancelled"]),
  paid: new Set(["in-delivery", "refunded"]),
  "in-delivery": new Set(["delivered", "refunded"]),
  delivered: new Set(["monitoring", "closed"]),
  monitoring: new Set(["closed"]),
  lost: new Set(),
  cancelled: new Set(),
  refunded: new Set(),
  closed: new Set(),
};

const DEFAULT_PIPELINE = path.join(".tmp", "strategy-assurance", "pipeline.json");

function usage() {
  return `Track Strategy Assurance proposals and recurring revenue locally.

Usage:
  npm run assurance:pipeline -- import --engagement PATH [--pipeline PATH] [--at YYYY-MM-DD]
  npm run assurance:pipeline -- advance --id ID --status STATUS [event evidence] [--at YYYY-MM-DD] [--pipeline PATH]
  npm run assurance:pipeline -- summary [--as-of YYYY-MM-DD] [--pipeline PATH] [--json]

Commands:
  import     Add a generated engagement.json. Re-importing the identical record is idempotent.
  advance    Move one engagement through an allowed forward lifecycle transition.
  summary    Report stage counts, exact funnel conversions, revenue, expirations, and next actions.

Statuses and allowed transitions:
  proposal -> accepted | lost
  accepted -> paid | cancelled
  paid -> in-delivery | refunded
  in-delivery -> delivered | refunded
  delivered -> monitoring | closed
  monitoring -> closed

Options:
  --pipeline PATH        Default: ${DEFAULT_PIPELINE}
  --engagement PATH      Generated engagement.json to import
  --id ID                Stable engagementId to advance
  --status STATUS        Destination lifecycle status
  --amount NUMBER        Required for paid/refunded; actual USD cash amount
  --hours NUMBER         Required for delivered; actual delivery hours
  --at YYYY-MM-DD        Event/import date; default is current UTC date
  --as-of YYYY-MM-DD     Summary date; default is current UTC date
  --json                 Render summary as JSON instead of Markdown
  --help
`;
}

function utcDate(timestampMs) {
  return new Date(timestampMs).toISOString().slice(0, 10);
}

function isoDate(value, flag) {
  const text = String(value ?? "").trim();
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
  if (!match) throw new Error(`--${flag} must use YYYY-MM-DD`);
  const timestamp = Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  if (new Date(timestamp).toISOString().slice(0, 10) !== text) {
    throw new Error(`--${flag} must be a real calendar date`);
  }
  return text;
}

function singleLine(value, label, maximum = 300) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a non-empty string`);
  const text = value.trim();
  if (text.length > maximum) throw new Error(`${label} must be at most ${maximum} characters`);
  if (/[\u0000-\u001f\u007f]/.test(text)) throw new Error(`${label} cannot contain control characters or newlines`);
  return text;
}

function finitePositive(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1_000_000) {
    throw new Error(`${label} must be a positive number no greater than 1000000`);
  }
  return value;
}

function finiteCurrency(value, label) {
  const parsed = finitePositive(value, label);
  const rounded = Math.round(parsed * 100) / 100;
  if (Math.abs(parsed - rounded) > 1e-9) throw new Error(`${label} must use at most two decimal places`);
  return rounded;
}

function positiveOption(value, flag, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > maximum) {
    throw new Error(`--${flag} must be a positive number no greater than ${maximum}`);
  }
  return parsed;
}

function currencyOption(value, flag) {
  const parsed = positiveOption(value, flag, 1_000_000);
  const rounded = Math.round(parsed * 100) / 100;
  if (Math.abs(parsed - rounded) > 1e-9) throw new Error(`--${flag} must use at most two decimal places`);
  return rounded;
}

function objectValue(value, label) {
  if (value == null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value;
}

function parseOptions(tokens) {
  const options = {};
  const seen = new Set();
  let json = false;
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token === "--help") return { help: true };
    if (token === "--json") {
      if (seen.has("json")) throw new Error("Duplicate option: --json");
      seen.add("json");
      json = true;
      continue;
    }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const name = token.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (!new Set(["pipeline", "engagement", "id", "status", "amount", "hours", "at", "asOf"]).has(name)) {
      throw new Error(`Unknown option: ${token}`);
    }
    if (seen.has(name)) throw new Error(`Duplicate option: ${token}`);
    const value = tokens[index + 1];
    if (value == null || value.startsWith("--")) throw new Error(`${token} requires a value`);
    options[name] = value;
    seen.add(name);
    index += 1;
  }
  return { ...options, json, help: false };
}

export function parseArgs(argv, nowMs = Date.now()) {
  if (argv.length === 0 || argv[0] === "--help") return { help: true };
  const command = argv[0];
  if (!new Set(["import", "advance", "summary"]).has(command)) throw new Error(`Unknown command: ${command}`);
  const options = parseOptions(argv.slice(1));
  if (options.help) return { help: true };
  const pipelinePath = path.resolve(singleLine(options.pipeline || DEFAULT_PIPELINE, "--pipeline", 1000));
  const today = utcDate(nowMs);

  if (command === "import") {
    if (options.json || options.id || options.status || options.amount || options.hours || options.asOf) {
      throw new Error("import received an option for another command");
    }
    if (!options.engagement) throw new Error("--engagement is required for import");
    return {
      help: false,
      command,
      pipelinePath,
      engagementPath: path.resolve(singleLine(options.engagement, "--engagement", 1000)),
      at: isoDate(options.at || today, "at"),
    };
  }
  if (command === "advance") {
    if (options.json || options.engagement || options.asOf) throw new Error("advance received an option for another command");
    const id = singleLine(options.id, "--id", 240);
    const status = singleLine(options.status, "--status", 40);
    if (!STATUS_SET.has(status)) throw new Error(`--status must be one of: ${PIPELINE_STATUSES.join(", ")}`);
    const hasAmount = options.amount != null;
    const hasHours = options.hours != null;
    if (status === "paid" || status === "refunded") {
      if (!hasAmount) throw new Error(`--amount is required when advancing to ${status}`);
      if (hasHours) throw new Error(`--hours is not valid when advancing to ${status}`);
    } else if (status === "delivered") {
      if (!hasHours) throw new Error("--hours is required when advancing to delivered");
      if (hasAmount) throw new Error("--amount is not valid when advancing to delivered");
    } else if (hasAmount || hasHours) {
      throw new Error(`--amount and --hours are not valid when advancing to ${status}`);
    }
    return {
      help: false,
      command,
      pipelinePath,
      id,
      status,
      at: isoDate(options.at || today, "at"),
      details: {
        ...(hasAmount ? { amount: currencyOption(options.amount, "amount") } : {}),
        ...(hasHours ? { hours: positiveOption(options.hours, "hours", 10_000) } : {}),
      },
    };
  }
  if (options.engagement || options.id || options.status || options.amount || options.hours || options.at) {
    throw new Error("summary received an option for another command");
  }
  return {
    help: false,
    command,
    pipelinePath,
    asOf: isoDate(options.asOf || today, "as-of"),
    json: options.json,
  };
}

function validateGeneratedEngagement(value) {
  const engagement = objectValue(value, "engagement");
  if (engagement.schemaVersion !== 1) throw new Error("engagement schemaVersion must be 1");
  const engagementId = singleLine(engagement.engagementId, "engagementId", 240);
  if (!/^[a-z0-9][a-z0-9-]+$/.test(engagementId)) throw new Error("engagementId must be lowercase letters, digits, and hyphens");
  if (engagement.engagementType !== "strategy-assurance-standard") {
    throw new Error("engagementType must be strategy-assurance-standard");
  }
  if (engagement.status !== "proposal") throw new Error("a generated engagement must enter the pipeline as proposal");
  const scope = objectValue(engagement.scope, "engagement.scope");
  const commercials = objectValue(engagement.commercials, "engagement.commercials");
  if (commercials.currency !== "USD") throw new Error("commercial currency must be USD");
  const proposalDate = isoDate(commercials.proposalDate, "engagement commercials proposalDate");
  const validThrough = isoDate(commercials.validThrough, "engagement commercials validThrough");
  if (validThrough < proposalDate) throw new Error("engagement validThrough must not precede proposalDate");
  return {
    engagementId,
    client: singleLine(engagement.client, "engagement.client"),
    provider: singleLine(engagement.provider, "engagement.provider"),
    decisionOwner: singleLine(engagement.decisionOwner, "engagement.decisionOwner"),
    strategy: singleLine(scope.strategy, "engagement.scope.strategy"),
    deployment: singleLine(scope.deployment, "engagement.scope.deployment"),
    standardReviewPrice: finiteCurrency(commercials.standardReviewPrice, "standardReviewPrice"),
    monitoringMonthlyPrice: finiteCurrency(commercials.monitoringMonthlyPrice, "monitoringMonthlyPrice"),
    proposalDate,
    validThrough,
  };
}

export function emptyPipeline() {
  return { schemaVersion: 1, updatedAt: null, engagements: [] };
}

function eventStatuses(record) {
  return new Set(record.events.map((event) => event.status));
}

function eventEvidence(status, value, label) {
  const hasAmount = Object.prototype.hasOwnProperty.call(value, "amount");
  const hasHours = Object.prototype.hasOwnProperty.call(value, "hours");
  if (status === "paid" || status === "refunded") {
    if (!hasAmount) throw new Error(`${label} requires an amount for ${status}`);
    if (hasHours) throw new Error(`${label} cannot include hours for ${status}`);
    return { amount: finiteCurrency(value.amount, `${label}.amount`) };
  }
  if (status === "delivered") {
    if (!hasHours) throw new Error(`${label} requires hours for delivered`);
    if (hasAmount) throw new Error(`${label} cannot include an amount for delivered`);
    const hours = finitePositive(value.hours, `${label}.hours`);
    if (hours > 10_000) throw new Error(`${label}.hours must not exceed 10000`);
    return { hours };
  }
  if (hasAmount || hasHours) throw new Error(`${label} cannot include amount or hours for ${status}`);
  return {};
}

export function validatePipeline(value) {
  const pipeline = objectValue(value, "pipeline");
  if (pipeline.schemaVersion !== 1) throw new Error("pipeline schemaVersion must be 1");
  if (pipeline.updatedAt !== null) isoDate(pipeline.updatedAt, "pipeline updatedAt");
  if (!Array.isArray(pipeline.engagements)) throw new Error("pipeline.engagements must be an array");
  const ids = new Set();
  for (const recordValue of pipeline.engagements) {
    const record = objectValue(recordValue, "pipeline engagement");
    const id = singleLine(record.id, "pipeline engagement id", 240);
    if (ids.has(id)) throw new Error(`duplicate pipeline engagement id: ${id}`);
    ids.add(id);
    singleLine(record.client, `${id}.client`);
    singleLine(record.provider, `${id}.provider`);
    singleLine(record.decisionOwner, `${id}.decisionOwner`);
    singleLine(record.strategy, `${id}.strategy`);
    singleLine(record.deployment, `${id}.deployment`);
    finiteCurrency(record.standardReviewPrice, `${id}.standardReviewPrice`);
    finiteCurrency(record.monitoringMonthlyPrice, `${id}.monitoringMonthlyPrice`);
    isoDate(record.proposalDate, `${id}.proposalDate`);
    const validThrough = isoDate(record.validThrough, `${id}.validThrough`);
    if (validThrough < record.proposalDate) throw new Error(`${id}.validThrough precedes proposalDate`);
    isoDate(record.importedAt, `${id}.importedAt`);
    if (pipeline.updatedAt == null || pipeline.updatedAt < record.importedAt) {
      throw new Error(`pipeline.updatedAt must include the import date for ${id}`);
    }
    if (!/^[a-f0-9]{64}$/.test(record.sourceDigest ?? "")) throw new Error(`${id}.sourceDigest must be SHA-256 hex`);
    if (!Array.isArray(record.events) || record.events.length === 0) throw new Error(`${id}.events must be non-empty`);
    let previousStatus = null;
    let previousAt = "";
    let netRecordedCashCents = 0;
    for (const eventValue of record.events) {
      const event = objectValue(eventValue, `${id} event`);
      if (!STATUS_SET.has(event.status)) throw new Error(`${id} has invalid event status: ${event.status}`);
      const at = isoDate(event.at, `${id} event date`);
      const evidence = eventEvidence(event.status, event, `${id} ${event.status} event`);
      if (event.status === "paid") netRecordedCashCents += Math.round(evidence.amount * 100);
      if (event.status === "refunded") {
        const refundCents = Math.round(evidence.amount * 100);
        if (refundCents > netRecordedCashCents) throw new Error(`${id} refund exceeds recorded paid cash`);
        netRecordedCashCents -= refundCents;
      }
      if (pipeline.updatedAt < at) throw new Error(`pipeline.updatedAt must include every event date for ${id}`);
      if (previousAt && at < previousAt) throw new Error(`${id} event dates must be nondecreasing`);
      if (previousStatus == null && event.status !== "proposal") throw new Error(`${id} must start at proposal`);
      if (previousStatus != null && !TRANSITIONS[previousStatus].has(event.status)) {
        throw new Error(`${id} has invalid transition ${previousStatus} -> ${event.status}`);
      }
      previousStatus = event.status;
      previousAt = at;
    }
    if (record.currentStatus !== previousStatus) throw new Error(`${id}.currentStatus must equal the final event status`);
  }
  return pipeline;
}

function sourceDigest(source) {
  return createHash("sha256").update(JSON.stringify(source)).digest("hex");
}

export function importEngagement(pipelineValue, source, importedAt) {
  const pipeline = validatePipeline(pipelineValue);
  const normalized = validateGeneratedEngagement(source);
  const at = isoDate(importedAt, "at");
  if (at < normalized.proposalDate) throw new Error(`import date ${at} precedes proposal date ${normalized.proposalDate}`);
  const digest = sourceDigest(source);
  const existing = pipeline.engagements.find((record) => record.id === normalized.engagementId);
  if (existing) {
    if (existing.sourceDigest !== digest) throw new Error(`engagement ${normalized.engagementId} already exists with different source evidence`);
    return { pipeline, changed: false, engagement: existing };
  }
  const record = {
    id: normalized.engagementId,
    client: normalized.client,
    provider: normalized.provider,
    decisionOwner: normalized.decisionOwner,
    strategy: normalized.strategy,
    deployment: normalized.deployment,
    standardReviewPrice: normalized.standardReviewPrice,
    monitoringMonthlyPrice: normalized.monitoringMonthlyPrice,
    proposalDate: normalized.proposalDate,
    validThrough: normalized.validThrough,
    importedAt: at,
    sourceDigest: digest,
    currentStatus: "proposal",
    events: [{ status: "proposal", at: normalized.proposalDate }],
  };
  const next = {
    schemaVersion: 1,
    updatedAt: pipeline.updatedAt == null || pipeline.updatedAt < at ? at : pipeline.updatedAt,
    engagements: [...pipeline.engagements, record].sort((left, right) => left.id.localeCompare(right.id)),
  };
  validatePipeline(next);
  return { pipeline: next, changed: true, engagement: record };
}

export function advanceEngagement(pipelineValue, idValue, statusValue, atValue, details = {}) {
  const pipeline = validatePipeline(pipelineValue);
  const id = singleLine(idValue, "id", 240);
  const status = singleLine(statusValue, "status", 40);
  if (!STATUS_SET.has(status)) throw new Error(`invalid destination status: ${status}`);
  const at = isoDate(atValue, "at");
  const evidence = eventEvidence(status, objectValue(details, "transition details"), "transition");
  const index = pipeline.engagements.findIndex((record) => record.id === id);
  if (index < 0) throw new Error(`unknown engagement id: ${id}`);
  const current = pipeline.engagements[index];
  if (current.currentStatus === status) {
    const lastEvent = current.events[current.events.length - 1];
    const sameEvidence = Object.entries(evidence).every(([key, value]) => lastEvent[key] === value);
    if (!sameEvidence) throw new Error(`engagement ${id} is already ${status} with different event evidence`);
    return { pipeline, changed: false, engagement: current };
  }
  if (!TRANSITIONS[current.currentStatus].has(status)) {
    throw new Error(`invalid transition ${current.currentStatus} -> ${status}`);
  }
  const lastAt = current.events[current.events.length - 1].at;
  if (at < lastAt) throw new Error(`transition date ${at} precedes the latest event ${lastAt}`);
  const updated = { ...current, currentStatus: status, events: [...current.events, { status, at, ...evidence }] };
  const engagements = [...pipeline.engagements];
  engagements[index] = updated;
  const next = {
    schemaVersion: 1,
    updatedAt: pipeline.updatedAt == null || pipeline.updatedAt < at ? at : pipeline.updatedAt,
    engagements,
  };
  validatePipeline(next);
  return { pipeline: next, changed: true, engagement: updated };
}

function ratio(numerator, denominator) {
  return denominator === 0 ? null : numerator / denominator;
}

function cents(value) {
  return Math.round(value * 100);
}

function nextAction(record, asOf) {
  switch (record.currentStatus) {
    case "proposal":
      return record.validThrough < asOf
        ? `Refresh or close expired proposal (expired ${record.validThrough})`
        : `Follow up before proposal expiry (${record.validThrough})`;
    case "accepted":
      return "Collect payment before kickoff";
    case "paid":
      return "Confirm complete evidence and start delivery";
    case "in-delivery":
      return "Complete and deliver the review";
    case "delivered":
      return "Offer monthly monitoring";
    case "monitoring":
      return "Deliver the current monthly assurance cycle";
    default:
      return null;
  }
}

export function summarizePipeline(pipelineValue, asOfValue) {
  const pipeline = validatePipeline(pipelineValue);
  const asOf = isoDate(asOfValue, "as-of");
  const counts = Object.fromEntries(PIPELINE_STATUSES.map((status) => [status, 0]));
  let openProposalValueCents = 0;
  let bookedReviewRevenueCents = 0;
  let grossCashCollectedCents = 0;
  let refundedCashCents = 0;
  let deliveredContractValueCents = 0;
  let deliveredNetCashCents = 0;
  let deliveryHours = 0;
  let currentMonthlyRecurringRevenueCents = 0;
  let accepted = 0;
  let paid = 0;
  let delivered = 0;
  let monitoring = 0;
  const expiredProposalIds = [];
  const actions = [];

  for (const record of pipeline.engagements) {
    counts[record.currentStatus] += 1;
    const statuses = eventStatuses(record);
    const recordGrossCashCents = record.events
      .filter((event) => event.status === "paid")
      .reduce((total, event) => total + cents(event.amount), 0);
    const recordRefundedCashCents = record.events
      .filter((event) => event.status === "refunded")
      .reduce((total, event) => total + cents(event.amount), 0);
    const recordDeliveryHours = record.events
      .filter((event) => event.status === "delivered")
      .reduce((total, event) => total + event.hours, 0);
    grossCashCollectedCents += recordGrossCashCents;
    refundedCashCents += recordRefundedCashCents;
    if (record.currentStatus === "proposal") {
      openProposalValueCents += cents(record.standardReviewPrice);
      if (record.validThrough < asOf) expiredProposalIds.push(record.id);
    }
    if (statuses.has("accepted")) accepted += 1;
    if (statuses.has("paid")) paid += 1;
    if (statuses.has("delivered")) delivered += 1;
    if (statuses.has("monitoring")) monitoring += 1;
    if (statuses.has("accepted") && !new Set(["cancelled", "refunded"]).has(record.currentStatus)) {
      bookedReviewRevenueCents += cents(record.standardReviewPrice);
    }
    if (statuses.has("delivered") && record.currentStatus !== "refunded") {
      deliveredContractValueCents += cents(record.standardReviewPrice);
      deliveredNetCashCents += recordGrossCashCents - recordRefundedCashCents;
      deliveryHours += recordDeliveryHours;
    }
    if (record.currentStatus === "monitoring") {
      currentMonthlyRecurringRevenueCents += cents(record.monitoringMonthlyPrice);
    }
    const action = nextAction(record, asOf);
    if (action) actions.push({ id: record.id, client: record.client, status: record.currentStatus, action });
  }

  actions.sort((left, right) => left.client.localeCompare(right.client) || left.id.localeCompare(right.id));
  expiredProposalIds.sort();
  const proposals = pipeline.engagements.length;
  return {
    schemaVersion: 2,
    asOf,
    totalEngagements: proposals,
    counts,
    funnel: {
      proposals,
      accepted,
      paid,
      delivered,
      monitoring,
      proposalToAccepted: ratio(accepted, proposals),
      acceptedToPaid: ratio(paid, accepted),
      paidToDelivered: ratio(delivered, paid),
      deliveredToMonitoring: ratio(monitoring, delivered),
    },
    revenue: {
      currency: "USD",
      openProposalValue: openProposalValueCents / 100,
      bookedReviewRevenue: bookedReviewRevenueCents / 100,
      grossCashCollected: grossCashCollectedCents / 100,
      refundedCash: refundedCashCents / 100,
      netCashCollected: (grossCashCollectedCents - refundedCashCents) / 100,
      deliveredContractValue: deliveredContractValueCents / 100,
      deliveredNetCash: deliveredNetCashCents / 100,
      deliveryHours,
      realizedReviewRevenuePerDeliveryHour: deliveryHours === 0 ? null : deliveredNetCashCents / 100 / deliveryHours,
      currentContractedMonthlyRecurringRevenue: currentMonthlyRecurringRevenueCents / 100,
      currentContractedAnnualRecurringRunRate: (currentMonthlyRecurringRevenueCents * 12) / 100,
    },
    expiredProposalIds,
    nextActions: actions,
  };
}

function amount(value) {
  const digits = Number.isInteger(value) ? 0 : 2;
  return `USD ${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: 2 })}`;
}

function percent(value) {
  return value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;
}

export function renderPipelineSummary(summary) {
  const activeCounts = PIPELINE_STATUSES.filter((status) => summary.counts[status] > 0)
    .map((status) => `| ${status} | ${summary.counts[status]} |`);
  const actionRows = summary.nextActions.map(
    (entry) => `| ${entry.client.replace(/\|/g, "\\|")} | ${entry.status} | ${entry.action.replace(/\|/g, "\\|")} |`,
  );
  return [
    "# Strategy Assurance Pipeline",
    "",
    `As of: ${summary.asOf}`,
    "",
    "## Revenue",
    "",
    `- Open proposal value: ${amount(summary.revenue.openProposalValue)}`,
    `- Booked review revenue: ${amount(summary.revenue.bookedReviewRevenue)}`,
    `- Gross cash collected: ${amount(summary.revenue.grossCashCollected)}`,
    `- Refunds recorded: ${amount(summary.revenue.refundedCash)}`,
    `- Net cash collected: ${amount(summary.revenue.netCashCollected)}`,
    `- Delivered contract value: ${amount(summary.revenue.deliveredContractValue)}`,
    `- Delivered net cash: ${amount(summary.revenue.deliveredNetCash)}`,
    `- Delivery hours: ${summary.revenue.deliveryHours.toLocaleString("en-US", { maximumFractionDigits: 2 })}`,
    `- Realized review revenue/hour: ${summary.revenue.realizedReviewRevenuePerDeliveryHour == null ? "N/A" : amount(summary.revenue.realizedReviewRevenuePerDeliveryHour)}`,
    `- Current contracted monitoring MRR: ${amount(summary.revenue.currentContractedMonthlyRecurringRevenue)}`,
    `- Current contracted monitoring ARR run rate: ${amount(summary.revenue.currentContractedAnnualRecurringRunRate)}`,
    "",
    "## Funnel",
    "",
    `- Proposal → accepted: ${percent(summary.funnel.proposalToAccepted)} (${summary.funnel.accepted}/${summary.funnel.proposals})`,
    `- Accepted → paid: ${percent(summary.funnel.acceptedToPaid)} (${summary.funnel.paid}/${summary.funnel.accepted})`,
    `- Paid → delivered: ${percent(summary.funnel.paidToDelivered)} (${summary.funnel.delivered}/${summary.funnel.paid})`,
    `- Delivered → monitoring: ${percent(summary.funnel.deliveredToMonitoring)} (${summary.funnel.monitoring}/${summary.funnel.delivered})`,
    "",
    "## Current stages",
    "",
    "| Status | Count |",
    "| --- | ---: |",
    ...(activeCounts.length > 0 ? activeCounts : ["| No engagements | 0 |"]),
    "",
    "## Next actions",
    "",
    "| Client | Status | Action |",
    "| --- | --- | --- |",
    ...(actionRows.length > 0 ? actionRows : ["| None | — | No active next action |"]),
    "",
  ].join("\n");
}

export async function loadPipeline(filePath, { allowMissing = false } = {}) {
  try {
    const parsed = JSON.parse(await readFile(filePath, "utf8"));
    return validatePipeline(parsed);
  } catch (error) {
    if (allowMissing && error?.code === "ENOENT") return emptyPipeline();
    if (error instanceof SyntaxError) throw new Error(`pipeline is not valid JSON: ${filePath}`);
    throw error;
  }
}

export async function savePipeline(filePath, pipelineValue) {
  const pipeline = validatePipeline(pipelineValue);
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(pipeline, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, filePath);
  } catch (error) {
    await unlink(temporaryPath).catch(() => {});
    throw error;
  }
}

async function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (args.help) {
      process.stdout.write(usage());
      return;
    }
    if (args.command === "import") {
      const pipeline = await loadPipeline(args.pipelinePath, { allowMissing: true });
      const engagement = JSON.parse(await readFile(args.engagementPath, "utf8"));
      const result = importEngagement(pipeline, engagement, args.at);
      if (result.changed) await savePipeline(args.pipelinePath, result.pipeline);
      process.stdout.write(`${JSON.stringify({ changed: result.changed, id: result.engagement.id, status: result.engagement.currentStatus }, null, 2)}\n`);
      return;
    }
    if (args.command === "advance") {
      const pipeline = await loadPipeline(args.pipelinePath);
      const result = advanceEngagement(pipeline, args.id, args.status, args.at, args.details);
      if (result.changed) await savePipeline(args.pipelinePath, result.pipeline);
      process.stdout.write(`${JSON.stringify({ changed: result.changed, id: result.engagement.id, status: result.engagement.currentStatus }, null, 2)}\n`);
      return;
    }
    const summary = summarizePipeline(await loadPipeline(args.pipelinePath), args.asOf);
    process.stdout.write(args.json ? `${JSON.stringify(summary, null, 2)}\n` : renderPipelineSummary(summary));
  } catch (error) {
    process.stderr.write(`Strategy Assurance pipeline failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
