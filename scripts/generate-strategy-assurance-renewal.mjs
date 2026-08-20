#!/usr/bin/env node

import { createHash } from "node:crypto";
import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { loadPipeline, validatePipeline } from "./strategy-assurance-pipeline.mjs";

const OUTPUT_FILES = ["monitoring-order.md", "monitoring-order.json"];
const DEFAULT_PIPELINE = path.join(".tmp", "strategy-assurance", "pipeline.json");

function usage() {
  return `Generate a local monitoring-renewal offer for a delivered Strategy Assurance engagement.

Usage:
  npm run assurance:renewal -- \\
    --id ENGAGEMENT_ID \\
    --start YYYY-MM-DD [options]

Required:
  --id ID                    Delivered engagementId from the local pipeline
  --start YYYY-MM-DD         Proposed first monitoring-cycle date

Options:
  --pipeline PATH            Default: ${DEFAULT_PIPELINE}
  --offer-date YYYY-MM-DD    Default: current UTC date
  --valid-days NUMBER        Default: 14; maximum: 90
  --months NUMBER            Initial committed cycles; default: 3; maximum: 24
  --output PATH              Default: .tmp/strategy-assurance/renewals/<id>-<start>
  --force                    Replace only the two generated renewal files
  --help

The command creates an offer only. It does not send, sign, invoice, charge, or advance the pipeline.
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
  if (new Date(timestamp).toISOString().slice(0, 10) !== text) throw new Error(`--${flag} must be a real calendar date`);
  return text;
}

function addUtcDays(date, days) {
  return utcDate(Date.parse(`${date}T00:00:00.000Z`) + days * 86_400_000);
}

function singleLine(value, flag, maximum = 500) {
  const text = String(value ?? "").trim();
  if (!text) throw new Error(`--${flag} is required`);
  if (text.length > maximum) throw new Error(`--${flag} must be at most ${maximum} characters`);
  if (/[\u0000-\u001f\u007f]/.test(text)) throw new Error(`--${flag} cannot contain control characters or newlines`);
  return text;
}

function positiveInteger(value, flag, maximum) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > maximum) {
    throw new Error(`--${flag} must be a whole number from 1 through ${maximum}`);
  }
  return parsed;
}

function markdownText(value) {
  return value.replace(/\\/g, "\\\\").replace(/([*_`[\]<>|])/g, "\\$1");
}

function amount(value) {
  const digits = Number.isInteger(value) ? 0 : 2;
  return `USD ${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: 2 })}`;
}

export function parseArgs(argv, nowMs = Date.now()) {
  if (argv.length === 0 || argv.includes("--help")) return { help: true };
  const values = {};
  let force = false;
  const seen = new Set();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--force") {
      if (seen.has("force")) throw new Error("Duplicate option: --force");
      seen.add("force");
      force = true;
      continue;
    }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const name = token.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (!new Set(["id", "start", "pipeline", "offerDate", "validDays", "months", "output"]).has(name)) {
      throw new Error(`Unknown option: ${token}`);
    }
    if (seen.has(name)) throw new Error(`Duplicate option: ${token}`);
    const value = argv[index + 1];
    if (value == null || value.startsWith("--")) throw new Error(`${token} requires a value`);
    values[name] = value;
    seen.add(name);
    index += 1;
  }

  const id = singleLine(values.id, "id", 240);
  const start = isoDate(values.start, "start");
  const offerDate = isoDate(values.offerDate || utcDate(nowMs), "offer-date");
  if (start < offerDate) throw new Error("--start must not precede --offer-date");
  const validDays = positiveInteger(values.validDays ?? 14, "valid-days", 90);
  const months = positiveInteger(values.months ?? 3, "months", 24);
  const defaultOutput = path.join(".tmp", "strategy-assurance", "renewals", `${id}-${start}`);
  return {
    help: false,
    id,
    start,
    offerDate,
    validDays,
    validThrough: addUtcDays(offerDate, validDays),
    months,
    force,
    pipelinePath: path.resolve(singleLine(values.pipeline || DEFAULT_PIPELINE, "pipeline", 1000)),
    outputDir: path.resolve(singleLine(values.output || defaultOutput, "output", 1000)),
  };
}

export function buildRenewalOffer(pipelineValue, config) {
  const pipeline = validatePipeline(pipelineValue);
  const engagement = pipeline.engagements.find((record) => record.id === config.id);
  if (!engagement) throw new Error(`unknown engagement id: ${config.id}`);
  if (engagement.currentStatus !== "delivered") {
    throw new Error(`engagement ${config.id} must be delivered before offering monitoring; current status is ${engagement.currentStatus}`);
  }
  const digest = createHash("sha256")
    .update(`${engagement.id}|${config.start}|${config.months}|${engagement.monitoringMonthlyPrice}`)
    .digest("hex")
    .slice(0, 16);
  const renewalId = `strategy-assurance-monitoring-${digest}`;
  const initialContractValue = (Math.round(engagement.monitoringMonthlyPrice * 100) * config.months) / 100;
  const offer = {
    schemaVersion: 1,
    renewalOfferType: "strategy-assurance-monitoring",
    status: "offered",
    renewalId,
    parentEngagementId: engagement.id,
    client: engagement.client,
    provider: engagement.provider,
    decisionOwner: engagement.decisionOwner,
    strategy: engagement.strategy,
    deployment: engagement.deployment,
    offerDate: config.offerDate,
    validThrough: config.validThrough,
    proposedStart: config.start,
    initialMonthlyCycles: config.months,
    currency: "USD",
    monthlyPrice: engagement.monitoringMonthlyPrice,
    initialContractValue,
    proposedPaymentTiming: "before-each-monthly-cycle",
    externalActionPerformed: false,
  };
  const markdown = `# Strategy Assurance Monitoring Order

**Status:** OFFERED — acceptance required
**Renewal ID:** ${renewalId}
**Parent engagement:** ${engagement.id}
**Client:** ${markdownText(engagement.client)}
**Provider:** ${markdownText(engagement.provider)}
**Decision owner:** ${markdownText(engagement.decisionOwner)}
**Offer date:** ${config.offerDate}
**Valid through:** ${config.validThrough}
**Proposed monitoring start:** ${config.start}

## Scope

Monthly monitoring covers the reviewed ${markdownText(engagement.strategy)} strategy and ${markdownText(engagement.deployment)} deployment only. Each monthly cycle includes:

1. One exchange-reconciled revenue update.
2. Drift, control-state, and material-change review against the delivered baseline.
3. A concise change and exception summary.
4. One 30-minute readout.

It excludes remediation implementation, incident response, a new strategy or deployment review, custody, discretionary trading, and continuous on-call coverage.

## Commercials

- Monthly fee: ${amount(engagement.monitoringMonthlyPrice)} per deployment
- Initial committed cycles: ${config.months}
- Initial contract value: ${amount(initialContractValue)}
- Proposed payment timing: before each monthly cycle begins

The client may activate monitoring only by accepting this order and the governing agreement. Tax, cancellation, refund, confidentiality, liability, and dispute terms remain subject to that agreement. This offer is not a tax invoice, receipt, payment confirmation, investment advice, or performance guarantee.

## Access boundary

Prefer customer-run exports. Any temporary access must remain least-privilege and read-only with an agreed expiry. Never provide withdrawal or trading permission, seed phrases, private keys, or unrestricted cloud credentials.

## Acceptance

After both parties accept, record the pipeline transition from delivered to monitoring using the actual acceptance date. Generating this offer does not advance the pipeline or perform billing.

| Client | Provider |
| --- | --- |
| ${markdownText(engagement.client)} | ${markdownText(engagement.provider)} |
| Name: ____________________ | Name: ____________________ |
| Title: ____________________ | Title: ____________________ |
| Signature: ____________________ | Signature: ____________________ |
| Date: ____________________ | Date: ____________________ |
`;
  return {
    offer,
    files: {
      "monitoring-order.md": markdown,
      "monitoring-order.json": `${JSON.stringify(offer, null, 2)}\n`,
    },
  };
}

export async function writeRenewalOffer(pipelineValue, config) {
  const built = buildRenewalOffer(pipelineValue, config);
  if (!config.force) {
    try {
      await access(config.outputDir);
      throw new Error(`Output directory already exists: ${config.outputDir}. Use --force to replace generated renewal files.`);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  await mkdir(config.outputDir, { recursive: true });
  for (const fileName of OUTPUT_FILES) {
    await writeFile(path.join(config.outputDir, fileName), built.files[fileName], {
      encoding: "utf8",
      flag: config.force ? "w" : "wx",
    });
  }
  return { outputDir: config.outputDir, files: [...OUTPUT_FILES], offer: built.offer };
}

async function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    if (config.help) {
      process.stdout.write(usage());
      return;
    }
    const result = await writeRenewalOffer(await loadPipeline(config.pipelinePath), config);
    process.stdout.write(`${JSON.stringify({ outputDir: result.outputDir, files: result.files, renewalId: result.offer.renewalId }, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`Strategy Assurance renewal generation failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
