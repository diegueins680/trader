#!/usr/bin/env node

import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const FILE_NAMES = [
  "proposal.md",
  "evidence-request.md",
  "outreach.md",
  "payment-request.md",
  "payment-request.json",
  "engagement.json",
];
const REQUIRED_TEXT_FLAGS = ["client", "provider", "strategy", "deployment", "decisionOwner"];

function usage() {
  return `Generate a tailored Strategy Assurance commercial kit locally.

Usage:
  node scripts/generate-strategy-assurance-kit.mjs \\
    --client "Client legal name" \\
    --provider "Provider legal name" \\
    --decision-owner "Name, title" \\
    --strategy "Strategy name/version" \\
    --deployment "Deployment name/region" [options]

Required:
  --client TEXT
  --provider TEXT
  --decision-owner TEXT
  --strategy TEXT
  --deployment TEXT

Scope options:
  --repository TEXT             Default: Customer-provided repository
  --commit TEXT                 Default: To be confirmed at kickoff
  --venue TEXT                  Default: Binance
  --account-label TEXT          Default: Read-only account in scope
  --asset CODE                  Default: USDT
  --start YYYY-MM-DD            Optional revenue-period start
  --end YYYY-MM-DD              Optional revenue-period end
  --infrastructure-cost TEXT    Default: Client-supplied

Commercial options:
  --proposal-date YYYY-MM-DD    Default: current UTC date
  --valid-days NUMBER           Default: 14
  --price NUMBER                Default: 2500
  --monitoring-price NUMBER     Default: 399
  --turnaround-days NUMBER      Default: 5
  --output PATH                 Default: .tmp/strategy-assurance/<client>-<date>
  --force                       Replace only the six generated files in an existing directory
  --help
`;
}

function flagName(raw) {
  return raw.replace(/^--/, "").replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function cleanText(value, flag, { required = false, maxLength = 240 } = {}) {
  const cleaned = String(value ?? "").trim();
  if (required && !cleaned) throw new Error(`--${flag} is required`);
  if (cleaned.length > maxLength) throw new Error(`--${flag} must be at most ${maxLength} characters`);
  if (/[\u0000-\u001f\u007f]/.test(cleaned)) throw new Error(`--${flag} cannot contain control characters or newlines`);
  return cleaned;
}

function positiveNumber(value, flag) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1_000_000) {
    throw new Error(`--${flag} must be a positive number no greater than 1000000`);
  }
  const rounded = Math.round(parsed * 100) / 100;
  if (Math.abs(parsed - rounded) > 1e-9) throw new Error(`--${flag} must use at most two decimal places`);
  return rounded;
}

function positiveInteger(value, flag, maximum) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > maximum) {
    throw new Error(`--${flag} must be a whole number from 1 through ${maximum}`);
  }
  return parsed;
}

function isoDate(value, flag, { optional = false } = {}) {
  const cleaned = cleanText(value, flag);
  if (optional && !cleaned) return "";
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(cleaned);
  if (!match) throw new Error(`--${flag} must use YYYY-MM-DD`);
  const timestamp = Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  if (new Date(timestamp).toISOString().slice(0, 10) !== cleaned) {
    throw new Error(`--${flag} must be a real calendar date`);
  }
  return cleaned;
}

function utcDate(timestampMs) {
  return new Date(timestampMs).toISOString().slice(0, 10);
}

function addUtcDays(date, days) {
  const timestamp = Date.parse(`${date}T00:00:00.000Z`);
  return utcDate(timestamp + days * 86_400_000);
}

function slug(value, maxLength = 80) {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase()
    .slice(0, maxLength) || "client";
}

function markdownText(value) {
  return value.replace(/\\/g, "\\\\").replace(/([*_`[\]<>|])/g, "\\$1");
}

function currency(value) {
  const digits = Number.isInteger(value) ? 0 : 2;
  return `USD ${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: 2 })}`;
}

export function parseArgs(argv, nowMs = Date.now()) {
  const raw = {};
  const seen = new Set();
  let force = false;
  let help = false;

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--help") {
      help = true;
      continue;
    }
    if (token === "--force") {
      force = true;
      continue;
    }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const name = flagName(token);
    const supported = new Set([
      "client",
      "provider",
      "decisionOwner",
      "strategy",
      "deployment",
      "repository",
      "commit",
      "venue",
      "accountLabel",
      "asset",
      "start",
      "end",
      "infrastructureCost",
      "proposalDate",
      "validDays",
      "price",
      "monitoringPrice",
      "turnaroundDays",
      "output",
    ]);
    if (!supported.has(name)) throw new Error(`Unknown option: ${token}`);
    if (seen.has(name)) throw new Error(`Duplicate option: ${token}`);
    const value = argv[index + 1];
    if (value == null || value.startsWith("--")) throw new Error(`${token} requires a value`);
    raw[name] = value;
    seen.add(name);
    index += 1;
  }

  if (help) return { help: true };
  for (const name of REQUIRED_TEXT_FLAGS) {
    const dashed = name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
    raw[name] = cleanText(raw[name], dashed, { required: true });
  }

  const proposalDate = isoDate(raw.proposalDate || utcDate(nowMs), "proposal-date");
  const validDays = positiveInteger(raw.validDays ?? 14, "valid-days", 90);
  const start = isoDate(raw.start, "start", { optional: true });
  const end = isoDate(raw.end, "end", { optional: true });
  if (start && end && start > end) throw new Error("--start must not be after --end");
  const asset = cleanText(raw.asset || "USDT", "asset", { required: true, maxLength: 12 }).toUpperCase();
  if (!/^[A-Z0-9]{2,12}$/.test(asset)) throw new Error("--asset must contain 2-12 uppercase letters or digits");

  const defaultOutput = path.join(".tmp", "strategy-assurance", `${slug(raw.client)}-${proposalDate}`);
  return {
    help: false,
    force,
    outputDir: path.resolve(cleanText(raw.output || defaultOutput, "output", { required: true, maxLength: 1000 })),
    client: raw.client,
    provider: raw.provider,
    decisionOwner: raw.decisionOwner,
    strategy: raw.strategy,
    deployment: raw.deployment,
    repository: cleanText(raw.repository || "Customer-provided repository", "repository"),
    commit: cleanText(raw.commit || "To be confirmed at kickoff", "commit"),
    venue: cleanText(raw.venue || "Binance", "venue", { required: true }),
    accountLabel: cleanText(raw.accountLabel || "Read-only account in scope", "account-label", { required: true }),
    asset,
    start,
    end,
    infrastructureCost: cleanText(raw.infrastructureCost || "Client-supplied", "infrastructure-cost"),
    proposalDate,
    validDays,
    validThrough: addUtcDays(proposalDate, validDays),
    price: positiveNumber(raw.price ?? 2500, "price"),
    monitoringPrice: positiveNumber(raw.monitoringPrice ?? 399, "monitoring-price"),
    turnaroundDays: positiveInteger(raw.turnaroundDays ?? 5, "turnaround-days", 30),
  };
}

function revenuePeriod(config) {
  if (config.start && config.end) return `${config.start} through ${config.end}`;
  if (config.start) return `${config.start} through an end date agreed at kickoff`;
  if (config.end) return `an agreed start date through ${config.end}`;
  return "To be agreed at kickoff";
}

function buildProposal(config) {
  const client = markdownText(config.client);
  const provider = markdownText(config.provider);
  return `# Strategy Assurance Proposal and Statement of Work

## Engagement

**Client:** ${client}
**Provider:** ${provider}
**Proposal date:** ${config.proposalDate}
**Valid through:** ${config.validThrough}
**Client decision owner:** ${markdownText(config.decisionOwner)}

## Objective

Provide an evidence-backed technical and economic assessment of one algorithmic crypto-trading deployment so ${client} can decide whether to operate it as-is, operate with restrictions, continue paper trading, or stop pending remediation.

This engagement is engineering assurance. It is not investment, legal, tax, regulatory, or accounting advice; an audit opinion; custody; discretionary trading; or a guarantee of profitability or future performance.

## Fixed scope

- **Strategy:** ${markdownText(config.strategy)}
- **Repository:** ${markdownText(config.repository)}
- **Reviewed commit:** ${markdownText(config.commit)}
- **Exchange account:** ${markdownText(config.venue)} — ${markdownText(config.accountLabel)}
- **Deployment:** ${markdownText(config.deployment)}
- **Revenue period:** ${revenuePeriod(config)}
- **Settlement asset:** ${config.asset}
- **Infrastructure cost:** ${markdownText(config.infrastructureCost)}

The standard review covers one strategy, one exchange account, and one production deployment. Additional scope requires written change approval before work begins.

## Deliverables

1. Exchange-reconciled revenue report with machine-readable JSON plus daily and symbol CSV exports.
2. Strategy-evidence assessment covering walk-forward design, costs, activity, selection bias, holdout discipline, and backtest-to-production gaps.
3. Execution and risk-control assessment covering orders, fees/funding, exposure, loss/drawdown halts, stale data, and restart ownership.
4. Deployment assessment covering credentials, executor boundaries, persistence, readiness, observability, rollback, and verification evidence.
5. Red/amber/green findings ledger with evidence, impact, owner, recommended action, and disposition.
6. Decision memo with an operate, restrict, paper-only, or stop-pending-remediation conclusion.
7. One 60-minute readout and one written clarification round requested within seven calendar days after delivery.

## Schedule

Delivery is targeted within ${config.turnaroundDays} business days after all required inputs and access are complete. Missing, contradictory, or inaccessible evidence pauses the delivery clock and is recorded in writing.

## Fees and payment

**Fixed fee:** ${currency(config.price)}, due before kickoff.

The fee excludes remediation implementation, incident response, additional strategies or deployments, third-party charges, and continuous monitoring. After delivery, the client may elect monthly monitoring at ${currency(config.monitoringPrice)} per reviewed deployment under a separate written order.

## Client responsibilities and access boundary

The client supplies the reviewed source revision, deployment configuration, strategy evidence, exchange evidence, cost inputs, and incident history. Exchange access must be customer-run exports or temporary least-privilege read-only access. Never provide withdrawal permission, trading permission, seed phrases, private keys, or unrestricted cloud credentials. Temporary access must have an agreed expiry and be revoked after delivery.

The client remains responsible for trading, capital allocation, accepted risk, legal compliance, and accounting treatment. Findings describe only the reviewed evidence, revision, environment, and period. Commercial and legal terms should be reviewed for the parties' jurisdictions and business structures.

## Acceptance

By signing, the parties approve this scope, fee, schedule, and access boundary, subject to their governing agreement.

| Client | Provider |
| --- | --- |
| ${client} | ${provider} |
| Name: ____________________ | Name: ____________________ |
| Title: ____________________ | Title: ____________________ |
| Signature: ____________________ | Signature: ____________________ |
| Date: ____________________ | Date: ____________________ |
`;
}

function buildEvidenceRequest(config) {
  return `# Strategy Assurance Evidence Request

**Client:** ${markdownText(config.client)}
**Strategy:** ${markdownText(config.strategy)}
**Deployment:** ${markdownText(config.deployment)}
**Decision owner:** ${markdownText(config.decisionOwner)}

The ${config.turnaroundDays}-business-day delivery clock starts when the applicable inputs below are complete and accessible.

## Decision and scope

Please confirm:

1. What operating decision must this review unblock?
2. Is the deployment live, paper, or pre-production, and what capital can it affect?
3. Which strategy, exchange account, deployment, repository, and exact commit are authoritative?
4. Who owns remediation and signs off on retained risk?
5. Are there known incidents, accounting disagreements, or urgent constraints?

## Evidence checklist

- [ ] Repository access or customer-supplied archive, including exact commit ${markdownText(config.commit)}
- [ ] Production manifests and non-secret runtime configuration for ${markdownText(config.deployment)}
- [ ] Backtest, optimizer, walk-forward, holdout, and cost-assumption evidence
- [ ] Intended production universe, intervals, sizing, and risk limits
- [ ] Exchange income and trade exports for ${revenuePeriod(config)} in ${config.asset}
- [ ] Infrastructure cost for the same period: ${markdownText(config.infrastructureCost)}
- [ ] Relevant incidents, operator runbooks, monitoring, and rollback evidence
- [ ] A technical owner available for clarification

## Secure access

Prefer customer-run exports. If temporary access is necessary, use a read-only credential limited to the required account-data endpoints, record its purpose and expiry, and revoke it after delivery. Do not send withdrawal-enabled or trading-enabled credentials, seed phrases, private wallet keys, unrestricted cloud credentials, or secrets embedded in tickets and reports.

## Completion confirmation

Reply with the evidence locations, the exact reviewed commit, any unavailable items, and the person authorized to resolve scope questions. Missing evidence is documented as a limitation or finding; it is never silently inferred.
`;
}

function buildOutreach(config) {
  return `# Strategy Assurance Outreach

## Subject

Evidence-backed review for ${markdownText(config.strategy)}

## Initial message

Hi ${markdownText(config.decisionOwner)},

I offer a fixed ${config.turnaroundDays}-business-day engineering review of one automated trading deployment. For ${markdownText(config.strategy)}, the review would reconcile exchange P&L and fees, test the strategy evidence and production risk controls, and end with a prioritized operate, restrict, paper-only, or stop-pending-remediation memo.

The fixed price is ${currency(config.price)}. This is engineering assurance—not investment advice or a return guarantee. If that operating decision would be useful, I can send the exact evidence checklist and scope.

Regards,
${markdownText(config.provider)}

## Follow-up

Hi ${markdownText(config.decisionOwner)},

Following up on the Strategy Assurance review for ${markdownText(config.strategy)}. The useful outcome is a fast, evidence-backed operating decision and a ranked remediation plan; it does not require handing over trading or withdrawal authority. The proposal is valid through ${config.validThrough}.

If this is not a current priority, no action is needed.
`;
}

function buildPaymentRequest(config, engagementId) {
  return `# Strategy Assurance Pro Forma Payment Request

**This is a prepayment request, not a tax invoice or receipt.**

**Reference:** ${engagementId}
**Client:** ${markdownText(config.client)}
**Provider:** ${markdownText(config.provider)}
**Issue date:** ${config.proposalDate}
**Proposal valid through:** ${config.validThrough}
**Payment timing:** Before kickoff

| Description | Amount |
| --- | ---: |
| Fixed-scope Strategy Assurance review for ${markdownText(config.strategy)} / ${markdownText(config.deployment)} | ${currency(config.price)} |
| **Total due** | **${currency(config.price)}** |

## Settlement

The provider will supply approved payment instructions through a separately agreed secure channel. Include the reference above with payment. Do not place bank credentials, card data, account passwords, private keys, or payment-provider secrets in the repository, engagement record, tickets, or report artifacts.

Kickoff occurs only after cleared payment and complete required inputs. If payment is initiated after the proposal-validity date, request written confirmation that scope and price remain available. The provider will issue any legally required tax invoice or receipt separately under the applicable jurisdiction and governing agreement.

This request does not alter the signed proposal, create investment or accounting advice, or authorize custody, trading, or withdrawal access.
`;
}

export function buildCommercialKit(config) {
  const engagementId = [
    "strategy-assurance",
    slug(config.client, 48),
    slug(config.strategy, 48),
    slug(config.deployment, 48),
    config.proposalDate,
  ].join("-");
  const engagement = {
    schemaVersion: 1,
    engagementId,
    engagementType: "strategy-assurance-standard",
    status: "proposal",
    client: config.client,
    provider: config.provider,
    decisionOwner: config.decisionOwner,
    scope: {
      strategy: config.strategy,
      deployment: config.deployment,
      repository: config.repository,
      reviewedCommit: config.commit,
      venue: config.venue,
      accountLabel: config.accountLabel,
      settlementAsset: config.asset,
      revenuePeriod: { start: config.start || null, end: config.end || null },
      infrastructureCost: config.infrastructureCost,
    },
    commercials: {
      currency: "USD",
      standardReviewPrice: config.price,
      monitoringMonthlyPrice: config.monitoringPrice,
      paymentTiming: "before-kickoff",
      proposalDate: config.proposalDate,
      validThrough: config.validThrough,
    },
    delivery: {
      turnaroundBusinessDays: config.turnaroundDays,
      startsAfterCompleteInputs: true,
      decisionOutcomes: ["operate", "restrict", "paper-only", "stop-pending-remediation"],
    },
    accessBoundary: {
      preferred: "customer-run-exports",
      temporaryAccess: "least-privilege-read-only",
      forbidden: ["withdrawal", "trading", "seed-phrase", "private-key", "unrestricted-cloud"],
    },
  };
  const paymentRequest = {
    schemaVersion: 1,
    paymentRequestType: "strategy-assurance-pro-forma",
    status: "unpaid",
    engagementId,
    issueDate: config.proposalDate,
    validThrough: config.validThrough,
    dueWhen: "before-kickoff",
    currency: "USD",
    amount: config.price,
    client: config.client,
    provider: config.provider,
    description: `Fixed-scope Strategy Assurance review for ${config.strategy} / ${config.deployment}`,
    externalPaymentActionPerformed: false,
  };
  return {
    engagement,
    paymentRequest,
    files: {
      "proposal.md": buildProposal(config),
      "evidence-request.md": buildEvidenceRequest(config),
      "outreach.md": buildOutreach(config),
      "payment-request.md": buildPaymentRequest(config, engagementId),
      "payment-request.json": `${JSON.stringify(paymentRequest, null, 2)}\n`,
      "engagement.json": `${JSON.stringify(engagement, null, 2)}\n`,
    },
  };
}

export async function writeCommercialKit(config) {
  const kit = buildCommercialKit(config);
  if (!config.force) {
    try {
      await access(config.outputDir);
      throw new Error(`Output directory already exists: ${config.outputDir}. Use --force to replace generated files.`);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  await mkdir(config.outputDir, { recursive: true });
  for (const fileName of FILE_NAMES) {
    await writeFile(path.join(config.outputDir, fileName), kit.files[fileName], { encoding: "utf8", flag: config.force ? "w" : "wx" });
  }
  return { outputDir: config.outputDir, files: [...FILE_NAMES], engagement: kit.engagement };
}

async function main() {
  let config;
  try {
    config = parseArgs(process.argv.slice(2));
    if (config.help) {
      process.stdout.write(usage());
      return;
    }
    const result = await writeCommercialKit(config);
    process.stdout.write(`${JSON.stringify({ outputDir: result.outputDir, files: result.files }, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`Strategy Assurance kit generation failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
