#!/usr/bin/env node

import { createHash } from "node:crypto";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_QUEUE = path.join("docs", "strategy-assurance-prospect-queue.json");
const DEFAULT_CHECKLIST = path.join("docs", "strategy-assurance-pre-live-checklist.md");
const MAX_PROSPECTS = 3;

function usage() {
  return `Prepare a local, human-reviewed Strategy Assurance outreach package.

Usage:
  npm run assurance:outreach -- \\
    --provider "Provider legal name" \\
    --sender "Name, title" \\
    --prospect "OctoBot" [--prospect "Another organization"] [options]

Required:
  --provider TEXT             Provider identity used in the draft
  --sender TEXT               Human sender name and title
  --prospect TEXT             Exact queue organization; repeat up to three times

Options:
  --campaign-date YYYY-MM-DD  Default: current UTC date
  --queue PATH                Default: ${DEFAULT_QUEUE}
  --checklist PATH            Default: ${DEFAULT_CHECKLIST}
  --output PATH               Default: .tmp/strategy-assurance/outreach/<date>-<prospects>
  --include-watchlist         Permit an explicitly selected watchlist organization
  --force                     Replace only the same campaign's generated files
  --help

The command prepares drafts and records only. It performs no outreach, form submission,
community join, affiliation claim, pipeline transition, or other external action.
`;
}

function singleLine(value, flag, maximum = 240) {
  const text = String(value ?? "").trim();
  if (!text) throw new Error(`--${flag} is required`);
  if (text.length > maximum) throw new Error(`--${flag} must be at most ${maximum} characters`);
  if (/[\u0000-\u001f\u007f]/.test(text)) throw new Error(`--${flag} cannot contain control characters or newlines`);
  return text;
}

function isoDate(value, flag) {
  const text = singleLine(value, flag, 10);
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
  if (!match) throw new Error(`--${flag} must use YYYY-MM-DD`);
  const timestamp = Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  if (new Date(timestamp).toISOString().slice(0, 10) !== text) throw new Error(`--${flag} must be a real calendar date`);
  return text;
}

function utcDate(timestampMs) {
  return new Date(timestampMs).toISOString().slice(0, 10);
}

export function addBusinessDays(date, days) {
  let timestamp = Date.parse(`${isoDate(date, "date")}T00:00:00.000Z`);
  let remaining = days;
  if (!Number.isSafeInteger(remaining) || remaining < 0 || remaining > 365) {
    throw new Error("business days must be a whole number from 0 through 365");
  }
  while (remaining > 0) {
    timestamp += 86_400_000;
    const day = new Date(timestamp).getUTCDay();
    if (day !== 0 && day !== 6) remaining -= 1;
  }
  return utcDate(timestamp);
}

function slug(value, maximum = 80) {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase()
    .slice(0, maximum) || "prospect";
}

function markdownText(value) {
  return value.replace(/\\/g, "\\\\").replace(/([*_`[\]<>|])/g, "\\$1");
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function queueText(value, label, maximum = 2000) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`prospect queue ${label} must be a non-empty string`);
  if (value.length > maximum || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)) {
    throw new Error(`prospect queue ${label} is invalid`);
  }
  return value.trim();
}

function httpsUrl(value, label) {
  const text = queueText(value, label, 2000);
  let parsed;
  try {
    parsed = new URL(text);
  } catch {
    throw new Error(`prospect queue ${label} must be a valid URL`);
  }
  if (parsed.protocol !== "https:") throw new Error(`prospect queue ${label} must use HTTPS`);
  return text;
}

export function validateProspectQueue(value) {
  if (!isObject(value) || value.schemaVersion !== 1) throw new Error("prospect queue schemaVersion must be 1");
  const researchedAt = isoDate(value.researchedAt, "queue-researched-at");
  if (value.externalActionsPerformed !== false) throw new Error("prospect queue must state that no external actions were performed");
  if (!Array.isArray(value.prospects) || value.prospects.length === 0 || value.prospects.length > 100) {
    throw new Error("prospect queue must contain from 1 through 100 prospects");
  }

  const names = new Set();
  const ranks = new Set();
  const prospects = value.prospects.map((prospect, index) => {
    const label = `prospects[${index}]`;
    if (!isObject(prospect)) throw new Error(`prospect queue ${label} must be an object`);
    const organization = queueText(prospect.organization, `${label}.organization`, 120);
    const nameKey = organization.toLocaleLowerCase("en-US");
    if (names.has(nameKey)) throw new Error(`prospect queue organization is duplicated: ${organization}`);
    names.add(nameKey);
    if (!Number.isSafeInteger(prospect.rank) || prospect.rank <= 0 || ranks.has(prospect.rank)) {
      throw new Error(`prospect queue ${label}.rank must be a unique positive integer`);
    }
    ranks.add(prospect.rank);
    if (!isObject(prospect.score)) throw new Error(`prospect queue ${label}.score must be an object`);
    const scoreMaximums = {
      audienceFit: 3,
      organizationAccess: 2,
      commercialAlignment: 3,
      currentActivity: 2,
      conflictPenalty: 3,
    };
    for (const [field, maximum] of Object.entries(scoreMaximums)) {
      if (!Number.isSafeInteger(prospect.score[field]) || prospect.score[field] < 0 || prospect.score[field] > maximum) {
        throw new Error(`prospect queue ${label}.score.${field} must be a whole number from 0 through ${maximum}`);
      }
    }
    const total =
      prospect.score.audienceFit +
      prospect.score.organizationAccess +
      prospect.score.commercialAlignment +
      prospect.score.currentActivity -
      prospect.score.conflictPenalty;
    if (prospect.score.total !== total) throw new Error(`prospect queue ${label}.score.total does not match its components`);
    if (!isObject(prospect.officialRoute)) throw new Error(`prospect queue ${label}.officialRoute must be an object`);
    if (!Array.isArray(prospect.evidence) || prospect.evidence.length === 0 || prospect.evidence.length > 20) {
      throw new Error(`prospect queue ${label}.evidence must contain from 1 through 20 entries`);
    }
    const evidence = prospect.evidence.map((entry, evidenceIndex) => {
      const evidenceLabel = `${label}.evidence[${evidenceIndex}]`;
      if (!isObject(entry)) throw new Error(`prospect queue ${evidenceLabel} must be an object`);
      return {
        claim: queueText(entry.claim, `${evidenceLabel}.claim`),
        url: httpsUrl(entry.url, `${evidenceLabel}.url`),
        checkedAt: isoDate(entry.checkedAt, `${evidenceLabel}.checkedAt`),
      };
    });
    return {
      rank: prospect.rank,
      organization,
      kind: queueText(prospect.kind, `${label}.kind`, 120),
      priority: queueText(prospect.priority, `${label}.priority`, 40),
      status: queueText(prospect.status, `${label}.status`, 120),
      score: { ...prospect.score },
      officialRoute: {
        label: queueText(prospect.officialRoute.label, `${label}.officialRoute.label`, 240),
        url: httpsUrl(prospect.officialRoute.url, `${label}.officialRoute.url`),
        rule: queueText(prospect.officialRoute.rule, `${label}.officialRoute.rule`),
      },
      evidence,
      hypothesis: queueText(prospect.hypothesis, `${label}.hypothesis`),
      likelyObjection: queueText(prospect.likelyObjection, `${label}.likelyObjection`),
      firstAsk: queueText(prospect.firstAsk, `${label}.firstAsk`),
      manualNextAction: queueText(prospect.manualNextAction, `${label}.manualNextAction`),
    };
  });

  return { schemaVersion: 1, researchedAt, externalActionsPerformed: false, prospects };
}

export function parseArgs(argv, nowMs = Date.now()) {
  if (argv.length === 0 || argv.includes("--help")) return { help: true };
  const values = {};
  const prospectNames = [];
  const seen = new Set();
  let force = false;
  let includeWatchlist = false;

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--force" || token === "--include-watchlist") {
      const name = token.slice(2);
      if (seen.has(name)) throw new Error(`Duplicate option: ${token}`);
      seen.add(name);
      if (token === "--force") force = true;
      else includeWatchlist = true;
      continue;
    }
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const name = token.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (!new Set(["provider", "sender", "prospect", "campaignDate", "queue", "checklist", "output"]).has(name)) {
      throw new Error(`Unknown option: ${token}`);
    }
    if (name !== "prospect" && seen.has(name)) throw new Error(`Duplicate option: ${token}`);
    const argument = argv[index + 1];
    if (argument == null || argument.startsWith("--")) throw new Error(`${token} requires a value`);
    if (name === "prospect") prospectNames.push(singleLine(argument, "prospect", 120));
    else {
      values[name] = argument;
      seen.add(name);
    }
    index += 1;
  }

  const provider = singleLine(values.provider, "provider");
  const sender = singleLine(values.sender, "sender");
  if (prospectNames.length === 0) throw new Error("at least one --prospect is required");
  if (prospectNames.length > MAX_PROSPECTS) throw new Error(`no more than ${MAX_PROSPECTS} --prospect options are allowed`);
  const uniqueNames = new Set(prospectNames.map((name) => name.toLocaleLowerCase("en-US")));
  if (uniqueNames.size !== prospectNames.length) throw new Error("--prospect values must be unique");
  const campaignDate = isoDate(values.campaignDate || utcDate(nowMs), "campaign-date");
  const campaignSlug = prospectNames.map((name) => slug(name, 36)).join("-");
  const defaultOutput = path.join(".tmp", "strategy-assurance", "outreach", `${campaignDate}-${campaignSlug}`);
  return {
    help: false,
    provider,
    sender,
    prospectNames,
    campaignDate,
    force,
    includeWatchlist,
    queuePath: path.resolve(singleLine(values.queue || DEFAULT_QUEUE, "queue", 1000)),
    checklistPath: path.resolve(singleLine(values.checklist || DEFAULT_CHECKLIST, "checklist", 1000)),
    outputDir: path.resolve(singleLine(values.output || defaultOutput, "output", 1000)),
  };
}

function messageSubject(prospect) {
  if (prospect.status === "permission-required") return "Permission request: pre-live strategy evidence checklist";
  if (prospect.status.includes("activity-check")) return "Current program and independent pre-live clinic";
  return "Partnership evaluation: independent strategy readiness review";
}

function buildInitialMessage(prospect, config, earliestFollowUpDate) {
  return `# Strategy Assurance outreach draft

**Status:** DRAFT — HUMAN REVIEW REQUIRED — NOT SENT
**Organization:** ${markdownText(prospect.organization)}
**Official route:** [${markdownText(prospect.officialRoute.label)}](${prospect.officialRoute.url})
**Route rule:** ${markdownText(prospect.officialRoute.rule)}
**Prepared:** ${config.campaignDate}
**Earliest follow-up:** ${earliestFollowUpDate}

## Subject

${messageSubject(prospect)}

## Message

Hello ${markdownText(prospect.organization)} team,

I am reaching out on behalf of ${markdownText(config.provider)}. We provide a fixed five-business-day engineering review of one automated trading strategy, one exchange account, and one deployment. The review reconciles exchange economics, tests strategy evidence and production controls, and ends with an operate, restrict, paper-only, or stop-pending-remediation memo. It is not investment advice, a certification, or a return guarantee.

The possible fit, based only on your public materials, is this: ${markdownText(prospect.hypothesis)}

${markdownText(prospect.firstAsk)}

If useful, I can share a one-page pre-live evidence checklist and the exact fixed scope. If this is not relevant, no response is needed.

Regards,
${markdownText(config.sender)}
${markdownText(config.provider)}

## Human review before sending

- Re-open the official route and confirm its rules and current activity.
- Confirm this organization and channel permit the proposed contact.
- Remove or rewrite any statement that is not supported by the recipient's current public material.
- Check applicable privacy, marketing, sanctions, and professional-services requirements.
- Send only through the official organization route; never solicit individual community members.

No message has been sent by this command.
`;
}

function buildFollowUpMessage(prospect, config, earliestFollowUpDate) {
  return `# Strategy Assurance single follow-up draft

**Status:** DRAFT — DO NOT SEND BEFORE ${earliestFollowUpDate}
**Organization:** ${markdownText(prospect.organization)}
**Initial draft date:** ${config.campaignDate}
**Maximum follow-ups:** 1

## Subject

Following up: independent strategy readiness review

## Message

Hello ${markdownText(prospect.organization)} team,

I am following up once on my organization-level note dated ${config.campaignDate}. The proposed review is limited to an operator's own strategy, exchange evidence, production controls, and deployment; it does not provide signals, custody, account operation, certification, or return claims.

I will not contact users or community members, share promotional material in a community, or represent an affiliation without approval. If this is not relevant, no action is needed and I will close the request.

Regards,
${markdownText(config.sender)}
${markdownText(config.provider)}

No follow-up has been sent by this command.
`;
}

export function buildOutreachCampaign(queueValue, checklistText, config) {
  const queue = validateProspectQueue(queueValue);
  if (typeof checklistText !== "string" || !checklistText.trim() || checklistText.length > 100_000) {
    throw new Error("pre-live checklist must be a non-empty text file no larger than 100000 characters");
  }
  const selected = config.prospectNames.map((requestedName) => {
    const key = requestedName.toLocaleLowerCase("en-US");
    const prospect = queue.prospects.find((candidate) => candidate.organization.toLocaleLowerCase("en-US") === key);
    if (!prospect) throw new Error(`unknown prospect organization: ${requestedName}`);
    if (prospect.priority === "watchlist" && !config.includeWatchlist) {
      throw new Error(`${prospect.organization} is a watchlist prospect; use --include-watchlist only after reviewing channel overlap`);
    }
    return prospect;
  });
  const campaignId = `strategy-assurance-outreach-${config.campaignDate}-${selected.map((prospect) => slug(prospect.organization, 36)).join("-")}`;
  const earliestFollowUpDate = addBusinessDays(config.campaignDate, 5);
  const files = {
    "pre-live-checklist.md": checklistText.endsWith("\n") ? checklistText : `${checklistText}\n`,
  };
  const records = [];

  for (const prospect of selected) {
    const directory = slug(prospect.organization);
    const initialPath = path.posix.join(directory, "initial.md");
    const followUpPath = path.posix.join(directory, "follow-up.md");
    const recordPath = path.posix.join(directory, "record.json");
    const initial = buildInitialMessage(prospect, config, earliestFollowUpDate);
    const followUp = buildFollowUpMessage(prospect, config, earliestFollowUpDate);
    const record = {
      schemaVersion: 1,
      outreachRecordType: "strategy-assurance-organization",
      status: "prepared",
      campaignId,
      organization: prospect.organization,
      queueRank: prospect.rank,
      queueKind: prospect.kind,
      queuePriority: prospect.priority,
      queueStatus: prospect.status,
      queueResearchedAt: queue.researchedAt,
      campaignDate: config.campaignDate,
      earliestFollowUpDate,
      maximumFollowUps: 1,
      provider: config.provider,
      sender: config.sender,
      officialRoute: prospect.officialRoute,
      sourceEvidence: prospect.evidence,
      fitHypothesis: prospect.hypothesis,
      likelyObjection: prospect.likelyObjection,
      manualNextAction: prospect.manualNextAction,
      messageFiles: {
        initial: initialPath,
        initialSha256: sha256(initial),
        followUp: followUpPath,
        followUpSha256: sha256(followUp),
        checklist: "pre-live-checklist.md",
        checklistSha256: sha256(files["pre-live-checklist.md"]),
      },
      manualOutcome: {
        sentAt: null,
        sentVia: null,
        responseStatus: null,
        responseAt: null,
        notes: null,
      },
      externalActionsPerformed: false,
    };
    files[initialPath] = initial;
    files[followUpPath] = followUp;
    files[recordPath] = `${JSON.stringify(record, null, 2)}\n`;
    records.push({
      organization: prospect.organization,
      directory,
      status: "prepared",
      officialRouteUrl: prospect.officialRoute.url,
      earliestFollowUpDate,
      record: recordPath,
    });
  }

  const campaign = {
    schemaVersion: 1,
    campaignType: "strategy-assurance-organization-outreach",
    campaignId,
    status: "prepared",
    campaignDate: config.campaignDate,
    queueResearchedAt: queue.researchedAt,
    provider: config.provider,
    sender: config.sender,
    prospectOrganizations: selected.map((prospect) => prospect.organization),
    limits: {
      maximumProspects: MAX_PROSPECTS,
      selectedProspects: selected.length,
      minimumBusinessDaysBeforeFollowUp: 5,
      maximumFollowUpsPerProspect: 1,
    },
    prospects: records,
    actionsNotPerformed: [
      "message-send",
      "form-submission",
      "community-join",
      "affiliation-claim",
      "member-data-collection",
      "pipeline-transition",
    ],
    externalActionsPerformed: false,
  };
  files["campaign.json"] = `${JSON.stringify(campaign, null, 2)}\n`;
  return { campaign, files };
}

export async function writeOutreachCampaign(queueValue, checklistText, config) {
  const built = buildOutreachCampaign(queueValue, checklistText, config);
  let outputExists = false;
  try {
    await access(config.outputDir);
    outputExists = true;
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  if (outputExists && !config.force) {
    throw new Error(`Output directory already exists: ${config.outputDir}. Use --force to replace this campaign's generated files.`);
  }
  if (outputExists && config.force) {
    try {
      const existing = JSON.parse(await readFile(path.join(config.outputDir, "campaign.json"), "utf8"));
      if (existing.campaignId !== built.campaign.campaignId) {
        throw new Error("--force may replace only files belonging to the same campaignId");
      }
    } catch (error) {
      if (error?.code === "ENOENT") {
        throw new Error("--force requires an existing campaign.json from the same generated campaign");
      }
      throw error;
    }
  }

  await mkdir(config.outputDir, { recursive: true });
  const fileNames = Object.keys(built.files).sort();
  for (const fileName of fileNames) {
    const target = path.join(config.outputDir, ...fileName.split("/"));
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, built.files[fileName], { encoding: "utf8", flag: config.force ? "w" : "wx" });
  }
  return { outputDir: config.outputDir, files: fileNames, campaign: built.campaign };
}

async function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    if (config.help) {
      process.stdout.write(usage());
      return;
    }
    const queue = JSON.parse(await readFile(config.queuePath, "utf8"));
    const checklist = await readFile(config.checklistPath, "utf8");
    const result = await writeOutreachCampaign(queue, checklist, config);
    process.stdout.write(
      `${JSON.stringify({ outputDir: result.outputDir, campaignId: result.campaign.campaignId, files: result.files }, null, 2)}\n`,
    );
  } catch (error) {
    process.stderr.write(`Strategy Assurance outreach preparation failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
