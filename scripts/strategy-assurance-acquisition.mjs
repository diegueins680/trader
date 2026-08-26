#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { addBusinessDays } from "./generate-strategy-assurance-outreach.mjs";

export const ACQUISITION_STATUSES = [
  "prepared",
  "contacted",
  "follow-up-sent",
  "responded",
  "qualified",
  "proposed",
  "disqualified",
  "closed-no-response",
];

const STATUS_SET = new Set(ACQUISITION_STATUSES);
const TRANSITIONS = {
  prepared: new Set(["contacted", "disqualified"]),
  contacted: new Set(["follow-up-sent", "responded"]),
  "follow-up-sent": new Set(["responded", "closed-no-response"]),
  responded: new Set(["qualified", "disqualified"]),
  qualified: new Set(["proposed", "disqualified"]),
  proposed: new Set(),
  disqualified: new Set(),
  "closed-no-response": new Set(),
};

const REQUIRED_NON_ACTIONS = [
  "message-send",
  "form-submission",
  "community-join",
  "affiliation-claim",
  "member-data-collection",
  "pipeline-transition",
];
const DEFAULT_REGISTRY = path.join(".tmp", "strategy-assurance", "acquisition.json");

function usage() {
  return `Track Strategy Assurance acquisition evidence locally.

Usage:
  npm run assurance:acquisition -- import --campaign PATH [--registry PATH] [--at YYYY-MM-DD]
  npm run assurance:acquisition -- advance --id ID --status STATUS --evidence TEXT [event options]
  npm run assurance:acquisition -- summary [--as-of YYYY-MM-DD] [--registry PATH] [--json]

Commands:
  import     Validate and import a generated outreach campaign package.
  advance    Record one real, evidenced forward lifecycle transition.
  summary    Report exact acquisition conversions, source performance, and next actions.

Statuses and allowed transitions:
  prepared -> contacted | disqualified
  contacted -> follow-up-sent | responded
  follow-up-sent -> responded | closed-no-response
  responded -> qualified | disqualified
  qualified -> proposed | disqualified

Options:
  --registry PATH        Default: ${DEFAULT_REGISTRY}
  --campaign PATH        Generated campaign.json to import
  --id ID                Stable lead id to advance
  --status STATUS        Destination lifecycle status
  --evidence TEXT        Required transition evidence reference; do not paste message bodies or secrets
  --channel TEXT         Required for contacted and follow-up-sent
  --engagement PATH      Required for proposed; generated engagement.json
  --at YYYY-MM-DD        Event/import date; default: current UTC date
  --as-of YYYY-MM-DD     Summary date; default: current UTC date
  --json                 Render summary as JSON instead of Markdown
  --help

The command records evidence only. It never sends outreach, submits a form, joins a
community, claims an affiliation, imports a proposal into the commercial pipeline,
or changes external state.
`;
}

function utcDate(timestampMs) {
  return new Date(timestampMs).toISOString().slice(0, 10);
}

function isoDate(value, label) {
  const text = String(value ?? "").trim();
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
  if (!match) throw new Error(`${label} must use YYYY-MM-DD`);
  const timestamp = Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  if (new Date(timestamp).toISOString().slice(0, 10) !== text) throw new Error(`${label} must be a real calendar date`);
  return text;
}

function singleLine(value, label, maximum = 500) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a non-empty string`);
  const text = value.trim();
  if (text.length > maximum) throw new Error(`${label} must be at most ${maximum} characters`);
  if (/[\u0000-\u001f\u007f]/.test(text)) throw new Error(`${label} cannot contain control characters or newlines`);
  return text;
}

function objectValue(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function digestObject(value) {
  return sha256(JSON.stringify(value));
}

function sha256Value(value, label) {
  const text = singleLine(value, label, 64);
  if (!/^[a-f0-9]{64}$/.test(text)) throw new Error(`${label} must be SHA-256 hex`);
  return text;
}

function httpsUrl(value, label) {
  const text = singleLine(value, label, 2000);
  let parsed;
  try {
    parsed = new URL(text);
  } catch {
    throw new Error(`${label} must be a valid URL`);
  }
  if (parsed.protocol !== "https:") throw new Error(`${label} must use HTTPS`);
  return text;
}

function relativeFile(value, label) {
  const text = singleLine(value, label, 500);
  if (text.includes("\\") || path.posix.isAbsolute(text) || path.posix.normalize(text) !== text || text.startsWith("../")) {
    throw new Error(`${label} must be a normalized relative path`);
  }
  return text;
}

function safeId(value, label, maximum = 240) {
  const text = singleLine(value, label, maximum);
  if (!/^[a-z0-9][a-z0-9-]+$/.test(text)) throw new Error(`${label} must use lowercase letters, digits, and hyphens`);
  return text;
}

function validateRoute(value, label) {
  const route = objectValue(value, label);
  return {
    label: singleLine(route.label, `${label}.label`, 240),
    url: httpsUrl(route.url, `${label}.url`),
    rule: singleLine(route.rule, `${label}.rule`, 2000),
  };
}

function validateEvidenceList(value, label) {
  if (!Array.isArray(value) || value.length === 0 || value.length > 20) {
    throw new Error(`${label} must contain from 1 through 20 entries`);
  }
  return value.map((item, index) => {
    const evidence = objectValue(item, `${label}[${index}]`);
    return {
      claim: singleLine(evidence.claim, `${label}[${index}].claim`, 2000),
      url: httpsUrl(evidence.url, `${label}[${index}].url`),
      checkedAt: isoDate(evidence.checkedAt, `${label}[${index}].checkedAt`),
    };
  });
}

function campaignHeader(value) {
  const campaign = objectValue(value, "campaign");
  if (campaign.schemaVersion !== 1) throw new Error("campaign schemaVersion must be 1");
  if (campaign.campaignType !== "strategy-assurance-organization-outreach") {
    throw new Error("campaignType must be strategy-assurance-organization-outreach");
  }
  if (campaign.status !== "prepared") throw new Error("campaign status must be prepared");
  if (campaign.externalActionsPerformed !== false) throw new Error("campaign must state that no external action was performed");
  const actions = Array.isArray(campaign.actionsNotPerformed) ? campaign.actionsNotPerformed : [];
  for (const action of REQUIRED_NON_ACTIONS) {
    if (!actions.includes(action)) throw new Error(`campaign actionsNotPerformed must include ${action}`);
  }
  if (!Array.isArray(campaign.prospects) || campaign.prospects.length === 0 || campaign.prospects.length > 3) {
    throw new Error("campaign must contain from 1 through 3 prospects");
  }
  if (!Array.isArray(campaign.prospectOrganizations) || campaign.prospectOrganizations.length !== campaign.prospects.length) {
    throw new Error("campaign prospectOrganizations must align with prospects");
  }
  const campaignDate = isoDate(campaign.campaignDate, "campaignDate");
  const queueResearchedAt = isoDate(campaign.queueResearchedAt, "queueResearchedAt");
  if (queueResearchedAt > campaignDate) throw new Error("queueResearchedAt must not follow campaignDate");
  return {
    source: campaign,
    campaignId: safeId(campaign.campaignId, "campaignId"),
    campaignDate,
    queueResearchedAt,
    provider: singleLine(campaign.provider, "campaign.provider", 240),
    sender: singleLine(campaign.sender, "campaign.sender", 240),
    prospectOrganizations: campaign.prospectOrganizations.map((item, index) =>
      singleLine(item, `campaign.prospectOrganizations[${index}]`, 120),
    ),
    prospects: campaign.prospects,
  };
}

function messageFiles(value, label) {
  const files = objectValue(value, label);
  return {
    initial: relativeFile(files.initial, `${label}.initial`),
    initialSha256: sha256Value(files.initialSha256, `${label}.initialSha256`),
    followUp: relativeFile(files.followUp, `${label}.followUp`),
    followUpSha256: sha256Value(files.followUpSha256, `${label}.followUpSha256`),
    checklist: relativeFile(files.checklist, `${label}.checklist`),
    checklistSha256: sha256Value(files.checklistSha256, `${label}.checklistSha256`),
  };
}

function packageFile(files, fileName, digest, label) {
  if (!Object.prototype.hasOwnProperty.call(files, fileName) || typeof files[fileName] !== "string") {
    throw new Error(`${label} is missing package file: ${fileName}`);
  }
  if (sha256(files[fileName]) !== digest) throw new Error(`${label} digest does not match package file: ${fileName}`);
}

export function validateOutreachPackage(campaignValue, filesValue) {
  const header = campaignHeader(campaignValue);
  const files = objectValue(filesValue, "campaign files");
  const organizationKeys = new Set();
  const records = header.prospects.map((summaryValue, index) => {
    const label = `campaign.prospects[${index}]`;
    const summary = objectValue(summaryValue, label);
    const organization = singleLine(summary.organization, `${label}.organization`, 120);
    if (organization !== header.prospectOrganizations[index]) throw new Error(`${label} does not align with prospectOrganizations`);
    const organizationKey = organization.toLocaleLowerCase("en-US");
    if (organizationKeys.has(organizationKey)) throw new Error(`duplicate campaign organization: ${organization}`);
    organizationKeys.add(organizationKey);
    if (summary.status !== "prepared") throw new Error(`${label}.status must be prepared`);
    const recordPath = relativeFile(summary.record, `${label}.record`);
    if (typeof files[recordPath] !== "string") throw new Error(`campaign package is missing record: ${recordPath}`);
    let recordValue;
    try {
      recordValue = JSON.parse(files[recordPath]);
    } catch {
      throw new Error(`campaign record is not valid JSON: ${recordPath}`);
    }
    const record = objectValue(recordValue, recordPath);
    if (record.schemaVersion !== 1 || record.outreachRecordType !== "strategy-assurance-organization") {
      throw new Error(`${recordPath} has an unsupported outreach record schema`);
    }
    if (record.status !== "prepared" || record.externalActionsPerformed !== false) {
      throw new Error(`${recordPath} must remain prepared with no external action`);
    }
    if (record.campaignId !== header.campaignId || record.organization !== organization) {
      throw new Error(`${recordPath} identity does not match the campaign`);
    }
    if (record.provider !== header.provider || record.sender !== header.sender) {
      throw new Error(`${recordPath} provider or sender does not match the campaign`);
    }
    const campaignDate = isoDate(record.campaignDate, `${recordPath}.campaignDate`);
    if (campaignDate !== header.campaignDate) throw new Error(`${recordPath}.campaignDate does not match the campaign`);
    const queueResearchedAt = isoDate(record.queueResearchedAt, `${recordPath}.queueResearchedAt`);
    if (queueResearchedAt !== header.queueResearchedAt) throw new Error(`${recordPath}.queueResearchedAt does not match`);
    const earliestFollowUpDate = isoDate(record.earliestFollowUpDate, `${recordPath}.earliestFollowUpDate`);
    if (earliestFollowUpDate !== addBusinessDays(header.campaignDate, 5)) {
      throw new Error(`${recordPath}.earliestFollowUpDate must be five business days after campaignDate`);
    }
    if (summary.earliestFollowUpDate !== earliestFollowUpDate) throw new Error(`${label}.earliestFollowUpDate does not match record`);
    if (record.maximumFollowUps !== 1) throw new Error(`${recordPath}.maximumFollowUps must be 1`);
    const route = validateRoute(record.officialRoute, `${recordPath}.officialRoute`);
    if (summary.officialRouteUrl !== route.url) throw new Error(`${label}.officialRouteUrl does not match record`);
    const sourceEvidence = validateEvidenceList(record.sourceEvidence, `${recordPath}.sourceEvidence`);
    if (!Number.isSafeInteger(record.queueRank) || record.queueRank <= 0) {
      throw new Error(`${recordPath}.queueRank must be a positive integer`);
    }
    const outcome = objectValue(record.manualOutcome, `${recordPath}.manualOutcome`);
    for (const field of ["sentAt", "sentVia", "responseStatus", "responseAt", "notes"]) {
      if (outcome[field] !== null) throw new Error(`${recordPath}.manualOutcome.${field} must be null at import`);
    }
    const messages = messageFiles(record.messageFiles, `${recordPath}.messageFiles`);
    packageFile(files, messages.initial, messages.initialSha256, recordPath);
    packageFile(files, messages.followUp, messages.followUpSha256, recordPath);
    packageFile(files, messages.checklist, messages.checklistSha256, recordPath);
    return {
      source: record,
      recordPath,
      organization,
      queueRank: record.queueRank,
      queueKind:
        typeof record.queueKind === "string" && record.queueKind.trim()
          ? singleLine(record.queueKind, `${recordPath}.queueKind`, 120)
          : "unknown",
      queuePriority: singleLine(record.queuePriority, `${recordPath}.queuePriority`, 40),
      queueStatus: singleLine(record.queueStatus, `${recordPath}.queueStatus`, 120),
      campaignDate,
      queueResearchedAt,
      earliestFollowUpDate,
      provider: header.provider,
      sender: header.sender,
      route,
      sourceEvidence,
      fitHypothesis: singleLine(record.fitHypothesis, `${recordPath}.fitHypothesis`, 2000),
      likelyObjection: singleLine(record.likelyObjection, `${recordPath}.likelyObjection`, 2000),
      manualNextAction: singleLine(record.manualNextAction, `${recordPath}.manualNextAction`, 2000),
    };
  });
  return {
    campaign: header,
    records,
    packageDigest: digestObject({ campaign: header.source, records: records.map((record) => record.source) }),
  };
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if (error instanceof SyntaxError) throw new Error(`${label} is not valid JSON: ${filePath}`);
    throw error;
  }
}

export async function loadOutreachPackage(campaignPath) {
  const campaign = await readJson(campaignPath, "campaign");
  const header = campaignHeader(campaign);
  const root = path.dirname(campaignPath);
  const files = {};
  for (let index = 0; index < header.prospects.length; index += 1) {
    const summary = objectValue(header.prospects[index], `campaign.prospects[${index}]`);
    const recordPath = relativeFile(summary.record, `campaign.prospects[${index}].record`);
    files[recordPath] = await readFile(path.join(root, ...recordPath.split("/")), "utf8");
    let record;
    try {
      record = JSON.parse(files[recordPath]);
    } catch {
      throw new Error(`campaign record is not valid JSON: ${recordPath}`);
    }
    const messages = messageFiles(objectValue(record, recordPath).messageFiles, `${recordPath}.messageFiles`);
    for (const fileName of [messages.initial, messages.followUp, messages.checklist]) {
      if (!Object.prototype.hasOwnProperty.call(files, fileName)) {
        files[fileName] = await readFile(path.join(root, ...fileName.split("/")), "utf8");
      }
    }
  }
  return { campaign, files, validated: validateOutreachPackage(campaign, files) };
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
    if (!new Set(["registry", "campaign", "id", "status", "evidence", "channel", "engagement", "at", "asOf"]).has(name)) {
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
  const registryPath = path.resolve(singleLine(options.registry || DEFAULT_REGISTRY, "--registry", 1000));
  const today = utcDate(nowMs);
  if (command === "import") {
    if (options.json || options.id || options.status || options.evidence || options.channel || options.engagement || options.asOf) {
      throw new Error("import received an option for another command");
    }
    if (!options.campaign) throw new Error("--campaign is required for import");
    return {
      help: false,
      command,
      registryPath,
      campaignPath: path.resolve(singleLine(options.campaign, "--campaign", 1000)),
      at: isoDate(options.at || today, "--at"),
    };
  }
  if (command === "advance") {
    if (options.json || options.campaign || options.asOf) throw new Error("advance received an option for another command");
    const status = singleLine(options.status, "--status", 40);
    if (!STATUS_SET.has(status) || status === "prepared") {
      throw new Error(`--status must be one of: ${ACQUISITION_STATUSES.filter((item) => item !== "prepared").join(", ")}`);
    }
    const channelRequired = status === "contacted" || status === "follow-up-sent";
    const engagementRequired = status === "proposed";
    if (channelRequired && !options.channel) throw new Error(`--channel is required when advancing to ${status}`);
    if (!channelRequired && options.channel) throw new Error(`--channel is not valid when advancing to ${status}`);
    if (engagementRequired && !options.engagement) throw new Error("--engagement is required when advancing to proposed");
    if (!engagementRequired && options.engagement) throw new Error(`--engagement is not valid when advancing to ${status}`);
    return {
      help: false,
      command,
      registryPath,
      id: safeId(options.id, "--id"),
      status,
      evidence: singleLine(options.evidence, "--evidence", 500),
      channel: channelRequired ? singleLine(options.channel, "--channel", 240) : null,
      engagementPath: engagementRequired ? path.resolve(singleLine(options.engagement, "--engagement", 1000)) : null,
      at: isoDate(options.at || today, "--at"),
    };
  }
  if (options.campaign || options.id || options.status || options.evidence || options.channel || options.engagement || options.at) {
    throw new Error("summary received an option for another command");
  }
  return {
    help: false,
    command,
    registryPath,
    asOf: isoDate(options.asOf || today, "--as-of"),
    json: options.json,
  };
}

function validateGeneratedEngagement(value) {
  const engagement = objectValue(value, "engagement");
  if (engagement.schemaVersion !== 1 || engagement.engagementType !== "strategy-assurance-standard") {
    throw new Error("engagement must be a schemaVersion 1 strategy-assurance-standard record");
  }
  if (engagement.status !== "proposal") throw new Error("engagement status must be proposal");
  const commercials = objectValue(engagement.commercials, "engagement.commercials");
  return {
    engagementId: safeId(engagement.engagementId, "engagement.engagementId"),
    client: singleLine(engagement.client, "engagement.client", 300),
    provider: singleLine(engagement.provider, "engagement.provider", 300),
    proposalDate: isoDate(commercials.proposalDate, "engagement.commercials.proposalDate"),
    sourceDigest: digestObject(engagement),
  };
}

export function emptyAcquisitionRegistry() {
  return { schemaVersion: 1, updatedAt: null, leads: [] };
}

function normalizeTransitionEvidence(status, detailsValue, label = "transition") {
  const details = objectValue(detailsValue, `${label} details`);
  const evidence = singleLine(details.evidence, `${label}.evidence`, 500);
  const hasChannel = details.channel != null;
  const hasEngagement = details.engagement != null;
  if (status === "contacted" || status === "follow-up-sent") {
    if (!hasChannel) throw new Error(`${label} requires channel for ${status}`);
    if (hasEngagement) throw new Error(`${label} cannot include engagement for ${status}`);
    return { evidence, channel: singleLine(details.channel, `${label}.channel`, 240) };
  }
  if (status === "proposed") {
    if (!hasEngagement) throw new Error(`${label} requires engagement for proposed`);
    if (hasChannel) throw new Error(`${label} cannot include channel for proposed`);
    const engagement = validateGeneratedEngagement(details.engagement);
    return {
      evidence,
      engagementId: engagement.engagementId,
      engagementClient: engagement.client,
      engagementProvider: engagement.provider,
      engagementProposalDate: engagement.proposalDate,
      engagementSourceDigest: engagement.sourceDigest,
    };
  }
  if (hasChannel || hasEngagement) throw new Error(`${label} cannot include channel or engagement for ${status}`);
  return { evidence };
}

function validateEventEvidence(status, event, label) {
  if (status === "prepared") {
    return { packageDigest: sha256Value(event.packageDigest, `${label}.packageDigest`) };
  }
  if (status === "proposed") {
    return {
      evidence: singleLine(event.evidence, `${label}.evidence`, 500),
      engagementId: safeId(event.engagementId, `${label}.engagementId`),
      engagementClient: singleLine(event.engagementClient, `${label}.engagementClient`, 300),
      engagementProvider: singleLine(event.engagementProvider, `${label}.engagementProvider`, 300),
      engagementProposalDate: isoDate(event.engagementProposalDate, `${label}.engagementProposalDate`),
      engagementSourceDigest: sha256Value(event.engagementSourceDigest, `${label}.engagementSourceDigest`),
    };
  }
  const details = {
    evidence: event.evidence,
    ...(Object.prototype.hasOwnProperty.call(event, "channel") ? { channel: event.channel } : {}),
  };
  return normalizeTransitionEvidence(status, details, label);
}

export function validateAcquisitionRegistry(value) {
  const registry = objectValue(value, "acquisition registry");
  if (registry.schemaVersion !== 1) throw new Error("acquisition registry schemaVersion must be 1");
  if (registry.updatedAt !== null) isoDate(registry.updatedAt, "registry.updatedAt");
  if (!Array.isArray(registry.leads)) throw new Error("registry.leads must be an array");
  const ids = new Set();
  const organizations = new Set();
  const engagementIds = new Set();
  for (const recordValue of registry.leads) {
    const record = objectValue(recordValue, "lead");
    const id = safeId(record.id, "lead.id");
    if (ids.has(id)) throw new Error(`duplicate lead id: ${id}`);
    ids.add(id);
    const organization = singleLine(record.organization, `${id}.organization`, 120);
    const organizationKey = organization.toLocaleLowerCase("en-US");
    if (organizations.has(organizationKey)) throw new Error(`duplicate acquisition organization: ${organization}`);
    organizations.add(organizationKey);
    safeId(record.campaignId, `${id}.campaignId`);
    singleLine(record.provider, `${id}.provider`, 240);
    singleLine(record.sender, `${id}.sender`, 240);
    if (!Number.isSafeInteger(record.queueRank) || record.queueRank <= 0) throw new Error(`${id}.queueRank must be positive`);
    singleLine(record.queueKind, `${id}.queueKind`, 120);
    singleLine(record.queuePriority, `${id}.queuePriority`, 40);
    singleLine(record.queueStatus, `${id}.queueStatus`, 120);
    const campaignDate = isoDate(record.campaignDate, `${id}.campaignDate`);
    const queueResearchedAt = isoDate(record.queueResearchedAt, `${id}.queueResearchedAt`);
    if (queueResearchedAt > campaignDate) throw new Error(`${id}.queueResearchedAt follows campaignDate`);
    const earliestFollowUpDate = isoDate(record.earliestFollowUpDate, `${id}.earliestFollowUpDate`);
    if (earliestFollowUpDate !== addBusinessDays(campaignDate, 5)) throw new Error(`${id}.earliestFollowUpDate is inconsistent`);
    validateRoute(record.officialRoute, `${id}.officialRoute`);
    validateEvidenceList(record.sourceEvidence, `${id}.sourceEvidence`);
    singleLine(record.fitHypothesis, `${id}.fitHypothesis`, 2000);
    singleLine(record.likelyObjection, `${id}.likelyObjection`, 2000);
    singleLine(record.manualNextAction, `${id}.manualNextAction`, 2000);
    const importedAt = isoDate(record.importedAt, `${id}.importedAt`);
    if (importedAt < campaignDate) throw new Error(`${id}.importedAt precedes campaignDate`);
    sha256Value(record.sourceDigest, `${id}.sourceDigest`);
    if (registry.updatedAt == null || registry.updatedAt < importedAt) throw new Error(`registry.updatedAt must include ${id} import`);
    if (!Array.isArray(record.events) || record.events.length === 0) throw new Error(`${id}.events must be non-empty`);
    let previousStatus = null;
    let previousAt = "";
    let contactedAt = null;
    let followUpAt = null;
    for (let index = 0; index < record.events.length; index += 1) {
      const event = objectValue(record.events[index], `${id}.events[${index}]`);
      const status = singleLine(event.status, `${id}.events[${index}].status`, 40);
      if (!STATUS_SET.has(status)) throw new Error(`${id} has invalid status: ${status}`);
      const at = isoDate(event.at, `${id}.events[${index}].at`);
      const evidence = validateEventEvidence(status, event, `${id}.${status}`);
      if (registry.updatedAt < at) throw new Error(`registry.updatedAt must include every ${id} event`);
      if (previousAt && at < previousAt) throw new Error(`${id} event dates must be nondecreasing`);
      if (previousStatus == null && status !== "prepared") throw new Error(`${id} must begin at prepared`);
      if (previousStatus != null && !TRANSITIONS[previousStatus].has(status)) {
        throw new Error(`${id} has invalid transition ${previousStatus} -> ${status}`);
      }
      if (status === "prepared" && at !== campaignDate) throw new Error(`${id} prepared date must equal campaignDate`);
      if (status === "contacted") contactedAt = at;
      if (status === "follow-up-sent") {
        const actualEarliestFollowUp = contactedAt == null ? earliestFollowUpDate : [earliestFollowUpDate, addBusinessDays(contactedAt, 5)].sort().at(-1);
        if (at < actualEarliestFollowUp) throw new Error(`${id} follow-up precedes ${actualEarliestFollowUp}`);
        followUpAt = at;
      }
      if (status === "closed-no-response") {
        const earliestClose = followUpAt == null ? null : addBusinessDays(followUpAt, 5);
        if (earliestClose == null || at < earliestClose) throw new Error(`${id} no-response closure is premature`);
      }
      if (status === "proposed") {
        if (evidence.engagementProvider !== record.provider) throw new Error(`${id} proposal provider does not match lead provider`);
        if (at < evidence.engagementProposalDate) throw new Error(`${id} proposal event precedes engagement proposalDate`);
        if (engagementIds.has(evidence.engagementId)) throw new Error(`duplicate proposed engagementId: ${evidence.engagementId}`);
        engagementIds.add(evidence.engagementId);
      }
      previousStatus = status;
      previousAt = at;
    }
    if (record.currentStatus !== previousStatus) throw new Error(`${id}.currentStatus must equal its final event status`);
  }
  return registry;
}

export function importOutreachCampaign(registryValue, packageValue, importedAtValue) {
  const registry = validateAcquisitionRegistry(registryValue);
  const validated = packageValue?.validated ?? validateOutreachPackage(packageValue?.campaign, packageValue?.files);
  const importedAt = isoDate(importedAtValue, "importedAt");
  if (importedAt < validated.campaign.campaignDate) throw new Error("import date precedes campaign date");
  const existingCampaign = registry.leads.filter((lead) => lead.campaignId === validated.campaign.campaignId);
  if (existingCampaign.length > 0) {
    if (existingCampaign.length !== validated.records.length) throw new Error("campaign is only partially present in the registry");
    for (const record of validated.records) {
      const id = `strategy-assurance-lead-${sha256(`${validated.campaign.campaignId}|${record.organization}`).slice(0, 16)}`;
      const existing = existingCampaign.find((lead) => lead.id === id);
      const digest = digestObject({ campaignId: validated.campaign.campaignId, record: record.source });
      if (!existing || existing.sourceDigest !== digest) throw new Error("campaign already exists with different source evidence");
    }
    return { registry, changed: false, leads: existingCampaign };
  }
  const existingOrganizations = new Set(registry.leads.map((lead) => lead.organization.toLocaleLowerCase("en-US")));
  for (const record of validated.records) {
    if (existingOrganizations.has(record.organization.toLocaleLowerCase("en-US"))) {
      throw new Error(`organization already exists in acquisition registry: ${record.organization}`);
    }
  }
  const leads = validated.records.map((record) => {
    const id = `strategy-assurance-lead-${sha256(`${validated.campaign.campaignId}|${record.organization}`).slice(0, 16)}`;
    return {
      id,
      campaignId: validated.campaign.campaignId,
      organization: record.organization,
      provider: record.provider,
      sender: record.sender,
      queueRank: record.queueRank,
      queueKind: record.queueKind,
      queuePriority: record.queuePriority,
      queueStatus: record.queueStatus,
      queueResearchedAt: record.queueResearchedAt,
      campaignDate: record.campaignDate,
      earliestFollowUpDate: record.earliestFollowUpDate,
      officialRoute: record.route,
      sourceEvidence: record.sourceEvidence,
      fitHypothesis: record.fitHypothesis,
      likelyObjection: record.likelyObjection,
      manualNextAction: record.manualNextAction,
      importedAt,
      sourceDigest: digestObject({ campaignId: validated.campaign.campaignId, record: record.source }),
      currentStatus: "prepared",
      events: [{ status: "prepared", at: record.campaignDate, packageDigest: validated.packageDigest }],
    };
  });
  const next = {
    schemaVersion: 1,
    updatedAt: registry.updatedAt == null || registry.updatedAt < importedAt ? importedAt : registry.updatedAt,
    leads: [...registry.leads, ...leads].sort((left, right) => left.id.localeCompare(right.id)),
  };
  validateAcquisitionRegistry(next);
  return { registry: next, changed: true, leads };
}

function sameEvidence(event, at, evidence) {
  return event.at === at && Object.entries(evidence).every(([key, value]) => event[key] === value);
}

function actualEarliestFollowUp(lead) {
  const contacted = lead.events.find((event) => event.status === "contacted");
  if (contacted == null) return lead.earliestFollowUpDate;
  return [lead.earliestFollowUpDate, addBusinessDays(contacted.at, 5)].sort().at(-1);
}

export function advanceLead(registryValue, idValue, statusValue, atValue, detailsValue) {
  const registry = validateAcquisitionRegistry(registryValue);
  const id = safeId(idValue, "id");
  const status = singleLine(statusValue, "status", 40);
  if (!STATUS_SET.has(status) || status === "prepared") throw new Error(`invalid destination status: ${status}`);
  const at = isoDate(atValue, "at");
  const evidence = normalizeTransitionEvidence(status, detailsValue);
  const index = registry.leads.findIndex((lead) => lead.id === id);
  if (index < 0) throw new Error(`unknown lead id: ${id}`);
  const current = registry.leads[index];
  if (current.currentStatus === status) {
    const lastEvent = current.events[current.events.length - 1];
    if (!sameEvidence(lastEvent, at, evidence)) throw new Error(`lead ${id} is already ${status} with different evidence or date`);
    return { registry, changed: false, lead: current };
  }
  if (!TRANSITIONS[current.currentStatus].has(status)) throw new Error(`invalid transition ${current.currentStatus} -> ${status}`);
  const lastAt = current.events[current.events.length - 1].at;
  if (at < lastAt) throw new Error(`transition date ${at} precedes latest event ${lastAt}`);
  if (status === "follow-up-sent" && at < actualEarliestFollowUp(current)) {
    throw new Error(`follow-up cannot be recorded before ${actualEarliestFollowUp(current)}`);
  }
  if (status === "closed-no-response") {
    const followUp = current.events.find((event) => event.status === "follow-up-sent");
    const earliestClose = addBusinessDays(followUp.at, 5);
    if (at < earliestClose) throw new Error(`closed-no-response cannot be recorded before ${earliestClose}`);
  }
  if (status === "proposed") {
    if (evidence.engagementProvider !== current.provider) throw new Error("engagement provider does not match lead provider");
    if (at < evidence.engagementProposalDate) throw new Error("proposal event precedes engagement proposalDate");
    const duplicate = registry.leads.some((lead) =>
      lead.events.some((event) => event.status === "proposed" && event.engagementId === evidence.engagementId),
    );
    if (duplicate) throw new Error(`engagement is already linked to another lead: ${evidence.engagementId}`);
  }
  const updated = {
    ...current,
    currentStatus: status,
    events: [...current.events, { status, at, ...evidence }],
  };
  const leads = [...registry.leads];
  leads[index] = updated;
  const next = {
    schemaVersion: 1,
    updatedAt: registry.updatedAt == null || registry.updatedAt < at ? at : registry.updatedAt,
    leads,
  };
  validateAcquisitionRegistry(next);
  return { registry: next, changed: true, lead: updated };
}

function ratio(numerator, denominator) {
  return denominator === 0 ? null : numerator / denominator;
}

function hasStatus(lead, status) {
  return lead.events.some((event) => event.status === status);
}

function sourceSummary(leads) {
  const grouped = new Map();
  for (const lead of leads) {
    const current = grouped.get(lead.queueKind) ?? {
      kind: lead.queueKind,
      prepared: 0,
      contacted: 0,
      responded: 0,
      qualified: 0,
      proposed: 0,
    };
    current.prepared += 1;
    for (const status of ["contacted", "responded", "qualified", "proposed"]) {
      if (hasStatus(lead, status)) current[status] += 1;
    }
    grouped.set(lead.queueKind, current);
  }
  return [...grouped.values()]
    .map((entry) => ({
      ...entry,
      contactedToResponded: ratio(entry.responded, entry.contacted),
      contactedToProposed: ratio(entry.proposed, entry.contacted),
    }))
    .sort((left, right) => right.proposed - left.proposed || right.responded - left.responded || left.kind.localeCompare(right.kind));
}

function nextAction(lead, asOf) {
  switch (lead.currentStatus) {
    case "prepared":
      return "Review the official route and draft; record contacted only after a real send or submission";
    case "contacted":
      return asOf < actualEarliestFollowUp(lead)
        ? `Wait for response; follow-up is not eligible before ${actualEarliestFollowUp(lead)}`
        : "Record a response or send the one permitted follow-up manually";
    case "follow-up-sent": {
      const event = lead.events.find((item) => item.status === "follow-up-sent");
      const earliestClose = addBusinessDays(event.at, 5);
      return asOf < earliestClose
        ? `Wait for response; no-response closure is not eligible before ${earliestClose}`
        : "Record a response or close as no response";
    }
    case "responded":
      return "Complete qualification and record qualified or disqualified";
    case "qualified":
      return "Generate and review the commercial kit, then commit the proposal handoff";
    case "proposed":
      return "Reconcile the linked engagement with the commercial pipeline";
    default:
      return null;
  }
}

export function summarizeAcquisition(registryValue, asOfValue) {
  const registry = validateAcquisitionRegistry(registryValue);
  const asOf = isoDate(asOfValue, "asOf");
  if (registry.updatedAt != null && asOf < registry.updatedAt) {
    throw new Error(`asOf ${asOf} precedes latest registry evidence ${registry.updatedAt}`);
  }
  const counts = Object.fromEntries(ACQUISITION_STATUSES.map((status) => [status, 0]));
  const milestones = { contacted: 0, followUpSent: 0, responded: 0, qualified: 0, proposed: 0 };
  const nextActions = [];
  const followUpEligibleLeadIds = [];
  const closeEligibleLeadIds = [];
  for (const lead of registry.leads) {
    counts[lead.currentStatus] += 1;
    if (hasStatus(lead, "contacted")) milestones.contacted += 1;
    if (hasStatus(lead, "follow-up-sent")) milestones.followUpSent += 1;
    if (hasStatus(lead, "responded")) milestones.responded += 1;
    if (hasStatus(lead, "qualified")) milestones.qualified += 1;
    if (hasStatus(lead, "proposed")) milestones.proposed += 1;
    if (lead.currentStatus === "contacted" && asOf >= actualEarliestFollowUp(lead)) followUpEligibleLeadIds.push(lead.id);
    if (lead.currentStatus === "follow-up-sent") {
      const followUp = lead.events.find((event) => event.status === "follow-up-sent");
      if (asOf >= addBusinessDays(followUp.at, 5)) closeEligibleLeadIds.push(lead.id);
    }
    const action = nextAction(lead, asOf);
    if (action) nextActions.push({ id: lead.id, organization: lead.organization, status: lead.currentStatus, action });
  }
  nextActions.sort((left, right) => left.organization.localeCompare(right.organization) || left.id.localeCompare(right.id));
  followUpEligibleLeadIds.sort();
  closeEligibleLeadIds.sort();
  const prepared = registry.leads.length;
  return {
    schemaVersion: 1,
    asOf,
    totalLeads: prepared,
    counts,
    funnel: {
      prepared,
      ...milestones,
      preparedToContacted: ratio(milestones.contacted, prepared),
      contactedToResponded: ratio(milestones.responded, milestones.contacted),
      respondedToQualified: ratio(milestones.qualified, milestones.responded),
      qualifiedToProposed: ratio(milestones.proposed, milestones.qualified),
      contactedToProposed: ratio(milestones.proposed, milestones.contacted),
      followUpToResponded: ratio(
        registry.leads.filter((lead) => hasStatus(lead, "follow-up-sent") && hasStatus(lead, "responded")).length,
        milestones.followUpSent,
      ),
    },
    sources: sourceSummary(registry.leads),
    followUpEligibleLeadIds,
    closeEligibleLeadIds,
    nextActions,
  };
}

function percent(value) {
  return value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;
}

function tableText(value) {
  return String(value).replace(/\|/g, "\\|");
}

export function renderAcquisitionSummary(summary) {
  const stageRows = ACQUISITION_STATUSES.filter((status) => summary.counts[status] > 0).map(
    (status) => `| ${status} | ${summary.counts[status]} |`,
  );
  const sourceRows = summary.sources.map(
    (source) =>
      `| ${tableText(source.kind)} | ${source.prepared} | ${source.contacted} | ${source.responded} | ${source.qualified} | ${source.proposed} | ${percent(source.contactedToResponded)} | ${percent(source.contactedToProposed)} |`,
  );
  const actionRows = summary.nextActions.map(
    (entry) => `| ${tableText(entry.organization)} | ${entry.status} | ${tableText(entry.action)} |`,
  );
  return [
    "# Strategy Assurance Acquisition Pipeline",
    "",
    `As of: ${summary.asOf}`,
    "",
    "## Funnel",
    "",
    `- Prepared → contacted: ${percent(summary.funnel.preparedToContacted)} (${summary.funnel.contacted}/${summary.funnel.prepared})`,
    `- Contacted → responded: ${percent(summary.funnel.contactedToResponded)} (${summary.funnel.responded}/${summary.funnel.contacted})`,
    `- Responded → qualified: ${percent(summary.funnel.respondedToQualified)} (${summary.funnel.qualified}/${summary.funnel.responded})`,
    `- Qualified → proposed: ${percent(summary.funnel.qualifiedToProposed)} (${summary.funnel.proposed}/${summary.funnel.qualified})`,
    `- Contacted → proposed: ${percent(summary.funnel.contactedToProposed)} (${summary.funnel.proposed}/${summary.funnel.contacted})`,
    `- Follow-up → responded: ${percent(summary.funnel.followUpToResponded)}`,
    "",
    "## Current stages",
    "",
    "| Status | Count |",
    "| --- | ---: |",
    ...(stageRows.length > 0 ? stageRows : ["| No leads | 0 |"]),
    "",
    "## Source performance",
    "",
    "| Source kind | Prepared | Contacted | Responded | Qualified | Proposed | Contact→response | Contact→proposal |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...(sourceRows.length > 0 ? sourceRows : ["| No sources | 0 | 0 | 0 | 0 | 0 | N/A | N/A |"]),
    "",
    "## Next actions",
    "",
    "| Organization | Status | Action |",
    "| --- | --- | --- |",
    ...(actionRows.length > 0 ? actionRows : ["| None | — | No active next action |"]),
    "",
  ].join("\n");
}

export async function loadAcquisitionRegistry(filePath, { allowMissing = false } = {}) {
  try {
    return validateAcquisitionRegistry(JSON.parse(await readFile(filePath, "utf8")));
  } catch (error) {
    if (allowMissing && error?.code === "ENOENT") return emptyAcquisitionRegistry();
    if (error instanceof SyntaxError) throw new Error(`acquisition registry is not valid JSON: ${filePath}`);
    throw error;
  }
}

export async function saveAcquisitionRegistry(filePath, registryValue) {
  const registry = validateAcquisitionRegistry(registryValue);
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(registry, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
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
      const registry = await loadAcquisitionRegistry(args.registryPath, { allowMissing: true });
      const packageValue = await loadOutreachPackage(args.campaignPath);
      const result = importOutreachCampaign(registry, packageValue, args.at);
      if (result.changed) await saveAcquisitionRegistry(args.registryPath, result.registry);
      process.stdout.write(
        `${JSON.stringify({ changed: result.changed, campaignId: packageValue.validated.campaign.campaignId, leads: result.leads.map((lead) => ({ id: lead.id, organization: lead.organization, status: lead.currentStatus })) }, null, 2)}\n`,
      );
      return;
    }
    if (args.command === "advance") {
      const registry = await loadAcquisitionRegistry(args.registryPath);
      const engagement = args.engagementPath == null ? null : await readJson(args.engagementPath, "engagement");
      const result = advanceLead(registry, args.id, args.status, args.at, {
        evidence: args.evidence,
        ...(args.channel == null ? {} : { channel: args.channel }),
        ...(engagement == null ? {} : { engagement }),
      });
      if (result.changed) await saveAcquisitionRegistry(args.registryPath, result.registry);
      process.stdout.write(
        `${JSON.stringify({ changed: result.changed, id: result.lead.id, organization: result.lead.organization, status: result.lead.currentStatus }, null, 2)}\n`,
      );
      return;
    }
    const summary = summarizeAcquisition(await loadAcquisitionRegistry(args.registryPath), args.asOf);
    process.stdout.write(args.json ? `${JSON.stringify(summary, null, 2)}\n` : renderAcquisitionSummary(summary));
  } catch (error) {
    process.stderr.write(`Strategy Assurance acquisition failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
