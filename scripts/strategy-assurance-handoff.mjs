#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  advanceLead,
  loadAcquisitionRegistry,
  saveAcquisitionRegistry,
  validateAcquisitionRegistry,
} from "./strategy-assurance-acquisition.mjs";
import {
  importEngagement,
  loadPipeline,
  savePipeline,
  validatePipeline,
} from "./strategy-assurance-pipeline.mjs";

const DEFAULT_REGISTRY = path.join(".tmp", "strategy-assurance", "acquisition.json");
const DEFAULT_PIPELINE = path.join(".tmp", "strategy-assurance", "pipeline.json");

function usage() {
  return `Commit and reconcile Strategy Assurance proposal handoffs locally.

Usage:
  npm run assurance:handoff -- commit --lead ID --engagement PATH --evidence TEXT [options]
  npm run assurance:handoff -- reconcile [--as-of YYYY-MM-DD] [--json] [options]

Commands:
  commit       Link one qualified lead to a generated engagement and import it into the commercial pipeline.
  reconcile    Report qualified leads, missing or inconsistent imports, linked value, and next actions.

Options:
  --registry PATH        Default: ${DEFAULT_REGISTRY}
  --pipeline PATH        Default: ${DEFAULT_PIPELINE}
  --lead ID              Qualified acquisition lead id
  --engagement PATH      Generated engagement.json
  --evidence TEXT        Required proposal/sending evidence reference; never paste secrets or message bodies
  --at YYYY-MM-DD        Real proposal handoff date; default: current UTC date
  --as-of YYYY-MM-DD     Reconciliation date; default: current UTC date
  --json                 Render reconciliation as JSON instead of Markdown
  --help

Commit validates both complete next states before writing. It writes the commercial
pipeline first and the acquisition link second, so an interrupted second write can be
completed by rerunning the identical command. The command never generates or sends a
proposal, signs an agreement, requests payment, charges a client, or changes external state.
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
    if (!new Set(["registry", "pipeline", "lead", "engagement", "evidence", "at", "asOf"]).has(name)) {
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
  if (!new Set(["commit", "reconcile"]).has(command)) throw new Error(`Unknown command: ${command}`);
  const options = parseOptions(argv.slice(1));
  if (options.help) return { help: true };
  const common = {
    help: false,
    command,
    registryPath: path.resolve(singleLine(options.registry || DEFAULT_REGISTRY, "--registry", 1000)),
    pipelinePath: path.resolve(singleLine(options.pipeline || DEFAULT_PIPELINE, "--pipeline", 1000)),
  };
  const today = utcDate(nowMs);
  if (command === "commit") {
    if (options.json || options.asOf) throw new Error("commit received an option for reconcile");
    return {
      ...common,
      leadId: singleLine(options.lead, "--lead", 240),
      engagementPath: path.resolve(singleLine(options.engagement, "--engagement", 1000)),
      evidence: singleLine(options.evidence, "--evidence", 500),
      at: isoDate(options.at || today, "--at"),
    };
  }
  if (options.lead || options.engagement || options.evidence || options.at) {
    throw new Error("reconcile received an option for commit");
  }
  return { ...common, asOf: isoDate(options.asOf || today, "--as-of"), json: options.json };
}

function proposedEvent(lead) {
  return lead.events.find((event) => event.status === "proposed") ?? null;
}

function assertLinkedEvidence(lead, pipelineEngagement) {
  const event = proposedEvent(lead);
  if (event == null) throw new Error(`lead ${lead.id} has no proposed event`);
  const differences = [];
  if (event.engagementId !== pipelineEngagement.id) differences.push("engagement id");
  if (event.engagementSourceDigest !== pipelineEngagement.sourceDigest) differences.push("source digest");
  if (event.engagementClient !== pipelineEngagement.client) differences.push("client");
  if (event.engagementProvider !== pipelineEngagement.provider) differences.push("provider");
  if (event.engagementProposalDate !== pipelineEngagement.proposalDate) differences.push("proposal date");
  if (differences.length > 0) throw new Error(`proposal handoff evidence differs for: ${differences.join(", ")}`);
}

export function prepareProposalHandoff(registryValue, pipelineValue, leadIdValue, engagementValue, atValue, evidenceValue) {
  const registry = validateAcquisitionRegistry(registryValue);
  const pipeline = validatePipeline(pipelineValue);
  const leadId = singleLine(leadIdValue, "leadId", 240);
  const at = isoDate(atValue, "at");
  const evidence = singleLine(evidenceValue, "evidence", 500);

  const pipelineResult = importEngagement(pipeline, engagementValue, at);
  const acquisitionResult = advanceLead(registry, leadId, "proposed", at, { evidence, engagement: engagementValue });
  assertLinkedEvidence(acquisitionResult.lead, pipelineResult.engagement);

  return {
    registry: acquisitionResult.registry,
    pipeline: pipelineResult.pipeline,
    acquisitionChanged: acquisitionResult.changed,
    pipelineChanged: pipelineResult.changed,
    lead: acquisitionResult.lead,
    engagement: pipelineResult.engagement,
  };
}

function cents(value) {
  return Math.round(value * 100);
}

function dollars(value) {
  return Math.round(value) / 100;
}

function linkedCash(engagement) {
  let net = 0;
  for (const event of engagement.events) {
    if (event.status === "paid") net += cents(event.amount);
    if (event.status === "refunded") net -= cents(event.amount);
  }
  return net;
}

function linkDifferences(event, engagement) {
  const differences = [];
  if (event.engagementSourceDigest !== engagement.sourceDigest) differences.push("sourceDigest");
  if (event.engagementClient !== engagement.client) differences.push("client");
  if (event.engagementProvider !== engagement.provider) differences.push("provider");
  if (event.engagementProposalDate !== engagement.proposalDate) differences.push("proposalDate");
  return differences;
}

function ratio(numerator, denominator) {
  return denominator === 0 ? null : numerator / denominator;
}

export function reconcileProposalHandoffs(registryValue, pipelineValue, asOfValue) {
  const registry = validateAcquisitionRegistry(registryValue);
  const pipeline = validatePipeline(pipelineValue);
  const asOf = isoDate(asOfValue, "asOf");
  for (const [label, updatedAt] of [["acquisition", registry.updatedAt], ["pipeline", pipeline.updatedAt]]) {
    if (updatedAt != null && asOf < updatedAt) throw new Error(`asOf ${asOf} precedes latest ${label} evidence ${updatedAt}`);
  }

  const byEngagementId = new Map(pipeline.engagements.map((engagement) => [engagement.id, engagement]));
  const linkedIds = new Set();
  const linked = [];
  const missingPipelineImports = [];
  const inconsistentLinks = [];
  const qualifiedAwaitingProposal = [];

  for (const lead of registry.leads) {
    if (lead.currentStatus === "qualified") {
      qualifiedAwaitingProposal.push({ id: lead.id, organization: lead.organization });
    }
    const event = proposedEvent(lead);
    if (event == null) continue;
    const engagement = byEngagementId.get(event.engagementId);
    if (engagement == null) {
      missingPipelineImports.push({ id: lead.id, organization: lead.organization, engagementId: event.engagementId });
      continue;
    }
    const differences = linkDifferences(event, engagement);
    if (differences.length > 0) {
      inconsistentLinks.push({
        id: lead.id,
        organization: lead.organization,
        engagementId: event.engagementId,
        differences,
      });
      continue;
    }
    linkedIds.add(engagement.id);
    linked.push({
      leadId: lead.id,
      organization: lead.organization,
      engagementId: engagement.id,
      status: engagement.currentStatus,
      standardReviewPrice: engagement.standardReviewPrice,
      netCashCollected: dollars(linkedCash(engagement)),
    });
  }

  const unlinkedPipelineEngagements = pipeline.engagements
    .filter((engagement) => !linkedIds.has(engagement.id))
    .map((engagement) => ({ engagementId: engagement.id, client: engagement.client, status: engagement.currentStatus }));
  const proposedLeadCount = registry.leads.filter((lead) => proposedEvent(lead) != null).length;
  const qualifiedMilestoneCount = registry.leads.filter((lead) => lead.events.some((event) => event.status === "qualified")).length;

  for (const values of [linked, missingPipelineImports, inconsistentLinks, qualifiedAwaitingProposal]) {
    values.sort((left, right) => left.organization.localeCompare(right.organization) || left.id?.localeCompare(right.id) || 0);
  }
  unlinkedPipelineEngagements.sort((left, right) => left.client.localeCompare(right.client) || left.engagementId.localeCompare(right.engagementId));

  return {
    schemaVersion: 1,
    asOf,
    healthy: missingPipelineImports.length === 0 && inconsistentLinks.length === 0,
    counts: {
      qualifiedMilestones: qualifiedMilestoneCount,
      qualifiedAwaitingProposal: qualifiedAwaitingProposal.length,
      proposedLeads: proposedLeadCount,
      linkedEngagements: linked.length,
      missingPipelineImports: missingPipelineImports.length,
      inconsistentLinks: inconsistentLinks.length,
      unlinkedPipelineEngagements: unlinkedPipelineEngagements.length,
    },
    conversions: {
      proposedToLinked: ratio(linked.length, proposedLeadCount),
      qualifiedToLinked: ratio(linked.length, qualifiedMilestoneCount),
    },
    value: {
      currency: "USD",
      linkedStandardReviewValue: dollars(linked.reduce((total, item) => total + cents(item.standardReviewPrice), 0)),
      linkedNetCashCollected: dollars(linked.reduce((total, item) => total + cents(item.netCashCollected), 0)),
    },
    qualifiedAwaitingProposal,
    missingPipelineImports,
    inconsistentLinks,
    linked,
    unlinkedPipelineEngagements,
  };
}

function percent(value) {
  return value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;
}

function currency(value) {
  return `USD ${value.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 2 })}`;
}

function tableText(value) {
  return String(value).replace(/\|/g, "\\|");
}

export function renderProposalHandoffReconciliation(summary) {
  const actionRows = [
    ...summary.qualifiedAwaitingProposal.map((item) => `| ${tableText(item.organization)} | qualified | Generate and review the commercial kit, then commit the handoff |`),
    ...summary.missingPipelineImports.map((item) => `| ${tableText(item.organization)} | proposed | Import ${tableText(item.engagementId)} or rerun the identical handoff |`),
    ...summary.inconsistentLinks.map((item) => `| ${tableText(item.organization)} | inconsistent | Resolve ${tableText(item.differences.join(", "))} before relying on funnel value |`),
  ];
  const unlinkedRows = summary.unlinkedPipelineEngagements.map(
    (item) => `| ${tableText(item.client)} | ${tableText(item.engagementId)} | ${item.status} | Confirm direct/referral provenance or link its acquisition evidence |`,
  );
  return [
    "# Strategy Assurance Proposal Handoff",
    "",
    `As of: ${summary.asOf}`,
    `Status: ${summary.healthy ? "healthy" : "attention required"}`,
    "",
    "## Conversion and value",
    "",
    `- Proposed → linked: ${percent(summary.conversions.proposedToLinked)} (${summary.counts.linkedEngagements}/${summary.counts.proposedLeads})`,
    `- Qualified → linked: ${percent(summary.conversions.qualifiedToLinked)} (${summary.counts.linkedEngagements}/${summary.counts.qualifiedMilestones})`,
    `- Linked standard-review value: ${currency(summary.value.linkedStandardReviewValue)}`,
    `- Linked net cash collected: ${currency(summary.value.linkedNetCashCollected)}`,
    "",
    "## Required actions",
    "",
    "| Organization | State | Action |",
    "| --- | --- | --- |",
    ...(actionRows.length > 0 ? actionRows : ["| None | — | No acquisition handoff action required |"]),
    "",
    "## Pipeline engagements without acquisition links",
    "",
    "These may be valid direct or referral engagements; they are reported rather than treated as inconsistent.",
    "",
    "| Client | Engagement | Status | Action |",
    "| --- | --- | --- | --- |",
    ...(unlinkedRows.length > 0 ? unlinkedRows : ["| None | — | — | No provenance review required |"]),
    "",
  ].join("\n");
}

async function readEngagement(filePath) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    if (error instanceof SyntaxError) throw new Error(`engagement is not valid JSON: ${filePath}`);
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
    const registry = await loadAcquisitionRegistry(args.registryPath);
    const pipeline = await loadPipeline(args.pipelinePath, { allowMissing: true });
    if (args.command === "commit") {
      const engagement = await readEngagement(args.engagementPath);
      const result = prepareProposalHandoff(registry, pipeline, args.leadId, engagement, args.at, args.evidence);
      if (result.pipelineChanged) await savePipeline(args.pipelinePath, result.pipeline);
      if (result.acquisitionChanged) {
        try {
          await saveAcquisitionRegistry(args.registryPath, result.registry);
        } catch (error) {
          throw new Error(`commercial pipeline may already contain the validated engagement; rerun this identical command to complete the acquisition link: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
      process.stdout.write(`${JSON.stringify({
        changed: result.pipelineChanged || result.acquisitionChanged,
        pipelineChanged: result.pipelineChanged,
        acquisitionChanged: result.acquisitionChanged,
        leadId: result.lead.id,
        engagementId: result.engagement.id,
        status: result.engagement.currentStatus,
      }, null, 2)}\n`);
      return;
    }
    const summary = reconcileProposalHandoffs(registry, pipeline, args.asOf);
    process.stdout.write(args.json ? `${JSON.stringify(summary, null, 2)}\n` : renderProposalHandoffReconciliation(summary));
  } catch (error) {
    process.stderr.write(`Strategy Assurance handoff failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  await main();
}
