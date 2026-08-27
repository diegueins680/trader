import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  advanceLead,
  emptyAcquisitionRegistry,
  importOutreachCampaign,
  loadAcquisitionRegistry,
  parseArgs,
  renderAcquisitionSummary,
  saveAcquisitionRegistry,
  summarizeAcquisition,
  validateOutreachPackage,
} from "../scripts/strategy-assurance-acquisition.mjs";
import {
  buildCommercialKit,
  parseArgs as parseKitArgs,
} from "../scripts/generate-strategy-assurance-kit.mjs";
import {
  buildOutreachCampaign,
  parseArgs as parseOutreachArgs,
  writeOutreachCampaign,
} from "../scripts/generate-strategy-assurance-outreach.mjs";

const REPOSITORY_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ACQUISITION_SCRIPT = fileURLToPath(new URL("../scripts/strategy-assurance-acquisition.mjs", import.meta.url));

async function fixtures(prospects = ["OctoBot", "Jesse"], campaignDate = "2026-08-17") {
  const queue = JSON.parse(
    await fs.readFile(path.join(REPOSITORY_ROOT, "docs", "strategy-assurance-prospect-queue.json"), "utf8"),
  );
  const checklist = await fs.readFile(
    path.join(REPOSITORY_ROOT, "docs", "strategy-assurance-pre-live-checklist.md"),
    "utf8",
  );
  const args = [
    "--provider",
    "Assurance Labs",
    "--sender",
    "Avery Chen, Principal",
    ...prospects.flatMap((prospect) => ["--prospect", prospect]),
    "--campaign-date",
    campaignDate,
  ];
  const config = parseOutreachArgs(args);
  const built = buildOutreachCampaign(queue, checklist, config);
  return { queue, checklist, config, built, packageValue: { campaign: built.campaign, files: built.files } };
}

function proposal(client = "Referred Operator", proposalDate = "2026-08-22") {
  return buildCommercialKit(
    parseKitArgs([
      "--client",
      client,
      "--provider",
      "Assurance Labs",
      "--decision-owner",
      `${client} CTO`,
      "--strategy",
      `${client} Strategy`,
      "--deployment",
      `${client} Production`,
      "--proposal-date",
      proposalDate,
    ]),
  ).engagement;
}

function advance(registry, id, status, at, details) {
  return advanceLead(registry, id, status, at, details).registry;
}

test("acquisition CLI arguments keep import, evidence transitions, and summary distinct", () => {
  const now = Date.UTC(2026, 7, 20);
  const imported = parseArgs(["import", "--campaign", "campaign.json"], now);
  assert.equal(imported.command, "import");
  assert.equal(imported.at, "2026-08-20");
  assert.match(imported.registryPath, /strategy-assurance\/acquisition\.json$/);

  const contacted = parseArgs(
    [
      "advance",
      "--id",
      "strategy-assurance-lead-1234",
      "--status",
      "contacted",
      "--evidence",
      "contact-form receipt 42",
      "--channel",
      "official contact form",
    ],
    now,
  );
  assert.equal(contacted.status, "contacted");
  assert.equal(contacted.channel, "official contact form");

  const summary = parseArgs(["summary", "--as-of", "2026-08-31", "--json"], now);
  assert.equal(summary.asOf, "2026-08-31");
  assert.equal(summary.json, true);

  assert.throws(() => parseArgs(["import"]), /--campaign is required/);
  assert.throws(
    () =>
      parseArgs([
        "advance",
        "--id",
        "strategy-assurance-lead-1234",
        "--status",
        "contacted",
        "--evidence",
        "receipt",
      ]),
    /--channel is required/,
  );
  assert.throws(
    () =>
      parseArgs([
        "advance",
        "--id",
        "strategy-assurance-lead-1234",
        "--status",
        "proposed",
        "--evidence",
        "proposal generated",
      ]),
    /--engagement is required/,
  );
  assert.throws(() => parseArgs(["summary", "--status", "qualified"]), /option for another command/);
});

test("campaign import verifies package hashes, is idempotent, and prevents duplicate organizations", async () => {
  const { packageValue } = await fixtures();
  const validated = validateOutreachPackage(packageValue.campaign, packageValue.files);
  assert.equal(validated.records.length, 2);
  assert.equal(validated.records[0].queueKind, "commercial-platform-channel");

  const first = importOutreachCampaign(emptyAcquisitionRegistry(), packageValue, "2026-08-18");
  assert.equal(first.changed, true);
  assert.equal(first.leads.length, 2);
  assert.equal(first.registry.leads.every((lead) => lead.currentStatus === "prepared"), true);
  const second = importOutreachCampaign(first.registry, packageValue, "2026-08-19");
  assert.equal(second.changed, false);
  assert.deepEqual(second.registry, first.registry);

  const tampered = { campaign: packageValue.campaign, files: { ...packageValue.files } };
  tampered.files["octobot/initial.md"] += "tampered\n";
  assert.throws(() => validateOutreachPackage(tampered.campaign, tampered.files), /digest does not match/);

  const later = await fixtures(["OctoBot"], "2026-08-18");
  assert.throws(
    () => importOutreachCampaign(first.registry, later.packageValue, "2026-08-19"),
    /organization already exists/,
  );
});

test("acquisition lifecycle enforces real evidence, wait periods, and proposal linkage", async () => {
  const imported = importOutreachCampaign(emptyAcquisitionRegistry(), (await fixtures()).packageValue, "2026-08-18");
  let registry = imported.registry;
  const octobotId = imported.leads.find((lead) => lead.organization === "OctoBot").id;
  const jesseId = imported.leads.find((lead) => lead.organization === "Jesse").id;

  assert.throws(() => advanceLead(registry, octobotId, "responded", "2026-08-20", { evidence: "reply" }), /prepared -> responded/);
  registry = advance(registry, octobotId, "contacted", "2026-08-20", {
    evidence: "contact-form receipt 42",
    channel: "official contact form",
  });
  assert.throws(
    () =>
      advanceLead(registry, octobotId, "follow-up-sent", "2026-08-24", {
        evidence: "sent-folder message 43",
        channel: "official contact form",
      }),
    /before 2026-08-27/,
  );
  registry = advance(registry, octobotId, "follow-up-sent", "2026-08-27", {
    evidence: "sent-folder message 43",
    channel: "official contact form",
  });
  assert.throws(
    () => advanceLead(registry, octobotId, "closed-no-response", "2026-09-02", { evidence: "no response observed" }),
    /before 2026-09-03/,
  );
  registry = advance(registry, octobotId, "closed-no-response", "2026-09-03", {
    evidence: "mailbox checked after follow-up window",
  });

  registry = advance(registry, jesseId, "contacted", "2026-08-20", {
    evidence: "moderator request message 11",
    channel: "official community route",
  });
  registry = advance(registry, jesseId, "responded", "2026-08-21", { evidence: "moderator reply 12" });
  registry = advance(registry, jesseId, "qualified", "2026-08-21", { evidence: "discovery note Q-12" });
  const wrongProviderProposal = proposal("Jesse", "2026-08-22");
  wrongProviderProposal.provider = "Different Provider";
  assert.throws(
    () =>
      advanceLead(registry, jesseId, "proposed", "2026-08-22", {
        evidence: "proposal package",
        engagement: wrongProviderProposal,
      }),
    /provider does not match/,
  );
  const wrongClientProposal = proposal("Different Client", "2026-08-22");
  assert.throws(
    () =>
      advanceLead(registry, jesseId, "proposed", "2026-08-22", {
        evidence: "proposal package",
        engagement: wrongClientProposal,
      }),
    /client does not match/,
  );
  const validProposal = proposal("Jesse");
  registry = advance(registry, jesseId, "proposed", "2026-08-22", {
    evidence: "engagement package generated and reviewed",
    engagement: validProposal,
  });
  const proposed = registry.leads.find((lead) => lead.id === jesseId);
  assert.equal(proposed.currentStatus, "proposed");
  assert.equal(proposed.events.at(-1).engagementId, validProposal.engagementId);
});

test("acquisition summary reports exact source conversions and dated next actions", async () => {
  const imported = importOutreachCampaign(emptyAcquisitionRegistry(), (await fixtures()).packageValue, "2026-08-18");
  let registry = imported.registry;
  const octobotId = imported.leads.find((lead) => lead.organization === "OctoBot").id;
  const jesseId = imported.leads.find((lead) => lead.organization === "Jesse").id;

  registry = advance(registry, octobotId, "contacted", "2026-08-18", {
    evidence: "contact receipt O-1",
    channel: "official contact form",
  });
  registry = advance(registry, octobotId, "follow-up-sent", "2026-08-25", {
    evidence: "follow-up receipt O-2",
    channel: "official contact form",
  });
  registry = advance(registry, jesseId, "contacted", "2026-08-18", {
    evidence: "community message J-1",
    channel: "official community route",
  });
  registry = advance(registry, jesseId, "responded", "2026-08-19", { evidence: "reply J-2" });
  registry = advance(registry, jesseId, "qualified", "2026-08-20", { evidence: "discovery J-3" });

  const summary = summarizeAcquisition(registry, "2026-09-01");
  assert.equal(summary.totalLeads, 2);
  assert.equal(summary.counts["follow-up-sent"], 1);
  assert.equal(summary.counts.qualified, 1);
  assert.deepEqual(summary.funnel, {
    prepared: 2,
    contacted: 2,
    followUpSent: 1,
    responded: 1,
    qualified: 1,
    proposed: 0,
    preparedToContacted: 1,
    contactedToResponded: 0.5,
    respondedToQualified: 1,
    qualifiedToProposed: 0,
    contactedToProposed: 0,
    followUpToResponded: 0,
  });
  assert.equal(summary.sources.length, 2);
  assert.deepEqual(summary.closeEligibleLeadIds, [octobotId]);
  assert.equal(summary.nextActions.find((action) => action.id === jesseId).action, "Generate and review the commercial kit, then commit the proposal handoff");
  assert.match(renderAcquisitionSummary(summary), /Contacted → responded: 50\.0% \(1\/2\)/);
  assert.match(renderAcquisitionSummary(summary), /commercial-platform-channel/);
  assert.throws(() => summarizeAcquisition(registry, "2026-08-19"), /precedes latest registry evidence/);
});

test("acquisition CLI imports a package, persists atomically, and records only explicit events", async () => {
  const { queue, checklist, config } = await fixtures(["OctoBot"]);
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-acquisition-"));
  const campaignDir = path.join(temporaryRoot, "campaign");
  const registryPath = path.join(temporaryRoot, "state", "acquisition.json");
  const run = (...args) => spawnSync(process.execPath, [ACQUISITION_SCRIPT, ...args], { encoding: "utf8" });
  try {
    await writeOutreachCampaign(queue, checklist, { ...config, outputDir: campaignDir });
    const campaignPath = path.join(campaignDir, "campaign.json");
    const imported = run("import", "--campaign", campaignPath, "--registry", registryPath, "--at", "2026-08-18");
    assert.equal(imported.status, 0, imported.stderr);
    const id = JSON.parse(imported.stdout).leads[0].id;

    const contacted = run(
      "advance",
      "--id",
      id,
      "--status",
      "contacted",
      "--evidence",
      "contact receipt CLI-1",
      "--channel",
      "official contact form",
      "--at",
      "2026-08-20",
      "--registry",
      registryPath,
    );
    assert.equal(contacted.status, 0, contacted.stderr);
    const summarized = run("summary", "--registry", registryPath, "--as-of", "2026-08-20", "--json");
    assert.equal(summarized.status, 0, summarized.stderr);
    assert.equal(JSON.parse(summarized.stdout).counts.contacted, 1);

    const registry = await loadAcquisitionRegistry(registryPath);
    assert.equal(registry.leads[0].events.at(-1).evidence, "contact receipt CLI-1");
    await saveAcquisitionRegistry(registryPath, registry);
    assert.deepEqual(await fs.readdir(path.dirname(registryPath)), ["acquisition.json"]);
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});
