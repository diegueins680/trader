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
  saveAcquisitionRegistry,
} from "../scripts/strategy-assurance-acquisition.mjs";
import {
  parseArgs,
  prepareProposalHandoff,
  reconcileProposalHandoffs,
  renderProposalHandoffReconciliation,
} from "../scripts/strategy-assurance-handoff.mjs";
import {
  buildCommercialKit,
  parseArgs as parseKitArgs,
} from "../scripts/generate-strategy-assurance-kit.mjs";
import {
  buildOutreachCampaign,
  parseArgs as parseOutreachArgs,
} from "../scripts/generate-strategy-assurance-outreach.mjs";
import {
  emptyPipeline,
  importEngagement,
  loadPipeline,
} from "../scripts/strategy-assurance-pipeline.mjs";

const REPOSITORY_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HANDOFF_SCRIPT = fileURLToPath(new URL("../scripts/strategy-assurance-handoff.mjs", import.meta.url));

async function qualifiedLead() {
  const queue = JSON.parse(
    await fs.readFile(path.join(REPOSITORY_ROOT, "docs", "strategy-assurance-prospect-queue.json"), "utf8"),
  );
  const checklist = await fs.readFile(
    path.join(REPOSITORY_ROOT, "docs", "strategy-assurance-pre-live-checklist.md"),
    "utf8",
  );
  const outreachConfig = parseOutreachArgs([
    "--provider",
    "Assurance Labs",
    "--sender",
    "Avery Chen, Principal",
    "--prospect",
    "OctoBot",
    "--campaign-date",
    "2026-08-17",
  ]);
  const built = buildOutreachCampaign(queue, checklist, outreachConfig);
  const imported = importOutreachCampaign(
    emptyAcquisitionRegistry(),
    { campaign: built.campaign, files: built.files },
    "2026-08-18",
  );
  const id = imported.leads[0].id;
  let registry = advanceLead(imported.registry, id, "contacted", "2026-08-18", {
    evidence: "contact receipt H-1",
    channel: "official organization route",
  }).registry;
  registry = advanceLead(registry, id, "responded", "2026-08-19", { evidence: "reply H-2" }).registry;
  registry = advanceLead(registry, id, "qualified", "2026-08-20", { evidence: "discovery record H-3" }).registry;
  return { registry, id };
}

function engagement(client = "OctoBot", proposalDate = "2026-08-21") {
  return buildCommercialKit(
    parseKitArgs([
      "--client",
      client,
      "--provider",
      "Assurance Labs",
      "--decision-owner",
      `${client} decision owner`,
      "--strategy",
      `${client} strategy`,
      "--deployment",
      `${client} production`,
      "--proposal-date",
      proposalDate,
    ]),
  ).engagement;
}

test("handoff arguments separate commit evidence from read-only reconciliation", () => {
  const now = Date.UTC(2026, 7, 21);
  const committed = parseArgs(
    ["commit", "--lead", "strategy-assurance-lead-1234", "--engagement", "engagement.json", "--evidence", "sent record P-1"],
    now,
  );
  assert.equal(committed.command, "commit");
  assert.equal(committed.at, "2026-08-21");
  assert.match(committed.pipelinePath, /strategy-assurance\/pipeline\.json$/);

  const reconciled = parseArgs(["reconcile", "--as-of", "2026-08-31", "--json"], now);
  assert.equal(reconciled.command, "reconcile");
  assert.equal(reconciled.json, true);
  assert.throws(() => parseArgs(["commit", "--lead", "lead", "--engagement", "engagement.json"], now), /--evidence/);
  assert.throws(() => parseArgs(["reconcile", "--evidence", "not valid"], now), /option for commit/);
});

test("proposal handoff validates both states and is identical-rerun idempotent", async () => {
  const { registry, id } = await qualifiedLead();
  const source = engagement();
  const first = prepareProposalHandoff(registry, emptyPipeline(), id, source, "2026-08-21", "reviewed proposal sent P-1");
  assert.equal(first.acquisitionChanged, true);
  assert.equal(first.pipelineChanged, true);
  assert.equal(first.lead.currentStatus, "proposed");
  assert.equal(first.engagement.currentStatus, "proposal");
  assert.equal(first.lead.events.at(-1).engagementSourceDigest, first.engagement.sourceDigest);

  const recovered = prepareProposalHandoff(
    registry,
    first.pipeline,
    id,
    source,
    "2026-08-21",
    "reviewed proposal sent P-1",
  );
  assert.equal(recovered.pipelineChanged, false);
  assert.equal(recovered.acquisitionChanged, true);
  assert.equal(recovered.lead.currentStatus, "proposed");

  const second = prepareProposalHandoff(
    first.registry,
    first.pipeline,
    id,
    source,
    "2026-08-21",
    "reviewed proposal sent P-1",
  );
  assert.equal(second.acquisitionChanged, false);
  assert.equal(second.pipelineChanged, false);

  const wrongProvider = structuredClone(source);
  wrongProvider.provider = "Different Provider";
  await assert.rejects(
    async () => prepareProposalHandoff(registry, emptyPipeline(), id, wrongProvider, "2026-08-21", "proposal P-2"),
    /provider does not match/,
  );
  assert.throws(
    () => prepareProposalHandoff(emptyAcquisitionRegistry(), emptyPipeline(), id, source, "2026-08-21", "proposal P-3"),
    /unknown lead id/,
  );
});

test("handoff reconciliation exposes leakage, inconsistent evidence, and direct pipeline provenance", async () => {
  const { registry, id } = await qualifiedLead();
  const source = engagement();
  const waiting = reconcileProposalHandoffs(registry, emptyPipeline(), "2026-08-21");
  assert.equal(waiting.healthy, true);
  assert.equal(waiting.counts.qualifiedAwaitingProposal, 1);
  assert.equal(waiting.conversions.qualifiedToLinked, 0);

  const proposed = advanceLead(registry, id, "proposed", "2026-08-21", {
    evidence: "reviewed proposal sent P-4",
    engagement: source,
  }).registry;
  const missing = reconcileProposalHandoffs(proposed, emptyPipeline(), "2026-08-21");
  assert.equal(missing.healthy, false);
  assert.equal(missing.counts.missingPipelineImports, 1);
  assert.match(renderProposalHandoffReconciliation(missing), /rerun the identical handoff/);

  const altered = structuredClone(source);
  altered.decisionOwner = "Different decision owner";
  const mismatchedPipeline = importEngagement(emptyPipeline(), altered, "2026-08-21").pipeline;
  const inconsistent = reconcileProposalHandoffs(proposed, mismatchedPipeline, "2026-08-21");
  assert.equal(inconsistent.counts.inconsistentLinks, 1);
  assert.deepEqual(inconsistent.inconsistentLinks[0].differences, ["sourceDigest"]);
  assert.equal(inconsistent.counts.unlinkedPipelineEngagements, 1);

  const direct = engagement("Direct Referral");
  const directPipeline = importEngagement(emptyPipeline(), direct, "2026-08-21").pipeline;
  const provenance = reconcileProposalHandoffs(registry, directPipeline, "2026-08-21");
  assert.equal(provenance.healthy, true);
  assert.equal(provenance.counts.unlinkedPipelineEngagements, 1);
});

test("handoff CLI persists pipeline before acquisition and reconciles exact linked value", async () => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-handoff-"));
  const registryPath = path.join(temporaryRoot, "acquisition.json");
  const pipelinePath = path.join(temporaryRoot, "pipeline.json");
  const engagementPath = path.join(temporaryRoot, "engagement.json");
  const run = (...args) => spawnSync(process.execPath, [HANDOFF_SCRIPT, ...args], { encoding: "utf8" });
  try {
    const { registry, id } = await qualifiedLead();
    const source = engagement();
    await saveAcquisitionRegistry(registryPath, registry);
    await fs.writeFile(engagementPath, `${JSON.stringify(source, null, 2)}\n`);

    const committed = run(
      "commit",
      "--lead",
      id,
      "--engagement",
      engagementPath,
      "--evidence",
      "reviewed proposal sent CLI-P-1",
      "--at",
      "2026-08-21",
      "--registry",
      registryPath,
      "--pipeline",
      pipelinePath,
    );
    assert.equal(committed.status, 0, committed.stderr);
    assert.deepEqual(
      {
        pipelineChanged: JSON.parse(committed.stdout).pipelineChanged,
        acquisitionChanged: JSON.parse(committed.stdout).acquisitionChanged,
      },
      { pipelineChanged: true, acquisitionChanged: true },
    );
    assert.equal((await loadPipeline(pipelinePath)).engagements[0].id, source.engagementId);

    const repeated = run(
      "commit",
      "--lead",
      id,
      "--engagement",
      engagementPath,
      "--evidence",
      "reviewed proposal sent CLI-P-1",
      "--at",
      "2026-08-21",
      "--registry",
      registryPath,
      "--pipeline",
      pipelinePath,
    );
    assert.equal(repeated.status, 0, repeated.stderr);
    assert.equal(JSON.parse(repeated.stdout).changed, false);

    const reconciled = run(
      "reconcile",
      "--registry",
      registryPath,
      "--pipeline",
      pipelinePath,
      "--as-of",
      "2026-08-21",
      "--json",
    );
    assert.equal(reconciled.status, 0, reconciled.stderr);
    const summary = JSON.parse(reconciled.stdout);
    assert.equal(summary.healthy, true);
    assert.equal(summary.conversions.proposedToLinked, 1);
    assert.equal(summary.value.linkedStandardReviewValue, 2500);
    assert.equal(summary.value.linkedNetCashCollected, 0);
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});
