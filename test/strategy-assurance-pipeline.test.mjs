import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  buildCommercialKit,
  parseArgs as parseKitArgs,
} from "../scripts/generate-strategy-assurance-kit.mjs";
import {
  advanceEngagement,
  emptyPipeline,
  importEngagement,
  loadPipeline,
  parseArgs,
  renderPipelineSummary,
  savePipeline,
  summarizePipeline,
} from "../scripts/strategy-assurance-pipeline.mjs";

const PIPELINE_SCRIPT = fileURLToPath(new URL("../scripts/strategy-assurance-pipeline.mjs", import.meta.url));

function engagementFor(client, proposalDate = "2026-08-01", extraArgs = []) {
  const config = parseKitArgs([
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
    ...extraArgs,
  ]);
  return buildCommercialKit(config).engagement;
}

function advance(pipeline, id, status, at, details = {}) {
  return advanceEngagement(pipeline, id, status, at, details).pipeline;
}

test("pipeline CLI arguments keep import, advance, and summary scopes distinct", () => {
  const now = Date.UTC(2026, 7, 14);
  const imported = parseArgs(["import", "--engagement", "engagement.json"], now);
  assert.equal(imported.command, "import");
  assert.equal(imported.at, "2026-08-14");
  assert.match(imported.pipelinePath, /\.tmp\/strategy-assurance\/pipeline\.json$/);

  const advanced = parseArgs(["advance", "--id", "engagement-1", "--status", "accepted"], now);
  assert.equal(advanced.status, "accepted");
  assert.equal(advanced.at, "2026-08-14");
  const paid = parseArgs(
    ["advance", "--id", "engagement-1", "--status", "paid", "--amount", "2450.50"],
    now,
  );
  assert.deepEqual(paid.details, { amount: 2450.5 });

  const summary = parseArgs(["summary", "--as-of", "2026-08-31", "--json"], now);
  assert.equal(summary.asOf, "2026-08-31");
  assert.equal(summary.json, true);

  assert.throws(() => parseArgs(["import"]), /--engagement is required/);
  assert.throws(() => parseArgs(["advance", "--status", "delivered"]), /non-empty string/);
  assert.throws(
    () => parseArgs(["advance", "--id", "engagement-1", "--status", "paid"]),
    /--amount is required/,
  );
  assert.throws(
    () => parseArgs(["advance", "--id", "engagement-1", "--status", "paid", "--amount", "1.001"]),
    /two decimal places/,
  );
  assert.throws(
    () => parseArgs(["advance", "--id", "engagement-1", "--status", "delivered"]),
    /--hours is required/,
  );
  assert.throws(
    () => parseArgs(["advance", "--id", "engagement-1", "--status", "accepted", "--amount", "1"]),
    /not valid/,
  );
  assert.throws(
    () => parseArgs(["advance", "--id", "engagement-1", "--status", "invented"]),
    /must be one of/,
  );
  assert.throws(() => parseArgs(["summary", "--status", "paid"]), /option for another command/);
});

test("pipeline import is idempotent and rejects identity collisions", () => {
  const source = engagementFor("Acme Capital");
  const first = importEngagement(emptyPipeline(), source, "2026-08-02");
  assert.equal(first.changed, true);
  assert.equal(first.engagement.currentStatus, "proposal");

  const second = importEngagement(first.pipeline, source, "2026-08-03");
  assert.equal(second.changed, false);
  assert.deepEqual(second.pipeline, first.pipeline);

  const changed = structuredClone(source);
  changed.commercials.standardReviewPrice = 9999;
  assert.throws(() => importEngagement(first.pipeline, changed, "2026-08-03"), /different source evidence/);
});

test("pipeline enforces forward commercial and delivery transitions", () => {
  const source = engagementFor("Acme Capital");
  const imported = importEngagement(emptyPipeline(), source, "2026-08-02");
  const id = imported.engagement.id;

  assert.throws(
    () => advanceEngagement(imported.pipeline, id, "paid", "2026-08-03", { amount: 2400 }),
    /proposal -> paid/,
  );
  let pipeline = advance(imported.pipeline, id, "accepted", "2026-08-03");
  assert.equal(advanceEngagement(pipeline, id, "accepted", "2026-08-03").changed, false);
  assert.throws(
    () => advanceEngagement(pipeline, id, "paid", "2026-08-02", { amount: 2400 }),
    /precedes the latest event/,
  );
  pipeline = advance(pipeline, id, "paid", "2026-08-04", { amount: 2400 });
  assert.equal(advanceEngagement(pipeline, id, "paid", "2026-08-04", { amount: 2400 }).changed, false);
  assert.throws(
    () => advanceEngagement(pipeline, id, "paid", "2026-08-04", { amount: 2500 }),
    /different event evidence/,
  );
  pipeline = advance(pipeline, id, "in-delivery", "2026-08-05");
  pipeline = advance(pipeline, id, "delivered", "2026-08-10", { hours: 12.5 });
  pipeline = advance(pipeline, id, "monitoring", "2026-08-11");
  assert.equal(pipeline.engagements[0].currentStatus, "monitoring");
  assert.throws(
    () => advanceEngagement(pipeline, id, "refunded", "2026-08-12", { amount: 100 }),
    /monitoring -> refunded/,
  );
});

test("pipeline summary reports exact funnel, revenue, expirations, and next actions", () => {
  let pipeline = emptyPipeline();
  const acme = engagementFor("Acme Capital", "2026-08-01");
  const beta = engagementFor("Beta Trading", "2026-07-01", ["--price", "3000"]);
  const gamma = engagementFor("Gamma Systems", "2026-08-01", ["--price", "4000"]);
  const delta = engagementFor("Delta Research", "2026-08-01", ["--price", "3500"]);

  let result = importEngagement(pipeline, acme, "2026-08-02");
  pipeline = result.pipeline;
  const acmeId = result.engagement.id;
  result = importEngagement(pipeline, beta, "2026-08-02");
  pipeline = result.pipeline;
  const betaId = result.engagement.id;
  result = importEngagement(pipeline, gamma, "2026-08-02");
  pipeline = result.pipeline;
  const gammaId = result.engagement.id;
  result = importEngagement(pipeline, delta, "2026-08-02");
  pipeline = result.pipeline;
  const deltaId = result.engagement.id;

  for (const [status, at, details] of [
    ["accepted", "2026-08-03", {}],
    ["paid", "2026-08-04", { amount: 2400 }],
    ["in-delivery", "2026-08-05", {}],
    ["delivered", "2026-08-10", { hours: 12.5 }],
    ["monitoring", "2026-08-11", {}],
  ]) {
    pipeline = advance(pipeline, acmeId, status, at, details);
  }
  pipeline = advance(pipeline, gammaId, "accepted", "2026-08-03");
  pipeline = advance(pipeline, gammaId, "cancelled", "2026-08-04");
  pipeline = advance(pipeline, deltaId, "accepted", "2026-08-03");
  pipeline = advance(pipeline, deltaId, "paid", "2026-08-04", { amount: 3300 });
  pipeline = advance(pipeline, deltaId, "in-delivery", "2026-08-05");
  assert.throws(
    () => advanceEngagement(pipeline, deltaId, "refunded", "2026-08-06", { amount: 3301 }),
    /refund exceeds recorded paid cash/,
  );
  pipeline = advance(pipeline, deltaId, "refunded", "2026-08-06", { amount: 1200 });

  const summary = summarizePipeline(pipeline, "2026-08-31");
  assert.equal(summary.schemaVersion, 2);
  assert.equal(summary.totalEngagements, 4);
  assert.equal(summary.counts.monitoring, 1);
  assert.equal(summary.counts.proposal, 1);
  assert.equal(summary.counts.cancelled, 1);
  assert.equal(summary.counts.refunded, 1);
  assert.deepEqual(summary.funnel, {
    proposals: 4,
    accepted: 3,
    paid: 2,
    delivered: 1,
    monitoring: 1,
    proposalToAccepted: 3 / 4,
    acceptedToPaid: 2 / 3,
    paidToDelivered: 1 / 2,
    deliveredToMonitoring: 1,
  });
  assert.deepEqual(summary.revenue, {
    currency: "USD",
    openProposalValue: 3000,
    bookedReviewRevenue: 2500,
    grossCashCollected: 5700,
    refundedCash: 1200,
    netCashCollected: 4500,
    deliveredContractValue: 2500,
    deliveredNetCash: 2400,
    deliveryHours: 12.5,
    realizedReviewRevenuePerDeliveryHour: 192,
    currentContractedMonthlyRecurringRevenue: 399,
    currentContractedAnnualRecurringRunRate: 4788,
  });
  assert.deepEqual(summary.expiredProposalIds, [betaId]);
  assert.deepEqual(summary.nextActions.map((entry) => [entry.client, entry.status]), [
    ["Acme Capital", "monitoring"],
    ["Beta Trading", "proposal"],
  ]);

  const markdown = renderPipelineSummary(summary);
  assert.match(markdown, /Net cash collected: USD 4,500/);
  assert.match(markdown, /Realized review revenue\/hour: USD 192/);
  assert.match(markdown, /Current contracted monitoring MRR: USD 399/);
  assert.match(markdown, /Proposal → accepted: 75\.0% \(3\/4\)/);
  assert.match(markdown, /Refresh or close expired proposal/);
});

test("pipeline persistence is atomic at the file boundary and round-trips validated state", async () => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-pipeline-"));
  const pipelinePath = path.join(temporaryRoot, "nested", "pipeline.json");
  try {
    const source = engagementFor("Acme Capital");
    const pipeline = importEngagement(emptyPipeline(), source, "2026-08-02").pipeline;
    await savePipeline(pipelinePath, pipeline);
    assert.deepEqual(await loadPipeline(pipelinePath), pipeline);
    assert.deepEqual(await fs.readdir(path.dirname(pipelinePath)), ["pipeline.json"]);
    await assert.rejects(() => loadPipeline(path.join(temporaryRoot, "missing.json")), /ENOENT/);
    assert.deepEqual(
      await loadPipeline(path.join(temporaryRoot, "missing.json"), { allowMissing: true }),
      emptyPipeline(),
    );
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("pipeline CLI carries cash and delivery evidence through import, advance, and summary", async () => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-pipeline-cli-"));
  const pipelinePath = path.join(temporaryRoot, "pipeline.json");
  const engagementPath = path.join(temporaryRoot, "engagement.json");
  const run = (...args) => spawnSync(process.execPath, [PIPELINE_SCRIPT, ...args], { encoding: "utf8" });
  try {
    const engagement = engagementFor("Acme Capital");
    await fs.writeFile(engagementPath, `${JSON.stringify(engagement, null, 2)}\n`);
    const imported = run(
      "import",
      "--engagement",
      engagementPath,
      "--pipeline",
      pipelinePath,
      "--at",
      "2026-08-02",
    );
    assert.equal(imported.status, 0, imported.stderr);
    const id = JSON.parse(imported.stdout).id;

    for (const args of [
      ["advance", "--id", id, "--status", "accepted", "--at", "2026-08-03"],
      ["advance", "--id", id, "--status", "paid", "--amount", "2400", "--at", "2026-08-04"],
      ["advance", "--id", id, "--status", "in-delivery", "--at", "2026-08-05"],
      ["advance", "--id", id, "--status", "delivered", "--hours", "12", "--at", "2026-08-10"],
    ]) {
      const advanced = run(...args, "--pipeline", pipelinePath);
      assert.equal(advanced.status, 0, advanced.stderr);
    }

    const summarized = run("summary", "--pipeline", pipelinePath, "--as-of", "2026-08-10", "--json");
    assert.equal(summarized.status, 0, summarized.stderr);
    const summary = JSON.parse(summarized.stdout);
    assert.equal(summary.schemaVersion, 2);
    assert.equal(summary.revenue.netCashCollected, 2400);
    assert.equal(summary.revenue.deliveryHours, 12);
    assert.equal(summary.revenue.realizedReviewRevenuePerDeliveryHour, 200);
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});
