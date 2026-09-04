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
  savePipeline,
} from "../scripts/strategy-assurance-pipeline.mjs";
import {
  buildRenewalOffer,
  parseArgs,
  writeRenewalOffer,
} from "../scripts/generate-strategy-assurance-renewal.mjs";

const RENEWAL_SCRIPT = fileURLToPath(new URL("../scripts/generate-strategy-assurance-renewal.mjs", import.meta.url));

function proposalEngagement(extraArgs = []) {
  return buildCommercialKit(
    parseKitArgs([
      "--client",
      "Acme Capital",
      "--provider",
      "Assurance Labs",
      "--decision-owner",
      "Avery Chen, CTO",
      "--strategy",
      "Basis Router v3",
      "--deployment",
      "production-us-east",
      "--proposal-date",
      "2026-08-01",
      ...extraArgs,
    ]),
  ).engagement;
}

function deliveredPipeline(extraArgs = []) {
  const imported = importEngagement(emptyPipeline(), proposalEngagement(extraArgs), "2026-08-02");
  const id = imported.engagement.id;
  let pipeline = advanceEngagement(imported.pipeline, id, "accepted", "2026-08-03").pipeline;
  pipeline = advanceEngagement(pipeline, id, "paid", "2026-08-04", { amount: 2500 }).pipeline;
  pipeline = advanceEngagement(pipeline, id, "in-delivery", "2026-08-05").pipeline;
  pipeline = advanceEngagement(pipeline, id, "delivered", "2026-08-10", { hours: 10 }).pipeline;
  return { pipeline, id };
}

test("renewal arguments bound offer validity, cycles, dates, and output", () => {
  const config = parseArgs(
    ["--id", "engagement-1", "--start", "2026-09-01", "--months", "6"],
    Date.UTC(2026, 7, 15),
  );
  assert.equal(config.offerDate, "2026-08-15");
  assert.equal(config.validThrough, "2026-08-29");
  assert.equal(config.months, 6);
  assert.match(config.outputDir, /renewals\/engagement-1-2026-09-01$/);

  assert.throws(() => parseArgs(["--id", "x", "--start", "2026-08-14"], Date.UTC(2026, 7, 15)), /must not precede/);
  assert.throws(
    () => parseArgs(["--id", "x", "--start", "2026-09-01", "--months", "0"], Date.UTC(2026, 7, 15)),
    /whole number/,
  );
  assert.throws(() => parseArgs(["--start", "2026-09-01"]), /--id is required/);
});

test("renewal offer uses delivered scope and exact contracted monitoring value", () => {
  const { pipeline, id } = deliveredPipeline();
  const config = parseArgs(
    ["--id", id, "--offer-date", "2026-08-15", "--start", "2026-09-01", "--months", "3"],
  );
  const built = buildRenewalOffer(pipeline, config);

  assert.equal(built.offer.parentEngagementId, id);
  assert.equal(built.offer.monthlyPrice, 399);
  assert.equal(built.offer.initialContractValue, 1197);
  assert.equal(built.offer.externalActionPerformed, false);
  assert.match(built.files["monitoring-order.md"], /Initial contract value: USD 1,197/);
  assert.match(built.files["monitoring-order.md"], /does not advance the pipeline or perform billing/);
  assert.match(built.files["monitoring-order.md"], /Never provide withdrawal or trading permission/);

  const custom = deliveredPipeline(["--monitoring-price", "399.99"]);
  const customBuilt = buildRenewalOffer(custom.pipeline, { ...config, id: custom.id });
  assert.equal(customBuilt.offer.initialContractValue, 1199.97);

  const proposal = importEngagement(emptyPipeline(), proposalEngagement(), "2026-08-02").pipeline;
  assert.throws(() => buildRenewalOffer(proposal, config), /must be delivered before offering monitoring/);
});

test("renewal writer creates a complete package and refuses implicit overwrite", async () => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-renewal-"));
  const outputDir = path.join(temporaryRoot, "offer");
  try {
    const { pipeline, id } = deliveredPipeline();
    const config = parseArgs([
      "--id",
      id,
      "--offer-date",
      "2026-08-15",
      "--start",
      "2026-09-01",
      "--output",
      outputDir,
    ]);
    const result = await writeRenewalOffer(pipeline, config);
    assert.deepEqual(result.files, ["monitoring-order.md", "monitoring-order.json"]);
    assert.deepEqual((await fs.readdir(outputDir)).sort(), [...result.files].sort());
    await assert.rejects(() => writeRenewalOffer(pipeline, config), /already exists/);

    const forced = parseArgs([
      "--id",
      id,
      "--offer-date",
      "2026-08-15",
      "--start",
      "2026-09-01",
      "--output",
      outputDir,
      "--force",
    ]);
    await writeRenewalOffer(pipeline, forced);
    assert.match(await fs.readFile(path.join(outputDir, "monitoring-order.md"), "utf8"), /Monitoring Order/);
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("renewal CLI reads the local pipeline and writes an offer without mutating lifecycle state", async () => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-renewal-cli-"));
  const pipelinePath = path.join(temporaryRoot, "pipeline.json");
  const outputDir = path.join(temporaryRoot, "renewal");
  try {
    const { pipeline, id } = deliveredPipeline();
    await savePipeline(pipelinePath, pipeline);
    const result = spawnSync(
      process.execPath,
      [
        RENEWAL_SCRIPT,
        "--id",
        id,
        "--pipeline",
        pipelinePath,
        "--offer-date",
        "2026-08-15",
        "--start",
        "2026-09-01",
        "--output",
        outputDir,
      ],
      { encoding: "utf8" },
    );
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).files.length, 2);
    const persisted = JSON.parse(await fs.readFile(pipelinePath, "utf8"));
    assert.equal(persisted.engagements[0].currentStatus, "delivered");
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});
