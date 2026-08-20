import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildCommercialKit,
  parseArgs,
  writeCommercialKit,
} from "../scripts/generate-strategy-assurance-kit.mjs";

const REQUIRED_ARGS = [
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
];

test("commercial-kit arguments produce bounded deterministic scope", () => {
  const config = parseArgs(
    [
      ...REQUIRED_ARGS,
      "--proposal-date",
      "2026-08-14",
      "--start",
      "2026-07-01",
      "--end",
      "2026-07-31",
      "--asset",
      "usdt",
    ],
    Date.UTC(2030, 0, 1),
  );

  assert.equal(config.proposalDate, "2026-08-14");
  assert.equal(config.validThrough, "2026-08-28");
  assert.equal(config.price, 2500);
  assert.equal(config.monitoringPrice, 399);
  assert.equal(config.turnaroundDays, 5);
  assert.equal(config.asset, "USDT");
  assert.match(config.outputDir, /acme-capital-2026-08-14$/);
});

test("commercial-kit arguments reject ambiguous or unsafe commercial inputs", () => {
  const newlineArgs = [...REQUIRED_ARGS];
  newlineArgs[1] = "Injected\nheading";
  assert.throws(() => parseArgs(REQUIRED_ARGS.slice(0, -2)), /--deployment is required/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--start", "2026-02-30"]), /real calendar date/);
  assert.throws(
    () => parseArgs([...REQUIRED_ARGS, "--start", "2026-09-01", "--end", "2026-08-01"]),
    /must not be after/,
  );
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--price", "0"]), /positive number/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--price", "2500.001"]), /two decimal places/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--valid-days", "14.5"]), /whole number/);
  assert.throws(() => parseArgs(newlineArgs), /control characters/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--client", "Duplicate"]), /Duplicate option/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--unknown", "value"]), /Unknown option/);
});

test("commercial kit is tailored, machine-readable, and preserves the credential boundary", () => {
  const config = parseArgs([
    ...REQUIRED_ARGS,
    "--proposal-date",
    "2026-08-14",
    "--repository",
    "ssh://example.invalid/strategy.git",
    "--commit",
    "abc1234",
    "--infrastructure-cost",
    "125 USDT supplied by client",
  ]);
  const kit = buildCommercialKit(config);
  const allMarkdown = [
    kit.files["proposal.md"],
    kit.files["evidence-request.md"],
    kit.files["outreach.md"],
    kit.files["payment-request.md"],
  ].join("\n");

  assert.match(kit.files["proposal.md"], /Acme Capital/);
  assert.match(kit.files["proposal.md"], /USD 2,500, due before kickoff/);
  assert.match(kit.files["outreach.md"], /fixed 5-business-day engineering review/);
  assert.match(kit.files["evidence-request.md"], /Do not send withdrawal-enabled or trading-enabled credentials/);
  assert.doesNotMatch(allMarkdown, /\[[A-Z][A-Z _-]{2,}\]/);

  const engagement = JSON.parse(kit.files["engagement.json"]);
  assert.equal(engagement.schemaVersion, 1);
  assert.equal(
    engagement.engagementId,
    "strategy-assurance-acme-capital-basis-router-v3-production-us-east-2026-08-14",
  );
  assert.equal(engagement.scope.reviewedCommit, "abc1234");
  assert.deepEqual(engagement.accessBoundary.forbidden, [
    "withdrawal",
    "trading",
    "seed-phrase",
    "private-key",
    "unrestricted-cloud",
  ]);
  const paymentRequest = JSON.parse(kit.files["payment-request.json"]);
  assert.equal(paymentRequest.engagementId, engagement.engagementId);
  assert.equal(paymentRequest.amount, 2500);
  assert.equal(paymentRequest.externalPaymentActionPerformed, false);
  assert.match(kit.files["payment-request.md"], /not a tax invoice or receipt/i);
  assert.match(kit.files["payment-request.md"], /Do not place bank credentials/);
});

test("commercial kit writes a complete package and never overwrites without explicit force", async () => {
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-kit-"));
  const outputDir = path.join(temporaryRoot, "client-kit");
  try {
    const config = parseArgs([...REQUIRED_ARGS, "--proposal-date", "2026-08-14", "--output", outputDir]);
    const result = await writeCommercialKit(config);
    assert.deepEqual(result.files, [
      "proposal.md",
      "evidence-request.md",
      "outreach.md",
      "payment-request.md",
      "payment-request.json",
      "engagement.json",
    ]);
    assert.deepEqual((await fs.readdir(outputDir)).sort(), [...result.files].sort());
    await assert.rejects(() => writeCommercialKit(config), /already exists/);

    await fs.writeFile(path.join(outputDir, "proposal.md"), "stale proposal\n");
    const forced = parseArgs([...REQUIRED_ARGS, "--proposal-date", "2026-08-14", "--output", outputDir, "--force"]);
    await writeCommercialKit(forced);
    assert.match(await fs.readFile(path.join(outputDir, "proposal.md"), "utf8"), /Strategy Assurance Proposal/);
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});
