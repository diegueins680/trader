import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  addBusinessDays,
  buildOutreachCampaign,
  parseArgs,
  validateProspectQueue,
  writeOutreachCampaign,
} from "../scripts/generate-strategy-assurance-outreach.mjs";

const REPOSITORY_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const QUEUE_PATH = path.join(REPOSITORY_ROOT, "docs", "strategy-assurance-prospect-queue.json");
const CHECKLIST_PATH = path.join(REPOSITORY_ROOT, "docs", "strategy-assurance-pre-live-checklist.md");

async function fixtures() {
  return {
    queue: JSON.parse(await fs.readFile(QUEUE_PATH, "utf8")),
    checklist: await fs.readFile(CHECKLIST_PATH, "utf8"),
  };
}

const REQUIRED_ARGS = [
  "--provider",
  "Assurance Labs",
  "--sender",
  "Avery Chen, Principal",
  "--prospect",
  "OctoBot",
  "--campaign-date",
  "2026-08-17",
];

test("outreach arguments bound campaign size, identity, and dates", () => {
  const config = parseArgs(REQUIRED_ARGS, Date.UTC(2030, 0, 1));
  assert.equal(config.campaignDate, "2026-08-17");
  assert.deepEqual(config.prospectNames, ["OctoBot"]);
  assert.equal(config.includeWatchlist, false);
  assert.match(config.outputDir, /2026-08-17-octobot$/);
  assert.equal(addBusinessDays("2026-08-17", 5), "2026-08-24");
  assert.equal(addBusinessDays("2026-08-14", 5), "2026-08-21");
});

test("outreach arguments reject unsafe or high-volume preparation", () => {
  const invalidDateArgs = [...REQUIRED_ARGS];
  invalidDateArgs[invalidDateArgs.length - 1] = "2026-02-30";
  const newlineArgs = [...REQUIRED_ARGS];
  newlineArgs[1] = "Injected\nheading";
  assert.throws(() => parseArgs(REQUIRED_ARGS.slice(0, 4)), /at least one --prospect/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--prospect", "OctoBot"]), /must be unique/);
  assert.throws(
    () =>
      parseArgs([
        ...REQUIRED_ARGS,
        "--prospect",
        "Hummingbot Botcamp",
        "--prospect",
        "QuantConnect Integration Partners",
        "--prospect",
        "Jesse",
      ]),
    /no more than 3/,
  );
  assert.throws(() => parseArgs(invalidDateArgs), /real calendar date/);
  assert.throws(() => parseArgs(newlineArgs), /control characters/);
  assert.throws(() => parseArgs([...REQUIRED_ARGS, "--unknown", "value"]), /Unknown option/);
});

test("the checked-in prospect queue is coherent and produces review-only drafts", async () => {
  const { queue, checklist } = await fixtures();
  const validated = validateProspectQueue(queue);
  assert.equal(validated.prospects.length, 9);
  const config = parseArgs(REQUIRED_ARGS);
  const built = buildOutreachCampaign(queue, checklist, config);
  const record = JSON.parse(built.files["octobot/record.json"]);

  assert.equal(built.campaign.externalActionsPerformed, false);
  assert.deepEqual(built.campaign.actionsNotPerformed, [
    "message-send",
    "form-submission",
    "community-join",
    "affiliation-claim",
    "member-data-collection",
    "pipeline-transition",
  ]);
  assert.equal(record.status, "prepared");
  assert.equal(record.earliestFollowUpDate, "2026-08-24");
  assert.equal(record.manualOutcome.sentAt, null);
  assert.match(built.files["octobot/initial.md"], /DRAFT — HUMAN REVIEW REQUIRED — NOT SENT/);
  assert.match(built.files["octobot/initial.md"], /Would the team evaluate one co-branded readiness-review pilot/);
  assert.match(built.files["octobot/initial.md"], /not investment advice, a certification, or a return guarantee/);
  assert.match(built.files["octobot/follow-up.md"], /No follow-up has been sent/);
  assert.match(built.files["pre-live-checklist.md"], /Minimum go-live boundary/);
});

test("outreach preparation rejects unknown and unacknowledged watchlist prospects", async () => {
  const { queue, checklist } = await fixtures();
  const unknownArgs = [...REQUIRED_ARGS];
  unknownArgs[5] = "Unknown org";
  assert.throws(
    () => buildOutreachCampaign(queue, checklist, parseArgs(unknownArgs)),
    /unknown prospect organization/,
  );
  const watchlistArgs = [
    "--provider",
    "Assurance Labs",
    "--sender",
    "Avery Chen, Principal",
    "--prospect",
    "Enflux",
    "--campaign-date",
    "2026-08-17",
  ];
  assert.throws(() => buildOutreachCampaign(queue, checklist, parseArgs(watchlistArgs)), /watchlist prospect/);
  assert.equal(
    buildOutreachCampaign(queue, checklist, parseArgs([...watchlistArgs, "--include-watchlist"])).campaign.prospects[0]
      .organization,
    "Enflux",
  );
});

test("outreach writer creates a complete package and limits forced replacement to the same campaign", async () => {
  const { queue, checklist } = await fixtures();
  const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), "trader-assurance-outreach-"));
  const outputDir = path.join(temporaryRoot, "campaign");
  try {
    const config = parseArgs([...REQUIRED_ARGS, "--output", outputDir]);
    const result = await writeOutreachCampaign(queue, checklist, config);
    assert.deepEqual(result.files, [
      "campaign.json",
      "octobot/follow-up.md",
      "octobot/initial.md",
      "octobot/record.json",
      "pre-live-checklist.md",
    ]);
    await assert.rejects(() => writeOutreachCampaign(queue, checklist, config), /already exists/);

    const forced = parseArgs([...REQUIRED_ARGS, "--output", outputDir, "--force"]);
    await writeOutreachCampaign(queue, checklist, forced);
    assert.match(await fs.readFile(path.join(outputDir, "octobot", "initial.md"), "utf8"), /No message has been sent/);

    const mismatched = parseArgs([
      "--provider",
      "Assurance Labs",
      "--sender",
      "Avery Chen, Principal",
      "--prospect",
      "Jesse",
      "--campaign-date",
      "2026-08-17",
      "--output",
      outputDir,
      "--force",
    ]);
    await assert.rejects(() => writeOutreachCampaign(queue, checklist, mismatched), /same campaignId/);
  } finally {
    await fs.rm(temporaryRoot, { recursive: true, force: true });
  }
});
