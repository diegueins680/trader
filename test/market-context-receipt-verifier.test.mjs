import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import {
  appendFile,
  cp,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const repositoryRoot = new URL("..", import.meta.url).pathname;
const bundleScript = join(
  repositoryRoot,
  "scripts/research/verify_market_context_bundle.py",
);
const receiptScript = join(
  repositoryRoot,
  "scripts/research/verify_market_context_receipt.py",
);
const fixture = join(
  repositoryRoot,
  "test/fixtures/market-context-source-v1",
);

function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

function git(...args) {
  return execFileSync("git", args, { cwd: repositoryRoot });
}

function run(script, args) {
  return spawnSync("python3", [script, ...args], {
    cwd: repositoryRoot,
    encoding: "utf8",
  });
}

function artifactPaths(manifest) {
  return [
    manifest.artifacts.serverTimeBefore.logicalPath,
    manifest.artifacts.exchangeInfo.logicalPath,
    manifest.artifacts.ticker24hr.logicalPath,
    ...manifest.artifacts.peerKlines.map(({ logicalPath }) => logicalPath),
    manifest.artifacts.serverTimeAfter.logicalPath,
  ];
}

async function buildEvidence() {
  const root = await mkdtemp(join(tmpdir(), "trader-market-context-receipt-"));
  const archive = join(root, "archive");
  await cp(fixture, archive, { recursive: true });
  await rm(join(archive, "expected-panel.csv"));
  await rm(join(archive, "expected-receipt.json"));
  const manifestPath = join(archive, "source-manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const commit = git("rev-parse", "HEAD").toString("utf8").trim();
  manifest.codeCommit = commit;
  const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(manifestPath, manifestBytes);
  const status = {
    schemaId: "binance_usdm_market_context_collection_status_v1",
    schemaVersion: 1,
    sourceId: "binance-usdm-public-market-data",
    startedAtMs: manifest.startedAtMs - 10,
    codeCommit: commit,
    provenanceTrackedClean: true,
    collectorPath: "scripts/research/collect_market_context.py",
    collectorSha256: sha256(
      git("show", `${commit}:scripts/research/collect_market_context.py`),
    ),
    verifierPath: "scripts/research/market_context_source.py",
    verifierSha256: sha256(
      git("show", `${commit}:scripts/research/market_context_source.py`),
    ),
    runtime: { python: "3.13.7" },
    outcomeUse: false,
    modelUse: false,
    orderUse: false,
    liveAuthorizationUse: false,
    state: "complete_unverified",
    completedAtMs: manifest.decisionTime + 10,
    completedArtifactPaths: artifactPaths(manifest),
    sourceManifestPublished: true,
    sourceManifestSha256: sha256(manifestBytes),
    eligiblePopulationCount: manifest.population.eligibleSymbols.length,
  };
  const statusPath = join(archive, "collection-status.json");
  await writeFile(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  const receipt = join(root, "bundle-receipt.json");
  const panel = join(root, "derived-panel.csv");
  const generated = run(bundleScript, [
    "verify",
    "--status",
    statusPath,
    "--panel-output",
    panel,
    "--receipt-output",
    receipt,
  ]);
  assert.equal(generated.status, 0, generated.stderr);
  return { archive, manifest, panel, receipt, root, statusPath };
}

function verify(receipt, archive) {
  return run(receiptScript, ["--receipt", receipt, "--archive", archive]);
}

test("market-context receipt replay binds the exact frozen archive", async () => {
  const { archive, panel, receipt } = await buildEvidence();
  const first = verify(receipt, archive);
  assert.equal(first.status, 0, first.stderr);
  const result = JSON.parse(first.stdout);
  assert.equal(
    result.schemaId,
    "binance_usdm_market_context_receipt_verification_v1",
  );
  assert.equal(result.collectionStatusVerified, true);
  assert.equal(result.sourceBundleVerified, true);
  assert.equal(result.panelSha256, sha256(await readFile(panel)));
  assert.ok(result.archiveFilesVerified > 2);
  assert.ok(result.archiveBytesVerified > 0);
  for (const field of [
    "researchAdmission",
    "experimentUse",
    "holdoutUse",
    "modelUse",
    "promotionUse",
    "deploymentUse",
    "orderUse",
    "liveAuthorizationUse",
  ]) {
    assert.equal(result[field], false);
  }
  const second = verify(receipt, archive);
  assert.equal(second.status, 0, second.stderr);
  assert.equal(second.stdout, first.stdout);

  const sourceText = await readFile(receiptScript, "utf8");
  assert.match(sourceText, /deliberately offline and read-only/);
  assert.doesNotMatch(sourceText, /urllib|urlopen|requests\.|httpx|socket\./);
  assert.doesNotMatch(sourceText, /collect_market_context/);
});

test("market-context receipt replay rejects changed archive evidence", async () => {
  const changedRaw = await buildEvidence();
  const rawPath = join(
    changedRaw.archive,
    changedRaw.manifest.artifacts.exchangeInfo.logicalPath,
  );
  await appendFile(rawPath, "\n");
  let result = verify(changedRaw.receipt, changedRaw.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /byte count changed|SHA-256 changed/);

  const changedStatus = await buildEvidence();
  await appendFile(changedStatus.statusPath, "\n");
  result = verify(changedStatus.receipt, changedStatus.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /status SHA-256 disagrees/);

  const extra = await buildEvidence();
  await writeFile(join(extra.archive, "unregistered.json"), "{}\n");
  result = verify(extra.receipt, extra.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /inventory is incomplete or contains extras/);
});

test("market-context receipt replay rejects malformed or authorizing receipts", async () => {
  const authorizing = await buildEvidence();
  let value = JSON.parse(await readFile(authorizing.receipt, "utf8"));
  value.orderUse = true;
  await writeFile(authorizing.receipt, `${JSON.stringify(value)}\n`);
  let result = verify(authorizing.receipt, authorizing.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /forbidden downstream authority/);

  const wrongVerifier = await buildEvidence();
  value = JSON.parse(await readFile(wrongVerifier.receipt, "utf8"));
  value.bundleVerifierSha256 = "0".repeat(64);
  await writeFile(wrongVerifier.receipt, `${JSON.stringify(value)}\n`);
  result = verify(wrongVerifier.receipt, wrongVerifier.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /not bound to the collection commit/);

  const wrongCount = await buildEvidence();
  value = JSON.parse(await readFile(wrongCount.receipt, "utf8"));
  value.panelRows = true;
  await writeFile(wrongCount.receipt, `${JSON.stringify(value)}\n`);
  result = verify(wrongCount.receipt, wrongCount.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /positive integer/);

  const forgedPanel = await buildEvidence();
  value = JSON.parse(await readFile(forgedPanel.receipt, "utf8"));
  value.panelSha256 = "1".repeat(64);
  await writeFile(forgedPanel.receipt, `${JSON.stringify(value)}\n`);
  result = verify(forgedPanel.receipt, forgedPanel.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /disagrees with replayed verification/);

  const unknown = await buildEvidence();
  value = JSON.parse(await readFile(unknown.receipt, "utf8"));
  value.unregistered = false;
  await writeFile(unknown.receipt, `${JSON.stringify(value)}\n`);
  result = verify(unknown.receipt, unknown.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /keys are incompatible/);
});

test("market-context receipt replay rejects ambiguous paths and JSON", async () => {
  const duplicate = await buildEvidence();
  const valid = await readFile(duplicate.receipt, "utf8");
  await writeFile(
    duplicate.receipt,
    `{\n  "schemaId": "duplicate",${valid.slice(1)}`,
  );
  let result = verify(duplicate.receipt, duplicate.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /duplicate JSON key/);

  const nonFinite = await buildEvidence();
  const finiteText = await readFile(nonFinite.receipt, "utf8");
  await writeFile(
    nonFinite.receipt,
    finiteText.replace(/"panelRows": \d+/, '"panelRows": NaN'),
  );
  result = verify(nonFinite.receipt, nonFinite.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /non-finite JSON constant/);

  const linkedReceipt = await buildEvidence();
  const receiptLink = join(linkedReceipt.root, "linked-receipt.json");
  await symlink(linkedReceipt.receipt, receiptLink);
  result = verify(receiptLink, linkedReceipt.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /non-symlink file/);

  const linkedArchive = await buildEvidence();
  const archiveLink = join(linkedArchive.root, "linked-archive");
  await symlink(linkedArchive.archive, archiveLink);
  result = verify(linkedArchive.receipt, archiveLink);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /non-symlink directory/);

  const receiptInside = await buildEvidence();
  const insidePath = join(receiptInside.archive, "bundle-receipt.json");
  await cp(receiptInside.receipt, insidePath);
  result = verify(insidePath, receiptInside.archive);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /inventory is incomplete or contains extras/);
});
