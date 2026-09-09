import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import {
  cp,
  mkdir,
  mkdtemp,
  readFile,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

const repositoryRoot = new URL("..", import.meta.url).pathname;
const script = join(
  repositoryRoot,
  "scripts/research/verify_market_context_bundle.py",
);
const fixture = join(
  repositoryRoot,
  "test/fixtures/market-context-source-v1",
);

function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

function gitAt(root, ...args) {
  return execFileSync("git", args, { cwd: root });
}

function runVerifierAt(verifier, root, status, ...args) {
  return spawnSync(
    "python3",
    [verifier, "verify", "--status", status, ...args],
    { cwd: root, encoding: "utf8" },
  );
}

function runVerifier(status, ...args) {
  return runVerifierAt(script, repositoryRoot, status, ...args);
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

async function buildBundle(commitOverride, gitRoot = repositoryRoot) {
  const root = await mkdtemp(join(tmpdir(), "trader-market-context-bundle-"));
  const bundle = join(root, "bundle");
  await cp(fixture, bundle, { recursive: true });
  const manifestPath = join(bundle, "source-manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const commit = (
    commitOverride === undefined
      ? gitAt(gitRoot, "rev-parse", "HEAD")
      : Buffer.from(commitOverride)
  ).toString("utf8").trim();
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
      gitAt(
        gitRoot,
        "show",
        `${commit}:scripts/research/collect_market_context.py`,
      ),
    ),
    verifierPath: "scripts/research/market_context_source.py",
    verifierSha256: sha256(
      gitAt(
        gitRoot,
        "show",
        `${commit}:scripts/research/market_context_source.py`,
      ),
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
  const statusPath = join(bundle, "collection-status.json");
  await writeFile(statusPath, `${JSON.stringify(status, null, 2)}\n`);
  return { bundle, manifestPath, status, statusPath };
}

test("offline bundle verifier binds complete collector and source evidence", async () => {
  const { bundle, statusPath } = await buildBundle();
  const panel = join(bundle, "derived-panel.csv");
  const receipt = join(bundle, "bundle-receipt.json");
  let result = runVerifier(
    statusPath,
    "--panel-output",
    panel,
    "--receipt-output",
    receipt,
  );
  assert.equal(result.status, 0, result.stderr);
  const firstPanel = await readFile(panel);
  const firstReceipt = await readFile(receipt);
  const parsed = JSON.parse(firstReceipt);
  assert.equal(parsed.schemaId, "binance_usdm_market_context_bundle_receipt_v1");
  assert.equal(parsed.collectionStatusVerified, true);
  assert.equal(parsed.sourceBundleVerified, true);
  assert.equal(parsed.researchAdmission, false);
  assert.equal(parsed.sourceVerificationReceipt.outcomeUse, false);
  for (const field of [
    "experimentUse",
    "holdoutUse",
    "modelUse",
    "promotionUse",
    "deploymentUse",
    "orderUse",
    "liveAuthorizationUse",
  ]) {
    assert.equal(parsed[field], false);
  }
  assert.equal(parsed.panelSha256, sha256(firstPanel));
  assert.match(parsed.collectionStatusSha256, /^[0-9a-f]{64}$/);
  assert.match(parsed.sourceLicenseManifestSha256, /^[0-9a-f]{64}$/);

  result = runVerifier(
    statusPath,
    "--panel-output",
    panel,
    "--receipt-output",
    receipt,
  );
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(await readFile(panel), firstPanel);
  assert.deepEqual(await readFile(receipt), firstReceipt);

  const source = await readFile(script, "utf8");
  assert.doesNotMatch(source, /urllib|urlopen|requests\.|httpx|socket\./);
  assert.match(source, /This module is deliberately offline/);
  assert.doesNotMatch(source, /collect_bundle\s*\(/);
});

test("bundle verifier rejects incomplete, changed, or authorizing status", async () => {
  for (const state of [
    "collecting",
    "partial_failure",
    "cleanup_pending",
    "cleanup_failure",
  ]) {
    const { status, statusPath } = await buildBundle();
    status.state = state;
    await writeFile(statusPath, `${JSON.stringify(status)}\n`);
    const result = runVerifier(statusPath);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /not complete_unverified/);
  }

  const authorizing = await buildBundle();
  authorizing.status.orderUse = true;
  await writeFile(
    authorizing.statusPath,
    `${JSON.stringify(authorizing.status)}\n`,
  );
  let result = runVerifier(authorizing.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /forbidden downstream authority/);

  const changedManifest = await buildBundle();
  const manifest = JSON.parse(await readFile(changedManifest.manifestPath, "utf8"));
  manifest.decisionTime += 1;
  await writeFile(changedManifest.manifestPath, `${JSON.stringify(manifest)}\n`);
  result = runVerifier(changedManifest.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /source-manifest SHA-256 changed/);

  const wrongBlob = await buildBundle();
  wrongBlob.status.collectorSha256 = "0".repeat(64);
  await writeFile(wrongBlob.statusPath, `${JSON.stringify(wrongBlob.status)}\n`);
  result = runVerifier(wrongBlob.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /recorded Git commit/);

  const incomplete = await buildBundle();
  incomplete.status.completedArtifactPaths.pop();
  await writeFile(incomplete.statusPath, `${JSON.stringify(incomplete.status)}\n`);
  result = runVerifier(incomplete.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /artifact inventory disagrees/);

  const boolCount = await buildBundle();
  boolCount.status.eligiblePopulationCount = true;
  await writeFile(boolCount.statusPath, `${JSON.stringify(boolCount.status)}\n`);
  result = runVerifier(boolCount.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /eligiblePopulationCount must be/);

  const driftRoot = await mkdtemp(join(tmpdir(), "trader-bundle-drift-repo-"));
  const trackedPaths = [
    "scripts/research/collect_market_context.py",
    "scripts/research/market_context_source.py",
    "scripts/research/verify_market_context_bundle.py",
    "research-notes/market-prediction-2026-09-04/data-source-license-manifest.json",
  ];
  for (const path of trackedPaths) {
    const destination = join(driftRoot, path);
    await mkdir(dirname(destination), { recursive: true });
    await cp(join(repositoryRoot, path), destination);
  }
  gitAt(driftRoot, "init", "--quiet");
  gitAt(driftRoot, "add", ...trackedPaths);
  gitAt(
    driftRoot,
    "-c",
    "commit.gpgsign=false",
    "-c",
    "user.name=CI Fixture",
    "-c",
    "user.email=ci-fixture@example.invalid",
    "commit",
    "--quiet",
    "-m",
    "fixture",
  );
  const collectionCommit = gitAt(driftRoot, "rev-parse", "HEAD");
  const driftScript = join(
    driftRoot,
    "scripts/research/verify_market_context_bundle.py",
  );
  await writeFile(
    driftScript,
    Buffer.concat([
      await readFile(driftScript),
      Buffer.from("\n# simulated post-collection verifier drift\n"),
    ]),
  );
  const versionDrift = await buildBundle(collectionCommit, driftRoot);
  result = runVerifierAt(
    driftScript,
    driftRoot,
    versionDrift.statusPath,
  );
  assert.equal(result.status, 1);
  assert.match(result.stderr, /bundle-verifier bytes disagree/);
});

test("bundle verifier enforces strict status and protects frozen inputs", async () => {
  const duplicate = await buildBundle();
  const validStatus = await readFile(duplicate.statusPath, "utf8");
  await writeFile(
    duplicate.statusPath,
    `{\n  "schemaId": "duplicate",${validStatus.slice(1)}`,
  );
  let result = runVerifier(duplicate.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /duplicate JSON key/);

  const nonFinite = await buildBundle();
  const nonFiniteText = await readFile(nonFinite.statusPath, "utf8");
  await writeFile(
    nonFinite.statusPath,
    nonFiniteText.replace(/"completedAtMs": \d+/, '"completedAtMs": NaN'),
  );
  result = runVerifier(nonFinite.statusPath);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /non-finite JSON constant/);

  const linked = await buildBundle();
  const linkedStatus = join(linked.bundle, "linked-collection-status.json");
  await symlink(linked.statusPath, linkedStatus);
  result = runVerifier(linkedStatus);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /non-symlink/);

  const protectedBundle = await buildBundle();
  const before = await readFile(protectedBundle.statusPath);
  result = runVerifier(
    protectedBundle.statusPath,
    "--panel-output",
    join(protectedBundle.bundle, "derived.csv"),
    "--receipt-output",
    protectedBundle.statusPath,
  );
  assert.equal(result.status, 1);
  assert.match(result.stderr, /cannot overwrite frozen bundle evidence/);
  assert.deepEqual(await readFile(protectedBundle.statusPath), before);
});
