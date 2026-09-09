import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const repositoryRoot = new URL("..", import.meta.url).pathname;
const script = join(repositoryRoot, "scripts/research/market_context_source.py");
const fixture = join(repositoryRoot, "test/fixtures/market-context-source-v1");

function runVerifier(manifest, ...args) {
  return spawnSync(
    "python3",
    [script, "verify", "--manifest", manifest, ...args],
    { cwd: repositoryRoot, encoding: "utf8" },
  );
}

async function fixtureCopy() {
  const root = await mkdtemp(join(tmpdir(), "trader-market-context-source-"));
  const copy = join(root, "fixture");
  await cp(fixture, copy, { recursive: true });
  return copy;
}

async function readManifest(root) {
  return JSON.parse(await readFile(join(root, "source-manifest.json"), "utf8"));
}

async function writeManifest(root, manifest) {
  await writeFile(
    join(root, "source-manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
}

function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

test("market-context source verifier deterministically derives the Haskell panel boundary", async () => {
  const root = await fixtureCopy();
  const manifest = join(root, "source-manifest.json");
  const panel = join(root, "derived-panel.csv");
  const receipt = join(root, "verification-receipt.json");
  const expectedPanel = await readFile(join(fixture, "expected-panel.csv"));
  const expectedReceipt = await readFile(join(fixture, "expected-receipt.json"));

  const first = runVerifier(
    manifest,
    "--panel-output",
    panel,
    "--receipt-output",
    receipt,
  );
  assert.equal(first.status, 0, first.stderr);
  assert.deepEqual(await readFile(panel), expectedPanel);
  assert.deepEqual(await readFile(receipt), expectedReceipt);
  const parsedReceipt = JSON.parse(first.stdout);
  assert.deepEqual(parsedReceipt, JSON.parse(expectedReceipt));
  assert.equal(parsedReceipt.verifierPath, "scripts/research/market_context_source.py");
  assert.match(parsedReceipt.verifierSha256, /^[0-9a-f]{64}$/);
  assert.match(parsedReceipt.sourceManifestCodeCommit, /^[0-9a-f]{40}$/);
  assert.equal(parsedReceipt.outcomeUse, false);
  assert.equal(parsedReceipt.modelUse, false);
  assert.equal(parsedReceipt.orderUse, false);
  assert.equal(parsedReceipt.liveAuthorizationUse, false);

  const second = runVerifier(
    manifest,
    "--panel-output",
    panel,
    "--receipt-output",
    receipt,
  );
  assert.equal(second.status, 0, second.stderr);
  assert.deepEqual(await readFile(panel), expectedPanel);
  assert.deepEqual(await readFile(receipt), expectedReceipt);

  const source = await readFile(script, "utf8");
  assert.doesNotMatch(source, /urllib|urlopen|requests\.|httpx|socket\./);
  assert.match(source, /This module is deliberately offline/);
});

test("market-context source verifier rejects tampering, partial population, and non-causal evidence", async () => {
  const tampered = await fixtureCopy();
  await writeFile(join(tampered, "raw/ticker-24hr.json"), "[]\n");
  let result = runVerifier(join(tampered, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /ticker24hr (byte count|SHA-256) changed/);

  const incomplete = await fixtureCopy();
  const incompleteManifest = await readManifest(incomplete);
  incompleteManifest.population.eligibleSymbols.pop();
  await writeManifest(incomplete, incompleteManifest);
  result = runVerifier(join(incomplete, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /population disagrees with exchangeInfo/);

  const changedParams = await fixtureCopy();
  const changedParamsManifest = await readManifest(changedParams);
  changedParamsManifest.artifacts.peerKlines[0].params.limit = 1;
  await writeManifest(changedParams, changedParamsManifest);
  result = runVerifier(join(changedParams, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /endpoint or parameters changed/);

  const missingWeight = await fixtureCopy();
  const missingWeightManifest = await readManifest(missingWeight);
  missingWeightManifest.artifacts.exchangeInfo.usedWeight1m = 0;
  await writeManifest(missingWeight, missingWeightManifest);
  result = runVerifier(join(missingWeight, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /usedWeight1m must be an integer >= 1/);

  const late = await fixtureCopy();
  const lateManifest = await readManifest(late);
  lateManifest.decisionTime = lateManifest.completedAtMs - 1;
  await writeManifest(late, lateManifest);
  result = runVerifier(join(late, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /decision is unavailable/);

  const extraRaw = await fixtureCopy();
  await writeFile(join(extraRaw, "raw/unregistered.json"), "{}\n");
  result = runVerifier(join(extraRaw, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /unregistered files/);

  const nestedRaw = await fixtureCopy();
  await mkdir(join(nestedRaw, "raw/unregistered"));
  result = runVerifier(join(nestedRaw, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /unregistered files/);

  const misaligned = await fixtureCopy();
  const misalignedManifest = await readManifest(misaligned);
  misalignedManifest.barOpenTime += 1;
  misalignedManifest.barEndTime += 1;
  await writeManifest(misaligned, misalignedManifest);
  result = runVerifier(join(misaligned, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /bar window is incompatible/);
});

test("market-context source verifier rejects strict-JSON and market-value failures after matching hashes", async () => {
  const invalidJson = await fixtureCopy();
  const invalidJsonManifest = await readManifest(invalidJson);
  const invalidServer = Buffer.from('{"serverTime":NaN}\n');
  await writeFile(join(invalidJson, "raw/server-time-before.json"), invalidServer);
  invalidJsonManifest.artifacts.serverTimeBefore.bytes = invalidServer.length;
  invalidJsonManifest.artifacts.serverTimeBefore.sha256 = sha256(invalidServer);
  await writeManifest(invalidJson, invalidJsonManifest);
  let result = runVerifier(join(invalidJson, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /non-finite JSON constant is prohibited/);

  const nonFiniteTicker = await fixtureCopy();
  const tickerManifest = await readManifest(nonFiniteTicker);
  const tickerPath = join(nonFiniteTicker, "raw/ticker-24hr.json");
  const ticker = JSON.parse(await readFile(tickerPath, "utf8"));
  ticker[0].quoteVolume = "NaN";
  const tickerBytes = Buffer.from(`${JSON.stringify(ticker)}\n`);
  await writeFile(tickerPath, tickerBytes);
  tickerManifest.artifacts.ticker24hr.bytes = tickerBytes.length;
  tickerManifest.artifacts.ticker24hr.sha256 = sha256(tickerBytes);
  await writeManifest(nonFiniteTicker, tickerManifest);
  result = runVerifier(join(nonFiniteTicker, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /quoteVolume must be finite/);

  const staleTicker = await fixtureCopy();
  const staleTickerManifest = await readManifest(staleTicker);
  const staleTickerPath = join(staleTicker, "raw/ticker-24hr.json");
  const staleTickerRows = JSON.parse(await readFile(staleTickerPath, "utf8"));
  staleTickerRows.find(({ symbol }) => symbol === "DOGEUSDT").closeTime =
    staleTickerManifest.artifacts.ticker24hr.requestStartedAtMs -
    staleTickerManifest.maxClockSkewMs -
    1;
  const staleTickerBytes = Buffer.from(`${JSON.stringify(staleTickerRows)}\n`);
  await writeFile(staleTickerPath, staleTickerBytes);
  staleTickerManifest.artifacts.ticker24hr.bytes = staleTickerBytes.length;
  staleTickerManifest.artifacts.ticker24hr.sha256 = sha256(staleTickerBytes);
  await writeManifest(staleTicker, staleTickerManifest);
  result = runVerifier(join(staleTicker, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /ticker event exceeds the declared collection window/);

  const wrongGrid = await fixtureCopy();
  const wrongGridManifest = await readManifest(wrongGrid);
  const klinePath = join(wrongGrid, "raw/BTCUSDT-klines.json");
  const klines = JSON.parse(await readFile(klinePath, "utf8"));
  klines[1][0] += 1;
  const klineBytes = Buffer.from(`${JSON.stringify(klines)}\n`);
  await writeFile(klinePath, klineBytes);
  wrongGridManifest.artifacts.peerKlines[0].bytes = klineBytes.length;
  wrongGridManifest.artifacts.peerKlines[0].sha256 = sha256(klineBytes);
  await writeManifest(wrongGrid, wrongGridManifest);
  result = runVerifier(join(wrongGrid, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /peer kline grid changed/);

  const wrongShape = await fixtureCopy();
  const wrongShapeManifest = await readManifest(wrongShape);
  const shapePath = join(wrongShape, "raw/BTCUSDT-klines.json");
  const shapeKlines = JSON.parse(await readFile(shapePath, "utf8"));
  shapeKlines[1].push("unexpected");
  const shapeBytes = Buffer.from(`${JSON.stringify(shapeKlines)}\n`);
  await writeFile(shapePath, shapeBytes);
  wrongShapeManifest.artifacts.peerKlines[0].bytes = shapeBytes.length;
  wrongShapeManifest.artifacts.peerKlines[0].sha256 = sha256(shapeBytes);
  await writeManifest(wrongShape, wrongShapeManifest);
  result = runVerifier(join(wrongShape, "source-manifest.json"));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /peer kline 1 is malformed/);
});

test("market-context source verifier cannot overwrite raw evidence or its source manifest", async () => {
  const root = await fixtureCopy();
  const manifest = join(root, "source-manifest.json");
  const rawTicker = join(root, "raw/ticker-24hr.json");
  const before = await readFile(rawTicker);
  const beforeManifest = await readFile(manifest);
  const receipt = join(root, "receipt.json");

  let result = runVerifier(
    manifest,
    "--panel-output",
    rawTicker,
    "--receipt-output",
    receipt,
  );
  assert.equal(result.status, 1);
  assert.match(result.stderr, /cannot overwrite/);
  assert.deepEqual(await readFile(rawTicker), before);

  result = runVerifier(
    manifest,
    "--panel-output",
    manifest,
    "--receipt-output",
    receipt,
  );
  assert.equal(result.status, 1);
  assert.match(result.stderr, /cannot overwrite/);
  assert.deepEqual(await readFile(manifest), beforeManifest);
});
