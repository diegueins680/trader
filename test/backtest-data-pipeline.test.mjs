import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const repoRoot = path.resolve(import.meta.dirname, "..");
const pipeline = path.join(repoRoot, "scripts", "fetch-data-pipeline.sh");
const endTimeMs = 28_799_999;

const baseRows = [
  [0, "100", "101", "99", "100.5", "12", 14_399_999, "1200", 5, "6", "600", "0"],
  [14_400_000, "100.5", "102", "100", "101", "13", endTimeMs, "1313", 7, "7", "707", "0"],
];

function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

function runPipeline(env) {
  return new Promise((resolve, reject) => {
    const child = spawn("bash", [pipeline], {
      cwd: repoRoot,
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.setEncoding("utf8").on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (status) => resolve({ status, stdout, stderr }));
  });
}

async function withKlineServer(run) {
  let rows = structuredClone(baseRows);
  const requests = [];
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    requests.push(url);
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(rows));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.equal(typeof address, "object");
  try {
    await run({
      baseUrl: `http://127.0.0.1:${address.port}/api/v3/klines`,
      requests,
      setRows(nextRows) {
        rows = structuredClone(nextRows);
      },
    });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test("backtest data pipeline is fixed-window deterministic and hash-bound", async () => {
  await withKlineServer(async ({ baseUrl, requests }) => {
    const dataDir = await mkdtemp(path.join(os.tmpdir(), "trader-backtest-data-"));
    const env = {
      BINANCE_KLINES_URL: baseUrl,
      DATA_DIR: dataDir,
      END_TIME_MS: String(endTimeMs),
      KLINE_LIMIT: "2",
      SYMBOLS: "BTCUSDT",
    };

    const first = await runPipeline(env);
    assert.equal(first.status, 0, first.stderr);
    assert.match(first.stdout, /source_rows_sha256=[a-f0-9]{64}/);
    assert.match(first.stdout, /csv_sha256=[a-f0-9]{64}/);
    assert.match(first.stdout, /randomness=none/);

    const csvPath = path.join(dataDir, "BTCUSDT-4h-2.csv");
    const manifestPath = path.join(dataDir, "backtest-data-manifest-v1.json");
    const firstCsv = await readFile(csvPath);
    const firstManifest = await readFile(manifestPath);
    const manifest = JSON.parse(firstManifest);
    assert.deepEqual(manifest.randomness, { seed: null, used: false });
    assert.equal(manifest.entries[0].csvSha256, sha256(firstCsv));
    assert.equal(manifest.entries[0].endTimeMs, endTimeMs);
    assert.equal(manifest.entries[0].lastCloseTimeMs, endTimeMs);
    assert.equal(manifest.entries[0].rows, 2);

    const second = await runPipeline({ ...env, EXPECTED_MANIFEST: manifestPath });
    assert.equal(second.status, 0, second.stderr);
    assert.deepEqual(await readFile(csvPath), firstCsv);
    assert.deepEqual(await readFile(manifestPath), firstManifest);

    const verified = await runPipeline({ VERIFY_MANIFEST: manifestPath });
    assert.equal(verified.status, 0, verified.stderr);
    assert.match(verified.stdout, /VERIFIED .* entries=1 sha256=[a-f0-9]{64}/);

    assert.equal(requests.length, 2);
    for (const request of requests) {
      assert.equal(request.pathname, "/api/v3/klines");
      assert.equal(request.searchParams.get("symbol"), "BTCUSDT");
      assert.equal(request.searchParams.get("interval"), "4h");
      assert.equal(request.searchParams.get("limit"), "2");
      assert.equal(request.searchParams.get("endTime"), String(endTimeMs));
    }
  });
});

test("backtest data pipeline rejects moving windows, drift, and incomplete bars", async () => {
  const missingEnd = await runPipeline({ SYMBOLS: "BTCUSDT" });
  assert.equal(missingEnd.status, 2);
  assert.match(missingEnd.stderr, /END_TIME_MS is required/);

  await withKlineServer(async ({ baseUrl, setRows }) => {
    const dataDir = await mkdtemp(path.join(os.tmpdir(), "trader-backtest-drift-"));
    const env = {
      BINANCE_KLINES_URL: baseUrl,
      DATA_DIR: dataDir,
      END_TIME_MS: String(endTimeMs),
      KLINE_LIMIT: "2",
      SYMBOLS: "BTCUSDT",
    };
    const initial = await runPipeline(env);
    assert.equal(initial.status, 0, initial.stderr);

    const csvPath = path.join(dataDir, "BTCUSDT-4h-2.csv");
    const manifestPath = path.join(dataDir, "backtest-data-manifest-v1.json");
    const initialCsv = await readFile(csvPath);
    const initialManifest = await readFile(manifestPath);

    const revised = structuredClone(baseRows);
    revised[1][4] = "101.25";
    setRows(revised);
    const drifted = await runPipeline({ ...env, EXPECTED_MANIFEST: manifestPath });
    assert.equal(drifted.status, 1);
    assert.match(drifted.stderr, /does not match expected manifest/);
    assert.deepEqual(await readFile(csvPath), initialCsv);
    assert.deepEqual(await readFile(manifestPath), initialManifest);

    const incomplete = structuredClone(baseRows);
    incomplete[1][6] = endTimeMs + 1;
    setRows(incomplete);
    const incompleteDir = await mkdtemp(path.join(os.tmpdir(), "trader-backtest-incomplete-"));
    const rejected = await runPipeline({ ...env, DATA_DIR: incompleteDir, EXPECTED_MANIFEST: "" });
    assert.equal(rejected.status, 1);
    assert.match(rejected.stderr, /not completed at endTimeMs/);

    const nonFinite = structuredClone(baseRows);
    nonFinite[1][4] = "NaN";
    setRows(nonFinite);
    const nonFiniteDir = await mkdtemp(path.join(os.tmpdir(), "trader-backtest-nonfinite-"));
    const invalidNumber = await runPipeline({
      ...env,
      DATA_DIR: nonFiniteDir,
      EXPECTED_MANIFEST: "",
    });
    assert.equal(invalidNumber.status, 1);
    assert.match(invalidNumber.stderr, /must be finite numeric evidence/);

    await writeFile(csvPath, Buffer.concat([initialCsv, Buffer.from("tamper\n")]));
    const tampered = await runPipeline({ VERIFY_MANIFEST: manifestPath });
    assert.equal(tampered.status, 1);
    assert.match(tampered.stderr, /CSV hash mismatch/);
  });

  const credentialedEndpoint = await runPipeline({
    BINANCE_KLINES_URL: "https://user:secret@api.binance.com/api/v3/klines",
    DATA_DIR: await mkdtemp(path.join(os.tmpdir(), "trader-backtest-credential-url-")),
    END_TIME_MS: String(endTimeMs),
    KLINE_LIMIT: "2",
    SYMBOLS: "BTCUSDT",
  });
  assert.equal(credentialedEndpoint.status, 1);
  assert.match(credentialedEndpoint.stderr, /must not contain credentials/);
  assert.doesNotMatch(credentialedEndpoint.stderr, /secret/);
});
