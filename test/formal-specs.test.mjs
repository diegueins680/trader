import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { loadAndVerify, matchesGlob, verifyManifest } from "../scripts/verify-formal-specs.mjs";

const canonicalVerification = loadAndVerify();

test("formal specification registry is coherent and covers production files", async () => {
  const result = await canonicalVerification;
  assert.deepEqual(result.errors, []);
  assert.equal(result.ok, true);
  assert.ok(result.statistics.specifications >= 30);
  assert.ok(result.statistics.features >= 80);
  assert.ok(result.statistics.implementationFiles >= 150);
});

test("formal glob matching keeps single and recursive wildcards distinct", () => {
  assert.equal(matchesGlob("haskell/app/Main.hs", "haskell/app/*.hs"), true);
  assert.equal(matchesGlob("haskell/app/Trader/LSTM.hs", "haskell/app/*.hs"), false);
  assert.equal(matchesGlob("haskell/app/Trader/Predictors/KNN.hs", "haskell/app/Trader/Predictors/**"), true);
});

test("formal verifier rejects duplicate clauses, dependency cycles, and missing evidence", async () => {
  const clause = { id: "C1", statement: "x = x" };
  const base = {
    schemaVersion: 1,
    semantics: { transitionSystem: "F : I x S -> O x S'" },
    coverage: { areas: [], explicit: [] },
    globalInvariants: [{ id: "G1", statement: "true" }],
    specifications: [
      {
        id: "X1",
        title: "one",
        features: ["one"],
        criticality: "correctness",
        implementation: ["missing.file"],
        uses: ["G1"],
        dependsOn: ["X2"],
        requires: [clause],
        ensures: [{ ...clause, id: "C2" }],
        invariants: [{ ...clause, id: "C3" }],
        failures: [{ ...clause, id: "C4" }],
        evidence: [{ level: "regression", path: "missing.test" }],
      },
      {
        id: "X2",
        title: "two",
        features: ["two"],
        criticality: "correctness",
        implementation: ["missing.file"],
        uses: ["G1"],
        dependsOn: ["X1"],
        requires: [{ ...clause, id: "C1" }],
        ensures: [{ ...clause, id: "C5" }],
        invariants: [{ ...clause, id: "C6" }],
        failures: [{ ...clause, id: "C7" }],
        evidence: [{ level: "regression", path: "missing.test" }],
      },
    ],
  };
  const result = await verifyManifest(base);
  assert.equal(result.ok, false);
  assert.ok(result.errors.some((error) => error.includes("duplicate contract clause IDs")));
  assert.ok(result.errors.some((error) => error.includes("dependency cycle")));
  assert.ok(result.errors.some((error) => error.includes("missing evidence")));
});

test("Hetzner deployment safety is represented by the canonical formal registry", async () => {
  const result = await canonicalVerification;
  assert.equal(result.ok, true);
  const compose = await readFile(new URL("../deploy/hetzner/docker-compose.yml", import.meta.url), "utf8");
  const backend = await readFile(new URL("../haskell/app/Main.hs", import.meta.url), "utf8");
  const bridge = await readFile(new URL("../deploy/hetzner/webhook-bridge/bridge.py", import.meta.url), "utf8");
  assert.doesNotMatch(compose, /^\s+- --binance-live\s*$/m);
  assert.match(compose, /TRADER_BINANCE_LIVE must be true\/false\/1\/0/);
  assert.match(backend, /tradeAllowedByEnv && botTradeEnabledFromApi/);
  assert.match(backend, /bsTradeEnabled = tradeEnabled/);
  assert.match(backend, /argBinanceLive = argBinanceLive baseArgs && pick \(apBinanceLive p\) True/);
  assert.match(backend, /live = argBinanceLive baseArgs && fromMaybe True \(abcpBinanceLive params\)/);
  assert.match(bridge, /MAX_BODY_BYTES = 64 \* 1024/);
  assert.doesNotMatch(bridge, /print\(payload|print\(phone|print\(api_key/);
  execFileSync(
    "python3",
    [
      "-c",
      [
        "import runpy",
        "m=runpy.run_path('deploy/hetzner/webhook-bridge/bridge.py')",
        "msg=m['event_message']({'type':'bot.halt','symbol':'BTCUSDT','error':'secret','apiKey':'secret'})",
        "assert msg == 'Trader alert: event=bot.halt | symbol=BTCUSDT'",
      ].join(";"),
    ],
    { cwd: new URL("..", import.meta.url), env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } },
  );
});
