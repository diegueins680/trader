import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  loadAndVerify,
  loadAndVerifyRiskRegister,
  matchesGlob,
  verifyManifest,
  verifyRiskRegister,
  verifyRiskRegisterSources,
} from "../scripts/verify-formal-specs.mjs";

const canonicalVerification = loadAndVerify();
const canonicalRiskVerification = loadAndVerifyRiskRegister();

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

test("canonical risk register has unique ordered IDs and coherent projections", async () => {
  const result = await canonicalRiskVerification;
  assert.deepEqual(result.errors, []);
  assert.equal(result.ok, true);
  assert.ok(result.statistics.entries >= 27);
});

test("risk-register verifier rejects duplicate projection IDs and status drift", () => {
  const register = {
    schemaVersion: 1,
    severities: ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    statuses: ["OPEN", "MITIGATED", "CLOSED"],
    entries: [
      { id: "A-001", severity: "HIGH", status: "OPEN" },
      { id: "B-001", severity: "MEDIUM", status: "CLOSED" },
    ],
  };
  const markdownSource = `
| ID | Risk | Severity | Owner | Status | Next Action |
|---|---|---|---|---|---|
| A-001 | one | LOW | owner | OPEN | action |
| A-001 | duplicate | LOW | owner | OPEN | action |
| B-001 | two | MEDIUM | owner | CLOSED | action |
`;
  const haskellSource = `
data RiskID
    = A_001
    | B_001
    deriving (Eq)
riskIdText = \\case
    A_001 -> "A-001"
    B_001 -> "B-001"
riskRegister =
    [ riskEntry A_001 MEDIUM CLOSED "one" "owner" "action"
    , riskEntry B_001 MEDIUM CLOSED "two" "owner" "action"
    ]
`;
  const result = verifyRiskRegisterSources(register, { markdownSource, haskellSource });
  assert.equal(result.ok, false);
  assert.ok(result.errors.some((error) => error.includes("Markdown risk register: duplicate risk IDs: A-001")));
  assert.ok(result.errors.some((error) => error.includes("A-001 severity LOW does not match canonical HIGH")));
  assert.ok(result.errors.some((error) => error.includes("A-001 severity MEDIUM does not match canonical HIGH")));
  assert.ok(result.errors.some((error) => error.includes("A-001 status CLOSED does not match canonical OPEN")));
});

test("risk-register verifier rejects malformed canonical metadata and projection paths", async () => {
  const malformed = {
    schemaVersion: 2,
    severities: ["CRITICAL"],
    statuses: ["OPEN", "PENDING"],
    entries: [
      { id: "B-001", severity: "URGENT", status: "PENDING" },
      { id: "A-001", severity: "HIGH", status: "OPEN" },
      { id: "A-001", severity: "HIGH", status: "OPEN" },
    ],
  };
  const sourceResult = verifyRiskRegisterSources(malformed, { markdownSource: "", haskellSource: "" });
  assert.equal(sourceResult.ok, false);
  assert.ok(sourceResult.errors.includes("risk register schemaVersion must be 1"));
  assert.ok(sourceResult.errors.some((error) => error.startsWith("risk register severities must be")));
  assert.ok(sourceResult.errors.some((error) => error.startsWith("risk register statuses must be")));
  assert.ok(sourceResult.errors.includes("duplicate canonical risk IDs: A-001"));
  assert.ok(sourceResult.errors.includes("canonical risk entries must be sorted by ID"));
  assert.ok(sourceResult.errors.includes("B-001: invalid severity URGENT"));
  assert.ok(sourceResult.errors.includes("B-001: invalid status PENDING"));

  const pathResult = await verifyRiskRegister({
    schemaVersion: 1,
    severities: ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    statuses: ["OPEN", "MITIGATED", "CLOSED"],
    entries: [{ id: "A-001", severity: "HIGH", status: "OPEN" }],
    projections: { markdown: "wrong.md", haskell: "wrong.hs" },
  });
  assert.equal(pathResult.ok, false);
  assert.ok(pathResult.errors.some((error) => error.startsWith("risk register Markdown projection must be")));
  assert.ok(pathResult.errors.some((error) => error.startsWith("risk register Haskell projection must be")));
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
