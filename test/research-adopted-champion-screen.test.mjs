import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";

const registrationUrl = new URL("../research-notes/registrations/adopted-champion-screen-v1.json", import.meta.url);
const holdoutRegistrationUrl = new URL("../research-notes/registrations/residual-momentum-funding-only-v1.json", import.meta.url);
const carryRegistrationUrl = new URL("../research-notes/registrations/cross-sectional-funding-carry-v1.json", import.meta.url);
const harRvRegistrationUrl = new URL("../research-notes/registrations/har-rv-risk-gate-v1.json", import.meta.url);
const screenScript = fileURLToPath(new URL("../scripts/research/champion_screen.py", import.meta.url));
const gitignoreUrl = new URL("../data/research/.gitignore", import.meta.url);

const readJson = async (url) => JSON.parse(await readFile(url, "utf8"));
const FUTURE_FAMILY_START_MS = Date.parse("2027-01-21T00:00:00Z");

test("champion screen phases never touch sealed, carry or future-family symbols/windows", async () => {
  const reg = await readJson(registrationUrl);
  const holdout = await readJson(holdoutRegistrationUrl);
  const carry = await readJson(carryRegistrationUrl);
  const harRv = await readJson(harRvRegistrationUrl);
  const restricted = new Set([...holdout.universe.symbols, ...carry.universe.symbols]);
  assert.equal(Date.parse(harRv.dataset.startInclusiveUtc), FUTURE_FAMILY_START_MS);

  for (const [name, phase] of Object.entries(reg.phases)) {
    for (const symbol of phase.symbols) {
      assert.ok(!restricted.has(symbol), `${name} phase uses restricted symbol ${symbol}`);
    }
    assert.ok(phase.startMs < phase.endExclusiveMs, `${name} window is empty`);
    assert.ok(phase.endExclusiveMs <= FUTURE_FAMILY_START_MS, `${name} overlaps HAR-RV/OFI/missingness data`);
  }
  assert.ok(reg.phases.retrospective.endExclusiveMs <= reg.phases.prospective.startMs);
  assert.deepEqual(reg.excludedCombos.map((c) => c.symbol).sort(), ["ADAUSDT", "AVAXUSDT"]);
});

test("champion screen retrospective phase can only reject", async () => {
  const reg = await readJson(registrationUrl);
  assert.match(reg.protocol.decisionScope, /can only REJECT or be INCONCLUSIVE/);
  assert.equal(reg.protocol.rejectionRule.maxNetReturnForRejection, 0);
  assert.equal(reg.protocol.rejectionRule.maxDrawdown, 0.2);
  assert.equal(reg.phases.prospective.oneShot, true);
  assert.ok(reg.protocol.seeds.includes(reg.protocol.liveSeed));
});

test("live champion parameters stay out of the public repository", async () => {
  const reg = await readJson(registrationUrl);
  const ignored = await readFile(gitignoreUrl, "utf8");
  const relative = reg.snapshot.path.replace(/^data\/research\//, "");
  assert.ok(ignored.split("\n").includes(relative), "snapshot must be gitignored");
  assert.match(reg.snapshot.sha256, /^[0-9a-f]{64}$/);
});

test("champion screen harness selftest", () => {
  const out = execFileSync("python3", [screenScript, "selftest"], { encoding: "utf8" });
  assert.match(out, /selftest ok/);
});
